"""
Module D — Dual-Bias Compositional Drift  [khung đã kiểm chứng bằng keyword-probe:
8/13 aspect đổi kết luận sau hiệu chỉnh thành phần; mùa vụ khôi phục đúng
(điều hoà đỉnh T6 2,22× · hồ bơi đỉnh T8 1,73× · phòng phẳng 1,05×)].

Input : quad đã gắn trọng số w (outputs/reliability/quads_weighted.jsonl.gz)
        HOẶC pool_quads.*.jsonl.gz (w=conf_seq) — cùng schema.

Hai kênh, mỗi kênh có cả bản thô (raw) và bản hiệu chỉnh thành phần (adj):
  π  prevalence — SHARE-OF-COMPLAINTS: tỷ trọng aspect trong tổng lượt phàn nàn
     (quad có φ=NEG hoặc sentiment=negative), trên thang CLR.
  ν  valence    — NEGATIVITY RATE: P(negative | có nhắc aspect), trên MỌI quad,
     direct standardization trên strata (tử/mẫu tính trong từng ô rồi lấy
     trung bình theo tỷ trọng tham chiếu). Đây là estimand trung tâm của paper.

Bước  : D-a hiệu chỉnh recall — CHƯA IMPLEMENT (stub, chỉ log). --final sẽ chặn
        D-b direct standardization trên strata (tham chiếu = gộp toàn kỳ)
        khử mùa vụ (trừ trung bình tháng-trong-năm) -> OLS trend + best-split
        changepoint -> permutation null RIÊNG TỪNG CHUỖI cho CẢ HAI thống kê
        (p_perm = trend, p_perm_changepoint = bước nhảy — hai kiểm định khác nhau)
        BH-FDR trên thống kê TREND: lưới adj (π-adj ∪ ν-adj) là kết quả chính;
        lưới raw FDR riêng để so sánh like-for-like trong injection test (E3a)
        Bootstrap CI cho slope π-adj (resample review TRONG strata)

Chạy:  python3 -m prism.module_d_drift --quads outputs/reliability/quads_weighted.jsonl.gz \
           --level taxonomy_code --cohort corpus
Ra  :  outputs/drift/drift_results.<cohort>.<level>.json  (đổi bằng --out)
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random

from . import config as C
from . import utils as U

log = U.get_logger("prism.D")


# ------------------------------------------------------------------ aggregation
def load_quads(path, level: str, cohort_hotels: set[str]):
    """
    Trả về:
      cell[(period, stratum)][aspect] = tổng w                (kênh π, chỉ complaint)
      vcell[(period, stratum)][aspect][sentiment] = tổng w    (kênh ν, mọi quad)
      reviews[(period, stratum)] = list (review_uid, {aspect: w}) — cho bootstrap
    """
    cell = collections.defaultdict(lambda: collections.defaultdict(float))
    vcell = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.Counter()))
    rev_acc = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(float)))
    n_used = 0
    n_bad_code = 0
    for q in U.read_jsonl(path):
        if cohort_hotels and q["hotel_id"] not in cohort_hotels:
            continue
        # Lọc w TRƯỚC khi tra taxonomy: apply_weights chủ ý đặt w=0 cho code ngoài
        # taxonomy, nhưng nếu tra CODE2CAT trước thì KeyError xảy ra trước cả nhánh
        # bỏ đó -> cả run chết ở một dòng bất kỳ giữa 1,9M dòng.
        w = q.get("w", q.get("conf_seq", 1.0))
        if w <= 0:
            continue
        code = q["taxonomy_code"]
        cat = C.CODE2CAT.get(code)
        if cat is None:
            n_bad_code += 1
            continue          # code ngoài taxonomy -> bỏ, có đếm, không crash
        a = code if level == "taxonomy_code" else cat
        if level == "taxonomy_code" and a not in C.CODES_REPORTABLE:
            a = cat                                   # gộp code hiếm lên category
        per, st = q["period"], tuple(q["stratum"])
        vcell[(per, st)][a][q["sentiment"]] += w
        if q["phi"] == "NEG" or q["sentiment"] == "negative":
            cell[(per, st)][a] += w
            rev_acc[(per, st)][q["review_uid"]][a] += w
        n_used += 1
    reviews = {k: list(v.items()) for k, v in rev_acc.items()}
    log.info("dùng %d quad", n_used)
    if n_bad_code:
        log.warning("bỏ %d quad có taxonomy_code ngoài CODE2CAT", n_bad_code)
    if not n_used:
        raise SystemExit(f"KHÔNG có quad nào dùng được từ {path} — file rỗng, "
                         f"cohort không khớp, hoặc w=0 hết. Dừng thay vì ghi ra "
                         f"kết quả trên tập rỗng.")
    return cell, vcell, reviews


def reference_composition(cell) -> dict:
    ref = collections.Counter()
    for (_, st), aspects in cell.items():
        ref[st] += sum(aspects.values())
    tot = sum(ref.values())
    return {st: v / tot for st, v in ref.items()} if tot else {}


def reference_composition_valence(vcell) -> dict:
    ref = collections.Counter()
    for (_, st), d in vcell.items():
        ref[st] += sum(sum(sv.values()) for sv in d.values())
    tot = sum(ref.values())
    return {st: v / tot for st, v in ref.items()} if tot else {}


def shares_by_period(cell, ref, aspects, min_n: int, adjust: bool) -> dict[str, dict]:
    """
    π theo kỳ; adjust=True -> direct standardization; False -> thô.

    Ô nhỏ KHÔNG bị loại mà được SHRINK về share gộp của kỳ (prior Dirichlet,
    cường độ = min_n). Loại ô đột ngột (bản cũ) làm mix strata đóng góp tự trôi
    theo thời gian khi một stratum teo dần -> sinh trend giả ngay trong kênh adj
    (đã tái hiện được bằng composition injection trên nền null). Shrinkage giữ
    mọi stratum đóng góp liên tục: ô rỗng đóng góp đúng share gộp (trung tính).
    """
    out = {}
    periods = sorted({p for (p, _) in cell})
    for per in periods:
        pooled = {a: 0.0 for a in aspects}
        for (p, _), counts in cell.items():
            if p != per:
                continue
            for a, v in counts.items():
                pooled[a] += v
        ptot = sum(pooled.values())
        s0 = {a: (pooled[a] / ptot if ptot else 1.0 / len(aspects)) for a in aspects}
        if adjust:
            acc = {a: 0.0 for a in aspects}
            for st, wref in ref.items():
                counts = cell.get((per, st)) or {}
                tot = sum(counts.values())
                den = tot + min_n
                for a in aspects:
                    sh = ((counts.get(a, 0.0) + min_n * s0[a]) / den
                          if den > 0 else s0[a])
                    acc[a] += wref * sh
        else:
            acc = pooled
        t = sum(acc.values()) or 1.0
        out[per] = {a: acc[a] / t for a in aspects}
    return out


def valence_by_period(vcell, ref, aspects, periods, min_w: float,
                      adjust: bool) -> dict[str, dict]:
    """
    ν_{a,t} = P(negative | nhắc a, t). adjust=True: direct standardization —
    tỷ lệ trong từng stratum được SHRINK về tỷ lệ gộp của kỳ (prior Beta,
    cường độ = min_w) rồi lấy trung bình theo tỷ trọng tham chiếu. Cùng lý do
    với shares_by_period: loại ô nhỏ đột ngột sinh trend giả khi stratum teo dần.
    Kỳ không có quad nào của aspect -> None.
    """
    out = {}
    for per in periods:
        row = {}
        for a in aspects:
            neg0 = tot0 = 0.0
            for (p, st), d in vcell.items():
                if p != per:
                    continue
                sv = d.get(a)
                if not sv:
                    continue
                neg0 += sv.get("negative", 0.0)
                tot0 += sum(sv.values())
            if tot0 <= 0:
                row[a] = None
                continue
            nu0 = neg0 / tot0
            if adjust:
                num = den = 0.0
                for st, wref in ref.items():
                    sv = vcell.get((per, st), {}).get(a) or {}
                    tot = sum(sv.values())
                    cw = tot + min_w
                    nu = ((sv.get("negative", 0.0) + min_w * nu0) / cw
                          if cw > 0 else nu0)
                    num += wref * nu
                    den += wref
                row[a] = num / den if den > 0 else None
            else:
                row[a] = nu0
        out[per] = row
    return out


# ------------------------------------------------------------- trend machinery
def deseason(series: list[float], periods: list[str]) -> list[float]:
    moy = collections.defaultdict(list)
    for v, p in zip(series, periods):
        moy[p[5:7]].append(v)
    mm = {m: sum(v) / len(v) for m, v in moy.items()}
    gm = sum(series) / len(series)
    return [v - (mm[p[5:7]] - gm) for v, p in zip(series, periods)]


def ols_trend(series: list[float]) -> tuple[float, float]:
    n = len(series)
    if n < 2:
        return 0.0, 0.0                             # chuỗi 1 điểm: sxx = 0
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(series) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, series)) / sxx
    res = [y - (my + b * (x - mx)) for x, y in zip(xs, series)]
    se = math.sqrt(sum(r * r for r in res) / (n - 2) / sxx) if n > 2 else 0.0
    return b * 12, (b / se if se else 0.0)          # slope/năm, t


def best_split(series: list[float]) -> tuple[int, float]:
    best_i, best_t = -1, 0.0
    for i in range(4, len(series) - 4):             # tối thiểu 4 điểm mỗi phía
        t = abs(U.welch_t(series[:i], series[i:]))
        if t > best_t:
            best_i, best_t = i, t
    return best_i, best_t


def permutation_p(series: list[float], periods: list[str],
                  obs_trend_t: float, obs_cp_t: float,
                  n_perm: int, rng: random.Random) -> tuple[float, float]:
    """Null RIÊNG cho chuỗi này: xáo trộn thứ tự thời gian, giữ nguyên giá trị.

    Trả (p của thống kê TREND, p của thống kê CHANGEPOINT) — HAI kiểm định khác
    nhau, tính trong cùng một vòng permutation.

    Bản cũ chỉ permute thống kê changepoint rồi báo cáo p đó cạnh slope OLS, nên
    `significant_after_fdr` thực chất là "có bước nhảy", trong khi verdict lại gate
    trên dấu slope: aspect trôi đơn điệu mạnh mà không có bước nhảy bị gọi là
    "phẳng", còn bước nhảy sạch với slope ~0 được gọi là "XU HƯỚNG THẬT".
    """
    hit_trend = hit_cp = 0
    for _ in range(n_perm):
        sh = series[:]
        rng.shuffle(sh)
        de = deseason(sh, periods)
        _, t_cp = best_split(de)
        _, t_tr = ols_trend(de)
        if abs(t_tr) >= obs_trend_t:
            hit_trend += 1
        if t_cp >= obs_cp_t:
            hit_cp += 1
    return ((hit_trend + 1) / (n_perm + 1), (hit_cp + 1) / (n_perm + 1))


def channel_stats(series: list[float | None], periods: list[str],
                  n_perm: int, rng: random.Random) -> dict | None:
    """Trend + changepoint + permutation-p cho một chuỗi; None nếu <12 điểm.

    `p_perm` là p của THỐNG KÊ TREND (khớp với slope_per_year/t_stat được báo cáo
    và với verdict). `p_perm_changepoint` là p của bước nhảy, báo cáo riêng.
    """
    pts = [(v, p) for v, p in zip(series, periods) if v is not None]
    if len(pts) < 12:
        return None
    vals = [v for v, _ in pts]
    pers = [p for _, p in pts]
    de = deseason(vals, pers)
    slope, t = ols_trend(de)
    ci, ct = best_split(de)
    p_tr, p_cp = permutation_p(vals, pers, abs(t), ct, n_perm, rng)
    return {"slope_per_year": round(slope, 4), "t_stat": round(t, 2),
            "changepoint": pers[ci] if ci >= 0 else None,
            "welch_t": round(ct, 2),
            "p_perm": round(p_tr, 5),                 # kiểm định TREND (chính)
            "p_perm_changepoint": round(p_cp, 5)}     # kiểm định BƯỚC NHẢY


# ----------------------------------------------------------------------- main
def run(args) -> None:
    C.ensure_dirs()
    rng = random.Random(args.seed)
    cohort_file = C.STORE_DIR / "hotel_cohorts.json"
    if cohort_file.exists():
        cohorts = json.loads(cohort_file.read_text())
        keep = set(cohorts.get(args.cohort) or [])
    else:
        if args.cohort != "corpus":
            raise SystemExit(f"cohort '{args.cohort}' cần {cohort_file} — chạy Module A trước")
        keep = set()

    cell, vcell, reviews = load_quads(args.quads, args.level, keep)
    aspects = sorted({a for d in cell.values() for a in d}
                     | {a for d in vcell.values() for a in d})
    periods = sorted({p for (p, _) in vcell})
    if len(periods) < 12:
        log.warning("chỉ %d kỳ — kết quả trend sẽ yếu", len(periods))
    ref = reference_composition(cell)
    ref_v = reference_composition_valence(vcell)

    # D-a: hiệu chỉnh recall — CHƯA IMPLEMENT (stub).
    # Log phải nói đúng sự thật: bản cũ in "áp dụng recall_table.json" trong khi
    # chỉ đọc file rồi bỏ, nên người đọc log tin rằng số đã hiệu chỉnh recall.
    rec_path = C.DRIFT_DIR / "recall_table.json"
    recall_adjusted = False
    if rec_path.exists():
        log.warning("D-a: THẤY %s nhưng CHƯA áp dụng (bước hiệu chỉnh recall vẫn là "
                    "stub) -> mọi số π/ν dưới đây CHƯA hiệu chỉnh recall", rec_path.name)
        if args.final:
            raise SystemExit(
                "--final: không sinh số cuối khi D-a còn là stub. Implement bước áp "
                "recall_table (nhân 1/ρ̂(sentiment, length_bin) vào w trong load_quads) "
                "hoặc bỏ --final và ghi rõ đây là giới hạn của bài.")
    else:
        log.info("D-a: BỎ QUA (chưa có recall_table.json từ tập audit D0)")

    S_adj = shares_by_period(cell, ref, aspects, args.min_stratum, adjust=True)
    S_raw = shares_by_period(cell, ref, aspects, args.min_stratum, adjust=False)
    V_adj = valence_by_period(vcell, ref_v, aspects, periods,
                              args.min_valence_w, adjust=True)
    V_raw = valence_by_period(vcell, ref_v, aspects, periods,
                              args.min_valence_w, adjust=False)

    pi_periods = sorted({p for (p, _) in cell})
    results = []
    for a in aspects:
        row = {"aspect": a, "level": args.level, "cohort": args.cohort}
        # kênh π (prevalence trong complaint composition, thang CLR)
        for tag, S in (("raw", S_raw), ("adj", S_adj)):
            series = [U.clr(S[p])[a] for p in pi_periods]
            st = channel_stats(series, pi_periods, args.n_perm, rng)
            row[tag] = st or {"p_perm": None}
        # kênh ν (negativity rate, direct standardization)
        for tag, V in (("val_raw", V_raw), ("val_adj", V_adj)):
            series = [V[p].get(a) for p in periods]
            row[tag] = channel_stats(series, periods, args.n_perm, rng)
        row["nu_raw_series"] = [None if V_raw[p].get(a) is None
                                else round(V_raw[p][a], 4) for p in periods]
        row["nu_adj_series"] = [None if V_adj[p].get(a) is None
                                else round(V_adj[p][a], 4) for p in periods]
        row["pi_adj_series"] = [round(S_adj[p][a], 5) for p in pi_periods]
        row["pi_raw_series"] = [round(S_raw[p][a], 5) for p in pi_periods]
        results.append(row)

    # Bootstrap CI cho slope π-adj: resample review trong strata MỘT LẦN mỗi
    # vòng rồi lấy slope cho MỌI aspect từ cùng bản resample — vừa nhanh gấp
    # |aspects| lần, vừa đúng hơn (các aspect chia sẻ cùng phương án resample,
    # bảo toàn tương quan giữa các share của một composition).
    if args.n_boot:
        keys = list(reviews)
        slopes_by_aspect: dict[str, list[float]] = {a: [] for a in aspects}
        for _ in range(args.n_boot):
            cell_b = collections.defaultdict(lambda: collections.defaultdict(float))
            for k in keys:
                revs = reviews[k]
                for _ in range(len(revs)):
                    _, aw = revs[rng.randrange(len(revs))]
                    for aa, w in aw.items():
                        cell_b[k][aa] += w
            Sb = shares_by_period(cell_b, ref, aspects, args.min_stratum, True)
            for a in aspects:
                sb = deseason([U.clr(Sb[p]).get(a, 0.0) for p in pi_periods],
                              pi_periods)
                slopes_by_aspect[a].append(ols_trend(sb)[0])
        for row in results:
            sl = sorted(slopes_by_aspect[row["aspect"]])
            if sl and row["adj"].get("t_stat") is not None:
                row["adj"]["slope_ci95"] = [round(sl[int(0.025 * len(sl))], 4),
                                            round(sl[int(0.975 * len(sl))], 4)]

    # BH-FDR — lưới CHÍNH: hợp nhất (π-adj ∪ ν-adj) trên mọi aspect.
    # Lưới raw FDR RIÊNG (π-raw ∪ ν-raw): chỉ để so sánh like-for-like (E3a),
    # không phải kết quả chính.
    def apply_fdr(entries):
        qvals = U.benjamini_hochberg([p for _, p in entries], args.fdr)
        for (st, _), qv in zip(entries, qvals):
            st["p_fdr"] = round(qv, 5)
            st["significant_after_fdr"] = bool(qv <= args.fdr)

    adj_entries, raw_entries = [], []
    for row in results:
        for tag, bucket in (("adj", adj_entries), ("val_adj", adj_entries),
                            ("raw", raw_entries), ("val_raw", raw_entries)):
            st = row.get(tag)
            if st and st.get("p_perm") is not None:
                bucket.append((st, st["p_perm"]))
    apply_fdr(adj_entries)
    apply_fdr(raw_entries)

    def sig(row, tag):
        st = row.get(tag)
        return bool(st and st.get("significant_after_fdr"))

    n_sig = n_sig_val = 0
    for row in results:
        n_sig += sig(row, "adj")
        n_sig_val += sig(row, "val_adj")
        for prefix, r_tag, a_tag in (("verdict", "raw", "adj"),
                                     ("verdict_val", "val_raw", "val_adj")):
            r_sig, a_sig = sig(row, r_tag), sig(row, a_tag)
            # `or 0.0` gộp "thiếu số" với "bằng 0" -> nhánh cuối gán nhãn
            # "ĐẢO DẤU" (một kết luận rất mạnh) cho cả trường hợp sa*sr == 0.
            sr = (row.get(r_tag) or {}).get("slope_per_year")
            sa = (row.get(a_tag) or {}).get("slope_per_year")
            if a_sig and r_sig and sr is not None and sa is not None and sa * sr > 0:
                row[prefix] = "XU HƯỚNG THẬT"
            elif r_sig and not a_sig:
                row[prefix] = "GIẢ (do thành phần)"
            elif a_sig and not r_sig:
                row[prefix] = "BỊ CHE, lộ ra sau hiệu chỉnh"
            elif a_sig and r_sig and sr is not None and sa is not None and sa * sr < 0:
                row[prefix] = "ĐẢO DẤU sau hiệu chỉnh"
            elif a_sig:
                row[prefix] = "không xác định (thiếu slope hoặc slope = 0)"
            else:
                row[prefix] = "phẳng"

    out = (U.Path(args.out) if args.out
           else C.DRIFT_DIR / f"drift_results.{args.cohort}.{args.level}.json")
    U.write_json(out, {"config": vars(args),
                       "periods": periods,          # trục của nu_*_series
                       "pi_periods": pi_periods,    # trục của pi_*_series
                       # p_perm/p_fdr là kiểm định TREND; bước nhảy ở p_perm_changepoint
                       "test_statistic": "ols_trend_t (permutation, null riêng từng chuỗi)",
                       "recall_adjusted": recall_adjusted,
                       "n_significant_after_fdr": n_sig,
                       "n_significant_valence_fdr": n_sig_val,
                       "results": results})
    log.info("=> %s  (π-adj %d/%d · ν-adj %d/%d aspect vượt FDR)",
             out, n_sig, len(results), n_sig_val, len(results))
    for r in sorted(results, key=lambda r: -abs(r["adj"].get("t_stat") or 0)):
        va = r.get("val_adj") or {}
        log.info("  %-22s π: raw t=%+6.1f adj t=%+6.1f q=%s %s | ν: adj t=%s q=%s %s",
                 r["aspect"], r["raw"].get("t_stat") or 0, r["adj"].get("t_stat") or 0,
                 r["adj"].get("p_fdr"), r["verdict"],
                 va.get("t_stat"), va.get("p_fdr"), r.get("verdict_val"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quads", required=True)
    ap.add_argument("--level", default="taxonomy_code",
                    choices=["taxonomy_code", "aspect_category"])
    ap.add_argument("--cohort", default="corpus",
                    choices=C.COHORTS)
    ap.add_argument("--min-stratum", type=int, default=C.MIN_STRATUM_N,
                    help="CƯỜNG ĐỘ PRIOR shrink ô strata về share gộp của kỳ (kênh π, "
                         "đơn vị tổng w) — KHÔNG phải ngưỡng lọc. Số LỚN HƠN = shrink "
                         "MẠNH HƠN = mọi chuỗi phẳng hơn.")
    ap.add_argument("--min-valence-w", type=float, default=C.MIN_VALENCE_CELL_W,
                    help="CƯỜNG ĐỘ PRIOR shrink ô về tỷ lệ gộp của kỳ (kênh ν) — "
                         "KHÔNG phải ngưỡng lọc; xem --min-stratum")
    ap.add_argument("--n-perm", type=int, default=C.N_PERMUTATION)
    ap.add_argument("--n-boot", type=int, default=0,
                    help="0 = tắt bootstrap (bật 1000 cho kết quả cuối)")
    ap.add_argument("--fdr", type=float, default=C.FDR_ALPHA)
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    ap.add_argument("--final", action="store_true",
                    help="đánh dấu đây là run sinh SỐ CUỐI: dừng nếu còn bước nào "
                         "là stub (hiện tại: D-a hiệu chỉnh recall)")
    ap.add_argument("--out", default=None,
                    help="đường dẫn output; mặc định drift_results.<cohort>.<level>.json")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
