"""
E3/E4 — giao thức đánh giá bằng injection và negative control (thí nghiệm lõi).

Vì KHÔNG có drift ground truth, ta tạo drift/không-drift có kiểm soát trên chính
dữ liệu thật rồi kiểm tra pipeline trả lời đúng:

  E3a composition : XÁO TRỘN timestamp trước (tạo nền null — dữ liệu thật có thể
                    chứa drift thật, áp injection thẳng lên nó thì không thể đòi
                    adj im lặng), rồi RESAMPLE tỷ trọng strata trôi dần
                    -> PASS = adj KHÔNG báo drift (0 aspect) VÀ raw CÓ báo (>=1),
                       cả hai đo bằng CÙNG máy suy diễn (permutation + FDR),
                       không phải t-threshold cho raw vs FDR cho adj (táo vs cam)
  E3c valence     : sau t0, lật δ% quad của 1 aspect pos->neg (KHÔNG đổi φ)
                    -> kênh ν-adj phải bắt được, đo detection theo δ ∈ {2,5,10,20}%
                    LƯU Ý: kênh π cũng bị ảnh hưởng theo thiết kế (load_quads nhận
                    quad vào π khi sentiment==negative) — xem inject_valence
  E4  shuffle     : xáo timestamp Ở MỨC REVIEW, lặp --repeats lần với seed khác
                    -> FPR thực nghiệm trung bình ≈ α (1 lần chạy với ~14 aspect
                       có độ phân giải quá thô để ước lượng FPR)

Chạy:  python3 -m prism.eval_injection --quads <file> --test {composition,valence,shuffle}
Ra  :  <--out-dir>/injection_<test>.json  (mặc định outputs/drift; truyền --out-dir
       khi chạy thử để KHÔNG ghi đè drift_results chính)
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys

from . import config as C
from . import utils as U

log = U.get_logger("prism.E3")


def load(path):
    return list(U.read_jsonl(path))


def dump(quads, path):
    U.write_jsonl(path, quads)
    return path


def inject_composition(quads, rng):
    """Kéo tỷ trọng stratum VN giảm tuyến tính theo kỳ (mô phỏng đúng B1 89,7→16,1%),
    bằng resample-with-drop — KHÔNG đổi nội dung bất kỳ quad nào."""
    periods = sorted({q["period"] for q in quads})
    idx = {p: i for i, p in enumerate(periods)}
    out = []
    for q in quads:
        frac = idx[q["period"]] / max(len(periods) - 1, 1)
        if q["stratum"][0] == "VN":
            keep_p = 1.0 - 0.8 * frac         # VN giữ 100% đầu kỳ -> 20% cuối kỳ
        else:
            keep_p = 0.2 + 0.8 * frac         # non-VN ngược lại
        if rng.random() < keep_p:
            out.append(q)
    return out


def inject_valence(quads, aspect, delta, t0, rng):
    """Lật δ% quad pos->neg của 1 aspect sau t0. Chỉ đổi sentiment, giữ nguyên φ.

    LƯU Ý: "chỉ đổi một biến" đúng ở mức TRƯỜNG DỮ LIỆU, không đúng ở mức ESTIMAND.
    load_quads nhận quad vào kênh π theo điều kiện `phi == "NEG" or sentiment ==
    "negative"`, nên mọi quad bị lật cũng đi vào tử số của π. Cột
    `prevalence_detected_within_3` trong report E3c vì thế bị nhiễu THEO THIẾT KẾ —
    nó không phải bằng chứng "π giữ im lặng khi chỉ có valence drift".
    """
    out = []
    for q in quads:
        q = dict(q)
        if (q["taxonomy_code"] == aspect and q["period"] >= t0
                and q["sentiment"] == "positive" and rng.random() < delta):
            q["sentiment"] = "negative"
        out.append(q)
    return out


def inject_shuffle(quads, rng):
    """Xáo timestamp Ở MỨC REVIEW: permute period giữa các review_uid, rồi phát
    xuống mọi quad của cùng review.

    Bản cũ shuffle list period rồi zip theo TỪNG QUAD, nên hai quad của cùng một
    review lạc sang hai tháng khác nhau. Hai hệ quả: (a) load_quads gom
    rev_acc[(period,stratum)][review_uid] nên bootstrap resample trên các mảnh
    review bị xé — đơn vị resample nhỏ hơn thật; (b) với E4, xáo theo quad phá
    tương quan TRONG review, làm null độc lập hơn dữ liệu thật -> phương sai chuỗi
    bị hạ -> FPR thực nghiệm bị ước lượng THẤP và negative control PASS quá dễ.
    Đây là điểm tựa của lập luận "khung không sinh drift giả", nên phải đúng.
    """
    period_of: dict[str, str] = {}
    for q in quads:                          # period của review = period đầu tiên thấy
        period_of.setdefault(q["review_uid"], q["period"])
    uids = sorted(period_of)                 # sorted -> tái lập được với cùng seed
    pool = [period_of[u] for u in uids]
    rng.shuffle(pool)
    remap = dict(zip(uids, pool))
    return [dict(q, period=remap[q["review_uid"]]) for q in quads]


def run_drift(quad_file, tag, n_perm=300, out_dir=None):
    out = U.Path(out_dir or C.DRIFT_DIR) / f"drift_results.{tag}.json"
    subprocess.run([sys.executable, "-m", "prism.module_d_drift",
                    "--quads", str(quad_file), "--level", "taxonomy_code",
                    "--cohort", "corpus", "--n-perm", str(n_perm),
                    "--out", str(out)], check=True)
    return json.loads(out.read_text())


def _sig(row, tag) -> bool:
    st = row.get(tag)
    return bool(st and st.get("significant_after_fdr"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quads", required=True)
    ap.add_argument("--test", required=True,
                    choices=["composition", "valence", "shuffle"])
    ap.add_argument("--aspect", default="AM_FOOD")
    ap.add_argument("--t0", default="2023-09")
    ap.add_argument("--repeats", type=int, default=5,
                    help="số lần lặp shuffle (E4) với seed khác nhau")
    ap.add_argument("--n-perm", type=int, default=300)
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    ap.add_argument("--out-dir", default=None,
                    help="thư mục output; mặc định outputs/drift. Smoke test PHẢI "
                         "truyền thư mục riêng để không ghi đè kết quả thật.")
    args = ap.parse_args()
    out_dir = U.Path(args.out_dir) if args.out_dir else C.DRIFT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    quads = load(args.quads)
    log.info("nạp %d quad", len(quads))

    if args.test == "composition":
        # nền null trước: xáo timestamp để drift DUY NHẤT còn lại là cái ta tiêm
        base = inject_shuffle(quads, rng)
        inj = inject_composition(base, rng)
        f = dump(inj, out_dir / "quads_inj_composition.jsonl.gz")
        res = run_drift(f, "inj_composition", args.n_perm, out_dir)
        # PASS chặt: adj im lặng hoàn toàn, raw báo — cùng máy suy diễn FDR
        n_raw = sum(_sig(r, "raw") or _sig(r, "val_raw") for r in res["results"])
        n_adj = sum(_sig(r, "adj") or _sig(r, "val_adj") for r in res["results"])
        if n_adj > 0:
            verdict = "FAIL"            # adj báo drift trên nền null -> hiệu chỉnh hỏng
        elif n_raw >= 1:
            verdict = "PASS"            # raw báo, adj im lặng -> đúng như thiết kế
        else:
            verdict = "INCONCLUSIVE"    # cả hai im lặng: injection quá yếu để raw
                                        # bắt được (mix aspect giữa strata quá giống
                                        # nhau) — không nói được gì về adj
        rep = {"n_raw_significant_fdr": n_raw, "n_adj_significant_fdr": n_adj,
               "n_aspects": len(res["results"]), "verdict": verdict,
               "note": "PASS = adj im lặng (0) VÀ raw báo (>=1), cả hai sau FDR, "
                       "trên nền null đã xáo timestamp"}
    elif args.test == "valence":
        rep = {"aspect": args.aspect, "t0": args.t0, "detections": {}}
        for delta in (0.02, 0.05, 0.10, 0.20):
            inj = inject_valence(quads, args.aspect, delta, args.t0, rng)
            f = dump(inj, out_dir / f"quads_inj_val{int(delta*100)}.jsonl.gz")
            res = run_drift(f, f"inj_val{int(delta*100)}", args.n_perm, out_dir)
            row = next(r for r in res["results"] if r["aspect"] == args.aspect)

            def within3(st):
                cp = (st or {}).get("changepoint")
                if not cp:
                    return False
                return abs(int(cp[:4]) * 12 + int(cp[5:7])
                           - int(args.t0[:4]) * 12 - int(args.t0[5:7])) <= 3

            val, prev = row.get("val_adj") or {}, row.get("adj") or {}
            rep["note"] = ("prevalence_* bị nhiễu THEO THIẾT KẾ: quad lật pos->neg "
                           "cũng vào kênh π qua điều kiện sentiment==negative")
            rep["detections"][f"{int(delta*100)}%"] = {
                "valence_detected_within_3": _sig(row, "val_adj") and within3(val),
                "valence_changepoint": val.get("changepoint"),
                "valence_p_fdr": val.get("p_fdr"),
                "prevalence_detected_within_3": _sig(row, "adj") and within3(prev),
                "prevalence_p_fdr": prev.get("p_fdr"),
            }
    else:   # shuffle
        fprs, runs = [], []
        for r_i in range(args.repeats):
            rng_i = random.Random(args.seed + r_i)
            inj = inject_shuffle(quads, rng_i)
            f = dump(inj, out_dir / f"quads_inj_shuffle_r{r_i}.jsonl.gz")
            res = run_drift(f, f"inj_shuffle_r{r_i}", args.n_perm, out_dir)
            n_tests = sum(1 for r in res["results"]
                          for t in ("adj", "val_adj")
                          if r.get(t) and r[t].get("p_fdr") is not None)
            n_hit = sum(_sig(r, t) for r in res["results"] for t in ("adj", "val_adj"))
            fpr = n_hit / max(n_tests, 1)
            fprs.append(fpr)
            runs.append({"repeat": r_i, "fpr": round(fpr, 4),
                         "n_tests": n_tests, "n_flagged": n_hit})
            log.info("shuffle lặp %d/%d: FPR=%.4f", r_i + 1, args.repeats, fpr)
        mean_fpr = sum(fprs) / len(fprs)
        rep = {"empirical_fpr": round(mean_fpr, 4), "alpha": C.FDR_ALPHA,
               "repeats": runs,
               "verdict": "PASS" if mean_fpr <= C.FDR_ALPHA * 2 else "FAIL"}

    U.write_json(out_dir / f"injection_{args.test}.json", rep)
    log.info("KẾT QUẢ %s: %s", args.test, json.dumps(rep, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
