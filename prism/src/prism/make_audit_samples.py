"""
Tạo 2 mẫu annotation người (tách biệt — không được trộn lẫn):

  D0    ~300 review annotate ĐẦY ĐỦ (toàn bộ text pos+neg, không cắt)
        -> ước lượng hàm recall ρ̂(sentiment, length_bin) cho hiệu chỉnh D-a.
        Phân tầng theo length_bin × có/không text_neg.

  AUDIT ~300 pseudo-quad chấm đúng/sai
        -> fit cầu nối Module C (bridge) và hiệu chỉnh reliability.
        Phân tầng theo conf_seq (thấp/vừa/cao) × phi × provenance_flip.

Chạy:  python3 -m prism.make_audit_samples --quads outputs/extract/pool_quads.T-unbiased.jsonl.gz
Ra  :  outputs/reliability/d0_sample_300.jsonl        (annotate: thêm trường "gold_quads": [...])
       outputs/reliability/audit_sample_300.jsonl     (annotate: thêm trường "correct": 0|1)
"""
from __future__ import annotations

import argparse
import collections
import random

from . import config as C
from . import utils as U

log = U.get_logger("prism.audit")


def d0_sample(n_total: int, rng: random.Random) -> None:
    strata: dict[tuple, list] = collections.defaultdict(list)
    seen: collections.Counter = collections.Counter()
    CAP = 200
    for r in U.read_jsonl(C.STORE_DIR / "reviews.jsonl.gz"):
        if r["in_gold"] or not r["has_text"]:
            continue
        key = (U.length_bin(r["n_words"]), bool(r["text_neg"]))
        # reservoir sampling chuẩn (Algorithm R) từng ô — mỗi phần tử của ô
        # có xác suất được giữ ĐÚNG bằng nhau; xác suất cố định (bản cũ) làm
        # mẫu lệch về phía các dòng đến sau
        seen[key] += 1
        bucket = strata[key]
        if len(bucket) < CAP:
            bucket.append(r)
        else:
            j = rng.randrange(seen[key])
            if j < CAP:
                bucket[j] = r
    per_cell = max(1, n_total // max(len(strata), 1))
    rows = []
    for key, bucket in sorted(strata.items()):
        take = rng.sample(bucket, min(per_cell, len(bucket)))
        for r in take:
            rows.append({
                "review_uid": r["review_uid"], "period": r["period"],
                "length_bin": key[0], "has_neg": key[1],
                "text_pos": r["text_pos"], "text_neg": r["text_neg"],
                "gold_quads": [],       # <-- người annotate điền TOÀN BỘ quad
            })
    rng.shuffle(rows)
    out = C.RELIAB_DIR / "d0_sample_300.jsonl"
    U.write_jsonl(out, rows[:n_total])
    log.info("D0: %d review (%d ô strata) -> %s", min(len(rows), n_total), len(strata), out)


def quad_audit_sample(quad_file, n_total: int, rng: random.Random) -> None:
    def band(c): return "lo" if c < 0.6 else "mid" if c < 0.85 else "hi"
    strata: dict[tuple, list] = collections.defaultdict(list)
    seen: collections.Counter = collections.Counter()
    CAP = 300
    for q in U.read_jsonl(quad_file):
        key = (band(q["conf_seq"]), q["phi"], q["provenance_flip"])
        seen[key] += 1
        bucket = strata[key]
        if len(bucket) < CAP:
            bucket.append(q)
        else:
            j = rng.randrange(seen[key])
            if j < CAP:
                bucket[j] = q
    per_cell = max(1, n_total // max(len(strata), 1))
    rows = []
    for key, bucket in sorted(strata.items()):
        for q in rng.sample(bucket, min(per_cell, len(bucket))):
            rows.append({
                "quad_uid": U.quad_uid(q),   # PHẢI trùng khoá module_c dùng đối chiếu
                "text": q.get("text"), "phi": q["phi"],
                "aspect_term": q.get("aspect_term"),
                "taxonomy_code": q["taxonomy_code"],
                "opinion_term": q.get("opinion_term"),
                "sentiment": q["sentiment"],
                "conf_band": key[0], "provenance_flip": key[2],
                "correct": None,        # <-- người annotate điền 0|1
            })
    rng.shuffle(rows)
    out = C.RELIAB_DIR / "audit_sample_300.jsonl"
    U.write_jsonl(out, rows[:n_total])
    log.info("AUDIT: %d quad (%d ô strata) -> %s", min(len(rows), n_total), len(strata), out)


def main_with_args(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quads", default=None,
                    help="pool_quads.*.jsonl.gz — bỏ trống nếu chỉ cần D0")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    ap.add_argument("--only", choices=["both", "d0", "audit"], default="both",
                    help="'audit' = chỉ sinh mẫu AUDIT (§3.6). D0 (§3.2) đọc "
                         "reviews.jsonl.gz và GHI ĐÈ template đã giao annotate, "
                         "nên đừng kéo nó theo khi chỉ cần AUDIT.")
    args = ap.parse_args(argv)
    C.ensure_dirs()
    if args.only in ("both", "d0"):
        d0_sample(args.n, random.Random(args.seed))
    if args.only in ("both", "audit"):
        if not args.quads:
            ap.error("--only audit cần --quads pool_quads.<cohort>.jsonl.gz")
        # RNG RIÊNG cho AUDIT: nếu dùng chung stream với D0 thì mẫu AUDIT phụ
        # thuộc vào việc D0 có chạy cùng lệnh hay không -> cùng --seed cho ra
        # hai file khác nhau, không tái lập được.
        quad_audit_sample(args.quads, args.n, random.Random(args.seed + 1))


def main() -> None:
    main_with_args()


if __name__ == "__main__":
    main()
