"""
Module B (self-training) — vòng lặp teacher-student có kiểm soát.

Mỗi vòng:
  1. infer trên một mẫu pool (cohort chỉ định, KHÔNG gồm dòng in_gold)
  2. chọn pseudo-label: conf_seq >= --tau  VÀ  p_posterior >= --tau-post
     (posterior mềm thay bộ lọc cứng; quad provenance_flip vẫn được giữ nếu đủ posterior)
  3. trộn D_gold ∪ D_pseudo (giới hạn tỷ lệ --max-ratio pseudo/gold) rồi train tiếp
  4. đo dev macro-F1; DỪNG khi dev giảm hoặc phân phối lớp pseudo lệch quá --max-skew

Chạy:  python3 -m prism.module_b_selftrain --rounds 2 --cohort B-anchor
"""
from __future__ import annotations

import argparse
import collections
import random
import subprocess
import sys

from . import config as C
from . import utils as U
from .module_b_data import linearize, TASK_PREFIX

log = U.get_logger("prism.B.selftrain")


def pseudo_rows(quad_file, tau: float, tau_post: float, max_n: int, seed: int):
    """Gom quad theo (review_uid, phi) thành instance t2t; lọc theo ngưỡng."""
    by_unit: dict[tuple, list[dict]] = collections.defaultdict(list)
    for q in U.read_jsonl(quad_file):
        if q["conf_seq"] >= tau and q["p_posterior"] >= tau_post:
            by_unit[(q["review_uid"], q["phi"])].append(q)
    rows = []
    for (uid, phi), quads in by_unit.items():
        rows.append({
            "instance_id": f"PSEUDO_{uid}_{phi}",
            "input": TASK_PREFIX + quads[0]["text"],
            "target": linearize(quads),
            "n_quads": len(quads),
            "sent_dist": collections.Counter(q["sentiment"] for q in quads),
        })
    random.Random(seed).shuffle(rows)
    return rows[:max_n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--cohort", default="B-anchor")
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--tau-post", type=float, default=0.8)
    ap.add_argument("--max-ratio", type=float, default=3.0,
                    help="tối đa pseudo = ratio × |gold train| mỗi vòng")
    ap.add_argument("--max-skew", type=float, default=0.10,
                    help="dừng nếu %%negative của pseudo lệch quá mức này so với gold (15,4%%)")
    ap.add_argument("--infer-limit", type=int, default=200_000)
    args = ap.parse_args()

    gold_train = list(U.read_jsonl(C.EXTRACT_DIR / "train.t2t.jsonl"))
    gold_neg_rate = 0.154        # [đo] phân phối gold
    ckpt = str(C.MODEL_DIR / "seed_extractor")
    history = []

    for rnd in range(1, args.rounds + 1):
        log.info("=== VÒNG %d/%d — teacher=%s ===", rnd, args.rounds, ckpt)
        # 1. infer
        subprocess.run([sys.executable, "-m", "prism.module_b_infer",
                        "--ckpt", ckpt, "--cohort", args.cohort,
                        "--limit", str(args.infer_limit)], check=True)
        qf = C.EXTRACT_DIR / f"pool_quads.{args.cohort}.jsonl.gz"

        # 2. chọn pseudo + guardrail phân phối lớp
        max_n = int(args.max_ratio * len(gold_train))
        pseudo = pseudo_rows(qf, args.tau, args.tau_post, max_n, C.RANDOM_SEED + rnd)
        sent = collections.Counter()
        for r in pseudo:
            sent.update(r.pop("sent_dist"))
        tot = sum(sent.values()) or 1
        neg_rate = sent["negative"] / tot
        log.info("pseudo: %d instance · phân phối %s · neg_rate=%.3f (gold %.3f)",
                 len(pseudo), dict(sent), neg_rate, gold_neg_rate)
        if abs(neg_rate - gold_neg_rate) > args.max_skew:
            log.warning("DỪNG: phân phối pseudo lệch quá %.2f — dấu hiệu error propagation",
                        args.max_skew)
            break

        # 3. train student trên gold ∪ pseudo
        mixed = C.EXTRACT_DIR / f"selftrain_round{rnd}.t2t.jsonl"
        U.write_jsonl(mixed, gold_train + pseudo)
        new_ckpt = str(C.MODEL_DIR / f"selftrain_round{rnd}")
        subprocess.run([sys.executable, "-m", "prism.module_b_train",
                        "--model", ckpt, "--train-file", str(mixed),
                        "--out", new_ckpt, "--epochs", "3"], check=True)

        # 4. đánh giá dev — dừng khi giảm
        subprocess.run([sys.executable, "-m", "prism.module_b_eval",
                        "--ckpt", new_ckpt,
                        "--test-file", str(C.EXTRACT_DIR / "dev.t2t.jsonl"),
                        "--gold-file", str(C.SPLIT_DIR / "dev.jsonl"),
                        "--out-prefix", str(C.EXTRACT_DIR / f"dev_round{rnd}")],
                       check=True)
        import json as _json
        rep = _json.loads((C.EXTRACT_DIR / f"dev_round{rnd}_report.json").read_text())
        f1 = rep["overall"]["F1"]
        history.append({"round": rnd, "dev_f1": f1, "n_pseudo": len(pseudo),
                        "neg_rate": round(neg_rate, 4)})
        log.info("vòng %d dev F1 = %.4f", rnd, f1)
        if len(history) > 1 and f1 < history[-2]["dev_f1"]:
            log.warning("DỪNG: dev F1 giảm (%.4f -> %.4f)", history[-2]["dev_f1"], f1)
            break
        ckpt = new_ckpt

    U.write_json(C.EXTRACT_DIR / "selftrain_history.json",
                 {"history": history, "final_ckpt": ckpt, "args": vars(args)})
    log.info("kết thúc. checkpoint cuối: %s", ckpt)


if __name__ == "__main__":
    main()
