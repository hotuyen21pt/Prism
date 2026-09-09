"""
Module B (self-training) — vòng lặp teacher-student có kiểm soát.

Vòng 0: đo dev F1 của TEACHER GỐC làm mốc — không có mốc này thì vòng 1 luôn được
nhận kể cả khi nó làm dev tụt, và final_ckpt có thể tệ hơn seed mà không ai biết.

Mỗi vòng tiếp theo:
  1. infer trên MẪU NGẪU NHIÊN của pool (cohort chỉ định, KHÔNG gồm dòng in_gold);
     --sample-seed đổi theo vòng nên vòng sau thấy dữ liệu mới
  2. chọn pseudo-label: conf_seq >= --tau  VÀ  p_posterior >= --tau-post
     (posterior mềm thay bộ lọc cứng; quad provenance_flip vẫn được giữ nếu đủ posterior)
     + thêm --empty-ratio instance target-RỖNG từ unit không có quad nào đạt ngưỡng
  3. trộn D_gold ∪ D_pseudo (giới hạn tỷ lệ --max-ratio pseudo/gold) rồi train tiếp
  4. đo dev F1; CHỈ nhận checkpoint khi vượt mốc TỐT NHẤT đã thấy (kể cả seed).
     DỪNG khi không vượt, hoặc khi phân phối lớp pseudo lệch quá --max-skew

Chạy:  python3 -m prism.module_b_selftrain --rounds 2 --cohort B-anchor
Ra  :  outputs/extract/selftrain_history.json — final_ckpt + final_dev_f1
       (kaggle_pipeline.find_ckpt đọc file này để chọn checkpoint cho infer)
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import subprocess
import sys

from . import config as C
from . import utils as U
from .module_b_data import linearize, parse_linearized, TASK_PREFIX

log = U.get_logger("prism.B.selftrain")


def gold_negative_rate(gold_rows: list[dict]) -> float:
    """Tỷ lệ quad mang sentiment=negative trong gold train — ĐO TẠI CHỖ.

    Trước đây hardcode 0.154; nếu gold được cập nhật thì guardrail --max-skew so
    với một con số đã chết và không còn phát hiện được error propagation.
    Đọc trực tiếp từ target đã tuyến tính hoá nên không cần file gold thứ hai.
    """
    n_neg = n_tot = 0
    for r in gold_rows:
        for q in parse_linearized(r["target"]):
            n_tot += 1
            n_neg += q["sentiment"] == "negative"
    return n_neg / n_tot if n_tot else 0.0


def eval_dev(ckpt: str, out_prefix: str) -> float:
    """Chạy module_b_eval trên dev, trả overall F1."""
    subprocess.run([sys.executable, "-m", "prism.module_b_eval",
                    "--ckpt", ckpt,
                    "--test-file", str(C.EXTRACT_DIR / "dev.t2t.jsonl"),
                    "--gold-file", str(C.SPLIT_DIR / "dev.jsonl"),
                    "--out-prefix", str(C.EXTRACT_DIR / out_prefix)], check=True)
    rep = json.loads((C.EXTRACT_DIR / f"{out_prefix}_report.json").read_text())
    return rep["overall"]["F1"]


def pseudo_rows(quad_file, tau: float, tau_post: float, max_n: int, seed: int,
                empty_ratio: float = 0.0):
    """Gom quad theo (review_uid, phi) thành instance t2t; lọc theo ngưỡng.

    Unit mà MỌI quad đều dưới ngưỡng không bị bỏ hẳn: một phần của chúng được giữ
    lại làm ví dụ TARGET RỖNG (theo tỷ lệ empty_ratio so với số instance có quad).
    Bỏ hết như bản cũ dạy student rằng "input nào cũng có quad để trích" -> student
    over-generate, precision tụt trong khi recall tăng, và guardrail --max-skew
    (chỉ xem phân phối cực tính) không phát hiện được.
    """
    by_unit: dict[tuple, list[dict]] = collections.defaultdict(list)
    empty_units: dict[tuple, str] = {}
    for q in U.read_jsonl(quad_file):
        key = (q["review_uid"], q["phi"])
        if q["conf_seq"] >= tau and q["p_posterior"] >= tau_post:
            by_unit[key].append(q)
        else:
            empty_units.setdefault(key, q["text"])

    def row(iid, text, quads):
        return {
            "instance_id": iid,
            "input": TASK_PREFIX + text,
            "target": linearize(quads),
            "n_quads": len(quads),
            "sent_dist": collections.Counter(q["sentiment"] for q in quads),
        }

    rows = [row(f"PSEUDO_{uid}_{phi}", quads[0]["text"], quads)
            for (uid, phi), quads in by_unit.items()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:max_n]

    if empty_ratio > 0:
        # chỉ unit KHÔNG có quad nào vượt ngưỡng mới đủ điều kiện làm target rỗng
        cand = [(k, t) for k, t in empty_units.items() if k not in by_unit]
        rng.shuffle(cand)
        n_empty = min(len(cand), int(empty_ratio * len(rows)))
        rows += [row(f"PSEUDO_EMPTY_{uid}_{phi}", text, [])
                 for (uid, phi), text in cand[:n_empty]]
        rng.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--cohort", default="B-anchor")
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--tau-post", type=float, default=0.8)
    ap.add_argument("--max-ratio", type=float, default=3.0,
                    help="tối đa pseudo = ratio × |gold train| mỗi vòng")
    ap.add_argument("--max-skew", type=float, default=0.10,
                    help="dừng nếu %%negative của pseudo lệch quá mức này so với gold")
    ap.add_argument("--infer-limit", type=int, default=200_000)
    ap.add_argument("--empty-ratio", type=float, default=0.1,
                    help="tỷ lệ instance target-RỖNG thêm vào pseudo (chống over-generate)")
    ap.add_argument("--batch", type=int, default=32,
                    help="batch generate của bước infer bên trong (giảm khi ít VRAM)")
    ap.add_argument("--score-batch", type=int, default=48,
                    help="batch rescore của bước infer bên trong")
    args = ap.parse_args()

    gold_train = list(U.read_jsonl(C.EXTRACT_DIR / "train.t2t.jsonl"))
    gold_neg_rate = gold_negative_rate(gold_train)
    log.info("gold train: %d instance · neg_rate=%.4f (đo tại chỗ, không hardcode)",
             len(gold_train), gold_neg_rate)
    seed_ckpt = str(C.MODEL_DIR / "seed_extractor")
    ckpt = seed_ckpt
    history = []

    # Mốc so sánh BẮT BUỘC: dev F1 của teacher gốc. Không có mốc này thì vòng 1
    # luôn được nhận, kể cả khi nó làm dev F1 tụt -> final_ckpt tệ hơn seed mà
    # không log nào nói ra.
    best_f1 = eval_dev(seed_ckpt, "dev_round0")
    history.append({"round": 0, "ckpt": seed_ckpt, "dev_f1": best_f1,
                    "note": "teacher gốc (mốc so sánh)"})
    log.info("vòng 0 (seed) dev F1 = %.4f", best_f1)

    for rnd in range(1, args.rounds + 1):
        log.info("=== VÒNG %d/%d — teacher=%s ===", rnd, args.rounds, ckpt)
        # 1. infer — seed mẫu ĐỔI theo vòng để vòng sau thấy dữ liệu mới
        subprocess.run([sys.executable, "-m", "prism.module_b_infer",
                        "--ckpt", ckpt, "--cohort", args.cohort,
                        "--batch", str(args.batch),
                        "--score-batch", str(args.score_batch),
                        "--limit", str(args.infer_limit),
                        "--sample-seed", str(C.RANDOM_SEED + rnd)], check=True)
        qf = C.EXTRACT_DIR / C.pool_quads_name(args.cohort)

        # 2. chọn pseudo + guardrail phân phối lớp
        max_n = int(args.max_ratio * len(gold_train))
        pseudo = pseudo_rows(qf, args.tau, args.tau_post, max_n,
                             C.RANDOM_SEED + rnd, args.empty_ratio)
        n_empty = sum(1 for r in pseudo if r["n_quads"] == 0)
        sent = collections.Counter()
        for r in pseudo:
            sent.update(r.pop("sent_dist"))
        tot = sum(sent.values()) or 1
        neg_rate = sent["negative"] / tot
        log.info("pseudo: %d instance (%d target rỗng) · phân phối %s · "
                 "neg_rate=%.3f (gold %.3f)",
                 len(pseudo), n_empty, dict(sent), neg_rate, gold_neg_rate)
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

        # 4. đánh giá dev — CHỈ nhận checkpoint mới khi vượt mốc TỐT NHẤT đã thấy
        #    (kể cả seed teacher), không phải chỉ so với vòng ngay trước.
        f1 = eval_dev(new_ckpt, f"dev_round{rnd}")
        history.append({"round": rnd, "ckpt": new_ckpt, "dev_f1": f1,
                        "n_pseudo": len(pseudo), "n_empty": n_empty,
                        "neg_rate": round(neg_rate, 4)})
        log.info("vòng %d dev F1 = %.4f (tốt nhất đang là %.4f)", rnd, f1, best_f1)
        if f1 <= best_f1:
            log.warning("DỪNG: dev F1 không vượt mốc tốt nhất (%.4f -> %.4f). "
                        "GIỮ checkpoint %s", best_f1, f1, ckpt)
            break
        best_f1, ckpt = f1, new_ckpt

    U.write_json(C.EXTRACT_DIR / "selftrain_history.json",
                 {"history": history, "final_ckpt": ckpt,
                  "final_dev_f1": best_f1, "seed_ckpt": seed_ckpt,
                  "args": vars(args)})
    log.info("kết thúc. checkpoint cuối: %s (dev F1 = %.4f)", ckpt, best_f1)
    if ckpt == seed_ckpt:
        log.warning("Self-train KHÔNG cải thiện được teacher gốc — infer chính thức "
                    "nên dùng seed_extractor.")


if __name__ == "__main__":
    main()
