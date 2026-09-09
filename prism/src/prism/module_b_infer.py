"""
Module B (suy luận pool) — chạy extractor trên store, áp provenance posterior.

Mỗi review trong store cho tối đa 2 đơn vị suy luận: (text_pos, φ=POS) và
(text_neg, φ=NEG) — nguồn trường đi kèm từng đoạn, không phải cả review.

Provenance posterior (KHÔNG lọc cứng — xem docs/method_table_q1.md §B.3):
    P(s | x, φ)  ∝  P_model(s | x)^λ · P(s | φ)^(1-λ)
với P(s|φ) đo trên gold: POS=(93,1 / 3,6 / 3,3)  NEG=(21,7 / 14,2 / 64,1).

P_model(s|x) là XÁC SUẤT THẬT của model: với mỗi quad sinh ra, chấm điểm lại
3 biến thể target (chỉ đổi sentiment của quad đó) bằng teacher forcing rồi
softmax trên tổng log-prob. KHÔNG dùng one-hot xấp xỉ — nếu dùng one-hot,
"posterior" chỉ là hàm tất định của (s dự đoán, φ) và λ mất ý nghĩa.
(--no-rescore để quay về xấp xỉ one-hot khi cần tốc độ; kết quả chính thức
 của paper PHẢI chạy với rescore.)

Quad có s hậu nghiệm ≠ s model → gắn cờ provenance_flip (giữ lại, để audit).

Chạy:  python3 -m prism.module_b_infer --ckpt models/seed_extractor \
           --cohort T-unbiased           # chạy cohort nhỏ trước, corpus sau
Ra  :  outputs/extract/pool_quads.<cohort>.jsonl.gz
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json as _json
import math
import random

from . import config as C
from . import utils as U
from .module_b_data import parse_linearized, linearize, TASK_PREFIX

log = U.get_logger("prism.B.infer")

# Nếu 3 biến thể sentiment của cùng một quad cho log-prob gần như bằng nhau thì
# chúng đã tokenize giống nhau — nghĩa là quad đang xét bị TRUNCATE khỏi target
# (unit nhiều quad, vượt MAX_TGT_TOKENS). Khi đó p_model là uniform và posterior
# thoái hoá thành prior thuần: sentiment do φ quyết định hoàn toàn, im lặng.
DEGENERATE_LP_EPS = 1e-6


def apply_provenance(sent_scores: dict[str, float], phi: str,
                     lam: float) -> tuple[str, float]:
    """Kết hợp P_model 3 lớp với prior P(s|φ). Trả (s*, posterior)."""
    prior = C.PROVENANCE_PRIOR[phi]
    post = {s: lam * math.log(max(sent_scores.get(s, 1e-9), 1e-9))
               + (1 - lam) * math.log(prior[s]) for s in C.SENTIMENTS}
    z = max(post.values())
    exp = {s: math.exp(v - z) for s, v in post.items()}
    tot = sum(exp.values())
    best = max(exp, key=exp.get)
    return best, exp[best] / tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(C.MODEL_DIR / "seed_extractor"))
    ap.add_argument("--cohort", default="T-unbiased",
                    choices=C.COHORTS)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--score-batch", type=int, default=48,
                    help="batch cho bước rescore sentiment")
    ap.add_argument("--lam", type=float, default=C.PROVENANCE_LAMBDA)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = không giới hạn; >0 = lấy MẪU NGẪU NHIÊN đều "
                         "(reservoir) trên toàn cohort, không phải N unit đầu file")
    ap.add_argument("--sample-seed", type=int, default=C.RANDOM_SEED,
                    help="seed của mẫu --limit; self-train PHẢI đổi seed mỗi vòng "
                         "để vòng sau thấy dữ liệu mới")
    ap.add_argument("--no-rescore", action="store_true",
                    help="bỏ bước chấm điểm P_model(s|x) thật (nhanh hơn, "
                         "posterior thoái hoá thành hàm tất định — chỉ để debug)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.ckpt).to(device).eval()
    log.info("device=%s cohort=%s λ=%.2f rescore=%s",
             device, args.cohort, args.lam, not args.no_rescore)

    cohorts = _json.loads((C.STORE_DIR / "hotel_cohorts.json").read_text())
    keep = set(cohorts.get(args.cohort) or [])   # rỗng = corpus = mọi hotel

    def units():
        for r in U.read_jsonl(C.STORE_DIR / "reviews.jsonl.gz"):
            if r["in_gold"]:            # BLOCKLIST: không suy luận trên dòng gold
                continue
            if keep and r["hotel_id"] not in keep:
                continue
            for phi, key in (("POS", "text_pos"), ("NEG", "text_neg")):
                if r[key]:
                    yield {"review_uid": r["review_uid"], "hotel_id": r["hotel_id"],
                           "period": r["period"], "stratum": r["stratum"],
                           "phi": phi, "text": r[key], "score": r["score"],
                           "has_photo": r["has_photo"], "n_words": r["n_words"]}

    def _score(inputs: list[str], targets: list[str]) -> tuple[list[float], list[float]]:
        """Chấm điểm target | input bằng teacher forcing, theo chunk.

        Trả (tổng log-prob, TRUNG BÌNH log-prob) cho từng cặp — hai thống kê duy
        nhất mà pipeline cần, tính trong CÙNG một lượt forward để không phải chạy
        model hai lần trên cùng dữ liệu (trước đây là hai hàm sao chép nhau).

        Thay cho generate(output_scores=True)+compute_transition_scores — vốn giữ
        logits full-vocab (~250k) cho MỌI bước sinh nên OOM. Ở đây chỉ gather
        log-prob của đúng token đã chọn, chunk theo score_batch."""
        sums: list[float] = []
        means: list[float] = []
        for i in range(0, len(inputs), args.score_batch):
            enc = tok(inputs[i:i + args.score_batch], return_tensors="pt",
                      padding=True, truncation=True,
                      max_length=C.MAX_SRC_TOKENS).to(device)
            lab = tok(targets[i:i + args.score_batch], return_tensors="pt",
                      padding=True, truncation=True,
                      max_length=C.MAX_TGT_TOKENS).input_ids.to(device)
            lab_m = lab.clone()
            lab_m[lab_m == tok.pad_token_id] = -100
            with torch.no_grad():
                logits = model(**enc, labels=lab_m).logits
            lp = torch.log_softmax(logits.float(), dim=-1)
            tok_lp = lp.gather(-1, lab.unsqueeze(-1)).squeeze(-1)
            mask = (lab_m != -100).float()
            tot = (tok_lp * mask).sum(-1)
            sums.extend(tot.tolist())
            means.extend((tot / mask.sum(-1).clamp(min=1)).tolist())
        return sums, means

    def seq_logprobs(inputs: list[str], targets: list[str]) -> list[float]:
        """Tổng log-prob của target | input."""
        return _score(inputs, targets)[0]

    def seq_conf(inputs: list[str], targets: list[str]) -> list[float]:
        """conf_seq = exp(trung bình log-prob token) — độ tin cậy mức chuỗi."""
        return [math.exp(m) for m in _score(inputs, targets)[1]]

    stats = collections.Counter()

    def flush(batch, fout):
        enc = tok([TASK_PREFIX + b["text"] for b in batch], return_tensors="pt",
                  padding=True, truncation=True,
                  max_length=C.MAX_SRC_TOKENS).to(device)
        with torch.no_grad():
            seqs = model.generate(**enc, max_length=C.MAX_TGT_TOKENS, num_beams=4)

        texts_out = [tok.decode(s, skip_special_tokens=True) for s in seqs]
        # seq-level confidence tính bằng teacher forcing (nhẹ VRAM, không giữ full-vocab)
        confs = seq_conf([TASK_PREFIX + b["text"] for b in batch], texts_out)
        parsed = [(b, confs[i], parse_linearized(texts_out[i]))   # (unit, conf, quads)
                  for i, b in enumerate(batch)]

        # P_model(s|x) thật: chấm điểm 3 biến thể sentiment cho từng quad
        p_models: dict[tuple[int, int], dict[str, float]] = {}
        degenerate: set[tuple[int, int]] = set()
        if not args.no_rescore:
            jobs, ins, tgts = [], [], []
            for bi, (b, _, quads) in enumerate(parsed):
                for qi in range(len(quads)):
                    for s in C.SENTIMENTS:
                        variant = [dict(q) for q in quads]
                        variant[qi]["sentiment"] = s
                        jobs.append((bi, qi, s))
                        ins.append(TASK_PREFIX + b["text"])
                        tgts.append(linearize(variant))
            if jobs:
                lps = seq_logprobs(ins, tgts)
                acc: dict[tuple[int, int], dict[str, float]] = {}
                for (bi, qi, s), lp in zip(jobs, lps):
                    acc.setdefault((bi, qi), {})[s] = lp
                for key, d in acc.items():
                    # 3 biến thể cho log-prob y hệt nhau => quad này đã bị truncate
                    # khỏi target, p_model sẽ là uniform và posterior = prior thuần.
                    # Đánh dấu thay vì để nó trôi vào dữ liệu như một số bình thường.
                    if max(d.values()) - min(d.values()) < DEGENERATE_LP_EPS:
                        degenerate.add(key)
                        continue
                    z = max(d.values())
                    e = {s: math.exp(v - z) for s, v in d.items()}
                    t = sum(e.values())
                    p_models[key] = {s: v / t for s, v in e.items()}

        for bi, (b, conf, quads) in enumerate(parsed):
            for qi, q in enumerate(quads):
                scored = p_models.get((bi, qi))
                is_degen = (bi, qi) in degenerate
                # Fallback one-hot xấp xỉ: dùng khi --no-rescore, hoặc khi rescore
                # thoái hoá vì truncate. Trong cả hai trường hợp λ mất ý nghĩa nên
                # phải ghi cờ ra dữ liệu để Module C/D lọc được.
                p_model = scored or {
                    s: (0.9 if s == q["sentiment"] else 0.05) for s in C.SENTIMENTS}
                stats["quads"] += 1
                if is_degen:
                    stats["p_model_truncated"] += 1
                elif scored is None:
                    stats["p_model_onehot"] += 1
                s_post, p_post = apply_provenance(p_model, b["phi"], args.lam)
                q.update({
                    "p_model_exact": scored is not None,
                    "p_model_truncated": is_degen,
                    "text": b["text"],
                    "review_uid": b["review_uid"], "hotel_id": b["hotel_id"],
                    "period": b["period"], "stratum": b["stratum"],
                    "phi": b["phi"], "score": b["score"],
                    "has_photo": b["has_photo"], "n_words": b["n_words"],
                    "conf_seq": round(conf, 4),
                    "sentiment_model": q["sentiment"],
                    "p_model": {s: round(v, 4) for s, v in p_model.items()},
                    "sentiment": s_post,
                    "p_posterior": round(p_post, 4),
                    "provenance_flip": s_post != q["sentiment"],
                })
                fout.write(_json.dumps(q, ensure_ascii=False) + "\n")

    def sampled_units():
        """--limit > 0: lấy mẫu ĐỀU trên toàn cohort bằng reservoir sampling
        (Algorithm R). Bản cũ cắt N unit ĐẦU FILE, mà store ghi theo thứ tự pool
        (gom theo hotel) — nên mẫu lệch về một tập con hotel, và mọi vòng self-train
        đọc lại đúng cùng slice đó (không có dữ liệu mới, chỉ tự khuếch đại lỗi)."""
        if not args.limit:
            yield from units()
            return
        rng = random.Random(args.sample_seed)
        pool: list[dict] = []
        seen = 0
        for u in units():
            seen += 1
            if len(pool) < args.limit:
                pool.append(u)
            else:
                j = rng.randrange(seen)
                if j < args.limit:
                    pool[j] = u
        log.info("mẫu %d/%d đơn vị (reservoir, seed=%d)",
                 len(pool), seen, args.sample_seed)
        rng.shuffle(pool)
        yield from pool

    out_path = C.EXTRACT_DIR / C.pool_quads_name(args.cohort)
    n_units = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as fout:
        batch = []
        for u in sampled_units():
            batch.append(u); n_units += 1
            if len(batch) == args.batch:
                flush(batch, fout); batch = []
                if n_units % (args.batch * 50) == 0:
                    log.info("  ... %d đơn vị", n_units)
        if batch:
            flush(batch, fout)

    n_quads = stats["quads"]
    log.info("xong %d đơn vị · %d quad -> %s", n_units, n_quads, out_path)
    if stats["p_model_truncated"]:
        log.warning("%d/%d quad (%.2f%%) bị TRUNCATE khỏi target lúc rescore -> "
                    "p_model thoái hoá, posterior = prior thuần (cờ p_model_truncated). "
                    "Đây là unit nhiều quad vượt MAX_TGT_TOKENS=%d.",
                    stats["p_model_truncated"], n_quads,
                    100 * stats["p_model_truncated"] / max(n_quads, 1),
                    C.MAX_TGT_TOKENS)
    if stats["p_model_onehot"]:
        log.warning("%d/%d quad dùng xấp xỉ one-hot (--no-rescore) — λ KHÔNG có ý "
                    "nghĩa, không dùng cho số chính thức",
                    stats["p_model_onehot"], n_quads)
    if n_quads == 0:
        log.error("KHÔNG có quad nào được ghi — file output rỗng. Kiểm tra checkpoint "
                  "và cohort trước khi chạy Module C/D trên file này.")


if __name__ == "__main__":
    main()
