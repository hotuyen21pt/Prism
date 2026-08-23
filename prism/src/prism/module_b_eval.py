"""
Module B (đánh giá) — E1/E1b: F1 exact-match quad trên gold test,
tách theo ngôn ngữ / độ dài / cực tính / thành phần / category.

Exact match = trùng cả 4 thành phần (aspect_term chuẩn hoá, taxonomy_code,
opinion_term chuẩn hoá, sentiment) — cùng quy ước với evaluation/quadruple_metrics.py
của HAMoS.

Chạy:  python3 -m prism.module_b_eval --ckpt models/seed_extractor \
           --test-file outputs/extract/test.t2t.jsonl
Ra  :  outputs/extract/eval_report.json (+ .by_language / .by_length / .by_polarity)
"""
from __future__ import annotations

import argparse
import collections

from . import config as C
from . import utils as U
from .module_b_data import parse_linearized, TASK_PREFIX

log = U.get_logger("prism.B.eval")


def quad_key(q: dict) -> tuple:
    return (U.norm_text(q.get("aspect_term") or ""), q["taxonomy_code"],
            U.norm_text(q.get("opinion_term") or ""), q["sentiment"])


def prf(n_gold: int, n_pred: int, n_hit: int) -> dict:
    p = n_hit / n_pred if n_pred else 0.0
    r = n_hit / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"P": round(p, 4), "R": round(r, 4), "F1": round(f, 4),
            "gold": n_gold, "pred": n_pred, "hit": n_hit}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(C.MODEL_DIR / "seed_extractor"))
    ap.add_argument("--test-file", default=str(C.EXTRACT_DIR / "test.t2t.jsonl"))
    ap.add_argument("--gold-file", default=str(C.SPLIT_DIR / "test.jsonl"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-tgt", type=int, default=192)
    ap.add_argument("--out-prefix", default=str(C.EXTRACT_DIR / "eval"))
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.ckpt).to(device).eval()

    # ngôn ngữ theo review (từ gold meta)
    langs = {r["source_review_id"]: (r.get("languages") or ["?"])[0]
             for r in U.read_jsonl(C.GOLD_META)}
    gold_rows = {r["instance_id"]: r for r in U.read_jsonl(args.gold_file)}
    rows = list(U.read_jsonl(args.test_file))

    preds: dict[str, list[dict]] = {}
    with torch.no_grad():
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            enc = tok([r["input"] for r in chunk], return_tensors="pt",
                      padding=True, truncation=True, max_length=160).to(device)
            out = model.generate(**enc, max_length=args.max_tgt, num_beams=4)
            for r, o in zip(chunk, out):
                preds[r["instance_id"]] = parse_linearized(
                    tok.decode(o, skip_special_tokens=True))
            if i % (args.batch * 20) == 0:
                log.info("  ... %d/%d", i, len(rows))

    def agg(filt) -> dict:
        ng = np = nh = 0
        for iid, g in gold_rows.items():
            if not filt(g):
                continue
            gset = collections.Counter(quad_key(q) for q in g["quads"])
            pset = collections.Counter(quad_key(q) for q in preds.get(iid, []))
            ng += sum(gset.values()); np += sum(pset.values())
            nh += sum((gset & pset).values())
        return prf(ng, np, nh)

    seg_words = lambda g: len(g["text"].split())
    report = {
        "overall": agg(lambda g: True),
        "by_language": {lv: agg(lambda g, lv=lv: langs.get(g["source_review_id"]) == lv)
                        for lv in ("en", "vi", "other")},
        "by_length": {f"L{i}": agg(lambda g, i=i: C.LENGTH_BINS[i][0] <= seg_words(g)
                                   < C.LENGTH_BINS[i][1])
                      for i in range(len(C.LENGTH_BINS))},
        # cực tính (mức instance): instance vào bucket s nếu CÓ quad gold mang s
        "by_polarity": {s: agg(lambda g, s=s: any(q["sentiment"] == s
                                                  for q in g["quads"]))
                        for s in C.SENTIMENTS},
        # cực tính (mức QUAD): P/R/F1 chỉ trên quad mang đúng s — đây là số đo
        # trực tiếp cho recall lệch theo cực tính (M3a); bucket instance ở trên
        # trộn lẫn quad khác cực trong cùng instance nên không dùng được cho M3a
        "by_polarity_quad": {},
        "by_category": {cat: None for cat in C.CATEGORIES},
        "ckpt": args.ckpt,
    }
    # match chỉ trên tập con quad thoả bộ lọc (mức quad, không phải instance)
    def agg_quads(qfilter) -> dict:
        ng = np = nh = 0
        for iid, g in gold_rows.items():
            gset = collections.Counter(quad_key(q) for q in g["quads"] if qfilter(q))
            pset = collections.Counter(quad_key(q) for q in preds.get(iid, [])
                                       if qfilter(q))
            ng += sum(gset.values()); np += sum(pset.values())
            nh += sum((gset & pset).values())
        return prf(ng, np, nh)

    for cat in C.CATEGORIES:
        report["by_category"][cat] = agg_quads(
            lambda q, cat=cat: q.get("aspect_category") == cat)
    for s in C.SENTIMENTS:
        report["by_polarity_quad"][s] = agg_quads(
            lambda q, s=s: q["sentiment"] == s)

    U.write_json(U.Path(f"{args.out_prefix}_report.json"), report)
    U.write_jsonl(U.Path(f"{args.out_prefix}_preds.jsonl"),
                  ({"instance_id": k, "quads": v} for k, v in preds.items()))
    log.info("overall: %s", report["overall"])
    for k, v in report["by_language"].items():
        log.info("  lang %-6s %s", k, v)


if __name__ == "__main__":
    main()
