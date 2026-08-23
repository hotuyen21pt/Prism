"""
Module B (huấn luyện) — fine-tune mT5 làm seed extractor trên gold train.

Yêu cầu: torch + transformers (xem README §0). Tự chọn device cuda/mps/cpu.

Chạy:  python3 -m prism.module_b_train --model google/mt5-base --epochs 10
Ra  :  models/seed_extractor/  (checkpoint + training_log.json)

Sau huấn luyện, đánh giá bằng:  python3 -m prism.module_b_eval
"""
from __future__ import annotations

import argparse
import json
import random
import time

from . import config as C
from . import utils as U

log = U.get_logger("prism.B.train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/mt5-base",
                    help="mT5 vì gold đa ngữ (80,5%% en · 16,2%% vi)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-src", type=int, default=160)   # segment mean 13 từ, max 147
    ap.add_argument("--max-tgt", type=int, default=192)   # max 23 quad/segment
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    ap.add_argument("--train-file", default=str(C.EXTRACT_DIR / "train.t2t.jsonl"),
                    help="đổi sang chrono_train.t2t.jsonl cho probe E1c")
    ap.add_argument("--dev-file", default=str(C.EXTRACT_DIR / "dev.t2t.jsonl"))
    ap.add_argument("--out", default=str(C.MODEL_DIR / "seed_extractor"))
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    log.info("device=%s  model=%s", device, args.model)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)

    class T2T(Dataset):
        def __init__(self, path):
            self.rows = list(U.read_jsonl(path))
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, i):
            return self.rows[i]

    def collate(batch):
        enc = tok([b["input"] for b in batch], max_length=args.max_src,
                  truncation=True, padding=True, return_tensors="pt")
        lab = tok([b["target"] for b in batch], max_length=args.max_tgt,
                  truncation=True, padding=True, return_tensors="pt").input_ids
        lab[lab == tok.pad_token_id] = -100
        enc["labels"] = lab
        return enc

    train_dl = DataLoader(T2T(args.train_file), batch_size=args.batch,
                          shuffle=True, collate_fn=collate)
    dev_dl = DataLoader(T2T(args.dev_file), batch_size=args.batch, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total = len(train_dl) * args.epochs // args.grad_accum
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)

    best_dev, hist = float("inf"), []
    outdir = U.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tr_loss = time.time(), 0.0
        for step, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            tr_loss += loss.item() * args.grad_accum
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        model.eval()
        dv_loss = 0.0
        with torch.no_grad():
            for batch in dev_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                dv_loss += model(**batch).loss.item()
        tr_loss /= len(train_dl); dv_loss /= len(dev_dl)
        hist.append({"epoch": ep, "train_loss": round(tr_loss, 4),
                     "dev_loss": round(dv_loss, 4), "sec": round(time.time() - t0)})
        log.info("epoch %d  train=%.4f  dev=%.4f  (%.0fs)", ep, tr_loss, dv_loss,
                 time.time() - t0)
        if dv_loss < best_dev:                      # early keep-best theo dev loss
            best_dev = dv_loss
            model.save_pretrained(outdir); tok.save_pretrained(outdir)
            log.info("  ↳ saved best -> %s", outdir)

    U.write_json(outdir / "training_log.json", {
        "args": vars(args), "history": hist, "best_dev_loss": best_dev,
        "device": device,
    })


if __name__ == "__main__":
    main()
