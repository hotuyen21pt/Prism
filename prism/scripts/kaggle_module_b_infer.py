"""Kaggle entry point cho PRISM Module B — suy luận pool (infer).

Thiết lập môi trường Kaggle rồi gọi prism.module_b_infer:
  - clone repo (nếu chưa có)
  - vá bug KeyError 'sentiment_model' cho bản GitHub cũ (idempotent)
  - nạp checkpoint từ /kaggle/input (tìm model.safetensors)
  - nạp store của Module A vào PRISM_WORK_DIR/store; nếu chỉ có reviews.jsonl
    (chưa nén) thì tự nén thành reviews.jsonl.gz vì code đọc bằng gzip
  - batch nhỏ + expandable_segments để tránh OOM (mT5 vocab ~250k)

Chạy trong 1 cell Kaggle (bật GPU):
    !python /kaggle/working/Prism/prism/scripts/kaggle_module_b_infer.py \
        --cohort T-unbiased --batch 4 --score-batch 8
hoặc import và gọi main().
"""
from __future__ import annotations

import argparse
import glob
import gzip
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Chống phân mảnh VRAM — phải đặt TRƯỚC khi torch khởi tạo CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_REPO = "https://github.com/hotuyen21pt/Prism.git"


def clone_repo(url: str, directory: Path) -> Path:
    for project_root in (directory, directory / "prism"):
        if (project_root / "src" / "prism").is_dir():
            return project_root
    directory.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", url, str(directory)], check=True)
    return directory / "prism" if (directory / "prism").is_dir() else directory


def patch_infer_bug(project_root: Path) -> None:
    """Sửa KeyError 'sentiment_model' (dòng gán 'provenance_flip')."""
    path = project_root / "src" / "prism" / "module_b_infer.py"
    text = path.read_text(encoding="utf-8")
    fixed = text.replace('s_post != q["sentiment_model"]', 's_post != q["sentiment"]')
    if fixed != text:
        path.write_text(fixed, encoding="utf-8")
        print("patched module_b_infer.py")


def find_one(name: str, roots: list[str]) -> str | None:
    for root in roots:
        hits = [p for p in glob.glob(f"{root}/**/{name}", recursive=True)
                if os.path.isfile(p)]
        if hits:
            return hits[0]
    return None


def prepare_store(src_roots: list[str], store_dst: Path) -> None:
    store_dst.mkdir(parents=True, exist_ok=True)

    cohorts = find_one("hotel_cohorts.json", src_roots)
    if not cohorts:
        raise FileNotFoundError("Không thấy hotel_cohorts.json trong /kaggle/input")
    shutil.copy(cohorts, store_dst / "hotel_cohorts.json")

    dst = store_dst / "reviews.jsonl.gz"
    gz = find_one("reviews.jsonl.gz", src_roots)
    plain = find_one("reviews.jsonl", src_roots)
    if gz:
        shutil.copy(gz, dst)
    elif plain:
        print("nén reviews.jsonl -> reviews.jsonl.gz ...")
        with open(plain, "rb") as fin, gzip.open(dst, "wb") as fout:
            shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)
    else:
        raise FileNotFoundError("Không thấy reviews.jsonl(.gz) trong /kaggle/input")
    print("store ready:", os.listdir(store_dst))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-url", default=DEFAULT_REPO)
    ap.add_argument("--repo-dir", default="/kaggle/working/Prism")
    ap.add_argument("--work-dir", default="/kaggle/working/outputs")
    ap.add_argument("--ckpt", help="mặc định: tự tìm model.safetensors trong /kaggle/input")
    ap.add_argument("--cohort", default="T-unbiased",
                    choices=["A-dense", "B-anchor", "T-unbiased", "corpus"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--score-batch", type=int, default=8)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--no-rescore", action="store_true")
    args = ap.parse_args()

    repo_root = clone_repo(args.repo_url, Path(args.repo_dir))
    patch_infer_bug(repo_root)

    os.environ["PRISM_TABSA_ROOT"] = str(repo_root)
    os.environ["PRISM_WORK_DIR"] = args.work_dir
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(repo_root / "src"))

    roots = [r for r in ("/kaggle/input", "/kaggle/working") if os.path.isdir(r)]

    ckpt = args.ckpt or find_one("model.safetensors", roots)
    if not ckpt:
        raise FileNotFoundError("Không thấy model.safetensors — attach dataset checkpoint chưa?")
    ckpt = os.path.dirname(ckpt)
    print("CKPT =", ckpt)

    prepare_store(roots, Path(args.work_dir) / "store")
    (Path(args.work_dir) / "extract").mkdir(parents=True, exist_ok=True)

    import torch
    print("cuda:", torch.cuda.is_available())

    from prism import module_b_infer

    argv = ["prism.module_b_infer", "--ckpt", ckpt, "--cohort", args.cohort,
            "--batch", str(args.batch), "--score-batch", str(args.score_batch)]
    if args.lam is not None:
        argv += ["--lam", str(args.lam)]
    if args.no_rescore:
        argv += ["--no-rescore"]
    sys.argv = argv
    print("out:", f"{args.work_dir}/extract/pool_quads.{args.cohort}.jsonl.gz")
    module_b_infer.main()


if __name__ == "__main__":
    main()
