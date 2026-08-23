"""Kaggle entry point for PRISM Module B training.

Expected input files:
  outputs/extract/train.t2t.jsonl
  outputs/extract/dev.t2t.jsonl

The files may be placed anywhere under /kaggle/input or /kaggle/working.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_REPO = "https://github.com/hotuyen21pt/Prism.git"


def clone_repo(url: str, directory: Path) -> Path:
    for project_root in (directory, directory / "prism"):
        if (project_root / "src" / "prism").is_dir():
            return project_root
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"Repository directory is not empty: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(directory)],
        check=True,
    )
    return directory


def find_file(name: str, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    roots = [Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()]
    matches = [path for root in roots if root.exists()
               for path in root.rglob(name) if path.is_file()]
    if not matches:
        raise FileNotFoundError(
            f"Could not find {name}. Upload the PRISM extract dataset or pass "
            f"--{name.removesuffix('.t2t.jsonl').replace('_', '-')}-file."
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-url", default=DEFAULT_REPO,
                        help="GitHub repository to clone before training")
    parser.add_argument("--repo-dir", default="/kaggle/working/Prism",
                        help="Local clone directory")
    parser.add_argument("--train-file", help="Path to train.t2t.jsonl")
    parser.add_argument("--dev-file", help="Path to dev.t2t.jsonl")
    parser.add_argument("--out", default="/kaggle/working/models/seed_extractor")
    parser.add_argument("--model", default="google/mt5-small")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-src", type=int, default=160)
    parser.add_argument("--max-tgt", type=int, default=192)
    args = parser.parse_args()

    repo_root = clone_repo(args.repo_url, Path(args.repo_dir))
    os.environ.setdefault("PRISM_TABSA_ROOT", str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))

    train_file = find_file("train.t2t.jsonl", args.train_file)
    dev_file = find_file("dev.t2t.jsonl", args.dev_file)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from prism import module_b_train

    sys.argv = [
        "prism.module_b_train",
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--grad-accum", str(args.grad_accum),
        "--lr", str(args.lr),
        "--max-src", str(args.max_src),
        "--max-tgt", str(args.max_tgt),
        "--train-file", str(train_file),
        "--dev-file", str(dev_file),
        "--out", args.out,
    ]
    print(f"train: {train_file}")
    print(f"dev:   {dev_file}")
    print(f"model: {args.model}")
    print(f"out:   {args.out}")
    module_b_train.main()


if __name__ == "__main__":
    main()