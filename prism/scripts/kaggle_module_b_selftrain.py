"""Kaggle entry point cho PRISM Module B — self-training teacher/student.

Bọc prism.module_b_selftrain cho môi trường Kaggle. Vòng lặp self-train tự gọi
lại module_b_infer / module_b_train / module_b_eval qua subprocess
(`python -m prism.…`), nên script này phải:

  - clone repo (nếu chưa có) và đặt PYTHONPATH để `-m prism.…` chạy được ở tiến trình con
  - vá bug KeyError 'sentiment_model' của module_b_infer (idempotent)
  - vá bước infer của self-train để TRUYỀN --batch/--score-batch nhỏ (mặc định
    self-train không truyền -> dùng batch 32/48 -> OOM mT5 trên T4). Idempotent.
  - đặt env PRISM_TABSA_ROOT / PRISM_WORK_DIR / PRISM_MODEL_DIR để config trỏ đúng
    thư mục dữ liệu đã stage, và tiến trình con kế thừa được
  - stage input: train.t2t.jsonl, dev.t2t.jsonl -> WORK_DIR/extract ;
    dev.jsonl -> data/raw ; teacher checkpoint -> MODEL_DIR/seed_extractor
  - nạp store của Module A (reviews.jsonl(.gz) + hotel_cohorts.json) cho bước infer
  - expandable_segments để giảm phân mảnh VRAM

Input cần attach vào Kaggle (đặt ở đâu dưới /kaggle/input cũng được, script tự tìm):
  - checkpoint teacher: model.safetensors (+ config/tokenizer cùng thư mục)
  - train.t2t.jsonl, dev.t2t.jsonl        (Module B, dạng text2text)
  - dev.jsonl                              (gold để eval dừng sớm)
  - reviews.jsonl(.gz), hotel_cohorts.json (store Module A, cho bước infer pool)

Chạy trong 1 cell Kaggle (bật GPU):
    !python /kaggle/working/Prism/prism/scripts/kaggle_module_b_selftrain.py \
        --rounds 2 --cohort B-anchor --infer-batch 4 --infer-score-batch 8
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


def patch_selftrain_infer_batch(project_root: Path, batch: int, score_batch: int) -> None:
    """Chèn --batch/--score-batch vào lệnh infer của self-train (mặc định không có
    -> dùng batch 32/48 -> OOM). Idempotent: bỏ qua nếu đã có --score-batch."""
    path = project_root / "src" / "prism" / "module_b_selftrain.py"
    text = path.read_text(encoding="utf-8")
    if "--score-batch" in text:
        return
    needle = ('                        "--ckpt", ckpt, "--cohort", args.cohort,\n'
              '                        "--limit", str(args.infer_limit)], check=True)')
    repl = ('                        "--ckpt", ckpt, "--cohort", args.cohort,\n'
            f'                        "--batch", "{batch}", "--score-batch", "{score_batch}",\n'
            '                        "--limit", str(args.infer_limit)], check=True)')
    if needle not in text:
        raise RuntimeError("Không khớp được lệnh infer trong module_b_selftrain.py — "
                           "file có thể đã đổi; kiểm tra lại patch.")
    path.write_text(text.replace(needle, repl), encoding="utf-8")
    print(f"patched module_b_selftrain.py infer batch -> {batch}/{score_batch}")


def find_one(name: str, roots: list[str]) -> str | None:
    for root in roots:
        hits = [p for p in glob.glob(f"{root}/**/{name}", recursive=True)
                if os.path.isfile(p)]
        if hits:
            return hits[0]
    return None


def require(name: str, roots: list[str]) -> str:
    hit = find_one(name, roots)
    if not hit:
        raise FileNotFoundError(f"Không thấy {name} trong {roots} — attach dataset chưa?")
    return hit


def prepare_store(src_roots: list[str], store_dst: Path) -> None:
    store_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(require("hotel_cohorts.json", src_roots), store_dst / "hotel_cohorts.json")

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


def stage(name: str, roots: list[str], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(require(name, roots), dst)
    print(f"staged {name} -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-url", default=DEFAULT_REPO)
    ap.add_argument("--repo-dir", default="/kaggle/working/Prism")
    ap.add_argument("--work-dir", default="/kaggle/working/outputs")
    ap.add_argument("--model-dir", default="/kaggle/working/models")
    ap.add_argument("--seed-ckpt", help="mặc định: tự tìm model.safetensors trong /kaggle/input")
    # tham số self-train (khớp module_b_selftrain)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--cohort", default="B-anchor",
                    choices=["A-dense", "B-anchor", "T-unbiased", "corpus"])
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--tau-post", type=float, default=0.8)
    ap.add_argument("--max-ratio", type=float, default=3.0)
    ap.add_argument("--max-skew", type=float, default=0.10)
    ap.add_argument("--infer-limit", type=int, default=200_000)
    # batch nhỏ cho bước infer (tránh OOM)
    ap.add_argument("--infer-batch", type=int, default=4)
    ap.add_argument("--infer-score-batch", type=int, default=8)
    args = ap.parse_args()

    repo_root = clone_repo(args.repo_url, Path(args.repo_dir))
    patch_infer_bug(repo_root)
    patch_selftrain_infer_batch(repo_root, args.infer_batch, args.infer_score_batch)

    model_dir = Path(args.model_dir)
    os.environ["PRISM_TABSA_ROOT"] = str(repo_root)
    os.environ["PRISM_WORK_DIR"] = args.work_dir
    os.environ["PRISM_MODEL_DIR"] = str(model_dir)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # để tiến trình con (`python -m prism.…`) import được prism từ src
    src = str(repo_root / "src")
    os.environ["PYTHONPATH"] = src + os.pathsep + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, src)

    roots = [r for r in ("/kaggle/input", "/kaggle/working") if os.path.isdir(r)]
    work = Path(args.work_dir)

    # teacher checkpoint -> MODEL_DIR/seed_extractor (self-train khởi đầu từ đây)
    # --seed-ckpt: file model.safetensors, HOẶC thư mục chứa nó (vd output notebook train)
    if args.seed_ckpt:
        p = Path(args.seed_ckpt)
        seed = str(p) if p.is_file() else require("model.safetensors", [str(p)])
    else:
        seed = require("model.safetensors", roots)
    seed_dir = Path(seed).parent if os.path.isfile(seed) else Path(seed)
    dst_ckpt = model_dir / "seed_extractor"
    if dst_ckpt.resolve() != seed_dir.resolve():
        if dst_ckpt.exists():
            shutil.rmtree(dst_ckpt)
        shutil.copytree(seed_dir, dst_ckpt)
    print("seed_extractor =", dst_ckpt)

    # stage các file text2text + gold dev + store
    stage("train.t2t.jsonl", roots, work / "extract" / "train.t2t.jsonl")
    stage("dev.t2t.jsonl", roots, work / "extract" / "dev.t2t.jsonl")
    stage("dev.jsonl", roots, repo_root / "data" / "raw" / "dev.jsonl")
    prepare_store(roots, work / "store")
    (work / "extract").mkdir(parents=True, exist_ok=True)

    import torch
    print("cuda:", torch.cuda.is_available())

    from prism import module_b_selftrain

    sys.argv = [
        "prism.module_b_selftrain",
        "--rounds", str(args.rounds),
        "--cohort", args.cohort,
        "--tau", str(args.tau),
        "--tau-post", str(args.tau_post),
        "--max-ratio", str(args.max_ratio),
        "--max-skew", str(args.max_skew),
        "--infer-limit", str(args.infer_limit),
    ]
    print("history ->", f"{args.work_dir}/extract/selftrain_history.json")
    module_b_selftrain.main()


if __name__ == "__main__":
    main()
