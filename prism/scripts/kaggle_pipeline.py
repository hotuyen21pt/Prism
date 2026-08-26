"""PRISM — orchestrator chạy TỪNG STEP trên Kaggle (mỗi cell = 1 step).

Triết lý: chạy 1 step / 1 cell để lỗi step nào chỉ hỏng step đó, không kéo sập cả
pipeline. Mọi output ghi vào /kaggle/working (WORK_DIR/MODEL_DIR) -> trở thành output
notebook, attach làm input cho step sau.

Thứ tự đầy đủ (bất biến):
    store -> data -> train -> selftrain -> infer -> photos
          -> c_verifier -> c_apply_verifier -> c_bridge -> c_apply -> drift [-> injection]

Chạy 1 step:
    !python /kaggle/working/Prism/prism/scripts/kaggle_pipeline.py --step infer \
        --ckpt /kaggle/input/.../selftrain_round2 --cohort T-unbiased

Chạy hết trong 1 cell (BỎ QUA step đã có output sẵn dưới /kaggle/working):
    !python /kaggle/working/Prism/prism/scripts/kaggle_pipeline.py --step all --cohort T-unbiased

Input: tự tìm theo TÊN FILE đệ quy khắp /kaggle/input + /kaggle/working (kể cả output
notebook step trước đã attach), rồi stage vào đúng WORK_DIR/MODEL_DIR mà config trông đợi.
Thiếu file nào -> báo lỗi rõ tên file & step cần chạy trước.

Phụ thuộc dữ liệu ngoài repo (attach khi cần):
  - store : hotel_booking_unlabeled.jsonl + {train,dev,test}.jsonl + gold ABSA
  - data / c_verifier : hamos-mabsa/ (gold quads + ảnh) — set PRISM_HAMOS_ROOT tự động
  - photos + c_* : nên chạy CÙNG SESSION vì pool_image_index.json chứa đường dẫn tuyệt đối.
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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_REPO = "https://github.com/hotuyen21pt/Prism.git"
SEARCH_ROOTS = ["/kaggle/input", "/kaggle/working"]

# thứ tự chạy khi --step all (injection là tùy chọn, không nằm trong 'all')
PIPELINE = ["store", "data", "train", "selftrain", "infer", "photos",
            "c_verifier", "c_apply_verifier", "c_bridge", "c_apply", "drift"]
STEPS = PIPELINE + ["injection", "all"]


# --------------------------------------------------------------------- lấy repo
def _existing_repo(*cands: Path) -> Path | None:
    for cand in cands:
        for root in (cand, cand / "prism"):
            if (root / "src" / "prism").is_dir():
                return root
    return None


def clone_repo(url: str, directory: Path) -> Path:
    found = _existing_repo(directory)
    if found:
        return found
    for root in SEARCH_ROOTS:
        for hit in glob.glob(f"{root}/**/src/prism/config.py", recursive=True):
            repo = Path(hit).parents[2]
            print("dùng repo có sẵn:", repo)
            return repo
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git", "clone", "--depth", "1", url, str(directory)], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(
            "Không lấy được source prism: clone lỗi và không thấy repo dưới "
            "/kaggle/input. Bật internet hoặc attach output notebook có thư mục Prism."
        ) from e
    return directory / "prism" if (directory / "prism").is_dir() else directory


# --------------------------------------------------------------------- tìm file
def find_one(name: str, roots: list[str]) -> str | None:
    for root in roots:
        hits = [p for p in glob.glob(f"{root}/**/{name}", recursive=True)
                if os.path.isfile(p)]
        if hits:
            return hits[0]
    return None


def find_ckpt(roots: list[str]) -> str | None:
    """Thư mục checkpoint HF hợp lệ = có model.safetensors + config.json.
    Ưu tiên student self-train, rồi output train."""
    dirs = []
    for root in roots:
        for p in glob.glob(f"{root}/**/model.safetensors", recursive=True):
            d = os.path.dirname(p)
            if os.path.isfile(os.path.join(d, "config.json")):
                dirs.append(d)
    if not dirs:
        return None

    def rank(d: str):
        low = d.lower()
        return ("selftrain" not in low, "module-b-train" not in low,
                "seed_extractor" not in low, len(d))

    return sorted(set(dirs), key=rank)[0]


def find_hamos(roots: list[str]) -> str | None:
    """Gốc hamos-mabsa sao cho <root>/annotations/quads.jsonl hoặc
    <root>/data/annotations/quads.jsonl tồn tại (khớp config._hamos_file)."""
    for root in roots:
        for hit in glob.glob(f"{root}/**/annotations/quads.jsonl", recursive=True):
            base = Path(hit).parents[1]                 # .../<X>/annotations/quads.jsonl -> <X>
            return str(base.parent if base.name == "data" else base)
    for root in roots:
        for hit in glob.glob(f"{root}/**/hamos-mabsa", recursive=True):
            if os.path.isdir(hit):
                return hit
    return None


def require(name: str, roots: list[str], hint: str = "") -> str:
    hit = find_one(name, roots)
    if not hit:
        raise FileNotFoundError(f"Không thấy {name} trong /kaggle/input. {hint}")
    return hit


def copy_in(name: str, dst: Path, roots: list[str], hint: str = "") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(require(name, roots, hint), dst)
    print(f"  staged {name} -> {dst}")


def copy_opt(name: str, dst: Path, roots: list[str]) -> bool:
    hit = find_one(name, roots)
    if not hit:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(hit, dst)
    print(f"  staged {name} -> {dst}")
    return True


# ------------------------------------------------------------------------ patch
def patch_infer_bug(repo: Path) -> None:
    p = repo / "src" / "prism" / "module_b_infer.py"
    t = p.read_text(encoding="utf-8")
    f = t.replace('s_post != q["sentiment_model"]', 's_post != q["sentiment"]')
    if f != t:
        p.write_text(f, encoding="utf-8")
        print("patched module_b_infer.py")


def patch_selftrain_batch(repo: Path, batch: int, sbatch: int) -> None:
    p = repo / "src" / "prism" / "module_b_selftrain.py"
    t = p.read_text(encoding="utf-8")
    if "--score-batch" in t:
        return
    needle = ('                        "--ckpt", ckpt, "--cohort", args.cohort,\n'
              '                        "--limit", str(args.infer_limit)], check=True)')
    repl = ('                        "--ckpt", ckpt, "--cohort", args.cohort,\n'
            f'                        "--batch", "{batch}", "--score-batch", "{sbatch}",\n'
            '                        "--limit", str(args.infer_limit)], check=True)')
    if needle in t:
        p.write_text(t.replace(needle, repl), encoding="utf-8")
        print(f"patched selftrain infer batch -> {batch}/{sbatch}")


# ------------------------------------------------------------------------ store
def prepare_store(store_dst: Path, roots: list[str]) -> None:
    store_dst.mkdir(parents=True, exist_ok=True)
    copy_in("hotel_cohorts.json", store_dst / "hotel_cohorts.json", roots,
            "Cần store Module A. Chạy step 'store' trước hoặc attach dataset store.")
    dst = store_dst / "reviews.jsonl.gz"
    gz = find_one("reviews.jsonl.gz", roots)
    plain = find_one("reviews.jsonl", roots)
    if gz:
        shutil.copy(gz, dst)
    elif plain:
        print("  nén reviews.jsonl -> .gz")
        with open(plain, "rb") as fi, gzip.open(dst, "wb") as fo:
            shutil.copyfileobj(fi, fo, length=16 * 1024 * 1024)
    else:
        raise FileNotFoundError("Không thấy reviews.jsonl(.gz) — chạy step 'store' trước.")
    print("  store ready:", os.listdir(store_dst))


# -------------------------------------------------------------------- chạy module
def run_module(module: str, argv: list[str], env: dict) -> None:
    cmd = [sys.executable, "-m", module] + argv
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def run_script(path: Path, argv: list[str], env: dict) -> None:
    cmd = [sys.executable, str(path)] + argv
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


# ---------------------------------------------------------------- sentinel (all)
def sentinel(step: str, work: Path, model_dir: Path, cohort: str, level: str) -> Path:
    pool = f"pool_quads.{cohort}.jsonl.gz"
    return {
        "store":           work / "store" / "reviews.jsonl.gz",
        "data":            work / "extract" / "train.t2t.jsonl",
        "train":           model_dir / "seed_extractor" / "model.safetensors",
        "selftrain":       work / "extract" / "selftrain_history.json",
        "infer":           work / "extract" / pool,
        "photos":          work / "reliability" / "pool_image_index.json",
        "c_verifier":      work / "reliability" / "verifier.pkl",
        "c_apply_verifier": work / "reliability" / f"pool_quads_vimg.{cohort}.jsonl.gz",
        "c_bridge":        work / "reliability" / "bridge.pkl",
        "c_apply":         work / "reliability" / "quads_weighted.jsonl.gz",
        "drift":           work / "drift" / f"drift_results.{cohort}.{level}.json",
    }[step]


# ------------------------------------------------------------------ 1 step logic
def do_step(step: str, args, repo: Path, env: dict,
            work: Path, model_dir: Path, roots: list[str], hamos: str | None) -> None:
    raw = repo / "data" / "raw"
    pool = f"pool_quads.{args.cohort}.jsonl.gz"
    vimg = f"pool_quads_vimg.{args.cohort}.jsonl.gz"
    reliab = work / "reliability"
    print(f"\n=== STEP: {step} | cohort={args.cohort} ===")

    if step == "store":
        copy_in("hotel_booking_unlabeled.jsonl", raw / "hotel_booking_unlabeled.jsonl",
                roots, "Attach pool thô.")
        for s in ("train", "dev", "test"):
            copy_in(f"{s}.jsonl", raw / f"{s}.jsonl", roots, "Attach gold split.")
        copy_opt("hotel_absa_labeled.jsonl", raw / "hotel_absa_labeled.jsonl", roots)
        run_module("prism.module_a_store", [], env)

    elif step == "data":
        if not hamos:
            raise FileNotFoundError("Module B-data cần hamos-mabsa (gold quads). Attach chưa?")
        for s in ("train", "dev", "test"):
            copy_in(f"{s}.jsonl", raw / f"{s}.jsonl", roots, "Attach gold split.")
        copy_opt("hotel_absa_labeled.jsonl", raw / "hotel_absa_labeled.jsonl", roots)
        run_module("prism.module_b_data", [], env)

    elif step == "train":
        copy_in("train.t2t.jsonl", work / "extract" / "train.t2t.jsonl", roots,
                "Chạy step 'data' trước.")
        copy_in("dev.t2t.jsonl", work / "extract" / "dev.t2t.jsonl", roots)
        run_module("prism.module_b_train",
                   ["--epochs", str(args.epochs), "--batch", str(args.batch),
                    "--grad-accum", str(args.grad_accum),
                    "--out", str(model_dir / "seed_extractor")], env)

    elif step == "selftrain":
        patch_selftrain_batch(repo, args.infer_batch, args.infer_score_batch)
        copy_in("train.t2t.jsonl", work / "extract" / "train.t2t.jsonl", roots)
        copy_in("dev.t2t.jsonl", work / "extract" / "dev.t2t.jsonl", roots)
        copy_in("dev.jsonl", raw / "dev.jsonl", roots, "gold dev cho eval.")
        prepare_store(work / "store", roots)
        seed = args.ckpt or find_ckpt(roots)
        if not seed:
            raise FileNotFoundError("Không thấy seed checkpoint — chạy step 'train' trước.")
        seed_dir = Path(seed if os.path.isdir(seed) else os.path.dirname(seed))
        dst = model_dir / "seed_extractor"
        if dst.resolve() != seed_dir.resolve():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(seed_dir, dst)
        print("  seed_extractor =", dst)
        run_module("prism.module_b_selftrain",
                   ["--rounds", str(args.rounds), "--cohort", args.cohort,
                    "--infer-limit", str(args.limit or 200000)], env)

    elif step == "infer":
        prepare_store(work / "store", roots)
        ckpt = args.ckpt or find_ckpt(roots)
        if not ckpt or not os.path.isdir(ckpt):
            raise FileNotFoundError("Không thấy checkpoint (model.safetensors+config.json). "
                                    "Chạy step 'selftrain' (hoặc 'train') trước.")
        print("  CKPT =", ckpt)
        argv = ["--ckpt", ckpt, "--cohort", args.cohort,
                "--batch", str(args.batch), "--score-batch", str(args.score_batch)]
        if args.limit:
            argv += ["--limit", str(args.limit)]
        run_module("prism.module_b_infer", argv, env)

    elif step == "photos":
        prepare_store(work / "store", roots)
        run_script(repo / "scripts" / "download_pool_photos.py",
                   ["--cohort", args.cohort, "--limit", str(args.limit)], env)

    elif step == "c_verifier":
        if not hamos:
            raise FileNotFoundError("train_verifier cần hamos-mabsa (gold ảnh). Attach chưa?")
        for s in ("train", "dev", "test"):
            copy_opt(f"{s}.jsonl", raw / f"{s}.jsonl", roots)
        run_module("prism.module_c_reliability", ["--stage", "train_verifier"], env)

    elif step == "c_apply_verifier":
        copy_in(pool, work / "extract" / pool, roots, "Chạy step 'infer' trước.")
        copy_in("verifier.pkl", reliab / "verifier.pkl", roots, "Chạy step 'c_verifier' trước.")
        copy_in("pool_image_index.json", reliab / "pool_image_index.json", roots,
                "Chạy step 'photos' trước (cùng session để giữ đường dẫn ảnh).")
        run_module("prism.module_c_reliability",
                   ["--stage", "apply_verifier", "--quads", str(work / "extract" / pool),
                    "--image-index", str(reliab / "pool_image_index.json"),
                    "--out", str(reliab / vimg)], env)

    elif step == "c_bridge":
        # bridge CẦN file có v_image = output của apply_verifier (pool_quads_vimg)
        copy_in(vimg, reliab / vimg, roots, "Chạy step 'c_apply_verifier' trước.")
        copy_opt("audit_sample_300.jsonl", reliab / "human_audit_300.jsonl", roots)
        run_module("prism.module_c_reliability",
                   ["--stage", "bridge", "--quads", str(reliab / vimg),
                    "--audit", str(reliab / "human_audit_300.jsonl")], env)

    elif step == "c_apply":
        copy_in(pool, work / "extract" / pool, roots, "Chạy step 'infer' trước.")
        copy_in("bridge.pkl", reliab / "bridge.pkl", roots, "Chạy step 'c_bridge' trước.")
        run_module("prism.module_c_reliability",
                   ["--stage", "apply", "--quads", str(work / "extract" / pool),
                    "--out", str(reliab / "quads_weighted.jsonl.gz")], env)

    elif step == "drift":
        quads = reliab / "quads_weighted.jsonl.gz"
        if not copy_opt("quads_weighted.jsonl.gz", quads, roots):
            copy_in(pool, work / "extract" / pool, roots,
                    "Cần quads_weighted (step 'c_apply') hoặc pool_quads (step 'infer').")
            quads = work / "extract" / pool
        run_module("prism.module_d_drift",
                   ["--quads", str(quads), "--cohort", args.cohort, "--level", args.level], env)

    elif step == "injection":
        quads = reliab / "quads_weighted.jsonl.gz"
        copy_in("quads_weighted.jsonl.gz", quads, roots, "Chạy step 'c_apply' trước.")
        run_module("prism.eval_injection", ["--quads", str(quads), "--test", args.test], env)

    print(f"=== DONE '{step}'. Output dưới {work} và {model_dir} ===")


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True, choices=STEPS)
    ap.add_argument("--repo-url", default=DEFAULT_REPO)
    ap.add_argument("--repo-dir", default="/kaggle/working/Prism")
    ap.add_argument("--work-dir", default="/kaggle/working/outputs")
    ap.add_argument("--model-dir", default="/kaggle/working/models")
    ap.add_argument("--cohort", default="T-unbiased")
    ap.add_argument("--ckpt", help="checkpoint cho infer/selftrain (mặc định: tự tìm)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--score-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--infer-batch", type=int, default=2)
    ap.add_argument("--infer-score-batch", type=int, default=4)
    ap.add_argument("--level", default="taxonomy_code")
    ap.add_argument("--test", default="composition")
    ap.add_argument("--force", action="store_true",
                    help="với --step all: chạy lại cả step đã có output")
    args = ap.parse_args()

    repo = clone_repo(args.repo_url, Path(args.repo_dir))
    patch_infer_bug(repo)

    work = Path(args.work_dir)
    model_dir = Path(args.model_dir)
    roots = [r for r in SEARCH_ROOTS if os.path.isdir(r)]

    env = dict(os.environ)
    env["PRISM_TABSA_ROOT"] = str(repo)
    env["PRISM_WORK_DIR"] = str(work)
    env["PRISM_MODEL_DIR"] = str(model_dir)
    env["PYTHONPATH"] = str(repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    hamos = find_hamos(roots)
    if hamos:
        env["PRISM_HAMOS_ROOT"] = hamos
        print("HAMOS_ROOT =", hamos)
    (work / "extract").mkdir(parents=True, exist_ok=True)

    if args.step != "all":
        do_step(args.step, args, repo, env, work, model_dir, roots, hamos)
        return

    # --- all: chạy tuần tự, BỎ QUA step đã có output ---
    print(f"\n########## RUN ALL (cohort={args.cohort}) ##########")
    for step in PIPELINE:
        out = sentinel(step, work, model_dir, args.cohort, args.level)
        if out.exists() and not args.force:
            print(f"[skip] {step}: đã có output {out}")
            continue
        try:
            do_step(step, args, repo, env, work, model_dir, roots, hamos)
        except Exception as e:                       # noqa: BLE001
            print(f"\n!!! step '{step}' LỖI: {e}")
            print("Dừng chuỗi 'all'. Sửa/attach thiếu rồi chạy lại "
                  "(step đã xong sẽ tự bỏ qua).")
            raise
    print("\n########## ALL DONE ##########")


if __name__ == "__main__":
    main()
