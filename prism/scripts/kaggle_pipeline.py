"""PRISM — orchestrator chạy TỪNG STEP trên Kaggle (mỗi cell = 1 step).

Triết lý: chạy 1 step / 1 cell để lỗi step nào chỉ hỏng step đó, không kéo sập cả
pipeline. Mọi output ghi vào /kaggle/working (WORK_DIR/MODEL_DIR) -> trở thành output
notebook, attach làm input cho step sau.

Thứ tự đầy đủ (bất biến):
    store -> data -> train -> selftrain -> infer -> photos
          -> c_verifier -> c_apply_verifier -> c_bridge -> c_apply -> drift [-> injection]

Chen giữa infer và c_bridge là MỘT bước người (không có trong 'all'):
    --step audit_sample  -> outputs/reliability/audit_sample_300.jsonl
    người điền "correct": 0|1 cho 300 quad -> attach lại -> chạy lại c_bridge.
Không có file đó, c_bridge luôn NO-GO và c_apply chạy fallback w=conf_seq
(bài lùi về A+B+D) — vẫn ra kết quả, nhưng phải báo cáo đúng là không có Module C.

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
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Log/print của script này toàn tiếng Việt. Console Windows mặc định cp1252 nên
# một dòng có dấu là UnicodeEncodeError -> giết cả run. Ép UTF-8, mất dấu chứ
# không mất run. (Kaggle vốn UTF-8, nhưng script cũng chạy được ở local.)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

DEFAULT_REPO = "https://github.com/hotuyen21pt/Prism.git"
SEARCH_ROOTS = ["/kaggle/input", "/kaggle/working"]

# Tên file phải KHỚP với src/prism/config.py — orchestrator tìm input THEO TÊN,
# nên lệch tên là FileNotFoundError dù file có đó. Giữ đồng bộ với
# config.pool_quads_name / vimg_quads_name / HUMAN_AUDIT_NAME.
def POOL_NAME(cohort: str) -> str:
    return f"pool_quads.{cohort}.jsonl.gz"


def VIMG_NAME(cohort: str) -> str:
    return f"pool_quads_vimg.{cohort}.jsonl.gz"


AUDIT_NAME = "audit_sample_300.jsonl"

# thứ tự chạy khi --step all (injection là tùy chọn, không nằm trong 'all')
PIPELINE = ["store", "data", "train", "selftrain", "infer", "photos",
            "c_verifier", "c_apply_verifier", "c_bridge", "c_apply", "drift"]
# audit_sample KHÔNG nằm trong 'all': nó chỉ sinh template, phần còn lại là
# công người (annotate 'correct'), không tự động hoá được.
STEPS = PIPELINE + ["audit_sample", "injection", "all"]


# --------------------------------------------------------------------- lấy repo
def _existing_repo(*cands: Path) -> Path | None:
    for cand in cands:
        for root in (cand, cand / "prism"):
            if (root / "src" / "prism").is_dir():
                return root
    return None


def _print_commit(repo: Path) -> None:
    try:
        out = subprocess.run(["git", "-C", str(repo.parent), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode == 0:
            print("  repo commit:", out.stdout.strip())
    except Exception:                                # noqa: BLE001
        pass


def clone_repo(url: str, directory: Path) -> Path:
    """Lấy source prism. CLONE TRƯỚC, chỉ fallback sang repo có sẵn khi clone lỗi.

    Bản cũ ưu tiên bất kỳ repo tìm thấy dưới /kaggle/input, nên một output notebook
    cũ có thư mục Prism sẽ khiến cả run dùng CODE CŨ — trong khi cell setup ở README
    vừa clone bản mới về /kaggle/working. Không có log nào nói ra.
    """
    found = _existing_repo(directory)
    if found:
        print("dùng repo đã có tại", found)
        _print_commit(found)
        return found

    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git", "clone", "--depth", "1", url, str(directory)], check=True)
        repo = directory / "prism" if (directory / "prism").is_dir() else directory
        print("đã clone repo mới ->", repo)
        _print_commit(repo)
        return repo
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[!] clone lỗi ({e}) — thử tìm repo dưới {SEARCH_ROOTS}")

    for root in SEARCH_ROOTS:
        for hit in sorted(glob.glob(f"{root}/**/src/prism/config.py", recursive=True)):
            repo = Path(hit).parents[2]
            print(f"[!] DÙNG REPO CÓ SẴN (có thể là code CŨ): {repo}")
            _print_commit(repo)
            return repo
    raise RuntimeError(
        "Không lấy được source prism: clone lỗi và không thấy repo dưới "
        "/kaggle/input. Bật internet hoặc attach output notebook có thư mục Prism.")


# --------------------------------------------------------------------- tìm file
def find_one(name: str, roots: list[str]) -> str | None:
    """Tìm file theo tên. Ưu tiên /kaggle/working (output step trước trong cùng
    session) rồi tới file MỚI NHẤT.

    glob(recursive=True) không bảo đảm thứ tự, nên `hits[0]` của bản cũ chọn tuỳ ý.
    Trên Kaggle rất thường attach cùng lúc dataset store + output notebook trước,
    và CẢ HAI đều chứa train.t2t.jsonl / dev.jsonl / hotel_cohorts.json — chọn sai
    là train trên split cũ mà log vẫn in "staged ..." như bình thường.
    """
    hits = []
    for prio, root in enumerate(roots):
        for p in glob.glob(f"{root}/**/{name}", recursive=True):
            if os.path.isfile(p):
                # /kaggle/working đứng trước /kaggle/input; trong cùng root: mới nhất
                hits.append((0 if root.rstrip("/").endswith("working") else 1,
                             -os.path.getmtime(p), prio, p))
    if not hits:
        return None
    hits.sort()
    if len(hits) > 1:
        print(f"  [!] {len(hits)} bản của {name}, dùng bản đầu:")
        for _, negmt, _, p in hits[:5]:
            print(f"      {p}  (mtime {-negmt:.0f})")
    return hits[0][3]


def find_ckpt(roots: list[str]) -> str | None:
    """Thư mục checkpoint HF hợp lệ = có model.safetensors + config.json.

    Thứ tự ưu tiên: (1) final_ckpt ghi trong selftrain_history.json — đây là
    checkpoint self-train ĐÃ VƯỢT dev F1 tốt nhất; (2) selftrain_round<N> với N
    LỚN NHẤT; (3) output train / seed_extractor.

    Bản cũ tie-break bằng len(path) rồi sorted() theo chữ, nên round1 thắng round2
    khi hai đường dẫn dài bằng nhau -> infer chính thức chạy checkpoint vòng trước,
    ngược đúng bất biến ghi trong README.
    """
    dirs = []
    for root in roots:
        for p in glob.glob(f"{root}/**/model.safetensors", recursive=True):
            d = os.path.dirname(p)
            if os.path.isfile(os.path.join(d, "config.json")):
                dirs.append(d)
    if not dirs:
        return None
    dirs = sorted(set(dirs))

    def _prefer(cands: list[str]) -> str:
        """Nhiều thư mục cùng tên: ưu tiên /kaggle/working (output step trước trong
        cùng session) rồi tới bản MỚI NHẤT. Khớp theo basename một mình là chưa đủ —
        trên Kaggle rất thường có dataset CŨ trong /input và output MỚI trong
        /working mang ĐÚNG cùng tên, và sorted() theo chữ thì /input thắng."""
        if len(cands) > 1:
            print(f"  [!] {len(cands)} thư mục cùng tên, ưu tiên /working + mới nhất:")
            for c in cands:
                print(f"      {c}")
        return sorted(cands, key=lambda d: ("working" not in d.replace("\\", "/"),
                                            -os.path.getmtime(d)))[0]

    # (1) tin selftrain_history.json trước tiên
    hist = find_one("selftrain_history.json", roots)
    if hist:
        try:
            with open(hist, encoding="utf-8") as f:
                h = json.load(f)
            want_path = str(h.get("final_ckpt") or "")
            want = os.path.basename(want_path)
            # khớp ĐƯỜNG DẪN ĐẦY ĐỦ trước — chính xác nhất khi chạy cùng session
            exact = [d for d in dirs
                     if os.path.normpath(d) == os.path.normpath(want_path)]
            same_name = [d for d in dirs if want and os.path.basename(d) == want]
            hit = exact or same_name
            if hit:
                d = _prefer(hit)
                print(f"  ckpt theo selftrain_history.json: {d} "
                      f"(dev F1 = {h.get('final_dev_f1')})")
                return d
            if want:
                print(f"  [!] selftrain_history.json trỏ tới '{want}' nhưng không "
                      f"thấy thư mục đó trong input — rơi về xếp hạng theo tên")
        except Exception as e:                       # noqa: BLE001
            print(f"  [!] không đọc được {hist}: {e}")

    def rank(d: str):
        low = os.path.basename(d).lower()
        m = re.search(r"round(\d+)", low)
        return (
            "selftrain" not in d.lower(),            # selftrain trước
            -(int(m.group(1)) if m else -1),         # round LỚN NHẤT trước
            "module-b-train" not in d.lower(),
            "seed_extractor" not in d.lower(),
            "working" not in d.replace("\\", "/"),   # /working trước /input
            -os.path.getmtime(d),                    # mới nhất trước
        )

    best = sorted(dirs, key=rank)[0]
    print(f"  ckpt theo xếp hạng tên/mtime: {best}")
    return best


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


def _safe_copy(src: str, dst: Path) -> bool:
    """Copy src->dst, nhưng BỎ QUA nếu nguồn trùng đích (đã stage sẵn từ step
    trước dưới /kaggle/working). Tránh shutil.SameFileError."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if os.path.abspath(src) == os.path.abspath(dst):
        return False
    shutil.copy(src, dst)
    return True


def copy_in(name: str, dst: Path, roots: list[str], hint: str = "") -> None:
    if _safe_copy(require(name, roots, hint), dst):
        print(f"  staged {name} -> {dst}")
    else:
        print(f"  ok {name} (đã có sẵn tại {dst})")


def copy_opt(name: str, dst: Path, roots: list[str]) -> bool:
    hit = find_one(name, roots)
    if not hit:
        return False
    if _safe_copy(hit, dst):
        print(f"  staged {name} -> {dst}")
    else:
        print(f"  ok {name} (đã có sẵn tại {dst})")
    return True


# KHÔNG có hàm patch source ở đây nữa.
# Trước đây orchestrator ghi đè src/prism/*.py lúc chạy bằng so khớp chuỗi:
#   - patch_infer_bug: needle đã không còn tồn tại -> no-op hoàn toàn (code chết
#     nhưng vẫn có quyền ghi vào src/)
#   - patch_selftrain_batch: needle là 2 dòng KÈM đúng 24 space thụt lề; reformat
#     hay đổi tên biến là không khớp -> không patch, KHÔNG cảnh báo -> self-train
#     gọi infer với default batch 32/48 -> OOM trên T4 sau khi train xong vòng 1.
# module_b_selftrain nay có --batch/--score-batch thật, truyền thẳng ở step selftrain.

# ------------------------------------------------------------------------ store
def prepare_store(store_dst: Path, roots: list[str]) -> None:
    store_dst.mkdir(parents=True, exist_ok=True)
    copy_in("hotel_cohorts.json", store_dst / "hotel_cohorts.json", roots,
            "Cần store Module A. Chạy step 'store' trước hoặc attach dataset store.")
    dst = store_dst / "reviews.jsonl.gz"
    gz = find_one("reviews.jsonl.gz", roots)
    plain = find_one("reviews.jsonl", roots)
    if gz:
        _safe_copy(gz, dst)
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
# Ngưỡng "file coi như rỗng": .jsonl.gz rỗng vẫn nặng ~53 byte (header gzip), nên
# os.path.exists() một mình không đủ. Đã gặp thật: pool_quads.T-unbiased.jsonl.gz
# tồn tại với 0 dòng -> `--step all` in "[skip] infer: đã có output" rồi chạy
# Module C/D trên tập rỗng.
MIN_SENTINEL_BYTES = 200


def sentinel_ok(path: Path) -> bool:
    return path.exists() and path.stat().st_size >= MIN_SENTINEL_BYTES


def sentinel(step: str, work: Path, model_dir: Path, cohort: str, level: str) -> Path:
    pool = POOL_NAME(cohort)
    return {
        "store":           work / "store" / "reviews.jsonl.gz",
        "data":            work / "extract" / "train.t2t.jsonl",
        "train":           model_dir / "seed_extractor" / "model.safetensors",
        "selftrain":       work / "extract" / "selftrain_history.json",
        "infer":           work / "extract" / pool,
        "photos":          work / "reliability" / "pool_image_index.json",
        "audit_sample":    work / "reliability" / AUDIT_NAME,
        "c_verifier":      work / "reliability" / "verifier.pkl",
        "c_apply_verifier": work / "reliability" / VIMG_NAME(cohort),
        "c_bridge":        work / "reliability" / "bridge.pkl",
        "c_apply":         work / "reliability" / "quads_weighted.jsonl.gz",
        "drift":           work / "drift" / f"drift_results.{cohort}.{level}.json",
    }[step]


# ------------------------------------------------------------------ 1 step logic
def do_step(step: str, args, repo: Path, env: dict,
            work: Path, model_dir: Path, roots: list[str], hamos: str | None) -> None:
    raw = repo / "data" / "raw"
    pool = POOL_NAME(args.cohort)
    vimg = VIMG_NAME(args.cohort)
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
                   ["--model", args.model, "--epochs", str(args.epochs),
                    "--batch", str(args.batch), "--grad-accum", str(args.grad_accum),
                    "--out", str(model_dir / "seed_extractor")], env)

    elif step == "selftrain":
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
                    "--batch", str(args.infer_batch),
                    "--score-batch", str(args.infer_score_batch),
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
                   ["--stage", "apply_verifier", "--cohort", args.cohort,
                    "--quads", str(work / "extract" / pool),
                    "--image-index", str(reliab / "pool_image_index.json"),
                    "--out", str(reliab / vimg)], env)

    elif step == "audit_sample":
        # §3.6/bước 7b: sinh TEMPLATE 300 quad để người chấm đúng/sai. Không có
        # file này thì c_bridge NO-GO vĩnh viễn (không chấm được Spearman).
        # --only audit: D0 (§3.2) là mẫu KHÁC, đọc reviews.jsonl.gz và ghi đè
        # template đã giao annotate — không được kéo theo.
        copy_in(pool, work / "extract" / pool, roots, "Chạy step 'infer' trước.")
        run_module("prism.make_audit_samples",
                   ["--quads", str(work / "extract" / pool), "--only", "audit"], env)
        print(f"  -> {reliab / AUDIT_NAME}: GIAO ANNOTATE (điền 'correct': 0|1),"
              " attach lại rồi chạy lại step 'c_bridge'.")

    elif step == "c_bridge":
        # bridge CẦN file có v_image = output của apply_verifier (pool_quads_vimg)
        copy_in(vimg, reliab / vimg, roots, "Chạy step 'c_apply_verifier' trước.")
        # MỘT tên duy nhất cho file audit (config.HUMAN_AUDIT_NAME) — không rename
        copy_opt(AUDIT_NAME, reliab / AUDIT_NAME, roots)
        run_module("prism.module_c_reliability",
                   ["--stage", "bridge", "--cohort", args.cohort,
                    "--quads", str(reliab / vimg),
                    "--audit", str(reliab / AUDIT_NAME)], env)

    elif step == "c_apply":
        copy_in(pool, work / "extract" / pool, roots, "Chạy step 'infer' trước.")
        # bridge.pkl là TUỲ CHỌN. c_bridge chỉ ghi ra cái tên này khi verdict=GO;
        # NO-GO thì nó ghi bridge_NOGO.pkl. Hợp đồng đã ghi trong apply_weights:
        # thiếu bridge -> w = conf_seq thô + cờ w_source, bài lùi về A+B+D.
        # Hard-require ở đây làm nhánh đó không bao giờ chạy được và giết luôn
        # 'all' ngay trước drift.
        if not copy_opt("bridge.pkl", reliab / "bridge.pkl", roots):
            print("  ! không có bridge.pkl (Module C NO-GO hoặc chưa chạy c_bridge)"
                  " -> apply sẽ dùng w=conf_seq thô, w_source=conf_seq."
                  " KHÔNG được báo cáo đây là kết quả có Module C.")
        run_module("prism.module_c_reliability",
                   ["--stage", "apply", "--cohort", args.cohort,
                    "--quads", str(work / "extract" / pool),
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
    ap.add_argument("--model", default="google/mt5-small", help="base model cho step train")
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
        if sentinel_ok(out) and not args.force:
            print(f"[skip] {step}: đã có output {out} ({out.stat().st_size} byte)")
            continue
        if out.exists():
            print(f"[rerun] {step}: {out} tồn tại nhưng RỖNG "
                  f"({out.stat().st_size} byte) -> chạy lại")
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
