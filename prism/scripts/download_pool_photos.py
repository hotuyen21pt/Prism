"""
Tải ảnh pool cho Module C2 (apply_verifier) — resumable, có rate-limit.

Đọc outputs/store/reviews.jsonl.gz, chọn review has_photo & không in_gold thuộc
cohort chỉ định, tải ẢNH ĐẦU TIÊN của mỗi review về
outputs/reliability/pool_images/<review_uid>.jpg và ghi index
outputs/reliability/pool_image_index.json ({review_uid: đường dẫn tuyệt đối}).

Chạy lại an toàn: ảnh đã có trên đĩa được bỏ qua, index được ghi lại đầy đủ.

Chạy:  python3 scripts/download_pool_photos.py --cohort T-unbiased --limit 0
       (limit 0 = không giới hạn; --sleep 0.4 giây giữa 2 request)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prism import config as C          # noqa: E402
from prism import utils as U           # noqa: E402

log = U.get_logger("prism.download")

UA = "Mozilla/5.0 (research; PRISM pipeline; contact: see repo)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="T-unbiased",
                    choices=["A-dense", "B-anchor", "T-unbiased", "corpus"])
    ap.add_argument("--limit", type=int, default=0, help="0 = không giới hạn")
    ap.add_argument("--sleep", type=float, default=0.4,
                    help="giây nghỉ giữa 2 request (lịch sự với CDN)")
    ap.add_argument("--timeout", type=float, default=15.0)
    args = ap.parse_args()

    C.ensure_dirs()
    img_dir = C.RELIAB_DIR / "pool_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    index_path = C.RELIAB_DIR / "pool_image_index.json"

    cohorts = json.loads((C.STORE_DIR / "hotel_cohorts.json").read_text())
    keep = set(cohorts.get(args.cohort) or [])

    index: dict[str, str] = {}
    n_new = n_skip = n_err = 0
    for r in U.read_jsonl(C.STORE_DIR / "reviews.jsonl.gz"):
        if not r["has_photo"] or r["in_gold"]:
            continue
        if keep and r["hotel_id"] not in keep:
            continue
        if not r.get("photo_urls"):
            continue
        dest = img_dir / f"{r['review_uid']}.jpg"
        if dest.exists() and dest.stat().st_size > 0:
            index[r["review_uid"]] = str(dest)
            n_skip += 1
            continue
        if args.limit and n_new >= args.limit:
            continue
        url = r["photo_urls"][0]
        if not url.startswith(("http://", "https://")):
            n_err += 1
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                dest.write_bytes(resp.read())
            index[r["review_uid"]] = str(dest)
            n_new += 1
            if n_new % 200 == 0:
                log.info("  ... %d ảnh mới (bỏ qua %d đã có, lỗi %d)",
                         n_new, n_skip, n_err)
                U.write_json(index_path, index)     # checkpoint giữa chừng
            time.sleep(args.sleep)
        except Exception as e:                      # noqa: BLE001
            n_err += 1
            if n_err <= 20:
                log.warning("lỗi %s: %s", url[:80], e)

    U.write_json(index_path, index)
    log.info("xong: %d mới · %d đã có · %d lỗi -> %s (%d entry)",
             n_new, n_skip, n_err, index_path, len(index))


if __name__ == "__main__":
    main()
