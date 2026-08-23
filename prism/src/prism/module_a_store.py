"""
Module A — Temporal Quad Store.

Biến pool 1,95M dòng raw thành corpus sạch cho toàn pipeline:
  A1 parse ngày (regex tiếng Việt)      A5 ép kiểu hotel_id
  A2 cắt cửa sổ 2022-03 → 2025-02       A6 lọc/lưu cờ text-bearing
  A3 dedup                              A7 (langid thật — tuỳ chọn, cần fasttext)
  A4 GOLD BLOCKLIST  [BLOCKING]         A8 gắn stratum + cohort

Chạy:  python3 -m prism.module_a_store
Ra  :  outputs/store/reviews.jsonl.gz  +  outputs/store/store_report.json
       outputs/store/hotel_cohorts.json
"""
from __future__ import annotations

import collections
import json

from . import config as C
from . import utils as U

log = U.get_logger("prism.A")


# ------------------------------------------------------------------ gold blocklist
def build_gold_blocklist() -> tuple[set[str], dict[str, set[str]], dict[str, str]]:
    """
    Ba khoá chặn leakage (gold ⊂ pool đã xác nhận — vd H3977209_R00003):
      1. (hotel_id, iso_date)            — chặn thô, an toàn nhất
      2. hash 120 ký tự đầu text gold    — chặn khớp nội dung
      3. map hotel -> gold_split         — để gắn cohort T-unbiased
    """
    keys_date: set[str] = set()
    keys_text: dict[str, set[str]] = collections.defaultdict(set)
    hotel_split: dict[str, str] = {}

    split_of: dict[str, str] = {}
    for name in ("train", "dev", "test"):
        for row in U.read_jsonl(C.SPLIT_DIR / f"{name}.jsonl"):
            split_of[row["source_review_id"]] = name

    for row in U.read_jsonl(C.GOLD_META):
        hid = str(row["hotel_id"])
        rid = row["source_review_id"]
        if row.get("review_date"):
            keys_date.add(f"{hid}|{row['review_date']}")
        keys_text[hid].add(U.norm_text(row["review_text"])[:120])
        sp = split_of.get(rid)
        # một hotel chỉ nằm trong đúng một split (đã kiểm chứng hotel-disjoint)
        if sp:
            hotel_split[hid] = sp
    log.info("blocklist: %d khoá (hotel,date) · %d hotel gold", len(keys_date), len(keys_text))
    return keys_date, keys_text, hotel_split


def is_gold_leak(hid: str, iso: str | None, text_pos: str, text_neg: str,
                 keys_date: set[str], keys_text: dict[str, set[str]]) -> bool:
    if hid not in keys_text:
        return False
    if iso and f"{hid}|{iso}" in keys_date:
        return True
    for t in (text_pos, text_neg, f"{text_pos} {text_neg}".strip()):
        if t and U.norm_text(t)[:120] in keys_text[hid]:
            return True
    return False


# ------------------------------------------------------------------------- main
def build_store() -> None:
    C.ensure_dirs()
    keys_date, keys_text, hotel_split = build_gold_blocklist()

    stats = collections.Counter()
    seen_dedup: set[str] = set()
    pool_count: collections.Counter = collections.Counter()   # hotel -> n (để tính cohort)
    out_path = C.STORE_DIR / "reviews.jsonl.gz"

    def rows():
        for i, d in enumerate(U.read_jsonl(C.POOL_JSONL)):
            if i and i % 200_000 == 0:
                log.info("  ... %d dòng", i)
            stats["read"] += 1
            hid = str(d.get("hotel_id") or "")

            parsed = U.parse_review_date(d.get("review_date"))
            if parsed is None:                       # [đo] 184 dòng
                stats["drop_no_date"] += 1
                continue
            iso, period = parsed
            if not U.in_window(period):              # A2: cắt 2022-02 và 2025-03
                stats["drop_window"] += 1
                continue

            text_pos = U.nfc(d.get("review_positive"))
            text_neg = U.nfc(d.get("review_negative"))

            # A3 dedup — [đo] 15.137 dòng trùng (0,78%)
            dk = U.text_hash(f"{hid}|{d.get('name')}|{iso}|{text_pos}|{text_neg}")
            if dk in seen_dedup:
                stats["drop_dup"] += 1
                continue
            seen_dedup.add(dk)

            # A4 gold blocklist [BLOCKING]
            leaked = is_gold_leak(hid, iso, text_pos, text_neg, keys_date, keys_text)
            if leaked:
                stats["flag_gold_leak"] += 1

            has_text = bool(text_pos or text_neg)    # [đo] 59,0% pool
            stats["has_text" if has_text else "no_text"] += 1
            n_words = len(f"{text_pos} {text_neg}".split())
            pool_count[hid] += 1

            yield {
                "review_uid": f"{hid}_{iso.replace('-', '')}_{stats['read']:08d}",
                "hotel_id": hid,
                "iso_date": iso,
                "period": period,
                "text_pos": text_pos or None,
                "text_neg": text_neg or None,
                "score": U.parse_score(d.get("review_score")),
                "stars_rating": d.get("stars_rating"),
                "stratum": list(U.make_stratum(d.get("country"), d.get("state"), n_words)),
                "country_bloc": U.country_bloc(d.get("country")),
                "traveller": U.traveller_type(d.get("state")),
                "n_words": n_words,
                "nights": U.parse_nights(d.get("date")),
                "room": U.nfc(d.get("room")) or None,
                "has_text": has_text,
                "has_photo": bool(d.get("review_photo")),
                "photo_urls": list(d["review_photo"].keys()) if d.get("review_photo") else [],
                "in_gold": leaked,                    # cờ — KHÔNG dùng các dòng này để train/self-train
                "gold_split": hotel_split.get(hid),   # train/dev/test/None (cấp hotel)
            }

    n = U.write_jsonl(out_path, rows())
    log.info("đã ghi %d review -> %s", n, out_path)

    # A8: cohort theo hotel
    gold_hotels = set(hotel_split)
    cohorts = {
        "A-dense":    sorted(h for h in gold_hotels if pool_count.get(h, 0) >= 1000),
        "B-anchor":   sorted(h for h in gold_hotels if pool_count.get(h, 0) >= 300),
        "T-unbiased": sorted(h for h, sp in hotel_split.items() if sp == "test"),
        "corpus":     [],   # rỗng = mọi hotel
    }
    U.write_json(C.STORE_DIR / "hotel_cohorts.json", cohorts)

    report = {
        "stats": dict(stats),
        "n_written": n,
        "n_hotels": len(pool_count),
        "cohort_sizes": {k: (len(v) or len(pool_count)) for k, v in cohorts.items()},
        "window": [C.WINDOW_START, C.WINDOW_END],
        "gold_leak_flagged": stats["flag_gold_leak"],
    }
    U.write_json(C.STORE_DIR / "store_report.json", report)
    log.info("report: %s", json.dumps(report["stats"], ensure_ascii=False))
    log.info("cohorts: %s", {k: report["cohort_sizes"][k] for k in cohorts})


if __name__ == "__main__":
    build_store()
