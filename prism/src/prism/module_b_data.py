"""
Module B (phần dữ liệu) — chuyển gold splits sang định dạng text-to-text cho extractor.

Định dạng target tuyến tính hoá (một chuỗi cho mọi quad của segment):
    <quad> aspect | taxonomy_code | opinion | sentiment </quad> <quad> ... </quad>
Aspect/opinion implicit biểu diễn bằng token 'NULL'.

Chạy:  python3 -m prism.module_b_data
Ra  :  outputs/extract/{train,dev,test}.t2t.jsonl  +  chronological probe splits
"""
from __future__ import annotations

import collections

from . import config as C
from . import utils as U

log = U.get_logger("prism.B.data")

QUAD_OPEN, QUAD_CLOSE, SEP, NULL = "<quad>", "</quad>", " | ", "NULL"
TASK_PREFIX = "extract quads: "


def linearize(quads: list[dict]) -> str:
    parts = []
    for q in quads:
        parts.append(
            f"{QUAD_OPEN} {q.get('aspect_term') or NULL}{SEP}{q['taxonomy_code']}"
            f"{SEP}{q.get('opinion_term') or NULL}{SEP}{q['sentiment']} {QUAD_CLOSE}"
        )
    return " ".join(parts)


def parse_linearized(s: str) -> list[dict]:
    """Nghịch đảo của linearize — dùng lúc inference. Bỏ qua quad hỏng định dạng."""
    out = []
    for chunk in s.split(QUAD_OPEN)[1:]:
        body = chunk.split(QUAD_CLOSE)[0].strip()
        fields = [f.strip() for f in body.split("|")]
        if len(fields) != 4:
            continue
        a, code, o, sent = fields
        if code not in C.CODE2CAT or sent not in C.SENTIMENTS:
            continue    # taxonomy hard filter ngay tại parse
        out.append({
            "aspect_term": None if a == NULL else a,
            "taxonomy_code": code,
            "aspect_category": C.CODE2CAT[code],
            "opinion_term": None if o == NULL else o,
            "sentiment": sent,
            "aspect_implicit": a == NULL,
            "opinion_implicit": o == NULL,
        })
    return out


def review_dates() -> dict[str, str]:
    return {r["source_review_id"]: r.get("review_date") or ""
            for r in U.read_jsonl(C.GOLD_META)}


def build() -> None:
    C.ensure_dirs()
    dates = review_dates()
    stats = collections.Counter()

    all_rows: dict[str, list[dict]] = {}
    for split in ("train", "dev", "test"):
        rows = []
        for r in U.read_jsonl(C.SPLIT_DIR / f"{split}.jsonl"):
            rid = r["source_review_id"]
            rows.append({
                "instance_id": r["instance_id"],
                "source_review_id": rid,
                "review_date": dates.get(rid, ""),
                "input": TASK_PREFIX + r["text"],
                "target": linearize(r["quads"]),
                "n_quads": len(r["quads"]),
            })
            stats[f"{split}_segments"] += 1
            stats[f"{split}_quads"] += len(r["quads"])
        # round-trip check: linearize -> parse phải khôi phục đủ số quad hợp lệ
        for row, src in zip(rows, U.read_jsonl(C.SPLIT_DIR / f"{split}.jsonl")):
            got = parse_linearized(row["target"])
            want = [q for q in src["quads"] if q["taxonomy_code"] in C.CODE2CAT]
            if len(got) != len(want):
                stats["roundtrip_mismatch"] += 1
        U.write_jsonl(C.EXTRACT_DIR / f"{split}.t2t.jsonl", rows)
        all_rows[split] = rows

    # E1c — chronological probe: train ≤2024-06, test ≥2024-07 (chỉ từ train+dev gốc,
    # KHÔNG đụng test gốc để test chuẩn vẫn nguyên vẹn)
    pool = all_rows["train"] + all_rows["dev"]
    early = [r for r in pool if r["review_date"][:7] <= "2024-06"]
    late  = [r for r in pool if r["review_date"][:7] >= "2024-07"]
    U.write_jsonl(C.EXTRACT_DIR / "chrono_train.t2t.jsonl", early)
    U.write_jsonl(C.EXTRACT_DIR / "chrono_test.t2t.jsonl", late)
    stats["chrono_train"], stats["chrono_test"] = len(early), len(late)

    U.write_json(C.EXTRACT_DIR / "data_report.json", dict(stats))
    log.info("%s", dict(stats))


if __name__ == "__main__":
    build()
