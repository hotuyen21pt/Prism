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

# Ký tự phân cách PHẢI được escape trong nội dung term, nếu không quad bị mất trắng:
# aspect_term "giá | chất lượng" sinh ra 5 field -> parse_linearized bỏ cả quad im lặng.
# Escape ở cả hai chiều để round-trip là song ánh với mọi nội dung term.
_ESCAPES = ((QUAD_OPEN, "&lt;quad&gt;"), (QUAD_CLOSE, "&lt;/quad&gt;"), ("|", "&#124;"))


def esc_term(s: str) -> str:
    for raw, safe in _ESCAPES:
        s = s.replace(raw, safe)
    return s


def unesc_term(s: str) -> str:
    for raw, safe in _ESCAPES:
        s = s.replace(safe, raw)
    return s


def linearize(quads: list[dict]) -> str:
    parts = []
    for q in quads:
        parts.append(
            f"{QUAD_OPEN} {esc_term(q.get('aspect_term') or NULL)}{SEP}{q['taxonomy_code']}"
            f"{SEP}{esc_term(q.get('opinion_term') or NULL)}{SEP}{q['sentiment']} {QUAD_CLOSE}"
        )
    return " ".join(parts)


def parse_linearized(s: str) -> list[dict]:
    """Nghịch đảo của linearize — dùng lúc inference. Bỏ qua quad hỏng định dạng."""
    out = []
    for chunk in s.split(QUAD_OPEN)[1:]:
        body = chunk.split(QUAD_CLOSE)[0].strip()
        fields = [f.strip() for f in body.split("|")]
        if len(fields) < 4:
            continue
        # Target của ta luôn có đúng 4 field (term đã escape). Nhưng model TỰ SINH
        # có thể chèn '|' thô vào term -> >4 field. Thay vì bỏ cả quad (mất recall
        # lệch theo loại term, không lệch theo aspect nên không hiện ở bảng nào),
        # neo vào hai field xác định được: sentiment là field CUỐI, taxonomy_code là
        # field duy nhất thuộc CODE2CAT. Phần trước code = aspect, phần giữa = opinion.
        sent = fields[-1]
        code_at = [i for i, f in enumerate(fields[:-1]) if f in C.CODE2CAT]
        if len(code_at) != 1 or sent not in C.SENTIMENTS:
            continue    # taxonomy hard filter ngay tại parse
        ci = code_at[0]
        code = fields[ci]
        a = unesc_term(" | ".join(fields[:ci]).strip())
        o = unesc_term(" | ".join(fields[ci + 1:-1]).strip())
        if not a or not o:
            continue    # thiếu hẳn vị trí aspect hoặc opinion -> quad hỏng
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
    # KHÔNG đụng test gốc để test chuẩn vẫn nguyên vẹn).
    # Review THIẾU ngày có review_date == "" và ""[:7] < mọi chuỗi -> nếu không lọc
    # riêng thì chúng rơi hết vào early, làm tập early nhiễm dữ liệu không xác định
    # được thời gian. Tách hẳn ra và đếm.
    pool = all_rows["train"] + all_rows["dev"]
    dated  = [r for r in pool if len(r["review_date"]) >= 7]
    undated = [r for r in pool if len(r["review_date"]) < 7]
    early = [r for r in dated if r["review_date"][:7] <= "2024-06"]
    late  = [r for r in dated if r["review_date"][:7] >= "2024-07"]
    U.write_jsonl(C.EXTRACT_DIR / "chrono_train.t2t.jsonl", early)
    U.write_jsonl(C.EXTRACT_DIR / "chrono_test.t2t.jsonl", late)
    stats["chrono_train"], stats["chrono_test"] = len(early), len(late)
    stats["chrono_undated_excluded"] = len(undated)
    if not dated:
        # ĐÃ GẶP THẬT trên hamos-mabsa hiện tại: metadata/reviews.jsonl KHÔNG có
        # trường review_date, nên gold không có ngày và probe E1c bất khả thi.
        # Trước khi sửa, ""[:7] == "" <= "2024-06" là True nên TOÀN BỘ train+dev
        # rơi vào chrono_train và chrono_test rỗng — split sai mà không ai biết.
        log.error("E1c KHÔNG chạy được: 0/%d segment có review_date. %s không có "
                  "trường 'review_date' — cần metadata/reviews_with_dates.jsonl. "
                  "chrono_train/chrono_test được ghi RỖNG, đừng train trên chúng.",
                  len(pool), C.GOLD_META)
    elif undated:
        log.warning("E1c: %d/%d segment thiếu review_date -> LOẠI khỏi cả "
                    "chrono_train và chrono_test", len(undated), len(pool))

    U.write_json(C.EXTRACT_DIR / "data_report.json", dict(stats))
    log.info("%s", dict(stats))


if __name__ == "__main__":
    build()
