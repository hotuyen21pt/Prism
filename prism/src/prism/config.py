"""
PRISM — cấu hình tập trung.

Mọi hằng số/ngưỡng của pipeline nằm ở đây, không rải rác trong code.
Các giá trị có chú thích "[đo]" là số đã đo trực tiếp trên dữ liệu trong quá trình
audit (xem docs/method_table_q1.md), không phải giá trị đặt tay.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- paths
# Bố cục src-layout: file này nằm ở <repo>/src/prism/config.py -> repo root = parents[2]
TABSA_ROOT = Path(os.environ.get("PRISM_TABSA_ROOT", Path(__file__).resolve().parents[2]))
HAMOS_ROOT = Path(os.environ.get("PRISM_HAMOS_ROOT", TABSA_ROOT.parent / "hamos-mabsa"))


def _hamos_file(name: str, fallback: Path | None = None) -> Path:
    for path in (HAMOS_ROOT / "data" / name, HAMOS_ROOT / name):
        if path.exists():
            return path
    return fallback or HAMOS_ROOT / "data" / name

POOL_JSONL   = TABSA_ROOT / "data/raw/hotel_booking_unlabeled.jsonl"
SPLIT_DIR    = TABSA_ROOT / "data/raw"                          # train/dev/test.jsonl
GOLD_META    = _hamos_file(
    "metadata/reviews_with_dates.jsonl",
    _hamos_file("metadata/reviews.jsonl",
                TABSA_ROOT / "data/raw/hotel_absa_labeled.jsonl"),
)
GOLD_QUADS   = _hamos_file("annotations/quads.jsonl")
GOLD_SEGS    = _hamos_file("metadata/segments.jsonl")
GOLD_IMAGES  = _hamos_file("metadata/images.jsonl")
IMAGE_DIR    = _hamos_file("images")

WORK_DIR     = Path(os.environ.get("PRISM_WORK_DIR", TABSA_ROOT / "outputs"))
STORE_DIR    = WORK_DIR / "store"          # Module A
EXTRACT_DIR  = WORK_DIR / "extract"        # Module B
RELIAB_DIR   = WORK_DIR / "reliability"    # Module C
DRIFT_DIR    = WORK_DIR / "drift"          # Module D
MODEL_DIR    = Path(os.environ.get("PRISM_MODEL_DIR", TABSA_ROOT / "models"))

# --------------------------------------------------------------------- time window
# [đo] pool trải 2022-02 → 2025-03; cắt hai biên vì dữ liệu dở dang:
#      2022-02 chỉ 1.391 review / 549 hotel; 2025-03 chỉ 11.211 review (crawl dừng 10/03).
WINDOW_START = "2022-03"
WINDOW_END   = "2025-02"          # 36 kỳ tháng

# ------------------------------------------------------------------------ taxonomy
CATEGORIES = ["FACILITY", "AMENITY", "SERVICE", "EXPERIENCE", "LOYALTY", "BRANDING"]

# [đo] 14 code có >=300 quad gold -> đủ mẫu để báo cáo riêng
CODES_REPORTABLE = [
    "FAC_ROOM", "FAC_VIEW_LOCATION", "AM_POOL", "FAC_BUILDING", "SER_ATTITUDE",
    "AM_FOOD", "FAC_BATH", "EXP_OVERALL", "AM_ROOM_UTIL", "FAC_ENV",
    "LOY_RETURN", "FAC_INTERIOR", "EXP_VALUE", "SER_SUPPORT",
]
# 17 code còn lại: gộp lên category cha khi báo cáo drift
CODE2CAT = {
    "FAC_ROOM": "FACILITY", "FAC_VIEW_LOCATION": "FACILITY", "FAC_BUILDING": "FACILITY",
    "FAC_BATH": "FACILITY", "FAC_ENV": "FACILITY", "FAC_INTERIOR": "FACILITY",
    "FAC_CLIMATE": "FACILITY", "FAC_SECURITY": "FACILITY",
    "AM_POOL": "AMENITY", "AM_FOOD": "AMENITY", "AM_ROOM_UTIL": "AMENITY",
    "AM_TRANSPORT": "AMENITY", "AM_WELLNESS": "AMENITY", "AM_ENT": "AMENITY",
    "AM_WIFI": "AMENITY", "AM_UTILITY": "AMENITY",
    "SER_ATTITUDE": "SERVICE", "SER_SUPPORT": "SERVICE", "SER_OPERATION": "SERVICE",
    "SER_COMM": "SERVICE", "SER_PROFESSIONALISM": "SERVICE",
    "EXP_OVERALL": "EXPERIENCE", "EXP_VALUE": "EXPERIENCE", "EXP_EMOTION": "EXPERIENCE",
    "EXP_SAFETY": "EXPERIENCE",
    "LOY_RETURN": "LOYALTY", "LOY_RECOMMEND": "LOYALTY", "LOY_PREFERENCE": "LOYALTY",
    "BRA_CONSISTENCY": "BRANDING", "BRA_LUXURY": "BRANDING", "BRA_REPUTE": "BRANDING",
}
ALL_CODES = sorted(CODE2CAT)
SENTIMENTS = ["positive", "neutral", "negative"]

# ------------------------------------------------------- Module B: field provenance
# [đo] trên 14.129 quad gold xác định được nguồn trường (docs/method_table_q1.md §B.1).
#      Bất đối xứng 93,1 / 64,1 -> KHÔNG giả định đối xứng, KHÔNG dùng làm bộ lọc cứng.
PROVENANCE_PRIOR = {
    "POS": {"positive": 0.931, "neutral": 0.036, "negative": 0.033},
    "NEG": {"positive": 0.217, "neutral": 0.142, "negative": 0.641},
}
PROVENANCE_LAMBDA = 0.7      # trọng số model vs prior; chọn lại trên dev (Module B)

# ------------------------------------------------------------------ Module D: strata
COUNTRY_BLOCS = ["VN", "WEST", "ASIA", "OTH"]
WEST_COUNTRIES = {
    "pháp", "úc", "đức", "vương quốc anh", "mỹ", "tây ban nha", "hà lan", "ý", "bỉ",
    "canada", "thụy sĩ", "thuỵ sĩ", "áo", "đan mạch", "thụy điển", "thuỵ điển",
    "na uy", "ireland", "new zealand", "ba lan", "séc", "bồ đào nha", "phần lan",
}
ASIA_COUNTRIES = {
    "nhật bản", "hàn quốc", "trung quốc", "thái lan", "singapore", "malaysia",
    "đài loan", "hồng kông", "indonesia", "philippines", "ấn độ", "campuchia", "lào",
}
# [đo] 4 loại khách chiếm 99,7%: Cặp đôi 40,8% · Gia đình 24,0% · Khách lẻ 18,9% · Nhóm 16,0%
TRAVELLER_TYPES = ["Cặp đôi", "Phòng gia đình", "Khách lẻ", "Nhóm", "NA"]
LENGTH_BINS = [(0, 15), (15, 45), (45, 10**9)]     # L0 / L1 / L2 theo số từ
USE_LENGTH_IN_STRATA = False   # True -> 60 ô thay vì 20 (ablation robustness)
MIN_STRATUM_N = 30             # ô dưới ngưỡng này bị bỏ qua trong kỳ đó
MIN_VALENCE_CELL_W = 10        # kênh ν: ô (kỳ×stratum×aspect) dưới tổng w này bị bỏ

# ------------------------------------------------------------------ Module D: thống kê
N_BOOTSTRAP   = 1000
N_PERMUTATION = 1000
FDR_ALPHA     = 0.05
T_THRESHOLD   = 2.5            # ngưỡng |t| coi là có ý nghĩa trước FDR
RANDOM_SEED   = 20260821

# ---------------------------------------------------------------------- cohort
COHORT_DEFS = {
    "A-dense":    dict(needs_gold=True,  min_pool=1000),   # [đo] 276 hotel
    "B-anchor":   dict(needs_gold=True,  min_pool=300),    # [đo] 1.221 hotel
    "T-unbiased": dict(gold_split="test", min_pool=0), # [đo] 514 hotel — extractor chưa thấy
    "corpus":     dict(needs_gold=False, min_pool=0),      # toàn bộ 10.631 hotel
}


@dataclass
class RunConfig:
    """Cấu hình một lần chạy; ghi kèm output để tái lập."""
    window_start: str = WINDOW_START
    window_end: str = WINDOW_END
    use_length_in_strata: bool = USE_LENGTH_IN_STRATA
    min_stratum_n: int = MIN_STRATUM_N
    n_bootstrap: int = N_BOOTSTRAP
    n_permutation: int = N_PERMUTATION
    fdr_alpha: float = FDR_ALPHA
    seed: int = RANDOM_SEED
    provenance_lambda: float = PROVENANCE_LAMBDA
    encoder_version: str = "unset"
    extractor_ckpt: str = "unset"
    notes: str = ""

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def ensure_dirs() -> None:
    for d in (WORK_DIR, STORE_DIR, EXTRACT_DIR, RELIAB_DIR, DRIFT_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
