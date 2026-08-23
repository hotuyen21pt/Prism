"""PRISM — tiện ích dùng chung: parse ngày VN, chuẩn hoá text, strata, I/O, thống kê."""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import config as C

# ------------------------------------------------------------------------ logging
def get_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(name)-22s %(levelname)-7s %(message)s",
                                         datefmt="%H:%M:%S"))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
    return lg


# ---------------------------------------------------------------------------- I/O
def read_jsonl(path: Path | str) -> Iterator[dict]:
    path = Path(path)
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path | str, rows: Iterable[dict], gzip_out: bool = False) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    op = gzip.open if gzip_out or path.suffix == ".gz" else open
    n = 0
    with op(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------------- normalisation
def nfc(s: str | None) -> str:
    return unicodedata.normalize("NFC", (s or "")).strip()


def norm_text(s: str | None) -> str:
    """Chuẩn hoá mạnh dùng để so khớp (bỏ dấu câu, gộp khoảng trắng, hạ chữ)."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def text_hash(s: str) -> str:
    return hashlib.md5(norm_text(s).encode("utf-8")).hexdigest()


def quad_uid(q: dict) -> str:
    """
    Định danh ổn định cho một pseudo-quad. PHẢI dùng chung hàm này ở mọi nơi
    (make_audit_samples tạo mẫu audit, module_c bridge đối chiếu audit) —
    hai bên tự dựng khoá riêng sẽ không bao giờ khớp nhau.
    """
    return (f"{q['review_uid']}_{q['phi']}_{q['taxonomy_code']}"
            f"_{text_hash(str(q.get('opinion_term')))[:6]}")


# ------------------------------------------------------------------- date parsing
_VI_DATE = re.compile(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})")
_ISO     = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_SLASH   = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")


def parse_review_date(raw: str | None) -> tuple[str, str] | None:
    """
    Trả (iso_date, period) với period = 'YYYY-MM'. None nếu không parse được.

    Pool Booking dùng chuỗi tiếng Việt 'Ngày đánh giá: ngày D tháng M năm YYYY'
    ([đo] 1.949.420/1.949.604 = 99,99% khớp mẫu này; 184 dòng null).
    """
    if not raw:
        return None
    m = _VI_DATE.search(raw)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
    else:
        m = _ISO.search(raw)
        if m:
            y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
        else:
            m = _SLASH.search(raw)
            if not m:
                return None
            d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return f"{y}-{mo:02d}-{d:02d}", f"{y}-{mo:02d}"


_SCORE = re.compile(r"(\d{1,2})(?:[.,](\d))?")


def parse_score(raw: str | None) -> float | None:
    """'Đạt điểm 9,0' -> 9.0 . Pool dùng dấu phẩy thập phân."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    m = _SCORE.search(str(raw).replace("Đạt điểm", ""))
    if not m:
        return None
    v = float(f"{m.group(1)}.{m.group(2) or 0}")
    return v if 0.0 <= v <= 10.0 else None


_NIGHTS = re.compile(r"(\d+)\s*đêm")


def parse_nights(raw: str | None) -> int | None:
    m = _NIGHTS.search(raw or "")
    return int(m.group(1)) if m else None


# ------------------------------------------------------------------------- strata
def country_bloc(country: str | None) -> str:
    c = nfc(country).lower()
    if c == "việt nam":
        return "VN"
    if c in C.WEST_COUNTRIES:
        return "WEST"
    if c in C.ASIA_COUNTRIES:
        return "ASIA"
    return "OTH"


def traveller_type(state: str | None) -> str:
    s = nfc(state)
    return s if s in C.TRAVELLER_TYPES else "NA"


def length_bin(n_words: int) -> str:
    for i, (lo, hi) in enumerate(C.LENGTH_BINS):
        if lo <= n_words < hi:
            return f"L{i}"
    return f"L{len(C.LENGTH_BINS) - 1}"


def make_stratum(country: str | None, state: str | None, n_words: int,
                 use_length: bool = C.USE_LENGTH_IN_STRATA) -> tuple[str, ...]:
    base = (country_bloc(country), traveller_type(state))
    return base + (length_bin(n_words),) if use_length else base


def in_window(period: str, start: str = C.WINDOW_START, end: str = C.WINDOW_END) -> bool:
    return start <= period <= end


def periods_in_window(start: str = C.WINDOW_START, end: str = C.WINDOW_END) -> list[str]:
    y0, m0 = map(int, start.split("-"))
    y1, m1 = map(int, end.split("-"))
    out, y, m = [], y0, m0
    while (y, m) <= (y1, m1):
        out.append(f"{y}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


# ------------------------------------------------------ compositional / statistics
def clr(shares: dict[str, float], eps: float = 1e-6) -> dict[str, float]:
    """
    Centered log-ratio. Bắt buộc vì các share cộng lại = 1: hồi quy trực tiếp trên
    share vi phạm ràng buộc simplex và sinh tương quan âm giả.
    """
    p = {k: max(v, eps) for k, v in shares.items()}
    g = math.exp(sum(math.log(v) for v in p.values()) / len(p))
    return {k: math.log(v / g) for k, v in p.items()}


def benjamini_hochberg(pvals: list[float], alpha: float = C.FDR_ALPHA) -> list[float]:
    """Trả p-value đã hiệu chỉnh (q-value), cùng thứ tự với input."""
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    q = [0.0] * n
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = n - rank + 1
        prev = min(prev, pvals[idx] * n / i)
        q[idx] = min(prev, 1.0)
    return q


def welch_t(a: list[float], b: list[float]) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    den = math.sqrt(va / na + vb / nb)
    return 0.0 if den == 0 else (ma - mb) / den
