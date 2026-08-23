"""
Smoke test — chạy toàn pipeline (trừ phần cần GPU/torch) trên mẫu nhỏ để bắt lỗi sớm.
KHÔNG thay thế run thật; chỉ xác nhận code chạy end-to-end với dữ liệu thật.

Chạy:  python3 -m prism.smoke_test
"""
from __future__ import annotations

import collections
import itertools
import json
import random
import subprocess
import sys

from . import config as C
from . import utils as U
from .module_a_store import build_gold_blocklist, is_gold_leak
from .module_b_data import build as build_b_data, linearize, parse_linearized

log = U.get_logger("prism.smoke")


def main() -> None:
    C.ensure_dirs()
    ok = []

    # 1. config paths
    for p in (C.POOL_JSONL, C.GOLD_QUADS, C.GOLD_META, C.SPLIT_DIR / "train.jsonl"):
        assert p.exists(), f"thiếu {p}"
    ok.append("paths")

    # 2. blocklist trên 20k dòng pool đầu
    kd, kt, hs = build_gold_blocklist()
    n_leak = 0
    for d in itertools.islice(U.read_jsonl(C.POOL_JSONL), 20000):
        parsed = U.parse_review_date(d.get("review_date"))
        iso = parsed[0] if parsed else None
        if is_gold_leak(str(d.get("hotel_id")), iso, U.nfc(d.get("review_positive")),
                        U.nfc(d.get("review_negative")), kd, kt):
            n_leak += 1
    log.info("blocklist bắt %d dòng leak trong 20k mẫu", n_leak)
    assert len(hs) == 3399, "hotel_split phải phủ 3.399 hotel gold"
    ok.append(f"blocklist({n_leak} leak/20k)")

    # 3. Module B data (round-trip toàn bộ gold)
    build_b_data()
    rep = json.loads((C.EXTRACT_DIR / "data_report.json").read_text())
    assert rep["train_quads"] + rep["dev_quads"] + rep["test_quads"] == 23995
    assert rep.get("roundtrip_mismatch", 0) == 0, "round-trip linearize hỏng"
    ok.append("b_data(23995 quads, roundtrip clean)")

    # 4. Module D trên pseudo-quad tổng hợp từ keyword-probe logic (mẫu 50k pool)
    rng = random.Random(C.RANDOM_SEED)
    quads = []
    kw = {"AM_FOOD": "breakfast", "FAC_ROOM": "room", "AM_WIFI": "wifi",
          "SER_ATTITUDE": "staff", "AM_POOL": "pool"}
    n = 0
    for d in U.read_jsonl(C.POOL_JSONL):
        parsed = U.parse_review_date(d.get("review_date"))
        if not parsed or not U.in_window(parsed[1]):
            continue
        neg = U.nfc(d.get("review_negative")).lower()
        if not neg:
            continue
        st = list(U.make_stratum(d.get("country"), d.get("state"), len(neg.split())))
        for code, term in kw.items():
            if term in neg:
                quads.append({"review_uid": f"S{n}", "hotel_id": str(d["hotel_id"]),
                              "period": parsed[1], "stratum": st, "phi": "NEG",
                              "taxonomy_code": code, "aspect_category": C.CODE2CAT[code],
                              "sentiment": "negative", "conf_seq": rng.uniform(.6, .99),
                              "w": 1.0, "n_words": len(neg.split()),
                              "score": None, "has_photo": False,
                              "provenance_flip": False, "p_posterior": 0.9})
        n += 1
        if n >= 50000:
            break
    f = C.DRIFT_DIR / "smoke_quads.jsonl.gz"
    U.write_jsonl(f, quads)
    log.info("smoke quads: %d từ %d review", len(quads), n)
    subprocess.run([sys.executable, "-m", "prism.module_d_drift",
                    "--quads", str(f), "--level", "taxonomy_code",
                    "--cohort", "corpus", "--n-perm", "100"], check=True)
    res = json.loads((C.DRIFT_DIR / "drift_results.corpus.taxonomy_code.json").read_text())
    assert len(res["results"]) >= 4
    ok.append(f"d_drift({len(res['results'])} aspects, {len(res['periods'])} periods)")

    # 5. injection shuffle làm negative control nhanh
    subprocess.run([sys.executable, "-m", "prism.eval_injection",
                    "--quads", str(f), "--test", "shuffle"], check=True)
    rep = json.loads((C.DRIFT_DIR / "injection_shuffle.json").read_text())
    ok.append(f"e4_shuffle(FPR={rep['empirical_fpr']}, {rep['verdict']})")

    log.info("SMOKE TEST PASS: %s", " · ".join(ok))


if __name__ == "__main__":
    main()
