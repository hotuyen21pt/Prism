"""
Smoke test — chạy toàn pipeline (trừ phần cần GPU/torch) trên mẫu nhỏ để bắt lỗi sớm.
KHÔNG thay thế run thật; chỉ xác nhận code chạy end-to-end với dữ liệu thật.

MỌI output của smoke nằm dưới outputs/drift/smoke/ — smoke KHÔNG được ghi vào
đường dẫn mặc định của Module D. Bản cũ gọi module_d_drift không truyền --out nên
ghi đè đúng drift_results.corpus.taxonomy_code.json, tức file kết quả chính thức,
cùng tên cùng schema và không cảnh báo gì.

Số kỳ vọng (số hotel gold, tổng quad) đọc từ tests/expected_counts.json để cập
nhật gold không làm smoke fail bằng AssertionError trống.

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
    smoke_dir = C.DRIFT_DIR / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    exp_path = C.TABSA_ROOT / "tests" / "expected_counts.json"
    exp = json.loads(exp_path.read_text(encoding="utf-8")) if exp_path.exists() else {}

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
    want_hotels = exp.get("gold_hotels_in_splits")
    if want_hotels is not None:
        assert len(hs) == want_hotels, (
            f"hotel_split phủ {len(hs)} hotel, kỳ vọng {want_hotels}. Nếu gold vừa "
            f"được cập nhật thì sửa {exp_path.name}, không sửa assert.")
    ok.append(f"blocklist({n_leak} leak/20k)")

    # 3. Module B data (round-trip toàn bộ gold)
    build_b_data()
    rep = json.loads((C.EXTRACT_DIR / "data_report.json").read_text())
    got_quads = rep["train_quads"] + rep["dev_quads"] + rep["test_quads"]
    want_quads = exp.get("total_gold_quads")
    if want_quads is not None:
        assert got_quads == want_quads, (
            f"tổng {got_quads} quad gold, kỳ vọng {want_quads}. Gold đổi thì sửa "
            f"{exp_path.name}.")
    assert rep.get("roundtrip_mismatch", 0) == 0, "round-trip linearize hỏng"
    ok.append(f"b_data({got_quads} quads, roundtrip clean)")

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
    f = smoke_dir / "smoke_quads.jsonl.gz"
    U.write_jsonl(f, quads)
    log.info("smoke quads: %d từ %d review", len(quads), n)
    drift_out = smoke_dir / "drift_results.smoke.json"
    subprocess.run([sys.executable, "-m", "prism.module_d_drift",
                    "--quads", str(f), "--level", "taxonomy_code",
                    "--cohort", "corpus", "--n-perm", "100",
                    "--out", str(drift_out)], check=True)
    res = json.loads(drift_out.read_text())
    assert len(res["results"]) >= 4
    ok.append(f"d_drift({len(res['results'])} aspects, {len(res['periods'])} periods)")

    # 5. injection shuffle làm negative control nhanh
    subprocess.run([sys.executable, "-m", "prism.eval_injection",
                    "--quads", str(f), "--test", "shuffle",
                    "--out-dir", str(smoke_dir)], check=True)
    rep = json.loads((smoke_dir / "injection_shuffle.json").read_text())
    ok.append(f"e4_shuffle(FPR={rep['empirical_fpr']}, {rep['verdict']})")

    log.info("SMOKE TEST PASS: %s", " · ".join(ok))
    log.info("mọi output smoke nằm dưới %s — kết quả thật KHÔNG bị đụng", smoke_dir)


if __name__ == "__main__":
    main()
