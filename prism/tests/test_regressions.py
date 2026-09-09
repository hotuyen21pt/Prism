"""
Test chống hồi quy cho các lỗi đã sửa trong docs/prism_code_review.

Mỗi test ở đây tương ứng một lỗi mà TRƯỚC KHI SỬA không có test nào bắt được —
đó là lý do chúng tồn tại lâu. Đừng nới lỏng chúng; nếu một test ở đây fail thì
lỗi cũ đã quay lại.
"""
import importlib.util
import json
import os
import random
import unittest
from pathlib import Path

from prism import config as C
from prism import utils as U
from prism.eval_injection import inject_shuffle
from prism.module_b_data import linearize, parse_linearized


class TestSeparatorInTerm(unittest.TestCase):
    """#8 — ký tự phân cách trong term làm mất trắng cả quad."""

    def _roundtrip(self, aspect, opinion):
        quads = [{"aspect_term": aspect, "taxonomy_code": "FAC_ROOM",
                  "opinion_term": opinion, "sentiment": "negative"}]
        return parse_linearized(linearize(quads))

    def test_pipe_in_aspect_survives(self):
        out = self._roundtrip("giá | chất lượng", "kém")
        self.assertEqual(len(out), 1, "quad có '|' trong aspect bị mất")
        self.assertEqual(out[0]["aspect_term"], "giá | chất lượng")
        self.assertEqual(out[0]["sentiment"], "negative")

    def test_pipe_in_opinion_survives(self):
        out = self._roundtrip("wifi", "chậm | hay mất")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["opinion_term"], "chậm | hay mất")

    def test_quad_tag_in_term_survives(self):
        out = self._roundtrip("a </quad> b", "x <quad> y")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["aspect_term"], "a </quad> b")
        self.assertEqual(out[0]["opinion_term"], "x <quad> y")

    def test_multi_quad_with_pipes_keeps_all(self):
        quads = [
            {"aspect_term": "giá | chất lượng", "taxonomy_code": "EXP_VALUE",
             "opinion_term": "ổn", "sentiment": "positive"},
            {"aspect_term": "phòng", "taxonomy_code": "FAC_ROOM",
             "opinion_term": "ồn", "sentiment": "negative"},
        ]
        out = parse_linearized(linearize(quads))
        self.assertEqual([q["taxonomy_code"] for q in out],
                         ["EXP_VALUE", "FAC_ROOM"])

    def test_escape_is_reversible(self):
        for s in ("a|b", "<quad>", "</quad>", "a | b </quad> c", "bình thường"):
            self.assertEqual(len(parse_linearized(linearize(
                [{"aspect_term": s, "taxonomy_code": "FAC_ROOM",
                  "opinion_term": s, "sentiment": "neutral"}]))), 1, s)

    def test_model_output_with_raw_pipe_still_parses(self):
        # model TỰ SINH '|' thô (không qua esc_term) -> neo vào code + sentiment
        s = "<quad> giá | chất lượng | EXP_VALUE | tạm được | positive </quad>"
        out = parse_linearized(s)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["taxonomy_code"], "EXP_VALUE")
        self.assertEqual(out[0]["sentiment"], "positive")

    def test_term_equal_to_a_taxonomy_code(self):
        """Term có thể TRÙNG một taxonomy_code. Nếu parse quét CODE2CAT thay vì đọc
        theo vị trí thì sẽ thấy 2 ứng viên và bỏ cả quad — quad hoàn toàn hợp lệ."""
        out = self._roundtrip("AM_POOL", "sạch")
        self.assertEqual(len(out), 1, "aspect trùng code bị mất quad")
        self.assertEqual(out[0]["aspect_term"], "AM_POOL")
        out = self._roundtrip("phòng", "FAC_ROOM")
        self.assertEqual(len(out), 1, "opinion trùng code bị mất quad")
        self.assertEqual(out[0]["opinion_term"], "FAC_ROOM")

    def test_escape_is_injective_on_sentinel_text(self):
        """Term chứa đúng chuỗi escape phải round-trip nguyên vẹn: '&' được escape
        ĐẦU TIÊN và unescape CUỐI CÙNG, nếu không 'a&#124;b' bị đọc thành 'a|b'."""
        for s in ("a&#124;b", "giá & chất lượng", "&amp;", "&lt;quad&gt;", "a&b|c"):
            out = self._roundtrip(s, s)
            self.assertEqual(len(out), 1, s)
            self.assertEqual(out[0]["aspect_term"], s, f"escape mất nội dung: {s!r}")
            self.assertEqual(out[0]["opinion_term"], s)

    def test_still_drops_genuinely_malformed(self):
        self.assertEqual(parse_linearized("<quad> thiếu | trường </quad>"), [])
        self.assertEqual(parse_linearized("<quad> a | XX_BAD | b | positive </quad>"), [])
        self.assertEqual(parse_linearized("<quad> a | FAC_ROOM | b | xxx </quad>"), [])


class TestShuffleIsReviewLevel(unittest.TestCase):
    """#6 — xáo timestamp phải ở MỨC REVIEW, không phải mức quad."""

    @staticmethod
    def _quads():
        out = []
        for i in range(50):                      # mỗi review 3 quad
            for j in range(3):
                out.append({"review_uid": f"R{i}", "hotel_id": "H1",
                            "period": f"2022-{(i % 12) + 1:02d}",
                            "stratum": ["VN", "Cặp đôi"], "phi": "NEG",
                            "taxonomy_code": "AM_FOOD", "aspect_category": "AMENITY",
                            "sentiment": "negative", "conf_seq": 0.9, "idx": j})
        return out

    def test_all_quads_of_a_review_share_one_period(self):
        inj = inject_shuffle(self._quads(), random.Random(1))
        by_uid = {}
        for q in inj:
            by_uid.setdefault(q["review_uid"], set()).add(q["period"])
        for uid, pers in by_uid.items():
            self.assertEqual(len(pers), 1,
                             f"{uid} bị xé sang {len(pers)} kỳ khác nhau")

    def test_review_level_period_multiset_preserved(self):
        quads = self._quads()
        inj = inject_shuffle(quads, random.Random(2))

        def review_periods(rows):
            d = {}
            for q in rows:
                d.setdefault(q["review_uid"], q["period"])
            return sorted(d.values())

        self.assertEqual(review_periods(quads), review_periods(inj))

    def test_content_untouched(self):
        quads = self._quads()
        inj = inject_shuffle(quads, random.Random(3))
        for b, a in zip(quads, inj):
            self.assertEqual(b["review_uid"], a["review_uid"])
            self.assertEqual(b["sentiment"], a["sentiment"])
            self.assertEqual(b["stratum"], a["stratum"])

    def test_deterministic_for_same_seed(self):
        q = self._quads()
        a = [x["period"] for x in inject_shuffle(q, random.Random(7))]
        b = [x["period"] for x in inject_shuffle(q, random.Random(7))]
        self.assertEqual(a, b)


class TestQuadUidParity(unittest.TestCase):
    """Phần C — khoá join giữa make_audit_samples và module_c phải TRÙNG.

    Hai bên dựng khoá bằng cùng hàm U.quad_uid; nếu lệch thì fit_bridge khớp 0 cặp
    -> khối Spearman bị bỏ, go/no-go không được chấm.
    """

    QUAD = {"review_uid": "H1_20230415_00000042", "phi": "NEG",
            "taxonomy_code": "FAC_ROOM", "opinion_term": "ồn quá",
            "conf_seq": 0.9, "p_posterior": 0.88, "n_words": 30,
            "provenance_flip": False, "sentiment": "negative"}

    def test_audit_sample_key_matches_bridge_key(self):
        from prism import make_audit_samples as MAS
        from prism import module_c_reliability as MC
        # đường của make_audit_samples: ghi quad_uid vào file audit
        written = U.quad_uid(self.QUAD)
        # đường của module_c: dựng lại từ pool quad
        rebuilt = U.quad_uid(dict(self.QUAD))
        self.assertEqual(written, rebuilt)
        # cả hai module phải dùng ĐÚNG hàm đó, không có bản sao nội bộ
        self.assertIs(MAS.U.quad_uid, U.quad_uid)
        self.assertIs(MC.U.quad_uid, U.quad_uid)

    def test_uid_ignores_fields_not_in_key(self):
        # thêm trường không thuộc khoá không được đổi uid (audit gán 'correct' sau)
        q2 = dict(self.QUAD, correct=1, w=0.5, v_image=0.7)
        self.assertEqual(U.quad_uid(self.QUAD), U.quad_uid(q2))


class TestBridgeFeatureParity(unittest.TestCase):
    """#4 — fit_bridge và apply_weights phải dùng CÙNG vector đặc trưng."""

    Q = {"conf_seq": 0.81, "p_posterior": 0.77, "phi": "POS",
         "n_words": 42, "provenance_flip": True}

    def test_single_definition(self):
        from prism.module_c_reliability import (BRIDGE_FEATURE_NAMES,
                                                bridge_features)
        x = bridge_features(self.Q)
        self.assertEqual(len(x), len(BRIDGE_FEATURE_NAMES))
        self.assertEqual(BRIDGE_FEATURE_NAMES,
                         ["conf_seq", "p_posterior", "phi_pos", "log_len", "prov_flip"])
        self.assertEqual(x[0], 0.81)
        self.assertEqual(x[2], 1.0)          # phi POS
        self.assertEqual(x[4], 1.0)          # provenance_flip

    def test_apply_weights_uses_the_same_helper(self):
        # apply_weights KHÔNG được dựng list literal riêng: bắt bằng cách đếm
        # số lần tên hàm xuất hiện trong source của module.
        import prism.module_c_reliability as MC
        src = Path(MC.__file__).read_text(encoding="utf-8")
        self.assertGreaterEqual(src.count("bridge_features("), 3,
                                "apply_weights phải gọi bridge_features, "
                                "không dựng vector riêng")

    def test_propensity_keeps_real_zero_score(self):
        from prism.module_c_reliability import propensity_features
        self.assertEqual(propensity_features(dict(self.Q, score=0.0))[1], 0.0)
        self.assertEqual(propensity_features(dict(self.Q, score=None))[1],
                         C.DEFAULT_SCORE)


class TestFindCkptPrefersLatestRound(unittest.TestCase):
    """#9 — orchestrator phải chọn selftrain_round LỚN NHẤT, không phải round1."""

    @staticmethod
    def _load_orchestrator():
        path = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_pipeline.py"
        spec = importlib.util.spec_from_file_location("kaggle_pipeline", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def setUp(self):
        import tempfile
        self.kp = self._load_orchestrator()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in ("selftrain_round1", "selftrain_round2", "seed_extractor"):
            d = self.root / "models" / name
            d.mkdir(parents=True)
            (d / "model.safetensors").write_bytes(b"x")
            (d / "config.json").write_text("{}")

    def tearDown(self):
        self.tmp.cleanup()

    def test_picks_highest_round(self):
        got = self.kp.find_ckpt([str(self.root)])
        self.assertEqual(os.path.basename(got), "selftrain_round2",
                         "phải chọn round2, round1 là checkpoint vòng trước")

    def test_history_final_ckpt_wins(self):
        (self.root / "selftrain_history.json").write_text(json.dumps(
            {"final_ckpt": "/somewhere/models/selftrain_round1",
             "final_dev_f1": 0.61}), encoding="utf-8")
        got = self.kp.find_ckpt([str(self.root)])
        self.assertEqual(os.path.basename(got), "selftrain_round1",
                         "selftrain_history.json phải được ưu tiên")

    def test_no_source_patching_functions_remain(self):
        # #10 — orchestrator không được quyền ghi vào src/
        self.assertFalse(hasattr(self.kp, "patch_infer_bug"))
        self.assertFalse(hasattr(self.kp, "patch_selftrain_batch"))

    def test_sentinel_rejects_empty_file(self):
        f = self.root / "pool_quads.T-unbiased.jsonl.gz"
        f.write_bytes(b"\x1f\x8b" + b"\x00" * 51)     # gzip rỗng ~53 byte
        self.assertFalse(self.kp.sentinel_ok(f),
                         "file .gz rỗng không được coi là output hợp lệ")
        f.write_bytes(b"\x00" * 5000)
        self.assertTrue(self.kp.sentinel_ok(f))


class TestCohortDefsAreUsed(unittest.TestCase):
    """#19 — ngưỡng cohort chỉ được định nghĩa ở config.COHORT_DEFS."""

    def test_every_cohort_has_full_spec(self):
        for name, spec in C.COHORT_DEFS.items():
            for key in ("needs_gold", "min_pool", "gold_split"):
                self.assertIn(key, spec, f"{name} thiếu khoá {key}")

    def test_module_a_reads_config_not_literals(self):
        import prism.module_a_store as A
        src = Path(A.__file__).read_text(encoding="utf-8")
        self.assertIn("C.COHORT_DEFS", src)
        self.assertNotIn(">= 1000", src)
        self.assertNotIn(">= 300", src)


class TestImageValidation(unittest.TestCase):
    """#22 — ảnh tải về phải được kiểm nội dung, không chỉ kích thước > 0."""

    @staticmethod
    def _load():
        path = Path(__file__).resolve().parents[1] / "scripts" / "download_pool_photos.py"
        spec = importlib.util.spec_from_file_location("download_pool_photos", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_rejects_html_error_page(self):
        m = self._load()
        self.assertFalse(m.is_image_bytes(b"<!DOCTYPE html><html><head><title>404"))
        self.assertFalse(m.is_image_bytes(b'{"error": "not found"}'))

    def test_accepts_real_image_headers(self):
        m = self._load()
        self.assertTrue(m.is_image_bytes(bytes.fromhex("ffd8ffe0") + b"0" * 20))
        self.assertTrue(m.is_image_bytes(bytes.fromhex("89504e470d0a1a0a") + b"0" * 20))
        self.assertTrue(m.is_image_bytes(b"RIFF\x00\x00\x00\x00WEBPVP8 "))


if __name__ == "__main__":
    unittest.main()


class TestSecondPassFixes(unittest.TestCase):
    """Vòng review thứ 2 — lỗi phát hiện trong chính các bản sửa của vòng 1."""

    @staticmethod
    def _kp():
        path = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_pipeline.py"
        spec = importlib.util.spec_from_file_location("kaggle_pipeline", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_find_ckpt_prefers_working_over_stale_input(self):
        """find_ckpt khớp final_ckpt theo BASENAME sẽ lấy bản CŨ trong /kaggle/input
        khi /kaggle/working có bản mới CÙNG TÊN — tình huống rất thường trên Kaggle."""
        import tempfile
        kp = self._kp()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for base in ("input/old-dataset/models", "working/models"):
                d = root / base / "selftrain_round2"
                d.mkdir(parents=True)
                (d / "model.safetensors").write_bytes(b"x")
                (d / "config.json").write_text("{}")
            want = root / "working" / "models" / "selftrain_round2"
            (root / "working" / "selftrain_history.json").write_text(
                json.dumps({"final_ckpt": str(want), "final_dev_f1": 0.63}),
                encoding="utf-8")
            got = kp.find_ckpt([str(root / "input"), str(root / "working")])
            self.assertIn("working", got.replace("\\", "/"),
                          "phải lấy checkpoint trong /working, không phải bản cũ /input")

    def test_find_ckpt_ranking_also_prefers_working(self):
        """Đường fallback (không có history) cũng phải ưu tiên /working."""
        import tempfile
        kp = self._kp()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for base in ("input/old/models", "working/models"):
                d = root / base / "selftrain_round1"
                d.mkdir(parents=True)
                (d / "model.safetensors").write_bytes(b"x")
                (d / "config.json").write_text("{}")
            got = kp.find_ckpt([str(root / "input"), str(root / "working")])
            self.assertIn("working", got.replace("\\", "/"))

    def test_filename_convention_matches_config(self):
        """kaggle_pipeline KHÔNG import được prism.config (nó chạy trước khi stage
        source), nên nó có bản sao quy ước tên file. Test này là thứ duy nhất giữ
        hai bên không lệch — lệch tên = FileNotFoundError dù file có đó."""
        kp = self._kp()
        for cohort in ("T-unbiased", "corpus", "B-anchor"):
            self.assertEqual(kp.POOL_NAME(cohort), C.pool_quads_name(cohort))
            self.assertEqual(kp.VIMG_NAME(cohort), C.vimg_quads_name(cohort))
        self.assertEqual(kp.AUDIT_NAME, C.HUMAN_AUDIT_NAME)
        self.assertEqual(kp.AUDIT_NAME, C.AUDIT_SAMPLE_NAME)

    def test_probe_reads_utf8(self):
        """probe dùng io.open(..., encoding='utf-8'): open() trần crash
        UnicodeDecodeError trên locale cp1252 của Windows."""
        src = (Path(__file__).resolve().parents[1]
               / "scripts" / "probe_complaint_composition.py").read_text(encoding="utf-8")
        self.assertNotIn("open(QUADS)", src)
        self.assertNotIn("open(POOL)", src)
        self.assertIn("encoding='utf-8'", src)

    def test_probe_uses_config_not_copies(self):
        """probe không được khai báo lại WEST/ASIA/MIN_STRATUM."""
        src = (Path(__file__).resolve().parents[1]
               / "scripts" / "probe_complaint_composition.py").read_text(encoding="utf-8")
        self.assertNotIn("WEST = {", src)
        self.assertNotIn("ASIA = {", src)
        self.assertIn("C.MIN_STRATUM_N", src)

    def test_injection_reuses_one_shuffle_file(self):
        """E4 lặp --repeats lần; mỗi bản quad là hàng trăm MB với corpus nên phải
        DÙNG LẠI một file, không ghi ra mỗi lần lặp một file."""
        import prism.eval_injection as EI
        src = Path(EI.__file__).read_text(encoding="utf-8")
        self.assertNotIn('quads_inj_shuffle_r{r_i}', src)
        self.assertIn('"quads_inj_shuffle.jsonl.gz"', src)

    def test_verifier_cache_is_bounded(self):
        """Cache embedding ảnh phải có chặn trên (bản gốc giữ toàn bộ)."""
        import prism.module_c_reliability as MC
        src = Path(MC.__file__).read_text(encoding="utf-8")
        self.assertIn("CACHE_MAX", src)
        self.assertIn("popitem(last=False)", src)
