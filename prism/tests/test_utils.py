"""Unit test cho prism.utils — chạy: PYTHONPATH=src python3 -m unittest discover tests"""
import math
import unittest

from prism import utils as U


class TestDateParsing(unittest.TestCase):
    def test_vietnamese_format(self):
        self.assertEqual(
            U.parse_review_date("Ngày đánh giá: ngày 5 tháng 9 năm 2022"),
            ("2022-09-05", "2022-09"))

    def test_iso_and_slash(self):
        self.assertEqual(U.parse_review_date("2023-01-15"), ("2023-01-15", "2023-01"))
        self.assertEqual(U.parse_review_date("15/01/2023"), ("2023-01-15", "2023-01"))

    def test_invalid(self):
        self.assertIsNone(U.parse_review_date(None))
        self.assertIsNone(U.parse_review_date(""))
        self.assertIsNone(U.parse_review_date("ngày 40 tháng 1 năm 2023"))
        self.assertIsNone(U.parse_review_date("ngày 1 tháng 13 năm 2023"))


class TestScoreParsing(unittest.TestCase):
    def test_comma_decimal(self):
        self.assertEqual(U.parse_score("Đạt điểm 9,0"), 9.0)
        self.assertEqual(U.parse_score("Đạt điểm 10"), 10.0)

    def test_passthrough_and_invalid(self):
        self.assertEqual(U.parse_score(8.5), 8.5)
        self.assertIsNone(U.parse_score(None))
        self.assertIsNone(U.parse_score("không có điểm"))


class TestStrata(unittest.TestCase):
    def test_country_bloc(self):
        self.assertEqual(U.country_bloc("Việt Nam"), "VN")
        self.assertEqual(U.country_bloc("Pháp"), "WEST")
        self.assertEqual(U.country_bloc("Hàn Quốc"), "ASIA")
        self.assertEqual(U.country_bloc("Brazil"), "OTH")
        self.assertEqual(U.country_bloc(None), "OTH")

    def test_length_bin(self):
        self.assertEqual(U.length_bin(0), "L0")
        self.assertEqual(U.length_bin(15), "L1")
        self.assertEqual(U.length_bin(45), "L2")
        self.assertEqual(U.length_bin(10_000), "L2")

    def test_periods_in_window_36_months(self):
        ps = U.periods_in_window("2022-03", "2025-02")
        self.assertEqual(len(ps), 36)
        self.assertEqual(ps[0], "2022-03")
        self.assertEqual(ps[-1], "2025-02")


class TestStatistics(unittest.TestCase):
    def test_bh_known_example(self):
        q = U.benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.9])
        for v in q[:4]:
            self.assertAlmostEqual(v, 0.05, places=10)
        self.assertAlmostEqual(q[4], 0.9, places=10)

    def test_bh_empty_and_order(self):
        self.assertEqual(U.benjamini_hochberg([]), [])
        q = U.benjamini_hochberg([0.9, 0.01])       # giữ nguyên thứ tự input
        self.assertGreater(q[0], q[1])

    def test_welch_t(self):
        self.assertEqual(U.welch_t([1.0], [2.0, 3.0]), 0.0)          # n<2
        self.assertEqual(U.welch_t([1, 1, 1], [1, 1, 1]), 0.0)       # var=0
        t = U.welch_t([0, 0.1, -0.1, 0.05], [5, 5.1, 4.9, 5.05])
        self.assertLess(t, -10)

    def test_clr_sums_to_zero(self):
        c = U.clr({"a": 0.5, "b": 0.3, "c": 0.2})
        self.assertAlmostEqual(sum(c.values()), 0.0, places=9)

    def test_clr_handles_zero_share(self):
        c = U.clr({"a": 1.0, "b": 0.0})
        self.assertTrue(all(math.isfinite(v) for v in c.values()))


class TestQuadUid(unittest.TestCase):
    Q = {"review_uid": "H1_20220901_00000001", "phi": "NEG",
         "taxonomy_code": "FAC_ROOM", "opinion_term": "ồn"}

    def test_stable(self):
        self.assertEqual(U.quad_uid(self.Q), U.quad_uid(dict(self.Q)))

    def test_sensitive_to_opinion(self):
        q2 = dict(self.Q, opinion_term="sạch")
        self.assertNotEqual(U.quad_uid(self.Q), U.quad_uid(q2))


if __name__ == "__main__":
    unittest.main()
