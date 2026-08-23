"""Unit test cho máy thống kê của Module D — dữ liệu tổng hợp nhỏ, không cần torch."""
import random
import unittest

from prism.module_d_drift import (best_split, channel_stats, deseason,
                                  ols_trend, shares_by_period,
                                  valence_by_period)


def _cell(entries):
    """entries: {(period, stratum): {aspect: w}}"""
    return dict(entries)


class TestShares(unittest.TestCase):
    def test_single_stratum_adjusted_equals_raw(self):
        cell = _cell({("2022-01", ("VN",)): {"x": 30.0, "y": 10.0},
                      ("2022-02", ("VN",)): {"x": 20.0, "y": 20.0}})
        ref = {("VN",): 1.0}
        adj = shares_by_period(cell, ref, ["x", "y"], min_n=0, adjust=True)
        raw = shares_by_period(cell, ref, ["x", "y"], min_n=0, adjust=False)
        for p in adj:
            for a in ("x", "y"):
                self.assertAlmostEqual(adj[p][a], raw[p][a], places=9)

    def test_standardization_removes_composition_shift(self):
        # VN nói về x 80/20, WEST nói 20/80 — tỷ trọng VN đổi 75%->25% giữa 2 kỳ.
        # Share thô đổi mạnh; share chuẩn hoá (ref 50/50) phải giống nhau 2 kỳ.
        cell = _cell({
            ("2022-01", ("VN",)):   {"x": 240.0, "y": 60.0},    # 300 (75%)
            ("2022-01", ("WEST",)): {"x": 20.0,  "y": 80.0},    # 100 (25%)
            ("2022-02", ("VN",)):   {"x": 80.0,  "y": 20.0},    # 100 (25%)
            ("2022-02", ("WEST",)): {"x": 60.0,  "y": 240.0},   # 300 (75%)
        })
        ref = {("VN",): 0.5, ("WEST",): 0.5}
        raw = shares_by_period(cell, ref, ["x", "y"], min_n=0, adjust=False)
        adj = shares_by_period(cell, ref, ["x", "y"], min_n=0, adjust=True)
        self.assertGreater(abs(raw["2022-01"]["x"] - raw["2022-02"]["x"]), 0.2)
        self.assertAlmostEqual(adj["2022-01"]["x"], adj["2022-02"]["x"], places=9)

    def test_missing_stratum_contributes_neutral_pooled_share(self):
        # Kỳ chỉ có VN; WEST vắng — với shrinkage, WEST đóng góp đúng share gộp
        # của kỳ => adjusted == pooled (không sinh bước nhảy như khi loại ô).
        cell = _cell({("2022-01", ("VN",)): {"x": 30.0, "y": 10.0}})
        ref = {("VN",): 0.5, ("WEST",): 0.5}
        adj = shares_by_period(cell, ref, ["x", "y"], min_n=10, adjust=True)
        self.assertAlmostEqual(adj["2022-01"]["x"], 0.75, places=9)

    def test_large_prior_shrinks_to_pooled(self):
        cell = _cell({("2022-01", ("VN",)):   {"x": 9.0, "y": 1.0},
                      ("2022-01", ("WEST",)): {"x": 1.0, "y": 9.0}})
        ref = {("VN",): 0.5, ("WEST",): 0.5}
        adj = shares_by_period(cell, ref, ["x", "y"], min_n=10**6, adjust=True)
        self.assertAlmostEqual(adj["2022-01"]["x"], 0.5, places=6)


class TestValence(unittest.TestCase):
    VCELL = {
        ("2022-01", ("VN",)):   {"x": {"negative": 8.0, "positive": 2.0}},
        ("2022-01", ("WEST",)): {"x": {"negative": 2.0, "positive": 8.0}},
    }

    def test_raw_is_pooled_rate(self):
        v = valence_by_period(self.VCELL, {("VN",): 0.5, ("WEST",): 0.5},
                              ["x"], ["2022-01"], min_w=0, adjust=False)
        self.assertAlmostEqual(v["2022-01"]["x"], 0.5, places=9)

    def test_adjusted_weights_by_reference(self):
        # ref 90% WEST -> ν chuẩn hoá phải nghiêng về tỷ lệ của WEST (0,2)
        v = valence_by_period(self.VCELL, {("VN",): 0.1, ("WEST",): 0.9},
                              ["x"], ["2022-01"], min_w=0, adjust=True)
        self.assertAlmostEqual(v["2022-01"]["x"], 0.1 * 0.8 + 0.9 * 0.2, places=9)

    def test_absent_aspect_is_none(self):
        v = valence_by_period(self.VCELL, {("VN",): 1.0}, ["z"],
                              ["2022-01"], min_w=0, adjust=True)
        self.assertIsNone(v["2022-01"]["z"])


class TestTrendMachinery(unittest.TestCase):
    def test_deseason_removes_pure_monthly_pattern(self):
        periods = [f"{y}-{m:02d}" for y in (2022, 2023) for m in range(1, 13)]
        series = [float(int(p[5:7]) % 3) for p in periods]   # chỉ có mùa vụ
        de = deseason(series, periods)
        self.assertLess(max(de) - min(de), 1e-9)

    def test_ols_trend_recovers_slope(self):
        series = [0.01 * i for i in range(24)]
        slope, t = ols_trend(series)
        self.assertAlmostEqual(slope, 0.12, places=9)        # 0,01/kỳ × 12
        self.assertGreater(abs(t), 100)

    def test_best_split_finds_step(self):
        # cần nhiễu nhỏ: hai nửa phương sai 0 làm welch_t=0 (guard den==0)
        rng = random.Random(7)
        series = ([0.0 + 0.01 * rng.random() for _ in range(10)]
                  + [1.0 + 0.01 * rng.random() for _ in range(10)])
        i, t = best_split(series)
        self.assertEqual(i, 10)
        self.assertGreater(t, 10)

    def test_channel_stats_short_series_returns_none(self):
        rng = random.Random(0)
        periods = [f"2022-{m:02d}" for m in range(1, 9)]
        self.assertIsNone(channel_stats([0.1] * 8, periods, 10, rng))

    def test_channel_stats_flat_series_not_significant(self):
        rng = random.Random(0)
        periods = [f"{y}-{m:02d}" for y in (2022, 2023, 2024) for m in range(1, 13)]
        series = [0.5 + 0.001 * rng.random() for _ in periods]
        st = channel_stats(series, periods, 200, rng)
        self.assertGreater(st["p_perm"], 0.05)


if __name__ == "__main__":
    unittest.main()
