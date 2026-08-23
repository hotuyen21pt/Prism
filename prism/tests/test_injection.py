"""Bất biến của các hàm injection (E3/E4) — dữ liệu tổng hợp nhỏ."""
import collections
import random
import unittest

from prism.eval_injection import (inject_composition, inject_shuffle,
                                  inject_valence)


def make_quads(n=400, seed=0):
    rng = random.Random(seed)
    periods = [f"2022-{m:02d}" for m in range(1, 13)]
    out = []
    for i in range(n):
        out.append({
            "review_uid": f"R{i}", "hotel_id": f"H{i % 20}",
            "period": rng.choice(periods),
            "stratum": [rng.choice(["VN", "WEST"]), "Cặp đôi"],
            "phi": rng.choice(["POS", "NEG"]),
            "taxonomy_code": rng.choice(["AM_FOOD", "FAC_ROOM"]),
            "aspect_category": "AMENITY",
            "sentiment": rng.choice(["positive", "negative"]),
            "conf_seq": 0.9,
        })
    return out


class TestValenceInjection(unittest.TestCase):
    def test_only_sentiment_changes_phi_untouched(self):
        quads = make_quads()
        inj = inject_valence(quads, "AM_FOOD", 1.0, "2022-06", random.Random(1))
        self.assertEqual(len(inj), len(quads))
        for before, after in zip(quads, inj):
            self.assertEqual(before["phi"], after["phi"])
            if (before["taxonomy_code"] == "AM_FOOD"
                    and before["period"] >= "2022-06"
                    and before["sentiment"] == "positive"):
                self.assertEqual(after["sentiment"], "negative")   # δ=100%
            else:
                self.assertEqual(after["sentiment"], before["sentiment"])

    def test_nothing_changes_before_t0_or_other_aspect(self):
        quads = make_quads()
        inj = inject_valence(quads, "AM_FOOD", 1.0, "2022-06", random.Random(1))
        for b, a in zip(quads, inj):
            if b["taxonomy_code"] != "AM_FOOD" or b["period"] < "2022-06":
                self.assertEqual(a["sentiment"], b["sentiment"])


class TestShuffleInjection(unittest.TestCase):
    def test_preserves_period_multiset_and_content(self):
        quads = make_quads()
        inj = inject_shuffle(quads, random.Random(2))
        self.assertEqual(collections.Counter(q["period"] for q in quads),
                         collections.Counter(q["period"] for q in inj))
        for b, a in zip(quads, inj):
            self.assertEqual(b["sentiment"], a["sentiment"])
            self.assertEqual(b["stratum"], a["stratum"])


class TestCompositionInjection(unittest.TestCase):
    def test_drops_rows_without_editing_content(self):
        quads = make_quads(n=2000)
        inj = inject_composition(quads, random.Random(3))
        self.assertLess(len(inj), len(quads))
        originals = {id(q) for q in quads}
        for q in inj:                       # chỉ resample, không sửa nội dung
            self.assertIn(id(q), originals)

    def test_vn_share_declines_over_time(self):
        quads = make_quads(n=20000, seed=4)
        inj = inject_composition(quads, random.Random(5))
        def vn_share(rows, per):
            sub = [q for q in rows if q["period"] == per]
            return sum(q["stratum"][0] == "VN" for q in sub) / max(len(sub), 1)
        self.assertGreater(vn_share(inj, "2022-01"), vn_share(inj, "2022-12") + 0.3)


if __name__ == "__main__":
    unittest.main()
