"""Round-trip linearize/parse của Module B — không cần dữ liệu thật."""
import unittest

from prism.module_b_data import linearize, parse_linearized


class TestRoundTrip(unittest.TestCase):
    def test_explicit_quad(self):
        quads = [{"aspect_term": "phòng", "taxonomy_code": "FAC_ROOM",
                  "opinion_term": "sạch sẽ", "sentiment": "positive"}]
        out = parse_linearized(linearize(quads))
        self.assertEqual(len(out), 1)
        q = out[0]
        self.assertEqual(q["aspect_term"], "phòng")
        self.assertEqual(q["taxonomy_code"], "FAC_ROOM")
        self.assertEqual(q["aspect_category"], "FACILITY")
        self.assertEqual(q["opinion_term"], "sạch sẽ")
        self.assertEqual(q["sentiment"], "positive")
        self.assertFalse(q["aspect_implicit"])

    def test_implicit_aspect_and_opinion(self):
        quads = [{"aspect_term": None, "taxonomy_code": "AM_WIFI",
                  "opinion_term": None, "sentiment": "negative"}]
        q = parse_linearized(linearize(quads))[0]
        self.assertIsNone(q["aspect_term"])
        self.assertIsNone(q["opinion_term"])
        self.assertTrue(q["aspect_implicit"])
        self.assertTrue(q["opinion_implicit"])

    def test_multi_quad_order_preserved(self):
        quads = [
            {"aspect_term": "staff", "taxonomy_code": "SER_ATTITUDE",
             "opinion_term": "friendly", "sentiment": "positive"},
            {"aspect_term": "pool", "taxonomy_code": "AM_POOL",
             "opinion_term": "dirty", "sentiment": "negative"},
        ]
        out = parse_linearized(linearize(quads))
        self.assertEqual([q["taxonomy_code"] for q in out],
                         ["SER_ATTITUDE", "AM_POOL"])

    def test_invalid_code_and_sentiment_dropped(self):
        s = ("<quad> x | XX_BAD | y | positive </quad> "
             "<quad> x | FAC_ROOM | y | tệ_lắm </quad> "
             "<quad> x | FAC_ROOM | y | negative </quad>")
        out = parse_linearized(s)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["sentiment"], "negative")

    def test_malformed_returns_empty(self):
        self.assertEqual(parse_linearized(""), [])
        self.assertEqual(parse_linearized("không có quad nào ở đây"), [])
        self.assertEqual(parse_linearized("<quad> thiếu | trường </quad>"), [])


if __name__ == "__main__":
    unittest.main()
