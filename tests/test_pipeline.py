"""Offline tests: run with  python -m unittest discover -s tests -v   (no API key needed)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clarifysql.config import PipelineOptions                    # noqa: E402
from clarifysql.db import Database                                # noqa: E402
from clarifysql.metrics import auroc, results_match               # noqa: E402
from clarifysql.pipeline import ClarifySQL                        # noqa: E402
from clarifysql.retrieval import SchemaIndex                      # noqa: E402
from clarifysql.trust import TrustSignals, compute_trust          # noqa: E402
from tests.fake_llm import FakeLLM, js                            # noqa: E402

DB = Database(ROOT / "data" / "ecommerce.db")
INDEX = SchemaIndex(DB, None, ROOT / ".cache")


def make(handler, **opts):
    return ClarifySQL(DB, FakeLLM(handler), INDEX, PipelineOptions(**opts))


def amb(score=0.0, answerable=True, **kw):
    return js(answerable=answerable, ambiguity=score, reason="r", clarifying_question=kw.get("cq", ""),
              options=kw.get("options", []), unanswerable_reason=kw.get("why", ""))


class PipelineTests(unittest.TestCase):
    def test_clear_question_answers_directly_with_high_trust(self):
        def h(prompt, system, json_mode, sample):
            if "query-understanding" in (system or ""):
                return amb(0.05)
            if "explain database queries" in (system or "").lower() or "shop managers" in (system or ""):
                return js(summary="I counted all customers.", assumptions=[])
            return "SELECT COUNT(*) FROM customers;"
        res = make(h).ask("How many customers are there?")
        self.assertEqual(res.status, "answer")
        self.assertEqual(res.rows, [(200,)])
        self.assertEqual(res.repairs, 0)
        self.assertEqual(res.agreement, 1.0)
        self.assertEqual(res.trust.level, "High")
        self.assertIn("customers", res.explanation.tables)

    def test_vague_question_asks_then_answers(self):
        def h(prompt, system, json_mode, sample):
            if "query-understanding" in (system or ""):
                return amb(0.9, cq="Best by sales, rating or profit?", options=["sales", "rating", "profit"])
            if json_mode:
                return js(summary="Top rated products.", assumptions=["Used average rating"])
            return "SELECT p.product_name, AVG(r.rating) AS score FROM products p JOIN reviews r ON r.product_id=p.product_id GROUP BY p.product_name ORDER BY score DESC LIMIT 5;"
        app = make(h)
        r1 = app.ask("Show me the best products")
        self.assertEqual(r1.status, "clarify")
        self.assertEqual(r1.options, ["sales", "rating", "profit"])
        r2 = app.ask("Show me the best products", clarification_answer="rating", clarification_question=r1.clarification_question)
        self.assertEqual(r2.status, "answer")
        self.assertEqual(len(r2.rows), 5)

    def test_unanswerable_from_ambiguity_module(self):
        h = lambda p, s, j, n: amb(0.0, answerable=False, why="No phone column.")
        res = make(h).ask("Show customer phone numbers.")
        self.assertEqual(res.status, "unanswerable")
        self.assertIn("phone", res.message.lower())

    def test_unanswerable_from_generator(self):
        def h(prompt, system, json_mode, sample):
            return amb(0.0) if "query-understanding" in (system or "") else "UNANSWERABLE: no salary data"
        self.assertEqual(make(h).ask("Show employee salaries.").status, "unanswerable")

    def test_repair_loop_fixes_a_broken_query(self):
        def h(prompt, system, json_mode, sample):
            if "query-understanding" in (system or ""):
                return amb(0.0)
            if json_mode:
                return js(summary="Counted cancelled orders.", assumptions=[])
            if "PROBLEMS FOUND" in prompt:
                return "SELECT COUNT(*) FROM orders WHERE status = 'cancelled';"
            return "SELECT COUNT(*) FROM ordrs WHERE status = 'cancelled';"
        res = make(h, agreement_samples=0).ask("How many orders were cancelled?")
        self.assertEqual(res.status, "answer")
        self.assertEqual(res.repairs, 1)
        self.assertTrue(res.validated)
        self.assertEqual(len(res.attempts), 2)
        self.assertIn(res.trust.level, ("Medium", "High"))

    def test_repair_disabled_keeps_error(self):
        def h(prompt, system, json_mode, sample):
            return amb(0.0) if "query-understanding" in (system or "") else "SELECT COUNT(*) FROM ordrs;"
        res = make(h, use_repair=False, agreement_samples=0).ask("How many orders?")
        self.assertEqual(res.status, "error")
        self.assertEqual(res.repairs, 0)

    def test_disagreeing_samples_lower_trust(self):
        def h(prompt, system, json_mode, sample):
            if "query-understanding" in (system or ""):
                return amb(0.0)
            if json_mode:
                return js(summary="x", assumptions=[])
            return "SELECT COUNT(*) FROM orders;" if sample == 0 else f"SELECT COUNT(*) FROM orders WHERE order_id < {sample * 100};"
        res = make(h).ask("How many orders are there?")
        self.assertEqual(res.status, "answer")
        self.assertAlmostEqual(res.agreement, 1 / 3, places=2)
        self.assertNotEqual(res.trust.level, "High")

    def test_clarification_ablation_never_asks(self):
        def h(prompt, system, json_mode, sample):
            if json_mode:
                return js(summary="x", assumptions=[])
            return "SELECT COUNT(*) FROM customers;"
        res = make(h, use_clarification=False, agreement_samples=0).ask("Show me the best products")
        self.assertEqual(res.status, "answer")

    def test_baseline(self):
        h = lambda p, s, j, n: "```sql\nSELECT COUNT(*) FROM customers;\n```"
        res = make(h).baseline("How many customers are there?")
        self.assertEqual((res.status, res.rows), ("answer", [(200,)]))


class UnitTests(unittest.TestCase):
    def test_results_match(self):
        self.assertTrue(results_match([(1, "a")], [("a", 1)]))
        self.assertTrue(results_match([("a", 5.0, 9)], [("a", 5)]))
        self.assertFalse(results_match([(1,)], [(2,)]))
        self.assertTrue(results_match([], []))

    def test_auroc(self):
        self.assertEqual(auroc([0.9, 0.8, 0.2], [True, True, False]), 1.0)
        self.assertIsNone(auroc([0.5], [True]))

    def test_trust_levels(self):
        good = compute_trust(TrustSignals(repairs=0, agreement=1.0, n_candidates=3, retrieval_confidence=0.9))
        bad = compute_trust(TrustSignals(repairs=2, agreement=0.33, n_candidates=3, suspicious="empty", retrieval_confidence=0.3))
        self.assertEqual(good.level, "High")
        self.assertEqual(bad.level, "Low")
        self.assertEqual(compute_trust(TrustSignals(validated=False)).level, "Low")

    def test_retrieval_returns_tables(self):
        r = INDEX.retrieve("How many customers are from Hyderabad?")
        self.assertIn("customers", r.tables)

    def test_database_is_read_only(self):
        with self.assertRaises(Exception):
            DB.execute("DELETE FROM orders")


if __name__ == "__main__":
    unittest.main()
