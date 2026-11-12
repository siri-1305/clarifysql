"""Quick health check. Run:  python selfcheck.py            (no API key needed)
                              python selfcheck.py --online   (also tests the Groq API)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
failures = 0


def report(ok, msg):
    global failures
    print(("  PASS  " if ok else "  FAIL  ") + msg)
    failures += 0 if ok else 1


print("1. Packages")
for mod in ("streamlit", "groq", "sqlglot", "numpy", "pandas", "dotenv"):
    try:
        m = __import__(mod, fromlist=["x"])
        report(True, f"{mod} {getattr(m, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        report(False, f"{mod} missing -> run: pip install -r requirements.txt  ({type(e).__name__})")

from clarifysql.config import Settings                      # noqa: E402
from clarifysql.db import Database                          # noqa: E402
from clarifysql.metrics import results_match                # noqa: E402
from clarifysql.validator import HAS_SQLGLOT, validate      # noqa: E402

print("2. Database")
s = Settings()
try:
    db = Database(s.db_path)
    report(True, "tables: " + ", ".join(f"{t.name}({t.row_count})" for t in db.tables.values()))
except Exception as e:  # noqa: BLE001
    report(False, f"cannot open database: {e}")
    sys.exit(1)

print("3. Validator accepts every reference SQL in questions.json (no false alarms)")
qs = json.loads(s.questions_path.read_text(encoding="utf-8"))
sqls = []
for q in qs:
    if q.get("expected_sql"):
        sqls.append((q["id"], q["expected_sql"]))
    for it in q.get("interpretations", []):
        sqls.append((q["id"], it["expected_sql"]))
bad = []
for qid, sql in sqls:
    rep = validate(sql, db)
    if not rep.ok:
        bad.append((qid, [i.message for i in rep.errors]))
report(not bad, f"{len(sqls) - len(bad)}/{len(sqls)} reference queries pass" + (f"; problems: {bad[:3]}" if bad else ""))
report(HAS_SQLGLOT, "sqlglot loaded (full schema + filter-value checks active)")

print("4. Validator catches mistakes")
tests = {
    "wrong table": ("SELECT * FROM ordrs;", "unknown_table"),
    "wrong column": ("SELECT o.total FROM orders o;", "unknown_column"),
    "wrong filter value": ("SELECT * FROM customers WHERE city = 'Bangalore';", "bad_value"),
    "double-quoted text": ('SELECT * FROM products p JOIN categories c ON c.category_id=p.category_id WHERE c.category_name = "Electronics";', "unknown_column"),
    "not a SELECT": ("DROP TABLE orders;", "not_select"),
}
for name, (sql, code) in tests.items():
    rep = validate(sql, db)
    got = {i.code for i in rep.issues}
    if HAS_SQLGLOT:
        report(code in got or not rep.ok, f"{name}: flagged ({', '.join(sorted(got)) or 'executed'})")
    else:
        report(not rep.ok or code == "bad_value", f"{name}: (reduced mode) {'flagged' if not rep.ok else 'not caught without sqlglot'}")

print("5. Offline pipeline tests (fake LLM)")
import unittest                                             # noqa: E402
suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
r = unittest.TextTestRunner(verbosity=0, stream=open("/dev/null" if sys.platform != "win32" else "NUL", "w")).run(suite)
report(r.wasSuccessful(), f"{r.testsRun} tests, {len(r.failures)} failures, {len(r.errors)} errors")

if "--online" in sys.argv:
    print("6. Groq API")
    from clarifysql.llm import GroqLLM, LLMError          # noqa: E402
    try:
        llm = GroqLLM(s)
        out = llm.generate("Reply with the single word: ready", sample=999)
        report(True, f"chat model '{llm.model}' answered: {out[:40]!r}")
        report(True, "Groq has no embedding models; schema retrieval uses the local embedder")
    except LLMError as e:
        report(False, str(e))

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{failures} CHECK(S) FAILED - see FAIL lines above"))
sys.exit(1 if failures else 0)
