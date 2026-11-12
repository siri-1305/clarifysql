# ClarifySQL - clarification-aware, retrieval-augmented Text-to-SQL with a trust badge

Final-year major project, Dept. of CSE, RGUKT Srikakulam.

A chat app where a non-SQL user asks questions about an e-commerce database.
It **asks a follow-up only when the question is vague**, retrieves **only the relevant tables**, writes and **validates/repairs** the SQL,
then **explains in plain English** how it understood the question and shows a **High / Medium / Low trust badge**.

```
Question -> Ambiguity check -> Schema retrieval (RAG) -> SQL generation -> Validate & repair -> Explanation + Trust badge
              (ask only if       (embeddings +            (Gemini)        (sqlglot + SQLite,     (plain English,
               score >= 0.5)      vector search)                           max 2 repairs)         badge from 4 signals)
```

---

## STEP-BY-STEP: HOW TO RUN IT

### Step 1 - Install Python
You need **Python 3.10, 3.11 or 3.12**. Check in a terminal (Windows: Command Prompt or PowerShell):
```
python --version
```
If it says "not found" (or shows Python 2/3.8), install Python from python.org. On Windows tick **"Add Python to PATH"** in the installer.
(On Mac/Linux you may need to type `python3` instead of `python` everywhere below.)

### Step 2 - Open the project folder in a terminal
Unzip `ClarifySQL.zip`, then open a terminal *inside* the `ClarifySQL` folder (VS Code: File > Open Folder, then Terminal > New Terminal).
You should see `app.py` when you run `dir` (Windows) or `ls` (Mac/Linux).

### Step 3 - Create a virtual environment (keeps packages tidy)
Windows:
```
python -m venv venv
venv\Scripts\activate
```
Mac/Linux:
```
python3 -m venv venv
source venv/bin/activate
```
Your prompt now starts with `(venv)`. (If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` once, then activate again.)

### Step 4 - Install the packages
```
pip install -r requirements.txt
```
This installs streamlit, google-genai, sqlglot, numpy, pandas, python-dotenv. Needs internet, takes 1-3 minutes.

### Step 5 - Get a free Gemini API key and put it in `.env`
1. Open https://aistudio.google.com/apikey , sign in with a Google account, click **Create API key**, copy it.
2. Make your settings file by copying the example:
   - Windows: `copy .env.example .env`
   - Mac/Linux: `cp .env.example .env`
3. Open `.env` in a text editor and replace `PASTE_YOUR_KEY_HERE` with your key. No quotes, no spaces:
   ```
   GEMINI_API_KEY=AIza...yourkey...
   ```
4. Save. **Never share or upload this file** (it is already in `.gitignore`).

### Step 6 - Run the health check (do this before the app)
```
python selfcheck.py --online
```
It checks the packages, the database, that the validator accepts all 60 reference queries, that it catches mistakes, runs 14 offline tests,
and (with `--online`) calls Gemini once. You want to see **ALL CHECKS PASSED**.
If a line says FAIL, read the troubleshooting table at the bottom.

### Step 7 - Start the app
```
streamlit run app.py
```
Your browser opens at http://localhost:8501 (if not, open that address yourself). The first start takes a few seconds while the schema index is built.
Stop the app with `Ctrl+C` in the terminal.

### Step 8 - Try this demo (good for your viva)
| Type this | What should happen |
|---|---|
| `How many orders were cancelled?` | Answered directly (no follow-up) with a number, a plain-English explanation and a trust badge. |
| `Show me the best products` | It asks *"Best by what?"* with option buttons. Click **rating** -> top-rated products, explanation says what was assumed. |
| `Who are our top customers?` | Follow-up (total spend vs number of orders), then the answer. |
| `List customers from Bangalore` | Database stores `Bengaluru`. The validator flags the wrong value and the repair loop fixes it (see *Details > Repair history*). |
| `How many orders were completed?` | `completed` is stored as `delivered`; handled by the same repair loop. |
| `Show customer phone numbers` | Politely refused: there is no phone column. |

Click **Details** under any answer to see ambiguity score, retrieved tables, repairs, agreement and number of LLM calls.
Use the sidebar toggles to switch modules off live (e.g. turn off "Validate and repair" and ask the Bangalore question again).

### Step 9 - Run the evaluation for your report
```
python evaluate.py --limit 3
```
First a small trial (3 questions per type). If it works, run everything:
```
python evaluate.py
```
It compares: `baseline` (plain LLM-to-SQL), `full` (ClarifySQL), and three ablations `no_clarify`, `no_retrieval`, `no_repair`.
Results are saved in `results/`: `summary_<time>.md` (tables for your report), `summary_<time>.json`, and `records_<time>.json` (every question, SQL, result and badge).

What is measured:
- **Execution accuracy** (same result rows as the reference SQL) overall and for clear / tricky / vague questions.
- **Clarification recall** (vague questions where it asked) and **unnecessary clarification rate** (clear questions where it asked anyway).
- **Your key test:** accuracy of answers with a **High** badge vs **Medium** vs **Low**, plus the AUROC of the trust score.

Notes: a full run is roughly 1,000+ API calls. Answers are cached in `.cache/`, so you can stop with `Ctrl+C` and rerun without paying again.
If you hit rate limits (HTTP 429 messages), the code waits and retries automatically; for a slower, safer pace add `LLM_MIN_INTERVAL=6` to `.env`.

---

## What is inside

```
ClarifySQL/
  app.py                  Streamlit chat interface (frontend)
  evaluate.py             baseline + ablation evaluation, trust-badge analysis
  selfcheck.py            health check (run this first)
  requirements.txt  .env.example
  data/                   ecommerce.db, questions.json (62 labelled cases), make_dataset.py   <- your files
  clarifysql/             backend
    config.py             settings + pipeline switches (ablation flags)
    llm.py                Gemini client: retries, rate limiting, model fallback, disk cache, embeddings
    db.py                 read-only SQLite access, schema + value profiling
    ambiguity.py          ambiguity score = 0.8 x LLM judgement + 0.2 x vague-word cue; threshold 0.5
    retrieval.py          schema RAG: embed tables/columns, vector search, value grounding, FK bridging
    generator.py          SQL generation, repair prompt, baseline
    validator.py          sqlglot syntax + schema + filter-value checks, then real execution
    explain.py            plain-English explanation (tables, filters, assumptions)
    trust.py              High/Medium/Low badge from 4 signals
    pipeline.py           ties everything together
    metrics.py            result comparison and AUROC
  tests/                  offline tests with a fake LLM (no API key needed)
```

### How it maps to your abstract
| Abstract says | Where it is |
|---|---|
| Score ambiguity, ask only if high | `ambiguity.py`, threshold slider in the app |
| Retrieve relevant tables/columns (RAG) | `retrieval.py` (Gemini embeddings + numpy cosine search; local fallback) |
| Validate syntax, schema consistency, filter values | `validator.py` |
| Bounded repair loop returning diagnostics to the LLM | `pipeline.py` (max 2 repairs, diagnostics from `validator.py`) |
| Plain-language explanation | `explain.py` |
| Trust from repairs, agreement, suspicious results, retrieval confidence | `trust.py` (weights 0.25 / 0.35 / 0.20 / 0.20) |
| Baseline + ablations, execution accuracy, does the badge separate right from wrong | `evaluate.py` |

### Things worth knowing (and saying honestly in your report)
- The trust weights (0.35 agreement, 0.25 repairs, 0.20 result sanity, 0.20 retrieval) and the High/Medium cut-offs (0.80 / 0.55) are **design choices, not learned values**. `evaluate.py` measures whether they work; you may tune them in `trust.py` / `config.py` and report that.
- The retrieval-confidence scaling in `retrieval.py` (`CONF_RANGE`) is heuristic and depends on the embedding model.
- The test set has 62 cases on one database: good for a project, too small for strong statistical claims. Report numbers as indicative, and run `evaluate.py` more than once if you want to show variation.
- Your abstract mentions Spider/BIRD. This code evaluates on your ecommerce database. Adding a Spider/BIRD subset needs a small loader that gives each database to `Database(path)` and its questions to `ClarifySQL.ask`; the pipeline itself is database-agnostic.
- Gemini model names change. Defaults are `gemini-3.5-flash` (chat) and `gemini-embedding-2` (embeddings), with fallbacks. Google retired `gemini-2.5-flash` in October 2026. If a model is retired, change `GEMINI_MODELS` / `GEMINI_EMBED_MODELS` in `.env`. Gemini 3.x models work best with default sampling settings, so the code does not set temperature.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not recognized | Reinstall Python with "Add to PATH", or use `py` / `python3`. |
| `pip install` fails | Check internet; run `python -m pip install --upgrade pip` and retry. |
| App says "No API key found" | `.env` must exist (not `.env.example`) and the line must be `GEMINI_API_KEY=yourkey`. Restart the app after editing. |
| "API key was rejected" | Create a new key at aistudio.google.com/apikey and paste it again. |
| "Model ... is not available" | Put a current model name first in `GEMINI_MODELS` in `.env` (see ai.google.dev/gemini-api/docs/models). |
| Many `429` / slow answers | Free-tier limit. Wait, or add `LLM_MIN_INTERVAL=6` to `.env`, or use fewer agreement samples (sidebar slider). |
| Sidebar warns "sqlglot is not installed" | Run `pip install sqlglot` inside the activated venv. |
| Port already in use | `streamlit run app.py --server.port 8502` |
| Want to start fresh | Delete the `.cache` folder (clears cached LLM answers and the schema index). |
| Regenerate the data | `cd data` then `python make_dataset.py` (rewrites `ecommerce.db` **and** `questions.json` together). |
