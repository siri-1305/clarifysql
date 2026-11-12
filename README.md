# ClarifySQL - clarification-aware, retrieval-augmented Text-to-SQL with a trust badge

Final-year major project, Dept. of CSE, RGUKT Srikakulam.

A chat app where a non-SQL user asks questions about an e-commerce database.
It **asks a follow-up only when the question is vague**, retrieves **only the relevant tables**, writes and **validates/repairs** the SQL,
then **explains in plain English** how it understood the question and shows a **High / Medium / Low trust badge**.

```
Question -> Ambiguity check -> Schema retrieval (RAG) -> SQL generation -> Validate & repair -> Explanation + Trust badge
              (ask only if       (embeddings +            (Groq)          (sqlglot + SQLite,     (plain English,
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
This installs streamlit, groq, sqlglot, numpy, pandas, python-dotenv. Needs internet, takes 1-3 minutes.

### Step 5 - Get a free Groq API key and put it in `.env`
1. Open https://console.groq.com/keys , sign in, click **Create API Key**, copy it.
2. Make your settings file by copying the example:
   - Windows: `copy .env.example .env`
   - Mac/Linux: `cp .env.example .env`
3. Open `.env` in a text editor and replace `PASTE_YOUR_KEY_HERE` with your key. No quotes, no spaces:
   ```
   GROQ_API_KEY=gsk_...yourkey...
   ```
4. Save. **Never share or upload this file** (it is already in `.gitignore`).

### Step 6 - Run the health check (do this before the app)
```
python selfcheck.py --online
```
It checks the packages, the database, that the validator accepts all 60 reference queries, that it catches mistakes, runs 14 offline tests,
and (with `--online`) calls Groq once. You want to see **ALL CHECKS PASSED**.
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
    llm.py                Groq client: retries, token-aware rate limiting, model fallback, disk cache
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



---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not recognized | Reinstall Python with "Add to PATH", or use `py` / `python3`. |
| `pip install` fails | Check internet; run `python -m pip install --upgrade pip` and retry. |
| App says "No API key found" | `.env` must exist and the line must be `GROQ_API_KEY=yourkey`. Restart the app after editing. |
| "API key was rejected" | Create a new key at console.groq.com/keys and paste it again. |
| "Model ... is not available" | Put a current model name first in `GROQ_MODELS` in `.env` (see console.groq.com/docs/models). |
| Many `429` / slow answers | Free-tier limit. Wait, add `LLM_MIN_INTERVAL=4` to `.env`, or use fewer agreement samples (sidebar slider). Daily limit: add a second model to `GROQ_MODELS`. |
| Sidebar warns "sqlglot is not installed" | Run `pip install sqlglot` inside the activated venv. |
| Port already in use | `streamlit run app.py --server.port 8502` |
| Want to start fresh | Delete the `.cache` folder (clears cached LLM answers and the schema index). |
| Regenerate the data | `cd data` then `python make_dataset.py` (rewrites `ecommerce.db` **and** `questions.json` together). |
