"""ClarifySQL - Streamlit chat interface.   Run:  streamlit run app.py"""
import json
from dataclasses import replace

import pandas as pd
import streamlit as st

from clarifysql.config import PipelineOptions, Settings
from clarifysql.db import Database
from clarifysql.llm import GeminiLLM, LLMError
from clarifysql.pipeline import ClarifySQL, PipelineResult
from clarifysql.retrieval import SchemaIndex
from clarifysql.validator import HAS_SQLGLOT

st.set_page_config(page_title="ClarifySQL", page_icon="🔎", layout="wide")

BADGE_COLORS = {"High": "#1a7f37", "Medium": "#b7791f", "Low": "#c0392b"}


# ----------------------------------------------------------------------------- backend (built once)
@st.cache_resource(show_spinner="Starting ClarifySQL (reading the database and building the schema index)...")
def load_backend():
    settings = Settings()
    db = Database(settings.db_path)
    llm = GeminiLLM(settings)
    index = SchemaIndex(db, llm, settings.cache_dir)
    return settings, db, llm, index


try:
    settings, db, llm, index = load_backend()
except (LLMError, FileNotFoundError) as e:
    st.title("🔎 ClarifySQL")
    st.error(f"Could not start: {e}")
    st.markdown("**Fix:** open the file named `.env` in the project folder, put your key on the line "
                "`GEMINI_API_KEY=...`, save, and restart with `streamlit run app.py`.")
    st.stop()


def example_questions():
    try:
        qs = json.loads(settings.questions_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    pick = {"C14": "clear", "C07": "clear", "V01": "vague", "V02": "vague", "V10": "vague", "T01": "tricky",
            "T02": "tricky", "T12": "unanswerable"}
    return [(pick[q["id"]], q["question"]) for q in qs if q["id"] in pick]


# ----------------------------------------------------------------------------- session state
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("pending", None)      # open clarification: {"question", "clarification_question", "options"}
ss.setdefault("queue", None)        # work to do on the next run: {"kind": "question"|"clarify"|"skip", "text": str}
ss.setdefault("counter", 0)


def queue_work(kind: str, text: str = ""):
    ss.queue = {"kind": kind, "text": text}


def new_chat():
    ss.messages, ss.pending, ss.queue = [], None, None


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🔎 ClarifySQL")
    st.caption("Asks only when needed. Tells you how far to trust each answer.")
    st.button("🗑️ New chat", on_click=new_chat)

    st.subheader("Modules")
    use_clar = st.toggle("Ask follow-up questions when vague", value=True)
    use_ret = st.toggle("Schema retrieval (RAG)", value=True)
    use_rep = st.toggle("Validate and repair SQL", value=True)
    thr = st.slider("Ambiguity threshold", 0.1, 0.9, 0.5, 0.05,
                    help="A follow-up question is asked only when the ambiguity score is at or above this value.")
    n_samples = st.slider("Extra queries for agreement check", 0, 3, 2,
                          help="More samples = a more reliable trust badge but more LLM calls.")
    show_sql = st.checkbox("Show the SQL", value=True)

    st.subheader("System")
    st.write(f"Chat model: `{llm.model}`")
    st.write(f"Retrieval: `{index.api_model if index.api_vecs is not None else 'local fallback'}`")
    if not HAS_SQLGLOT:
        st.warning("sqlglot is not installed - schema and filter-value checks are reduced.")

    ex = example_questions()
    if ex:
        st.subheader("Try these")
        for i, (kind, q) in enumerate(ex):
            st.button(f"{q}", key=f"ex{i}", on_click=queue_work, args=("question", q), help=f"({kind} question)")

    with st.expander("Database schema"):
        st.code(db.schema_text(), language="text")

options = PipelineOptions(use_clarification=use_clar, use_retrieval=use_ret, use_repair=use_rep,
                          ambiguity_threshold=thr, agreement_samples=n_samples)


# ----------------------------------------------------------------------------- rendering helpers
def unique_columns(cols):
    seen, out = {}, []
    for c in cols:
        seen[c] = seen.get(c, 0) + 1
        out.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
    return out


def render_badge(res: PipelineResult):
    t = res.trust
    st.markdown(f"<span style='background:{BADGE_COLORS[t.level]};color:white;padding:4px 14px;border-radius:14px;"
                f"font-weight:600'>{t.level} trust</span> &nbsp; <span style='color:gray'>score {t.score:.2f}</span>",
                unsafe_allow_html=True)
    if t.level == "Low":
        st.warning("Please double-check this result before relying on it.")
    elif t.level == "Medium":
        st.info("Looks reasonable, but worth a quick check.")
    with st.expander("Why this trust level?"):
        for r in t.reasons:
            st.write("• " + r)


def render_result(res: PipelineResult):
    if res.status == "unanswerable":
        st.info(f"I can't answer that from this database. {res.message}")
        return
    if res.status == "error":
        st.error(res.message or "Something went wrong.")
        if res.sql and show_sql:
            st.code(res.sql, language="sql")
        return
    if res.explanation:
        st.markdown(res.explanation.summary)
        if res.explanation.assumptions:
            st.markdown("**Assumptions I made:** " + "; ".join(res.explanation.assumptions))
    if res.trust:
        render_badge(res)
    if res.rows:
        df = pd.DataFrame(res.rows, columns=unique_columns(res.columns))
        st.dataframe(df)
        if res.truncated:
            st.caption(f"Showing the first {len(res.rows)} rows only.")
    else:
        st.write("The query returned no rows.")
    if show_sql and res.sql:
        with st.expander("SQL"):
            st.code(res.sql, language="sql")
    with st.expander("Details (how this was produced)"):
        if res.ambiguity:
            st.write(f"Ambiguity score: **{res.ambiguity.score:.2f}** (threshold {thr:.2f}) - {res.ambiguity.reason}")
        if res.retrieval:
            st.write(f"Tables retrieved: **{', '.join(res.retrieval.tables)}** "
                     f"(confidence {res.retrieval.confidence:.2f}, {res.retrieval.method})"
                     + (" - had to fall back to the full schema" if res.retrieval_expanded else ""))
        st.write(f"Automatic repairs: **{res.repairs}** | Agreement: "
                 f"**{'n/a' if res.agreement is None else f'{res.agreement:.0%} of {res.n_candidates}'}** | "
                 f"LLM calls: **{res.llm_calls}** | Time: **{res.seconds:.1f}s**")
        if res.explanation and res.explanation.filters:
            st.write("Filters applied: " + "; ".join(f"`{f}`" for f in res.explanation.filters))
        if len(res.attempts) > 1:
            st.write("Repair history:")
            for i, a in enumerate(res.attempts, 1):
                st.code(a.sql, language="sql")
                if a.errors:
                    st.caption("Problems: " + " | ".join(a.errors))


def render_message(i: int, m: dict, is_last: bool):
    with st.chat_message(m["role"]):
        if m["kind"] == "text":
            st.markdown(m["content"])
        elif m["kind"] == "clarify":
            st.markdown(m["content"])
            if is_last and ss.pending:
                cols = st.columns(max(1, min(4, len(ss.pending["options"]))))
                for j, opt in enumerate(ss.pending["options"]):
                    cols[j % len(cols)].button(opt, key=f"opt_{i}_{j}", on_click=queue_work, args=("clarify", opt))
                st.button("Just answer it directly", key=f"skip_{i}", on_click=queue_work, args=("skip",))
                st.caption("Click an option, or type your own answer below.")
        elif m["kind"] == "result":
            render_result(m["result"])


# ----------------------------------------------------------------------------- main page
st.title("Ask your data a question")
st.caption("Example: *Show me the best products* - I will ask what 'best' means - or *How many orders were cancelled?* - answered directly.")

if not ss.messages:
    ss.messages.append({"role": "assistant", "kind": "text",
                        "content": "Hi! Ask me anything about the shop's customers, products, orders and reviews. "
                                   "I'll ask a quick question only if yours is vague, and I'll tell you how far to trust each answer."})

for i, m in enumerate(ss.messages):
    render_message(i, m, is_last=(i == len(ss.messages) - 1))

job = ss.queue
if job:
    ss.queue = None
    app = ClarifySQL(db, llm, index, options)
    pending = ss.pending
    if job["kind"] == "question":
        shown_user, ss.pending = job["text"], None
    elif job["kind"] == "clarify" and pending:
        shown_user = job["text"]
    elif job["kind"] == "skip" and pending:
        shown_user = "Just answer it directly"
    elif job["text"]:  # an answer arrived but the open question was lost: treat the text as a new question
        shown_user, job = job["text"], {"kind": "question", "text": job["text"]}
        ss.pending = None
    else:
        st.rerun()

    with st.chat_message("user"):
        st.markdown(shown_user)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if job["kind"] == "question":
                res = app.ask(job["text"])
            elif job["kind"] == "clarify":
                res = app.ask(pending["question"], clarification_answer=job["text"],
                              clarification_question=pending["clarification_question"])
            else:
                res = ClarifySQL(db, llm, index, replace(options, use_clarification=False)).ask(pending["question"])

    ss.messages.append({"role": "user", "kind": "text", "content": shown_user})
    if res.status == "clarify":
        ss.pending = {"question": job["text"], "clarification_question": res.clarification_question, "options": res.options}
        ss.messages.append({"role": "assistant", "kind": "clarify", "content": f"**{res.clarification_question}**"})
    else:
        ss.pending = None
        ss.messages.append({"role": "assistant", "kind": "result", "result": res})
    st.rerun()

placeholder = "Type your answer to my question..." if ss.pending else "Ask a question about the data..."
prompt = st.chat_input(placeholder)
if prompt:
    queue_work("clarify" if ss.pending else "question", prompt.strip())
    st.rerun()
