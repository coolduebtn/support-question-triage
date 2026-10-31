"""Streamlit viewer: replays saved evaluation runs, so it needs no API key. Start with `uv run triage ui`."""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from triage.ui.data import load_bundle, verdict

ROOT = Path.cwd()
CORE = ["A", "A+", "B"]
ICON = {"correct": ":green[correct]", "wrong": ":red[wrong]", "error": ":orange[error]", "not run": ":gray[not run]"}
ACTION_BADGE = {"answer": ("ANSWER", "green"), "clarify": ("CLARIFY", "blue"), "escalate": ("ESCALATE", "orange"),
                "error": ("ERROR", "red")}

st.set_page_config(page_title="Support Question Triage", layout="wide")


@st.cache_data(ttl=30)
def bundle(root: str, split: str):
    return load_bundle(Path(root) / "results", Path(root) / "data", split)


def badge(action: str):
    text, color = ACTION_BADGE.get(action, (action.upper(), "gray"))
    st.badge(text, color=color)


def page_links(b, ids):
    for pid in dict.fromkeys(ids):
        title, url = b.page(pid)
        st.markdown(f"- [{title}]({url})" if url else f"- {title}")


def card(b, label: str, qid: str):
    row = b.predictions.get(label, {}).get(qid)
    st.subheader(label)
    v = verdict(row)
    if row is None:
        st.caption("not run on this question")
        return
    badge(row["pred"])
    st.markdown(f"vs gold: {ICON[v]}")
    if row.get("error"):
        st.warning(row["error"][:300])
    st.markdown("**Why:** " + (row["reason"] or "-"))
    if row.get("missing_detail"):
        st.markdown("**Asks for:** " + row["missing_detail"])
    if row.get("draft"):
        with st.expander("Draft" + (" follow-up" if row["pred"] == "clarify" else " answer"), expanded=True):
            st.write(row["draft"])
    j = {k: x for k, x in (row.get("judge") or {}).items() if x is not None}
    if j:
        st.caption("Judge: " + ", ".join(f"{k} {'yes' if x else 'no'}" for k, x in j.items()))
    with st.expander("Pages it saw"):
        page_links(b, row.get("retrieved_pages", []))
    st.caption(f"cost ${row['cost_usd']:.4f} · {row['latency_s']:.1f}s")


def compare_tab(b):
    qids = b.qids()
    left, right = st.columns([1, 2])
    with left:
        klass = st.radio("Show questions where the correct action is", ["any", "answer", "clarify", "escalate"],
                         horizontal=True)
        differ = st.checkbox("only where A, A+ and B disagree")
        wrong_b = st.checkbox("only where B is wrong")
    def keep(q):
        if klass != "any" and b.gold(q) != klass:
            return False
        rows = [b.predictions.get(a, {}).get(q) for a in CORE]
        if differ and len({r["pred"] for r in rows if r}) < 2:
            return False
        return not (wrong_b and verdict(b.predictions.get("B", {}).get(q)) != "wrong")
    shown = [q for q in qids if keep(q)]
    if not shown:
        st.info("No questions match these filters.")
        return
    with right:
        qid = st.selectbox(f"Question ({len(shown)} shown)", shown,
                           format_func=lambda q: f"[{b.gold(q)}] " + b.question_text(q).split("\n")[0][:80])
    st.markdown("### Question")
    st.text(b.question_text(qid)[:1500])
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Correct action (gold):**")
        badge(b.gold(qid))
    with g2:
        if b.expected_pages(qid):
            with st.expander("Pages that answer it (gold evidence)"):
                page_links(b, b.expected_pages(qid))
        if b.reference_answer(qid):
            with st.expander("Reference answer"):
                st.write(b.reference_answer(qid))
    extra = st.multiselect("Also compare retrieval variants", [a for a in b.predictions if a not in CORE])
    shown_approaches = CORE + extra
    for col, label in zip(st.columns(len(shown_approaches)), shown_approaches):
        with col:
            card(b, label, qid)


def score_table(b):
    rows = []
    for label, m in b.metrics.items():
        errors = sum(verdict(r) == "error" for r in b.predictions[label].values())
        ci = m.get("macro_f1_ci") or [None, None]
        rows.append({"approach": label, "macro-F1": m["macro_f1"], "CI low": ci[0], "CI high": ci[1],
                     "unsafe answers": m.get("unsafe_answer_rate"), "over-ask": m.get("over_ask_rate"),
                     "over-escalation": m.get("over_escalation_rate"), "recall@5": m.get("recall@5"),
                     "judged correct": m.get("judged_correctness"), "citation support": m.get("citation_support"),
                     "errors": errors})
    return pd.DataFrame(rows).set_index("approach")


def scoreboard_tab(b):
    df = score_table(b)
    bad = df[df["errors"] > 0]
    if len(bad):
        st.warning("Rows with errors are not valid generation results (the API call failed): "
                   + ", ".join(f"{i} ({int(r['errors'])} of {len(b.predictions[i])})" for i, r in bad.iterrows()))
    st.dataframe(df.style.format("{:.2f}", na_rep="n/a", subset=df.columns[:-1]), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Action macro-F1** (higher is better)")
        st.bar_chart(df.loc[[a for a in CORE if a in df.index], ["macro-F1"]])
    with c2:
        st.markdown("**Unsafe-answer rate** (lower is better): answered when it should have escalated or asked")
        st.bar_chart(df.loc[[a for a in CORE if a in df.index], ["unsafe answers"]])
    st.markdown("**What each approach did, by correct action**")
    for label in [a for a in CORE if a in b.predictions]:
        gold, pred = zip(*[(r["gold"], r["pred"]) for r in b.predictions[label].values()])
        st.caption(label)
        st.dataframe(pd.crosstab(pd.Series(gold, name="correct"), pd.Series(pred, name="predicted")))


def failures_tab(b):
    label = st.selectbox("Approach", [a for a in b.predictions] or ["B"], index=min(2, len(b.predictions) - 1))
    wrong = [(q, r) for q, r in b.predictions.get(label, {}).items() if verdict(r) in ("wrong", "error")]
    st.write(f"{len(wrong)} of {len(b.predictions.get(label, {}))} questions are wrong or errored for **{label}**.")
    for qid, r in wrong:
        with st.expander(f"correct: {r['gold']} → predicted: {r['pred']} · " + b.question_text(qid).split("\n")[0][:70]):
            st.text(b.question_text(qid)[:800])
            st.markdown("**Why it chose that:** " + (r["reason"] or r.get("error") or "-"))


def live_tab():
    st.write("Triage a new question with approach B. This calls the model API, so it needs a working key "
             "and a built index (`triage ingest --index`).")
    text = st.text_area("Support question", height=140)
    if st.button("Triage it", disabled=not text.strip()):
        with st.spinner("Retrieving and deciding..."):
            r = subprocess.run([sys.executable, "-m", "triage.cli", "ask", text], capture_output=True, text=True,
                               cwd=ROOT)
        st.code(r.stdout or r.stderr or "(no output)")


def main():
    st.title("Support Question Triage")
    st.caption("For each support question: ANSWER with citations, CLARIFY with one question, or ESCALATE to a human. "
               "A is plain RAG, A+ adds a similarity threshold, B is a typed LLM decision. "
               "Everything below is replayed from saved runs.")
    split = st.sidebar.selectbox("Question set", ["dev", "test"])
    b = bundle(str(ROOT), split)
    if not b.predictions:
        st.error(f"No saved {split} runs in results/runs. Run `triage eval --split {split}` first.")
        return
    st.sidebar.markdown("Runs shown (latest per approach)")
    st.sidebar.caption("\n".join(f"{a}: {r}" for a, r in b.run_ids.items()))
    t1, t2, t3, t4 = st.tabs(["Compare on a question", "Scoreboard", "Where it fails", "Try your own"])
    with t1:
        compare_tab(b)
    with t2:
        scoreboard_tab(b)
    with t3:
        failures_tab(b)
    with t4:
        live_tab()


main()
