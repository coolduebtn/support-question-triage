"""Streamlit viewer: replays saved evaluation runs, so it needs no API key. Start with `uv run triage ui`."""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from triage.ui.data import explain_failure, live_problem, load_bundle, verdict

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
    st.text(b.question_text(qid)[:1500] or "(question text is not part of the public data)")
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
    retrieval_section(b)
    risk_section(b)


def retrieval_section(b):
    st.markdown("#### Retrieval variants (policy A, answerable questions only)")
    rows = b.retrieval_rows()
    if not rows:
        st.info("No retrieval variants have been run on this split yet.")
        return
    df = pd.DataFrame(rows).rename(columns={"label": "variant", "n": "questions", "hit": "hit@5",
                                            "recall": "recall@5", "mrr": "MRR"}).set_index("variant")
    st.dataframe(df.style.format({"hit@5": "{:.2f}", "recall@5": "{:.2f}", "MRR": "{:.2f}"}), width="stretch")
    st.caption("hit = a gold page is in the top 5; recall = share of the gold pages found; MRR = rank of the first gold page. "
               f"With only {rows[0]['n']} answerable questions, differences of a few points are not reliable; "
               "paired confidence intervals are in the README.")


def risk_section(b):
    st.markdown("#### Risk and coverage (policy A, sweeping the top similarity score)")
    curve = b.risk_coverage("A")
    if not curve:
        st.info("Policy A has not been run on this split.")
        return
    df = pd.DataFrame(curve, columns=["coverage", "risk"]).set_index("coverage")
    st.line_chart(df)
    st.caption("Coverage = share of questions answered. Risk = share of those answers that should have been "
               "an escalation or a follow-up. If risk stays high as coverage falls, similarity alone cannot tell "
               "answerable questions apart.")


def failures_tab(b):
    label = st.selectbox("Approach", [a for a in b.predictions] or ["B"], index=min(2, len(b.predictions) - 1))
    wrong = [(q, r) for q, r in b.predictions.get(label, {}).items() if verdict(r) in ("wrong", "error")]
    st.write(f"{len(wrong)} of {len(b.predictions.get(label, {}))} questions are wrong or errored for **{label}**.")
    for qid, r in wrong:
        with st.expander(f"correct: {r['gold']} → predicted: {r['pred']} · " + b.question_text(qid).split("\n")[0][:70]):
            st.caption(f"Likely cause: {b.cause(label, qid).replace('_', ' ')}")
            st.text(b.question_text(qid)[:800])
            st.markdown("**Why it chose that:** " + (r["reason"] or r.get("error") or "-"))


STYLE = """
  rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.4;
  node [shape=box, style="rounded,filled", fillcolor="#eef2f7", color="#8a97a8", fontname="Helvetica", fontsize=11, margin="0.12,0.07"];
  edge [color="#8a97a8", arrowsize=0.7, fontname="Helvetica", fontsize=10];
"""

ARCH_DIAGRAM = "digraph G {" + STYLE + """
  q [label="Support question"];
  r [label="Retrieve\\nOpenAI embeddings + Chroma vector index\\n(top 5 chunks of wiki and docs)"];
  a [label="A: always answer\\nLLM drafts from the chunks"];
  g [label="A+: similarity gate\\nbest cosine below 0.6 ?", shape=diamond, style=filled];
  b [label="B: typed LLM decision\\nvalidated output, up to 2 retries", fillcolor="#e3f1e6"];
  ans [label="ANSWER\\ncited draft", fillcolor="#e3f1e6"];
  cl [label="CLARIFY\\none follow-up question", fillcolor="#e3ecf9"];
  es [label="ESCALATE\\nhand to a human", fillcolor="#fbe9d5"];
  q -> r; r -> a; r -> g; r -> b;
  a -> ans;
  g -> es [label="yes"]; g -> ans [label="no (same draft as A)"];
  b -> ans; b -> cl; b -> es;
}"""

METHOD_DIAGRAM = "digraph G {" + STYLE + """
  data [label="Data\\n5,397 chunks (wiki + docs)\\n259 mailing-list threads"];
  gold [label="Gold set: 117 questions\\nANSWER / CLARIFY / ESCALATE\\n(LLM-drafted, mechanically checked,\\nowner-graded blind subset)"];
  dev [label="Dev: 52\\ntune thresholds,\\nchoose approach", fillcolor="#e3f1e6"];
  test [label="Test: 65\\nheld out, run once\\n(B three times)", fillcolor="#fbe9d5"];
  ladder [label="Retrieval ladder (dev)\\nvector, + cleaning, + BM25, + reranker\\nkeep a step only if the paired\\nbootstrap interval excludes zero"];
  appr [label="Approaches\\nA, A+, B"];
  sel [label="Selection rule\\nfixed before the test run:\\nmacro-F1, unsafe-answer, cost cap,\\nties go to the simpler system"];
  final [label="Selected: B\\nscored on test", fillcolor="#fbe9d5"];
  data -> gold -> dev; gold -> test;
  dev -> ladder -> appr -> sel -> final; test -> final;
}"""


def method_tab():
    st.markdown("### Architecture and approaches")
    st.graphviz_chart(ARCH_DIAGRAM, width="stretch")
    st.markdown("All three approaches share the same retrieval, so any difference between them comes from the decision "
                "step.  \n"
                "**A** is plain RAG: it always answers.  \n"
                "**A+** adds a similarity threshold (0.6, tuned on dev): if the best match is weak it escalates, "
                "otherwise it answers exactly like A.  \n"
                "**B** makes one typed decision: ANSWER with citations, CLARIFY with one follow-up, or ESCALATE. "
                "Validators require that an answer cites a chunk it was given and that a clarify names the missing "
                "detail; a failed check triggers up to two retries.  \n"
                "A LangGraph checklist flow (C) was planned as a stretch and not built.")
    st.markdown("### How it was tested")
    st.graphviz_chart(METHOD_DIAGRAM, width="stretch")
    st.markdown("**Gold set.** 117 questions, each labelled ANSWER, CLARIFY or ESCALATE. Two LLMs drafted the labels, "
                "a script checked them against the source pages, and a third model adjudicated disagreements. "
                "The owner graded a blind subset. 20 CLARIFY questions are constructed, because real ones are rare "
                "(4 of 97), and they are flagged in the data.  \n"
                "**Dev and test.** Thresholds and the choice of approach come from the 52 dev questions. "
                "The 65 test questions are scored once per approach, and B three times.  \n"
                "**Retrieval ladder.** Email cleaning, BM25 hybrid search and a reranker were each added and scored. "
                "None passed the keep rule, so every approach uses plain vector search.  \n"
                "**Selection rule.** Written down before the test run. The best dev macro-F1 wins among approaches "
                "inside the limits for over-asking, over-escalation, unsafe answers and cost. "
                "Ties inside the minimum detectable effect go to the simpler system.  \n"
                "**LLM judge.** Scores answer correctness and citation support. It is a Claude model, while the system "
                "that writes the answers is an OpenAI model, so the judge does not grade its own family. "
                "It agreed with 10 of 15 owner grades, so it is a rough signal, not ground truth.")


def live_tab():
    st.write("Triage a new question with approach B. This calls the model API, so it needs an OpenAI key and a built "
             "search index. About $0.003 per question and 10 to 15 seconds.")
    problem = live_problem(ROOT)
    if problem:
        st.warning(problem)
    text = st.text_area("Support question", height=140)
    if st.button("Triage it", disabled=bool(problem) or not text.strip()):
        with st.spinner("Retrieving and deciding..."):
            r = subprocess.run([sys.executable, "-m", "triage.cli", "ask", text], capture_output=True, text=True,
                               cwd=ROOT)
        if getattr(r, "returncode", 0) != 0:
            st.error(explain_failure(r.stderr))
            with st.expander("Details"):
                st.code(r.stderr or "(no output)")
        else:
            st.code(r.stdout or r.stderr or "(no output)")


def main():
    st.title("Support Question Triage")
    st.caption("For each support question: ANSWER with citations, CLARIFY with one question, or ESCALATE to a human.  \n"
               "**A** is plain RAG.  \n"
               "**A+** adds a similarity threshold.  \n"
               "**B** is a typed LLM decision.  \n"
               "Everything below is replayed from saved runs.")
    split = st.sidebar.selectbox("Question set", ["dev", "test"])
    b = bundle(str(ROOT), split)
    if not b.predictions:
        st.error(f"No saved {split} runs in results/runs. Run `triage eval --split {split}` first.")
        return
    if split == "test":
        st.info("The test set is scored once per approach. Where an approach was run several times (B, three repeats), "
                "only the first repeat is shown here.")
    st.sidebar.markdown("Runs shown (latest per approach)")
    st.sidebar.caption("  \n".join(f"**{a}**: {r}" for a, r in b.run_ids.items()))
    t0, t1, t2, t3, t4 = st.tabs(["Methodology", "Compare on a question", "Scoreboard", "Where it fails",
                                   "Try your own"])
    with t0:
        method_tab()
    with t1:
        compare_tab(b)
    with t2:
        scoreboard_tab(b)
    with t3:
        failures_tab(b)
    with t4:
        live_tab()


main()
