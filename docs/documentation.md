# Support Question Triage: Documentation (draft; results sections filled from real runs)

## 1. Problem and hypothesis
Support emails to an IT team are often answerable from the wiki, or lack the details needed to answer. A plain RAG pipeline always
answers, so it produces confident replies to incomplete or undocumented questions. The system must choose one action per question:
ANSWER with citations, CLARIFY with one targeted follow-up, or ESCALATE with the closest pages. Hypothesis: an explicit decision step
improves action macro-F1 and lowers unsafe answers without too many unnecessary follow-ups.

## 2. Data and ingestion
- Wiki: Apache OFBiz Confluence space `OFBIZ` via the REST API, one crawl, snapshot hash recorded (README). Docs: 204 asciidoc pages from the apache/ofbiz-framework GitHub repo (the docs site disallows bots).
- Chunking: split by heading; each chunk keeps its page title and heading path as a breadcrumb prefix; at most 500 tokens, paragraph-aware.
- Questions: thread openings from the public user list, PII scrubbed, announcements, CVEs, votes and non-support posts removed, near-duplicates removed, email quotes and signatures stripped for the cleaned form.

## 3. Retrieval design
Baseline: OpenAI embeddings in local Chroma (cosine). Then, one at a time, each evaluated before the next: email cleaning of the query,
BM25 (`bm25s`) merged by reciprocal rank fusion (k=60), a local cross-encoder reranker over the top 20. Retrieval is scored by hit, recall@k and MRR
on page-level gold evidence.

## 4. Orchestration
A always answers; A+ escalates when the best first-stage vector cosine is under a threshold tuned on dev; B asks one typed Pydantic AI call
for {action, reason, missing_detail, citations, draft} with output validators (an answer must cite a provided excerpt, a clarify must name the
missing detail) and retries; C (stretch) is a LangGraph checklist flow with a human-approval interrupt. A framework stays only if it improves
results or reduces code.

## 5. Evaluation methodology
- Gold: 117 questions (97 real, plus 20 CLARIFY items made by deleting key details from real questions), split by thread before labelling (52 dev, 65 test). LLM-drafted labels from two annotators with verbatim
  evidence quotes; code verifies every quote against the frozen wiki; disagreements go to an adjudicator; the owner labels a blind subset first
  and reviews flagged items and an audit sample. Agreement statistics and owner-versus-LLM agreement are reported.
- Primary metric: action macro-F1 with stratified bootstrap confidence intervals. Also unsafe-answer, over-ask and over-escalation rates,
  recall@k, cost and latency per question. LLM judge: answer correctness, citation support, follow-up targeting, validated on owner-graded items.
- Rules fixed before test: dev-only tuning; selection rule in `docs/decisions.md`; test run once per finalist with 3 repeats; no claim inside the minimum detectable effect.

## 6. Results
[Filled from `results/ablation.md` and `results/final.md` after the real runs. Not written before the runs.]

## 7. Failure analysis
Failures found so far are in the README failure log (12 entries). The main ones, with causes:

- **CLARIFY is too rare to evaluate firmly** (4 real items, 20 invented from real questions). Claims about CLARIFY are indicative only.
- **Baseline A is unsafe by design** (unsafe-answer 1.00); the threshold A+ halves that but cannot ask for detail.
- **Retrieval additions are not proven.** Cleaning, BM25 and the reranker each move recall by less than the detectable effect on 15 items; the reranker moved it down.
- **Judge is stricter than the owner** (11 of 15 agree; all disagreements are false "incorrect").
- **Spend limit** invalidated two generation rows; they are marked and will be re-run.

The error table and risk-coverage analysis are added after the test runs.

## 8. Decisions and limitations
See README (Deviations, Limitations) and `docs/decisions.md`.
