# Design Doc: Support Question Triage (Answer / Clarify / Escalate)

Status: Revision 3, after two independent reviews. Awaiting owner approval; nothing is implemented yet. (The 1-2 page course design doc is the condensed version of sections 1-4; the rest is working detail.)
Author: Chandan Agarwal. Date: 2026-10-29.
Items marked **[my call]** are the assistant's proposals, not decisions the owner has made.

## 1. Problem Statement

An IT support team gets emails whose answers are often already in the wiki, and emails missing the details needed to answer them. A plain RAG pipeline always answers, so on an incomplete question or an undocumented one it risks a confident wrong reply.

**Task.** For each incoming question, choose one action:
- **ANSWER**: the docs contain the answer. Reply with a cited draft.
- **CLARIFY**: the docs could answer once one specific detail is known (version, environment, error text). Ask one targeted follow-up.
- **ESCALATE**: the docs do not cover it. Hand to a human with the closest pages.

**Hypothesis.** An explicit decision step gets more questions handled correctly (higher macro-F1, fewer unsafe answers) without asking unnecessary follow-ups.

**Stand-in data (capstone scope).** Public data replaces the internal wiki and mailbox:
- Wiki: Apache OFBiz Confluence space `OFBIZ` (705 pages) **[my call: plus the official docs site docs.ofbiz.apache.org as a second corpus, see Risks]**.
- Questions: the public `user@ofbiz.apache.org` list (about 350 root threads in 3 years, via lists.apache.org).
- The wiki source and question input sit behind small interfaces so they can later point at an internal Confluence and mailbox.

**Non-goals.** Sending email, handling replies to follow-ups, any internal or work data, a polished UI. Replies are drafts only.
**Anti-goals.** A system that scores well by asking or escalating everything. Over-ask rate and cost are constrained, not just F1.

## 2. Architecture

Plain Python first, frameworks only where they earn their place.

```
ingest:  Confluence REST -> HTML cleanup -> heading-based chunks (title + heading path kept) -> embeddings -> Chroma (+ BM25 index)
query :  email -> cleaning (strip quotes/signatures) -> retrieve top-k with scores -> DECISION POLICY -> action + draft
```

Retrieval is identical across all approaches; only the decision policy changes.

| ID | Approach | Decision rule |
|----|----------|---------------|
| A | Vanilla RAG (baseline) | Always answers |
| A+ | Score threshold | Escalate if top retrieval score is below a threshold tuned on dev; no extra LLM call |
| B | LLM decision (Pydantic AI) | Typed output {action, reason, missing_detail, citations}, with validation and retries |
| C (stretch) | Required-detail checklist (LangGraph) | Check topic-specific required details, then ask for what is missing; human-approval node |

**Retrieval ladder (core, added one component at a time, each with its own scored row; keep/drop recorded with evidence, including "no measurable gain"):**
vector (baseline) -> + email cleaning -> + BM25 hybrid with reciprocal rank fusion -> + MiniLM cross-encoder reranker (stretch: bge-reranker-v2-m3). Each component is switched on alone, evaluated, compared with the previous row and kept or dropped before the next is added; the vector baseline is evaluated first and BM25 and the reranker are not part of the system under test until their own row. Order of evaluation: A on the vector baseline, then the judge, then A+ and B on the vector baseline (so the core comparison of approaches exists early), then the ladder rows, then the best policy is re-run on the best retriever and the final selection is made. In the ladder rows the policy is A (always answers), so retrieval metrics and judged correctness on ANSWER items are the evidence there; macro-F1 is informative for A+ and B. All keep/drop decisions use dev items only. Pooled dev+test recall is reported afterwards as description, never used to decide. A+ thresholds on the vector cosine score of the top result, which every retriever keeps, so fusion and reranker scores never need to be compared. LLM-judge scores (correctness, citation support, follow-up targeting) are columns in every row.

**Ablation harness.** One command re-runs the full eval for any configuration and appends a row to `results/ablation.md` (macro-F1, per-class P/R/F1, recall@k, unsafe-answer rate, over-ask rate, cost, latency, confidence intervals, keep/drop decision). Results come only from real runs and every row stores the config, git/data hash and raw predictions.

**Interface order [my call]:** CLI and eval harness first; Streamlit last, as a thin layer over the same pipeline.

## 3. Evaluation Criteria

**Gold set.** 60-80 real questions, verbatim from the list (apart from PII scrubbing). Split by thread before any annotation or tuning: about 30 dev, the rest (about 30-50) held-out test. After labelling, and before any system run, the split may be redrawn once (logged) if any class has fewer than 5 dev items. (Revised from 20 dev after review: 20 items give about 6 per class, too few to tune on.) Label definitions, models and sampling seed are written down before any labelling.

**Labelling without hand-labelling (honest protocol).**
1. Questions are never written or paraphrased by a model.
2. Two blind LLM annotators (two different OpenAI models, both different from the system model; the key is OpenAI-only, so self-preference bias is a stated limitation) each give label, verbatim evidence quotes (page and section), the missing detail (for CLARIFY) and a draft reference answer. Their evidence path is not tied to one retriever: full-text wiki search plus the union of vector and BM25 results. This reduces, but does not remove, bias toward pages that are easy to retrieve; the README states this and checks it against the owner's blind subset.
3. Code checks that every normalised quote (Unicode and whitespace) is at least 40 characters and a substring of the frozen wiki snapshot; a second model checks the quote supports the label. Failures are flagged for review, never silently relabelled. ESCALATE items are re-searched for a missed answer; CLARIFY items must name a missing detail that would change the answer.
4. The real thread outcome (answered, followed up, unanswered) is a weak third signal, reported but not used to set labels.
5. Report Cohen's kappa, raw agreement and a confusion matrix between annotators. Disagreements go to a stronger blind adjudicator.
6. **Owner blind subset:** the owner labels 20 randomly chosen questions (minimum 12) before seeing any model label. Then the owner reviews disputed and flagged items plus a seeded audit of 12 agreed items (at least 3 per class). Owner-versus-LLM agreement, with its confidence interval and n, is the label-quality figure.
7. README states plainly: labels are LLM-drafted and mechanically verified, with an owner-labelled blind subset of n items. The submitted one-pager said "hand-labelled"; the README explains the change and why.

**Quantitative (primary first):**
- Action macro-F1 over answer/clarify/escalate (fixed `labels=` so absent classes cannot change the score), with bootstrap 95% CIs.
- Unsafe-answer rate: answered when gold was CLARIFY or ESCALATE.
- Over-ask rate, over-escalation rate.
- Retrieval hit@k / recall@k / MRR on ANSWER items, scored at page and heading level, not chunk id.
- Cost and latency per question.
- Paired comparison between adjacent systems (exact McNemar or paired bootstrap); state the minimum detectable effect (about 10-15 points at this size) and make no claims inside it.

**LLM judge (qualitative):** answer correctness against the reference, citation support per claim, whether a follow-up asks for the detail actually missing. Binary single-criterion rubrics, temperature 0, a stronger model than the system (same vendor, so self-preference bias is a stated limitation). Validated before use against about 20 real system drafts that the owner grades personally, plus clearly marked synthetic wrong answers reported separately. With n this small, agreement and kappa are reported with confidence intervals as supporting evidence, not as a pass/fail gate. Judge columns: answer correctness where gold is ANSWER and the system answered; citation support on every drafted answer; follow-up targeting where gold and prediction are both CLARIFY.

**Selection rule (pre-registered in `docs/decisions.md` before any test run):** among configurations whose dev over-ask rate and cost per question are within limits set on dev, choose the best dev macro-F1; if two are within the minimum detectable effect, choose the simpler. Thresholds are tuned on dev only. Test is then run once per finalist (3 repeats with distinct cache entries) to confirm and report; the choice is not changed after seeing test. (Revised from "best test macro-F1" after review: choosing on test would be selection on the held-out set.) Errors (API failure, invalid output) count as wrong predictions and are reported.

**Failure analysis:** each iteration logs what was tried, the scores, why it failed and the pivot, in the README as we go. Expected finding to verify (not assume): a retrieval threshold can catch ESCALATE but not CLARIFY.

## 4. Framework Justification

| Choice | Reason | Alternative rejected |
|--------|--------|----------------------|
| Plain Python for A, A+, ingestion | Baseline first; avoids architecture-first bias | Framework before a baseline |
| Chroma (local, cosine set explicitly) | Free, reproducible, enough for ~thousands of chunks | Hosted vector DB: cost and setup, no gain here |
| `bm25s` for BM25 | `rank-bm25` has not been updated since 2022 | rank-bm25 |
| Reciprocal rank fusion (k=60) | Merges rank lists without score calibration | Weighted score sum (needs calibration) |
| Cross-encoder reranker, local: MiniLM first, bge-reranker-v2-m3 as a stretch comparison | A cross-encoder reads question and chunk together, so it ranks the first-stage candidates more precisely. Cohort key covers OpenAI only; MiniLM 0.1s/50 pairs on CPU, bge 2.3GB and about 0.5s on the Mac's GPU. Whether bge's extra cost buys a gain is measured, not assumed. Reranker scores are on a different scale from vector scores, so the A+ threshold is retuned after it is added | Cloud reranker |
| Pydantic AI for B | Typed, validated output with retries is exactly what the decision step needs | Raw JSON parsing |
| LangGraph for C (stretch) | Explicit check-then-ask flow and human-approval interrupt | smolagents: built for open-ended tool use, not a fixed decision flow |
| OpenAI models (cohort key) | Same system model across approaches; stronger judge model than the system (same vendor; limitation stated) | Mixed system models (unfair comparison) |
| Streamlit last | Better demo, trivial over a UI-agnostic core | UI before eval |

A framework stays only if it improves results or reduces code, and that is recorded in the ablation table.

Versions verified on PyPI on 2026-10-29: pydantic-ai 2.63.0, langgraph 1.2.18, chromadb 1.5.9, openai 3.41.0, bm25s 0.3.16, sentence-transformers 6.2.0, scikit-learn 1.9.1, streamlit 1.66.1; Python 3.11+. Model names and prices to be confirmed against the cohort key with `client.models.list()`.

## 5. Deliverables and Order of Work

Design doc (this) -> ingestion + frozen snapshot -> question collection and labelling protocol -> eval harness with A -> A+ -> B -> retrieval ladder -> C (stretch) -> README failure analysis, 4-5 page documentation -> CLI demo and 3-minute video -> Streamlit (optional).

## 6. Risks (feasibility check, 2026-10-29)

Based on the saved list-API sample (not yet an ingestion of the wiki):
- 349 root threads, of which about 36 look like announcements, CVEs or release notes, leaving roughly 310 candidate questions. 139 threads had no reply. Enough for 60-80 labelled items, with little spare.
- Many real questions are developer-level (service names, form widgets), which the wiki may not answer. **[my call]** Add docs.ofbiz.apache.org as a second corpus, otherwise ESCALATE may dominate. Decide after the real ingestion shows the class mix.
- The wiki is uneven: median about 243 words, many 2006-2010 proposals.
- Terms: public archives with no explicit research licence. Use for research, cite sources, redact names and emails, do not redistribute raw dumps.
- Fallback sources if the class mix is unusable: Apache Solr (856 threads), CloudStack (largest list).
- With about 30-50 test items, differences under roughly 15 points are noise; the report states the minimum detectable effect and makes no claim inside it. "Improves" means the paired bootstrap 95% interval of the difference excludes zero; otherwise the row says "no measurable gain".
- LLM-written reference answers are judged by an LLM, a known limitation. References on the owner's blind subset are owner-reviewed.
- Repeats (3 per finalist on test) are reported as mean and min-max; confidence intervals are bootstrapped on the first repeat.
- Deviations from the submitted one-pager (labelling method, BM25 and reranker moved into the core ladder, dev size) are listed in the README.
- OpenAI key not yet available; embedding and model choices are provisional.
