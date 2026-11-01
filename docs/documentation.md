# Support Question Triage: Documentation

## 1. Problem and hypothesis

Support emails sent to an IT team fall into three groups: some can be answered from the documentation, some lack the details needed to
answer, and some need a person. A plain retrieval-augmented (RAG) pipeline treats all of them the same way: it retrieves pages and writes
an answer. For the second and third groups, that answer is a confident reply to a question the documentation cannot settle.

The system therefore chooses one action per question:

- **ANSWER** with a drafted reply that cites the documentation,
- **CLARIFY** with one targeted follow-up question, or
- **ESCALATE** to a human, with the closest pages attached.

Nothing is ever sent. Replies are drafts for a person to review.

**Hypothesis.** An explicit decision step handles more questions correctly (higher action macro-F1) and gives fewer unsafe answers
than plain RAG, without asking unnecessary follow-ups.

The public stand-in for a private support desk is the [Apache OFBiz Confluence wiki](https://cwiki.apache.org/confluence/display/OFBIZ)
and the [OFBiz documentation sources](https://github.com/apache/ofbiz-framework), with questions taken from the
[`user@ofbiz.apache.org` mailing list](https://lists.apache.org/list.html?user@ofbiz.apache.org).

**How the evidence is built.** The question set, the metrics, the selection rule and the keep/drop rule for retrieval changes were written
down before any system was run (`docs/decisions.md`, `configs/limits.toml`). Settings are tuned on a dev set only. The test set is run once
per finalist. Every change to the pipeline gets its own scored row, so each claim has a comparison behind it.

## 2. Approaches compared

| Approach | What it does |
|---|---|
| **A** | Plain RAG. Retrieves the top 5 chunks and always answers. |
| **A+** | A plus a similarity threshold. If the best vector cosine score is below a threshold tuned on dev (0.6), it escalates instead of answering. |
| **B** | One typed LLM decision (Pydantic AI). The model returns `{action, reason, missing_detail, citations, draft}`. Output validators require that an answer cites an excerpt it was given and that a clarify names the missing detail, and a failed check triggers up to two retries. |
| C | Planned stretch: a LangGraph checklist flow. **Not built.** |

A, A+ and B use the same retrieval, so any difference between them comes from the decision step. Retrieval is improved separately,
one component at a time (section 5).

Models: system `gpt-5-mini` (A, A+, B); judge `claude-sonnet-5-5`; embeddings `text-embedding-3-small`; label annotators `gpt-5.4`
and `gpt-4.1` with a `gpt-5.5` adjudicator. The judge is a different vendor from the system model on purpose. If no Anthropic key is
available, the judge falls back to an OpenAI model (`gpt-5.4`) and the run records which judge was used.

## 3. Data and ingestion

- **Wiki:** the [Apache OFBiz Confluence space](https://cwiki.apache.org/confluence/display/OFBIZ) (key `OFBIZ`), 705 pages from one crawl through the REST API. The snapshot hash is recorded in the README.
- **Docs:** 204 asciidoc pages from the [apache/ofbiz-framework](https://github.com/apache/ofbiz-framework) GitHub repository. The docs website disallows bots, so the sources
  were taken from the repository instead.
- **Chunking:** split by heading. Each chunk keeps its page title and heading path as a prefix, and holds at most 500 tokens, split by paragraph.
  The index has 5,397 chunks over 845 pages (63 wiki pages and 1 docs page had no text, so they produced no chunks). The median chunk is 529 characters, because pages are split by many headings.
- **Questions:** 259 threads from the [`user@ofbiz.apache.org` list archive](https://lists.apache.org/list.html?user@ofbiz.apache.org). Personal details (emails, phone numbers, links, IP addresses, known names) were scrubbed.
  Announcements, CVEs, votes and non-support posts were removed, and near-duplicates were removed. Quoted text and signatures are stripped for the cleaned form
  of each question. Raw question text stays in a gitignored folder.

## 4. Gold set and labelling

**The gold set has 117 questions: 97 real and 20 constructed.** Of these, 52 are dev and 65 are test. Questions were split by thread
before any labelling, so replies from one conversation never land on both sides.

| | answer | clarify | escalate | total |
|---|---|---|---|---|
| dev | 15 | 13 | 24 | 52 |
| test | 13 | 11 | 41 | 65 |

Real CLARIFY questions are rare in the mailing list: only 4 of the 97 real questions are CLARIFY. To give CLARIFY a usable
sample, 20 constructed CLARIFY items were added by deleting key details (a version, an error message, the name of a module) from real answerable
questions. The set therefore has 24 CLARIFY items, 20 of them constructed. They carry a `synthetic` flag in the data and are never used when tuning
the A+ threshold. Results restricted to real questions are available (`triage eval --synthetic none`).

**Labelling method.** The labels are **not hand-labelled in full**. Two LLM annotators label each question blind from retrieved evidence. Code
checks that every evidence quote appears verbatim in the frozen documentation. Disagreements go to an adjudicator model. The project owner
labelled a blind subset first, without seeing any model label, and then reviewed a queue of flagged items.

| Source of the final label (real questions) | Count |
|---|---|
| Owner blind (labelled before seeing any model label) | 20 |
| Owner review (disputed, evidence-failed or audit items; 0 overrides) | 43 |
| Both annotators agreed, not reviewed | 34 |

- Owner blind labels against final labels: 17 of 20 agree (85%, Wilson 95% interval 0.64 to 0.95, kappa 0.44).
- Annotator agreement: kappa 0.46 (raw 0.70).
- The owner confirmed 43 of 43 reviewed items. This is an upper bound on label quality, because the owner saw the model drafts.

## 5. Retrieval design and the ladder

Baseline retrieval embeds the question and searches a local Chroma index by cosine similarity (top 5 chunks). Three components were added
one at a time, each scored on its own row against the row before it:

1. **Email cleaning** of the query (strip quotes, signatures and greetings).
2. **BM25** keyword search (`bm25s`), merged with the vector results by reciprocal rank fusion (k = 60).
3. **A cross-encoder reranker** (MiniLM) over the top 20 candidates.

**Keep/drop rule (pre-registered).** A component counts as an improvement only if the 95% paired bootstrap interval of its difference from
the previous row excludes zero on dev.

Retrieval-only results on the 15 answerable dev questions (page-level, k = 5):

| Step | hit@5 | recall@5 | MRR | Change in recall vs previous [95% interval] | Improves |
|---|---|---|---|---|---|
| Vector, raw email | 0.87 | 0.74 | 0.79 | n/a | n/a |
| + email cleaning | 0.93 | 0.81 | 0.84 | +0.067 [+0.000, +0.200] | no measurable gain |
| + BM25 with RRF | 1.00 | 0.88 | 0.91 | +0.067 [-0.067, +0.233] | no measurable gain |
| + MiniLM reranker | 0.87 | 0.75 | 0.77 | -0.128 [-0.339, +0.056] | no measurable gain |

**None of the three components passed the rule**, so retrieval stays as plain vector search for A+ and B. This is a finding about this
benchmark, not a claim that hybrid search or reranking do not work. Section 8 explains why the benchmark could not show a gain.

## 6. Evaluation method

**Primary metric:** action macro-F1 over the three classes, with a stratified bootstrap 95% interval. A system that always gives the same action
has a zero-width interval, because it has no variation to resample.

**Other metrics:**
- *Unsafe-answer rate:* the share of CLARIFY and ESCALATE questions that the system answered anyway.
- *Over-ask and over-escalation rates:* the share of answerable questions where the system asked a follow-up or escalated instead.
- *Retrieval:* hit, recall and MRR at the page level.
- *Cost and latency* per question.
- *Judged quality* (below).

**LLM judge.** Three binary rubrics: is the answer correct against the reference answer, is each citation supported by its excerpt, and does the
follow-up ask for the missing detail. Answer correctness is counted only where both the gold label and the prediction are ANSWER, so it is a quality
measure for answered questions and not a second accuracy.

**Judge validation.** The owner graded 15 real drafts and the judge was compared with those grades.

| Judge | Agreement with owner | Kappa [95% interval] |
|---|---|---|
| `gpt-5.5` (first judge) | 11 of 15 (73%) | 0.48 [0.23, 0.84] |
| `claude-sonnet-5-5` (used for the final results) | 10 of 15 (67%) | 0.39 [0.17, 0.71] |

Both judges are stricter than the owner. Every disagreement was the judge calling a draft incorrect where the owner called it correct, and neither judge
accepted a wrong answer. The difference between the two judges is one item. The judge was switched from `gpt-5.5` to `claude-sonnet-5-5` to cut cost
and to separate the judge from the system model's vendor. With 15 graded items, judge validation is report-only. Judged-correct figures are therefore
conservative, and the comparison between approaches is more reliable than the absolute levels.

**Statistics.** Paired bootstrap for differences between approaches; exact McNemar test on per-question correctness; a minimum detectable effect
(about 0.19 on dev, 0.17 on test) below which a difference is not claimed. A claim needs both a difference above the minimum detectable effect and an interval
that excludes zero.

**Selection rule (pre-registered).** Among configurations within the limits in `configs/limits.toml` (over-ask at most 0.25, over-escalation
at most 0.40, unsafe answers at most 0.60, cost at most $0.02 per question), choose the best dev macro-F1. If two are within the
minimum detectable effect, choose the simpler. The test set is run once per finalist, and the choice is not changed after seeing test.
Errors count as wrong predictions.

## 7. Results

### Dev set (52 questions, rows scored in sequence)

| Row | Macro-F1 [95% interval] | Unsafe-answer | Over-escalation | recall@5 | Judged correct | Citation support | Cost/q |
|---|---|---|---|---|---|---|---|
| A, vector baseline | 0.15 [0.15, 0.15] | 1.00 | 0.00 | 0.74 | 0.47 | 0.70 | $0.0026 |
| A+, threshold 0.6 | 0.29 [0.20, 0.38] | 0.46 | 0.40 | 0.74 | 0.44 | 0.70 | $0.0014 |
| **B, typed decision** | **0.66 [0.51, 0.78]** | **0.30** | 0.20 | 0.74 | 0.50 | 0.78 | $0.0025 |
| A + email cleaning | 0.15 | 1.00 | 0.00 | 0.81 | 0.67 | 0.73 | $0.0026 |
| A + BM25 with RRF | 0.15 | 1.00 | 0.00 | 0.88 | 0.60 | 0.68 | $0.0028 |
| A + MiniLM reranker | 0.15 | 1.00 | 0.00 | 0.75 | 0.47 | 0.71 | $0.0027 |

All rows ran with 0 errors. The retrieval rows (A + ...) always answer, so their macro-F1 is the same as A's. They are compared on retrieval
and judged quality only. Judged correctness is computed on just 15 answerable items, so its differences are not claimed.

**Selection on dev.** A fails the unsafe-answer limit. A+ and B are within all limits, and B has the higher macro-F1. B vs A+ is
+0.36 [+0.21, +0.51] (exact McNemar p = 0.0004, above the 0.19 minimum detectable effect), and B vs A is +0.51 [+0.36, +0.63]. A+ vs A
(+0.14 [+0.05, +0.23]) is below the minimum detectable effect and is not claimed. **B is selected.**

### Test set (65 questions, run once per finalist; B run three times)

| Approach | Macro-F1 [95% interval] | Unsafe-answer | Over-ask | Judged correct | Citation support | Cost/q |
|---|---|---|---|---|---|---|
| A | 0.11 [0.11, 0.11] | 1.00 | 0.00 | 0.62 | 0.67 | $0.0028 |
| A+ | 0.33 [0.25, 0.42] | 0.25 | 0.00 | 0.80 | 0.71 | $0.0007 |
| **B** (first repeat) | **0.62 [0.48, 0.74]** | **0.19** | 0.15 | 0.75 | 0.70 | $0.0027 |

Paired comparisons on test:

| Comparison | Difference in macro-F1 [95% interval] | Exact McNemar p | Claim |
|---|---|---|---|
| B vs A | +0.51 [+0.37, +0.64] | below 0.0001 | yes |
| A+ vs A | +0.22 [+0.14, +0.31] | 0.0005 | yes |
| B vs A+ | +0.28 [+0.12, +0.44] | 0.087 | yes on macro-F1, weaker on accuracy |

**B is not fully stable.** Over its three runs on test, macro-F1 was 0.62, 0.49 and 0.57 (mean 0.56, spread 0.13), and B gave the same action in
all three runs for 44 of 65 questions. The single-run figure of 0.62 is therefore the best of the three, and the mean is the more honest summary.
The drop from dev (0.66) to test is real and about 0.04 to 0.10 depending on which figure is used.

**What the results support.**
- Plain RAG (A) is unsafe by design: it answered every question, including all of the ones that needed a human or more detail.
- A threshold alone (A+) cuts unsafe answers a lot (1.00 to 0.25 on test) but cannot ask a follow-up, and it escalated many answerable questions.
- A typed decision (B) cut unsafe answers further (to 0.19 on test) and is the only approach that asks follow-ups at all.
- B's gain over A+ is smaller and less certain than its gain over A.

## 8. Failure analysis

Source: `results/failure_analysis.md`, built by `scripts/failure_analysis.py` from the saved runs.

**What B got wrong.** On dev B was wrong on 16 of 52; on test, 20 of 65. Automatic cause hints on test: 9 class-boundary mismatches, 6 unsafe
answers (answered an ESCALATE item), 4 missed ambiguities (answered a CLARIFY item), and 1 retrieval miss. Retrieval is rarely the cause: on test, only 1 of B's 20 errors
is a retrieval miss, so the remaining errors are decision errors. The hints are automatic and are confirmed or corrected by reading the rows.

| Per-class correct (test) | answer | escalate | clarify (all) | clarify, real | clarify, constructed |
|---|---|---|---|---|---|
| A | 13 of 13 | 0 of 41 | 0 of 11 | 0 of 3 | 0 of 8 |
| A+ | 5 of 13 | 30 of 41 | 0 of 11 | 0 of 3 | 0 of 8 |
| B | 8 of 13 | 31 of 41 | 6 of 11 | 2 of 3 | 4 of 8 |

**Main failures, with causes:**

1. **B tends to answer whenever the retrieved pages touch the topic.** Most missed CLARIFY items are cases where the excerpts discuss the
   subject, so B drafts an answer even though a key detail is missing. The same pattern explains several unsafe answers on questions that need
   a human, such as a security report: loosely related wiki pages are enough for B to answer.
2. **B's follow-ups are often generic.** The judge found that only 1 of 6 follow-ups on test (2 of 5 on dev) names the specific missing detail.
   B often asks "which part do you mean?" where the question needed a concrete detail such as an error message.
3. **A+ cannot ask and over-escalates.** It never predicted CLARIFY (0 of 11 on test). On dev it turned 9 of 13 CLARIFY items into escalations and escalated 6 of 15
   answerable questions. The risk-coverage curve for A shows that the share of answered questions that should not have been answered stays between
   56% and 73% once coverage reaches 10% on dev, so no better threshold could separate the classes.
4. **Some "failures" are correct refusals.** On two dev questions the right page was not retrieved, and B correctly said the excerpts did not
   cover the question. The error is in retrieval, not in the decision.
5. **B is noisy** between runs (section 7).

**Why hybrid search and the reranker showed no gain.** Plain vector search with cleaning already has a gold page in its top 5 for 14 of 15 answerable dev
questions (ranked first for 12), and hybrid raises that to 15 of 15, a single question, which leaves little room to improve. Hybrid's recall gain over cleaning (+0.067) is two questions improving and one getting worse. The
gold evidence was found by annotators using the union of vector and BM25 top results plus a keyword search, which favours pages those retrievers
already surface. The reranker (MiniLM) was trained on short web queries, while these questions are long emails, and it pushed the gold pages out of the top 5 on two questions (hit@5 fell from 1.00 to 0.87).
That explanation fits the numbers but was not tested separately. Nothing was tuned to force a gain.

## 9. Limitations and threats to validity

- **Small evaluation set.** Only 15 answerable dev questions and 13 answerable test questions. Differences of a few points are not claimed. Retrieval and
  judged-quality comparisons rest on very few items.
- **CLARIFY is mostly constructed.** 20 of 24 CLARIFY items were made by deleting details from real questions. They test whether a system notices
  missing information, but they are cleaner than real messy questions. Only 4 are real, so the CLARIFY conclusions are indicative.
- **Labels are LLM-drafted.** They were checked mechanically and reviewed by the owner (85% agreement on a blind subset), but they are not hand-labelled in full.
- **Evidence pooling bias.** Gold evidence came from the same kinds of retrievers that are being evaluated.
- **Vendor overlap.** The annotators and the system model are both OpenAI models. The judge is a Claude model, which reduces but does not remove self-preference
  bias. The judge is stricter than the owner and was validated on only 15 items.
- **One system model and one prompt.** B uses its first prompt version; no prompt search was done. Results are for `gpt-5-mini` and may differ with other models.
- **Gold evidence ids.** Some evidence entries had stored a chunk id instead of a page id, which understated page-level retrieval scores (25 of 85 entries on
  answerable questions). `Evidence` now canonicalises ids; dev retrieval figures above are corrected (recall@5 rose by about 0.05 to 0.07 per row), and no keep/drop decision or action metric changed. The test rows were refreshed once more, fully cached,
  so their retrieval figures are corrected too (recall@5 0.36 to 0.76, hit@5 0.46 to 0.92) while every action metric is identical; this is a third run of the test configs and a disclosed exception to the test-once rule.
- **Test used once.** Each finalist was run once on test (B three times). No decision was changed after seeing test.
- **Approximate-search fix after the test runs.** The vector index (HNSW) was found to drop a true top-5 chunk for a few questions at its default search width
  (3 of 117 questions differed from exact cosine). The width was raised from 100 to 400, after which none differ, and the dev rows were re-run (macro-F1 and the selection did not
  change; recall moved by up to 0.03). The test runs were repeated once with the corrected retriever, as a disclosed exception to the test-once rule. Only 1 of 65 test questions had different retrieval
  (`q_hz0ohskd9k`); A's and A+'s predictions were unchanged and B's changed on that question in one repeat. The tables in section 7 are from the re-run, and the
  first run's rows are kept in `results/ablation.md` (the headline figures were identical).
- **Not built:** approach C (LangGraph) and the bge reranker row, both planned as stretch items.

## 10. Reproduction and decisions

The README's "Setup and reproduction" section has the exact commands, the `.env` variables, and what works without a key. In short: `uv sync`, set the API keys in a local `.env` (never committed), then `triage ingest`, `triage collect`, `triage eval --policy ... --split dev|test`,
`triage report`, and `triage ui` for the viewer. `uv run pytest` runs the tests without any key. Rules and decisions are in `docs/decisions.md`;
deviations from the submitted one-pager (LLM-drafted labels, a second document source, constructed CLARIFY items, extra metrics) are in the README.
