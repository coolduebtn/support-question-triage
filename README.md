# Support Question Triage

For each incoming support question, decide whether to **ANSWER** from the documentation (with citations), **CLARIFY** (ask one
targeted follow-up) or **ESCALATE** to a human. Public stand-in data: the Apache OFBiz Confluence wiki and the `user@ofbiz.apache.org`
mailing list. Replies are drafts only; nothing is sent.

Hypothesis: an explicit decision step handles more questions correctly (higher macro-F1, fewer unsafe answers) without asking
unnecessary follow-ups. Approaches compared: **A** vanilla RAG (always answers), **A+** retrieval-score threshold, **B** typed LLM
decision (Pydantic AI), **C** required-detail checklist (LangGraph, stretch). Retrieval is improved one component at a time
(email cleaning, BM25 + RRF, cross-encoder reranker), each with its own scored row.

## Setup

```bash
uv sync
export OPENAI_API_KEY=... TRIAGE_CONTACT="you@example.org" TRIAGE_SYSTEM_MODEL=... TRIAGE_JUDGE_MODEL=... \
       TRIAGE_ANNOTATOR_MODELS=modelA,modelB TRIAGE_EMBED_MODEL=...
uv run triage ingest --fetch-only && uv run triage collect && uv run triage ingest --index
uv run triage label --select    # then follow docs/labelling-protocol.md; evaluate with `triage eval`, publish with `triage report`
uv run triage ask "The export is failing."
```

**Anthropic key is optional.** The only OpenAI key is required. This project's own runs used a Claude judge (`TRIAGE_JUDGE_MODEL=claude-sonnet-5-5` with `ANTHROPIC_API_KEY`), which keeps the judge from the same vendor as the system model. If `ANTHROPIC_API_KEY` is not set, the judge automatically uses the OpenAI model in `TRIAGE_JUDGE_FALLBACK_MODEL` (for example `gpt-5.4`, which must differ from `TRIAGE_SYSTEM_MODEL`) and prints a note. The judge model is recorded in every run's config, so results always say which judge produced them. Judge scores from different judges are not directly comparable.

Tests (no key needed): `uv run pytest`.

## Results

_Filled from `results/ablation.md` by `uv run triage report`. No numbers are typed by hand._

<!-- ablation:start -->
_No runs yet._
<!-- ablation:end -->

## Deviations from the submitted one-pager

- Labels are LLM-drafted and mechanically checked, with an owner-labelled blind subset, not hand-labelled in full (see Labelling disclosure).
- BM25 and the MiniLM reranker are part of the core ladder (added one at a time with a scored row each), not stretch.
- 117 questions (52 dev / 65 test, split by thread before labelling), with a second document source (the OFBiz asciidoc docs from GitHub, 204 pages) added to fix the class mix. 97 are real mailing-list questions. The other 20 are CLARIFY items made by deleting key details from real answerable questions (24 CLARIFY in all, 4 of them real); they are never used for tuning, and `--synthetic none` gives the real-only row.
- Extra metrics: unsafe-answer rate, over-ask and over-escalation rates, judged correctness / citation support / follow-up targeting.

## Limitations

- Small evaluation set: with 40 dev and 57 test questions the minimum detectable paired difference is roughly 20 points (computed in `triage.evals.stats.min_detectable_effect`), so smaller differences are reported as "no measurable gain".
- OpenAI-only models: annotators, system model and judge share a vendor, so self-preference bias is possible. The judge is a different, stronger model than the system model.
- Reference answers are LLM-written and judged by an LLM; the judge is validated on a small owner-graded set (report-only, n stated).
- Annotator evidence comes from the union of vector and BM25 results plus keyword search, which reduces but does not remove bias toward easily retrievable pages.

## Failure log

Things that went wrong, what I found, and what changed. Numbers are from the runs in `results/`.

1. **Class mix too skewed.** With the wiki only, 58 of 75 candidate questions were ESCALATE and only 2 were CLARIFY. Fix: added the OFBiz docs from GitHub plus 22 short, vague real questions. Result: 97 questions, but CLARIFY is still rare (4 real items in gold: 1 dev, 3 test). The plan's gate (every class at least 12) **still fails for CLARIFY**. I proceeded and report CLARIFY results as indicative only.
2. **Annotator disagreement.** The first draft gave Cohen's kappa 0.31 between the two annotators. After fixing the evidence format (real chunk ids) and the prompts, the second draft gave 0.46 (raw agreement 0.70). `gpt-4.1` almost never chose CLARIFY, which drives most disagreement. The first draft is archived, not deleted.
3. **Personal data in committed files.** A code review found phone numbers, a street address, a LinkedIn link and first names in `data/gold`. I strengthened the scrubbing, gitignored `data/gold` except `gold_public.jsonl` (no raw email text), and rewrote git history to purge it.
4. **Bugs found in review (all fixed test-first):** policy errors were not counted as wrong predictions; the threshold tuner ignored the clean flag; judge errors were counted as failures; `label --select` could overwrite frozen splits; the config hash omitted the embedding model and gold hash; one bad item crashed label drafting; an empty structured response was cached.
5. **Owner blind check.** Owner labels versus final LLM labels on 20 blind items: 17 agree (85%, Wilson 95% CI 0.64 to 0.95, kappa 0.44). 18 of the 20 were ESCALATE, so kappa is low even at 85% agreement. The later review of 43 queue items confirmed all 43 (0 overrides). That is an upper bound, not an independent measure, because the owner saw the drafts.
6. **Vanilla RAG is unsafe by construction.** A always answers, so its unsafe-answer rate is 1.00 and macro-F1 0.18. Its stratified bootstrap CI is [0.18, 0.18] because the bootstrap conditions on class counts and a constant predictor has no variance. This is not a precise estimate; it is a degenerate one.
7. **Threshold (A+) only partly helps.** Macro-F1 0.34, unsafe-answer 0.52. A single similarity score cannot tell "undocumented" from "missing detail", so it escalates but never clarifies.
8. **Reranker lowered recall.** On the 15 dev answer questions, recall@5 went 0.81 (hybrid) to 0.70 (+MiniLM), MRR 0.74 to 0.63. The change is inside noise ([-0.33, +0.07]), so I report "no measurable gain" and the pre-registered rule drops it. Likely cause: MiniLM is trained on web search and reorders good wiki and mailing-list matches. The bge reranker row was not run.
9. **BM25 and cleaning gains are inside noise too.** Email cleaning +0.07 recall [0.00, +0.20]; BM25/RRF +0.08 [-0.07, +0.27]. With 15 answer items the minimum detectable effect is about 0.2 or more, so neither is a proven improvement.
10. **API spend limit.** The project's OpenAI spend limit was hit mid-run (HTTP 429). The hybrid row had 12 errors and the reranker row 40 of 40, so their generation and judge columns are **invalid** and marked so. Retrieval metrics for those rows are valid (retrieval ran first and the retrieved pages were saved), so the retrieval-only ladder was built from cached embeddings with no LLM calls.
11. **Judge is stricter than the owner.** On 15 owner-graded real drafts the judge agreed 11 of 15 (73%), kappa 0.48 (CI 0.23 to 0.84). All 4 disagreements were the judge saying "incorrect" where the owner said "correct". Judged-correct rates are therefore conservative, and the ranking of policies is more reliable than the absolute levels. n=15, so this is report-only.
12. **Cost driver.** The `gpt-5.5` judge and adjudicator were the most expensive part of the run (the judge alone was about $8 of about $13 nominal, by the local cache tally). Choosing a cheaper judge earlier would have avoided hitting the limit.

_Still to add after the remaining runs: test-split results, error table, risk-coverage curve, probe results, and the final selection._

## Data and terms

_Public Apache OFBiz wiki and user mailing list; research use; sources cited; no redistribution of raw dumps._

- Wiki snapshot (OFBIZ space, fetched 2026-10-29): 705 pages, sha256 `7a14c557956971ae2f4223bda01c746f6e1b0d09caca18509886edfdf58c7556`. The snapshot file is gitignored; rerun `triage ingest --fetch-only` (the wiki may have changed since).
- Docs snapshot (OFBiz asciidoc sources from the apache/ofbiz-framework GitHub repository, via sparse clone, no crawling): 204 pages. The docs site itself was not crawled because its robots rules disallow bots.
- Chunking (heading-based, max 500 tokens, breadcrumb prefix): wiki plus docs give 5,397 chunks in the index. Most chunks are small because pages are split by many headings.
- Questions: 259 mailing-list threads collected, PII scrubbed (emails, phones, links, IPs, known names); 97 selected for the gold set.

## Labelling disclosure

Labels are **not hand-labelled in full**. Process: two blind LLM annotators (`gpt-5.4`, `gpt-4.1`) label each question from retrieved evidence without seeing replies; code checks every evidence quote against the frozen snapshot; disagreements go to an adjudicator (`gpt-5.5`); the owner labelled a blind subset first and then reviewed a queue.

| Source of final label | Count |
|---|---|
| Owner blind (labelled before seeing any LLM label) | 20 |
| Owner review (disputed, evidence-failed or audit items; 0 overrides) | 43 |
| LLM annotators agreed, not reviewed | 34 |
| **Total** | 97 |

- Final class counts: 65 escalate, 28 answer, 4 clarify.
- Owner blind versus final labels: 17 of 20 agree (85%, Wilson 95% CI 0.64 to 0.95, kappa 0.44).
- Annotator agreement: kappa 0.46 (raw 0.70) on 97 questions.
- Owner review confirmed 43 of 43; this is an upper bound because the owner saw the drafts.
- **Invented CLARIFY items:** 20 of the 24 CLARIFY questions are real questions with key details removed, so CLARIFY results rest mostly on constructed cases. They are flagged `synthetic` in the data and excluded from threshold tuning.
