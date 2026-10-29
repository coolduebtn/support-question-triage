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

Tests (no key needed): `uv run pytest`.

## Results

_Filled from `results/ablation.md` by `uv run triage report`. No numbers are typed by hand._

<!-- ablation:start -->
_No runs yet._
<!-- ablation:end -->

## Deviations from the submitted one-pager

- Labels are LLM-drafted and mechanically checked, with an owner-labelled blind subset, not hand-labelled in full (see Labelling disclosure).
- BM25 and the MiniLM reranker are part of the core ladder (added one at a time with a scored row each), not stretch.
- Dev set is 30 questions instead of about 20, so there are enough items per class to tune on.
- Extra metrics: unsafe-answer rate, over-ask and over-escalation rates, judged correctness / citation support / follow-up targeting.

## Limitations

- Small evaluation set: with about 30 to 45 test questions the minimum detectable paired difference is roughly 20 points (computed in `triage.evals.stats.min_detectable_effect`), so smaller differences are reported as "no measurable gain".
- OpenAI-only models: annotators, system model and judge share a vendor, so self-preference bias is possible. The judge is a different, stronger model than the system model.
- Reference answers are LLM-written and judged by an LLM; the judge is validated on a small owner-graded set (report-only, n stated).
- Annotator evidence comes from the union of vector and BM25 results plus keyword search, which reduces but does not remove bias toward easily retrievable pages.

## Failure log

_Attempts, failures and pivots, appended as they happen._

## Data and terms

_Public Apache OFBiz wiki and user mailing list; research use; sources cited; no redistribution of raw dumps._

- Wiki snapshot (OFBIZ space, fetched 2026-10-29): 705 pages, sha256 `7a14c557956971ae2f4223bda01c746f6e1b0d09caca18509886edfdf58c7556`. The snapshot file is gitignored; rerun `triage ingest --fetch-only` (the wiki may have changed since).
- Chunking (heading-based, max 500 tokens, breadcrumb prefix): 705 pages give 4,487 chunks; 63 pages have no text content. Chunk size median 131 tokens, p90 466, max 500. Most chunks are small because the wiki is split by many headings.

## Labelling disclosure

_Labels are LLM-drafted and mechanically checked, with an owner-labelled blind subset. Counts to be filled after labelling._
