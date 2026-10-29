# Support Question Triage

Decides whether to ANSWER, CLARIFY or ESCALATE an incoming support question over the Apache OFBiz wiki.

## Results

_Filled from `results/ablation.md` by script. No numbers are typed by hand._

## Failure log

_Attempts, failures and pivots, appended as they happen._

## Data and terms

_Public Apache OFBiz wiki and user mailing list; research use; sources cited; no redistribution of raw dumps._

- Wiki snapshot (OFBIZ space, fetched 2026-10-29): 705 pages, sha256 `7a14c557956971ae2f4223bda01c746f6e1b0d09caca18509886edfdf58c7556`. The snapshot file is gitignored; rerun `triage ingest --fetch-only` (the wiki may have changed since).
- Chunking (heading-based, max 500 tokens, breadcrumb prefix): 705 pages give 4,487 chunks; 63 pages have no text content. Chunk size median 131 tokens, p90 466, max 500. Most chunks are small because the wiki is split by many headings.

## Labelling disclosure

_Labels are LLM-drafted and mechanically checked, with an owner-labelled blind subset. Counts to be filled after labelling._
