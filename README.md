# Support Question Triage

For each incoming support question, decide whether to **ANSWER** from the documentation (with citations), **CLARIFY** (ask one
targeted follow-up) or **ESCALATE** to a human. Public stand-in data: the Apache OFBiz Confluence wiki and the `user@ofbiz.apache.org`
mailing list. Replies are drafts only; nothing is sent.

Hypothesis: an explicit decision step handles more questions correctly (higher macro-F1, fewer unsafe answers) without asking
unnecessary follow-ups. Approaches compared: **A** vanilla RAG (always answers), **A+** retrieval-score threshold, **B** typed LLM
decision (Pydantic AI), **C** required-detail checklist (LangGraph, stretch). Retrieval is improved one component at a time
(email cleaning, BM25 + RRF, cross-encoder reranker), each with its own scored row.

## Setup and reproduction

Requirements: Python 3.11 or newer, [uv](https://docs.astral.sh/uv/), and `git` (the docs snapshot is a sparse clone of the OFBiz repository).

### 1. Look at the results (no key, no data download)

```bash
uv sync
uv run pytest                                  # 252 tests
uv run triage ui                               # viewer at http://localhost:8501 (Methodology, Compare, Scoreboard, Where it fails)
uv run triage report                           # rebuilds the table in this README from results/ablation.md
uv run python scripts/test_comparisons.py      # paired comparisons and B's repeat stability on test, from saved runs
```

All of this works on a fresh clone. It reads the committed runs in `results/runs/`, the cleaned public gold file `data/gold/gold_public.jsonl` and `data/page_titles.json`. Two limits of the public files: the 20 constructed CLARIFY questions have no text in them, so the viewer shows a placeholder for those, and the "Try your own" tab needs a key and the index (steps 2 and 3). `scripts/failure_analysis.py` and `scripts/ui_audit.py` need the private full gold file, which is not in the repository (it holds raw question text). The output of the first is committed as `results/failure_analysis.md`.

### 2. Configure keys

Put these in a `.env` file in the project root (gitignored; never commit it or paste a key into the chat or into code). Values already in the environment win.

```bash
OPENAI_API_KEY=...                      # required for indexing, eval and ask
ANTHROPIC_API_KEY=...                   # optional: only for the Claude judge
TRIAGE_CONTACT=you@example.org          # used in the crawler User-Agent; use an address you are happy to expose to Apache
TRIAGE_SYSTEM_MODEL=gpt-5-mini          # answers and decisions (A, A+, B)
TRIAGE_EMBED_MODEL=text-embedding-3-small
TRIAGE_JUDGE_MODEL=claude-sonnet-5-5    # must differ from the system model
TRIAGE_JUDGE_FALLBACK_MODEL=gpt-5.4     # used when there is no ANTHROPIC_API_KEY
TRIAGE_ANNOTATOR_MODELS=gpt-5.4,gpt-4.1 # only for `triage label`
```

**The Anthropic key is optional.** This project's own runs used a Claude judge (`claude-sonnet-5-5`), which keeps the judge from a different vendor than the system model. Without `ANTHROPIC_API_KEY` the judge automatically uses `TRIAGE_JUDGE_FALLBACK_MODEL` (it must differ from `TRIAGE_SYSTEM_MODEL`) and prints a note. The judge model is recorded in every run's config. Scores from different judges are not directly comparable.

### 3. Build the corpus and the index

```bash
uv run triage ingest --fetch-only    # wiki snapshot, 705 pages (polite crawl, about one request a second)
uv run triage ingest --fetch-docs    # OFBiz asciidoc docs, 204 pages (sparse git clone)
uv run triage collect                # mailing-list threads, PII-scrubbed, into data/snapshot/ (gitignored)
uv run triage ingest --index         # chunk and embed (5,397 chunks), writes data/index/ and data/page_titles.json
```

The wiki and the list keep changing, so a fresh crawl will not be byte-identical to the snapshot behind the reported numbers (hashes are in "Data and terms" below). Expect small differences.

### 4. Ask a question, re-run the experiments

```bash
uv run triage ask "The export is failing."                  # approach B on one question; prints action, reason, citations, draft

# dev ladder (each command appends a scored row to results/ablation.md)
uv run triage eval --policy A  --split dev --judge --step "A (dev)"
uv run triage eval --policy A+ --split dev --judge --step "A+ (dev)"          # tunes the threshold on dev
uv run triage eval --policy B  --split dev --judge --step "B (dev)"
uv run triage eval --policy A  --split dev --judge --clean --step "A + cleaning (dev)"
uv run triage eval --policy A  --retriever hybrid --split dev --judge --clean --step "A + BM25/RRF (dev)"
uv run triage eval --policy A  --retriever rerank-minilm --split dev --judge --clean --step "A + reranker (dev)"
uv run triage ladder                                        # retrieval-only ladder with paired intervals (no LLM calls)
uv run triage compare RUN_ID_A RUN_ID_B                     # paired comparison of two runs

# test: once per finalist (the runner refuses to repeat a test config unless --allow-test-rerun)
uv run triage eval --policy A  --split test --judge --decision final
uv run triage eval --policy A+ --split test --judge --decision final
uv run triage eval --policy B  --split test --judge --repeats 3 --decision final
uv run triage report
```

`triage eval` uses the full gold file if you have it and otherwise the public one, printing a note. On the public file (97 real questions, no constructed CLARIFY items) the numbers will differ from the tables below. Each eval run is saved under `results/runs/<run id>/` (config, predictions, metrics) and the answering step costs about $0.003 per question with `gpt-5-mini` (the judge adds some). Identical calls are cached on disk, so repeating a run is nearly free. Run `--synthetic none` to score the real questions only.

Labelling (`triage label ...`, then `triage judge --make-sheet / --validate`) is a hand-in-the-loop process that needs the owner's blind grades; the steps are in `docs/labelling-protocol.md` and the owner's decisions are in "Labelling disclosure" below. It is not meant to be re-run by a reviewer.

## Results

**B, the typed decision, is the best of the three approaches.** It cuts unsafe answers from 1.00 (plain RAG) to 0.19 on the test set, and it is the only approach that ever asks a follow-up. It is not perfect: its follow-ups are often generic and it is noisy between runs. Details and caveats are in `docs/documentation.md` and the failure log below.

Test set (65 questions: 13 ANSWER, 11 CLARIFY, 41 ESCALATE; run once per approach, B three times, first repeat shown):

| Approach | Macro-F1 [95% CI] | Unsafe-answer | Over-ask | Judged correct | Cost/q |
|---|---|---|---|---|---|
| A: plain RAG | 0.11 [0.11, 0.11] | 1.00 | 0.00 | 0.62 | $0.0028 |
| A+: similarity threshold | 0.33 [0.25, 0.42] | 0.25 | 0.00 | 0.80 | $0.0007 |
| **B: typed decision** | **0.62 [0.48, 0.74]** | **0.19** | 0.15 | 0.75 | $0.0027 |

Across B's three test runs macro-F1 was 0.62, 0.49 and 0.57 (mean 0.56), so the single-run figure is the best of the three. Paired differences on test: B vs A +0.51 [+0.37, +0.64]; A+ vs A +0.22 [+0.14, +0.31]; B vs A+ +0.28 [+0.12, +0.44] (exact McNemar p = 0.087, so this one is weaker than the other two). On the 52 dev questions B was selected by the pre-registered rule (B vs A+ +0.36 [+0.21, +0.51]).

Retrieval upgrades, scored one at a time on the 15 answerable dev questions: email cleaning (recall@5 0.74 to 0.81), BM25 with RRF (to 0.88) and the MiniLM reranker (to 0.75, lower ranking quality) **showed no measurable gain** under the pre-registered rule, so A+ and B use plain vector search. See failure-log entry 13 for why.

Class counts, the real-versus-constructed split of CLARIFY, and every caveat are in the sections below. The viewer shows all runs side by side: `uv run triage ui`. Full scored table (generated by `uv run triage report` from `results/ablation.md`; it holds the current dev rows and the current test rows. Earlier rows are in `results/ablation_history.md`; see entries 14 and 18 for why they were re-run):

<!-- ablation:start -->
| step | split | config | macro-F1 [95% CI] | answer F1 (n) | clarify F1 (n) | escalate F1 (n) | unsafe-answer | over-ask | recall@k | judged correct | citation support | follow-up targeting | cost/q (USD) | latency p50 (s) | errors | decision | notes | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A vector baseline (dev) | dev | caa31fb7 | 0.15 [0.15, 0.15] | 0.45 (n=15) | 0.00 (n=13) | 0.00 (n=24) | 1.00 | 0.00 | 0.74 | 0.47 | 0.70 | n/a | 0.0026 | n/a | 0 | baseline |  | 20261101-201233-caa31fb7-dev |
| A+ threshold (dev) | dev | dbb65c21 | 0.29 [0.20, 0.38] | 0.44 (n=15) | 0.00 (n=13) | 0.44 (n=24) | 0.46 | 0.00 | 0.74 | 0.44 | 0.70 | n/a | 0.0014 | 0.00 | 0 | baseline |  | 20261101-201235-dbb65c21-dev |
| B typed decision (dev) | dev | a0ce2387 | 0.66 [0.51, 0.78] | 0.63 (n=15) | 0.53 (n=13) | 0.81 (n=24) | 0.30 | 0.00 | 0.74 | 0.50 | 0.78 | 0.40 | 0.0025 | n/a | 0 | baseline |  | 20261101-201238-a0ce2387-dev |
| A + email cleaning (dev) | dev | 9544a29b | 0.15 [0.15, 0.15] | 0.45 (n=15) | 0.00 (n=13) | 0.00 (n=24) | 1.00 | 0.00 | 0.81 | 0.67 | 0.73 | n/a | 0.0026 | n/a | 0 | baseline |  | 20261101-201240-9544a29b-dev |
| A + BM25/RRF (dev) | dev | b0ddb1bb | 0.15 [0.15, 0.15] | 0.45 (n=15) | 0.00 (n=13) | 0.00 (n=24) | 1.00 | 0.00 | 0.88 | 0.60 | 0.68 | n/a | 0.0028 | n/a | 0 | baseline |  | 20261101-201242-b0ddb1bb-dev |
| A + MiniLM reranker (dev) | dev | f122adb7 | 0.15 [0.15, 0.15] | 0.45 (n=15) | 0.00 (n=13) | 0.00 (n=24) | 1.00 | 0.00 | 0.75 | 0.47 | 0.71 | n/a | 0.0027 | n/a | 0 | baseline |  | 20261101-201250-f122adb7-dev |
| A vector baseline (test, refreshed after evidence-id fix) | test | caa31fb7 | 0.11 [0.11, 0.11] | 0.33 (n=13) | 0.00 (n=11) | 0.00 (n=41) | 1.00 | 0.00 | 0.76 | 0.62 | 0.67 | n/a | 0.0028 | n/a | 0 | final |  | 20261101-201827-caa31fb7-test |
| A+ threshold (test, refreshed after evidence-id fix) | test | dbb65c21 | 0.33 [0.25, 0.42] | 0.32 (n=13) | 0.00 (n=11) | 0.68 (n=41) | 0.25 | 0.00 | 0.76 | 0.80 | 0.71 | n/a | 0.0007 | 0.00 | 0 | final |  | 20261101-201829-dbb65c21-test |
| B typed decision (test, refreshed after evidence-id fix, 3 repeats) | test | a0ce2387 | 0.62 [0.48, 0.74] | 0.52 (n=13) | 0.52 (n=11) | 0.82 (n=41) | 0.19 | 0.15 | 0.76 | 0.75 | 0.70 | 0.17 | 0.0027 | n/a | 0 | final |  | 20261101-201832-a0ce2387-test |
<!-- ablation:end -->

## Deviations from the submitted one-pager

- Labels are LLM-drafted and mechanically checked, with an owner-labelled blind subset, not hand-labelled in full (see Labelling disclosure).
- BM25 and the MiniLM reranker are part of the core ladder (added one at a time with a scored row each), not stretch.
- 117 questions (52 dev / 65 test, split by thread before labelling), with a second document source (the OFBiz asciidoc docs from GitHub, 204 pages) added to fix the class mix. 97 are real mailing-list questions. The other 20 are CLARIFY items made by deleting key details from real answerable questions (24 CLARIFY in all, 4 of them real); they are never used for tuning, and `--synthetic none` gives the real-only row.
- Extra metrics: unsafe-answer rate, over-ask and over-escalation rates, judged correctness / citation support / follow-up targeting.
- Approach C (LangGraph) and the bge reranker row were planned as stretch items and were **not built**.
- The judge is a Claude model (`claude-sonnet-5-5`), validated against the owner on 15 items (67% agreement); the first judge was `gpt-5.5` (73%). See entry 11.
- Test was re-run once after a vector-search fix, a disclosed exception to the test-once rule (entry 14).

## Limitations

- Small evaluation set: with 52 dev and 65 test questions the minimum detectable paired difference is about 0.19 (dev) and 0.17 (test), computed in `triage.evals.stats.min_detectable_effect`, so smaller differences are reported as "no measurable gain". Only 15 dev and 13 test questions are answerable, so retrieval and judged-quality comparisons rest on very few items.
- Vendor overlap: the annotators and the system model are OpenAI models, so self-preference bias is possible. The judge is a Claude model (a different vendor), which reduces but does not remove it.
- CLARIFY evidence is mostly constructed: 20 of the 24 CLARIFY items were made by deleting details from real questions, so CLARIFY conclusions are indicative.
- Reference answers are LLM-written and judged by an LLM; the judge is validated on a small owner-graded set (report-only, n stated).
- Annotator evidence comes from the union of vector and BM25 results plus keyword search, which reduces but does not remove bias toward easily retrievable pages.

## Failure log

Things that went wrong, what I found, and what changed. Numbers are from the runs in `results/`.

1. **Class mix too skewed.** With the wiki only, 58 of 75 candidate questions were ESCALATE and only 2 were CLARIFY. Fix: added the OFBiz docs from GitHub plus 22 short, vague real questions. Result: 97 questions, but CLARIFY is still rare (4 real items in gold: 1 dev, 3 test). The plan's gate (every class at least 12) **still fails for CLARIFY**. I proceeded and report CLARIFY results as indicative only.
2. **Annotator disagreement.** The first draft gave Cohen's kappa 0.31 between the two annotators. After fixing the evidence format (real chunk ids) and the prompts, the second draft gave 0.46 (raw agreement 0.70). `gpt-4.1` almost never chose CLARIFY, which drives most disagreement. The first draft is archived, not deleted.
3. **Personal data in committed files.** A code review found phone numbers, a street address, a LinkedIn link and first names in `data/gold`. I strengthened the scrubbing, gitignored `data/gold` except `gold_public.jsonl` (the labels, evidence and the PII-scrubbed question text for the 97 real questions; the raw emails and the constructed CLARIFY items are not in it), and rewrote git history to purge it.
4. **Bugs found in review (all fixed test-first):** policy errors were not counted as wrong predictions; the threshold tuner ignored the clean flag; judge errors were counted as failures; `label --select` could overwrite frozen splits; the config hash omitted the embedding model and gold hash; one bad item crashed label drafting; an empty structured response was cached.
5. **Owner blind check.** Owner labels versus final LLM labels on 20 blind items: 17 agree (85%, Wilson 95% CI 0.64 to 0.95, kappa 0.44). 18 of the 20 were ESCALATE, so kappa is low even at 85% agreement. The later review of 43 queue items confirmed all 43 (0 overrides). That is an upper bound, not an independent measure, because the owner saw the drafts.
6. **Vanilla RAG is unsafe by construction.** A always answers, so its unsafe-answer rate is 1.00 and macro-F1 0.15 on dev (0.11 on test). Its stratified bootstrap CI is [0.15, 0.15] on dev because the bootstrap conditions on class counts and a constant predictor has no variance. This is not a precise estimate; it is a degenerate one.
7. **Threshold (A+) only partly helps.** Macro-F1 0.29 on dev (0.33 on test), unsafe-answer 0.46 on dev (0.25 on test). A single similarity score cannot tell "undocumented" from "missing detail", so it escalates but never clarifies: it got 0 of 13 CLARIFY items on dev and 0 of 11 on test, and it escalated 6 of 15 answerable dev questions. The risk-coverage curve for A shows the share of answered questions that should not have been answered stays between 56% and 73% once coverage reaches 10%, so a better threshold would not fix this.
8. **Reranker lowered recall.** On the 15 dev answer questions, recall@5 went 0.88 (hybrid) to 0.75 (+MiniLM), MRR 0.91 to 0.77. The change is inside noise ([-0.34, +0.06]), so I report "no measurable gain" and the pre-registered rule drops it. Likely cause: MiniLM is trained on web search and reorders good wiki and mailing-list matches. The bge reranker row was not run.
9. **BM25 and cleaning gains are inside noise too.** Email cleaning +0.07 recall [0.00, +0.20]; BM25/RRF +0.07 [-0.07, +0.23]. With 15 answer items the minimum detectable effect is about 0.2 or more, so neither is a proven improvement.
10. **API spend limit.** The project's OpenAI spend limit was hit mid-run (HTTP 429). The hybrid row had 12 errors and the reranker row 40 of 40, so their generation and judge columns were **invalid** and marked so. Retrieval metrics for those rows were valid, so the retrieval-only ladder was built from cached embeddings with no LLM calls. Once the limit was lifted, all dev rows were re-run on the merged question set with 0 errors.
11. **Judge is stricter than the owner.** On 15 owner-graded real drafts the first judge (`gpt-5.5`) agreed 11 of 15 (73%), kappa 0.48 (CI 0.23 to 0.84). The final judge, `claude-sonnet-5-5`, agreed 10 of 15 (67%), kappa 0.39 (CI 0.17 to 0.71); the difference is one item. In both cases every disagreement was the judge saying "incorrect" where the owner said "correct", and neither accepted a wrong answer. Judged-correct rates are therefore conservative, and the ranking of policies is more reliable than the absolute levels. n=15, so this is report-only.
12. **Cost driver.** The `gpt-5.5` judge and adjudicator were the most expensive part of the run (the judge alone was about $8 of about $13 nominal, by the local cache tally). Choosing a cheaper judge earlier would have avoided hitting the limit; the final results use a cheaper Claude judge.

13. **Hybrid search and the reranker did not show a proven gain, and we looked into why.** On the 15 answerable dev questions, recall@5 went 0.74 (raw vector) to 0.81 (+ cleaning) to 0.88 (+ BM25/RRF), but each step's paired interval includes zero (the cleaning step's lower bound is exactly 0.00), and the reranker lowered it to 0.75. Causes found from the per-question table:
    - *Little headroom.* Plain vector search with cleaning already has a gold page in its top 5 for 14 of 15 questions (ranked first for 12). Hybrid raises that to 15 of 15, which is one question. Hybrid and reranking mainly help when the first retriever misses often, and here it does not (one wiki plus the docs, questions with specific OFBiz terms).
    - *The recall change is a few questions.* Hybrid's recall gain over cleaning (+0.067) is two questions improving (`q_o5yxnf1txb`, `q_rqgw8xtl5z`) and one getting worse (`q_mffxrrfxtr`). Its MRR rose from 0.84 to 0.91, but that is also a handful of questions and was not tested separately. That is noise at n=15.
    - *The gold evidence was found with these retrievers.* The annotators' candidate pages were the union of vector and BM25 top-10 results plus a keyword search (`src/triage/labelling/annotate.py`), so a page neither retriever surfaces was unlikely to become gold. This favours the existing retrievers and makes any new one harder to credit.
    - *Possible fit problem for the reranker:* MiniLM was trained on short web queries, while these questions are long emails, and it pushed the gold pages out of the top 5 on two questions, `q_jnnpvjdofl` and `q_o5yxnf1txb` (hit@5 fell from 1.00 to 0.87). This fits the numbers but was not tested separately.
    This is "no proven gain on this benchmark", not "these methods do not work". We did not tune anything to force a gain.

14. **The approximate vector search sometimes dropped a true top-5 chunk.** Found while reconciling two MRR figures (0.74 from a live ladder run, 0.77 from a saved run): Chroma's HNSW index, at its default `ef_search` of 100, returned different top-5 results for a few questions in different processes (for one question a chunk with cosine 0.4035 appeared in one process and not in two others). Against exact brute-force cosine on all 117 questions, 3 differed at the default setting, and 0 differed at `ef_search` 400 in repeated fresh processes. The setting is now 400 and is applied when an index is built or opened (tests added). HNSW is kept so the design still scales to a larger wiki. Dev rows were re-run after the fix (macro-F1 and the selection did not change; recall moved by up to 0.03). Test was re-run once with the corrected retriever, a disclosed exception to the test-once rule: only 1 of 65 test questions had different retrieval, A's and A+'s predictions did not change, B changed on that question in one repeat, and every headline figure was identical. Both sets of rows are in `results/ablation.md`.

15. **B is noisy between runs.** Over three runs on the test set, B's macro-F1 was 0.62, 0.49 and 0.57 (mean 0.56, spread 0.13), and it chose the same action in all three runs for 44 of 65 questions. A single-run figure overstates B, and I report the mean next to it. The dev-to-test drop (0.66 to 0.62 on the first repeat, 0.56 on the mean) is real.
16. **B answers whenever the retrieved pages touch the topic.** B was wrong on 16 of 52 dev questions and 20 of 65 test questions. On dev, all 7 missed CLARIFY items are constructed ones where the excerpts discuss the subject so B drafts an answer despite a missing detail. Several unsafe answers are questions that need a human (a security report, a request for guidance) where loosely related wiki pages were enough. Error table: `results/failure_analysis.md`.
17. **B's follow-ups are often generic.** The judge found that only 1 of 6 follow-ups on test (2 of 5 on dev) names the specific missing detail; B often asks "which part do you mean?" where the question needed something concrete, such as an error message. B asks, and is right to ask, more often than the baselines (6 of 11 CLARIFY items on test, including 2 of the 3 real ones), but the quality of what it asks is the weak spot.

18. **Some gold evidence pointed at chunks, not pages, so retrieval scores were too low.** Found by the UI audit, which showed evidence "pages" with no title. In 25 of the 85 evidence entries on answerable questions (9 of 28 questions) the annotators had cited a chunk id (`<page_id>-<n>`) where a page id was expected, so page-level recall, hit and MRR could never match them. All 25 map to a real page by dropping the suffix. Fix: `Evidence` now canonicalises page ids when it is built (page ids never end in `-<digits>`, checked against all 845 pages), so every consumer gets the right id; the frozen gold file was not edited. Effect on dev retrieval: recall@5 0.69 to 0.74 (vector), 0.76 to 0.81 (+ cleaning), 0.81 to 0.88 (+ BM25), 0.70 to 0.75 (+ reranker); no keep/drop decision changed and no action metric (macro-F1, unsafe-answer, over-ask) is affected. The dev rows were re-run (fully cached, no new model calls) and the older rows are in `results/ablation_history.md`. The test rows were refreshed once more for the same reason (also fully cached: identical predictions and action metrics, only the retrieval columns changed, test recall@5 0.36 to 0.76 and hit@5 0.46 to 0.92). That is a third run of the test configs, a disclosed exception to the test-once rule, and it made no new model calls.

## Data and terms

_Public Apache OFBiz wiki and user mailing list; research use; sources cited; no redistribution of raw dumps._

Sources: [OFBiz Confluence wiki](https://cwiki.apache.org/confluence/display/OFBIZ) (space `OFBIZ`), [OFBiz documentation sources](https://github.com/apache/ofbiz-framework) (apache/ofbiz-framework), [`user@ofbiz.apache.org` archive](https://lists.apache.org/list.html?user@ofbiz.apache.org).

- Wiki snapshot (OFBIZ space, fetched 2026-10-29): 705 pages, sha256 `7a14c557956971ae2f4223bda01c746f6e1b0d09caca18509886edfdf58c7556`. The snapshot file is gitignored; rerun `triage ingest --fetch-only` (the wiki may have changed since).
- Docs snapshot (OFBiz asciidoc sources from the apache/ofbiz-framework GitHub repository, via sparse clone, no crawling): 204 pages. The docs site itself was not crawled because its robots rules disallow bots.
- Chunking (heading-based, max 500 tokens, breadcrumb prefix): wiki plus docs give 5,397 chunks from 845 pages (63 of the 705 wiki pages and 1 of the 204 docs pages had no text, so they produced no chunks). Most chunks are small because pages are split by many headings.
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
