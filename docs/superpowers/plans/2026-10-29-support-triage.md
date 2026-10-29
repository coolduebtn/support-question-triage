# Support Question Triage Implementation Plan (revision 3, after two independent reviews)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a triage system that decides ANSWER / CLARIFY / ESCALATE for support questions over the Apache OFBiz wiki, adding retrieval and decision components one at a time with a score table row after each.

**Architecture:** Plain-Python ingestion and retrieval shared by every approach. Only the decision policy differs (A, A+, B, C behind one `Policy` interface). One eval runner produces per-item predictions, metrics with confidence intervals, judge scores and a row in `results/ablation.md`. Gold labels come from a mechanically checked two-annotator LLM protocol plus a blind owner-labelled subset, and are described as such everywhere.

**Tech Stack:** Python 3.11+, uv, chromadb 1.5.x, bm25s, sentence-transformers (local cross-encoder), openai 3.x, pydantic-ai 2.x, langgraph 1.2.x, scikit-learn, scipy, numpy, httpx, beautifulsoup4, tiktoken, mail-parser-reply, matplotlib, pytest, streamlit (stretch).

**Spec:** `docs/design/2026-10-29-support-triage-design.md` (also published as a page). Submitted one-pager: `Capstone_Support_Question_Triage.pdf`. Course rubric: `Capstone Project Specification_ AI Engineering Cohort.pdf`. Review findings this revision addresses: `scratchpad/plan_review.md` (summarised in `docs/decisions.md`, Task 1).

## Core path vs stretch

Core, in this order: Tasks 1 to 16. Inside core, the **minimum comparison** is Tasks 1 to 13 plus 15 and 16: A, judge, A+ and B on the vector baseline with a final selection. The retrieval ladder (Task 14) is core by the owner's decision, but it runs after the policy comparison exists, so a time overrun can never leave the central comparison missing. Stretch (only after core evaluation is complete): Task 17 (C with LangGraph), Task 18 (Streamlit), the bge-reranker row (14d).

## Phases and owner time

Dates are not fixed (the deadline was extended); phases are ordered by dependency. Owner touch-points are consolidated into three sessions:
- **Phase 1, offline (no key):** Tasks 1 to 7. No owner time.
- **Phase 2, key day:** Task 5 smoke test, Task 6 index build, Task 8 draft labelling. **Owner session S1 (about 45 minutes):** blind-label 20 questions (minimum 12), then review the queue.
- **Phase 3, baselines:** Tasks 9 to 13. **Owner session S2 (about 30 minutes):** grade about 20 real system drafts for the judge validation (after Task 9 produces drafts).
- **Phase 4, ladder and finish:** Tasks 14, 15, 16. **Owner session S3 (about 20 minutes):** confirm error-table cause categories.
- Stretch only afterwards.

## Global Constraints

- Python `>=3.11`; dependencies managed with `uv`; all commands run through `uv run`.
- Data: Apache OFBiz Confluence space `OFBIZ` via `https://cwiki.apache.org/confluence/rest/api/space/OFBIZ/content/page`; questions from `lists.apache.org` (`list=user`, `domain=ofbiz.apache.org`). Second corpus `docs.ofbiz.apache.org` is added only if Task 10 (class-mix gate) says so.
- Crawl etiquette: single thread, 1 request/second, User-Agent contact read from env `TRIAGE_CONTACT` (never committed), cache every response in `data/raw/`, crawl once.
- Privacy and terms: no emails, sender names or raw thread dumps in any committed file. Committed data is limited to the gold questions (after scrub and a manual read) and hashes. Wiki snapshot and raw threads are gitignored; the snapshot sha256 is recorded so the run is auditable. README has a "Data and terms" section: public archives, research use, sources cited, no redistribution of raw dumps.
- Questions are verbatim from the list apart from PII scrubbing. No model writes or paraphrases a question.
- Same system model and same retrieval within each policy comparison (A, A+, B, C). The judge is a stronger model than the system model. The key is OpenAI-only, so cross-vendor independence is impossible; annotators are two different OpenAI models, and the README states self-preference bias as a limitation.
- Chroma collections use cosine distance set explicitly; `score = 1 - distance`. Every `Retrieved` carries `score_kind in {"cosine","rrf","cross_encoder"}`; scores of different kinds are never compared.
- Metrics: macro-F1 over `labels=["answer","clarify","escalate"]`; a prediction of `error` (API failure, invalid output after retries) is always counted wrong and reported as `n_errors`.
- Split: by thread, fixed by seed before any annotation, about 30 dev and the rest (about 30 to 50) test. After labelling and before any system run, the split may be redrawn once, logged in `docs/decisions.md`, if any class has fewer than 5 dev items. Test is never used for thresholds, few-shot examples or prompt iteration.
- Keep/drop rule for components: a row "improves" only if the paired bootstrap 95% interval of the difference from the previous row excludes zero on dev items; otherwise it is recorded as "no measurable gain". Ladder decisions use dev items only; pooled dev+test recall is reported afterwards as description. Selection rule (pre-registered in `docs/decisions.md` before any test run): among configurations whose dev over-ask rate and cost per question are within the limits in `configs/limits.toml`, choose the best dev macro-F1; if two are within the minimum detectable effect, choose the simpler one. Test is then run once per finalist for confirmation and reported; the choice is not changed after seeing test.
- Config freezing: `config_hash` covers prompt texts, snapshot hash, index hash, retrieval settings, model names, thresholds and seeds. The git commit is recorded in `config.json` but not hashed, so unrelated commits between the dev and test runs do not invalidate the freeze.
- A+ thresholds on `Retrieved.vector_cosine` (the vector cosine of the result's first-stage score, kept by every retriever), so fusion and reranker scores never need to be compared.
- The labelling protocol (`docs/labelling-protocol.md`) is committed before any labelling call is made.
- Repeats: report the mean and min-max over 3 repeats; confidence intervals are bootstrapped on repeat 0.
- All reported numbers come from real runs stored in `results/runs/<run_id>/`. Any "expected" value in this plan is an expectation to check, not a result.
- Labels are described as "LLM-drafted, mechanically checked, with a blind owner-labelled subset of n items and an owner-reviewed queue", never as hand-labelled for the whole set. The README keeps blind labels (`owner_blind`) and reviewed labels (`owner_review`) separate in every count.
- Models read from env (`TRIAGE_SYSTEM_MODEL`, `TRIAGE_JUDGE_MODEL`, `TRIAGE_ANNOTATOR_MODELS`, `TRIAGE_EMBED_MODEL`); names confirmed with `client.models.list()` on key day. Some models reject `temperature`; `LLMClient` records `temperature_supported` per model.
- pydantic-ai: `result.usage` was an attribute in the version checked in research (verify on key day), use `retries=`, set `PYDANTIC_AI_NO_BANNER=1`. Chroma: `embedding_function=None`, own embeddings, collection names at least 3 characters.

## Review Focus

1. Email empty, or empty after cleaning (signature or quote only): no crash, never ANSWER, counted.
2. LLM returns invalid output after retries, or the API errors or times out: item is an `error` prediction, counted wrong, run continues.
3. Wiki page with no headings, empty sections, duplicate titles or very long sections: chunks with unique ids and bounded size.
4. A class absent from a split or from predictions: metrics compute with zeros; stratified bootstrap keeps every class.
5. Test-set leakage: runner refuses `test` without a frozen config and prior dev run; test items never in prompts or tuning.
6. Quote verification loopholes (whitespace and Unicode differences, tiny quotes, quotes that exist but do not support the label): handled by normalisation, a minimum quote length and an entailment check.

---

## File Structure

```
pyproject.toml            uv project, pinned deps, pytest config
src/triage/
  config.py               Settings (env, paths, k, seeds)
  schemas.py              Page, Chunk, Thread, Question, Action, Retrieved, Decision, Annotation, ReviewItem, GoldItem, RunResult
  cli.py                  argparse entry; subcommands registered as tasks land: smoke | ingest | collect | label | eval | ask
  sources/confluence.py   fetch_space_pages, snapshot IO
  sources/mailing_list.py fetch_threads, scrub_pii, is_noise
  clean.py                clean_email
  ingest/chunking.py      chunk_page
  ingest/index.py         build_index, load_index (vector + BM25)
  retrieval.py            VectorRetriever, Bm25Retriever, rrf, HybridRetriever, Reranker, RerankedRetriever
  llm.py                  LLMClient (cache with repeat salt, usage, cost), FakeLLM
  policies/               base.py vanilla.py threshold.py llm_decision.py checklist.py
  labelling/              select.py annotate.py verify.py agreement.py review.py
  evals/                  metrics.py stats.py judge.py judge_validation.py runner.py ablation.py analysis.py
app/streamlit_app.py      stretch
tests/                    mirrors src; fixtures in tests/fixtures/
data/                     raw/ and snapshot/ (gitignored, hashes recorded), gold/ (committed, scrubbed)
configs/                  limits.toml, checklists.toml, prices.toml
results/                  ablation.md, figs/, runs/<run_id>/
docs/                     design, plan, decisions.md, labelling-protocol.md, documentation.md, demo-script.md
README.md                 setup, results table, failure log, data and terms, labelling disclosure
```

---

### Task 1: Scaffold, config, schemas, CLI skeleton, decisions log

**Files:** Create `pyproject.toml`, `.gitignore`, `src/triage/{__init__,config,schemas,cli}.py`, `tests/test_schemas.py`, `tests/test_cli.py`, `README.md` (headings only: Results, Failure log, Data and terms, Labelling disclosure), `docs/decisions.md`.

**Interfaces (all later tasks use these):**
- `Action(StrEnum)`: `ANSWER`, `CLARIFY`, `ESCALATE`; `ACTIONS == ["answer","clarify","escalate"]`.
- `Page(id,title,url,body_html,updated)`; `Chunk(id,page_id,page_title,heading_path:list[str],text,url)`; `Thread(tid,subject,body,n_replies:int,epoch:int,replies_text:list[str],permalink:str)`; `Question(qid,thread_id,text_raw,text_clean,permalink,split:Literal["dev","test"]|None)`.
- `Retrieved(chunk,score:float,score_kind:Literal["cosine","rrf","cross_encoder"],vector_cosine:float|None)`.
- `Decision(action:Action|Literal["error"],reason,missing_detail:str|None,citations:list[str],draft:str|None,error_detail:str|None)`.
- `Evidence(page_id,heading,quote)`; `Annotation(qid,annotator:str,label:Action,evidence:list[Evidence],missing_detail:str|None,reference_answer:str|None,flags:list[str])`; `ReviewItem(qid,annotations,reason:Literal["disputed","evidence_failed","audit","owner_blind"])`; `GoldItem(qid,text,label,evidence,missing_detail,reference_answer,split,source:Literal["agree","adjudicated","owner_blind","owner_review"])`.
- `RunResult(run_id,config_hash,split,predictions:list[dict],metrics:dict,n_errors:int,cost_per_q:float,latency_p50:float)`.
- `Settings` via `load_settings()`: `k=5`, `seed=13`, paths, model names, `contact`.
- `cli.main(argv)->int` with subcommand registry; `triage smoke` is a stub here, filled in Task 5.

- [ ] **Step 0:** the owner approves the design doc and this plan (explicit message) before any file is created.
- [ ] **Step 1:** `git init`; `uv init --package`; add dependencies; write `.gitignore` (data/raw, data/snapshot, .env, .chroma, results/runs/*/cache).
- [ ] **Step 2: Failing tests:** `test_actions_order`; `test_gold_item_rejects_unknown_label`; `test_decision_allows_error_action`; `test_load_settings_reads_env` (monkeypatch `TRIAGE_SYSTEM_MODEL`); `test_cli_unknown_subcommand_returns_nonzero`; `test_contact_not_in_repo` (grep tracked files for "@" in User-Agent strings returns none).
- [ ] **Step 3:** Run `uv run pytest -v`, expect FAIL. **Step 4:** Implement. **Step 5:** Run, expect pass.
- [ ] **Step 6:** Write `docs/decisions.md` with: this review's findings and decisions, pre-registered selection rule, split sizes, labelling disclosure wording, core/stretch line. Commit `chore: scaffold, schemas, decisions log`.

### Task 2: Wiki fetch and frozen snapshot

**Files:** Create `src/triage/sources/confluence.py`, `tests/test_confluence.py`, `tests/fixtures/confluence_page.json`.

**Interfaces:** Produces `fetch_space_pages(space:str, session, limit:int=200, delay:float=1.0) -> Iterator[Page]`; `write_snapshot(pages, path)->str` (sha256 of JSONL); `read_snapshot(path)->list[Page]`; `triage ingest --fetch-only`.

- [ ] **Step 1: Failing tests:** `test_fetch_paginates_until_no_next` (2 pages of results gives 3 `Page`s and 2 requests); `test_user_agent_uses_env_contact`; `test_snapshot_roundtrip_and_hash_stable`; `test_http_500_three_times_raises_fetch_error`.
- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement with `httpx`, `expand=body.storage,version`, 3 retries with backoff, raw response cache. **Step 4:** Run, expect pass.
- [ ] **Step 5:** Live run (no key): `uv run triage ingest --fetch-only`; compare the page count with the research figure (705, to be checked) and record the actual count and hash in README. Commit code only.

### Task 3: Heading-based chunking

**Files:** Create `src/triage/ingest/chunking.py`, `tests/test_chunking.py`.

**Interfaces:** `chunk_page(page:Page, max_tokens:int=500) -> list[Chunk]`.

- [ ] **Step 1: Failing tests:** `test_splits_on_headings_and_keeps_heading_path`; `test_embedding_text_prefixed_with_breadcrumb`; `test_page_without_headings_yields_chunks`; `test_long_section_split_under_max_tokens` (tiktoken count); `test_empty_page_yields_no_chunks`; `test_chunk_ids_unique_and_stable`; `test_duplicate_titles_get_distinct_ids`.
- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement with BeautifulSoup; strip macros; flatten tables to rows; split long sections on paragraph boundaries; no overlap. **Step 4:** Run, expect pass.
- [ ] **Step 5:** Print chunk count and size histogram for the real snapshot into README (evidence for chunk size). Commit.

### Task 4: Mailing list collection, PII scrub, noise filter, email cleaning

**Files:** Create `src/triage/sources/mailing_list.py`, `src/triage/clean.py`, tests, fixtures from `ds_research/stats_user_ofbiz.apache.org.json`.

**Interfaces:** `fetch_threads(list_name, domain, window="3y")->list[Thread]`; `scrub_pii(text:str, known_names:set[str])->str`; `is_noise(subject:str, body:str)->bool`; `clean_email(text:str)->str`; `triage collect`.

- [ ] **Step 1: Failing tests:** `test_scrub_removes_emails_ips_and_header_names`; `test_scrub_removes_names_seen_in_from_fields_inside_bodies`; `test_is_noise_flags_announce_cve_vote_release`; `test_is_noise_keeps_real_question`; `test_clean_email_strips_quotes_and_signature`; `test_clean_email_keeps_question`; `test_clean_email_empty_after_cleaning_returns_empty` (Review Focus 1); `test_future_epoch_filtered`.
- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement; `known_names` comes from the masked sender fields and thread participants. **Step 4:** Run, expect pass.
- [ ] **Step 5:** Live run (no key): `uv run triage collect`; research sample showed 349 roots and about 36 noise, check actuals. Raw threads stay gitignored. Commit code only.

### Task 5: LLM client, cost helper, key-day smoke test

**Files:** Create `src/triage/llm.py`, `configs/prices.toml`, `tests/test_llm.py`; extend `cli.py` (`smoke`).

**Interfaces:** `LLMClient(model, cache_dir, price)`; `complete(messages, *, temperature=0, response_format=None, repeat:int=0) -> LLMResult(text,parsed,usage,cost_usd,cached,latency_s)`; `embed(texts)->list[list[float]]`; `cost_from_usage(usage, price)->float` (also used for Pydantic AI usage); `FakeLLM`. The cache key includes model, messages, params and `repeat`.

- [ ] **Step 1: Failing tests:** `test_cache_hit_zero_cost_and_cached_flag`; `test_different_repeat_values_are_different_cache_entries`; `test_cost_from_usage_known_example`; `test_rate_limit_retried_three_times_then_raises`; `test_temperature_dropped_for_models_that_reject_it`; `test_fake_llm_scripted_order`.
- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement on `openai` SDK; cached results keep original latency in `latency_s` and `cached=True` so reports can exclude them. **Step 4:** Run, expect pass.
- [ ] **Step 5 (key day):** `uv run triage smoke` runs `models.list()`, one embedding, one chat call, one structured call, one Pydantic AI call, and prints model names, prices used and whether `result.usage` is an attribute. Fix `configs/prices.toml` from the pricing page. Commit.

### Task 6: Vector and BM25 indexes, retrievers, RRF

**Files:** Create `src/triage/ingest/index.py`, `src/triage/retrieval.py` (vector, BM25, rrf, hybrid), `tests/test_retrieval.py`.

**Interfaces:** `build_index(chunks, embed_fn, path)->IndexInfo(hash)`; `Retriever(Protocol).retrieve(query,k)->list[Retrieved]`; `VectorRetriever(path, embed_fn)`; `Bm25Retriever(chunks)` on `bm25s`; `rrf(rankings:list[list[str]], k:int=60)->list[tuple[str,float]]`; `HybridRetriever(vector, bm25, pool:int=20)` with `score_kind="rrf"`.

- [ ] **Step 1: Failing tests** with a fake bag-of-words embedder: `test_vector_returns_k_sorted_desc_with_cosine_kind`; `test_identical_text_scores_about_1`; `test_index_persists_and_reloads`; `test_empty_query_returns_empty` (Review Focus 1); `test_bm25_finds_exact_term_vector_misses`; `test_rrf_known_example`; `test_rrf_deduplicates_ids`; `test_hybrid_includes_doc_found_only_by_bm25`; `test_hybrid_score_kind_is_rrf`.
- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement. **Step 4:** Run, expect pass.
- [ ] **Step 5 (key day):** `uv run triage ingest --index` builds both indexes; record index hash and cost in README. Commit.

### Task 7: Metrics and statistics

**Files:** Create `src/triage/evals/{metrics,stats}.py`, tests.

**Interfaces:** `macro_f1(y_true,y_pred)->float` (labels fixed; `error` counts wrong); `per_class_prf(...)->dict` with support; `unsafe_answer_rate`, `over_ask_rate`, `over_escalation_rate` (return `None` when the denominator is 0, never 0); `retrieval_metrics(gold_pages:list[set[str]], ranked_pages:list[list[str]], k)->dict(hit,recall,mrr)`; `bootstrap_ci(y_true,y_pred,metric,n=2000,seed=13,stratify=True)->tuple[float,float]` (resamples within gold class so every class stays present); `mcnemar_exact(correct_a,correct_b)->float`; `paired_bootstrap_diff(...)`; `min_detectable_effect(n_per_class)->float`.

- [ ] **Step 1: Failing tests (hand-computed):** `test_macro_f1_perfect`; `test_macro_f1_missing_pred_class_zero_not_error`; `test_error_prediction_counts_wrong`; `test_matches_sklearn_fixed_labels`; `test_unsafe_rate_example` (2 of 5 gives 0.4); `test_rate_returns_none_on_empty_denominator`; `test_recall_mrr_example`; `test_bootstrap_seeded_and_contains_estimate`; `test_stratified_bootstrap_keeps_all_classes`; `test_mcnemar_known_table` (b=10,c=2 gives p about 0.039); `test_empty_input_raises`.
- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement with scikit-learn, numpy, `scipy.stats.binomtest`. **Step 4:** Run, expect pass. Commit.

### Task 8: Question selection, split, labelling pipeline and owner blind subset

**Files:** Create `src/triage/labelling/{select,annotate,verify,agreement,review}.py`, tests, `docs/labelling-protocol.md`.

**Interfaces:**
- `select_candidates(threads, n_target:int, seed)->list[Question]` drops noise, empty-after-cleaning and near-duplicates; `assign_split(questions, n_dev:int=30, seed)->dict` writes `data/gold/split.json` before any annotation.
- Annotator evidence path is not tied to a single retriever (this reduces, but does not remove, bias toward retrievable pages, which the README states): `annotate(question, tools:AnnotatorTools, llm)->Annotation` where `AnnotatorTools.search_wiki(query)->list[Snippet]` is plain full-text keyword search over the snapshot, and the prompt also includes the union of vector and BM25 top-10 results. Annotators never see thread replies.
- `normalise(text)->str` (Unicode NFKC, collapse whitespace, lowercase); `verify_evidence(annotation, snapshot, min_quote_chars:int=40)->Annotation` adds flag `evidence_failed` (and does not silently change the label) when a normalised quote is not a substring of the cited page or is shorter than the minimum; `check_entailment(annotation, llm)->Annotation` asks a different model whether the quote supports the label and answer, adding flag `unsupported`; CLARIFY items must have `missing_detail` and a second-annotator check that the detail would change the answer; ESCALATE items get a re-search with question keywords and are flagged `possible_answer` if any snippet scores above a lexical threshold.
- `thread_outcome_signal(thread)->Action|None` (weak third vote, reported not used to set labels).
- `agreement(a,b)->dict(kappa,raw,confusion,ac1)`.
- `docs/labelling-protocol.md` (definitions, models, seed, prompts, check rules) is committed before any labelling call. Every evidence quote also records `page_id` so retrieval gold is page-level.
- `build_owner_blind_sheet(questions, n:int=20, seed)->path` produces a sheet with question, search links and no model output. `build_review_queue(annotations, owner_blind)->list[ReviewItem]` covers disputed, flagged and a seeded audit of 12 agreed items (at least 3 per class), shown only after the owner's blind labels are recorded. `apply_review(answers)->list[GoldItem]` (owner blind labels become `source="owner_blind"`, reviewed items `owner_review`) and `owner_agreement(owner_blind, llm_labels_before_review)->dict` which compares the owner's blind labels with the LLM labels as they were before any review, so the agreement figure is not circular.

- [ ] **Step 1: Failing tests:** `test_select_drops_noise_and_empty_cleaned`; `test_split_by_thread_deterministic_disjoint_written_before_annotation`; `test_verify_normalises_whitespace_and_unicode`; `test_verify_flags_missing_quote`; `test_verify_flags_short_quote`; `test_entailment_failure_flags_unsupported` (FakeLLM); `test_clarify_without_missing_detail_invalid`; `test_escalate_with_matching_snippet_flagged_possible_answer`; `test_agreement_kappa_known_example`; `test_annotator_prompt_has_no_thread_replies`; `test_blind_sheet_contains_no_model_labels`; `test_review_queue_has_all_flagged_and_min_3_per_class_audit`; `test_gold_item_source_values`.
- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement. Annotators use two different OpenAI models; disagreements go to a stronger blind adjudicator; each item's final `source` is `agree`, `adjudicated` or `owner`. **Step 4:** Run, expect pass. Commit.
- [ ] **Step 5 (key day, owner involved):** (a) `triage label --draft` runs both annotators and adjudication; (b) owner labels the blind sheet (target 20, minimum 12, owner decides; time it) before seeing any model label; (c) `triage label --review` shows the review queue; (d) `triage label --apply`. Report: class counts, kappa between annotators, evidence and entailment failure rates, owner-versus-LLM agreement (blind subset versus pre-review LLM labels) with a confidence interval and n. If any class has fewer than 5 dev items, redraw the split once now, before any system run, and log it. Commit scrubbed `data/gold/gold.jsonl` after the owner has read it for names. **Gate G1.**

### Task 9: Policy interface, approach A and the eval runner

**Files:** Create `src/triage/policies/{base,vanilla}.py`, `src/triage/evals/{runner,ablation}.py`, tests.

**Interfaces:** `Policy(Protocol)`: `name:str`, `decide(question:str, retrieved:list[Retrieved])->Decision`. `VanillaRAG(llm,system_prompt)` always answers with citations. `run_eval(policy, retriever, gold, split, config, judge=None, repeats:int=1)->RunResult` writes `results/runs/<run_id>/{predictions.jsonl,metrics.json,config.json}`, passes `repeat=i` to the LLM client, wraps per-item exceptions as `Decision(action="error")`. `config_hash(config)->str` covers the fields in Global Constraints. `triage eval --policy NAME --retriever NAME --split dev|test` is added to the CLI here. `ConfigNotFrozenError` when `split="test"` and no dev run with that hash exists. `append_ablation_row(result, step, decision, notes)` columns: step, config hash, macro-F1 [95% CI], answer/clarify/escalate F1 with n, unsafe-answer, over-ask, recall@5, judged correctness, citation support, follow-up targeting, cost/q, latency p50 (uncached), n_errors, keep/drop, run id.

- [ ] **Step 1: Failing tests:** `test_vanilla_always_answers`; `test_runner_turns_policy_exception_into_error_prediction`; `test_runner_refuses_test_without_dev_run`; `test_config_hash_changes_when_prompt_changes`; `test_runner_writes_files`; `test_ablation_row_appended_not_overwritten`; `test_repeats_use_distinct_cache_entries`; `test_latency_excludes_cached_calls`.
- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement. **Step 4:** Run, expect pass.
- [ ] **Step 5 (key day):** run A on dev with the vector retriever; append row "A, vector baseline" (judge columns are filled in when Task 11 re-runs this row; the cached generations are reused so it costs only the judge calls). Expectation to check, not a result: macro-F1 low because A never produces clarify or escalate and the unsafe-answer rate is high. If not, investigate before continuing. Commit.

### Task 10: Class-mix gate and optional second corpus

**Files:** Optionally create `src/triage/sources/docs_site.py`; update `docs/decisions.md`.

- [ ] **Step 1:** Print class counts from gold (overall, dev, test). **Gate G2:** each class at least 12 overall and at least 6 in test, ESCALATE at most 60%. If violated because the wiki lacks the answers, implement `fetch_docs_site()` for `docs.ofbiz.apache.org` (same etiquette and `Page` schema, fixture test), re-snapshot, rebuild indexes, rerun labelling for affected items (annotators and owner blind sheet reused where the question is unchanged). If still unusable, amend the design doc to use Solr.
- [ ] **Step 2:** Record the numbers before and after in `docs/decisions.md` and the README failure log. Commit.

### Task 11: LLM judge and judge validation

**Files:** Create `src/triage/evals/{judge,judge_validation}.py`, tests, `data/gold/judge_validation.jsonl`.

**Interfaces:** `Judge(llm)`: `answer_correct(question, reference, draft)->JudgeVerdict(passed:bool,reason)`, `citation_supported(draft, cited_chunks)->list[JudgeVerdict]` (per claim), `followup_targets_missing(followup, missing_detail)->JudgeVerdict`; binary rubrics, temperature 0 where supported. `validate_judge(judge, labelled)->dict(agreement,kappa,kappa_ci,n,confusion)` per rubric. Item subsets: `answer_correct` on items where gold is ANSWER and the system answered; `citation_supported` on every drafted answer; `followup_targets_missing` on items where gold and prediction are both CLARIFY. `run_eval` accepts the judge and records `judged_correctness`, `citation_support`, `followup_targeting` with n for each.

- [ ] **Step 1: Failing tests:** `test_parses_verdict`; `test_invalid_json_retried_then_error_verdict`; `test_validate_reports_kappa_for_scripted_agreement` (16 of 20 agree gives raw 0.8); `test_judge_model_is_not_the_system_model`; `test_run_eval_records_judge_columns`.
- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement. **Step 4:** Run, expect pass.
- [ ] **Step 5 (key day, owner session S2):** the owner personally grades about 20 real system drafts from the Task 9 baseline run (correct / not, citation supported / not), and these are the validation set; deliberately wrong drafts built from real drafts are added, marked `synthetic: true` and reported separately. Report agreement, kappa, CI and n for each rubric and state the small-n limitation. **Gate G3 (report-only):** with n this small kappa is supporting evidence, not a pass/fail gate; if agreement is below 0.7 on `answer_correct`, revise the rubric once, log the attempt as a failure-log entry and rerun. Then re-run the Task 9 baseline row with the judge columns. Commit.

### Task 12: Approach A+ score threshold

**Files:** Create `src/triage/policies/threshold.py`, tests, `configs/limits.toml`.

**Interfaces:** `ThresholdRAG(base:Policy, threshold:float)` escalates when there are no results or the maximum `vector_cosine` among the results is below `threshold`; `Retrieved.vector_cosine` is the first-stage vector cosine, kept by every retriever (hybrid and reranked retrievers carry it through), so the threshold never compares fusion or reranker scores; `tune_threshold(dev_items, retriever, metric, grid, limits)->float` accepts dev items only. Limits (over-ask maximum, cost per question maximum) are written to `configs/limits.toml` and committed before any test run.

- [ ] **Step 1: Failing tests:** `test_escalates_below_threshold`; `test_delegates_at_or_above`; `test_empty_retrieval_escalates`; `test_uses_vector_cosine_even_when_primary_score_is_rrf`; `test_tune_rejects_test_items`; `test_tune_picks_best_in_grid`.
- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement. **Step 4:** Run, expect pass.
- [ ] **Step 5 (key day):** tune on dev on the vector baseline retriever (the first comparison of approaches); plot score distributions per class and the risk-coverage curve to `results/figs/`; run dev; append row. Log whether the hypothesis "threshold catches ESCALATE but not CLARIFY" holds with per-class recall. Commit.

### Task 13: Approach B, LLM decision with Pydantic AI

**Files:** Create `src/triage/policies/llm_decision.py`, `src/triage/policies/prompts/decision_v1.md`, tests.

**Interfaces:** `DecisionOut(BaseModel)`: `action`, `reason`, `missing_detail`, `citations:list[str]`, `draft`. `Deps(chunk_ids:set[str])` passed as `deps_type`; an `output_validator` taking `RunContext[Deps]` requires CLARIFY to have `missing_detail` and ANSWER to cite at least one id in `deps.chunk_ids` (raising `ModelRetry`). `LLMDecision(model, retries=2, price)` implements `Policy`; usage from `result.usage`, cost via `cost_from_usage`; results cached on `(config_hash, qid, repeat)` in the same cache directory so reruns and cost reporting match the other approaches.

- [ ] **Step 1: Failing tests** using `TestModel`/`FunctionModel` and `agent.override`: `test_answer_without_valid_citation_retried`; `test_clarify_without_missing_detail_retried`; `test_exhausted_retries_returns_error_decision`; `test_prompt_contains_chunks_and_cleaned_question`; `test_empty_question_never_answers`; `test_usage_and_cost_recorded`; `test_second_call_served_from_cache`.
- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement. Few-shot examples, if used, come only from dev items and are listed in `config.json`. **Step 4:** Run, expect pass.
- [ ] **Step 5 (key day):** at most 3 prompt versions on dev, each logged as a failure-log entry with its dev scores and the change made; freeze the best; append row "B, vector baseline". The minimum comparison (A, A+, B on the same vector retriever) now exists. Commit.

### Task 14: Retrieval ladder, one scored row per component

**Files:** Modify `src/triage/retrieval.py` (Reranker, RerankedRetriever), `src/triage/policies/vanilla.py` untouched; tests `tests/test_reranker.py`.

**Interfaces:** `Reranker(model_name)` wraps `sentence_transformers.CrossEncoder` (mps if available); `RerankedRetriever(base, reranker, pool:int=20)` returns `score_kind="cross_encoder"`.

- [ ] **Step 1: Failing tests:** `test_reranker_reorders_by_scores` (stub scorer); `test_reranked_kind_is_cross_encoder`; `test_pool_larger_than_k_then_truncated`.
- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement. **Step 4:** Run, expect pass.
- [ ] **Step 5 (key day), strictly one component at a time.** The vector baseline rows (A, A+, B) already exist from Tasks 9, 12 and 13. Each sub-step below adds exactly one component to the previous best configuration, runs policy A plus the judge (and, for the final configuration, B), appends its own row, compares it with the previous row, and records keep/drop **before** the next component is switched on. Show the owner the row after each sub-step (stop and report).
  - 14a: + email cleaning (query cleaning on, retriever unchanged). Row compared with the Task 9 baseline.
  - 14b: + BM25 with RRF (the code from Task 6 becomes part of the system under test only now). Row compared with 14a (or with the baseline if 14a was dropped).
  - 14c: + MiniLM cross-encoder reranker over the pool of 20. Row compared with 14b (or the previous kept row).
  - 14d (stretch): bge-reranker-v2-m3 instead of MiniLM. Row compared with 14c (or the previous kept row).
  All keep/drop decisions use dev items only (recall@5 and MRR on dev ANSWER items, judged correctness, macro-F1), with RRF k=60 and pool=20 fixed. A row is "kept" only if the paired bootstrap 95% interval of its improvement over the previous row excludes zero; otherwise it is recorded as "no measurable gain" and still shown (the owner chose to show the effect of every component). Pooled dev+test recall is reported afterwards as description only. After the ladder, re-run the best policy (B, and A+ with its threshold re-tuned on dev) on the best retriever and append those rows. Because the annotators used the union of vector and BM25 evidence, also report each retriever's recall on pages the owner-blind subset cites, as a bias check. Commit.

### Task 15: Final selection, held-out test run, failure analysis

**Files:** Create `src/triage/evals/analysis.py`, tests, `results/final.md`.

**Interfaces:** `select_final(dev_results, limits)->str` (candidates include each policy on the vector baseline and on the best retriever) implements the pre-registered rule (best dev macro-F1 within limits, simplicity tie-break inside the minimum detectable effect); `compare_pair(a,b)->dict(delta,ci,mcnemar_p,mde,claim:bool)` (`claim=False` inside the MDE); `error_table(result, gold)->list[dict]`; `risk_coverage(result)->list[tuple]`.

- [ ] **Step 1: Failing tests:** `test_select_final_respects_limits`; `test_tie_within_mde_prefers_simpler`; `test_compare_pair_no_claim_inside_mde`; `test_error_table_lists_every_misclassified_item_including_errors`.
- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement. **Step 4:** Run, expect pass.
- [ ] **Step 5 (key day):** run each finalist once on test with 3 repeats (distinct cache entries), produce the confidence-interval table, paired comparisons between adjacent steps, confusion matrices, risk-coverage plot and categorised error table. The owner reads the error table and confirms or corrects the cause categories (retrieval miss, class boundary, judge disagreement, label doubt). Commit.

### Task 16: CLI demo, documentation, demo recording, submission checklist

**Files:** Modify `src/triage/cli.py` (`ask`); create `docs/documentation.md`, `docs/demo-script.md`; finish `README.md`; test `tests/test_cli_ask.py`.

- [ ] **Step 1: Failing tests:** `test_ask_prints_action_reason_citations_with_fake_llm`; `test_ask_handles_empty_input`.
- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement `uv run triage ask "<email>"`. **Step 4:** Run, expect pass.
- [ ] **Step 5:** README: problem, setup in at most 5 commands, the ablation table inserted by script, the selection reasoning, failure log (attempts, failures, pivots), "Data and terms", "Labelling disclosure" (exact counts: `owner_blind` n, `owner_review` n, LLM-drafted n, blind-versus-LLM agreement with CI), limitations (OpenAI-only model families so self-preference bias, small n, LLM-written reference answers judged by an LLM, annotator evidence union still biased toward retrievable pages), and a "Deviations from the submitted one-pager" list (labelling method, BM25 and reranker moved into the core ladder, dev size 30 instead of 20, extra metrics). `docs/documentation.md` at most 5 pages (check by word count of about 2,500 words including tables). Demo script: 3 minutes using real held-out examples of each class. Record the demo video from a real run. Submission checklist in `docs/decisions.md`: repo access granted to reviewers, design doc, documentation, video link. Commit.

### Task 17 (stretch): Approach C, required-detail checklist with LangGraph

**Files:** Create `src/triage/policies/checklist.py`, `configs/checklists.toml`, tests.

**Interfaces:** `TriageState(question,retrieved,topic,required:list[str],missing:list[str],decision)`; nodes `classify_topic`, `check_required`, `decide`, `draft_followup`, `human_approval` (`interrupt()` with `MemorySaver` and `thread_id`); `ChecklistPolicy` implements `Policy`, auto-approving in eval mode.

- [ ] **Step 1: Failing tests:** `test_missing_detail_routes_to_followup`; `test_all_present_routes_to_answer_or_escalate`; `test_followup_names_missing_detail`; `test_human_approval_interrupts_and_resumes`; `test_deterministic_with_fake_llm`.
- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement; checklists derived from dev inspection only. **Step 4:** Run, expect pass.
- [ ] **Step 5 (key day):** dev row; kept only if it beats B on dev within limits. If added after Task 15, rerun the selection rule and document that the test was run once more for C. Commit.

### Task 18 (stretch): Streamlit interface

**Files:** Create `app/streamlit_app.py`, `tests/test_app_smoke.py`.

- [ ] **Step 1: Failing test:** `test_app_runs_headless` with `streamlit.testing.v1.AppTest` and FakeLLM: entering text shows the action badge and cited sources.
- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement a thin page (input, action badge, reason, citations, draft, toggle to compare A, A+, B). **Step 4:** Run, expect pass. Commit. Screenshots in README only from real runs.

---

## Gates (stop and report scores before continuing)

- **G1 after Task 8:** class counts, annotator kappa, evidence and entailment failure rates, owner-versus-LLM agreement.
- **G2 after Task 10:** class mix acceptable.
- **G3 after Task 11 (report-only):** judge agreement and kappa with confidence intervals and n.
- **After each of Tasks 9, 12, 13, 14a, 14b, 14c, 17:** the new ablation row with its keep/drop reasoning is shown to the owner.

## Self-review notes

- Spec coverage: data surface (2 to 6), core flow and orchestration (9, 12, 13, 17), comparative evaluation with justified choice (9 to 15), judge and quantitative metrics (7, 11), failure analysis (README log from Task 8 on, Task 15), demo and documentation (16), optional UI (18).
- Review findings addressed: cut line and key-day smoke (Task 5), independent annotator evidence and owner blind labels (8), pre-registered selection rule (Global Constraints, Task 15), ordering (CLI in 1, indexes in 6 before annotation in 8), repeat salt (5), Pydantic AI cost and deps (13), judge in the table (9, 11), evidence loopholes (8), score kinds and error handling (1, 7, 12), config hash (Global Constraints), missing types (1), privacy and terms (Global Constraints, 4, 16).
- Offline-first: Tasks 1 to 7 and every unit test run without a key. Steps marked "key day" are the only ones that spend money; the cache makes reruns free.
