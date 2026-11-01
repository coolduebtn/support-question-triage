# Decisions log

## Review-driven decisions (2026-10-29)
- Core vs stretch: Tasks 1-16 core; minimum comparison is A, A+, B on the vector baseline plus final selection. Retrieval ladder (cleaning, BM25+RRF, MiniLM reranker) is core but runs after. Stretch: C with LangGraph, Streamlit, bge reranker.
- Split: by thread, about 30 dev and the rest test; one logged redraw allowed after labelling and before any system run if a class has fewer than 5 dev items.
- Selection rule (pre-registered before any test run): among configurations whose dev over-ask rate and cost per question are within `configs/limits.toml`, choose the best dev macro-F1; if two are within the minimum detectable effect, choose the simpler. Test is run once per finalist to confirm and report; the choice is not changed after seeing test.
- Keep/drop rule: a component "improves" only if the paired bootstrap 95% interval of the difference from the previous row excludes zero on dev items; otherwise "no measurable gain".
- Labelling disclosure wording: "LLM-drafted, mechanically checked, with a blind owner-labelled subset of n items and an owner-reviewed queue". Never described as hand-labelled for the whole set. The submitted one-pager said hand-labelled; README lists this deviation.
- Errors count as wrong predictions. Config hash excludes the git commit.
- Judge validation is report-only (small n); synthetic wrong answers are marked and reported separately.

## Decisions made during the build (2026-10-30 to 2026-11-01)
- Split sizes: 40 dev and 57 test real questions (by thread). Constructed CLARIFY items (20; 12 dev and 8 test) were merged into the main gold set, flagged `synthetic`, and are excluded from threshold tuning. `triage eval --synthetic none` gives the real-only row.
- Judge: `claude-sonnet-5-5` (cheaper than the first judge, `gpt-5.5`, and a different vendor from the system model). If no Anthropic key is set, the OpenAI model in `TRIAGE_JUDGE_FALLBACK_MODEL` is used and the run records which judge produced it. Validated on 15 owner-graded items: 10 of 15 agree (kappa 0.39); the first judge agreed on 11 of 15 (kappa 0.48).
- Retrieval ladder outcome under the keep/drop rule: email cleaning, BM25 with RRF and the MiniLM reranker each showed no measurable gain, so A+ and B use plain vector search. Selection rule outcome on dev: B.
- Vector index: HNSW kept (so the design scales), with `ef_search` raised from 100 to 400 after the approximate search was found to drop a true top-5 chunk for a few questions.
- Gold evidence page ids are canonicalised (chunk ids had been stored as page ids for 25 of 85 entries), which corrected the retrieval scores but no action metric.
- Test was run more than once per config, each time disclosed in the README: first run; a re-run after the HNSW fix; and a cached refresh after the evidence-id fix (no new model calls, identical action metrics).
- Not built: approach C (LangGraph) and the bge reranker row.

## Submission checklist
- [ ] Repo access for reviewers  - [x] Design doc (`docs/design/`)  - [x] Documentation (`docs/documentation.md`, about 3,100 words; the limit is 4 to 5 pages, so it needs trimming)  - [ ] 3-minute demo video link
