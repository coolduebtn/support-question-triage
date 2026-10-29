# Decisions log

## Review-driven decisions (2026-10-29)
- Core vs stretch: Tasks 1-16 core; minimum comparison is A, A+, B on the vector baseline plus final selection. Retrieval ladder (cleaning, BM25+RRF, MiniLM reranker) is core but runs after. Stretch: C with LangGraph, Streamlit, bge reranker.
- Split: by thread, about 30 dev and the rest test; one logged redraw allowed after labelling and before any system run if a class has fewer than 5 dev items.
- Selection rule (pre-registered before any test run): among configurations whose dev over-ask rate and cost per question are within `configs/limits.toml`, choose the best dev macro-F1; if two are within the minimum detectable effect, choose the simpler. Test is run once per finalist to confirm and report; the choice is not changed after seeing test.
- Keep/drop rule: a component "improves" only if the paired bootstrap 95% interval of the difference from the previous row excludes zero on dev items; otherwise "no measurable gain".
- Labelling disclosure wording: "LLM-drafted, mechanically checked, with a blind owner-labelled subset of n items and an owner-reviewed queue". Never described as hand-labelled for the whole set. The submitted one-pager said hand-labelled; README lists this deviation.
- Errors count as wrong predictions. Config hash excludes the git commit.
- Judge validation is report-only (small n); synthetic wrong answers are marked and reported separately.

## Submission checklist
- [ ] Repo access for reviewers  - [ ] Design doc  - [ ] Documentation (max 5 pages)  - [ ] 3-minute demo video link
