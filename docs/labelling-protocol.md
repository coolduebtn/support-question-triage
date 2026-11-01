# Labelling protocol (committed before any labelling call)

**Disclosure.** Labels are LLM-drafted and mechanically checked, with an owner-labelled blind subset and an
owner-reviewed queue. They are not hand-labelled for the whole set. The submitted one-pager said "hand-labelled".

## Data
- Questions are real, verbatim thread openings from the public `user@ofbiz.apache.org` archive (PII scrubbed,
  announcement/CVE/vote threads and non-support posts removed by `select_candidates`). No model writes or paraphrases a question.
- Selection: seeded (seed 13), stratified by whether the thread got a reply, near-duplicates removed.
- Split by thread (40 dev and 57 test among the 97 real questions; 52 and 65 once the constructed CLARIFY items are added), written to `data/gold/split.json` before any annotation call.
  A single logged redraw is allowed after labelling and before any system run if a class has fewer than 5 dev items.

## Label definitions
- **answer**: the wiki snippets contain the answer; evidence is a verbatim quote (at least 40 characters) with its page id.
- **clarify**: the wiki could answer once one specific missing detail (version, environment, error text, feature) is known; the detail is named.
- **escalate**: the wiki does not cover it.

## Procedure
1. Two blind LLM annotators (two different OpenAI models, set in `TRIAGE_ANNOTATOR_MODELS`) each see the question plus
   evidence from the union of vector and BM25 top-10 plus a keyword-coverage search over the whole snapshot. They never see thread replies.
2. Mechanical checks (code): normalised quote (Unicode NFKC, whitespace, case) must be at least 40 characters and a substring of the cited page;
   ANSWER needs evidence; CLARIFY needs a named missing detail. Failures are flagged `evidence_failed` / `invalid_clarify`, never silently relabelled.
3. A second model checks that the quotes support the label (`unsupported` flag). At labelling time this was `gpt-5.5`; the judge used for the final evaluation was later switched to `claude-sonnet-5-5`.
4. ESCALATE items are re-searched for a missed answer (`possible_answer` flag).
5. Disagreement between the two annotators goes to a stronger blind adjudicator (`gpt-5.5` in this project's run).
6. The thread outcome (no reply, follow-up question, substantive reply) is a weak signal, reported, not used to set labels.
7. Reported: Cohen's kappa, Gwet's AC1, raw agreement, confusion matrix between annotators; evidence and entailment failure rates.

## Owner involvement
- **Blind subset:** the owner labels 20 randomly chosen questions (minimum 12) in `data/gold/owner_blind.md` before seeing any model output.
  Owner labels win over LLM labels (`source = owner_blind`).
- **Review queue:** disputed and flagged items plus a seeded audit of 12 agreed items (at least 3 per class), shown after the blind labels exist (`source = owner_review` where the owner overrides).
- **Label-quality figure:** owner blind labels versus the LLM labels as they were before any review, with n and a Wilson 95% interval.

## Known limitations
The annotators and the system model are both OpenAI models (self-preference bias; the evaluation judge is a Claude model), LLM-written reference answers,
annotator evidence still biased toward retrievable pages, small n. Stated again in the README.

## Constructed CLARIFY items (added after labelling)
Only 4 of the 97 real questions are CLARIFY, so 20 further CLARIFY items were constructed: a model deletes key details (and nothing else) from a real answerable question,
and the result is checked to be a deletion only. Each keeps the original question's thread and split, is flagged `synthetic` with `derived_from`, and is never used for threshold tuning.
The original question and the edited one are shown side by side in `data/gold/probes_trace.md`. They are merged into `data/gold/gold.jsonl`.

## Evidence ids
Annotators sometimes cited a chunk id (`<page_id>-<n>`) where a page id was expected. Evidence page ids are canonicalised when the record is built, by dropping a trailing `-<digits>`.
