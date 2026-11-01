# 3-minute demo script

All examples are from the held-out **test** split (65 questions, scored once per approach). Run the demo from the viewer (`uv run triage ui`, sidebar: **test**). It replays the saved runs, so what you show is exactly what the numbers were computed from and does not change between takes. Pick each question in the **Compare on a question** tab (use the "correct action" filter, then find the title in the dropdown).

Avoid `q_rfbj6762c9` (its text contains a map link with a place name and coordinates).

## Script

**0:00 Problem and idea (20 s)**
"Support teams get questions that should not all be answered the same way. Some have an answer in the docs, some need one follow-up question, some need a human. Plain RAG always answers. My hypothesis: an explicit decision step (ANSWER, CLARIFY or ESCALATE) beats that. I tested it on the Apache OFBiz wiki, docs and mailing list."

**0:20 ANSWER: B answers with citations (30 s)**
Question `q_g92sgt6j3m`, "What is the difference between Agreement and Term". Gold: ANSWER.
Show B's badge, the draft, and the "Pages it saw" list. Say: "B answers, cites the accounting docs, and the cited pages are the gold pages. A also answers here, so this is the easy case: the decision step costs nothing when the answer exists."
Backup: `q_fy79lk14np` (service `createAcctgTransForSalesInvoice`).

**0:50 ESCALATE: plain RAG answers confidently and has nothing to say (35 s)**
Question `q_9ypyrkz412`, "what is expected date for apache ofbiz 24.0.0". Gold: ESCALATE.
Show A and A+: both answer. Show B: ESCALATE, with the reason "the excerpts only cover much older releases (09.04, 10.04)". Say: "A cannot say 'I don't know'. It will produce a release date from documents about version 10. B sees that the evidence does not contain the answer and routes it to a human."

**1:25 CLARIFY: one targeted follow-up (30 s)**
Question `q_8y3q9lx35p`, "Assistant regarding ofbiz connection with AWS rds". Gold: CLARIFY.
Show A answering anyway, A+ escalating (top similarity 0.57 < 0.60), B asking for the `entityengine.xml` datasource block. Say: "This is one of only 4 real CLARIFY questions in the set, so I do not lean on this class. The other 20 CLARIFY items are constructed and flagged as such."

**1:55 Scoreboard (30 s)**
Open **Scoreboard**, test split. Read out: macro-F1 A 0.11, A+ 0.33, B 0.62 (first repeat); unsafe-answer rate A 1.00, A+ 0.25, B 0.19. "B beats A by +0.51 [+0.37, +0.64]. B versus A+ is +0.28 [+0.12, +0.44], but McNemar p is 0.087, so that gap is weaker evidence. B was run three times: macro-F1 0.62, 0.49, 0.57, so the honest number is about 0.56, not 0.62."

**2:25 Failure (25 s)**
Open **Where it fails**, approach B, and expand `q_7y2tgjclf9` ("User Authorization Required for executeMrp"). Gold: ESCALATE, B ANSWERED with a generic permissions walk-through, because the retrieved security pages looked relevant. Say: "B's main error is over-answering: 6 of its 20 test errors are unsafe answers. Retrieval is part of it, but the model also trusts plausible-looking context."
Backup: `q_m7g9qjsjz5` (gold CLARIFY, B answered).

**2:50 Close (10 s)**
"B was selected by a rule fixed before the test run, on dev only. Hybrid search and a reranker did not pass the keep/drop test, so the system uses plain vector retrieval. Limits: small n, labels are LLM-drafted and mechanically checked, with an owner-labelled blind subset. Everything is in the README."

## Notes for recording

- Stay at about 450 spoken words. The scoreboard section is the one to shorten if you run long.
- Say "held-out test set" once, early.
- Do not claim B is production-ready. The claim is that the decision step helps, with wide intervals.
- Optional live shot (not replayed, so output can vary): **Try your own** tab, or `uv run triage ask "<question>"`. It needs the OpenAI key in `.env` and the built index. Skip it unless you have already tried the exact question.
- Check before recording: viewer is on the **test** split, and the sidebar run IDs match the README table (A `20261101-201827-caa31fb7`, A+ `...201829-dbb65c21`, B `...201832-a0ce2387`).
