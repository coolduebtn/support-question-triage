# 3-minute demo script

All examples come from the held-out **test** set (65 questions). Run the demo in the viewer (`uv run triage ui`, sidebar: **test**). It replays saved runs, so it looks the same every take. Find each question in the **Compare on a question** tab: filter by correct action, then pick the title.

Avoid `q_rfbj6762c9` (its text has a map link with a place name and coordinates).

Spoken lines are in quotes. Everything else is for you.

## Script

**0:00 The problem (15 s)**
"Support questions should not all get the same treatment. Some can be answered from the docs. Some need one follow-up question. Some need a person. Normal RAG always answers. My idea: add a clear decision step. Answer, clarify, or escalate. I tested it on the Apache OFBiz wiki, docs and mailing list."

**0:15 Methodology (45 s)**
Open the **Methodology** tab. Spend about 10 seconds on the top diagram: retrieve once, then A answers, A+ checks a score, B decides. Then scroll to the second diagram and point along it.
1. "I have 5,397 chunks of documentation, and 259 real questions from the mailing list."
2. "My gold set has 117 questions, each labelled answer, clarify or escalate. Two AI models drafted the labels, code checked them, and I labelled a blind sample myself. Twenty clarify questions are made-up, because real ones are rare. They are flagged."
3. "I tuned everything on 52 dev questions. The 65 test questions were held back, and each approach ran on them once."
4. "A is plain RAG. A+ escalates when the best match is weak. B makes one structured decision."
5. "I tried better search: email cleaning, keyword search and a reranker. None gave a proven gain, so I kept simple vector search."
6. "I wrote the selection rule before the test run. An AI judge scores the answers. It is a Claude model, and the system is an OpenAI model, so the judge does not grade its own family. I checked it against 15 of my grades. It agreed on 10, so I treat it as a rough guide."

**1:00 ANSWER (20 s)**
First, in the sidebar, switch **Question set** from dev to **test**, and open **Compare on a question**. Say once: "These examples are from the held-out test set."
Question `q_g92sgt6j3m`, "What is the difference between Agreement and Term". Say the gold label first: "The right action is answer."
Show B's badge, the draft and "Pages it saw".
"B answers and cites the right accounting pages. A does too. This is the easy case. The decision step costs nothing when the answer exists."
Backup: `q_fy79lk14np`.

**1:20 ESCALATE (25 s)**
Question `q_9ypyrkz412`, "what is expected date for apache ofbiz 24.0.0". "The right action is escalate."
Show A and A+ answering. Then B: escalate, because the pages only cover old versions.
"Plain RAG cannot say 'I don't know'. It writes a release date from old documents. B sees the answer is not in the docs and sends it to a person."

**1:45 CLARIFY (25 s)**
Question `q_8y3q9lx35p`, the AWS RDS connection. "The right action is clarify."
Show A answering anyway, A+ escalating, and B asking for the `entityengine.xml` datasource block.
"B asks for the one thing it needs. But note: only 4 real questions are clarify, so I do not lean on this class."

**2:10 Scoreboard (25 s)**
Open **Scoreboard**, test split.
"Score on test, higher is better: A 0.11, A+ 0.33, B 0.62. B beats plain RAG clearly. B against A+ is weaker evidence. And I ran B three times: 0.62, 0.49, 0.57. So B's honest score is about 0.56."

**2:35 Failure (15 s)**
Open **Where it fails**, approach B, and expand `q_7y2tgjclf9` (executeMrp). "The right action is escalate."
"B answered with a generic permissions guide. Its main mistake is answering when it should hand off. That is 6 of its 20 test errors."
Backup: `q_m7g9qjsjz5`.

**2:50 Close (10 s)**
"I picked B with a rule I wrote before testing. The limits are small data and AI-drafted labels. Everything is in the README."

## Notes for recording

- About 450 spoken words. If you run long, shorten the three example beats first. Keep the methodology.
- Say "held-out test set" once, early.
- Do not call B production-ready. The claim is that the decision step helps, with wide error bars.
- Optional live shot (output can vary): **Try your own** tab, or `uv run triage ask "<question>"`. It needs the OpenAI key in `.env` and the built index. Only use a question you have already tried.
- Before recording, check the viewer is on **test** and the sidebar run IDs match the README table (A `20261101-201827-caa31fb7`, A+ `...201829-dbb65c21`, B `...201832-a0ce2387`).
