# Demo script

Run the demo in the viewer: `uv run triage ui`. It replays saved runs, so it looks the same every take. The first examples use the **dev** set (52 questions, used while building). The scoreboard then switches to the held-out **test** set (65 questions).

How to read this file: text in quotes is what you say. Everything else is what you do on screen.

Avoid `q_rfbj6762c9` (its text has a map link with a place name and coordinates).

## Script

### 1. The problem

"This idea comes from my company's IT support. Questions arrive all day, and they are not all the same. Some can be answered from the documentation. Some need one follow-up question first. Some need a person.

Normal RAG always answers, even when it should not. My idea is to add a clear decision step: answer, clarify, or escalate.

I cannot share my company's tickets. So for this capstone I changed the scope. I use public questions about Apache OFBiz, taken from its wiki, its docs and its mailing list."
My main goal was learning so i have kept the project simple to showcase the concepts learned during the cohort.

### 2. Methodology: the approaches

Open the **Methodology** tab. Stay on the top diagram and point at each approach as you speak.

"I compared three approaches: A, A+ and B.

Approach A is plain RAG. It always writes an answer. This is my baseline.

Approach A+ adds one simple check. If the best match is weak, it escalates instead of answering. I tuned that cutoff on development data only.

Approach B is the decision step. The model makes one structured choice: answer, clarify or escalate, and it gives a reason. If it answers, it must cite pages. Code checks the output and asks again if it is invalid.

All three approaches use the same search. It finds the five best chunks of documentation for the question. So any difference in the results comes from what happens after the search.

I also tried to improve the search with email cleaning, hybrid search and a reranker. I show the results in a moment."

### 3. Methodology: how it was tested

Scroll down to the second diagram and point along it.

"For data, I have 5,397 chunks of documentation, and 259 real questions from the mailing list.

My gold set has 117 questions. Each one is labelled answer, clarify or escalate. Two AI models drafted the labels, code checked them, and I labelled a blind sample myself. 

I tuned everything on 52 development questions. The other 65, the test questions, were held back. Each approach ran on them once. I wrote the selection rule before the test run.

To score the answers, I use an AI judge. The judge is a Claude model, and the system being tested is an OpenAI model. So the judge does not grade its own family. I checked it against 15 of my own grades. It agreed on 10, so I treat it as a rough guide."

### 4. Example: ANSWER (dev set)

The viewer opens on the **dev** set. Stay there. Open **Compare on a question**. Set "Show questions where the correct action is" to **answer**. The list is sorted by question ID, so the example is near the top.

"These examples are from my development set. These are the questions I used while building. The held-out test results come later."

Pick the 4th question, `q_6ggpkj7k5n`, "OutOfMemoryError in Webtools When Fetching Large Dataset".

"The right action here is answer."

Show B's badge, its draft and "Pages it saw".

"B answers with the JVM heap and cache guidance from the docs. A answers too. A+ wrongly escalates, because its best match scored 0.46, under its 0.60 cutoff. B reads the pages instead of trusting one number."

### 5. Example: CLARIFY (dev set)

Set the filter to **clarify**. Pick the 1st question, `q_0qls0o5kcl`, "Sale order creation without error".

"The right action here is clarify."

Show A answering anyway, A+ escalating, and B asking for the full error message and server log.

"The user says there is an error, but not which one. B asks for the one thing it needs. But only 4 real questions are clarify, so I do not lean on this class."

Note: the entries in the clarify list ending in "-probe" are the made-up questions. Skip those.

### 6. Example: ESCALATE (dev set)

Set the filter to **escalate**. Pick the 1st question, `q_13j39obxj7`, "Manufacturing reports".

"The right action here is escalate."

Show A answering, then A+ and B escalating. Read B's reason.

"The user asks where to report a bug. Plain RAG writes a confident answer anyway. B sees that the docs do not say where to report it, and sends it to a person."

### 7. Scoreboard

Open **Scoreboard**. First stay on the **dev** set.

"This is the score on the development set. Higher is better. A scores 0.15, A+ scores 0.29, and B scores 0.66."

Scroll down to **Retrieval variants**. This table exists only for dev, because I used dev to decide which search to keep.

"Remember I tried to improve the search? Here are the results, on the 15 answerable dev questions. Look at the recall column. Plain vector search finds 74% of the right pages. Email cleaning gets 81%. Hybrid search gets 88%, the best number. The reranker gets 75%, no better than plain search.

But 15 questions is small. My rule was to keep a step only if its confidence interval stays above zero. None did. Hybrid looked best, but the interval still included zero, so I cannot say it is real. So all three approaches use plain vector search. This does not mean hybrid search or reranking do not work. My data was too small to show it."

Now in the sidebar, switch **Question set** to **test**. Scroll back up to the scores.

"Now the held-out test set. Each approach ran on it once. A scores 0.11, A+ scores 0.33, and B scores 0.62. B beats plain RAG clearly. B against A+ is weaker evidence.

I also ran B three times: 0.62, 0.49 and 0.57. So B's honest score is about 0.56."

### 8. Live demo: Try your own

Open the **Try your own** tab. Type this question, which is not in my data: `I cannot connect OFBiz to my database. How do I fix it?`

"One last thing: a question that is not in my data. This runs live."

B should show **clarify**, asking which database and version.

"B does not guess. Setup steps differ by database, so it asks which one. This is live, so the wording can change a little, but the decision is the same."

Tested twice: clarify both times. If the live call fails while recording, cut and re-record this step. Do not read the full draft.

### 9. Limits and areas for improvement

"Here is what I would do next.

First, more data. My test set is small, so small differences are not reliable. And I need more real clarify questions, so I do not have to make them up.

Second, fewer unsafe answers. B's main mistake is answering when it should hand off. That is 6 of its 20 test errors. A stricter rule, or a second check before answering, could help.

Third, more approaches. I did not build a single agent that searches by itself, or a graph-based workflow. Both are good comparisons for later.

Fourth, real tickets. This used public Apache questions. The real test is my company's IT support questions."

### 10. Close

"So, does a decision step help? Yes. On the test set, B scored 0.62 and plain RAG scored 0.11. B knows when to answer, when to ask, and when to hand off to a person.

But this is a small test, so I would not call it production-ready. It is good evidence that the idea works.

The code, the data links and the full write-up are in the README. You can run everything yourself."

## Notes for recording

- There is no time limit. Take the time each part needs. If you need to cut, shorten the three examples before the problem or the approaches.
- Do not name your company. "My company's IT support" is enough.
- Say clearly that the three examples are dev questions, and that you switch to the held-out test set on the scoreboard.
- Do not call B production-ready. The claim is that the decision step helps, with wide error bars.
- The live step (step 8) needs the OpenAI key in `.env` and the built index. Its output can vary, so test the question just before recording. Fallback: `uv run triage ask "<question>"`.
- Before recording, check that the viewer starts on **dev** and that the sidebar run IDs match the README table (A `20261101-201827-caa31fb7`, A+ `...201829-dbb65c21`, B `...201832-a0ce2387`).
