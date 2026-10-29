# 3-minute demo script (fill examples from the real held-out test run)

1. (0:00) One sentence: decide ANSWER / CLARIFY / ESCALATE instead of always answering. Show the three-way table from the README.
2. (0:25) `uv run triage ask "<held-out ANSWER question>"` : action, reason, cited wiki page, draft reply.
3. (0:55) `uv run triage ask "<held-out CLARIFY question>"` : the targeted follow-up and the missing detail.
4. (1:25) `uv run triage ask "<held-out ESCALATE question>"` : escalation with closest pages.
5. (1:50) Show the ablation table (`results/ablation.md`): A, A+, B and each retrieval step, with confidence intervals and the keep/drop column.
6. (2:30) Show one error from the error table and the failure log entry that explains it. State the limitations (small n, labelling method).
7. (2:50) Close with the selected approach and why (pre-registered rule).
Record from a real run; use only examples that are in the test set and say so.
