Dev runs on the earlier 40-question set (before the CLARIFY items were merged in). Superseded by ablation.md.

| step | split | config | macro-F1 [95% CI] | answer F1 (n) | clarify F1 (n) | escalate F1 (n) | unsafe-answer | over-ask | recall@k | judged correct | citation support | follow-up targeting | cost/q (USD) | latency p50 (s) | errors | decision | notes | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A vector baseline | dev | e9887525 | 0.18 [0.18, 0.18] | 0.55 (n=15) | 0.00 (n=1) | 0.00 (n=24) | 1.00 | 0.00 | 0.66 | 0.47 | 0.67 | n/a | 0.0028 | n/a | 0 | baseline |  | 20261031-191155-e9887525-dev |
| A+ vector threshold | dev | fb7b1306 | 0.34 [0.22, 0.44] | 0.49 (n=15) | 0.00 (n=1) | 0.52 (n=24) | 0.52 | 0.00 | 0.66 | 0.44 | 0.70 | n/a | 0.0015 | 0.00 | 0 | compare |  | 20261031-191157-fb7b1306-dev |
| B LLM decision, vector | dev | 2bbd653a | 0.76 [0.62, 0.93] | 0.77 (n=15) | 0.67 (n=1) | 0.83 (n=24) | 0.16 | 0.00 | 0.66 | 0.50 | 0.76 | 1.00 | 0.0024 | n/a | 0 | compare | prompt v1 | 20261031-191349-2bbd653a-dev |
| A vector + email cleaning | dev | 504ad27b | 0.18 [0.18, 0.18] | 0.55 (n=15) | 0.00 (n=1) | 0.00 (n=24) | 1.00 | 0.00 | 0.73 | 0.67 | 0.71 | n/a | 0.0027 | n/a | 0 | pending |  | 20261031-191941-504ad27b-dev |
