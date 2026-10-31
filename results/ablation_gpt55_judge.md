| step | split | config | macro-F1 [95% CI] | answer F1 (n) | clarify F1 (n) | escalate F1 (n) | unsafe-answer | over-ask | recall@k | judged correct | citation support | follow-up targeting | cost/q (USD) | latency p50 (s) | errors | decision | notes | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A vector baseline | dev | 537ceac2 | 0.18 [0.18, 0.18] | 0.55 (n=15) | 0.00 (n=1) | 0.00 (n=24) | 1.00 | 0.00 | 0.66 | 0.47 | 0.66 | n/a | 0.0028 | n/a | 0 | baseline |  | 20261031-135452-537ceac2-dev |
| A+ vector threshold | dev | 9ea988c3 | 0.34 [0.22, 0.44] | 0.49 (n=15) | 0.00 (n=1) | 0.52 (n=24) | 0.52 | 0.00 | 0.66 | 0.44 | 0.69 | n/a | 0.0015 | 0.00 | 0 | compare |  | 20261031-135453-9ea988c3-dev |
| B LLM decision, vector | dev | 44081a7e | 0.76 [0.62, 0.93] | 0.77 (n=15) | 0.67 (n=1) | 0.83 (n=24) | 0.16 | 0.00 | 0.69 | 0.58 | 0.78 | 1.00 | 0.0025 | 17.32 | 0 | compare | prompt v1 | 20261031-135531-44081a7e-dev |
| A vector + email cleaning | dev | 65c1e367 | 0.18 [0.18, 0.18] | 0.55 (n=15) | 0.00 (n=1) | 0.00 (n=24) | 1.00 | 0.00 | 0.73 | 0.53 | 0.70 | n/a | 0.0027 | 10.38 | 0 | pending |  | 20261031-141523-65c1e367-dev |
| A + BM25/RRF hybrid | dev | 57a98f2b | 0.16 [0.10, 0.20] | 0.47 (n=15) | 0.00 (n=1) | 0.00 (n=24) | 0.72 | 0.00 | 0.81 | 0.56 | 0.68 | n/a | 0.0021 | 10.54 | 12 | pending |  | 20261031-143310-57a98f2b-dev |
| A + MiniLM reranker | dev | 980ad75a | 0.00 [0.00, 0.00] | 0.00 (n=15) | 0.00 (n=1) | 0.00 (n=24) | 0.00 | 0.00 | 0.70 | n/a | n/a | n/a | 0.0000 | n/a | 40 | pending |  | 20261031-143856-980ad75a-dev |
