Superseded rows (earlier runs). Recall, hit and MRR in these rows were computed before the gold evidence page ids were canonicalised (see README entry 18); macro-F1 and the other action metrics are unaffected.

| step | split | config | macro-F1 [95% CI] | answer F1 (n) | clarify F1 (n) | escalate F1 (n) | unsafe-answer | over-ask | recall@k | judged correct | citation support | follow-up targeting | cost/q (USD) | latency p50 (s) | errors | decision | notes | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A vector baseline (test) | test | caa31fb7 | 0.11 [0.11, 0.11] | 0.33 (n=13) | 0.00 (n=11) | 0.00 (n=41) | 1.00 | 0.00 | 0.36 | 0.62 | 0.67 | n/a | 0.0027 | 13.27 | 0 | final |  | 20261101-172648-caa31fb7-test |
| A+ threshold (test) | test | dbb65c21 | 0.33 [0.25, 0.42] | 0.32 (n=13) | 0.00 (n=11) | 0.68 (n=41) | 0.25 | 0.00 | 0.36 | 0.80 | 0.71 | n/a | 0.0007 | 0.00 | 0 | final |  | 20261101-172650-dbb65c21-test |
| B typed decision (test, 3 repeats) | test | a0ce2387 | 0.62 [0.48, 0.74] | 0.52 (n=13) | 0.52 (n=11) | 0.82 (n=41) | 0.19 | 0.15 | 0.36 | 0.75 | 0.70 | 0.17 | 0.0027 | 12.51 | 0 | final |  | 20261101-181034-a0ce2387-test |
| A vector baseline (exact top-5) | dev | caa31fb7 | 0.15 [0.15, 0.15] | 0.45 (n=15) | 0.00 (n=13) | 0.00 (n=24) | 1.00 | 0.00 | 0.69 | 0.47 | 0.70 | n/a | 0.0026 | 19.20 | 0 | baseline |  | 20261101-194030-caa31fb7-dev |
| A+ threshold (exact top-5) | dev | dbb65c21 | 0.29 [0.20, 0.38] | 0.44 (n=15) | 0.00 (n=13) | 0.44 (n=24) | 0.46 | 0.00 | 0.69 | 0.44 | 0.70 | n/a | 0.0014 | 0.00 | 0 | baseline |  | 20261101-194033-dbb65c21-dev |
| B typed decision (exact top-5) | dev | a0ce2387 | 0.66 [0.51, 0.78] | 0.63 (n=15) | 0.53 (n=13) | 0.81 (n=24) | 0.30 | 0.00 | 0.69 | 0.50 | 0.78 | 0.40 | 0.0025 | 12.95 | 0 | baseline |  | 20261101-194048-a0ce2387-dev |
| A + email cleaning (exact top-5) | dev | 9544a29b | 0.15 [0.15, 0.15] | 0.45 (n=15) | 0.00 (n=13) | 0.00 (n=24) | 1.00 | 0.00 | 0.76 | 0.67 | 0.73 | n/a | 0.0026 | 14.28 | 0 | baseline |  | 20261101-194121-9544a29b-dev |
| A + BM25/RRF (exact top-5) | dev | b0ddb1bb | 0.15 [0.15, 0.15] | 0.45 (n=15) | 0.00 (n=13) | 0.00 (n=24) | 1.00 | 0.00 | 0.81 | 0.60 | 0.68 | n/a | 0.0028 | 11.76 | 0 | baseline |  | 20261101-194201-b0ddb1bb-dev |
| A + MiniLM reranker (exact top-5) | dev | f122adb7 | 0.15 [0.15, 0.15] | 0.45 (n=15) | 0.00 (n=13) | 0.00 (n=24) | 1.00 | 0.00 | 0.70 | 0.47 | 0.71 | n/a | 0.0027 | n/a | 0 | baseline |  | 20261101-194208-f122adb7-dev |
