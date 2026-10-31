| step | n | hit@5 | recall@5 | MRR | recall change vs previous [95% CI] | improves |
|---|---|---|---|---|---|---|
| vector (raw email) | 15 | 0.80 | 0.66 | 0.71 | n/a | n/a |
| + email cleaning | 15 | 0.87 | 0.73 | 0.78 | +0.067 [+0.000, +0.200] | no measurable gain |
| + BM25 with RRF | 15 | 0.87 | 0.81 | 0.74 | +0.083 [-0.067, +0.267] | no measurable gain |
| + MiniLM reranker | 15 | 0.73 | 0.70 | 0.63 | -0.111 [-0.333, +0.067] | no measurable gain |
