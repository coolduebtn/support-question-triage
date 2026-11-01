| step | n | hit@5 | recall@5 | MRR | recall change vs previous [95% CI] | improves |
|---|---|---|---|---|---|---|
| vector (raw email) | 15 | 0.87 | 0.74 | 0.79 | n/a | n/a |
| + email cleaning | 15 | 0.93 | 0.81 | 0.84 | +0.067 [+0.000, +0.200] | no measurable gain |
| + BM25 with RRF | 15 | 1.00 | 0.88 | 0.91 | +0.067 [-0.067, +0.233] | no measurable gain |
| + MiniLM reranker | 15 | 0.87 | 0.75 | 0.77 | -0.128 [-0.339, +0.056] | no measurable gain |
