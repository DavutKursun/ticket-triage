# Train token lengths (xlm-roberta-base, incl. special tokens)

| source / language | n | p50 | p95 | p99 | max | > max_len |
|---|---|---|---|---|---|---|
| bitext / en | 19639 | 13 | 18 | 21 | 30 | 0 |
| synthetic / en | 275 | 65 | 104 | 123 | 144 | 2 |
| synthetic / tr | 106 | 57 | 92 | 110 | 117 | 0 |
| translated / tr | 300 | 14 | 21 | 23 | 29 | 0 |
| non-Bitext (synthetic + translated) | 681 | 46 | 94 | 114 | 144 | 2 |

Recommended max_len: **128** (smallest of 128/192/256 covering the non-Bitext p99 of 114).
