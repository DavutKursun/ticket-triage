# Train token lengths (xlm-roberta-base, incl. special tokens)

| source / language | n | p50 | p95 | p99 | max | > 128 | > 192 | > 256 |
|---|---|---|---|---|---|---|---|---|
| bitext / en | 19639 | 13 | 18 | 21 | 30 | 0 | 0 | 0 |
| synthetic / en | 320 | 61 | 100 | 122 | 144 | 2 | 0 | 0 |
| synthetic / tr | 127 | 54 | 91 | 108 | 117 | 0 | 0 | 0 |
| translated / tr | 300 | 14 | 21 | 23 | 29 | 0 | 0 | 0 |
| non-Bitext (synthetic + translated) | 747 | 35 | 93 | 113 | 144 | 2 | 0 | 0 |

The smallest of 128/192/256 covering the non-Bitext p99 of 113 is 128; the configured default is **192** (headroom for six-sentence hard-set tickets at inference). Train rows truncated: 2 at 128, 0 at 192, 0 at 256.
