# dev results — 495 silver-labelled rows (preliminary)

| engine | category acc | category macro-F1 | urgency macro-F1 | sentiment macro-F1 | p50 / p95 ms | tokens in / out per call | $ / 1,000 sync | $ / 1,000 cached | $ / 1,000 batch |
|---|---|---|---|---|---|---|---|---|---|
| llm_zero | 0.986 | 0.986 | 0.659 | 0.883 | 1305 / 2079 | 1358 / 77 | $0.1744 | – | $0.0872 |
| llm_few | 0.988 | 0.988 | 0.721 | 0.775 | 1035 / 1588 | 2212 / 43 (2127 cached) | $0.2426 | $0.0513 | $0.1213 |
| baseline | 1.000 | 1.000 | 0.676 | 0.871 | 3 / 3 | – | $0 marginal | – | – |

Category is scored against Bitext gold labels; urgency and sentiment against silver labels from claude-haiku-5-5 (their own quality is measured against the 200-row hand-check). Latency is per ticket on this machine, batch size 1; LLM latency includes the network round trip. Cost is measured tokens × list prices; 'cached' = measured cache reads at 0.1× and writes at 1.25×; 'batch' = Batch API at 50% off, uncached.
