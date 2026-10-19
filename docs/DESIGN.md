# Design notes — ticket-triage

## Why two paths
Clients ask "fine-tune or API?". The honest answer depends on volume, latency, privacy and how
often labels change. This project measures all four on the same data so the recommendation is a
table, not an opinion.

## Labels
- Category: gold from Bitext (11 categories; 27 intents collapsed).
- Urgency / sentiment: silver labels from an LLM (rubric in `ticket_triage/rubric.py`,
  `PROMPT_VERSION` recorded on every row). Quality measured on 200 hand-checked rows (Cohen's kappa).
  Known limitation: silver labels inherit the LLM's biases; the hard set is hand-labelled to
  counter that.
- Hard set: 300 realistic messages, hand-edited and hand-labelled. The number clients should trust.

## Structured output
Pydantic `Triage` with `Literal` enums; values normalised case-insensitively. Groq: JSON mode +
Pydantic + one repair retry (json_schema mode corrupted UTF-8 in the invoice project, so it is
opt-in). Claude: `messages.parse(output_format=Triage)`.

## Cost accounting
Measured input/output tokens per call × published list prices (`costs.py`, checked 2026-10-09).
The default engine runs on Groq's free tier; the README quotes the cost "as if" on Claude Haiku
so the comparison is meaningful for a paying client. Batch (50% off) and prompt-cache variants
are computed from the same token counts.

## Evaluation protocol (M2: dev only)
- Rows: the 495 silver-labelled dev messages (45 per category), the same rows for every engine. The test
  split and the hard set are not touched until the final run (`scripts/eval.py` refuses `--split test`
  without `data/hard/hard.csv` and `--final`).
- Category is scored against Bitext gold; urgency and sentiment against silver labels. Macro-F1 is taken over
  the classes present in the truth (dev silver has no `urgent` and no `positive`), so an absent class does not
  zero out a quarter of the score; a predicted class that never occurs still costs precision.
- Caveat: the silver labels come from claude-haiku-5-5, the same model the LLM engines run on, so their
  urgency/sentiment scores against silver are optimistic (self-agreement). The 200-row hand-check and the
  hand-labelled hard set are the correction; dev numbers are labelled preliminary everywhere.
- Latency is per ticket, batch size 1, on the development Mac; LLM latency includes the network round trip
  and any repair call. Predictions stream to `eval/results/<split>-<engine>.predictions.jsonl` so a run can
  resume and errors can be inspected row by row.

## Prompt caching accounting
The few-shot engine puts `cache_control` on its system block (rubric + 22 train examples, ~2.1k tokens with the
structured-output schema; Claude Haiku 5.5 caches prefixes from 512 tokens). Anthropic reports
`usage.input_tokens` as the *uncached remainder*, so the provider adds `cache_read_input_tokens` and
`cache_creation_input_tokens` back and `LLMResult.input_tokens` is always the full prompt, like the
OpenAI-compatible path. Cost lines: sync (all input at list price), cached (reads at 0.1×, writes at 1.25×, as
measured), batch (50% off, uncached). Structured outputs are not free: `count_tokens` puts the zero-shot prompt
at ~830 tokens, but the API bills ~1,350 per call once the `output_format` schema is attached.

## Baseline (D)
TF-IDF word 1–2-grams + char_wb 2–5-grams, logistic regression (C=4, balanced), one model per task. Category
trains on all 19,639 train rows (gold); urgency and sentiment on the 990 silver-labelled train rows. Training
takes under two seconds and the model is 1.8 MB. On dev the category task is essentially solved (accuracy
1.000 on the 495 scored rows, 0.999 on all 2,455): Bitext intents are templated, which is exactly why the hard
set exists.

## Fine-tuned path (M3)
`ticket_triage/ft_model.py` (shared by `scripts/train.py` and the Kaggle notebook): xlm-roberta-base, dropout,
three linear heads on the first-token state; loss = CE(category) + CE(urgency) + CE(sentiment). Plain PyTorch loop
so the two data problems stay visible:
- **Missing labels.** Category is gold on every train row (19,639); urgency/sentiment exist only on the 990
  silver-labelled rows plus the synthetic rows. Missing labels are `-100`; a fully masked batch contributes 0, never
  NaN. Labelled rows are oversampled ×4 per epoch (logged as the effective epoch size) so the small heads get
  enough updates without starving the category head.
- **Missing classes.** Silver train has 0 `urgent` and 1 `positive`. `scripts/make_synth.py` writes TRAIN-ONLY synthetic
  rows for `urgent`, `high` and `positive` from the rubric's exact definitions, then the production silver labeller
  labels them blind and only rows where both labels agree are kept (`data/silver/train_synth.jsonl`,
  `source="synthetic"`). They never enter dev, test or the hard set, so evaluation stays clean. Inverse-sqrt class
  weights (normalised to mean 1, `--no-class-weights` to disable) soften the remaining imbalance.
  Run of 2026-10-19: 360 planned (120 per class, 30% Turkish), 272 kept, 88 rejected, $0.12. Kept: urgent 108/120,
  high 77/120, positive 87/120. The labeller mostly disagreed on tone (planned neutral, read as negative) and on
  medium vs low for positive rows; rejected rows stay in `train_synth.rejected.jsonl` for inspection. With them the
  real run sees 19,911 train rows, 1,262 with urgency/sentiment (urgency 791 low / 222 medium / 141 high / 108
  urgent; sentiment 933 neutral / 241 negative / 88 positive) and an effective epoch of 23,697 rows = 741 steps.
- Training: AdamW 2e-5, 6% warmup, fp16 on CUDA, batch 32, 3 epochs, fixed seed; after each epoch category on the
  full dev split and urgency/sentiment on the silver dev rows (macro-F1 over classes present); best epoch = mean of
  the three macro-F1s. Saved: safetensors state dict, encoder config, tokenizer, `labels.json`, `dev_metrics.json`.
- The Kaggle run is not the first run: `make train-smoke` (`train.py --tiny`) builds a seeded 2-layer, 32-wide random
  XLM-R locally (`XLMRobertaConfig` sized to the tokenizer-only repo `hf-internal-testing/tiny-random-xlm-roberta`,
  no weights download) and trains it on 64 rows in the separate `.venv-train` (CPU torch, pinned); it must save a
  model and `dev_metrics.json`. Kaggle runs `--model-name xlm-roberta-base` unchanged. Export (M3-2): ONNX + int8 dynamic
  quantisation via optimum; latency measured batch 1 and 32 on CPU.

## Findings
### Dev, preliminary (M2) — `eval/results/dev-summary.md`
- Category: baseline 0 errors on 495; zero-shot 7 (`order`→`delivery` ×3, `order`→`contact` ×2, `delivery`→`shipping`,
  `feedback`→`refund`); few-shot 6 (`order`→`delivery` ×5, `feedback`→`refund`). "Where is my order" sits between
  the `order` (track order) and `delivery` (where is the parcel) definitions; a definition tweak is a candidate for the
  next `PROMPT_VERSION`, to be judged on the hard set, not on dev.
- Urgency: every engine under-predicts `medium` against silver. Zero-shot calls 414/495 rows `low` (silver: 347) with
  65 `medium`→`low` misses; few-shot 395 `low` and over-predicts `high` (41 vs 35); the baseline over-predicts
  `medium` (136 vs 113). The silver labels themselves agree only 82% (κ 0.54) with a second LLM, so part of this is
  label noise on templated one-liners — the hand-check decides how much.
- Sentiment: zero-shot 23 disagreements, few-shot 45, both symmetric `negative`↔`neutral`; the 22 examples did not
  help here. The baseline matches the silver distribution exactly (438/57) with 26 swaps.
- Confidence carries signal: zero-shot averages 0.85 overall but 0.64 on its category errors (few-shot 0.89 vs 0.79),
  usable for a review-queue threshold later.
- Tokens: zero-shot answers are longer (77 output tokens per call, p50 40, max 322) than few-shot (43); adaptive
  thinking shows up as the long tail. Few-shot's 2.1k-token system block was served from cache on 494 of 495 calls.
- Run: 990 LLM calls, 0 repairs, 0 rate-limit retries, 0 failed rows, $0.11 total at list price.

### Test and hard set
_Milestone 4._

## Next steps
Reply drafting, Zendesk/Freshdesk routing, active learning on low-confidence tickets.
