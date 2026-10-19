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
  medium vs low for positive rows; rejected rows stay in `train_synth.rejected.jsonl` for inspection.
- **Language and length shortcut (M3-2).** After M3-1 every Turkish and every >16-word labelled train row was synthetic
  and urgent/high/positive: Turkish rows were 69% urgent+high and 31% positive (English 16% / 5%), and only 32% of the
  >16-word rows were low/medium. A model could learn "Turkish or long ⇒ urgent/positive", which is exactly the hard
  set's profile. Two train-only fixes (`eval/results/train_confounds.md`, `scripts/train_confounds.py`):
  - plan `control` in `make_synth.py`: 240 messages in the same style, length and language mix with the ordinary label
    mix (80 low/neutral, 40 low/negative, 60 medium/neutral, 60 medium/negative), same blind verification. Only 109 were
    kept ($0.085): the silver labeller moved 54 planned-low rows to medium, 21 medium to low and 24 to high, and kept
    just 7 of 40 low/negative — the low/medium boundary on long, annoyed tickets is the fuzziest part of the rubric,
    worth watching in the hand-check;
  - `scripts/translate_train.py`: 300 silver train rows (27–28 per category, urgency in proportion) translated into
    customer-style Turkish by Haiku, labels carried over, 60 typed without Turkish characters ($0.012, 0 rejected).
  After: Turkish urgent+high 18% vs English 15% (gap 2.9 points, target ≤ 10) and positive 6% vs 5%; >16-word rows 51%
  low/medium (target ≥ 40%). No top-up was needed.
- **Short rows (M3-2b).** That still left no urgent and almost no positive row of ≤16 words (Bitext has none and neither
  do its translations), so "short ⇒ not urgent" was learnable and one-liners like "charged twice, fix it today" would be
  under-called in the demo. Plan `short`: 90 messages of 6–16 words in one or two sentences (40 urgent, 20 high, 30
  positive with low/medium urgency), ~30% Turkish with a third of those folded to ASCII before the blind verification;
  texts outside the word window are re-asked. Kept 66 ($0.030): urgent 34/40, high 13/20, positive 19/30 (the labeller
  read 9 medium/positive rows as low). ≤16-word labelled rows: urgent 0 → 34, urgent+high 6% → 10%, positive 0% → 2%;
  the language and length targets still hold (Turkish 21% vs English 17% urgent+high, >16-word rows 51% low/medium).
- **Real run data.** 20,386 train rows (19,639 Bitext + 447 synthetic + 300 translated), 1,737 with urgency/sentiment
  (urgency 1,076 low / 346 medium / 173 high / 142 urgent; sentiment 1,281 neutral / 349 negative / 107 positive),
  effective epoch 25,597 rows = 800 steps, 2,400 steps for 3 epochs. `--no-synth` / `--no-translated` drop either part.
- **max_len 192 (M3-2b).** xlm-roberta-base token counts (`eval/results/train_token_lengths.md`, incl. special tokens):
  Bitext p99 21, synthetic English p99 122 (max 144), synthetic Turkish p99 108, translated p99 23; the non-Bitext p99
  is 113, which 128 would cover, but six-sentence hard-set tickets need headroom at inference. Train rows truncated:
  2 at 128, 0 at 192. Collate pads per batch, so training barely pays for it. `TrainConfig.max_len` is the single
  default: it is saved in ft_config.json, `export_onnx.py` copies it into `onnx/ft_config.json`, and the engine reads
  it from there. The tiny smoke encoder has 130 position embeddings and caps itself at 128.
- Training: AdamW 2e-5, 6% warmup, fp16 on CUDA, batch 32, 3 epochs, fixed seed; after each epoch category on the
  full dev split and urgency/sentiment on the silver dev rows (macro-F1 over classes present); best epoch = mean of
  the three macro-F1s. When `data/hard/hard.csv` exists its dev half is scored every epoch as well (report only, never
  used for selection). Saved: safetensors state dict, encoder config, tokenizer, `labels.json`, `dev_metrics.json`;
  `train.py` adds `train_meta.json` (git commit, GPU, torch/transformers/CUDA versions, wall time, peak GPU memory).
- The Kaggle run is not the first run: `make train-smoke` (`train.py --tiny`) builds a seeded 2-layer, 32-wide random
  XLM-R locally (`XLMRobertaConfig` sized to the tokenizer-only repo `hf-internal-testing/tiny-random-xlm-roberta`,
  no weights download) and trains it on 64 rows in the separate `.venv-train` (CPU torch, pinned); it must save a
  model and `dev_metrics.json`. Kaggle runs `--model-name xlm-roberta-base` unchanged; `make notebook-smoke` executes
  the Kaggle notebook's own cells locally with the tiny model (no clone, pip or push).

### ONNX export and serving (M3-2)
- `scripts/export_onnx.py` exports the saved module (PyTorch fp32, CPU) with `torch.onnx.export`, opset 17, dynamic
  batch and sequence axes. The default (dynamo) exporter needs `onnxscript`, which we do not install, so it falls back
  to `dynamo=False`; whichever exporter wins must match PyTorch within 1e-3 on two input shapes. optimum is not used:
  a custom three-head module gains nothing from it.
- int8: `onnxruntime.quantization.quantize_dynamic`, QInt8 weights. MatMuls and the embedding Gather are both
  quantised, so xlm-roberta-base shrinks from ≈1.1 GB to ≈280 MB. If int8 category argmax agreement with PyTorch on
  dev is below 98%, the ladder tries per_channel=True, then QUInt8, and keeps the first that passes.
- `MODEL_DIR/onnx/` is self-contained (model.int8.onnx, tokenizer.json, labels.json, ft_config.json with max_len,
  pad id, exporter and quantisation); the fp32 file stays in `MODEL_DIR/fp32/` and is never uploaded.
  `export_report.json` records max |Δlogit| fp32 vs PyTorch, per-head argmax agreement (fp32 and int8), dev macro-F1
  PyTorch vs int8, file sizes, tokenizer parity (`tokenizers` on tokenizer.json vs the transformers tokenizer on 200 dev
  texts, every Turkish train row and a 5,000-character input) and batch-1 CPU latency on 2 threads, labelled with the
  machine it ran on. On the tiny smoke model: fp32 max |Δlogit| 3.6e-7, int8 category agreement 100%, tokenizer
  parity on every text, serving folder 0.52 MB.
- `ticket_triage/ft_engine.py` serves `onnx/` with onnxruntime + tokenizers + numpy only (no torch, no transformers),
  from a local folder or a Hub repo id (downloads `onnx/*` only). It returns the LLM engine's `TriageResponse` plus
  per-head probabilities; `confidence` is the category probability. Inputs are truncated to the training max_len, so a
  5,000-character message classifies cleanly; empty input is rejected. Caveat: dynamic quantisation scales
  activations per batch, so a batched result can differ from the single-message result in the last decimals
  (5.5e-4 on the fixture). Threads: `FT_THREADS`.
- Tests: `tests/fixtures/tiny_onnx/` (0.52 MB, the exported smoke model plus transformers' token ids for a
  cross-version tokenizer check) drives the engine tests in the main venv and CI; `.venv-train` adds an export
  parity test.

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
