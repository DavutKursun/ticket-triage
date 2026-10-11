# ticket-triage — support ticket classification: fine-tuned model vs LLM, with real costs

This file is the build spec. Read it fully before writing code. Work milestone by
milestone; after each milestone stop, show how to run it, and wait for Davut's OK.
Section 0 says where we are and what changed from the original plan.

## 0. Status (updated 2026-10-11)

**Starter is implemented and tested (16 tests, no network). Start at Milestone 1, step 0 below.**

Deviations from the original spec — all because Davut uses **no paid API**:
- LLM engine (A/B) and silver labelling run on **Groq free tier** (`openai/gpt-oss-120b`, OpenAI-compatible,
  JSON mode + Pydantic + one repair retry). Claude stays an optional provider (`LLM_PROVIDER=anthropic`,
  `messages.parse`, prompt caching). The README cost table quotes list prices for `claude-haiku-5-5`
  ("as if") from measured tokens; the free tier is labelled "free tier", never "$0.0000".
- No Batch API. `scripts/label_silver.py` is sequential, resumable (appends JSONL, skips done ids), and exits 3 on
  Groq's daily quota (200k tokens/day) so it can continue the next day. Silver set reduced from 4,000 to
  **2,000 messages (1,000 dev + 1,000 test)** plus a 24-row train pool for few-shot examples.
- Groq's quota is shared with docs-chat and invoice-to-sheet: do not run API scripts for two projects on the same day
  without checking with Davut.
- Model IDs verified 2026-10-09: `claude-haiku-5-5` ($0.10/$0.50 per MTok), `claude-sonnet-5-5` ($2/$10). Never
  downgrade to older IDs without asking.

Implemented:
- `ticket_triage/config.py`, `schema.py` (Triage / SilverLabel / TriageResponse, 12 categories — build_dataset.py
  asserts they match the dataset), `rubric.py` (versioned prompts, urgency + sentiment rubric, few-shot block),
  `providers.py` (OpenAICompatProvider, AnthropicProvider, QuotaError, complete_validated with one repair),
  `llm_engine.py` (zero/few-shot, budget guard), `baseline.py` (TF-IDF word+char + logreg, one model per task),
  `costs.py` (prices, batch/cache math, BudgetGuard), `data.py` (splits, silver, hard set, hand-check templates)
- `scripts/build_dataset.py`, `label_silver.py`, `agreement.py` (accuracy + kappa), `draft_hard.py`, `train_baseline.py`
- `tests/test_core.py` (FakeProvider), CI, Makefile, requirements split (core / full / dev / train)
- NOT yet: `ft_engine.py`, `scripts/train.py`, `export_onnx.py`, `eval.py`, `bench_latency.py`, notebook, FastAPI
  routes, Gradio, Dockerfile.

**Milestone 1, step 0 (do first):** `pip install -r requirements-dev.txt`, `pytest -q`, then
`python scripts/build_dataset.py`. If it exits 2 (category mismatch), update `CATEGORIES` in schema.py and
`CATEGORY_DEFINITIONS` in rubric.py to the dataset's actual set and rerun. Report the per-category × split table.
Then `python scripts/label_silver.py --split train --n 24 --limit 24` as a smoke test of the provider (24 calls),
report tokens used, repair rate and 5 sample labels with reasons. Stop.

macOS notes: Python 3.11 venv; torch is only needed for M3 (Kaggle) — do not install requirements-train.txt locally.

## 1. Goal

An API (and a demo page) that takes a customer message and returns its **category**,
**urgency** and **sentiment** as JSON, so a support team can route and prioritize
tickets automatically. Built two ways and compared honestly:

1. **LLM path**: Claude Haiku 5.5 with structured outputs (zero-shot and few-shot).
2. **Fine-tuned path**: a small multilingual transformer trained on a free Kaggle T4,
   served on CPU.

The README answers the question every client asks: *"Should I fine-tune my own model
or just call an API?"* — with accuracy, latency, and cost per 1,000 tickets for each
option, and a clear recommendation per situation (volume, latency, privacy).

Portfolio purpose: show real NLP skills (fine-tuning, evaluation, label quality),
production thinking (API, latency, cost), and the judgment to recommend the simpler
option when it wins.

## 2. Non-negotiables

- One held-out test set, touched once at the end, used for every approach.
- A "hard" test subset of real-looking messages (long, multi-issue, typos, Turkish),
  because the public dataset is templated and too easy; results reported on both.
- Labels for urgency and sentiment are created with an LLM (silver labels) and their
  quality is measured against 200 hand-checked examples; this is stated in the README.
- Cost per 1,000 tickets is computed from measured token usage and published prices,
  not guessed. Latency measured on the same machine for all approaches.
- The API has input validation, a max length, and returns a confidence; the demo has
  a budget guard for the LLM path.

## 3. Data

Primary: `bitext/Bitext-customer-support-llm-chatbot-training-dataset` on the
Hugging Face Hub (~27k customer messages, 27 intents grouped into 10 categories,
license CDLA-Sharing-1.0 — attribute it and share derived label files under the same
license). Use the **category** column as the gold label for the category task.

Build (`scripts/build_dataset.py`):
- Deduplicate, stratified split 80/10/10 by category → train / dev / test.
- **Silver labels** for urgency (`low | medium | high | urgent`) and sentiment
  (`negative | neutral | positive`): label a stratified 4,000-message subset with
  `claude-sonnet-5-5` using a written rubric (urgent = service down / money lost /
  deadline today; high = blocked; medium = inconvenience; low = question). Use the
  Batch API. Davut hand-checks 200 (100 dev, 100 test) in a simple CSV; report
  agreement (accuracy and Cohen's kappa) in the README. If agreement < 0.75 on a task,
  refine the rubric once and relabel.
- **Hard set** (300 messages): write/paraphrase realistic tickets — 2–6 sentences,
  two issues in one message, typos, mixed case, 60 in Turkish. Draft with Claude from
  the rubric, then Davut edits and labels every one by hand. Half goes to dev, half to
  test. This is the number clients should trust most.

## 4. Approaches

**A. LLM zero-shot** — `claude-haiku-5-5`, `client.messages.parse` with
`output_format=Triage` (Pydantic with `Literal[...]` enums for the three labels and a
`confidence: float`), a system prompt with the category definitions and the urgency
rubric. Compare enum values case-insensitively.
**B. LLM few-shot** — same, plus 2 examples per category in the system prompt
(from train only). Use prompt caching (`cache_control` on the system block) so the
long prompt is cheap on repeated calls; report cost with and without caching.
**C. Fine-tuned encoder** — `xlm-roberta-base` (multilingual) with three
classification heads (multi-task) trained with Hugging Face `transformers` on Kaggle
T4 (`notebooks/train_kaggle.ipynb` + `scripts/train.py` sharing one module). Loss =
sum of the three cross-entropies; 3 epochs; early stopping on dev macro-F1. If
multi-task causes trouble, fall back to three separate heads trained in sequence —
simpler is fine. Export the best model to ONNX with int8 dynamic quantization
(`optimum`) and measure CPU latency; keep the PyTorch model as fallback. Publish the
model to Davut's Hugging Face account with a model card.
**D. Baseline** — TF-IDF + logistic regression, to show what a 5-minute solution gets.

## 5. Evaluation (`scripts/eval.py`)

For each approach, on the test set and on the hard test set:
- Category: accuracy, macro-F1, confusion matrix (saved as PNG for DESIGN.md).
- Urgency and sentiment: macro-F1 against silver labels, and accuracy against the
  200 hand-checked examples.
- Latency: p50 and p95 per ticket on the same CPU (the Space's hardware class), batch
  size 1 for the API, batch 1 and 32 for the fine-tuned model.
- Cost per 1,000 tickets: LLM = measured input/output tokens × list price (sync and
  Batch API at 50% off, with and without prompt caching); fine-tuned = $0 marginal
  on the free Space, plus a "what a small VPS costs per month" line for context.
Then a short decision table: *API wins when…* / *fine-tuned wins when…* (volume,
latency, data privacy, offline, label drift).

## 6. Serving

- FastAPI app: `POST /classify {"text": "..."}` → `{category, urgency, sentiment,
  confidence, model, latency_ms}`; `POST /classify/batch` (≤ 100 texts);
  `?engine=finetuned|llm` (default finetuned). Pydantic validation, 2,000-char limit,
  structured error responses, request logging without message bodies.
- Gradio demo mounted on the same FastAPI app (`gr.mount_gradio_app`) with a text box,
  engine selector, sample tickets (including Turkish and a two-issue message), and a
  side-by-side button that runs both engines and shows both results + latency + cost.
- Deployed to Hugging Face Spaces with the Docker SDK (`Dockerfile` runs `uvicorn`
  on port 7860); `ANTHROPIC_API_KEY` as a secret; daily budget guard for the LLM path.

## 7. Repository layout

```
ticket-triage/
  app.py                        # FastAPI + mounted Gradio
  ticket_triage/
    schema.py, config.py, rubric.py (label definitions + prompt), llm_engine.py,
    ft_engine.py (PyTorch/ONNX inference), costs.py, data.py
  scripts/
    build_dataset.py, label_silver.py (Batch API), train.py, export_onnx.py, eval.py, bench_latency.py
  notebooks/train_kaggle.ipynb
  data/                         # splits + labels (CDLA-Sharing attribution), hand-check CSVs, hard set
  models/                       # not committed; downloaded from the Hub at startup
  eval/results/, tests/, docs/DESIGN.md, assets/demo.gif, README.md, requirements.txt,
  Dockerfile, .env.example, LICENSE, .github/workflows/ci.yml
```

## 8. Milestones

**M1 — data.** Download, splits, silver labelling via Batch API, hand-check CSV
prepared for Davut (he labels 200 + writes/edits the 300 hard messages — budget half a
day). Agreement numbers computed.
**M2 — LLM engines + baseline.** Approaches A, B, D; eval on dev only.
**M3 — fine-tuning.** Kaggle notebook, training script, ONNX export, model on the Hub,
dev results.
**M4 — final evaluation.** One run on test + hard test for all approaches; latency
benchmark; cost table; decision table; DESIGN.md with confusion matrices and error
examples.
**M5 — API, demo, ship.** FastAPI + Gradio, Dockerfile, Space, GIF, README, CI, LICENSE.

## 9. Acceptance criteria

- Fine-tuned category macro-F1 ≥ 0.95 on the test set and ≥ 0.80 on the hard set;
  LLM results reported whatever they are.
- Hand-check agreement reported for urgency and sentiment (target ≥ 0.75 kappa).
- Cost per 1,000 tickets and p50 latency reported for all approaches; decision table
  present.
- `POST /classify` responds in < 200 ms p50 with the ONNX model on the Space CPU.
- `pytest` green (engines mocked), CI green, README on one screen before setup.

## 10. Out of scope

Reply generation, routing integrations (Zendesk/Freshdesk — mention as paid
extensions), active learning. List as next steps.
