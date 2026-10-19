# ticket-triage — support ticket classification: fine-tuned model vs LLM, with real costs

This file is the build spec. Read it fully before writing code. Work milestone by
milestone; after each milestone stop, show how to run it, and wait for Davut's OK.
Section 0 says where we are and what changed from the original plan.

## 0. Status (updated 2026-10-19)

**Milestone 1 is 2/3 done (22 tests, no network): dataset built, 1,980 silver labels done, hand-check templates
written, 300 hard-set candidates drafted. Next is M1 3/3 — Davut's 200-row hand-check and his edit/label pass over
the hard set (see "Next" below).**

Deviations from the original spec:
- Silver labels come from **`claude-haiku-5-5`** (`LLM_PROVIDER=anthropic`, `messages.parse` + Pydantic, one repair
  retry), not Sonnet and not the Batch API: `scripts/label_silver.py` is sequential and resumable (appends JSONL,
  skips done ids), retries transient 429/5xx/network errors with capped backoff (`providers.with_backoff`) and exits
  3 on a daily quota. The 1,980 rows cost ≈$0.25 at list price (≈864 input / 84 output tokens per call, 0 repairs).
- Silver set is **1,980 messages = 990 train + 495 dev + 495 test** (equal per category; `--n 1000/500/500` rounds
  down to a multiple of 11). Train silver labels feed the baseline / fine-tuned urgency+sentiment heads and the
  few-shot pool. Hand-check files: `data/handcheck/{dev,test}_100.csv`, stratified by silver urgency, topped up to 100.
- Groq free tier (`openai/gpt-oss-120b`, `LLM_PROVIDER=openai_compat`, `LLM_REASONING_EFFORT=low`) was the labeller
  until 2026-10-16; its 939 train labels are kept in `data/silver/train.gpt-oss-120b.jsonl` (never loaded) as a
  cross-labeller check: vs Haiku, urgency agreement 0.82 (κ 0.54), sentiment 0.95 (κ 0.71). Groq's quota is shared
  with docs-chat and invoice-to-sheet: do not run API scripts for two projects on the same day without checking
  with Davut.
- The README cost table quotes list prices for `claude-haiku-5-5` from measured tokens; a free-tier model is
  labelled "free tier", never "$0.0000".
- Model IDs verified 2026-10-09: `claude-haiku-5-5` ($0.10/$0.50 per MTok), `claude-sonnet-5-5` ($2/$10). Never
  downgrade to older IDs without asking.
- Labels are skewed because Bitext is templated: across 1,980 rows there are 0 `urgent` and 1 `positive`
  (urgency ≈73% low / 22% medium / 7% high; sentiment ≈89% neutral). The hard set must supply those classes.

Implemented:
- `ticket_triage/config.py` (incl. `LLM_REASONING_EFFORT`), `schema.py` (Triage / SilverLabel / TriageResponse,
  11 categories — build_dataset.py asserts they match the dataset), `rubric.py` (versioned prompts, urgency +
  sentiment rubric, few-shot block), `providers.py` (OpenAICompatProvider with `reasoning_effort`, AnthropicProvider,
  QuotaError, `with_backoff`, `complete_validated` with one repair; `max_tokens` 1024 so Claude's adaptive thinking
  never truncates the JSON), `llm_engine.py` (zero/few-shot, budget guard), `baseline.py` (TF-IDF word+char + logreg,
  one model per task), `costs.py` (prices, batch/cache math, BudgetGuard), `data.py` (splits, silver, hard set,
  hand-check templates)
- `scripts/build_dataset.py`, `label_silver.py`, `agreement.py` (accuracy + kappa, `--make-templates`),
  `draft_hard.py` (quota-driven plan: category/urgency/sentiment/language/sentences/second issue/typos fixed per
  ticket, 10 tickets per call, `with_backoff`, resumable), `train_baseline.py`
- Data: `data/splits/` (24,554 deduplicated rows: 19,639 train / 2,455 dev / 2,460 test),
  `data/silver/{train,dev,test}.jsonl` (990 / 495 / 495, prompt 2026-10-11.2), `data/handcheck/{dev,test}_100.csv`,
  `data/hard/hard_draft.csv` (300 candidates: 48 urgent, 36 positive, 60 Turkish, 90 two-issue, 75 with typos,
  2–6 sentences, ≈$0.07) + `data/hard/REVIEW.md`
- `tests/test_core.py` (22 tests: FakeProvider, fake OpenAI client, backoff, Anthropic validation→repair,
  hand-check template), CI, Makefile,
  requirements split (core / full / dev / train)
- NOT yet: `ft_engine.py`, `scripts/train.py`, `export_onnx.py`, `eval.py`, `bench_latency.py`, notebook, FastAPI
  routes, Gradio, Dockerfile.

**Next (M1 3/3):** Davut fills `urgency`, `sentiment` (and optionally `note`) in `data/handcheck/dev_100.csv` and
`test_100.csv` without looking at the silver columns, then `python scripts/agreement.py` (target κ ≥ 0.75 per task;
below that, refine the rubric once, bump `PROMPT_VERSION`, relabel). Hard set: Davut edits and labels the 300
candidates in `data/hard/hard_draft.csv` following `data/hard/REVIEW.md` and saves `data/hard/hard.csv`. Then M2
(LLM engines + baseline, eval on dev only).

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
