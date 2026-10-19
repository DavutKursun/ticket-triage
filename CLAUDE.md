# ticket-triage — support ticket classification: fine-tuned model vs LLM, with real costs

This file is the build spec. Read it fully before writing code. Work milestone by
milestone; after each milestone stop, show how to run it, and wait for Davut's OK.
Section 0 says where we are and what changed from the original plan.

## 0. Status (updated 2026-10-19)

**M3 part 2 of 3 is done (incl. 2b: short synthetic rows, max_len 192): de-biased train augmentation, ONNX export, the
ONNX serving engine and the Kaggle notebook, all smoke-tested locally end to end (32 tests in the main venv, 38 in
`.venv-train`, no network). Next is M3-3: Davut
runs the notebook on Kaggle (runbook in "Next"), then the dev evaluation of the fine-tuned engine. Milestone 1's last
third (Davut's hand-check and hard-set pass) runs in parallel.**

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
- LLM engines A/B run on whatever `LLM_PROVIDER` selects — currently `claude-haiku-5-5`, as the original spec
  intended; the M2 dev numbers are Haiku's. Few-shot uses `cache_control` on the system block (2.1k tokens).
- The README cost table quotes list prices for `claude-haiku-5-5` from measured tokens; a free-tier model is
  labelled "free tier", never "$0.0000".
- Model IDs verified 2026-10-09: `claude-haiku-5-5` ($0.10/$0.50 per MTok), `claude-sonnet-5-5` ($2/$10). Never
  downgrade to older IDs without asking.
- Labels are skewed because Bitext is templated: across 1,980 rows there are 0 `urgent` and 1 `positive`
  (urgency ≈73% low / 22% medium / 7% high; sentiment ≈89% neutral). The hard set must supply those classes for
  evaluation; for training, `scripts/make_synth.py` added 272 TRAIN-ONLY synthetic rows (108 urgent, 77 high,
  87 positive; Haiku-written from the rubric, kept only when the silver labeller agrees blind; 88 rejected; $0.12)
  in `data/silver/train_synth.jsonl` (`source="synthetic"`, plan "minority"). Never in dev, test or the hard set.
- M3-2 removed a language/length shortcut those rows created (Turkish rows were 69% urgent+high vs 16% English; only
  32% of >16-word rows were low/medium): plan "control" added 109 long calm low/medium rows (240 planned, the labeller
  disagreed on low vs medium for most rejects; $0.085) and `scripts/translate_train.py` 300 Turkish translations of
  silver train rows (labels carried over, 60 without Turkish characters; $0.012). After: Turkish 18% vs English 15%
  urgent+high, >16-word rows 51% low/medium (`eval/results/train_confounds.md`).
- M3-2b: plan "short" added 66 one-line rows of 6–16 words (34 urgent, 13 high, 19 positive; 90 planned; $0.030), so
  ≤16-word rows are no longer never urgent (0 → 34 urgent). Targets still hold: Turkish 21% vs English 17% urgent+high.
- Fine-tuning runs in a separate `.venv-train` (`requirements-train.txt`, CPU torch, pinned, no optimum); torch never
  enters the main venv, and `tests/test_ft.py` skips there. Local smoke tests use `train.py --tiny` (2-layer random
  XLM-R built from the tokenizer-only `hf-internal-testing/tiny-random-xlm-roberta`); Kaggle uses
  `--model-name xlm-roberta-base` and installs `requirements-kaggle.txt` (never torch). max_len default 192
  (`TrainConfig`, carried by ft_config.json into export and the engine): headroom for six-sentence hard-set tickets;
  2 train rows exceed 128 tokens, none exceed 192 (`eval/results/train_token_lengths.md`).
- ONNX export uses `torch.onnx.export` directly (the default dynamo exporter needs onnxscript, so it falls back to
  `dynamo=False`) + onnxruntime `quantize_dynamic`; optimum is not used. Serving needs only onnxruntime + tokenizers.
- Hub repo `DavutKursun/ticket-triage-xlmr` (public, empty until the Kaggle push) was created with Davut's existing
  login; the Kaggle push uses a separate fine-grained token stored only as a Kaggle Secret.

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
  ticket, 10 tickets per call, `with_backoff`, resumable), `train_baseline.py` (timing, size, dev metrics),
  `eval.py` (resumable per-engine predictions, macro-F1 over classes present in the truth, p50/p95, $/1,000
  sync · cached · batch; refuses `--split test` until `data/hard/hard.csv` exists and `--final` is passed)
- M2 dev results (495 silver rows, `eval/results/dev-summary.md`, README table): baseline category 1.000 /
  urgency F1 0.676 / sentiment F1 0.871 / 3 ms; zero-shot 0.986 / 0.659 / 0.883 / 1.3 s p50 / $0.17 per 1,000;
  few-shot 0.988 / 0.721 / 0.775 / 1.0 s / $0.24 sync → $0.05 cached (99.8% hits). 990 calls, $0.11, 0 repairs.
- Data: `data/splits/` (24,554 deduplicated rows: 19,639 train / 2,455 dev / 2,460 test),
  `data/silver/{train,dev,test}.jsonl` (990 / 495 / 495, prompt 2026-10-11.2), `data/handcheck/{dev,test}_100.csv`,
  `data/hard/hard_draft.csv` (300 candidates: 48 urgent, 36 positive, 60 Turkish, 90 two-issue, 75 with typos,
  2–6 sentences, ≈$0.07) + `data/hard/REVIEW.md`
- `ticket_triage/ft_model.py` (TriageModel = encoder + dropout + heads 11/4/3 on the first token; labels -100 where
  missing; loss = CE + masked CE + masked CE with inverse-sqrt class weights; labelled rows oversampled ×4; AdamW
  2e-5, 6% warmup, fp16 on CUDA, batch 32, 3 epochs, seed 7; per-epoch dev eval, best epoch by mean macro-F1;
  saves safetensors + encoder config + tokenizer + labels.json + ft_config.json + dev_metrics.json; `load_model`;
  hard-dev metrics per epoch when `data/hard/hard.csv` exists, report only), `scripts/train.py` (thin wrapper,
  `--tiny`, `--no-synth`, `--no-translated`, writes `train_meta.json`), `scripts/make_synth.py` (plans minority,
  control, short), `translate_train.py`, `train_confounds.py`, `token_lengths.py`. Real run: 20,386 train rows, 1,737
  labelled, effective epoch 25,597 rows = 800 steps, 2,400 steps for 3 epochs.
- `scripts/export_onnx.py` (fp32 to `fp32/`, int8 to `onnx/` with tokenizer.json + labels.json + ft_config.json,
  QInt8 → per-channel → QUInt8 ladder at 98% category agreement, `export_report.json` with parity, sizes, tokenizer
  parity, CPU latency), `ticket_triage/ft_engine.py` (FTEngine: local folder or Hub repo id, `classify` /
  `classify_batch`, TriageResponse + per-head probabilities, `FT_THREADS`), engine `finetuned` in `eval.py`
  (`--engine finetuned --ft-model …`), `scripts/model_card.py`, `notebooks/train_kaggle.ipynb` (+ `scripts/notebook_smoke.py`
  runs its cells locally), `requirements-kaggle.txt`. Tiny smoke export: fp32 max |Δlogit| 4.8e-7, int8 agreement
  100%, tokenizer parity 607/607 (406 Turkish), serving folder 0.52 MB.
- `tests/test_core.py` (24 tests), `tests/test_ft_engine.py` (8 tests on the committed 0.52 MB fixture
  `tests/fixtures/tiny_onnx/`: schema, probabilities sum to 1, batch == single, 5,000-character input, Turkish, empty
  input, tokenizer ids vs transformers, folder resolution), `tests/test_ft.py` (7 torch tests incl. export parity;
  skipped without torch), CI, Makefile (`synth`, `translate`, `train-smoke`, `export-smoke`, `notebook-smoke`,
  `eval-ft-dev`), requirements split (core / full / dev / train / kaggle)
- NOT yet: the real training run, `bench_latency.py`, fine-tuned dev results, FastAPI routes, Gradio, Dockerfile.

**Next (M3-3): the real run on Kaggle, click by click (Davut)**
1. One time, Hugging Face: huggingface.co → avatar → Settings → Access Tokens → **Create new token** → type
   **Fine-grained** → name `kaggle-ticket-triage` → under *Repositories permissions* search and add
   `DavutKursun/ticket-triage-xlmr` and tick **Write access to contents/settings of selected repos** → Create →
   copy the token (shown once; paste it only into Kaggle, never into a chat or a file).
2. One time, Kaggle: the account must be phone-verified (Settings → Phone verification), otherwise GPU and Internet stay
   greyed out.
3. kaggle.com → **Create → New Notebook** → menu **File → Import Notebook** → **GitHub** tab (or *Link*) → paste
   `https://github.com/DavutKursun/ticket-triage/blob/main/notebooks/train_kaggle.ipynb` → **Import**.
4. Right sidebar → **Session options**: **Accelerator → GPU T4 x2** (one GPU is used), **Internet → On**.
5. Menu **Add-ons → Secrets → Add Secret**: Label `HF_TOKEN`, Value = the token → Save, and make sure the
   **Attached** box next to `HF_TOKEN` is ticked for this notebook.
6. Top right **Save Version** → Version type **Save & Run All (Commit)** → **Save**. Close the tab if you like: it runs
   headless. Expected runtime 20–35 min (pip ~2, model download ~1, training ~10–15 on one T4, CPU export ~5,
   push ~2); it uses well under 1 h of the 30 h weekly GPU quota.
7. Done when the version shows **Complete** and the last cell prints `pushed to https://huggingface.co/DavutKursun/ticket-triage-xlmr`.
   Check the repo has `onnx/model.int8.onnx`, `README.md`, `dev_metrics.json`, `export_report.json`. If the secret was
   missing the cell lists the files to download from the version's **Output** tab (folder `model/`) instead.
   On an error: open the version → **Logs**, copy the failing cell's output for Claude.
8. Locally (main venv, no torch): `make eval-ft-dev`, i.e.
   `.venv/bin/python scripts/eval.py --split dev --engine finetuned --ft-model DavutKursun/ticket-triage-xlmr` —
   downloads only `onnx/` (int8, ≈ 280 MB) into the Hugging Face cache and scores the 495 silver dev rows; results in
   `eval/results/dev-finetuned.json`. Without a push: put the downloaded `model/onnx/` folder at `models/ft/onnx` and
   pass `--ft-model models/ft/onnx`. Then M3-3 continues with `bench_latency.py`, dev results in the README and DESIGN.md.
**In parallel (M1 3/3, Davut):** fill `data/handcheck/{dev,test}_100.csv` without looking at the silver columns →
`python scripts/agreement.py` (target κ ≥ 0.75; below that refine the rubric once, bump `PROMPT_VERSION`, relabel);
edit/label `data/hard/hard_draft.csv` per `data/hard/REVIEW.md` → `data/hard/hard.csv`. Never stage those files from
Claude's side. The test split and the hard set's test half stay untouched until M4.

macOS notes: Python 3.11 venvs; torch lives only in `.venv-train` (CPU wheels) — never install requirements-train.txt
into the main `.venv`.

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
