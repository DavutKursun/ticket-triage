# ticket-triage — fine-tune your own model, or just call an API?

Classifies customer-support messages into **category · urgency · sentiment** (JSON) so a
support team can route and prioritise automatically — built **two ways and compared honestly**:
an LLM with structured outputs vs. a small multilingual transformer fine-tuned on a free GPU and
served on CPU. The README answers the question every client asks, with measured numbers.

> **Live demo:** _coming in Milestone 5_ · **Model:** _Hugging Face Hub link, Milestone 3_

## Results (test set · hard set) — filled in Milestone 4

| approach | category macro-F1 (test / hard) | urgency F1 | sentiment F1 | p50 latency | cost / 1,000 tickets |
|---|---|---|---|---|---|
| TF-IDF + logistic regression | – | – | – | – | $0 |
| LLM zero-shot | – | – | – | – | – |
| LLM few-shot (+ prompt cache) | – | – | – | – | – |
| Fine-tuned xlm-roberta-base (ONNX int8, CPU) | – | – | – | – | $0 marginal |

Silver-label quality (urgency / sentiment vs. 200 hand-checked): accuracy – / kappa –.

**Decision table** — _API wins when… / fine-tuned wins when…_ (Milestone 4).

## Dev results (silver labels) — preliminary, Milestone 2

495 silver-labelled dev messages (45 per category), the same rows for every engine. Category is scored against
Bitext gold; urgency and sentiment against silver labels from `claude-haiku-5-5`, so the LLM rows there are
self-agreement and optimistic. Test set and hard set stay untouched until Milestone 4.

| engine | category acc / macro-F1 | urgency macro-F1 | sentiment macro-F1 | p50 / p95 latency | tokens in / out per call | $ per 1,000 tickets: sync · cached · batch |
|---|---|---|---|---|---|---|
| TF-IDF + logistic regression (1.8 MB, trains in 1.7 s) | 1.000 / 1.000 | 0.676 | 0.871 | 3 / 3 ms | – | $0 marginal |
| LLM zero-shot (`claude-haiku-5-5`) | 0.986 / 0.986 | 0.659 | 0.883 | 1.3 / 2.1 s | 1,358 / 77 | $0.17 · – · $0.09 |
| LLM few-shot + prompt cache (22 train examples) | 0.988 / 0.988 | 0.721 | 0.775 | 1.0 / 1.6 s | 2,212 / 43 (2,127 cached) | $0.24 · $0.05 · $0.12 ($0.03 batch + cache) |

- Category on Bitext is solved by TF-IDF: the intents are templated. The LLMs' 6–7 errors are `order` → `delivery`
  confusions. Only the hard set can separate the approaches on this task.
- Urgency is the hard task for everyone (macro-F1 0.66–0.72): all engines under-predict `medium` relative to the
  silver labels. Few-shot examples help urgency (+0.06) but made sentiment noisier (−0.11, symmetric
  negative ↔ neutral swaps).
- Caching cut the few-shot price 4.7× ($0.24 → $0.05 per 1,000; 99.8% cache hits) and makes it cheaper than
  zero-shot. The LLM answers in about a second; the baseline in 3 ms. The whole LLM evaluation (990 calls) cost $0.11.

## How it works
1. **Data** — Bitext customer-support dataset (27k messages, 11 categories, CDLA-Sharing-1.0),
   deduplicated, stratified 80/10/10. Urgency and sentiment do not exist in the data: they are
   **silver labels** from an LLM with a written rubric, and their quality is measured against 200
   hand-checked examples. A 300-message **hard set** (long, multi-issue, typos, 20% Turkish) is
   hand-written and hand-labelled, because the public data is templated and too easy.
2. **Four engines** — TF-IDF baseline · LLM zero-shot · LLM few-shot · fine-tuned encoder
   (multi-task heads, exported to ONNX int8).
3. **One evaluation** on the held-out test set and the hard set; latency on the same CPU; cost
   from measured tokens × list prices.
4. **Serving** — FastAPI `POST /classify` (+ `/classify/batch`, `?engine=`), Gradio demo with a
   side-by-side button, Docker on Hugging Face Spaces.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                     # Groq free tier (LLM_PROVIDER=openai_compat) or Claude (LLM_PROVIDER=anthropic)
python scripts/build_dataset.py          # data/splits
python scripts/label_silver.py --split dev --n 1000 --delay 1     # resumable; stops on daily quota
python scripts/agreement.py --make-templates                       # hand-check CSVs
python scripts/train_baseline.py
python scripts/eval.py --split dev --engines llm_zero,llm_few,baseline --delay 0.5   # resumable; eval/results/
pytest -q
```

## Repository
`ticket_triage/` — schema, rubric (versioned prompts), providers (Groq / Claude), llm_engine, baseline, costs, data ·
`scripts/` — build_dataset, label_silver, agreement, draft_hard, train_baseline, eval (+ train / export_onnx / bench_latency from M3) ·
`data/` — splits (attributed), silver labels, hand-check CSVs, hard set · `docs/DESIGN.md` · `tests/`

## License
Code MIT. Data derived from Bitext is CDLA-Sharing-1.0 (see `data/splits/ATTRIBUTION.md`).
