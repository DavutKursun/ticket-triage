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
cp .env.example .env                     # free Groq key by default; no paid API needed
python scripts/build_dataset.py          # data/splits
python scripts/label_silver.py --split dev --n 1000 --delay 1     # resumable; stops on daily quota
python scripts/agreement.py --make-templates                       # hand-check CSVs
python scripts/train_baseline.py
pytest -q
```

## Repository
`ticket_triage/` — schema, rubric (versioned prompts), providers (Groq / Claude), llm_engine, baseline, costs, data ·
`scripts/` — build_dataset, label_silver, agreement, draft_hard, train_baseline (+ train / export_onnx / eval / bench_latency from M3) ·
`data/` — splits (attributed), silver labels, hand-check CSVs, hard set · `docs/DESIGN.md` · `tests/`

## License
Code MIT. Data derived from Bitext is CDLA-Sharing-1.0 (see `data/splits/ATTRIBUTION.md`).
