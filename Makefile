.PHONY: setup setup-dev test data silver baseline eval-dry eval-dev synth train-smoke app

PY_TRAIN ?= .venv-train/bin/python

PY ?= python

setup:
	$(PY) -m pip install -r requirements.txt
setup-dev:
	$(PY) -m pip install -r requirements-dev.txt
test:
	$(PY) -m pytest -q
data:             ## download Bitext, dedupe, split 80/10/10 → data/splits
	$(PY) scripts/build_dataset.py
silver:           ## label urgency+sentiment with the LLM (resumable; stops on daily quota)
	$(PY) scripts/label_silver.py --n 2000 --delay 1
baseline:         ## TF-IDF + logistic regression on dev
	$(PY) scripts/train_baseline.py
eval-dry:
	$(PY) scripts/eval.py --split dev --engines baseline --dry-run
eval-dev:         ## zero-shot, few-shot (cached) and baseline on the 495 silver-labelled dev rows
	$(PY) scripts/eval.py --split dev --engines llm_zero,llm_few,baseline --delay 0.5
synth:            ## synthetic train-only rows for urgent/high/positive, verified by the silver labeller
	$(PY) scripts/make_synth.py
train-smoke:      ## tiny model, 64 rows, 1 epoch on CPU — proves the training loop end to end (needs .venv-train)
	$(PY_TRAIN) scripts/train.py --tiny --limit 64 --epochs 1 --out models/smoke
app:
	$(PY) -m uvicorn app:app --port 7860
