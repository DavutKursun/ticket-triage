.PHONY: setup setup-dev test data silver baseline eval-dry app

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
app:
	$(PY) -m uvicorn app:app --port 7860
