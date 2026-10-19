.PHONY: setup setup-dev test data silver baseline eval-dry eval-dev synth translate train-smoke export-smoke notebook-smoke eval-ft-dev app

PY_TRAIN ?= .venv-train/bin/python
FT ?= DavutKursun/ticket-triage-xlmr

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
synth:            ## synthetic train-only rows (minority + control plans), verified by the silver labeller
	$(PY) scripts/make_synth.py
	$(PY) scripts/make_synth.py --plan control
translate:        ## 300 silver train rows translated into Turkish (train only)
	$(PY) scripts/translate_train.py
train-smoke:      ## tiny model, 64 rows, 1 epoch on CPU — proves the training loop end to end (needs .venv-train)
	$(PY_TRAIN) scripts/train.py --tiny --limit 64 --epochs 1 --out models/smoke
export-smoke: train-smoke  ## export the smoke model: fp32 + int8 ONNX, parity report (needs .venv-train)
	$(PY_TRAIN) scripts/export_onnx.py models/smoke --limit 200 --latency-n 40
notebook-smoke:   ## run the Kaggle notebook's cells locally on CPU with the tiny model (no clone, pip or push)
	$(PY_TRAIN) scripts/notebook_smoke.py
eval-ft-dev:      ## fine-tuned int8 ONNX engine on the 495 silver dev rows; FT = exported folder or Hub repo id
	$(PY) scripts/eval.py --split dev --engines finetuned --ft-model $(FT)
app:
	$(PY) -m uvicorn app:app --port 7860
