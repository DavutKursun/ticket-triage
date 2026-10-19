"""Serving engine (int8 ONNX + tokenizers + numpy, no torch) on the committed tiny fixture tests/fixtures/tiny_onnx/,
an export of the 2-layer random smoke model (scripts/export_onnx.py on models/smoke). Runs in the main venv and CI."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")

from ticket_triage.ft_engine import FTEngine, resolve_folder  # noqa: E402
from ticket_triage.schema import CATEGORIES, SENTIMENTS, URGENCIES, TriageResponse  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_onnx"
TEXTS = ["I was charged twice for order #48213 and need the duplicate refunded today.",
         "How do I change my delivery address?",
         "Siparişim hâlâ gelmedi, iade istiyorum! Lütfen yardım edin.",
         "Siparisim hala gelmedi, iade istiyorum",  # Turkish typed without Turkish characters
         "thanks a lot for the quick help with my invoice :)"]
# int8 dynamic quantisation scales activations per batch, so batched probabilities may differ in the last decimals
BATCH_TOL = 0.01  # measured 5.5e-4 on the fixture


@pytest.fixture(scope="module")
def eng() -> FTEngine:
    return FTEngine(FIXTURE, threads=1)


def test_fixture_is_small_and_complete():
    files = {p.name: p.stat().st_size for p in FIXTURE.iterdir() if p.is_file()}
    assert {"model.int8.onnx", "tokenizer.json", "labels.json", "ft_config.json"} <= set(files)
    assert sum(files.values()) < 1_000_000


def test_schema_and_probabilities(eng):
    r = eng.classify(TEXTS[0])
    assert isinstance(r, TriageResponse) and r.engine == "finetuned" and "model.int8.onnx" in r.model and r.latency_ms >= 0
    assert r.category in CATEGORIES and r.urgency in URGENCIES and r.sentiment in SENTIMENTS
    assert {t: list(d) for t, d in r.probabilities.items()} == {"category": list(CATEGORIES), "urgency": list(URGENCIES), "sentiment": list(SENTIMENTS)}
    for dist in r.probabilities.values():
        assert sum(dist.values()) == pytest.approx(1.0, abs=1e-9) and all(0.0 <= v <= 1.0 for v in dist.values())
    assert r.confidence == pytest.approx(max(r.probabilities["category"].values())) == pytest.approx(r.probabilities["category"][r.category])
    assert r.probabilities["urgency"][r.urgency] == max(r.probabilities["urgency"].values())


def test_batch_matches_single(eng):
    single = [eng.classify(t) for t in TEXTS]
    batch = eng.classify_batch(TEXTS, batch_size=3)
    assert [(b.category, b.urgency, b.sentiment) for b in batch] == [(s.category, s.urgency, s.sentiment) for s in single]
    for s, b in zip(single, batch, strict=True):
        for task, dist in s.probabilities.items():
            assert max(abs(dist[k] - b.probabilities[task][k]) for k in dist) < BATCH_TOL


def test_long_input_truncates_cleanly(eng):
    long = "My refund for order #48213 still has not arrived and nobody answers. " * 75
    assert len(long) >= 5000
    ids, mask = eng.encode([long])
    assert ids.shape == (1, eng.max_len) and int(mask.sum()) == eng.max_len
    assert eng.classify(long).category in CATEGORIES


def test_turkish_input(eng):
    for text in TEXTS[2:4]:
        r = eng.classify(text)
        assert r.category in CATEGORIES and sum(r.probabilities["sentiment"].values()) == pytest.approx(1.0)
    assert eng.encode([TEXTS[2]])[0].shape[1] > 2  # more than <s> </s>


def test_empty_input_rejected(eng):
    for bad in ("", "   \n", None):
        with pytest.raises(ValueError, match="empty"):
            eng.classify(bad)
    with pytest.raises(ValueError, match="empty"):
        eng.classify_batch(["refund please", " "])


def test_tokenizer_ids_match_transformers(eng):
    # expected ids from transformers' XLMRobertaTokenizerFast on this tokenizer.json (.venv-train, transformers 4.57.6)
    assert EXPECTED_IDS, "tests/fixtures/tiny_onnx/expected_ids.json is missing"
    for text, expected in EXPECTED_IDS.items():
        assert eng.tokenizer.encode(text).ids == expected


def test_resolve_folder(tmp_path):
    shutil.copytree(FIXTURE, tmp_path / "model" / "onnx")
    assert resolve_folder(tmp_path / "model") == tmp_path / "model" / "onnx"
    assert resolve_folder(tmp_path / "model" / "onnx") == tmp_path / "model" / "onnx"
    with pytest.raises(FileNotFoundError):
        resolve_folder(tmp_path)
    (tmp_path / "models").mkdir()
    with pytest.raises(FileNotFoundError, match="no exported model"):  # first component exists locally: never a Hub lookup
        resolve_folder(str(tmp_path / "models" / "missing"))


EXPECTED_IDS: dict[str, list[int]] = {}  # filled below from tests/fixtures/tiny_onnx/expected_ids.json
_exp = FIXTURE / "expected_ids.json"
if _exp.exists():
    import json

    EXPECTED_IDS.update(json.loads(_exp.read_text(encoding="utf-8")))
