"""Fine-tuning helpers: label masking, loss masking, class weights, oversampling. Skipped when torch is absent
(the main venv and CI stay torch-free); run them in .venv-train."""

from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from ticket_triage.ft_model import IGNORE, LABELS, TriageModel, class_weights, label_ids, multitask_loss, oversample_indices, tiny_config  # noqa: E402


def test_label_ids_mask_missing_urgency_and_sentiment():
    rows = [{"category": "refund", "urgency": "high", "sentiment": "negative"},
            {"category": "order", "urgency": "", "sentiment": ""},
            {"category": "account", "urgency": "Urgent ", "sentiment": None}]
    ids = label_ids(rows)
    assert ids.shape == (3, 3) and ids.dtype == np.int64
    assert ids[0].tolist() == [LABELS["category"].index("refund"), LABELS["urgency"].index("high"), LABELS["sentiment"].index("negative")]
    assert ids[1].tolist() == [LABELS["category"].index("order"), IGNORE, IGNORE]
    assert ids[2].tolist() == [LABELS["category"].index("account"), LABELS["urgency"].index("urgent"), IGNORE]
    with pytest.raises(ValueError, match="billing"):
        label_ids([{"category": "billing"}])


def test_multitask_loss_ignores_masked_rows_and_never_nans():
    torch.manual_seed(0)
    logits = tuple(torch.randn(4, n, requires_grad=True) for n in (11, 4, 3))
    labels = torch.tensor([[0, IGNORE, 2], [3, IGNORE, IGNORE], [5, IGNORE, 0], [7, IGNORE, IGNORE]])
    loss, parts = multitask_loss(logits, labels)
    assert torch.isfinite(loss) and float(parts["urgency"].detach()) == 0.0  # urgency fully masked contributes nothing, no NaN
    expected_sent = torch.nn.functional.cross_entropy(logits[2][[0, 2]], torch.tensor([2, 0]))
    assert torch.allclose(parts["sentiment"], expected_sent)
    assert torch.allclose(loss, parts["category"] + parts["sentiment"])
    loss.backward()
    assert torch.isfinite(logits[0].grad).all() and float(logits[1].grad.abs().sum()) == 0.0
    all_masked = torch.full((4, 3), IGNORE)
    all_masked[:, 0] = 1
    loss2, _ = multitask_loss(tuple(torch.randn(4, n) for n in (11, 4, 3)), all_masked)
    assert torch.isfinite(loss2)


def test_class_weights_inverse_sqrt_normalised():
    col = np.array([0, 0, 0, 0, 1, IGNORE, IGNORE])
    w = class_weights(col, 4)
    assert w.shape == (4,) and pytest.approx(float(w.mean()), abs=1e-6) == 1.0
    assert w[0] < w[1] and pytest.approx(float(w[1] / w[0]), rel=1e-6) == 2.0  # 1/sqrt(1) vs 1/sqrt(4)
    assert float(w[2]) == float(w[3])  # absent classes share the neutral weight
    assert class_weights(np.array([IGNORE, IGNORE]), 3).tolist() == [1.0, 1.0, 1.0]


def test_oversample_indices_repeats_labelled_rows_only():
    labelled = np.array([False, True, False, True, True, False, False, False, False, False])
    idx = oversample_indices(labelled, factor=4, rng=random.Random(1))
    assert len(idx) == 10 + 3 * 3
    counts = np.bincount(idx, minlength=10)
    assert counts[[1, 3, 4]].tolist() == [4, 4, 4] and counts[[0, 2, 5, 6, 7, 8, 9]].tolist() == [1] * 7
    again = oversample_indices(labelled, factor=4, rng=random.Random(2))
    assert sorted(again) == sorted(idx) and again != idx  # same multiset, different order per epoch
    assert len(oversample_indices(labelled, factor=1, rng=random.Random(0))) == 10


class _FakeTok:
    pad_token_id = 1

    def __len__(self):
        return 1005


def test_tiny_config_and_model_shapes():
    cfg = tiny_config(_FakeTok())
    assert (cfg.vocab_size, cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.intermediate_size, cfg.max_position_embeddings, cfg.pad_token_id) == (1005, 32, 2, 2, 64, 130, 1)
    torch.manual_seed(0)
    model = TriageModel(tokenizer=_FakeTok(), tiny=True)
    ids = torch.randint(5, 1000, (3, 20))
    logits = model(ids, torch.ones_like(ids))
    assert [tuple(x.shape) for x in logits] == [(3, 11), (3, 4), (3, 3)]
    assert sum(p.numel() for p in model.parameters()) < 200_000


def test_export_parity_tiny(tmp_path):
    """fp32 ONNX matches PyTorch, int8 runs with the same shapes, dynamic batch and sequence axes work."""
    pytest.importorskip("onnxruntime")
    import sys
    from pathlib import Path

    from transformers import PreTrainedTokenizerFast

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import export_onnx as ex

    tok = PreTrainedTokenizerFast(tokenizer_file=str(root / "tests" / "fixtures" / "tiny_onnx" / "tokenizer.json"), bos_token="<s>",
                                  eos_token="</s>", unk_token="<unk>", pad_token="<pad>", cls_token="<s>", sep_token="</s>", mask_token="<mask>")
    torch.manual_seed(0)
    model = TriageModel(tokenizer=tok, tiny=True).eval()
    info = ex.export_fp32(model, tok, tmp_path / "model.onnx", 17, 128)
    assert info["shape_check_max_abs_diff"] < 1e-4
    texts = ["refund please", "Siparişim hâlâ gelmedi, iade istiyorum!", "i was charged twice for order 48213, i need it back today " * 4]
    ref = ex.torch_logits(model, tok, texts, 128)
    assert ex.max_abs(ref, ex.ort_logits(ex.ort_session(tmp_path / "model.onnx"), tok, texts, 128))["all"] < 1e-4
    ex.quantize(tmp_path / "model.onnx", tmp_path / "model.int8.onnx", "QInt8", False)
    q = ex.ort_logits(ex.ort_session(tmp_path / "model.int8.onnx"), tok, texts, 128)
    assert {t: v.shape for t, v in q.items()} == {t: v.shape for t, v in ref.items()} == {"category": (3, 11), "urgency": (3, 4), "sentiment": (3, 3)}
    assert (tmp_path / "model.int8.onnx").stat().st_size < (tmp_path / "model.onnx").stat().st_size
