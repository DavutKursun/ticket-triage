"""Approach C at serving time: the fine-tuned multi-task encoder as int8 ONNX. Needs only onnxruntime, tokenizers and
numpy (no torch, no transformers), so the serving image stays small.

Source: a local folder (the exported ``onnx/`` folder, or a model folder that contains ``onnx/``) or a Hugging Face
repo id, from which only ``onnx/*`` is downloaded (huggingface_hub.snapshot_download). Threads: FT_THREADS (default:
onnxruntime decides). Inputs are truncated to the ``max_len`` the model was trained with (ft_config.json), so long
messages never fail; empty ones are rejected. int8 dynamic quantisation scales activations per batch, so a batched
result can differ from the single-message result in the last decimals.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from .config import MODELS, settings
from .schema import TriageResponse

TASKS = ("category", "urgency", "sentiment")


def resolve_folder(source: str | Path, revision: str | None = None) -> Path:
    """Local folder with ft_config.json + an .onnx file (or its onnx/ subfolder), else a Hub repo id."""
    p = Path(source).expanduser()
    for cand in (p / "onnx", p):
        if (cand / "ft_config.json").is_file() and any(cand.glob("*.onnx")):
            return cand
    s = str(source)
    local_looking = p.exists() or s.startswith((".", "/", "~")) or s.count("/") != 1 or Path(s.split("/")[0]).is_dir()
    if local_looking:
        raise FileNotFoundError(f"no exported model in {s!r}: expected ft_config.json and an .onnx file (or an onnx/ subfolder), "
                                "or a Hugging Face repo id like owner/name")
    from huggingface_hub import snapshot_download

    try:
        local = Path(snapshot_download(repo_id=s, revision=revision, allow_patterns=["onnx/*"], token=settings.hf_token or None)) / "onnx"
    except Exception as e:  # noqa: BLE001 — not found, no network, gated: one clear error for callers
        raise FileNotFoundError(f"could not download {s!r}/onnx from the Hugging Face Hub: {type(e).__name__}: {str(e).splitlines()[0][:160]}") from e
    if not (local / "ft_config.json").is_file():
        raise FileNotFoundError(f"{s!r} has no exported model yet (onnx/ft_config.json missing on the Hub)")
    return local


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class FTEngine:
    """Same response shape as the LLM engine (TriageResponse) plus per-head probabilities."""

    name = "finetuned"

    def __init__(self, source: str | Path | None = None, onnx_file: str | None = None, threads: int | None = None, revision: str | None = None) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        source = source or settings.ft_model_id or MODELS / "ft" / "onnx"
        self.folder = resolve_folder(source, revision)
        self.config = json.loads((self.folder / "ft_config.json").read_text(encoding="utf-8"))
        self.labels: dict[str, list[str]] = json.loads((self.folder / "labels.json").read_text(encoding="utf-8"))
        self.max_len = int(self.config["max_len"])
        self.tokenizer = Tokenizer.from_file(str(self.folder / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=self.max_len)
        self.pad_id = int(self.config.get("pad_token_id", self.tokenizer.token_to_id("<pad>") or 0))
        self.onnx_file = onnx_file or self.config.get("onnx_file", "model.int8.onnx")
        so = ort.SessionOptions()
        threads = threads or int(os.environ.get("FT_THREADS", "0") or 0) or None
        if threads:
            so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        self.session = ort.InferenceSession(str(self.folder / self.onnx_file), so, providers=["CPUExecutionProvider"])
        self.output_names = [o.name for o in self.session.get_outputs()]
        if sorted(self.output_names) != sorted(TASKS):
            raise ValueError(f"unexpected ONNX outputs {self.output_names}; expected {TASKS}")
        label = str(source) if not Path(str(source)).exists() else self.folder.parent.name if self.folder.name == "onnx" else self.folder.name
        self.model_name = f"{label} ({self.onnx_file})"

    # ------------------------------------------------------------------ core
    @staticmethod
    def _check(text) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("empty text")
        return text.strip()

    def encode(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        encs = self.tokenizer.encode_batch(texts)
        width = max(len(e.ids) for e in encs)
        ids = np.full((len(encs), width), self.pad_id, dtype=np.int64)
        mask = np.zeros((len(encs), width), dtype=np.int64)
        for k, e in enumerate(encs):
            ids[k, : len(e.ids)] = e.ids
            mask[k, : len(e.ids)] = 1
        return ids, mask

    def predict_proba(self, texts: list[str]) -> dict[str, np.ndarray]:
        ids, mask = self.encode(texts)
        outs = self.session.run(self.output_names, {"input_ids": ids, "attention_mask": mask})
        return {name: _softmax(np.asarray(x)) for name, x in zip(self.output_names, outs, strict=True)}

    def _response(self, probs: dict[str, np.ndarray], k: int, latency_ms: float) -> TriageResponse:
        best = {t: int(probs[t][k].argmax()) for t in TASKS}
        dist = {t: {lab: float(probs[t][k][j]) for j, lab in enumerate(self.labels[t])} for t in TASKS}
        return TriageResponse(category=self.labels["category"][best["category"]], urgency=self.labels["urgency"][best["urgency"]],
                              sentiment=self.labels["sentiment"][best["sentiment"]], confidence=min(1.0, float(probs["category"][k].max())),
                              model=self.model_name, engine="finetuned", latency_ms=round(latency_ms, 2), probabilities=dist)

    # ------------------------------------------------------------------ public
    def classify(self, text: str) -> TriageResponse:
        text = self._check(text)
        t0 = time.perf_counter()
        probs = self.predict_proba([text])
        return self._response(probs, 0, (time.perf_counter() - t0) * 1000)

    def classify_batch(self, texts: list[str], batch_size: int = 32) -> list[TriageResponse]:
        """Order preserved; latency_ms per item = its sub-batch's wall time / sub-batch size."""
        texts = [self._check(t) for t in texts]
        out: list[TriageResponse] = []
        for b in range(0, len(texts), max(1, batch_size)):
            chunk = texts[b : b + batch_size]
            t0 = time.perf_counter()
            probs = self.predict_proba(chunk)
            per = (time.perf_counter() - t0) * 1000 / len(chunk)
            out += [self._response(probs, k, per) for k in range(len(chunk))]
        return out

    respond = classify
