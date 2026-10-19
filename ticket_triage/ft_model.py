"""Approach C: xlm-roberta-base with three classification heads (multi-task), shared by scripts/train.py and the
Kaggle notebook. A plain PyTorch loop (no Trainer) so label masking and oversampling stay explicit.

Data (data.train_rows): category is gold on every train row; urgency/sentiment are silver on ~990 train rows plus the
train-only synthetic rows (data/silver/train_synth.jsonl, plans "minority" and "control") and the train-only Turkish
translations (data/silver/train_translated.jsonl). Missing labels are -100 and ignored by the loss. Rows that carry
urgency/sentiment are oversampled ×cfg.oversample per epoch so the two small heads get enough updates.
Loss = CE(category) + CE(urgency, w) + CE(sentiment, w) with w = inverse-sqrt class frequency normalised to mean 1
(cfg.class_weights=False disables). After each epoch: category on the full dev split, urgency/sentiment on the silver
dev rows, macro-F1 over the classes present in the truth; the best epoch is the mean of the three macro-F1s. When
data/hard/hard.csv exists, its dev half is scored every epoch too (report only, never used for selection).
Never reads the test split or the hard set's test half.
Saved to cfg.out: model.safetensors (model.pt fallback), encoder/config.json, tokenizer/, labels.json, ft_config.json,
dev_metrics.json.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, XLMRobertaConfig, get_linear_schedule_with_warmup

from .config import MODELS
from .schema import CATEGORIES, SENTIMENTS, URGENCIES

TASKS = ("category", "urgency", "sentiment")
LABELS: dict[str, list[str]] = {"category": list(CATEGORIES), "urgency": list(URGENCIES), "sentiment": list(SENTIMENTS)}
IGNORE = -100


@dataclass
class TrainConfig:
    model_name: str = "xlm-roberta-base"
    epochs: int = 3
    max_len: int = 192  # headroom for 6-sentence tickets at inference; p99 of long train rows is 114 (train_token_lengths.md)
    batch_size: int = 32
    eval_batch_size: int = 64
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup: float = 0.06
    dropout: float = 0.1
    oversample: int = 4
    class_weights: bool = True
    seed: int = 7
    limit: int = 0
    no_synth: bool = False
    no_translated: bool = False
    out: str = str(MODELS / "ft")
    fp16: bool | None = None  # None = on when CUDA is available
    log_every: int = 50
    tiny: bool = False  # smoke tests: 2-layer random XLM-R sized to the tokenizer, no weights download


TINY_TOKENIZER = "hf-internal-testing/tiny-random-xlm-roberta"  # tokenizer-only repo (1,005 tokens)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- labels, weights, sampling
def label_ids(rows: list[dict]) -> np.ndarray:
    """(n, 3) int64: category index, urgency index or -100, sentiment index or -100."""
    ids = np.full((len(rows), 3), IGNORE, dtype=np.int64)
    for k, r in enumerate(rows):
        if r["category"] not in LABELS["category"]:
            raise ValueError(f"unknown category {r['category']!r}")
        ids[k, 0] = LABELS["category"].index(r["category"])
        for j, task in enumerate(("urgency", "sentiment"), start=1):
            v = (r.get(task) or "").strip().lower()
            if v in LABELS[task]:
                ids[k, j] = LABELS[task].index(v)
    return ids


def class_weights(col: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-sqrt frequency over the labelled rows, normalised to mean 1; absent classes get 1 (they never appear)."""
    valid = col[col != IGNORE]
    counts = np.bincount(valid, minlength=n_classes).astype(float) if len(valid) else np.zeros(n_classes)
    w = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1.0)), 1.0)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def oversample_indices(labelled: np.ndarray, factor: int, rng: random.Random) -> list[int]:
    """Every row once plus (factor - 1) extra copies of rows that carry urgency/sentiment labels; shuffled."""
    idx = list(range(len(labelled)))
    idx += [int(i) for i in np.flatnonzero(labelled) for _ in range(max(int(factor), 1) - 1)]
    rng.shuffle(idx)
    return idx


def masked_ce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    """Cross-entropy over rows whose target is not -100; a fully masked batch contributes 0 (never NaN)."""
    mask = target != IGNORE
    if not bool(mask.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask].float(), target[mask], weight=weight)


def multitask_loss(logits: tuple, labels: torch.Tensor, weights: dict[str, torch.Tensor] | None = None):
    parts = {}
    total = None
    for j, task in enumerate(TASKS):
        part = masked_ce(logits[j], labels[:, j], (weights or {}).get(task))
        parts[task] = part
        total = part if total is None else total + part
    return total, parts


# ----------------------------------------------------------------------------- model, data
def tiny_config(tokenizer) -> XLMRobertaConfig:
    """2-layer, 32-wide XLM-R sized to the tokenizer: trains in seconds on CPU, exercises every code path."""
    return XLMRobertaConfig(vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                            max_position_embeddings=130, pad_token_id=tokenizer.pad_token_id)


def _build_encoder(model_name: str | None, config=None, tokenizer=None, tiny: bool = False):
    """Pretrained encoder without the pooler; ``tiny`` builds tiny_config() from scratch (seeded by the caller) and
    never downloads weights; ``config`` rebuilds a saved model (load_model)."""
    if tiny and config is None:
        config = tiny_config(tokenizer)
        print(f"NOTE: --tiny: random 2-layer XLM-R encoder ({config.vocab_size} vocab, hidden 32) — smoke tests only", flush=True)
    if config is None:
        return AutoModel.from_pretrained(model_name, add_pooling_layer=False)
    try:
        return AutoModel.from_config(config, add_pooling_layer=False)
    except TypeError:  # architecture without a pooler flag
        return AutoModel.from_config(config)


class TriageModel(nn.Module):
    """Encoder + dropout + three linear heads on the first-token hidden state. forward -> (category, urgency, sentiment) logits."""

    def __init__(self, model_name: str | None = None, config=None, dropout: float = 0.1, tokenizer=None, tiny: bool = False) -> None:
        super().__init__()
        self.encoder = _build_encoder(model_name, config, tokenizer, tiny)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({t: nn.Linear(hidden, len(LABELS[t])) for t in TASKS})

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        h = self.dropout(h)
        return tuple(self.heads[t](h) for t in TASKS)


class TriageDataset(Dataset):
    def __init__(self, rows: list[dict], ids: np.ndarray) -> None:
        self.rows, self.ids = rows, ids

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, k: int):
        return self.rows[k]["text"], self.ids[k]


class Collator:
    def __init__(self, tokenizer, max_len: int) -> None:
        self.tokenizer, self.max_len = tokenizer, max_len

    def __call__(self, batch):
        texts, ids = zip(*batch, strict=True)
        enc = self.tokenizer(list(texts), truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": torch.tensor(np.stack(ids), dtype=torch.long)}


def _spread(rows: list, k: int) -> list:
    """k evenly spaced rows, so a --limit sample covers every source and category instead of the first block."""
    return rows[:: max(1, len(rows) // k)][:k] if k > 0 else []


def hard_dev_rows() -> list[dict]:
    """The hard set's DEV half with complete hand labels, or [] when data/hard/hard.csv does not exist yet."""
    from .data import load_hard

    out = []
    for r in load_hard("dev").itertuples():
        c, u, s = (str(v).strip().lower() for v in (r.category, r.urgency, r.sentiment))
        if c in LABELS["category"] and u in LABELS["urgency"] and s in LABELS["sentiment"] and str(r.text).strip():
            out.append({"text": r.text, "category": c, "urgency": u, "sentiment": s})
    return out


def build_rows(cfg: TrainConfig) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Train rows (data.train_rows: Bitext + train-only synthetic/translated), dev category rows (full split), dev
    urgency/sentiment rows (silver), hard-dev rows (hand labels, report only; [] without hard.csv). Never touches the
    test split or the hard set's test half. --limit keeps an evenly spread labelled/unlabelled mix."""
    from .data import labelled, load_split, train_rows

    rows = train_rows(synth=not cfg.no_synth, translated=not cfg.no_translated)
    dev_cat = [{"text": r.text, "category": r.category} for r in load_split("dev").itertuples()]
    dev_us = [{"text": r.text, "urgency": r.urgency, "sentiment": r.sentiment} for r in labelled("dev").itertuples()]
    hard = hard_dev_rows()
    if cfg.limit:
        lab = [r for r in rows if r["urgency"]]
        unl = [r for r in rows if not r["urgency"]]
        half = min(cfg.limit // 2, len(lab))
        rows = _spread(lab, half) + _spread(unl, cfg.limit - half)
        dev_cat, dev_us, hard = _spread(dev_cat, cfg.limit), _spread(dev_us, cfg.limit), _spread(hard, cfg.limit)
    return rows, dev_cat, dev_us, hard


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def predict(model: TriageModel, tokenizer, texts: list[str], max_len: int, batch_size: int, device) -> dict[str, np.ndarray]:
    model.eval()
    out: dict[str, list] = {t: [] for t in TASKS}
    for b in range(0, len(texts), batch_size):
        enc = tokenizer(texts[b : b + batch_size], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
        logits = model(enc["input_ids"], enc["attention_mask"])
        for j, t in enumerate(TASKS):
            out[t].append(logits[j].argmax(-1).cpu().numpy())
    return {t: (np.concatenate(v) if v else np.array([], dtype=int)) for t, v in out.items()}


def _scores(y_true: list[str], y_pred: list[str]) -> dict:
    from sklearn.metrics import f1_score

    labels = sorted(set(y_true))  # classes present in the truth
    return {"n": len(y_true), "accuracy": round(float(np.mean(np.asarray(y_true) == np.asarray(y_pred))), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4), "labels_scored": labels}


def evaluate(model: TriageModel, tokenizer, cfg: TrainConfig, dev_cat: list[dict], dev_us: list[dict], device) -> dict:
    res: dict = {}
    if dev_cat:
        p = predict(model, tokenizer, [r["text"] for r in dev_cat], cfg.max_len, cfg.eval_batch_size, device)
        res["category"] = _scores([r["category"] for r in dev_cat], [LABELS["category"][k] for k in p["category"]])
    if dev_us:
        p = predict(model, tokenizer, [r["text"] for r in dev_us], cfg.max_len, cfg.eval_batch_size, device)
        for t in ("urgency", "sentiment"):
            res[t] = _scores([r[t] for r in dev_us], [LABELS[t][k] for k in p[t]])
    f1s = [res[t]["macro_f1"] for t in TASKS if t in res]
    res["mean_macro_f1"] = round(float(np.mean(f1s)), 4) if f1s else 0.0
    return res


# ----------------------------------------------------------------------------- save / load
def save_model(model: TriageModel, tokenizer, out: Path, cfg: TrainConfig) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    try:
        from safetensors.torch import save_file

        save_file(state, str(out / "model.safetensors"))
        (out / "model.pt").unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 — e.g. shared tensors: fall back to a plain state dict
        torch.save(state, out / "model.pt")
    model.encoder.config.save_pretrained(out / "encoder")
    tokenizer.save_pretrained(out / "tokenizer")
    (out / "labels.json").write_text(json.dumps(LABELS, indent=2), encoding="utf-8")
    (out / "ft_config.json").write_text(json.dumps({**asdict(cfg), "hidden_size": model.encoder.config.hidden_size,
                                                    "heads": {t: len(v) for t, v in LABELS.items()}}, indent=2), encoding="utf-8")
    return out


def load_model(out: Path, device: str = "cpu") -> tuple[TriageModel, object, dict, dict]:
    out = Path(out)
    cfg = json.loads((out / "ft_config.json").read_text(encoding="utf-8"))
    model = TriageModel(config=AutoConfig.from_pretrained(out / "encoder"), dropout=cfg.get("dropout", 0.1))
    if (out / "model.safetensors").exists():
        from safetensors.torch import load_file

        state = load_file(str(out / "model.safetensors"))
    else:
        state = torch.load(out / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    tokenizer = AutoTokenizer.from_pretrained(out / "tokenizer")
    labels = json.loads((out / "labels.json").read_text(encoding="utf-8"))
    return model.to(device).eval(), tokenizer, labels, cfg


# ----------------------------------------------------------------------------- training loop
def train(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = cfg.fp16 if cfg.fp16 is not None else device.type == "cuda"
    if cfg.tiny and cfg.max_len > 128:
        print(f"NOTE: --tiny has 130 position embeddings, so max_len {cfg.max_len} -> 128", flush=True)
        cfg.max_len = 128
    rows, dev_cat, dev_us, hard_dev = build_rows(cfg)
    ids = label_ids(rows)
    labelled_mask = ids[:, 1] != IGNORE
    n_lab, n_synth = int(labelled_mask.sum()), sum(r["source"] == "synthetic" for r in rows)
    by_source = {s: sum(r["source"] == s for r in rows) for s in ("bitext", "synthetic", "translated")}
    lab_by_source = {s: int(sum(1 for r, m in zip(rows, labelled_mask, strict=True) if m and r["source"] == s)) for s in by_source}
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = TriageModel(cfg.model_name, dropout=cfg.dropout, tokenizer=tokenizer, tiny=cfg.tiny).to(device)
    weights = None
    if cfg.class_weights:
        weights = {t: class_weights(ids[:, j], len(LABELS[t])).to(device) for j, t in enumerate(TASKS) if t != "category"}
    rng = random.Random(cfg.seed)
    dataset, collate = TriageDataset(rows, ids), Collator(tokenizer, cfg.max_len)
    epoch_size = len(rows) + n_lab * (max(cfg.oversample, 1) - 1)
    steps_per_epoch = math.ceil(epoch_size / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": cfg.weight_decay},
              {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=cfg.lr)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda") if fp16 else None
    print(f"device {device} | fp16 {fp16} | {cfg.model_name} | max_len {cfg.max_len} | train rows {len(rows)} {by_source} "
          f"({n_lab} with urgency/sentiment: {lab_by_source}) | effective epoch size {epoch_size} = {steps_per_epoch} steps × batch "
          f"{cfg.batch_size} (oversample ×{cfg.oversample}) | dev: {len(dev_cat)} category rows, {len(dev_us)} urgency/sentiment rows"
          f"{f', hard dev {len(hard_dev)} rows (report only)' if hard_dev else ''} | {cfg.epochs} epochs, {total_steps} steps", flush=True)
    if weights:
        print("class weights:", {t: [round(x, 2) for x in w.tolist()] for t, w in weights.items()}, flush=True)

    out = Path(cfg.out)
    history: list[dict] = []
    best: dict | None = None
    step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        idx = oversample_indices(labelled_mask, cfg.oversample, rng)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=idx, collate_fn=collate, num_workers=0)
        t0 = time.time()
        running = dict.fromkeys((*TASKS, "total"), 0.0)
        n_batches = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss, parts = multitask_loss(logits, labels, weights)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            step += 1
            n_batches += 1
            running["total"] += float(loss.item())
            for t in TASKS:
                running[t] += float(parts[t].item())
            if step % cfg.log_every == 0:
                print(f"  epoch {epoch} step {step}/{total_steps} loss {running['total'] / n_batches:.4f} "
                      f"(cat {running['category'] / n_batches:.3f} urg {running['urgency'] / n_batches:.3f} sent {running['sentiment'] / n_batches:.3f}) "
                      f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
        train_loss = {k: round(v / max(n_batches, 1), 4) for k, v in running.items()}
        metrics = evaluate(model, tokenizer, cfg, dev_cat, dev_us, device)
        rec = {"epoch": epoch, "train_loss": train_loss, "dev": metrics, "seconds": round(time.time() - t0, 1), "epoch_size": len(idx), "steps": n_batches}
        per_task = ", ".join(f"{t} {metrics[t]['macro_f1']}" for t in TASKS if t in metrics)
        print(f"epoch {epoch}: train loss {train_loss['total']} | dev mean macro-F1 {metrics['mean_macro_f1']} ({per_task}) | {rec['seconds']}s", flush=True)
        if hard_dev:  # report only — never used for model selection
            rec["hard_dev"] = evaluate(model, tokenizer, cfg, hard_dev, hard_dev, device)
            hd = rec["hard_dev"]
            print(f"  hard dev (report only, {len(hard_dev)} rows): mean macro-F1 {hd['mean_macro_f1']} ("
                  + ", ".join(f"{t} {hd[t]['macro_f1']}" for t in TASKS if t in hd) + ")", flush=True)
        history.append(rec)
        if best is None or metrics["mean_macro_f1"] > best["dev"]["mean_macro_f1"]:
            best = rec
            save_model(model, tokenizer, out, cfg)
            print(f"  best so far — saved to {out}", flush=True)
        summary = {"config": asdict(cfg), "device": str(device), "fp16": fp16, "train_rows": len(rows), "labelled_rows": n_lab,
                   "synthetic_rows": n_synth, "translated_rows": by_source["translated"], "rows_by_source": by_source,
                   "labelled_by_source": lab_by_source, "effective_epoch_size": epoch_size, "steps_per_epoch": steps_per_epoch,
                   "hard_dev_rows": len(hard_dev), "epochs": history, "best_epoch": best["epoch"], "best": best["dev"],
                   "best_hard_dev": best.get("hard_dev")}
        out.mkdir(parents=True, exist_ok=True)
        (out / "dev_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
