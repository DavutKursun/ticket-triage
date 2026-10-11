"""Dataset I/O: splits, silver labels, hard set, hand-check files.

Files (all UTF-8):
  data/splits/{train,dev,test}.jsonl   {"id", "text", "category", "intent"}            gold category (Bitext, CDLA-Sharing-1.0)
  data/silver/{dev,test,train}.jsonl   {"id", "urgency", "sentiment", "reason", "model", "prompt_version"}
  data/hard/hard.csv                   id,text,category,urgency,sentiment,language,split   hand-written and hand-labelled
  data/handcheck/{dev,test}_200.csv    id,text,category,silver_urgency,silver_sentiment,urgency,sentiment,note
                                       (Davut fills the last three columns)
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

from .config import HANDCHECK, HARD, SEED, SILVER, SPLITS
from .schema import CATEGORIES


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalise_text(t: str) -> str:
    return " ".join(str(t).split())


def dedupe_and_split(df: pd.DataFrame, seed: int = SEED, frac=(0.8, 0.1, 0.1)) -> dict[str, pd.DataFrame]:
    """Stratified by category. Expects columns text, category (lower-case), intent."""
    df = df.copy()
    df["text"] = df["text"].map(normalise_text)
    df = df[df["text"].str.len() > 0].drop_duplicates("text").reset_index(drop=True)
    df["id"] = df["text"].map(stable_id)
    unknown = set(df["category"]) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"categories not in schema.CATEGORIES: {sorted(unknown)} — update schema.py and rubric.py")
    rng = random.Random(seed)
    parts: dict[str, list] = {"train": [], "dev": [], "test": []}
    for _, g in df.groupby("category"):
        idx = list(g.index)
        rng.shuffle(idx)
        n = len(idx)
        a, b = int(n * frac[0]), int(n * (frac[0] + frac[1]))
        parts["train"] += idx[:a]
        parts["dev"] += idx[a:b]
        parts["test"] += idx[b:]
    return {k: df.loc[v, ["id", "text", "category", "intent"]].reset_index(drop=True) for k, v in parts.items()}


def load_split(name: str) -> pd.DataFrame:
    rows = read_jsonl(SPLITS / f"{name}.jsonl")
    return pd.DataFrame(rows, columns=["id", "text", "category", "intent"])


def load_silver(name: str) -> pd.DataFrame:
    rows = read_jsonl(SILVER / f"{name}.jsonl")
    return pd.DataFrame(rows, columns=["id", "urgency", "sentiment", "reason", "model", "prompt_version"])


def load_hard(split: str | None = None) -> pd.DataFrame:
    p = HARD / "hard.csv"
    if not p.exists():
        return pd.DataFrame(columns=["id", "text", "category", "urgency", "sentiment", "language", "split"])
    df = pd.read_csv(p, dtype=str).fillna("")
    return df if split is None else df[df["split"] == split].reset_index(drop=True)


def labelled(name: str) -> pd.DataFrame:
    """Split joined with its silver labels (inner join: only rows that have all three labels)."""
    return load_split(name).merge(load_silver(name)[["id", "urgency", "sentiment"]], on="id", how="inner")


def few_shot_examples(per_category: int = 2, seed: int = SEED) -> list[tuple[str, str, str, str]]:
    """From TRAIN only; needs silver labels on train for the chosen rows (label_silver.py --split train --n small)."""
    df = labelled("train")
    rng = random.Random(seed)
    out = []
    for c in CATEGORIES:
        g = df[df["category"] == c]
        take = g.sample(min(per_category, len(g)), random_state=rng.randint(0, 10**6)) if len(g) else g
        out += [(r.text, r.category, r.urgency, r.sentiment) for r in take.itertuples()]
    return out


def write_handcheck_template(name: str, n: int = 100, seed: int = SEED) -> Path:
    """A CSV Davut fills by hand: stratified by silver urgency so rare classes get checked."""
    df = labelled(name)
    rng = random.Random(seed)
    per = max(1, n // 4)
    chosen = []
    for u in ("urgent", "high", "medium", "low"):
        g = df[df["urgency"] == u]
        chosen.append(g.sample(min(per, len(g)), random_state=rng.randint(0, 10**6)))
    sel = pd.concat(chosen).head(n)
    HANDCHECK.mkdir(parents=True, exist_ok=True)
    p = HANDCHECK / f"{name}_{n}.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "text", "category", "silver_urgency", "silver_sentiment", "urgency", "sentiment", "note"])
        for r in sel.itertuples():
            w.writerow([r.id, r.text, r.category, r.urgency, r.sentiment, "", "", ""])
    return p


def read_handcheck(name: str, n: int = 100) -> pd.DataFrame:
    p = HANDCHECK / f"{name}_{n}.csv"
    df = pd.read_csv(p, dtype=str).fillna("")
    done = df[(df["urgency"] != "") & (df["sentiment"] != "")]
    for col in ("urgency", "sentiment"):
        done[col] = done[col].str.strip().str.lower()
    return done
