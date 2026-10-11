"""Approach D: TF-IDF (word + char n-grams) + logistic regression, one classifier per task.
What a competent person gets in five minutes — the bar every fancier approach must clear."""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .config import MODELS, SEED
from .schema import Triage, TriageResponse

TASKS = ("category", "urgency", "sentiment")


def _pipeline() -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, lowercase=True)),
    ])
    return Pipeline([("tfidf", feats), ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", random_state=SEED))])


class BaselineEngine:
    name = "baseline"

    def __init__(self, models: dict[str, Pipeline] | None = None) -> None:
        self.models = models or {}

    def fit(self, df: pd.DataFrame) -> "BaselineEngine":
        """df needs text + whichever of category/urgency/sentiment are present (non-empty)."""
        for task in TASKS:
            if task not in df.columns:
                continue
            sub = df[df[task].astype(str).str.len() > 0]
            if sub[task].nunique() < 2:
                continue
            self.models[task] = _pipeline().fit(sub["text"], sub[task])
        return self

    def predict_frame(self, texts: list[str]) -> pd.DataFrame:
        out = pd.DataFrame({"text": texts})
        for task, m in self.models.items():
            proba = m.predict_proba(texts)
            out[task] = m.classes_[proba.argmax(1)]
            out[f"{task}_conf"] = proba.max(1)
        return out

    def classify(self, text: str) -> Triage:
        f = self.predict_frame([text]).iloc[0]
        return Triage(category=f.get("category", "contact"), urgency=f.get("urgency", "low"), sentiment=f.get("sentiment", "neutral"),
                      confidence=float(f.get("category_conf", 0.0)))

    def respond(self, text: str) -> TriageResponse:
        t0 = time.perf_counter()
        t = self.classify(text)
        return TriageResponse(**t.model_dump(), model="tfidf_logreg", engine="baseline", latency_ms=round((time.perf_counter() - t0) * 1000, 1))

    def save(self, path: Path = MODELS / "baseline.joblib") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models, path)
        return path

    @classmethod
    def load(cls, path: Path = MODELS / "baseline.joblib") -> "BaselineEngine":
        return cls(joblib.load(path))


def macro_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro"))


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
