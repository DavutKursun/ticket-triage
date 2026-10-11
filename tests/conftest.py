from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ticket_triage.providers import LLMResult  # noqa: E402
from ticket_triage.schema import CATEGORIES  # noqa: E402


class FakeProvider:
    """Returns canned JSON; records calls. ``answers`` is a list consumed in order (last repeats)."""

    name = "fake"

    def __init__(self, answers: list[str | dict]) -> None:
        self.answers = [a if isinstance(a, str) else json.dumps(a) for a in answers]
        self.calls: list[tuple[str, str]] = []

    def complete(self, system, user, model, output_model, max_tokens=300, cache=False):
        self.calls.append((system, user))
        i = min(len(self.calls) - 1, len(self.answers) - 1)
        return LLMResult(self.answers[i], 120, 20, 0, model, 0.01)


@pytest.fixture
def fake_provider():
    return FakeProvider


@pytest.fixture
def tiny_frame() -> pd.DataFrame:
    rows = []
    words = {"account": "login password account", "refund": "refund money back", "delivery": "delivery parcel courier",
             "payment": "payment card charged", "order": "order cancel change", "invoice": "invoice download copy"}
    for cat in CATEGORIES:
        base = words.get(cat, cat.replace("_", " "))
        for i in range(6):
            rows.append({"id": f"{cat}{i}", "text": f"{base} please help {i} {'urgent now' if i % 2 else 'when you can'}",
                         "category": cat, "intent": cat, "urgency": "urgent" if i % 2 else "low",
                         "sentiment": "negative" if i % 3 == 0 else "neutral"})
    return pd.DataFrame(rows)
