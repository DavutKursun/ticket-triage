"""Price table, cost per call, free-tier labelling and a daily budget guard.

Prices in USD per million tokens (list prices checked 2026-10-09 at
https://platform.claude.com/docs/en/about-claude/pricing). Models not in the table
are treated as free tier (cost 0, labelled "free tier", never "$0.0000").
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}
BATCH_DISCOUNT = 0.5          # Batch API: 50% off both input and output
CACHE_READ_MULTIPLIER = 0.1   # prompt-cache read tokens cost 10% of input price
CACHE_WRITE_MULTIPLIER = 1.25


def cost_per_1000(model: str, input_tokens_per_call: float, output_tokens_per_call: float, cached_input_per_call: float = 0.0, batch: bool = False) -> float:
    """USD per 1,000 tickets at list prices; 0 for free-tier models."""
    if model not in PRICES_PER_MTOK:
        return 0.0
    i, o = PRICES_PER_MTOK[model]
    uncached = max(input_tokens_per_call - cached_input_per_call, 0.0)
    per_call = (uncached * i + cached_input_per_call * i * CACHE_READ_MULTIPLIER + output_tokens_per_call * o) / 1_000_000
    if batch:
        per_call *= BATCH_DISCOUNT
    return per_call * 1000


def priced_equivalent(model: str) -> str:
    """For a free-tier model, the paid model whose list price we quote 'as if' (README cost table)."""
    return model if model in PRICES_PER_MTOK else "claude-haiku-5-5"


def is_priced(model: str) -> bool:
    return model in PRICES_PER_MTOK


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICES_PER_MTOK:
        return 0.0
    i, o = PRICES_PER_MTOK[model]
    return (input_tokens * i + output_tokens * o) / 1_000_000


def format_cost(model: str, usd: float) -> str:
    return f"${usd:.4f}" if is_priced(model) else "free tier"


@dataclass
class Spend:
    date: str
    usd: float
    calls: int


class BudgetGuard:
    def __init__(self, path: str | Path, max_daily_usd: float, clock=time.time) -> None:
        self.path = Path(path)
        self.max_daily_usd = max_daily_usd
        self._clock = clock

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime(self._clock()))

    def read(self) -> Spend:
        if not self.path.exists():
            return Spend(self._today(), 0.0, 0)
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return Spend(self._today(), 0.0, 0)
        if d.get("date") != self._today():
            return Spend(self._today(), 0.0, 0)
        return Spend(d["date"], float(d.get("usd", 0.0)), int(d.get("calls", 0)))

    def can_spend(self, estimated_usd: float = 0.0) -> bool:
        return self.read().usd + estimated_usd <= self.max_daily_usd

    def record(self, usd: float) -> Spend:
        s = self.read()
        s = Spend(s.date, s.usd + usd, s.calls + 1)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".budget-", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"date": s.date, "usd": s.usd, "calls": s.calls}, fh)
        os.replace(tmp, self.path)
        return s
