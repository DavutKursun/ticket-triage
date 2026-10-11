"""Approaches A (zero-shot) and B (few-shot): one LLM call per message, Pydantic-validated."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .config import settings
from .costs import BudgetGuard, cost_usd
from .providers import LLMResult, Provider, complete_validated, get_provider
from .rubric import PROMPT_VERSION, TRIAGE_INSTRUCTION, TRIAGE_SYSTEM, few_shot_block
from .schema import Triage, TriageResponse


@dataclass
class LLMCall:
    triage: Triage
    results: list[LLMResult]
    repaired: bool

    @property
    def input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.results)

    @property
    def output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.results)

    @property
    def cache_read_tokens(self) -> int:
        return sum(r.cache_read_tokens for r in self.results)

    @property
    def elapsed(self) -> float:
        return sum(r.elapsed_seconds for r in self.results)


class LLMEngine:
    """``mode`` = "zero" | "few". Few-shot examples come from the train split only (see data.py)."""

    def __init__(self, provider: Provider | None = None, model: str | None = None, mode: str = "zero",
                 examples: list[tuple[str, str, str, str]] | None = None, budget: BudgetGuard | None = None) -> None:
        if provider is None:
            provider, model = get_provider(settings)
        self.provider, self.model, self.mode = provider, model or "", mode
        self.system = TRIAGE_SYSTEM
        if mode == "few":
            if not examples:
                raise ValueError("few-shot mode needs examples")
            self.system = TRIAGE_SYSTEM + "\n\n" + few_shot_block(examples)
        self.budget = budget
        self.prompt_version = PROMPT_VERSION

    @property
    def name(self) -> str:
        return f"llm_{self.mode}shot"

    def classify(self, text: str) -> LLMCall:
        text = text.strip()
        if not text:
            raise ValueError("empty text")
        if len(text) > settings.max_text_chars:
            raise ValueError(f"text longer than {settings.max_text_chars} characters")
        if self.budget is not None and not self.budget.can_spend():
            raise RuntimeError("daily budget for the LLM engine is exhausted; try the fine-tuned engine")
        user = TRIAGE_INSTRUCTION.format(text=text)
        obj, results, repaired = complete_validated(self.provider, self.system, user, self.model, Triage, cache=(self.mode == "few"))
        if self.budget is not None:
            self.budget.record(sum(cost_usd(self.model, r.input_tokens, r.output_tokens) for r in results))
        return LLMCall(obj, results, repaired)

    def respond(self, text: str) -> TriageResponse:
        t0 = time.perf_counter()
        call = self.classify(text)
        ms = (time.perf_counter() - t0) * 1000
        return TriageResponse(**call.triage.model_dump(), model=self.model, engine="llm", latency_ms=round(ms, 1))
