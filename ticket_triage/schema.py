"""Labels and the structured output schema.

CATEGORIES mirrors the Bitext ``category`` column (lower-cased). ``scripts/build_dataset.py``
asserts the dataset's actual set equals this list and prints the difference if not — fix the
list here, never silently drop a category.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

CATEGORIES: tuple[str, ...] = (
    "account", "cancel", "contact", "delivery", "feedback", "invoice", "order", "payment",
    "refund", "shipping", "subscription",
)
URGENCIES: tuple[str, ...] = ("low", "medium", "high", "urgent")
SENTIMENTS: tuple[str, ...] = ("negative", "neutral", "positive")

Category = Literal[
    "account", "cancel", "contact", "delivery", "feedback", "invoice", "order", "payment",
    "refund", "shipping", "subscription",
]
Urgency = Literal["low", "medium", "high", "urgent"]
Sentiment = Literal["negative", "neutral", "positive"]


def _norm(v):
    return v.strip().lower().replace(" ", "_").replace("-", "_") if isinstance(v, str) else v


class Triage(BaseModel):
    """What every engine returns for one message. Enum values compared case-insensitively."""

    category: Category
    urgency: Urgency
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0, description="Engine's confidence in the category label")

    @field_validator("category", "urgency", "sentiment", mode="before")
    @classmethod
    def _lower(cls, v):
        return _norm(v)


class SilverLabel(BaseModel):
    """LLM-produced urgency + sentiment for a message whose category is already gold."""

    urgency: Urgency
    sentiment: Sentiment
    reason: str = Field(default="", max_length=300)

    @field_validator("urgency", "sentiment", mode="before")
    @classmethod
    def _lower(cls, v):
        return _norm(v)


class TriageResponse(Triage):
    """API response: Triage plus provenance."""

    model: str
    engine: Literal["finetuned", "llm", "baseline"]
    latency_ms: float


def json_schema(model: type[BaseModel]) -> dict:
    """Strict JSON schema (no additional properties) for providers that enforce one."""
    s = model.model_json_schema()
    s["additionalProperties"] = False
    s.pop("description", None)
    return s
