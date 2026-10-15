"""Text LLM providers returning JSON for a Pydantic model.

* ``OpenAICompatProvider`` — any OpenAI-compatible chat API (Groq free tier by default).
  Uses JSON mode (``response_format={"type": "json_object"}``); the caller validates with
  Pydantic and retries once on failure. ``json_schema`` mode is opt-in (USE_JSON_SCHEMA=1)
  because Groq corrupted multibyte UTF-8 with it in the invoice project. ``reasoning_effort``
  (LLM_REASONING_EFFORT, default "low") is sent only when non-empty; it cuts hidden reasoning
  tokens for gpt-oss, which otherwise dominate output-token usage on tiny JSON replies.
* ``AnthropicProvider`` — ``client.messages.parse(output_format=Model)``; optional prompt
  caching of the system block (``cache=True``) for the few-shot engine.

Both return ``LLMResult(raw_json, input_tokens, output_tokens, cache_read_tokens, model, elapsed)``.
Network / API errors become ``ProviderError``; ``QuotaError`` (subclass) marks a daily quota
exhaustion so batch scripts can stop and resume tomorrow.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from .schema import json_schema


class ProviderError(RuntimeError):
    pass


class QuotaError(ProviderError):
    """Daily token quota exhausted (Groq 429 with 'tokens per day')."""


@dataclass
class LLMResult:
    raw_json: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    model: str
    elapsed_seconds: float


class Provider(Protocol):
    name: str

    def complete(self, system: str, user: str, model: str, output_model: type[BaseModel], max_tokens: int = 300, cache: bool = False) -> LLMResult: ...


def _wrap(exc: Exception) -> ProviderError:
    status = getattr(exc, "status_code", None)
    msg = str(exc)
    body = getattr(exc, "body", None)
    code = ""
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            code = err.get("code") or err.get("type") or ""
    text = f"HTTP {status}{f' ({code})' if code else ''}: {msg}" if status else f"{type(exc).__name__}: {msg}"
    if status == 429 and ("per day" in msg.lower() or "tpd" in msg.lower() or "daily" in msg.lower()):
        return QuotaError(text)
    return ProviderError(text)


class OpenAICompatProvider:
    name = "openai_compat"

    def __init__(self, api_key: str, base_url: str, client=None, reasoning_effort: str = "") -> None:
        if client is None:
            from openai import OpenAI

            if not api_key:
                raise ProviderError("LLM_API_KEY is not set (see .env.example)")
            client = OpenAI(api_key=api_key, base_url=base_url, max_retries=2)
        self._client = client
        self._use_schema = os.environ.get("USE_JSON_SCHEMA", "0") == "1"
        self._reasoning_effort = (reasoning_effort or "").strip()

    def complete(self, system: str, user: str, model: str, output_model: type[BaseModel], max_tokens: int = 300, cache: bool = False) -> LLMResult:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self._use_schema:
            fmt = {"type": "json_schema", "json_schema": {"name": output_model.__name__.lower(), "schema": json_schema(output_model), "strict": True}}
        else:
            fmt = {"type": "json_object"}
        kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "response_format": fmt}
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        t0 = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise _wrap(exc) from exc
        elapsed = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        u = getattr(resp, "usage", None)
        pin = int(getattr(u, "prompt_tokens", 0) or 0)
        pout = int(getattr(u, "completion_tokens", 0) or 0)
        details = getattr(u, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
        if not pin:
            pin, pout = (len(system) + len(user)) // 4, max(len(text) // 4, 1)
        return LLMResult(text, pin, pout, cached, model, elapsed)


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, client=None) -> None:
        if client is None:
            import anthropic

            if not api_key:
                raise ProviderError("ANTHROPIC_API_KEY is not set (see .env.example)")
            client = anthropic.Anthropic(api_key=api_key)
        self._client = client

    def complete(self, system: str, user: str, model: str, output_model: type[BaseModel], max_tokens: int = 300, cache: bool = False) -> LLMResult:
        sys_block: list[dict] = [{"type": "text", "text": system}]
        if cache:
            sys_block[0]["cache_control"] = {"type": "ephemeral"}
        t0 = time.perf_counter()
        try:
            resp = self._client.messages.parse(
                model=model, max_tokens=max_tokens, system=sys_block,
                messages=[{"role": "user", "content": user}], output_format=output_model,
            )
        except Exception as exc:
            raise _wrap(exc) from exc
        elapsed = time.perf_counter() - t0
        parsed = getattr(resp, "parsed_output", None)
        raw = parsed.model_dump_json() if parsed is not None else next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
        u = resp.usage
        cached = int(getattr(u, "cache_read_input_tokens", 0) or 0)
        return LLMResult(raw, u.input_tokens, u.output_tokens, cached, model, elapsed)


def get_provider(cfg) -> tuple[Provider, str]:
    if cfg.llm_provider == "anthropic":
        return AnthropicProvider(cfg.anthropic_api_key), cfg.anthropic_model
    if cfg.llm_provider == "openai_compat":
        if not cfg.llm_model:
            raise ProviderError("LLM_MODEL is not set (see .env.example)")
        # AnthropicProvider has no equivalent knob and ignores this setting.
        return OpenAICompatProvider(cfg.llm_api_key, cfg.llm_base_url, reasoning_effort=getattr(cfg, "llm_reasoning_effort", "")), cfg.llm_model
    raise ProviderError(f"Unknown LLM_PROVIDER {cfg.llm_provider!r}")


def parse_json_loose(text: str) -> dict:
    """Parse JSON, tolerating a ```json fence or prose around the object."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def complete_validated(provider: Provider, system: str, user: str, model: str, output_model: type[BaseModel], cache: bool = False):
    """Call, validate with Pydantic, retry once with the error appended. Returns (obj, results, repaired)."""
    from pydantic import ValidationError

    results = [provider.complete(system, user, model, output_model, cache=cache)]
    try:
        return output_model.model_validate(parse_json_loose(results[0].raw_json)), results, False
    except (ValidationError, ValueError) as first:
        fix = f"{user}\n\nYour previous answer was invalid: {str(first)[:400]}\nReturn only the corrected JSON object."
        results.append(provider.complete(system, fix, model, output_model, cache=cache))
        return output_model.model_validate(parse_json_loose(results[1].raw_json)), results, True
