"""Settings from environment variables / .env. Every knob lives here."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.split("#", 1)[0].strip().strip('"').strip("'"))


_load_dotenv()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SPLITS = DATA / "splits"
SILVER = DATA / "silver"
HARD = DATA / "hard"
HANDCHECK = DATA / "handcheck"
RESULTS = ROOT / "eval" / "results"
MODELS = ROOT / "models"

SEED = 7
HF_DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


@dataclass(frozen=True)
class Settings:
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "openai_compat"))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "openai/gpt-oss-120b"))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-haiku-5-5"))
    ft_model_id: str = field(default_factory=lambda: _env("FT_MODEL_ID", ""))
    hf_token: str = field(default_factory=lambda: _env("HF_TOKEN", ""))
    max_text_chars: int = field(default_factory=lambda: int(_env("MAX_TEXT_CHARS", "2000")))
    max_batch: int = field(default_factory=lambda: int(_env("MAX_BATCH", "100")))
    max_daily_usd: float = field(default_factory=lambda: float(_env("MAX_DAILY_USD", "1")))
    budget_file: Path = field(default_factory=lambda: Path(_env("BUDGET_FILE", ".budget.json")))


settings = Settings()
