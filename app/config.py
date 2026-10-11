"""Runtime settings, read from environment variables (and an optional .env file).

Kept deliberately tiny: one frozen dataclass, built once and cached.
"""

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()  # copies values from a local .env into os.environ (no-op if there is no file)

SYSTEM_PROMPT = (
    "You are a concise, helpful assistant. You have tools for the current time and "
    "for arithmetic. Use a tool whenever it gives a more reliable answer than "
    "guessing, and never do multi-step arithmetic in your head."
)


@dataclass(frozen=True)
class Settings:
    model_name: str
    checkpoint_db: str
    system_prompt: str


@lru_cache
def get_settings() -> Settings:
    # The Anthropic API key is not read here: ChatAnthropic picks up
    # ANTHROPIC_API_KEY from the environment by itself.
    return Settings(
        model_name=os.getenv("MODEL_NAME", "claude-sonnet-5-5"),
        checkpoint_db=os.getenv("CHECKPOINT_DB", "checkpoints.sqlite"),
        system_prompt=SYSTEM_PROMPT,
    )
