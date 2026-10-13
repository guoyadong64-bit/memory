from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, Field


class Settings(BaseModel):
    mode: Literal["mock", "live"] = "mock"
    llm_key: str = Field(default="", repr=False)
    llm_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    vision_model: str = ""
    llm_thinking: bool = False
    understanding_model: str = ""
    answer_model: str = ""
    jev_key: str = Field(default="", repr=False)
    jev_url: str = "https://api.typesafe.ai"
    jev_model: str = "jev-1.13.0"
    timeout: float = Field(default=30, gt=0)
    retries: int = Field(default=2, ge=0, le=5)
    store_path: Path = Path("runtime/memory_agent/memory.sqlite3")
    short_ttl_minutes: int = Field(default=60, ge=1, le=10080)
    session_turn_limit: int = Field(default=20, ge=1, le=100)
    top_k: int = Field(default=5, ge=1, le=20)
    reuse_confidence: float = Field(default=0.85, ge=0, le=1)
    relevance_threshold: float = Field(default=0.8, ge=0, le=1)
    retrieval_strategy: Literal["hypergraph","flat"] = "hypergraph"
    semantic_min_similarity: float = Field(default=.05,ge=0,le=1)
    embedding_model: str = ""
    embedding_url: str = ""
    embedding_key: str = Field(default="",repr=False)

    @classmethod
    def load(cls, env_file: Path = Path(".env")) -> "Settings":
        # Existing process environment overrides .env; never mutates os.environ.
        values = {**dotenv_values(env_file), **os.environ}
        names = {
            "mode": "MEMORY_AGENT_MODE", "llm_key": "MEMORY_LLM_API_KEY",
            "llm_url": "MEMORY_LLM_BASE_URL", "llm_model": "MEMORY_LLM_MODEL",
            "vision_model": "MEMORY_VISION_MODEL",
            "llm_thinking": "MEMORY_LLM_ENABLE_THINKING",
            "understanding_model": "MEMORY_UNDERSTANDING_MODEL",
            "answer_model": "MEMORY_ANSWER_MODEL", "jev_key": "TYPESAFE_API_KEY",
            "jev_url": "TYPESAFE_BASE_URL", "jev_model": "TYPESAFE_MODEL",
            "timeout": "MEMORY_HTTP_TIMEOUT_SECONDS", "retries": "MEMORY_HTTP_RETRIES",
            "store_path": "MEMORY_STORE_PATH", "top_k": "MEMORY_TOP_K",
            "short_ttl_minutes": "MEMORY_SHORT_TTL_MINUTES",
            "session_turn_limit": "MEMORY_SESSION_TURN_LIMIT",
            "reuse_confidence": "MEMORY_REUSE_CONFIDENCE",
            "relevance_threshold": "MEMORY_RELEVANCE_THRESHOLD",
            "retrieval_strategy":"MEMORY_RETRIEVAL_STRATEGY",
            "semantic_min_similarity":"MEMORY_SEMANTIC_MIN_SIMILARITY",
            "embedding_model":"MEMORY_EMBEDDING_MODEL",
            "embedding_url":"MEMORY_EMBEDDING_BASE_URL",
            "embedding_key":"MEMORY_EMBEDDING_API_KEY",
        }
        return cls(**{field: values[name] for field, name in names.items()
                      if values.get(name) not in (None, "")})

    def validate_live(self) -> None:
        missing = []
        for name, value in (("MEMORY_LLM_API_KEY", self.llm_key),
                            ("TYPESAFE_API_KEY", self.jev_key)):
            if not value.strip() or value.startswith(("YOUR_", "填写")):
                missing.append(name)
        if missing:
            raise ValueError("Fill these fields in .env: " + ", ".join(missing))
