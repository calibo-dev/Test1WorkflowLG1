from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict

from dotenv import load_dotenv

from latest_ai_development.secrets_manager import SecretsManager


# --------------------------------------------------
# LOAD .env FILE (IMPORTANT)
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# MODEL SETTINGS
# --------------------------------------------------

@dataclass(frozen=True)
class ModelSettings:
    provider: str
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    extra: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------
# MAIN CONFIG
# --------------------------------------------------

def get_model_settings(provider_name: str) -> ModelSettings:

    if not provider_name:
        raise ValueError("provider_name is required")

    provider_name = provider_name.lower()

    secrets_manager = SecretsManager()

    # --------------------------------------------------
    # DEFAULTS (NO YAML)
    # --------------------------------------------------

    defaults = {
        "openai": {
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "top_p": 1.0,
            "max_tokens": 2000,
            "secret_env": "OPENAI_API_KEY_SECRET",
        },

        "aws": {
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "top_p": 1.0,
            "max_tokens": 2000,
            "secret_env": "OPENAI_API_KEY_SECRET",
            "region_env": "AWS_REGION",
        },

        "ollama": {
            "model": "llama3",
            "temperature": 0.3,
            "top_p": 1.0,
            "max_tokens": 2000,
            "base_url_env": "OLLAMA_BASE_URL",
        },
    }

    if provider_name not in defaults:
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Supported: {list(defaults.keys())}"
        )

    cfg = defaults[provider_name]
    extra: Dict[str, Any] = {}

    # --------------------------------------------------
    # OPENAI
    # --------------------------------------------------

    if provider_name == "openai":
        secret_name = os.getenv(cfg["secret_env"])
        if secret_name:
            extra["api_key"] = secrets_manager.get_secret(secret_name)

    # --------------------------------------------------
    # AWS (Secrets Manager)
    # --------------------------------------------------

    elif provider_name == "aws":
        secret_name = os.getenv(cfg["secret_env"])
        region = os.getenv(cfg["region_env"], "us-east-1")

        if secret_name:
            extra["api_key"] = secrets_manager.get_secret(secret_name)

        extra["region"] = region

    # --------------------------------------------------
    # OLLAMA
    # --------------------------------------------------

    elif provider_name == "ollama":
        extra["base_url"] = os.getenv(
            cfg["base_url_env"],
            "http://localhost:11434"
        )

    # --------------------------------------------------
    # RETURN
    # --------------------------------------------------

    return ModelSettings(
        provider=provider_name,
        model=cfg["model"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        max_tokens=cfg["max_tokens"],
        extra=extra,
    )