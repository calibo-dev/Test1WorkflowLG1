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


def _resolve_api_key(
    *,
    secrets_manager: SecretsManager,
    direct_env_names: list[str],
    secret_env_names: list[str],
) -> str | None:
    """
    Resolve an API key using the following priority:

    1. Direct value from a plain environment variable
    2. Secret reference resolved through SecretsManager
    """
    for env_name in direct_env_names:
        value = os.getenv(env_name)
        if value:
            return value

    for env_name in secret_env_names:
        secret_name = os.getenv(env_name)
        if secret_name:
            return secrets_manager.get_secret(secret_name)

    return None


def get_model_settings(provider_name: str) -> ModelSettings:

    if not provider_name:
        raise ValueError("provider_name is required")

    provider_name = provider_name.lower()

    secrets_manager = SecretsManager()


    defaults = {
        "openai": {
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "top_p": 1.0,
            "max_tokens": 2000,
            "direct_api_key_envs": ["OPENAI_API_KEY"],
            "secret_envs": ["API_KEY_SECRET", "OPENAI_API_KEY_SECRET"],
        },

        "azure_openai": {
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "top_p": 1.0,
            "max_tokens": 2000,
            "direct_api_key_envs": ["AZURE_OPENAI_API_KEY"],
            "secret_envs": ["API_KEY_SECRET", "AZURE_OPENAI_API_KEY_SECRET"],
            "endpoint_envs": ["AZURE_OPENAI_ENDPOINT", "AZUREAI_OPENAI_URL"],
            "api_version_envs": ["AZURE_OPENAI_API_VERSION", "AZUREAI_OPENAI_API_VERSION"],
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
        api_key = _resolve_api_key(
            secrets_manager=secrets_manager,
            direct_env_names=cfg["direct_api_key_envs"],
            secret_env_names=cfg["secret_envs"],
        )
        if api_key:
            extra["api_key"] = api_key

    # --------------------------------------------------
    # AZURE OPENAI
    # --------------------------------------------------

    elif provider_name == "azure_openai":
        api_key = _resolve_api_key(
            secrets_manager=secrets_manager,
            direct_env_names=cfg["direct_api_key_envs"],
            secret_env_names=cfg["secret_envs"],
        )
        if api_key:
            extra["api_key"] = api_key

        for env_name in cfg["endpoint_envs"]:
            endpoint = os.getenv(env_name)
            if endpoint:
                extra["azure_endpoint"] = endpoint
                break

        for env_name in cfg["api_version_envs"]:
            api_version = os.getenv(env_name)
            if api_version:
                extra["api_version"] = api_version
                break

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