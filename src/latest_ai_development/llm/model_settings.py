from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict
from latest_ai_development.secrets_manager import SecretsManager

_CONFIG_PATH = "src/latest_ai_development/config/models.yaml"


def _load_config() -> Dict[str, Any]:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class ModelSettings:
    """
    Fully resolved LLM configuration.
    All values originate from models.yaml and env vars — no defaults in code.
    """
    provider: str
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    extra: Dict[str, Any] = field(default_factory=dict)


def get_model_settings(provider_name:str) -> ModelSettings:
    """
    Resolves ModelSettings entirely from PROVIDER env var + models.yaml.
    No step concept — provider block is the single source of truth.

    IMPORTANT FOR AI / FUTURE EDITS:
    - PROVIDER env var drives everything.
    - Each provider block in models.yaml must have: model, temperature, top_p, max_tokens.
    - Do not add Python-level fallbacks here.
    """
    config = _load_config()

    if not provider_name:
        raise ValueError("PROVIDER env var is not set.")
    provider_name = provider_name.lower()

    provider_cfg = config.get("providers", {}).get(provider_name)
    if provider_cfg is None:
        raise ValueError(
            f"Provider '{provider_name}' not found in models.yaml. "
            f"Available: {list(config.get('providers', {}).keys())}"
        )

    def require(key: str) -> Any:
        value = provider_cfg.get(key)
        if value is None:
            raise ValueError(
                f"'{key}' is not set for provider '{provider_name}' in models.yaml."
            )
        return value

    extra = _resolve_provider_extras(provider_name, provider_cfg)

    return ModelSettings(
        provider=provider_name,
        model=require("model"),
        temperature=float(require("temperature")),
        top_p=float(require("top_p")),
        max_tokens=int(require("max_tokens")),
        extra=extra,
    )


def _resolve_provider_extras(
    provider_name: str,
    provider_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resolve provider-specific runtime values from env vars.
    Env var names and fallback values come entirely from models.yaml.
    """
    extras: Dict[str, Any] = {}
    secrets_manager = SecretsManager()

    if api_key_env := provider_cfg.get("api_key_env"):
        extras["api_key"] = secrets_manager.get_secret(os.getenv(api_key_env))


    if provider_name == "azure":
        extras["azure_endpoint"] = os.getenv(
            provider_cfg["azure_endpoint_env"],
            provider_cfg.get("azure_endpoint_default"),
        )
        extras["api_version"] = os.getenv(
            provider_cfg["api_version_env"],
            provider_cfg.get("api_version_default"),
        )

    if provider_name == "ollama":
        extras["base_url"] = os.getenv(provider_cfg["base_url_env"])

    return extras
