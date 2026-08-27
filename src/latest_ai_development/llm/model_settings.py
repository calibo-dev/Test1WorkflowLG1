from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from latest_ai_development.config.settings import (
    get_models_config,
    get_settings,
    normalize_provider_name,
)
from latest_ai_development.secrets_manager import SecretsManager


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    extra: dict[str, Any] = field(default_factory=dict)


def _resolve_api_key(*, direct_value: str, secret_name: str, secrets_manager: SecretsManager) -> str | None:
    direct_value = direct_value.strip()
    if direct_value:
        return direct_value

    secret_name = secret_name.strip()
    if secret_name:
        return secrets_manager.get_secret(secret_name)

    return None


def _resolve_required_api_key(
    *,
    direct_value: str,
    secret_name: str,
    secrets_manager: SecretsManager,
) -> str:
    api_key = _resolve_api_key(
        direct_value=direct_value,
        secret_name=secret_name,
        secrets_manager=secrets_manager,
    )
    if not api_key:
        raise ValueError("API key not configured for the selected provider.")
    return api_key


def get_model_settings(provider_name: str | None = None) -> ModelSettings:
    settings = get_settings()
    models_cfg = get_models_config()
    secrets_manager = SecretsManager()

    provider = normalize_provider_name(provider_name or settings.provider)
    providers = models_cfg.get("providers", {})
    provider_cfg = providers.get(provider)

    if not isinstance(provider_cfg, dict):
        raise ValueError(f"Provider '{provider}' not configured in models.yaml")

    model_name = str(
        provider_cfg.get("chat_model")
        or models_cfg.get("default_model")
        or ""
    ).strip()
    if not model_name:
        raise ValueError(f"Provider '{provider}' must define chat_model in models.yaml")

    extra: dict[str, Any] = {}

    if provider == "openai":
        api_key = _resolve_required_api_key(
            direct_value=settings.openai_api_key,
            secret_name=settings.openai_api_key_secret,
            secrets_manager=secrets_manager,
        )
        extra["api_key"] = api_key

    elif provider == "anthropicai":
        api_key = _resolve_required_api_key(
            direct_value=settings.anthropicai_api_key,
            secret_name=settings.anthropicai_api_key_secret,
            secrets_manager=secrets_manager,
        )
        extra["api_key"] = api_key

    elif provider == "geminiai":
        api_key = _resolve_required_api_key(
            direct_value=settings.geminiai_api_key,
            secret_name=settings.geminiai_api_key_secret,
            secrets_manager=secrets_manager,
        )
        extra["api_key"] = api_key

    elif provider == "bedrockai":
        api_key = _resolve_required_api_key(
            direct_value=settings.bedrock_ai_api_key,
            secret_name=settings.bedrock_ai_api_key_secret,
            secrets_manager=secrets_manager,
        )
        extra["api_key"] = api_key
        extra["region_name"] = settings.aws_region

    elif provider == "ollama":
        extra["base_url"] = settings.ollama_base_url

    else:
        raise ValueError(
            f"Unsupported provider '{provider}'. Supported providers: {sorted(providers.keys())}"
        )

    return ModelSettings(
        provider=provider,
        model=model_name,
        temperature=float(provider_cfg.get("temperature", 0.3)),
        top_p=float(provider_cfg.get("top_p", 1.0)),
        max_tokens=int(provider_cfg.get("max_tokens", 2000)),
        extra=extra,
    )