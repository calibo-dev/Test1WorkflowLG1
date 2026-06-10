from __future__ import annotations

from latest_ai_development.config.settings import (
    get_models_config,
    get_settings,
)


def validate_configuration() -> None:
    """
    Validate application configuration.

    Checks:
    - models.yaml provider configuration
    """

    validate_model_configuration()


def validate_model_configuration() -> None:
    """
    Validate models.yaml and environment provider configuration.
    """

    settings = get_settings()
    models_cfg = get_models_config()

    default_provider = models_cfg.get("default_provider")
    providers = models_cfg.get("providers", {})

    if not providers:
        raise ValueError(
            "models.yaml must define at least one provider"
        )

    if not default_provider:
        raise ValueError(
            "models.yaml must define 'default_provider'"
        )

    if default_provider not in providers:
        raise ValueError(
            f"Default provider '{default_provider}' "
            f"is not defined in providers"
        )

    env_provider = settings.provider.lower()

    if env_provider not in providers:
        raise ValueError(
            f"Environment provider '{env_provider}' "
            f"is not defined in models.yaml"
        )

    provider_cfg = providers[env_provider]

    if not provider_cfg.get("chat_model"):
        raise ValueError(
            f"Provider '{env_provider}' must define "
            f"'chat_model'"
        )

    print(
        f"✓ Configuration validated successfully "
        f"(provider={env_provider})"
    )