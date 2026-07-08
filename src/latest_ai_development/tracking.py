from __future__ import annotations

import logging

from latest_ai_development.config.settings import get_settings
from latest_ai_development.secrets_manager import SecretsManager

logger = logging.getLogger("uvicorn.error")
_LANGSMITH_INITIALIZED = False


def _resolve_langsmith_api_key() -> str | None:
    settings = get_settings()
    api_key = settings.langsmith_api_key.strip()
    if api_key:
        return api_key

    secret_name = settings.langsmith_api_key_secret.strip()
    if not secret_name:
        return None

    return SecretsManager().get_secret(secret_name)


def initialize_langsmith_tracing() -> None:
    """
    Initialize LangSmith tracing for this template.

    LangSmith credentials are required. If configuration is missing or invalid,
    initialization fails fast so the app does not start in a partially configured
    state.
    """
    global _LANGSMITH_INITIALIZED

    import os

    if _LANGSMITH_INITIALIZED:
        return

    settings = get_settings()
    api_key = _resolve_langsmith_api_key()
    if not api_key:
        os.environ.pop("LANGSMITH_TRACING", None)
        raise ValueError(
            "LANGSMITH_API_KEY or LANGSMITH_API_KEY_SECRET must be configured "
            "for this template."
        )

    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_TRACING"] = "true"
    project = settings.langsmith_project.strip()
    if project:
        os.environ["LANGSMITH_PROJECT"] = project

    if settings.langsmith_endpoint.strip():
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint.strip()

    logger.info(
        "LangSmith tracing enabled | project=%s",
        project or "not set",
    )
    _LANGSMITH_INITIALIZED = True
