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

    LangSmith tracing is controlled by LANGSMITH_TRACING.

    - When LANGSMITH_TRACING=false, tracing is disabled and startup continues.
    - When LANGSMITH_TRACING=true, valid LangSmith credentials are required.
    """
    global _LANGSMITH_INITIALIZED

    import os

    if _LANGSMITH_INITIALIZED:
        return

    settings = get_settings()
    if not settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "false"
        logger.info("LangSmith tracing disabled by configuration")
        return

    try:
        api_key = _resolve_langsmith_api_key()
    except Exception as exc:
        os.environ["LANGSMITH_TRACING"] = "false"
        raise ValueError(
            "LANGSMITH_TRACING is enabled, but LangSmith credentials could not be resolved."
        ) from exc

    if not api_key:
        os.environ["LANGSMITH_TRACING"] = "false"
        raise ValueError(
            "LANGSMITH_TRACING is enabled, but LANGSMITH_API_KEY or "
            "LANGSMITH_API_KEY_SECRET is not configured."
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
