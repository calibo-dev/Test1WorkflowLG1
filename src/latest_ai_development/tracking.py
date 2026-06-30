from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from latest_ai_development.secrets_manager import SecretsManager

load_dotenv()

logger = logging.getLogger("uvicorn.error")

def _resolve_langsmith_api_key() -> str | None:
    api_key = os.getenv("LANGSMITH_API_KEY")
    if api_key:
        return api_key

    secret_name = os.getenv("LANGSMITH_API_KEY_SECRET")
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
    api_key = _resolve_langsmith_api_key()
    if not api_key:
        os.environ.pop("LANGSMITH_TRACING", None)
        raise ValueError(
            "LANGSMITH_API_KEY or LANGSMITH_API_KEY_SECRET must be configured "
            "for this template."
        )

    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_TRACING"] = "true"
    project = os.getenv("LANGSMITH_PROJECT")
    if project:
        os.environ["LANGSMITH_PROJECT"] = project

    logger.info(
        "LangSmith tracing enabled | project=%s",
        project or "not set",
    )
