from __future__ import annotations

import logging

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from latest_ai_development.llm.model_settings import (
    ModelSettings,
    get_fallback_model_settings,
    get_model_settings,
)


logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Create provider-specific LangChain chat models.
    Provider and model config are driven by centralized settings + models.yaml.

    IMPORTANT FOR AI / FUTURE EDITS:
    - Add new providers in src/latest_ai_development/config/models.yaml under `providers:`.
    - Do not create LLMs directly inside steps.
    - Steps should call LLMFactory().create_llm().
    """

    def create_llm(self, provider: str | None = None, *, section: str = "primary") -> BaseChatModel:
        """
        Resolve settings from centralized config and create the model.

        Supports:
        - OpenAI
        - Anthropic
        - Gemini
        - Ollama
        """
        settings = get_model_settings(provider_name=provider, section=section)
        return self.create(settings)

    def create(self, model_settings: ModelSettings) -> BaseChatModel:
        """Build a chat model from an already-resolved ModelSettings instance."""
        provider = model_settings.provider
        logger.info(
            "Creating chat model | provider=%s | model=%s",
            provider,
            model_settings.model,
        )
        logger.debug(
            "Chat model settings | provider=%s | temperature=%s | max_tokens=%s | extra_keys=%s",
            provider,
            model_settings.temperature,
            model_settings.max_tokens,
            sorted(model_settings.extra.keys()),
        )

        if provider == "openai":
            return ChatOpenAI(
                model=model_settings.model,
                temperature=model_settings.temperature,
                api_key=model_settings.extra.get("api_key"),
                max_tokens=model_settings.max_tokens,
            )

        if provider == "anthropicai":
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError as exc:
                raise ImportError(
                    "Anthropic provider requires the 'langchain-anthropic' package. "
                    "Run `uv sync` to install template dependencies."
                ) from exc

            return ChatAnthropic(
                model=model_settings.model,
                temperature=model_settings.temperature,
                api_key=model_settings.extra.get("api_key"),
                max_tokens=model_settings.max_tokens,
            )

        if provider == "geminiai":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as exc:
                raise ImportError(
                    "Gemini provider requires the 'langchain-google-genai' package. "
                    "Run `uv sync` to install template dependencies."
                ) from exc

            return ChatGoogleGenerativeAI(
                model=model_settings.model,
                temperature=model_settings.temperature,
                api_key=model_settings.extra.get("api_key"),
                max_tokens=model_settings.max_tokens,
            )

        if provider == "ollama":
            return ChatOllama(
                model=model_settings.model,
                temperature=model_settings.temperature,
                base_url=model_settings.extra.get("base_url"),
            )

        raise ValueError(f"Unsupported provider: {provider}")


def get_fallback_llms(provider: str | None = None) -> list[BaseChatModel]:
    factory = LLMFactory()
    return [factory.create(settings) for settings in get_fallback_model_settings(provider)]
