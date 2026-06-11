from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from latest_ai_development.llm.model_settings import (
    ModelSettings,
    get_model_settings,
)


class LLMFactory:
    """
    Creates LangChain chat models via init_chat_model.
    Provider and all config are driven by PROVIDER env var + models.yaml.

    IMPORTANT FOR AI / FUTURE EDITS:
    - Add new providers in models.yaml under `providers:`.
    - Do not create LLMs directly inside steps.
    - Steps should call LLMFactory().create_llm().
    """

    def create_llm(self, provider: str) -> BaseChatModel:
        """Resolve settings from env + models.yaml and create the model."""
        settings = get_model_settings(provider_name = provider)
        return self.create(settings)

    def create(self, model_settings: ModelSettings) -> BaseChatModel:
        """Build a chat model from an already-resolved ModelSettings instance."""
        kwargs = {
            "temperature": model_settings.temperature,
            "top_p": model_settings.top_p,
            "max_tokens": model_settings.max_tokens,
            **model_settings.extra,
        }

        return init_chat_model(
            model=model_settings.model,
            model_provider=model_settings.provider,
            configurable_fields=None,
            **kwargs,
        )
