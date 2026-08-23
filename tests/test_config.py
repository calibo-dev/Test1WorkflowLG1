from __future__ import annotations

import pytest

from latest_ai_development.config.settings import (
    Settings,
    get_graph_config,
    get_models_config,
    get_settings,
    normalize_provider_name,
)
from latest_ai_development.config.validators import validate_configuration
from latest_ai_development.llm.model_settings import get_model_settings


def test_settings_default_runtime_values(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("CONTEXT", raising=False)
    monkeypatch.delenv("OUTPUT_DIR", raising=False)

    settings = Settings(_env_file=None)

    assert settings.port == 8089
    assert settings.context == "/"
    assert settings.output_dir_path.name == "output"
    assert settings.normalized_api_root_path == ""


def test_settings_normalize_api_root_path_from_context(monkeypatch):
    monkeypatch.setenv("CONTEXT", "testing")

    settings = get_settings()

    assert settings.normalized_api_root_path == "/testing"


def test_settings_read_runtime_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "9090")
    monkeypatch.setenv("API_ROOT_PATH", "/platform")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("REPORT_OUTPUT_FILE", "final-report.md")
    monkeypatch.setenv("AZURE_KEY_VAULT_URL", "https://vault.example.vault.azure.net/")

    settings = get_settings()

    assert settings.api_host == "127.0.0.1"
    assert settings.port == 9090
    assert settings.normalized_api_root_path == "/platform"
    assert settings.output_dir_path == tmp_path
    assert settings.report_output_path == tmp_path / "final-report.md"
    assert settings.azure_key_vault_url == "https://vault.example.vault.azure.net/"


def test_validate_configuration_requires_azure_key_vault_url_when_cloud_provider_is_azure(
    monkeypatch,
):
    monkeypatch.setenv("PROVIDER", "OLLAMA")
    monkeypatch.setenv("CLOUD_PROVIDER", "AZURE")
    monkeypatch.setenv("AZURE_KEY_VAULT_URL", "")

    with pytest.raises(ValueError, match="AZURE_KEY_VAULT_URL is required"):
        validate_configuration()


def test_validate_configuration_requires_resolvable_provider_credentials(monkeypatch):
    monkeypatch.setenv("PROVIDER", "OPENAI")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY_SECRET", "")
    monkeypatch.setenv("CLOUD_PROVIDER", "")

    with pytest.raises(
        ValueError,
        match="API key not configured for the selected provider.",
    ):
        validate_configuration()


def test_models_config_is_loaded_from_yaml():
    models_config = get_models_config()

    assert models_config["default_provider"] == "ollama"
    assert "openai" in models_config["providers"]
    assert "anthropicai" in models_config["providers"]
    assert "geminiai" in models_config["providers"]
    assert "bedrockai" in models_config["providers"]
    assert "ollama" in models_config["providers"]


def test_normalize_provider_name_maps_provider_aliases():
    assert normalize_provider_name("anthropic") == "anthropicai"
    assert normalize_provider_name("ANTHROPIC") == "anthropicai"
    assert normalize_provider_name("gemini") == "geminiai"
    assert normalize_provider_name("GEMINI") == "geminiai"
    assert normalize_provider_name("bedrock") == "bedrockai"
    assert normalize_provider_name("openai") == "openai"


def test_graph_config_is_loaded_from_yaml():
    graph_config = get_graph_config()

    assert graph_config["entry_point"] == "validate_input"
    assert graph_config["final_output_field"] == "report"
    assert "router_agent" in graph_config["nodes"]
    assert graph_config["nodes"]["research"]["builder"].endswith(":ResearchNodeBuilder.build")


def test_model_settings_uses_models_yaml_defaults(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.internal:11434")

    model_settings = get_model_settings("ollama")

    assert model_settings.provider == "ollama"
    assert model_settings.model == "llama3"
    assert model_settings.temperature == 0.3
    assert model_settings.top_p == 1.0
    assert model_settings.max_tokens == 2000
    assert model_settings.extra["base_url"] == "http://ollama.internal:11434"


def test_model_settings_supports_openai_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")

    model_settings = get_model_settings("openai")
    models_config = get_models_config()

    assert model_settings.provider == "openai"
    assert model_settings.model == models_config["providers"]["openai"]["chat_model"]
    assert model_settings.extra["api_key"] == "openai-test-key"


def test_model_settings_supports_anthropic_provider(monkeypatch):
    monkeypatch.setenv("ANTHROPICAI_API_KEY", "anthropic-test-key")

    model_settings = get_model_settings("anthropic")

    assert model_settings.provider == "anthropicai"
    assert model_settings.model == "claude-sonnet-4-6"
    assert model_settings.extra["api_key"] == "anthropic-test-key"


def test_model_settings_supports_gemini_provider(monkeypatch):
    monkeypatch.setenv("GEMINIAI_API_KEY", "gemini-test-key")

    model_settings = get_model_settings("gemini")

    assert model_settings.provider == "geminiai"
    assert model_settings.model == "gemini-2.5-flash"
    assert model_settings.extra["api_key"] == "gemini-test-key"


def test_model_settings_supports_bedrock_provider(monkeypatch):
    monkeypatch.setenv("BEDROCK_AI_API_KEY", "bedrock-test-token")
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    model_settings = get_model_settings("bedrock")

    assert model_settings.provider == "bedrockai"
    assert model_settings.model == "us.anthropic.claude-opus-5"
    assert model_settings.extra["api_key"] == "bedrock-test-token"
    assert model_settings.extra["region_name"] == "us-east-1"
