from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_DIR = BASE_DIR / "src" / "latest_ai_development" / "config"

DEFAULT_OUTPUT_DIR = BASE_DIR / "output"

PROVIDER_ALIASES = {
    "anthropic": "anthropicai",
    "gemini": "geminiai",
}

def normalize_api_context(raw_context: str | None) -> str:
    """Return a stable URL prefix like '/testing' or ''."""
    if not raw_context:
        return ""

    normalized = raw_context.strip()
    if not normalized or normalized == "/":
        return ""

    return "/" + normalized.strip("/")


def normalize_provider_name(provider: str | None) -> str:
    """Return the canonical provider key used in models.yaml."""
    provider_key = (provider or "").strip().lower()
    return PROVIDER_ALIASES.get(provider_key, provider_key)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "latest-ai-development"
    app_env: str = Field(default="dev", alias="APP_ENV")
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    provider: str = Field(default="OLLAMA", alias="PROVIDER")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_api_key_secret: str = Field(default="", alias="OPENAI_API_KEY_SECRET")
    anthropicai_api_key: str = Field(default="", alias="ANTHROPICAI_API_KEY")
    anthropicai_api_key_secret: str = Field(default="", alias="ANTHROPICAI_API_KEY_SECRET")
    geminiai_api_key: str = Field(default="", alias="GEMINIAI_API_KEY")
    geminiai_api_key_secret: str = Field(default="", alias="GEMINIAI_API_KEY_SECRET")

    cloud_provider: str = Field(default="", alias="CLOUD_PROVIDER")
    azure_key_vault_url: str = Field(default="", alias="AZURE_KEY_VAULT_URL")
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")

    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_api_key_secret: str = Field(default="", alias="LANGSMITH_API_KEY_SECRET")
    langsmith_endpoint: str = Field(default="", alias="LANGSMITH_ENDPOINT")
    langsmith_project: str = Field(default="", alias="LANGSMITH_PROJECT")
    tracing_backend: str = Field(default="NONE", alias="TRACING_BACKEND")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_root_path: str = Field(default="", alias="API_ROOT_PATH")
    port: int = Field(default=8089, alias="PORT")

    context: str = Field(default="/", alias="CONTEXT")

    report_output_file: str = Field(default="report.md", alias="REPORT_OUTPUT_FILE")
    output_dir: str = Field(default=str(DEFAULT_OUTPUT_DIR), alias="OUTPUT_DIR")

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug_value(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "debug"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "prod", "production", ""}:
                return False

        return bool(value)

    @property
    def normalized_api_root_path(self) -> str:
        return normalize_api_context(self.api_root_path or self.context)

    @property
    def output_dir_path(self) -> Path:
        return Path(self.output_dir).expanduser()

    @property
    def report_output_path(self) -> Path:
        return self.output_dir_path / self.report_output_file

    @property
    def models_config_path(self) -> Path:
        return CONFIG_DIR / "models.yaml"

    @property
    def graph_config_path(self) -> Path:
        return CONFIG_DIR / "graph_config.yaml"


def ensure_runtime_dirs(settings: Settings) -> None:
    settings.output_dir_path.mkdir(parents=True, exist_ok=True)


def load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML object: {path}")

    return data


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    ensure_runtime_dirs(settings)
    return settings


@lru_cache
def get_models_config() -> dict[str, Any]:
    return load_yaml_file(get_settings().models_config_path)


@lru_cache
def get_graph_config() -> dict[str, Any]:
    return load_yaml_file(get_settings().graph_config_path)
