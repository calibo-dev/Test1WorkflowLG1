from __future__ import annotations

from pathlib import Path
from typing import Any
from functools import lru_cache

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Base directories
BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_DIR = BASE_DIR / "src" / "latest_ai_development" / "config"

DEFAULT_OUTPUT_DIR = BASE_DIR / "output"
DEFAULT_CONTEXT_DIR = BASE_DIR / "knowledge"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    app_name: str = "latest-ai-development"
    app_env: str = Field(default="dev", alias="APP_ENV")
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # LLM Provider
    provider: str = Field(default="OPENAI", alias="PROVIDER")

    # OpenAI
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_api_key_secret: str = Field(default="", alias="OPENAI_API_KEY_SECRET")

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    port: int = Field(default=8080, alias="PORT")

    # AWS
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")

    # Context / Knowledge
    context: str = Field(default=str(DEFAULT_CONTEXT_DIR), alias="CONTEXT")

    # Output
    report_output_file: str = Field(default="report.md", alias="REPORT_OUTPUT_FILE")
    output_dir: str = Field(default=str(DEFAULT_OUTPUT_DIR), alias="OUTPUT_DIR")

    # Path helpers
    @property
    def context_path(self) -> Path:
        return Path(self.context).expanduser()

    @property
    def output_dir_path(self) -> Path:
        return Path(self.output_dir).expanduser()

    @property
    def report_output_path(self) -> Path:
        return self.output_dir_path / self.report_output_file

    

    @property
    def models_config_path(self) -> Path:
        return CONFIG_DIR / "models.yaml"


# Runtime helpers
def ensure_runtime_dirs(settings: Settings) -> None:
    settings.context_path.mkdir(parents=True, exist_ok=True)
    settings.output_dir_path.mkdir(parents=True, exist_ok=True)


def load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML object: {path}")

    return data


# Public helpers
@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    ensure_runtime_dirs(settings)
    return settings


@lru_cache
def get_models_config() -> dict[str, Any]:
    return load_yaml_file(get_settings().models_config_path)