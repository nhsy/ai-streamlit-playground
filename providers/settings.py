"""Provider settings read from environment variables (app.py loads .env first)."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaSettings(BaseSettings):
    """OLLAMA_* environment variables."""

    model_config = SettingsConfigDict(env_prefix="OLLAMA_", extra="ignore")

    enabled: bool = True


class WatsonxSettings(BaseSettings):
    """WATSONX_* environment variables."""

    model_config = SettingsConfigDict(env_prefix="WATSONX_", extra="ignore")

    api_key: SecretStr | None = None
    project_id: str | None = None
    url: str = "https://eu-gb.ml.cloud.ibm.com"
    enabled: bool = True


class OpenRouterSettings(BaseSettings):
    """OPENROUTER_* environment variables."""

    model_config = SettingsConfigDict(env_prefix="OPENROUTER_", extra="ignore")

    api_key: SecretStr | None = None
    enabled: bool = True


class GeminiSettings(BaseSettings):
    """GEMINI_* environment variables."""

    model_config = SettingsConfigDict(env_prefix="GEMINI_", extra="ignore")

    api_key: SecretStr | None = None
    enabled: bool = True
