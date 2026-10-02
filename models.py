"""Pydantic models shared by the app, the views and the providers."""

import json
import logging
import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)

Role = Literal["system", "user", "assistant"]


# --- Chat -------------------------------------------------------------------


class ChatMessage(BaseModel):
    """
    One chat message.
    'content' is what the model sees; 'display' (optional) is what the chat shows instead.
    """

    role: Role
    content: str
    display: str | None = None

    @property
    def shown(self) -> str:
        """Text to show in the UI and exports."""
        return self.display if self.display is not None else self.content

    def payload(self) -> dict[str, str]:
        """Plain dict for provider SDKs: role and content only."""
        return {"role": self.role, "content": self.content}


class ChunkMessage(BaseModel):
    """Text carried by a stream chunk."""

    content: str = ""


class ChatChunk(BaseModel):
    """One chunk of a streamed chat response, in the same shape for every provider."""

    message: ChunkMessage = ChunkMessage()

    @classmethod
    def of(cls, text: str | None) -> "ChatChunk":
        """Chunk carrying the given text (None becomes an empty chunk)."""
        return cls(message=ChunkMessage(content=text or ""))


class GenerationOptions(BaseModel):
    """Sampling parameters passed to providers."""

    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, gt=0)


# --- Model metadata ---------------------------------------------------------


class ModelDetails(BaseModel):
    """Optional descriptive fields a provider may report for a model."""

    model_config = ConfigDict(extra="ignore")

    parameter_size: str | None = None
    quantization_level: str | None = None
    family: str | None = None
    display_name: str | None = None


class ModelInfo(BaseModel):
    """Model metadata used for tooltips; every field is optional."""

    model_config = ConfigDict(extra="ignore")

    size: int | None = None
    details: ModelDetails = ModelDetails()


# --- Config -----------------------------------------------------------------


class ProviderConfig(BaseModel):
    """Per-provider section of config.json."""

    default_model: str | None = None
    models: dict[str, str] = {}


class AppConfig(BaseModel):
    """Contents of config.json."""

    default_provider: str = ""
    default_model: str | None = None
    providers: dict[str, ProviderConfig] = {}
    templates: dict[str, str] = {}

    def provider(self, key: str) -> ProviderConfig:
        """Config for one provider, empty if it has no section."""
        return self.providers.get(key) or ProviderConfig()

    def default_model_for(self, key: str) -> str | None:
        """The provider's default model, falling back to the global default."""
        return self.provider(key).default_model or self.default_model


def load_app_config(path: str = "config.json") -> AppConfig:
    """Load and validate config.json, falling back to empty defaults if it is missing or invalid."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return AppConfig.model_validate(json.load(f))
        except (json.JSONDecodeError, OSError, ValidationError) as e:
            logger.warning("Error loading %s: %s", path, e)

    return AppConfig()
