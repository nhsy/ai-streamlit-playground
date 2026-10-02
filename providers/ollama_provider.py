"""Ollama provider implementation."""

from typing import Any, Iterator

import ollama

from models import ChatChunk, ChatMessage, GenerationOptions, ModelInfo

from .base import BaseProvider
from .settings import OllamaSettings


class OllamaProvider(BaseProvider):
    """Provider implementation for local Ollama models."""

    def __init__(self, settings: OllamaSettings | None = None):
        """
        Initialize Ollama provider.

        Args:
            settings: Provider settings; read from the environment if None
        """
        settings = settings or OllamaSettings()
        self._name = "Ollama (Local)"
        self._enabled = settings.enabled
        self._model_info_cache: dict[str, ModelInfo] = {}

    def is_available(self) -> bool:
        """
        Check if Ollama is running and accessible.

        Returns:
            bool: True if Ollama service is available and enabled
        """
        if not self._enabled:
            return False

        try:
            ollama.list()
            return True
        except Exception:  # pylint: disable=broad-exception-caught
            # This covers ConnectionError and other service-related issues
            return False

    def list_models(self) -> list[str]:
        """
        Get list of available Ollama models.

        Returns:
            list[str]: List of model names

        Raises:
            Exception: If Ollama is not running or connection fails
        """
        models_info = ollama.list()
        # Entries are SDK models (or plain dicts in tests); both support item access
        self._model_info_cache = {
            m["model"]: ModelInfo.model_validate(m.model_dump() if hasattr(m, "model_dump") else m)
            for m in models_info["models"]
        }
        return list(self._model_info_cache.keys())

    def get_model_info(self, model: str) -> ModelInfo:
        """
        Get info for a specific model from cache.

        Args:
            model: Model name

        Returns:
            ModelInfo: Cached model metadata, or empty if unknown
        """
        return self._model_info_cache.get(model) or ModelInfo()

    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        stream: bool = True,
        options: GenerationOptions | None = None,
    ) -> Iterator[ChatChunk]:
        """
        Send a chat completion request to Ollama.

        Args:
            model: Ollama model name
            messages: Conversation history as ChatMessage objects
            stream: Whether to stream the response
            options: Sampling options; defaults if None

        Yields:
            ChatChunk: Response chunks

        Raises:
            Exception: If the request fails
        """
        options = options or GenerationOptions()

        response = ollama.chat(
            model=model,
            messages=[m.payload() for m in messages],
            stream=stream,
            options=options.model_dump(include={"temperature", "top_p"}),
        )

        # A non-streaming call returns a single response rather than an iterator
        chunks = response if stream else [response]
        for chunk in chunks:
            yield ChatChunk.of(chunk["message"]["content"])

    def pull_model(self, model: str) -> Iterator[Any]:
        """
        Pull a model from the Ollama library.

        Args:
            model: Name of the model to pull

        Yields:
            Progress updates from the Ollama SDK
        """
        return ollama.pull(model, stream=True)

    def get_name(self) -> str:
        """
        Get the provider name.

        Returns:
            str: "Ollama (Local)"
        """
        return self._name
