"""Base abstract class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Iterator

from models import ChatChunk, ChatMessage, GenerationOptions, ModelInfo


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider is available and properly configured.

        Returns:
            bool: True if provider can be used, False otherwise
        """

    @abstractmethod
    def list_models(self) -> list[str]:
        """
        Get list of available models from the provider.

        Returns:
            list[str]: List of model names/identifiers

        Raises:
            Exception: If provider is not available or connection fails
        """

    def get_model_info(self, _model: str) -> ModelInfo:
        """
        Get metadata for a specific model.

        Args:
            model: Model identifier

        Returns:
            ModelInfo: Model metadata (empty by default)
        """
        return ModelInfo()

    @abstractmethod
    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        stream: bool = True,
        options: GenerationOptions | None = None,
    ) -> Iterator[ChatChunk]:
        """
        Send a chat completion request.

        Args:
            model: Model identifier
            messages: Conversation history as ChatMessage objects
            stream: Whether to stream the response
            options: Sampling options (temperature, top_p, max_tokens); defaults if None

        Yields:
            ChatChunk: Response chunks; the text is in chunk.message.content

        Raises:
            Exception: If the request fails
        """

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the display name of the provider.

        Returns:
            str: Provider name (e.g., "Ollama", "watsonx")
        """
