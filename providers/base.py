"""Base abstract class for LLM providers."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator


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
    def list_models(self) -> List[str]:
        """
        Get list of available models from the provider.

        Returns:
            List[str]: List of model names/identifiers

        Raises:
            Exception: If provider is not available or connection fails
        """

    def get_model_info(self, _model: str) -> Dict[str, Any]:
        """
        Get metadata for a specific model.

        Args:
            model: Model identifier

        Returns:
            Dict: Model metadata
        """
        return {}

    @abstractmethod
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = True,
        options: Dict[str, Any] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Send a chat completion request.

        Args:
            model: Model identifier
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream the response
            options: Provider-specific options (temperature, top_p, etc.)

        Yields:
            Dict containing response chunks with structure:
            {
                'message': {
                    'content': str  # The text content
                }
            }

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
