"""Ollama provider implementation."""
from typing import List, Dict, Any, Iterator
import ollama
from .base import LLMProvider


class OllamaProvider(LLMProvider):
    """Provider implementation for local Ollama models."""

    def __init__(self):
        """Initialize Ollama provider."""
        # pylint: disable=import-outside-toplevel
        import os
        self._name = "Ollama (Local)"
        self._enabled = os.getenv("OLLAMA_ENABLED", "true").lower() == "true"

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

    def list_models(self) -> List[str]:
        """
        Get list of available Ollama models.

        Returns:
            List[str]: List of model names

        Raises:
            Exception: If Ollama is not running or connection fails
        """
        models_info = ollama.list()
        return [m['model'] for m in models_info['models']]

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = True,
        options: Dict[str, Any] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Send a chat completion request to Ollama.

        Args:
            model: Ollama model name
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream the response
            options: Options like temperature, top_p

        Yields:
            Dict containing response chunks

        Raises:
            Exception: If the request fails
        """
        if options is None:
            options = {}

        # Ollama uses the same format, so we can pass through directly
        response = ollama.chat(
            model=model,
            messages=messages,
            stream=stream,
            options=options
        )

        # Yield chunks directly from Ollama
        yield from response

    def get_name(self) -> str:
        """
        Get the provider name.

        Returns:
            str: "Ollama (Local)"
        """
        return self._name
