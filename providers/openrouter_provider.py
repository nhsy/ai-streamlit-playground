"""OpenRouter provider implementation."""
import os
import json
from typing import List, Dict, Any, Iterator
from openai import OpenAI
from .base import BaseProvider


class OpenRouterProvider(BaseProvider):
    """Provider implementation for OpenRouter."""

    def __init__(self):
        """Initialize OpenRouter provider."""
        self._name = "OpenRouter"
        self._api_key = os.getenv("OPENROUTER_API_KEY")
        self._base_url = "https://openrouter.ai/api/v1"
        self._enabled = os.getenv("OPENROUTER_ENABLED", "true").lower() == "true"
        self._client = None
        self._config_models = {}

        if self._api_key:
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
            )
            # Load config to check for custom models
            self._load_config_models()

    def _load_config_models(self):
        """Load OpenRouter specific configuration from config.json."""
        config_path = "config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    provider_config = config.get("providers", {}).get("openrouter", {})
                    self._config_models = provider_config.get("models", {})
            except Exception:  # pylint: disable=broad-exception-caught
                # If config fails to load, we just won't have custom aliases
                self._config_models = {}

    def is_available(self) -> bool:
        """
        Check if OpenRouter is configured.

        Returns:
            bool: True if API key is set
        """
        if not self._enabled:
            return False
        return bool(self._api_key)

    def list_models(self) -> List[str]:
        """
        Get list of available OpenRouter models.
        Prioritizes models defined in config.json, otherwise returns recommended defaults.

        Returns:
            List[str]: List of model identifiers
        """
        if self._config_models:
            return list(self._config_models.keys())

        # Default fallback list if no config provided
        return [
            "google/gemini-2.0-flash-001",
            "google/gemini-2.5-pro",
            "openrouter/free",
            "deepseek/deepseek-chat",
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
        ]

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """
        Get info for a specific model.

        Args:
            model: Model identifier
        
        Returns:
            Dict: Model metadata
        """
        # Return the display name from config if available
        display_name = self._config_models.get(model, model)
        return {
            "details": {
                "display_name": display_name
            }
        }

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = True,
        options: Dict[str, Any] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Send a chat completion request to OpenRouter.

        Args:
            model: OpenRouter model identifier
            messages: List of message dicts
            stream: Whether to stream (always True for now)
            options: Model parameters

        Yields:
            Dict containing response chunks
        """
        if not self._client:
            raise RuntimeError("OpenRouter API key not configured.")

        if options is None:
            options = {}

        # Filter out system message to top level if needed, but OpenAI client handles it.
        # Just pass through provided options.

        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            temperature=options.get("temperature", 0.7),
            top_p=options.get("top_p", 0.9),
            # OpenRouter specific headers if needed
            extra_headers={
                "HTTP-Referer": "http://localhost:8501", # Optional
                "X-Title": "AI Streamlit Playground"     # Optional
            }
        )

        if stream:
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield {
                        'message': {
                            'content': content
                        }
                    }
        else:
            # Handle non-streaming if ever needed (though app uses stream=True)
            content = response.choices[0].message.content
            yield {
                'message': {
                    'content': content
                }
            }

    def get_name(self) -> str:
        """Get the provider name."""
        return self._name
