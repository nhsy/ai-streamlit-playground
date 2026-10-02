"""OpenRouter provider implementation."""

from typing import Iterator

from openai import OpenAI

from models import ChatChunk, ChatMessage, GenerationOptions, ModelDetails, ModelInfo, load_app_config

from .base import BaseProvider
from .settings import OpenRouterSettings


class OpenRouterProvider(BaseProvider):
    """Provider implementation for OpenRouter."""

    def __init__(self, settings: OpenRouterSettings | None = None):
        """
        Initialize OpenRouter provider.

        Args:
            settings: Provider settings; read from the environment if None
        """
        settings = settings or OpenRouterSettings()
        self._name = "OpenRouter"
        self._api_key = settings.api_key.get_secret_value() if settings.api_key else None
        self._base_url = "https://openrouter.ai/api/v1"
        self._enabled = settings.enabled
        self._client = None
        self._config_models: dict[str, str] = {}

        if self._api_key:
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
            )
            # Custom models and display names from config.json (empty if not configured)
            self._config_models = load_app_config().provider("openrouter").models

    def is_available(self) -> bool:
        """
        Check if OpenRouter is configured.

        Returns:
            bool: True if API key is set
        """
        if not self._enabled:
            return False
        return bool(self._api_key)

    def list_models(self) -> list[str]:
        """
        Get list of available OpenRouter models.
        Prioritizes models defined in config.json, otherwise returns recommended defaults.

        Returns:
            list[str]: List of model identifiers
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

    def get_model_info(self, model: str) -> ModelInfo:
        """
        Get info for a specific model.

        Args:
            model: Model identifier

        Returns:
            ModelInfo: Metadata carrying the display name from config (or the model id)
        """
        return ModelInfo(details=ModelDetails(display_name=self._config_models.get(model, model)))

    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        stream: bool = True,
        options: GenerationOptions | None = None,
    ) -> Iterator[ChatChunk]:
        """
        Send a chat completion request to OpenRouter.

        Args:
            model: OpenRouter model identifier
            messages: Conversation history as ChatMessage objects
            stream: Whether to stream (always True for now)
            options: Sampling options; defaults if None

        Yields:
            ChatChunk: Response chunks
        """
        if not self._client:
            raise RuntimeError("OpenRouter API key not configured.")

        options = options or GenerationOptions()

        # The OpenAI client accepts system messages inline, so payloads pass straight through
        response = self._client.chat.completions.create(
            model=model,
            messages=[m.payload() for m in messages],
            stream=stream,
            temperature=options.temperature,
            top_p=options.top_p,
            # OpenRouter specific headers if needed
            extra_headers={
                "HTTP-Referer": "http://localhost:8501",  # Optional
                "X-Title": "AI Streamlit Playground",  # Optional
            },
        )

        if stream:
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield ChatChunk.of(content)
        else:
            # Handle non-streaming if ever needed (though app uses stream=True)
            yield ChatChunk.of(response.choices[0].message.content)

    def get_name(self) -> str:
        """Get the provider name."""
        return self._name
