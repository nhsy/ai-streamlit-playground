"""Google Gemini provider implementation using google-genai SDK."""

from typing import Iterator

from google import genai
from google.genai import types

from models import ChatChunk, ChatMessage, GenerationOptions

from .base import BaseProvider
from .settings import GeminiSettings


class GeminiProvider(BaseProvider):
    """Provider implementation for Google Gemini models."""

    def __init__(self, settings: GeminiSettings | None = None):
        """
        Initialize Gemini provider.

        Args:
            settings: Provider settings; read from the environment if None
        """
        settings = settings or GeminiSettings()
        self._name = "Google Gemini"
        self._api_key = settings.api_key.get_secret_value() if settings.api_key else None
        self._enabled = settings.enabled
        self._client = None

        if self._api_key:
            self._client = genai.Client(api_key=self._api_key)

    def is_available(self) -> bool:
        """
        Check if Gemini is configured.

        Returns:
            bool: True if API key is set
        """
        if not self._enabled:
            return False
        return bool(self._api_key)

    def list_models(self) -> list[str]:
        """
        Get list of available Gemini models from the API.

        Returns:
            list[str]: List of model identifiers
        """
        if not self._client:
            return []

        try:
            # Dynamic listing using the SDK
            # We filter for models that are likely for content generation
            # The SDK returns models with 'models/' prefix usually, or just IDs.
            # We'll normalize to bare IDs if possible or keep as is.

            # Note: client.models.list() returns an iterator of Model objects
            models = list(self._client.models.list())

            # Filter and extract names
            # We look for 'generateContent' support roughly by name convention or validation
            # Current SDK might not expose simple capability flags easily on the list object
            # without extra calls, so we stick to known prefixes.
            model_ids = [
                m.name.split("/")[-1] for m in models if "gemini" in m.name.lower() and "vision" not in m.name.lower()
            ]

            if not model_ids:
                raise ValueError("No models found")

            return sorted(model_ids)

        except Exception:  # pylint: disable=broad-exception-caught
            # Fallback to curated list if API call fails
            return [
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite-preview-02-05",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-1.5-flash-8b",
            ]

    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        stream: bool = True,
        options: GenerationOptions | None = None,
    ) -> Iterator[ChatChunk]:
        """
        Send a chat completion request to Gemini.

        Args:
            model: Gemini model identifier
            messages: Conversation history as ChatMessage objects
            stream: Whether to stream (always True for now)
            options: Sampling options; if None, temperature 0.7 and top_p 0.95 are used

        Yields:
            ChatChunk: Response chunks
        """
        if not self._client:
            raise RuntimeError("Gemini API key not configured.")

        # Gemini keeps its own top_p default when no options are given
        temperature, top_p = (options.temperature, options.top_p) if options else (0.7, 0.95)

        # Convert messages to Gemini format if needed, OR relies on SDK's ability to handle
        # standard formats. The new SDK `models.generate_content` is versatile.

        # Extract system prompt if present
        system_instruction = None
        chat_history = []

        for msg in messages:
            role = msg.role
            content = msg.content
            if role == "system":
                system_instruction = content
            elif role == "user":
                chat_history.append(types.Content(role="user", parts=[types.Part.from_text(text=content)]))
            elif role == "assistant":
                chat_history.append(types.Content(role="model", parts=[types.Part.from_text(text=content)]))

        # The last message should be the prompt, so we pop it if we built a full history
        # actually for chat, we usually maintain history.
        # However, `generate_content` is stateless unless using `chats.create`.
        # Given the app structure builds the full context every time, we treat it
        # as single turn with history.

        # Simple concatenation for the current message (the last user message)
        # But wait, `messages` contains the WHOLE history ending with the latest user query.

        # New SDK approach:
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            system_instruction=system_instruction,
        )

        # We pass the full history (excluding system prompt which moved to config)

        response = self._client.models.generate_content_stream(
            model=model,
            contents=chat_history,
            config=config,
        )

        for chunk in response:
            if chunk.text:
                yield ChatChunk.of(chunk.text)

    def get_name(self) -> str:
        """Get the provider name."""
        return self._name
