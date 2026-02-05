"""Google Gemini provider implementation using google-genai SDK."""
import os
from typing import List, Dict, Any, Iterator
from google import genai
from google.genai import types
from .base import BaseProvider


class GeminiProvider(BaseProvider):
    """Provider implementation for Google Gemini models."""

    def __init__(self):
        """Initialize Gemini provider."""
        self._name = "Google Gemini"
        self._api_key = os.getenv("GEMINI_API_KEY")
        self._enabled = os.getenv("GEMINI_ENABLED", "true").lower() == "true"
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

    def list_models(self) -> List[str]:
        """
        Get list of available Gemini models from the API.

        Returns:
            List[str]: List of model identifiers
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
                m.name.split("/")[-1] for m in models
                if "gemini" in m.name.lower() and "vision" not in m.name.lower()
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
        messages: List[Dict[str, str]],
        stream: bool = True,
        options: Dict[str, Any] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Send a chat completion request to Gemini.

        Args:
            model: Gemini model identifier
            messages: List of message dicts
            stream: Whether to stream (always True for now)
            options: Model parameters

        Yields:
            Dict containing response chunks
        """
        if not self._client:
            raise RuntimeError("Gemini API key not configured.")

        if options is None:
            options = {}

        # Convert messages to Gemini format if needed, OR relies on SDK's ability to handle
        # standard formats. The new SDK `models.generate_content` is versatile.

        # Extract system prompt if present
        system_instruction = None
        chat_history = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                system_instruction = content
            elif role == "user":
                chat_history.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=content)]
                ))
            elif role == "assistant":
                chat_history.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=content)]
                ))

        # The last message should be the prompt, so we pop it if we built a full history
        # actually for chat, we usually maintain history.
        # However, `generate_content` is stateless unless using `chats.create`.
        # Given the app structure builds the full context every time, we treat it
        # as single turn with history.

        # Simple concatenation for the current message (the last user message)
        # But wait, `messages` contains the WHOLE history ending with the latest user query.

        # New SDK approach:
        config = types.GenerateContentConfig(
            temperature=options.get("temperature", 0.7),
            top_p=options.get("top_p", 0.95),
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
                yield {
                    'message': {
                        'content': chunk.text
                    }
                }

    def get_name(self) -> str:
        """Get the provider name."""
        return self._name
