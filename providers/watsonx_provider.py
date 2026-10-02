"""IBM watsonx provider implementation."""

from typing import Iterator

from models import ChatChunk, ChatMessage, GenerationOptions

from .base import BaseProvider
from .settings import WatsonxSettings


class WatsonxProvider(BaseProvider):
    """Provider implementation for IBM watsonx.ai models."""

    def __init__(self, settings: WatsonxSettings | None = None):
        """
        Initialize watsonx provider.

        Args:
            settings: Provider settings; read from the environment if None
        """
        settings = settings or WatsonxSettings()
        self._name = "IBM watsonx"
        self._api_key = settings.api_key.get_secret_value() if settings.api_key else None
        self._project_id = settings.project_id
        self._url = settings.url
        self._enabled = settings.enabled
        self._client = None
        self._credentials = None

        # Initialize client if credentials are available
        if self._api_key and self._project_id:
            self._initialize_client()

    def _initialize_client(self):
        """Initialize the watsonx client."""
        try:
            # pylint: disable=import-outside-toplevel
            from ibm_watsonx_ai import Credentials

            # pylint: disable=unused-import, import-outside-toplevel
            from ibm_watsonx_ai.foundation_models import ModelInference  # noqa: F401

            self._credentials = Credentials(url=self._url, api_key=self._api_key)
            # Client will be created per-request with specific model
        except ImportError:
            # Package not installed - watsonx will not be available
            self._credentials = None

    def is_available(self) -> bool:
        """
        Check if watsonx is properly configured.

        Returns:
            bool: True if credentials are set and valid
        """
        if not self._enabled:
            return False
        return bool(self._api_key and self._project_id and self._credentials)

    def list_models(self) -> list[str]:
        """
        Get list of available watsonx models.

        Returns:
            list[str]: List of model identifiers

        Raises:
            Exception: If credentials are not configured
        """
        if not self.is_available():
            raise RuntimeError(
                "watsonx credentials not configured. Set WATSONX_API_KEY and WATSONX_PROJECT_ID environment variables."
            )

        try:
            # pylint: disable=import-outside-toplevel
            from ibm_watsonx_ai import APIClient

            client = APIClient(self._credentials)

            # Fetch foundation models
            # pylint: disable=no-member
            models_df = client.foundation_models.get_entities()

            # Extract model IDs
            # The get_entities() returns a list of dictionaries in recent versions
            # we want to filter for chat/generate capabilities if possible,
            # but usually model_id is what we need.
            model_ids = []
            for model in models_df:
                model_id = model.get("model_id")
                if model_id:
                    model_ids.append(model_id)

            return sorted(model_ids)

        except Exception:  # pylint: disable=broad-exception-caught
            # Fallback to a curated list if API call fails
            # This ensures the app still works even if model listing fails
            return [
                "ibm/granite-3-8b-instruct",
                "ibm/granite-13b-chat-v2",
                "meta-llama/llama-3-3-70b-instruct",
                "meta-llama/llama-3-2-11b-vision-instruct",
                "mistralai/mistral-small-3-1-24b-instruct-2503",
            ]

    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        stream: bool = True,
        options: GenerationOptions | None = None,
    ) -> Iterator[ChatChunk]:
        """
        Send a chat completion request to watsonx.

        Args:
            model: watsonx model identifier
            messages: Conversation history as ChatMessage objects
            stream: Whether to stream the response
            options: Sampling options; defaults if None

        Yields:
            ChatChunk: Response chunks

        Raises:
            Exception: If the request fails or credentials are missing
        """
        if not self.is_available():
            raise RuntimeError(
                "watsonx credentials not configured. Set WATSONX_API_KEY and WATSONX_PROJECT_ID environment variables."
            )

        try:
            # pylint: disable=import-outside-toplevel
            from ibm_watsonx_ai.foundation_models import ModelInference
            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
        except ImportError as exc:
            raise ImportError(
                "ibm-watsonx-ai package not installed. Install it with: pip install ibm-watsonx-ai"
            ) from exc

        options = options or GenerationOptions()

        # Convert messages to watsonx format (concatenate into prompt)
        prompt = self._messages_to_prompt(messages)

        # Map options to watsonx parameters
        params = {
            GenParams.MAX_NEW_TOKENS: options.max_tokens,
            GenParams.TEMPERATURE: options.temperature,
            GenParams.TOP_P: options.top_p,
        }

        # Create model instance
        model_instance = ModelInference(
            model_id=model, credentials=self._credentials, project_id=self._project_id, params=params
        )

        if stream:
            # Stream response
            response_stream = model_instance.generate_text_stream(prompt=prompt)
            for chunk_text in response_stream:
                yield ChatChunk.of(chunk_text)
        else:
            # Non-streaming response
            response = model_instance.generate_text(prompt=prompt)
            yield ChatChunk.of(response)

    def _messages_to_prompt(self, messages: list[ChatMessage]) -> str:
        """
        Convert chat messages to a single prompt string.

        Args:
            messages: Conversation history as ChatMessage objects

        Returns:
            str: Formatted prompt string
        """
        prompt_parts = []

        for msg in messages:
            role = msg.role
            content = msg.content

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        # Add final assistant prompt
        prompt_parts.append("Assistant:")

        return "\n\n".join(prompt_parts)

    def get_name(self) -> str:
        """
        Get the provider name.

        Returns:
            str: "IBM watsonx"
        """
        return self._name
