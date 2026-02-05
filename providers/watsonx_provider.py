"""IBM watsonx provider implementation."""
import os
from typing import List, Dict, Any, Iterator
from .base import BaseProvider


class WatsonxProvider(BaseProvider):
    """Provider implementation for IBM watsonx.ai models."""

    def __init__(self):
        """Initialize watsonx provider with credentials from environment."""
        self._name = "IBM watsonx"
        self._api_key = os.getenv("WATSONX_API_KEY")
        self._project_id = os.getenv("WATSONX_PROJECT_ID")
        self._url = os.getenv("WATSONX_URL", "https://eu-gb.ml.cloud.ibm.com")
        self._enabled = os.getenv("WATSONX_ENABLED", "true").lower() == "true"
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
            from ibm_watsonx_ai.foundation_models import ModelInference

            self._credentials = Credentials(
                url=self._url,
                api_key=self._api_key
            )
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

    def list_models(self) -> List[str]:
        """
        Get list of available watsonx models.

        Returns:
            List[str]: List of model identifiers

        Raises:
            Exception: If credentials are not configured
        """
        if not self.is_available():
            raise RuntimeError(
                "watsonx credentials not configured. "
                "Set WATSONX_API_KEY and WATSONX_PROJECT_ID environment variables."
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
                model_id = model.get('model_id')
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
        messages: List[Dict[str, str]],
        stream: bool = True,
        options: Dict[str, Any] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Send a chat completion request to watsonx.

        Args:
            model: watsonx model identifier
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream the response
            options: Options like temperature, top_p

        Yields:
            Dict containing response chunks in Ollama-compatible format

        Raises:
            Exception: If the request fails or credentials are missing
        """
        if not self.is_available():
            raise RuntimeError(
                "watsonx credentials not configured. "
                "Set WATSONX_API_KEY and WATSONX_PROJECT_ID environment variables."
            )

        try:
            # pylint: disable=import-outside-toplevel
            from ibm_watsonx_ai.foundation_models import ModelInference
            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
        except ImportError as exc:
            raise ImportError(
                "ibm-watsonx-ai package not installed. "
                "Install it with: pip install ibm-watsonx-ai"
            ) from exc

        if options is None:
            options = {}

        # Convert messages to watsonx format (concatenate into prompt)
        prompt = self._messages_to_prompt(messages)

        # Map options to watsonx parameters
        params = {
            GenParams.MAX_NEW_TOKENS: options.get("max_tokens", 1024),
            GenParams.TEMPERATURE: options.get("temperature", 0.7),
            GenParams.TOP_P: options.get("top_p", 0.9),
        }

        # Create model instance
        model_instance = ModelInference(
            model_id=model,
            credentials=self._credentials,
            project_id=self._project_id,
            params=params
        )

        if stream:
            # Stream response
            response_stream = model_instance.generate_text_stream(prompt=prompt)
            for chunk_text in response_stream:
                # Convert to Ollama-compatible format
                yield {
                    'message': {
                        'content': chunk_text
                    }
                }
        else:
            # Non-streaming response
            response = model_instance.generate_text(prompt=prompt)
            yield {
                'message': {
                    'content': response
                }
            }

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert chat messages to a single prompt string.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            str: Formatted prompt string
        """
        prompt_parts = []

        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
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
