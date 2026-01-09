"""LLM abstraction layer supporting OpenAI and Anthropic."""

from abc import ABC, abstractmethod
from typing import Optional

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from app.config import settings


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def complete(
        self, prompt: str, system: Optional[str] = None, json_mode: bool = False
    ) -> str:
        """Generate a completion from the LLM.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            json_mode: If True, request JSON output format

        Returns:
            The generated text response
        """
        pass

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate an embedding for the given text.

        Args:
            text: The text to embed

        Returns:
            A list of floats representing the embedding vector
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = "gpt-4-turbo-preview"
        self.embedding_model = settings.embedding_model

    async def complete(
        self, prompt: str, system: Optional[str] = None, json_mode: bool = False
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,  # Low temperature for more consistent outputs
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    async def embed(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model=self.embedding_model, input=text
        )
        return response.data[0].embedding


class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider implementation.

    Note: Uses OpenAI for embeddings since Anthropic doesn't have an embedding API.
    """

    def __init__(self):
        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = "claude-3-sonnet-20240229"
        # Use OpenAI for embeddings (Anthropic doesn't have embedding API)
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.embedding_model = settings.embedding_model

    async def complete(
        self, prompt: str, system: Optional[str] = None, json_mode: bool = False
    ) -> str:
        # Build the prompt - for JSON mode, add instructions
        full_prompt = prompt
        if json_mode:
            full_prompt = f"{prompt}\n\nRespond with valid JSON only."

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": full_prompt}],
        )
        return response.content[0].text

    async def embed(self, text: str) -> list[float]:
        # Use OpenAI for embeddings
        response = await self.openai_client.embeddings.create(
            model=self.embedding_model, input=text
        )
        return response.data[0].embedding


def get_llm() -> LLMProvider:
    """Factory function to get the configured LLM provider.

    Returns:
        An instance of the configured LLM provider
    """
    if settings.llm_provider == "anthropic":
        return AnthropicProvider()
    return OpenAIProvider()
