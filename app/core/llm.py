"""LLM abstraction layer supporting OpenAI and Anthropic with failover."""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from openai import AsyncOpenAI, APIError as OpenAIError, RateLimitError as OpenAIRateLimitError
from anthropic import AsyncAnthropic, APIError as AnthropicError, RateLimitError as AnthropicRateLimitError

from app.config import settings
from app.utils.resilience import (
    with_retry,
    circuit_breakers,
    RateLimitError,
    ServiceUnavailableError,
    CircuitOpenError,
)

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging and circuit breaker."""
        pass

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
    """OpenAI LLM provider implementation with retry and circuit breaker."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = "gpt-4.1-mini"  # Fast, cheap, capable model
        self.embedding_model = settings.embedding_model
        self.circuit = circuit_breakers.get("openai")

    @property
    def name(self) -> str:
        return "openai"

    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    async def complete(
        self, prompt: str, system: Optional[str] = None, json_mode: bool = False
    ) -> str:
        """Generate completion with retry and circuit breaker."""
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

        try:
            # Use circuit breaker to wrap the API call
            async def _call():
                return await self.client.chat.completions.create(**kwargs)

            response = await self.circuit.call(_call)
            return response.choices[0].message.content

        except OpenAIRateLimitError as e:
            logger.warning(f"OpenAI rate limit: {e}")
            raise RateLimitError(str(e))

        except OpenAIError as e:
            if "500" in str(e) or "502" in str(e) or "503" in str(e):
                raise ServiceUnavailableError(f"OpenAI server error: {e}")
            raise

    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    async def embed(self, text: str) -> list[float]:
        """Generate embedding with retry."""
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model, input=text
            )
            return response.data[0].embedding

        except OpenAIRateLimitError as e:
            logger.warning(f"OpenAI embedding rate limit: {e}")
            raise RateLimitError(str(e))

        except OpenAIError as e:
            if "500" in str(e) or "502" in str(e) or "503" in str(e):
                raise ServiceUnavailableError(f"OpenAI server error: {e}")
            raise


class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider implementation with retry and circuit breaker.

    Note: Uses OpenAI for embeddings since Anthropic doesn't have an embedding API.
    """

    def __init__(self):
        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = "claude-haiku-4-5-20251001"  # Fast, cheap ($1/$5 per 1M tokens)
        # Use OpenAI for embeddings (Anthropic doesn't have embedding API)
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.embedding_model = settings.embedding_model
        self.circuit = circuit_breakers.get("anthropic")

    @property
    def name(self) -> str:
        return "anthropic"

    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    async def complete(
        self, prompt: str, system: Optional[str] = None, json_mode: bool = False
    ) -> str:
        """Generate completion with retry and circuit breaker."""
        # Build the prompt - for JSON mode, add instructions
        full_prompt = prompt
        if json_mode:
            full_prompt = f"{prompt}\n\nRespond with valid JSON only."

        try:
            async def _call():
                return await self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system or "You are a helpful assistant.",
                    messages=[{"role": "user", "content": full_prompt}],
                )

            response = await self.circuit.call(_call)
            return response.content[0].text

        except AnthropicRateLimitError as e:
            logger.warning(f"Anthropic rate limit: {e}")
            raise RateLimitError(str(e))

        except AnthropicError as e:
            if "500" in str(e) or "502" in str(e) or "503" in str(e):
                raise ServiceUnavailableError(f"Anthropic server error: {e}")
            raise

    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    async def embed(self, text: str) -> list[float]:
        """Generate embedding with retry (uses OpenAI)."""
        try:
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model, input=text
            )
            return response.data[0].embedding

        except OpenAIRateLimitError as e:
            logger.warning(f"OpenAI embedding rate limit: {e}")
            raise RateLimitError(str(e))


class LLMServiceWithFailover:
    """LLM service with automatic failover between providers.

    Tries primary provider first, falls back to secondary on failure.
    Uses circuit breakers to avoid repeated calls to failing services.
    """

    def __init__(self):
        # Set up primary and fallback based on config
        if settings.llm_provider == "anthropic":
            self.primary = AnthropicProvider()
            self.fallback = OpenAIProvider()
        else:
            self.primary = OpenAIProvider()
            self.fallback = AnthropicProvider()

        self._last_provider_used: Optional[str] = None

    @property
    def last_provider_used(self) -> Optional[str]:
        """Get the name of the last provider that was used."""
        return self._last_provider_used

    async def complete(
        self, prompt: str, system: Optional[str] = None, json_mode: bool = False
    ) -> str:
        """Generate completion with automatic failover.

        Tries primary provider first. If it fails (rate limit, server error,
        circuit open), falls back to secondary provider.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            json_mode: If True, request JSON output format

        Returns:
            The generated text response

        Raises:
            Exception: If both providers fail
        """
        # Try primary provider
        try:
            result = await self.primary.complete(prompt, system, json_mode)
            self._last_provider_used = self.primary.name
            return result

        except (RateLimitError, ServiceUnavailableError, CircuitOpenError) as e:
            logger.warning(
                f"Primary LLM ({self.primary.name}) failed: {e}. "
                f"Falling back to {self.fallback.name}"
            )

        except Exception as e:
            logger.warning(
                f"Primary LLM ({self.primary.name}) unexpected error: {e}. "
                f"Falling back to {self.fallback.name}"
            )

        # Try fallback provider
        try:
            result = await self.fallback.complete(prompt, system, json_mode)
            self._last_provider_used = self.fallback.name
            logger.info(f"Fallback to {self.fallback.name} succeeded")
            return result

        except Exception as e:
            logger.error(f"Fallback LLM ({self.fallback.name}) also failed: {e}")
            raise RuntimeError(
                f"All LLM providers failed. Primary ({self.primary.name}): circuit/error, "
                f"Fallback ({self.fallback.name}): {e}"
            )

    async def embed(self, text: str) -> list[float]:
        """Generate embedding with failover.

        Both providers use OpenAI for embeddings, so this primarily
        provides retry logic.

        Args:
            text: The text to embed

        Returns:
            Embedding vector
        """
        try:
            return await self.primary.embed(text)
        except Exception as e:
            logger.warning(f"Primary embedding failed: {e}, trying fallback")
            return await self.fallback.embed(text)


# Global instance for easy access
_llm_service: Optional[LLMServiceWithFailover] = None


def get_llm() -> LLMProvider:
    """Factory function to get the configured LLM provider.

    Returns:
        An instance of the configured LLM provider (without failover)
    """
    if settings.llm_provider == "anthropic":
        return AnthropicProvider()
    return OpenAIProvider()


def get_llm_with_failover() -> LLMServiceWithFailover:
    """Get the LLM service with automatic failover.

    Returns:
        LLMServiceWithFailover instance
    """
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMServiceWithFailover()
    return _llm_service
