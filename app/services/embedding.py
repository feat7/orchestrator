"""Embedding service for generating and caching embeddings."""

from typing import Optional
import hashlib

from app.core.llm import get_llm, LLMProvider
from app.services.cache import CacheService


class EmbeddingService:
    """Service for generating text embeddings with optional caching."""

    def __init__(
        self, llm: Optional[LLMProvider] = None, cache: Optional[CacheService] = None
    ):
        """Initialize the embedding service.

        Args:
            llm: Optional LLM provider (defaults to configured provider)
            cache: Optional cache service for embedding caching
        """
        self.llm = llm or get_llm()
        self.cache = cache

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text with optional caching.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        # Check cache first
        if self.cache:
            cache_key = self._get_cache_key(text)
            cached = await self.cache.get(cache_key)
            if cached:
                return cached

        # Generate embedding
        embedding = await self.llm.embed(text)

        # Cache for 1 hour
        if self.cache:
            await self.cache.set(cache_key, embedding, ttl=3600)

        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # For now, embed individually (could optimize with batching)
        return [await self.embed(text) for text in texts]

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text.

        Args:
            text: The text to hash

        Returns:
            Cache key string
        """
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"emb:{text_hash}"

    async def embed_for_storage(self, text: str) -> list[float]:
        """Generate embedding for database storage (no caching).

        Use this when storing embeddings in the database, as those
        don't need to be cached separately.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        return await self.llm.embed(text)
