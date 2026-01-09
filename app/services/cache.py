"""Redis cache service for caching embeddings and conversation context."""

from typing import Any, Optional
import json

import redis.asyncio as redis

from app.config import settings


class CacheService:
    """Redis-based caching service."""

    def __init__(self, redis_url: Optional[str] = None):
        """Initialize the cache service.

        Args:
            redis_url: Optional Redis URL (defaults to settings)
        """
        self.redis_url = redis_url or settings.redis_url
        self._redis: Optional[redis.Redis] = None

    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection.

        Returns:
            Redis client instance
        """
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url)
        return self._redis

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        try:
            r = await self._get_redis()
            value = await r.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception:
            # Cache failures shouldn't break the app
            return None

    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable)
            ttl: Time-to-live in seconds (default 1 hour)
        """
        try:
            r = await self._get_redis()
            await r.set(key, json.dumps(value), ex=ttl)
        except Exception:
            # Cache failures shouldn't break the app
            pass

    async def delete(self, key: str) -> None:
        """Delete value from cache.

        Args:
            key: Cache key
        """
        try:
            r = await self._get_redis()
            await r.delete(key)
        except Exception:
            pass

    async def get_conversation_context(
        self, conversation_id: str, limit: int = 5
    ) -> list[dict]:
        """Get last N messages for conversation context.

        Args:
            conversation_id: The conversation ID
            limit: Max messages to return

        Returns:
            List of message dictionaries
        """
        key = f"conv:{conversation_id}"
        data = await self.get(key)
        if data:
            return data[-limit:]
        return []

    async def add_to_conversation(
        self, conversation_id: str, message: dict
    ) -> None:
        """Add message to conversation context.

        Args:
            conversation_id: The conversation ID
            message: Message data to add
        """
        key = f"conv:{conversation_id}"
        messages = await self.get(key) or []
        messages.append(message)
        # Keep last 10 messages
        await self.set(key, messages[-10:], ttl=3600)

    async def cache_intent(
        self, query_hash: str, intent: dict, ttl: int = 300
    ) -> None:
        """Cache a classified intent.

        Args:
            query_hash: Hash of the query
            intent: The classified intent
            ttl: Cache TTL (default 5 minutes)
        """
        key = f"intent:{query_hash}"
        await self.set(key, intent, ttl=ttl)

    async def get_cached_intent(self, query_hash: str) -> Optional[dict]:
        """Get cached intent classification.

        Args:
            query_hash: Hash of the query

        Returns:
            Cached intent or None
        """
        key = f"intent:{query_hash}"
        return await self.get(key)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
