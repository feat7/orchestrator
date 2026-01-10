"""Resilience utilities: retry with backoff, circuit breaker, rate limiting."""

import asyncio
import random
import logging
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import TypeVar, Callable, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# Custom Exceptions
# =============================================================================

class ExternalAPIError(Exception):
    """Base exception for external API errors."""
    pass


class RateLimitError(ExternalAPIError):
    """Raised when rate limit is hit (HTTP 429)."""
    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class QuotaExceededError(ExternalAPIError):
    """Raised when daily quota is exceeded."""
    pass


class TokenExpiredError(ExternalAPIError):
    """Raised when OAuth token is expired or revoked."""
    pass


class ServiceUnavailableError(ExternalAPIError):
    """Raised when service is temporarily unavailable (5xx errors)."""
    pass


class CircuitOpenError(ExternalAPIError):
    """Raised when circuit breaker is open."""
    def __init__(self, message: str, retry_after: float = 0):
        super().__init__(message)
        self.retry_after = retry_after


# =============================================================================
# Retry Decorator with Exponential Backoff
# =============================================================================

def with_retry(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.25,
    retryable_exceptions: tuple = (
        RateLimitError,
        ServiceUnavailableError,
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
    ),
    non_retryable_exceptions: tuple = (
        QuotaExceededError,
        TokenExpiredError,
    ),
):
    """Decorator for retrying async functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 5)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay cap in seconds (default: 60.0)
        jitter: Random jitter factor ±percentage (default: 0.25 = ±25%)
        retryable_exceptions: Tuple of exception types to retry
        non_retryable_exceptions: Tuple of exception types to never retry

    Example:
        @with_retry(max_retries=3, base_delay=1.0)
        async def call_external_api():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except non_retryable_exceptions as e:
                    # Don't retry these - they won't succeed
                    logger.error(f"{func.__name__}: Non-retryable error: {e}")
                    raise

                except RateLimitError as e:
                    last_exception = e
                    # Use Retry-After header if provided
                    if e.retry_after:
                        delay = e.retry_after
                    else:
                        delay = min(base_delay * (2 ** attempt), max_delay)

                    # Add jitter to prevent thundering herd
                    delay *= (1 + random.uniform(-jitter, jitter))

                    logger.warning(
                        f"{func.__name__}: Rate limited, attempt {attempt + 1}/{max_retries + 1}, "
                        f"waiting {delay:.2f}s"
                    )

                    if attempt < max_retries:
                        await asyncio.sleep(delay)
                    else:
                        raise

                except retryable_exceptions as e:
                    last_exception = e
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    delay *= (1 + random.uniform(-jitter, jitter))

                    logger.warning(
                        f"{func.__name__}: Retryable error ({type(e).__name__}), "
                        f"attempt {attempt + 1}/{max_retries + 1}, waiting {delay:.2f}s"
                    )

                    if attempt < max_retries:
                        await asyncio.sleep(delay)
                    else:
                        raise

                except Exception as e:
                    # Unexpected exceptions - don't retry
                    logger.error(f"{func.__name__}: Unexpected error: {e}")
                    raise

            # All retries exhausted
            raise last_exception

        return wrapper
    return decorator


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation - requests allowed
    OPEN = "open"           # Failing - requests blocked
    HALF_OPEN = "half_open" # Testing - limited requests allowed


@dataclass
class CircuitBreaker:
    """Circuit breaker for external API calls.

    Prevents cascading failures by "opening" the circuit when too many
    failures occur, blocking requests until the service recovers.

    States:
        CLOSED: Normal operation, all requests pass through
        OPEN: Service is failing, all requests are rejected immediately
        HALF_OPEN: Testing if service recovered, allowing limited requests

    Example:
        breaker = CircuitBreaker(name="gmail", failure_threshold=5)

        try:
            result = await breaker.call(gmail_api.search, query="test")
        except CircuitOpenError:
            # Circuit is open, service is down
            return fallback_response()
    """

    name: str
    failure_threshold: int = 5       # Failures before opening circuit
    success_threshold: int = 2       # Successes in half-open before closing
    timeout: float = 30.0            # Seconds before trying half-open
    failure_window: float = 60.0     # Window in seconds to count failures

    # Internal state (not constructor args)
    state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    failures: list = field(default_factory=list, init=False)
    successes_in_half_open: int = field(default=0, init=False)
    last_failure_time: Optional[datetime] = field(default=None, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker.

        Args:
            func: Async function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitOpenError: If circuit is open
            Any exception from func
        """
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_try_half_open():
                    logger.info(f"Circuit {self.name}: Transitioning to HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                    self.successes_in_half_open = 0
                else:
                    retry_after = self._time_until_half_open()
                    raise CircuitOpenError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Service unavailable. Try again in {retry_after:.0f}s",
                        retry_after=retry_after
                    )

        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result

        except Exception as e:
            await self._record_failure()
            raise

    async def _record_success(self):
        """Record a successful call."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.successes_in_half_open += 1
                if self.successes_in_half_open >= self.success_threshold:
                    logger.info(f"Circuit {self.name}: Closing circuit (recovered)")
                    self.state = CircuitState.CLOSED
                    self.failures = []

    async def _record_failure(self):
        """Record a failed call."""
        async with self._lock:
            now = datetime.utcnow()
            self.failures.append(now)
            self.last_failure_time = now

            # Remove old failures outside the window
            cutoff = now - timedelta(seconds=self.failure_window)
            self.failures = [f for f in self.failures if f > cutoff]

            if self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                logger.warning(f"Circuit {self.name}: Failure in HALF_OPEN, reopening")
                self.state = CircuitState.OPEN

            elif len(self.failures) >= self.failure_threshold:
                logger.warning(
                    f"Circuit {self.name}: Opening circuit "
                    f"({len(self.failures)} failures in {self.failure_window}s)"
                )
                self.state = CircuitState.OPEN

    def _should_try_half_open(self) -> bool:
        """Check if enough time has passed to try half-open."""
        if not self.last_failure_time:
            return True
        elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout

    def _time_until_half_open(self) -> float:
        """Get seconds until circuit will try half-open."""
        if not self.last_failure_time:
            return 0
        elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return max(0, self.timeout - elapsed)

    def get_state(self) -> dict:
        """Get current circuit state for monitoring."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failures_in_window": len(self.failures),
            "failure_threshold": self.failure_threshold,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "time_until_half_open": self._time_until_half_open() if self.state == CircuitState.OPEN else 0,
        }


# =============================================================================
# Circuit Breaker Registry
# =============================================================================

class CircuitBreakerRegistry:
    """Registry for managing circuit breakers across services."""

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}

    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 30.0,
        failure_window: float = 60.0,
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                success_threshold=success_threshold,
                timeout=timeout,
                failure_window=failure_window,
            )
        return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)

    def get_all_states(self) -> list[dict]:
        """Get states of all circuit breakers."""
        return [breaker.get_state() for breaker in self._breakers.values()]

    def reset(self, name: str):
        """Reset a circuit breaker to closed state."""
        if name in self._breakers:
            breaker = self._breakers[name]
            breaker.state = CircuitState.CLOSED
            breaker.failures = []
            breaker.successes_in_half_open = 0
            breaker.last_failure_time = None


# Global registry instance
circuit_breakers = CircuitBreakerRegistry()

# Pre-create circuit breakers for known services
circuit_breakers.get_or_create("gmail", failure_threshold=5, timeout=30)
circuit_breakers.get_or_create("calendar", failure_threshold=5, timeout=30)
circuit_breakers.get_or_create("drive", failure_threshold=5, timeout=30)
circuit_breakers.get_or_create("openai", failure_threshold=3, timeout=60)
circuit_breakers.get_or_create("anthropic", failure_threshold=3, timeout=60)


# =============================================================================
# HTTP Error Detection Helpers
# =============================================================================

def classify_http_error(status_code: int, error_body: str = "") -> ExternalAPIError:
    """Classify HTTP error into appropriate exception type.

    Args:
        status_code: HTTP status code
        error_body: Response body for additional context

    Returns:
        Appropriate exception instance
    """
    if status_code == 429:
        # Rate limited
        return RateLimitError("Rate limit exceeded")

    elif status_code == 401:
        # Unauthorized - token expired or invalid
        return TokenExpiredError("Authentication failed - token may be expired")

    elif status_code == 403:
        # Forbidden - could be quota or permissions
        if "quota" in error_body.lower():
            return QuotaExceededError("API quota exceeded")
        return TokenExpiredError("Access denied - permissions may have been revoked")

    elif status_code >= 500:
        # Server error - retryable
        return ServiceUnavailableError(f"Service unavailable (HTTP {status_code})")

    else:
        # Other errors - not retryable
        return ExternalAPIError(f"API error (HTTP {status_code})")


def classify_google_error(exception: Exception) -> ExternalAPIError:
    """Classify Google API exception into appropriate type.

    Args:
        exception: Exception from Google API client

    Returns:
        Appropriate exception instance
    """
    error_str = str(exception).lower()

    # Check for common Google API error patterns
    if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
        return RateLimitError("Google API rate limit exceeded")

    elif "401" in error_str or "invalid credentials" in error_str:
        return TokenExpiredError("Google OAuth token expired or invalid")

    elif "403" in error_str:
        if "quota" in error_str:
            return QuotaExceededError("Google API quota exceeded")
        return TokenExpiredError("Google API access denied")

    elif "500" in error_str or "503" in error_str or "502" in error_str:
        return ServiceUnavailableError("Google API temporarily unavailable")

    elif "timeout" in error_str or "timed out" in error_str:
        return ServiceUnavailableError("Google API request timed out")

    else:
        return ExternalAPIError(f"Google API error: {exception}")


# =============================================================================
# Synchronous Retry Decorator (for Celery tasks)
# =============================================================================

def with_retry_sync(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.25,
    retryable_exceptions: tuple = (
        RateLimitError,
        ServiceUnavailableError,
        ConnectionError,
        TimeoutError,
    ),
    non_retryable_exceptions: tuple = (
        QuotaExceededError,
        TokenExpiredError,
    ),
):
    """Synchronous version of retry decorator for Celery tasks.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        jitter: Random jitter factor ±percentage
        retryable_exceptions: Tuple of exception types to retry
        non_retryable_exceptions: Tuple of exception types to never retry

    Example:
        @with_retry_sync(max_retries=3)
        def sync_gmail(user_id: str):
            ...
    """
    import time

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except non_retryable_exceptions as e:
                    logger.error(f"{func.__name__}: Non-retryable error: {e}")
                    raise

                except RateLimitError as e:
                    last_exception = e
                    if e.retry_after:
                        delay = e.retry_after
                    else:
                        delay = min(base_delay * (2 ** attempt), max_delay)

                    delay *= (1 + random.uniform(-jitter, jitter))

                    logger.warning(
                        f"{func.__name__}: Rate limited, attempt {attempt + 1}/{max_retries + 1}, "
                        f"waiting {delay:.2f}s"
                    )

                    if attempt < max_retries:
                        time.sleep(delay)
                    else:
                        raise

                except retryable_exceptions as e:
                    last_exception = e
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    delay *= (1 + random.uniform(-jitter, jitter))

                    logger.warning(
                        f"{func.__name__}: Retryable error ({type(e).__name__}), "
                        f"attempt {attempt + 1}/{max_retries + 1}, waiting {delay:.2f}s"
                    )

                    if attempt < max_retries:
                        time.sleep(delay)
                    else:
                        raise

                except Exception as e:
                    logger.error(f"{func.__name__}: Unexpected error: {e}")
                    raise

            raise last_exception

        return wrapper
    return decorator


def google_api_call_with_retry(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Any:
    """Execute a Google API call with retry logic.

    This is a helper for wrapping individual Google API calls in sync code.

    Args:
        func: Callable that makes the Google API request (should call .execute())
        max_retries: Maximum retry attempts
        base_delay: Initial delay for exponential backoff

    Returns:
        Result from the API call

    Raises:
        The classified exception after all retries exhausted

    Example:
        result = google_api_call_with_retry(
            lambda: service.users().messages().get(userId="me", id=msg_id).execute()
        )
    """
    import time

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()

        except Exception as e:
            error_str = str(e).lower()

            # Check for rate limit (429)
            if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                delay = base_delay * (2 ** attempt) * (1 + random.uniform(-0.25, 0.25))

                logger.warning(
                    f"Google API rate limit (429), attempt {attempt + 1}/{max_retries + 1}, "
                    f"waiting {delay:.2f}s"
                )

                if attempt < max_retries:
                    time.sleep(delay)
                    last_exception = RateLimitError(str(e))
                    continue
                else:
                    raise RateLimitError(str(e))

            # Check for server error (5xx)
            elif any(code in error_str for code in ["500", "502", "503", "504"]):
                delay = base_delay * (2 ** attempt) * (1 + random.uniform(-0.25, 0.25))

                logger.warning(
                    f"Google API server error, attempt {attempt + 1}/{max_retries + 1}, "
                    f"waiting {delay:.2f}s"
                )

                if attempt < max_retries:
                    time.sleep(delay)
                    last_exception = ServiceUnavailableError(str(e))
                    continue
                else:
                    raise ServiceUnavailableError(str(e))

            # Token expired - don't retry
            elif "401" in error_str or "invalid credentials" in error_str:
                raise TokenExpiredError("OAuth token expired or invalid")

            # Quota exceeded - don't retry
            elif "403" in error_str and "quota" in error_str:
                raise QuotaExceededError(str(e))

            # Connection/timeout errors - retry
            elif "timeout" in error_str or "connection" in error_str:
                delay = base_delay * (2 ** attempt) * (1 + random.uniform(-0.25, 0.25))

                logger.warning(
                    f"Google API connection error, attempt {attempt + 1}/{max_retries + 1}, "
                    f"waiting {delay:.2f}s"
                )

                if attempt < max_retries:
                    time.sleep(delay)
                    last_exception = ServiceUnavailableError(str(e))
                    continue
                else:
                    raise ServiceUnavailableError(str(e))

            # Other errors - don't retry
            else:
                raise

    if last_exception:
        raise last_exception
