"""Utility modules for the application."""

from app.utils.resilience import (
    # Exceptions
    ExternalAPIError,
    RateLimitError,
    QuotaExceededError,
    TokenExpiredError,
    ServiceUnavailableError,
    CircuitOpenError,
    # Retry decorators
    with_retry,
    with_retry_sync,
    # Circuit breaker
    CircuitBreaker,
    CircuitState,
    CircuitBreakerRegistry,
    circuit_breakers,
    # Helpers
    classify_http_error,
    classify_google_error,
    google_api_call_with_retry,
)

__all__ = [
    # Exceptions
    "ExternalAPIError",
    "RateLimitError",
    "QuotaExceededError",
    "TokenExpiredError",
    "ServiceUnavailableError",
    "CircuitOpenError",
    # Retry decorators
    "with_retry",
    "with_retry_sync",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerRegistry",
    "circuit_breakers",
    # Helpers
    "classify_http_error",
    "classify_google_error",
    "google_api_call_with_retry",
]
