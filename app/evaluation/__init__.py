"""Search quality evaluation module for Precision@K metrics."""

from app.evaluation.benchmark import (
    SEARCH_BENCHMARK,
    BenchmarkQuery,
    calculate_precision_at_k,
    run_search_benchmark,
)

__all__ = [
    "SEARCH_BENCHMARK",
    "BenchmarkQuery",
    "calculate_precision_at_k",
    "run_search_benchmark",
]
