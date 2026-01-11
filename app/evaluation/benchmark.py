"""Search benchmark for Precision@K evaluation.

This module provides ground-truth evaluation for search quality.
Each benchmark query has a relevance checker that determines if a
search result is relevant based on expected patterns.

Target: Precision@5 > 0.8
"""

from dataclasses import dataclass
from typing import Callable, Any, Optional
import re


@dataclass
class BenchmarkQuery:
    """A benchmark query with relevance criteria."""

    query: str
    service: str  # gmail, gcal, gdrive
    description: str  # Human-readable description
    relevance_check: Callable[[dict], bool]  # Function to check if result is relevant
    min_expected_results: int = 1  # Minimum results expected for valid test


def _check_keywords_in_fields(result: dict, keywords: list[str], fields: list[str]) -> bool:
    """Check if any keyword appears in any of the specified fields."""
    for field in fields:
        value = str(result.get(field, "")).lower()
        for keyword in keywords:
            if keyword.lower() in value:
                return True
    return False


def _check_sender_pattern(result: dict, pattern: str) -> bool:
    """Check if sender matches a pattern."""
    sender = str(result.get("sender", "")).lower()
    return pattern.lower() in sender


def _check_attendee_pattern(result: dict, pattern: str) -> bool:
    """Check if any attendee matches a pattern."""
    attendees = result.get("attendees", [])
    if isinstance(attendees, list):
        for attendee in attendees:
            if isinstance(attendee, dict):
                email = attendee.get("email", "")
            else:
                email = str(attendee)
            if pattern.lower() in email.lower():
                return True
    return False


def _check_mime_type(result: dict, mime_types: list[str]) -> bool:
    """Check if file matches expected MIME types."""
    mime = result.get("mime_type", "").lower()
    return any(m.lower() in mime for m in mime_types)


# =============================================================================
# Benchmark Queries
# These are customizable - update to match your synced data for accurate P@5
# =============================================================================

SEARCH_BENCHMARK: list[BenchmarkQuery] = [
    # ---------------------------------------------------------------------
    # Gmail Benchmarks - Customized for demo data
    # ---------------------------------------------------------------------
    BenchmarkQuery(
        query="ComfyUI workflow update",
        service="gmail",
        description="Find AI workflow update emails",
        relevance_check=lambda r: _check_keywords_in_fields(
            r, ["comfyui", "workflow", "update", "wan", "ltx", "qwen"],
            ["subject", "body_preview"]
        ),
    ),
    BenchmarkQuery(
        query="Patreon workflow notification",
        service="gmail",
        description="Find Patreon workflow emails",
        relevance_check=lambda r: (
            _check_keywords_in_fields(r, ["patreon", "workflow", "update"], ["subject", "body_preview"])
            or _check_sender_pattern(r, "patreon")
        ),
    ),
    BenchmarkQuery(
        query="Roboflow AI vision",
        service="gmail",
        description="Find Roboflow AI emails",
        relevance_check=lambda r: (
            _check_keywords_in_fields(r, ["roboflow", "vision", "ai", "model"], ["subject", "body_preview"])
            or _check_sender_pattern(r, "roboflow")
        ),
    ),
    BenchmarkQuery(
        query="LTX video generation",
        service="gmail",
        description="Find LTX video generation emails",
        relevance_check=lambda r: _check_keywords_in_fields(
            r, ["ltx", "video", "generation", "comfyui"],
            ["subject", "body_preview"]
        ),
    ),
    BenchmarkQuery(
        query="Qwen image editing",
        service="gmail",
        description="Find Qwen image editing emails",
        relevance_check=lambda r: _check_keywords_in_fields(
            r, ["qwen", "image", "edit", "text2img"],
            ["subject", "body_preview"]
        ),
    ),
    BenchmarkQuery(
        query="Instagram notification",
        service="gmail",
        description="Find Instagram notification emails",
        relevance_check=lambda r: (
            _check_keywords_in_fields(r, ["instagram", "reels", "posts"], ["subject", "body_preview"])
            or _check_sender_pattern(r, "instagram")
        ),
    ),
    BenchmarkQuery(
        query="Google account security",
        service="gmail",
        description="Find Google account emails",
        relevance_check=lambda r: (
            _check_keywords_in_fields(r, ["google", "account", "security", "privacy"], ["subject", "body_preview"])
            or _check_sender_pattern(r, "google.com")
        ),
    ),

    # ---------------------------------------------------------------------
    # Calendar Benchmarks - Generic (for when calendar data is synced)
    # ---------------------------------------------------------------------
    BenchmarkQuery(
        query="team standup meeting",
        service="gcal",
        description="Find team standup events",
        relevance_check=lambda r: _check_keywords_in_fields(
            r, ["standup", "stand-up", "daily", "team", "scrum", "sync"],
            ["title", "description"]
        ),
    ),
    BenchmarkQuery(
        query="client meeting",
        service="gcal",
        description="Find client meetings",
        relevance_check=lambda r: _check_keywords_in_fields(
            r, ["client", "customer", "meeting", "call"],
            ["title", "description"]
        ),
    ),
    BenchmarkQuery(
        query="one on one 1:1",
        service="gcal",
        description="Find 1:1 meetings",
        relevance_check=lambda r: _check_keywords_in_fields(
            r, ["1:1", "one on one", "1-on-1", "1on1", "sync"],
            ["title", "description"]
        ),
    ),
    BenchmarkQuery(
        query="project review",
        service="gcal",
        description="Find project review meetings",
        relevance_check=lambda r: _check_keywords_in_fields(
            r, ["review", "project", "sprint", "demo", "retrospective"],
            ["title", "description"]
        ),
    ),

    # ---------------------------------------------------------------------
    # Drive Benchmarks - Generic
    # ---------------------------------------------------------------------
    BenchmarkQuery(
        query="orchestrator document",
        service="gdrive",
        description="Find orchestrator documents",
        relevance_check=lambda r: _check_keywords_in_fields(
            r, ["orchestrat", "document", "doc"],
            ["name", "content_preview"]
        ),
    ),
    BenchmarkQuery(
        query="project document",
        service="gdrive",
        description="Find project documents",
        relevance_check=lambda r: _check_keywords_in_fields(
            r, ["project", "document", "doc"],
            ["name", "content_preview"]
        ) or _check_mime_type(r, ["document"]),
    ),
    BenchmarkQuery(
        query="spreadsheet data",
        service="gdrive",
        description="Find spreadsheet files",
        relevance_check=lambda r: _check_mime_type(r, ["spreadsheet", "sheet", "excel"]),
    ),
    BenchmarkQuery(
        query="presentation slides",
        service="gdrive",
        description="Find presentation files",
        relevance_check=lambda r: _check_mime_type(r, ["presentation", "slide", "powerpoint"]),
    ),
]


def calculate_precision_at_k(
    results: list[dict],
    relevance_check: Callable[[dict], bool],
    k: int = 5,
) -> tuple[float, int, int]:
    """Calculate Precision@K for a single query.

    Precision@K = (# of relevant results in top K) / K

    Args:
        results: Search results (ordered by rank)
        relevance_check: Function to determine if a result is relevant
        k: Number of top results to consider (default: 5)

    Returns:
        Tuple of (precision_at_k, relevant_count, total_in_top_k)
    """
    top_k = results[:k]
    if not top_k:
        return 0.0, 0, 0

    relevant_count = sum(1 for r in top_k if relevance_check(r))
    precision = relevant_count / k

    return precision, relevant_count, len(top_k)


async def run_search_benchmark(
    agents: dict,
    user_id: str,
    embedding_service: Any,
) -> dict:
    """Run all benchmark queries and calculate overall Precision@5.

    Args:
        agents: Dict mapping service name to agent instance
        user_id: User ID to run searches for
        embedding_service: Embedding service for query embedding

    Returns:
        Dict with overall P@5 and per-query results
    """
    results = []
    queries_with_results = 0

    for bq in SEARCH_BENCHMARK:
        # Map service name to agent
        agent = agents.get(bq.service)
        if not agent:
            continue

        try:
            # Run the search
            search_results = await agent.search(bq.query, user_id, {})

            # Skip if no results (can't evaluate)
            if not search_results:
                results.append({
                    "query": bq.query,
                    "service": bq.service,
                    "description": bq.description,
                    "precision_at_5": None,
                    "relevant_count": 0,
                    "total_results": 0,
                    "skipped": True,
                    "reason": "no_results",
                })
                continue

            queries_with_results += 1

            # Calculate P@5
            precision, relevant, total = calculate_precision_at_k(
                search_results,
                bq.relevance_check,
                k=5
            )

            results.append({
                "query": bq.query,
                "service": bq.service,
                "description": bq.description,
                "precision_at_5": round(precision, 3),
                "relevant_count": relevant,
                "total_results": len(search_results),
                "top_5_results": [
                    {
                        "id": r.get("id"),
                        "title": r.get("subject") or r.get("title") or r.get("name"),
                        "is_relevant": bq.relevance_check(r),
                        "score": r.get("similarity") or r.get("rrf_score"),
                    }
                    for r in search_results[:5]
                ],
            })

        except Exception as e:
            results.append({
                "query": bq.query,
                "service": bq.service,
                "description": bq.description,
                "precision_at_5": None,
                "error": str(e),
            })

    # Calculate overall P@5 (average across queries with results)
    valid_precisions = [r["precision_at_5"] for r in results if r.get("precision_at_5") is not None]

    if valid_precisions:
        overall_precision = sum(valid_precisions) / len(valid_precisions)
    else:
        overall_precision = None

    return {
        "overall_precision_at_5": round(overall_precision, 3) if overall_precision else None,
        "target_precision": 0.8,
        "meets_target": overall_precision >= 0.8 if overall_precision else False,
        "total_queries": len(SEARCH_BENCHMARK),
        "queries_evaluated": queries_with_results,
        "queries_skipped": len(SEARCH_BENCHMARK) - queries_with_results,
        "per_service": _aggregate_by_service(results),
        "details": results,
    }


def _aggregate_by_service(results: list[dict]) -> dict:
    """Aggregate P@5 results by service."""
    service_results = {}

    for r in results:
        service = r["service"]
        if service not in service_results:
            service_results[service] = {
                "precisions": [],
                "total_queries": 0,
                "evaluated": 0,
            }

        service_results[service]["total_queries"] += 1
        if r.get("precision_at_5") is not None:
            service_results[service]["precisions"].append(r["precision_at_5"])
            service_results[service]["evaluated"] += 1

    # Calculate average per service
    aggregated = {}
    for service, data in service_results.items():
        if data["precisions"]:
            avg = sum(data["precisions"]) / len(data["precisions"])
        else:
            avg = None

        aggregated[service] = {
            "precision_at_5": round(avg, 3) if avg else None,
            "queries_evaluated": data["evaluated"],
            "total_queries": data["total_queries"],
        }

    return aggregated
