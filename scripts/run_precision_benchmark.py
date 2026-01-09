#!/usr/bin/env python3
"""Run Precision@5 benchmark against synced data.

Usage:
    python scripts/run_precision_benchmark.py
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select, func
from app.db.database import async_session
from app.db.models import User, GmailCache, GcalCache, GdriveCache
from app.services.google.gmail import GmailService
from app.services.google.calendar import CalendarService
from app.services.google.drive import DriveService
from app.services.embedding import EmbeddingService
from app.agents.gmail import GmailAgent
from app.agents.gcal import GcalAgent
from app.agents.gdrive import GdriveAgent
from app.core.llm import get_llm
from app.evaluation.benchmark import run_search_benchmark, SEARCH_BENCHMARK


async def get_data_stats(db):
    """Get stats about synced data."""
    # Get user
    result = await db.execute(select(User).limit(1))
    user = result.scalar_one_or_none()

    if not user:
        return None, {}

    # Count items per service
    gmail_count = await db.execute(
        select(func.count(GmailCache.id)).where(GmailCache.user_id == user.id)
    )
    gcal_count = await db.execute(
        select(func.count(GcalCache.id)).where(GcalCache.user_id == user.id)
    )
    gdrive_count = await db.execute(
        select(func.count(GdriveCache.id)).where(GdriveCache.user_id == user.id)
    )

    stats = {
        "gmail": gmail_count.scalar() or 0,
        "gcal": gcal_count.scalar() or 0,
        "gdrive": gdrive_count.scalar() or 0,
    }

    return user, stats


async def run_individual_queries(agents, user_id, embedding_service):
    """Run benchmark queries individually and show detailed results."""
    print("\n" + "=" * 70)
    print("DETAILED QUERY RESULTS")
    print("=" * 70)

    for bq in SEARCH_BENCHMARK:
        agent = agents.get(bq.service)
        if not agent:
            continue

        print(f"\n{'─' * 70}")
        print(f"Query: \"{bq.query}\"")
        print(f"Service: {bq.service}")
        print(f"Description: {bq.description}")
        print("─" * 70)

        try:
            results = await agent.search(bq.query, user_id, {})

            if not results:
                print("  No results found")
                continue

            print(f"  Found {len(results)} results, showing top 5:")
            print()

            relevant_count = 0
            for i, r in enumerate(results[:5], 1):
                # Determine title based on service
                if bq.service == "gmail":
                    title = r.get("subject", "No subject")
                    extra = f"from: {r.get('sender', 'unknown')}"
                elif bq.service == "gcal":
                    title = r.get("title", "No title")
                    extra = f"time: {r.get('start_time', 'unknown')}"
                else:  # gdrive
                    title = r.get("name", "No name")
                    extra = f"type: {r.get('mime_type', 'unknown')}"

                # Check relevance
                is_relevant = bq.relevance_check(r)
                if is_relevant:
                    relevant_count += 1

                relevance_marker = "✓" if is_relevant else "✗"
                score = r.get("similarity") or r.get("rrf_score") or 0

                print(f"  {i}. [{relevance_marker}] {title[:50]}")
                print(f"      {extra}")
                print(f"      Score: {score:.3f}")

            precision = relevant_count / 5
            print()
            print(f"  Precision@5: {precision:.2f} ({relevant_count}/5 relevant)")

        except Exception as e:
            print(f"  Error: {e}")


async def main():
    print("=" * 70)
    print("PRECISION@5 BENCHMARK")
    print("Target: P@5 > 0.8")
    print("=" * 70)

    async with async_session() as db:
        # Get data stats
        user, stats = await get_data_stats(db)

        if not user:
            print("\n❌ No user found in database. Please sync data first.")
            return

        print(f"\nUser: {user.email}")
        print(f"User ID: {user.id}")
        print("\nSynced Data:")
        print(f"  Gmail: {stats['gmail']} emails")
        print(f"  Calendar: {stats['gcal']} events")
        print(f"  Drive: {stats['gdrive']} files")

        if sum(stats.values()) == 0:
            print("\n❌ No synced data found. Please sync data first.")
            return

        # Initialize services and agents
        print("\nInitializing services...")
        embedding_service = EmbeddingService()
        llm = get_llm()

        gmail_service = GmailService(db, None)
        calendar_service = CalendarService(db, None)
        drive_service = DriveService(db, None)

        gmail_agent = GmailAgent(gmail_service, embedding_service, llm)
        gcal_agent = GcalAgent(calendar_service, embedding_service)
        gdrive_agent = GdriveAgent(drive_service, embedding_service)

        agents = {
            "gmail": gmail_agent,
            "gcal": gcal_agent,
            "gdrive": gdrive_agent,
        }

        user_id = str(user.id)

        # Run individual queries with detailed output
        await run_individual_queries(agents, user_id, embedding_service)

        # Run full benchmark
        print("\n" + "=" * 70)
        print("OVERALL BENCHMARK RESULTS")
        print("=" * 70)

        results = await run_search_benchmark(agents, user_id, embedding_service)

        print(f"\nOverall Precision@5: {results['overall_precision_at_5']}")
        print(f"Target: {results['target_precision']}")
        print(f"Meets Target: {'✓ YES' if results['meets_target'] else '✗ NO'}")
        print(f"\nQueries Evaluated: {results['queries_evaluated']}/{results['total_queries']}")
        print(f"Queries Skipped: {results['queries_skipped']} (no results)")

        print("\nPer-Service Results:")
        for service, data in results['per_service'].items():
            p5 = data['precision_at_5']
            p5_str = f"{p5:.3f}" if p5 is not None else "N/A"
            print(f"  {service}: P@5 = {p5_str} ({data['queries_evaluated']}/{data['total_queries']} queries)")


if __name__ == "__main__":
    asyncio.run(main())
