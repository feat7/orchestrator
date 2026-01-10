#!/usr/bin/env python3
"""Explore synced data to understand what's available."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select, func
from app.db.database import async_session
from app.db.models import User, GmailCache, GcalCache, GdriveCache


async def main():
    async with async_session() as db:
        # Get user
        result = await db.execute(select(User).limit(1))
        user = result.scalar_one_or_none()

        if not user:
            print("No user found")
            return

        print(f"User: {user.email}\n")

        # Gmail - show unique senders and sample subjects
        print("=" * 60)
        print("GMAIL DATA")
        print("=" * 60)

        emails = await db.execute(
            select(GmailCache)
            .where(GmailCache.user_id == user.id)
            .order_by(GmailCache.received_at.desc())
            .limit(30)
        )
        emails = emails.scalars().all()

        print(f"\nTotal emails: {len(emails)} (showing first 30)")
        print("\nSample emails:")
        for i, email in enumerate(emails[:15], 1):
            subject = (email.subject or "No subject")[:50]
            sender = (email.sender or "unknown")[:30]
            print(f"  {i}. {subject}")
            print(f"      From: {sender}")

        # Unique senders
        senders = await db.execute(
            select(GmailCache.sender, func.count(GmailCache.id).label("count"))
            .where(GmailCache.user_id == user.id)
            .group_by(GmailCache.sender)
            .order_by(func.count(GmailCache.id).desc())
            .limit(10)
        )
        print("\n\nTop senders:")
        for sender, count in senders:
            print(f"  - {sender}: {count} emails")

        # Calendar
        print("\n" + "=" * 60)
        print("CALENDAR DATA")
        print("=" * 60)

        events = await db.execute(
            select(GcalCache)
            .where(GcalCache.user_id == user.id)
            .limit(10)
        )
        events = events.scalars().all()
        print(f"\nTotal events: {len(events)}")
        for event in events[:10]:
            print(f"  - {event.title}")

        # Drive
        print("\n" + "=" * 60)
        print("DRIVE DATA")
        print("=" * 60)

        files = await db.execute(
            select(GdriveCache)
            .where(GdriveCache.user_id == user.id)
            .limit(10)
        )
        files = files.scalars().all()
        print(f"\nTotal files: {len(files)}")
        for f in files:
            print(f"  - {f.name} ({f.mime_type})")


if __name__ == "__main__":
    asyncio.run(main())
