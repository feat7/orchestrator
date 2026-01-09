"""Seed database with mock emails, events, and files for testing.

Run with: python -m scripts.seed_mock_data
"""

import asyncio
from datetime import datetime, timedelta
from uuid import UUID
import random

from app.db.database import async_session
from app.db.models import User, GmailCache, GcalCache, GdriveCache, SyncStatus
from app.services.embedding import EmbeddingService
from app.core.llm import get_llm


# Demo user ID (matches the one in routes.py)
DEMO_USER_ID = UUID("00000000-0000-0000-0000-000000000001")


# Mock email data
MOCK_EMAILS = [
    {
        "email_id": "email_tk1234",
        "thread_id": "thread_tk1234",
        "subject": "Turkish Airlines Flight Confirmation - TK1234",
        "sender": "reservations@turkishairlines.com",
        "recipients": ["user@example.com"],
        "body_preview": "Your flight TK1234 Istanbul to NYC on Nov 5, 2024 at 10:30 AM is confirmed. Booking reference: ABC123. Please arrive at the airport 3 hours before departure.",
        "body_full": "Dear Passenger,\n\nYour flight TK1234 Istanbul to NYC on Nov 5, 2024 at 10:30 AM is confirmed.\n\nBooking reference: ABC123\nPassenger: John Doe\nRoute: Istanbul (IST) -> New York (JFK)\nDeparture: Nov 5, 2024 at 10:30 AM\nArrival: Nov 5, 2024 at 2:45 PM (local time)\n\nPlease arrive at the airport 3 hours before departure.\n\nFor cancellations or changes, contact support@turkishairlines.com\n\nBest regards,\nTurkish Airlines",
        "labels": ["INBOX", "CATEGORY_UPDATES"],
        "received_at": datetime.now() - timedelta(days=20),
    },
    {
        "email_id": "email_acme_q4",
        "thread_id": "thread_acme_q4",
        "subject": "Acme Corp Q4 Budget Review",
        "sender": "sarah@acme-corp.com",
        "recipients": ["user@example.com", "finance@acme-corp.com"],
        "body_preview": "Hi team, please review the attached Q4 budget proposal before our meeting tomorrow. Key points: Marketing budget increased by 15%, R&D allocation remains stable.",
        "body_full": "Hi team,\n\nPlease review the attached Q4 budget proposal before our meeting tomorrow.\n\nKey points:\n- Marketing budget increased by 15%\n- R&D allocation remains stable\n- Sales projections updated based on Q3 performance\n- New headcount requests included\n\nLet me know if you have any questions.\n\nBest,\nSarah",
        "labels": ["INBOX", "IMPORTANT"],
        "received_at": datetime.now() - timedelta(days=2),
    },
    {
        "email_id": "email_project_update",
        "thread_id": "thread_project",
        "subject": "Project Alpha - Weekly Update",
        "sender": "john@company.com",
        "recipients": ["user@example.com", "team@company.com"],
        "body_preview": "Weekly update on Project Alpha. Sprint velocity improved by 20%. Blockers: awaiting API documentation from vendor.",
        "body_full": "Team,\n\nWeekly update on Project Alpha:\n\nProgress:\n- Sprint velocity improved by 20%\n- Frontend milestone completed\n- Backend API 80% complete\n\nBlockers:\n- Awaiting API documentation from vendor\n- Need design approval for dashboard\n\nNext week:\n- Integration testing\n- User acceptance planning\n\nLet me know if you have questions.\n\nJohn",
        "labels": ["INBOX"],
        "received_at": datetime.now() - timedelta(days=1),
    },
    {
        "email_id": "email_invoice",
        "thread_id": "thread_invoice",
        "subject": "Invoice #12345 - Cloud Services",
        "sender": "billing@cloudprovider.com",
        "recipients": ["user@example.com"],
        "body_preview": "Your monthly invoice for cloud services is ready. Total amount: $1,234.56. Due date: Dec 15, 2024.",
        "body_full": "Dear Customer,\n\nYour monthly invoice for cloud services is ready.\n\nInvoice #: 12345\nPeriod: Nov 1-30, 2024\nTotal: $1,234.56\nDue: Dec 15, 2024\n\nServices:\n- Compute: $800.00\n- Storage: $300.00\n- Network: $134.56\n\nPay at: billing.cloudprovider.com\n\nThank you,\nCloud Provider",
        "labels": ["INBOX", "CATEGORY_UPDATES"],
        "received_at": datetime.now() - timedelta(days=5),
    },
    {
        "email_id": "email_meeting_notes",
        "thread_id": "thread_meeting",
        "subject": "Meeting Notes - Q4 Planning",
        "sender": "assistant@company.com",
        "recipients": ["user@example.com", "leadership@company.com"],
        "body_preview": "Meeting notes from today's Q4 planning session. Key decisions: Launch date confirmed for Jan 15, Marketing campaign starts Dec 1.",
        "body_full": "Meeting Notes - Q4 Planning\nDate: Today\nAttendees: Leadership Team\n\nKey Decisions:\n1. Launch date confirmed for Jan 15\n2. Marketing campaign starts Dec 1\n3. Hiring freeze lifted for engineering\n4. Budget approved for new tooling\n\nAction Items:\n- @John: Finalize timeline\n- @Sarah: Prepare press release\n- @Mike: Order equipment\n\nNext meeting: Next Monday",
        "labels": ["INBOX", "STARRED"],
        "received_at": datetime.now() - timedelta(hours=6),
    },
]


# Mock calendar events
MOCK_EVENTS = [
    {
        "event_id": "event_flight",
        "calendar_id": "primary",
        "title": "Istanbul - NYC Flight TK1234",
        "description": "Turkish Airlines flight TK1234. Booking ref: ABC123. Depart IST 10:30 AM, Arrive JFK 2:45 PM.",
        "start_time": datetime.now() + timedelta(days=30),
        "end_time": datetime.now() + timedelta(days=30, hours=12),
        "attendees": [],
        "location": "Istanbul Airport (IST)",
        "meeting_link": None,
        "status": "confirmed",
    },
    {
        "event_id": "event_acme_review",
        "calendar_id": "primary",
        "title": "Acme Corp Quarterly Review",
        "description": "Q4 budget review with Acme Corp team. Review proposal, discuss changes, finalize agreement.",
        "start_time": datetime.now() + timedelta(days=1, hours=2),
        "end_time": datetime.now() + timedelta(days=1, hours=3),
        "attendees": ["sarah@acme-corp.com", "john@company.com", "user@example.com"],
        "location": "Conference Room A",
        "meeting_link": "https://meet.google.com/abc-defg-hij",
        "status": "confirmed",
    },
    {
        "event_id": "event_standup",
        "calendar_id": "primary",
        "title": "Daily Standup",
        "description": "Daily team sync. Updates, blockers, plans.",
        "start_time": datetime.now() + timedelta(days=1, hours=-15),  # Tomorrow 9 AM
        "end_time": datetime.now() + timedelta(days=1, hours=-14, minutes=30),
        "attendees": ["team@company.com"],
        "location": None,
        "meeting_link": "https://meet.google.com/standup-123",
        "status": "confirmed",
    },
    {
        "event_id": "event_1on1",
        "calendar_id": "primary",
        "title": "1:1 with Sarah",
        "description": "Weekly sync with Sarah. Career discussion, project updates.",
        "start_time": datetime.now() + timedelta(days=3, hours=2),
        "end_time": datetime.now() + timedelta(days=3, hours=2, minutes=30),
        "attendees": ["sarah@company.com"],
        "location": None,
        "meeting_link": "https://meet.google.com/1on1-sarah",
        "status": "confirmed",
    },
    {
        "event_id": "event_lunch",
        "calendar_id": "primary",
        "title": "Team Lunch",
        "description": "Monthly team lunch at Italian place downtown.",
        "start_time": datetime.now() + timedelta(days=5, hours=4),
        "end_time": datetime.now() + timedelta(days=5, hours=5, minutes=30),
        "attendees": ["team@company.com"],
        "location": "Pasta Paradise, 123 Main St",
        "meeting_link": None,
        "status": "confirmed",
    },
    {
        "event_id": "event_client_call",
        "calendar_id": "primary",
        "title": "Client Call - Project Alpha Demo",
        "description": "Demo of Project Alpha progress to client. Show new features, get feedback.",
        "start_time": datetime.now() + timedelta(days=2, hours=3),
        "end_time": datetime.now() + timedelta(days=2, hours=4),
        "attendees": ["client@bigcorp.com", "john@company.com"],
        "location": None,
        "meeting_link": "https://zoom.us/j/123456789",
        "status": "confirmed",
    },
]


# Mock Drive files
MOCK_FILES = [
    {
        "file_id": "file_q4_budget",
        "name": "Q4_Budget_Proposal.pdf",
        "mime_type": "application/pdf",
        "content_preview": "Q4 Budget Proposal for Acme Corp Partnership\n\nExecutive Summary:\nThis proposal outlines the budget allocation for Q4 2024, with focus on marketing expansion and R&D investments.\n\nKey Allocations:\n- Marketing: $500,000 (+15%)\n- R&D: $800,000 (stable)\n- Operations: $300,000\n- Contingency: $100,000",
        "parent_folder": "Acme Corp",
        "web_link": "https://drive.google.com/file/d/q4budget",
        "owners": ["user@example.com"],
        "shared_with": ["sarah@acme-corp.com", "finance@company.com"],
        "modified_at": datetime.now() - timedelta(days=3),
    },
    {
        "file_id": "file_project_alpha",
        "name": "Project_Alpha_Specs.docx",
        "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "content_preview": "Project Alpha Technical Specifications\n\nOverview:\nProject Alpha is a next-generation platform for data analytics and visualization.\n\nArchitecture:\n- Microservices-based\n- Cloud-native deployment\n- Real-time data processing\n\nTimeline:\n- Phase 1: Q4 2024\n- Phase 2: Q1 2025",
        "parent_folder": "Projects",
        "web_link": "https://drive.google.com/file/d/projectalpha",
        "owners": ["user@example.com"],
        "shared_with": ["john@company.com", "team@company.com"],
        "modified_at": datetime.now() - timedelta(days=7),
    },
    {
        "file_id": "file_ooo",
        "name": "Out_of_Office_Schedule.xlsx",
        "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "content_preview": "Out of Office Schedule 2024\n\nDate | Person | Reason\nNov 5-10 | User | Travel to NYC\nNov 15 | Sarah | Personal\nDec 20-Jan 2 | All | Holiday Break",
        "parent_folder": "HR",
        "web_link": "https://drive.google.com/file/d/ooo",
        "owners": ["hr@company.com"],
        "shared_with": ["user@example.com", "team@company.com"],
        "modified_at": datetime.now() - timedelta(days=14),
    },
    {
        "file_id": "file_meeting_deck",
        "name": "Acme_Corp_Meeting_Deck.pptx",
        "mime_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "content_preview": "Acme Corp Partnership Meeting\n\nAgenda:\n1. Q3 Review\n2. Q4 Proposal\n3. Next Steps\n\nQ3 Highlights:\n- Revenue up 25%\n- Customer satisfaction 4.5/5\n- New features launched",
        "parent_folder": "Acme Corp",
        "web_link": "https://drive.google.com/file/d/acmedeck",
        "owners": ["user@example.com"],
        "shared_with": ["sarah@acme-corp.com"],
        "modified_at": datetime.now() - timedelta(days=1),
    },
    {
        "file_id": "file_contract",
        "name": "Service_Agreement_2024.pdf",
        "mime_type": "application/pdf",
        "content_preview": "Service Agreement\n\nThis agreement is between Company Inc and Acme Corp for consulting services.\n\nTerm: Jan 1, 2024 - Dec 31, 2024\nValue: $1,200,000\n\nServices:\n- Technical consulting\n- Implementation support\n- Training",
        "parent_folder": "Contracts",
        "web_link": "https://drive.google.com/file/d/contract",
        "owners": ["legal@company.com"],
        "shared_with": ["user@example.com", "sarah@acme-corp.com"],
        "modified_at": datetime.now() - timedelta(days=60),
    },
]


async def seed_data():
    """Seed the database with mock data."""
    print("Starting data seeding...")

    # Initialize embedding service
    llm = get_llm()
    embedding_service = EmbeddingService(llm=llm)

    async with async_session() as db:
        # Check if demo user exists
        from sqlalchemy import select

        result = await db.execute(select(User).where(User.id == DEMO_USER_ID))
        user = result.scalar_one_or_none()

        if not user:
            # Create demo user
            user = User(
                id=DEMO_USER_ID,
                email="demo@example.com",
            )
            db.add(user)
            print("Created demo user")

        # Seed emails
        print("Seeding emails...")
        for email_data in MOCK_EMAILS:
            # Check if already exists
            result = await db.execute(
                select(GmailCache).where(
                    GmailCache.user_id == DEMO_USER_ID,
                    GmailCache.email_id == email_data["email_id"],
                )
            )
            if result.scalar_one_or_none():
                print(f"  Email {email_data['email_id']} already exists, skipping")
                continue

            # Generate embedding
            text = f"{email_data['subject']} {email_data['body_preview']}"
            embedding = await embedding_service.embed_for_storage(text)

            email = GmailCache(
                user_id=DEMO_USER_ID,
                email_id=email_data["email_id"],
                thread_id=email_data["thread_id"],
                subject=email_data["subject"],
                sender=email_data["sender"],
                recipients=email_data["recipients"],
                body_preview=email_data["body_preview"],
                body_full=email_data["body_full"],
                embedding=embedding,
                received_at=email_data["received_at"],
                labels=email_data["labels"],
            )
            db.add(email)
            print(f"  Added email: {email_data['subject'][:50]}...")

        # Seed calendar events
        print("Seeding calendar events...")
        for event_data in MOCK_EVENTS:
            result = await db.execute(
                select(GcalCache).where(
                    GcalCache.user_id == DEMO_USER_ID,
                    GcalCache.event_id == event_data["event_id"],
                )
            )
            if result.scalar_one_or_none():
                print(f"  Event {event_data['event_id']} already exists, skipping")
                continue

            text = f"{event_data['title']} {event_data['description']}"
            embedding = await embedding_service.embed_for_storage(text)

            event = GcalCache(
                user_id=DEMO_USER_ID,
                event_id=event_data["event_id"],
                calendar_id=event_data["calendar_id"],
                title=event_data["title"],
                description=event_data["description"],
                start_time=event_data["start_time"],
                end_time=event_data["end_time"],
                attendees=event_data["attendees"],
                location=event_data["location"],
                meeting_link=event_data["meeting_link"],
                status=event_data["status"],
                embedding=embedding,
            )
            db.add(event)
            print(f"  Added event: {event_data['title'][:50]}...")

        # Seed Drive files
        print("Seeding Drive files...")
        for file_data in MOCK_FILES:
            result = await db.execute(
                select(GdriveCache).where(
                    GdriveCache.user_id == DEMO_USER_ID,
                    GdriveCache.file_id == file_data["file_id"],
                )
            )
            if result.scalar_one_or_none():
                print(f"  File {file_data['file_id']} already exists, skipping")
                continue

            text = f"{file_data['name']} {file_data['content_preview']}"
            embedding = await embedding_service.embed_for_storage(text)

            file = GdriveCache(
                user_id=DEMO_USER_ID,
                file_id=file_data["file_id"],
                name=file_data["name"],
                mime_type=file_data["mime_type"],
                content_preview=file_data["content_preview"],
                parent_folder=file_data["parent_folder"],
                web_link=file_data["web_link"],
                owners=file_data["owners"],
                shared_with=file_data["shared_with"],
                embedding=embedding,
                modified_at=file_data["modified_at"],
            )
            db.add(file)
            print(f"  Added file: {file_data['name']}")

        # Create sync status records
        print("Creating sync status records...")
        for service in ["gmail", "gcal", "gdrive"]:
            result = await db.execute(
                select(SyncStatus).where(
                    SyncStatus.user_id == DEMO_USER_ID,
                    SyncStatus.service == service,
                )
            )
            if not result.scalar_one_or_none():
                status = SyncStatus(
                    user_id=DEMO_USER_ID,
                    service=service,
                    last_sync_at=datetime.now(),
                    status="completed",
                )
                db.add(status)

        await db.commit()
        print("\nData seeding completed successfully!")
        print(f"  - {len(MOCK_EMAILS)} emails")
        print(f"  - {len(MOCK_EVENTS)} events")
        print(f"  - {len(MOCK_FILES)} files")


if __name__ == "__main__":
    asyncio.run(seed_data())
