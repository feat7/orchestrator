"""Seed database with mock emails, events, and files for testing.

Run with: python -m scripts.seed_mock_data

This script creates a demo user and populates the database with realistic
mock data for Gmail, Google Calendar, and Google Drive. The data includes
proper embeddings and TSVECTOR search_vector fields for hybrid search.

All dates are relative to the current date for realistic testing.
"""

import asyncio
from datetime import datetime, timedelta
from uuid import UUID

from app.db.database import async_session
from app.db.models import User, GmailCache, GcalCache, GdriveCache, SyncStatus
from app.services.embedding import EmbeddingService
from app.core.llm import get_llm


# Demo user ID (matches the one in routes.py)
DEMO_USER_ID = UUID("00000000-0000-0000-0000-000000000001")


# Helper functions for relative dates
def today_at(hour: int, minute: int = 0) -> datetime:
    """Get datetime for today at specified hour."""
    now = datetime.now()
    return now.replace(hour=hour, minute=minute, second=0, microsecond=0)


def yesterday_at(hour: int, minute: int = 0) -> datetime:
    """Get datetime for yesterday at specified hour."""
    return today_at(hour, minute) - timedelta(days=1)


def days_ago_at(days: int, hour: int, minute: int = 0) -> datetime:
    """Get datetime for N days ago at specified hour."""
    return today_at(hour, minute) - timedelta(days=days)


def days_from_now_at(days: int, hour: int, minute: int = 0) -> datetime:
    """Get datetime for N days from now at specified hour."""
    return today_at(hour, minute) + timedelta(days=days)


# Mock email data with relative dates
# Grouped by: TODAY, YESTERDAY, THIS WEEK (2-4 days), LAST WEEK (5-10 days), OLDER (2+ weeks)
MOCK_EMAILS = [
    # ===== TODAY =====
    {
        "email_id": "email_urgent_review",
        "thread_id": "thread_urgent",
        "subject": "URGENT: Contract needs your signature today",
        "sender": "legal@company.com",
        "recipients": ["user@example.com"],
        "body_preview": "Hi, the TechCorp contract needs to be signed by end of day. Please review and sign via DocuSign link.",
        "body_full": "Hi,\n\nThe TechCorp enterprise contract needs your signature today to meet their internal deadline.\n\nPlease review and sign: https://docusign.com/sign/techcorp\n\nKey points:\n- 3-year term\n- $500K ARR\n- Standard enterprise terms\n\nLet me know if you have any questions.\n\nThanks,\nLegal Team",
        "labels": ["INBOX", "IMPORTANT", "STARRED"],
        "received_at": today_at(8, 30),
    },
    {
        "email_id": "email_standup_notes",
        "thread_id": "thread_standup",
        "subject": "Daily Standup Notes - Today",
        "sender": "team-bot@company.com",
        "recipients": ["user@example.com", "team@company.com"],
        "body_preview": "Today's standup summary: 3 tasks completed, 2 in progress, 1 blocker identified.",
        "body_full": "Daily Standup Summary\n\nCompleted:\n- API endpoint for user preferences\n- Bug fix for login timeout\n- Documentation update\n\nIn Progress:\n- Dashboard redesign\n- Performance optimization\n\nBlockers:\n- Waiting for design approval on new icons\n\nNext standup: Tomorrow 9:00 AM",
        "labels": ["INBOX"],
        "received_at": today_at(9, 15),
    },
    {
        "email_id": "email_lunch_today",
        "thread_id": "thread_lunch",
        "subject": "Team lunch today at noon?",
        "sender": "sarah@company.com",
        "recipients": ["user@example.com", "team@company.com"],
        "body_preview": "Hey everyone! Want to grab lunch at the new Italian place today? Meet in lobby at 12?",
        "body_full": "Hey everyone!\n\nWant to grab lunch at the new Italian place down the street?\n\nMeet in lobby at 12:00 PM\nPlace: Pasta Paradise\nAddress: 123 Main St\n\nLet me know if you can make it!\n\n- Sarah",
        "labels": ["INBOX"],
        "received_at": today_at(10, 45),
    },
    # ===== YESTERDAY =====
    {
        "email_id": "email_project_update",
        "thread_id": "thread_project",
        "subject": "Project Alpha - Weekly Update",
        "sender": "john@company.com",
        "recipients": ["user@example.com", "team@company.com"],
        "body_preview": "Weekly update on Project Alpha. Sprint velocity improved by 20%. Blockers: awaiting API documentation from vendor.",
        "body_full": "Team,\n\nWeekly update on Project Alpha:\n\nProgress:\n- Sprint velocity improved by 20%\n- Frontend milestone completed\n- Backend API 80% complete\n\nBlockers:\n- Awaiting API documentation from vendor\n- Need design approval for dashboard\n\nNext week:\n- Integration testing\n- User acceptance planning\n\nLet me know if you have questions.\n\nJohn",
        "labels": ["INBOX"],
        "received_at": yesterday_at(16, 30),
    },
    {
        "email_id": "email_customer_feedback",
        "thread_id": "thread_feedback",
        "subject": "Customer Feedback: Enterprise Deal Closed!",
        "sender": "sales@company.com",
        "recipients": ["user@example.com", "team@company.com"],
        "body_preview": "Great news! TechCorp just signed a 3-year enterprise deal worth $500K ARR. They mentioned our API reliability as the deciding factor.",
        "body_full": "Team,\n\nExciting news to share!\n\nTechCorp has signed!\n\nDeal Details:\n- Contract: 3 years\n- ARR: $500,000\n- Seats: 500 users\n- Start Date: Next month\n\nWhy they chose us:\n1. API reliability (99.99% uptime)\n2. Enterprise security features\n3. Dedicated support\n4. Competitive pricing\n\nThis is our largest enterprise deal yet!\n\nSales Team",
        "labels": ["INBOX", "STARRED"],
        "received_at": yesterday_at(14, 0),
    },
    {
        "email_id": "email_deployment_complete",
        "thread_id": "thread_deploy",
        "subject": "Deployment Complete: v2.5.0 to Production",
        "sender": "devops@company.com",
        "recipients": ["user@example.com", "engineering@company.com"],
        "body_preview": "Version 2.5.0 has been successfully deployed to production. All health checks passing.",
        "body_full": "Deployment Summary\n\nVersion: 2.5.0\nEnvironment: Production\nStatus: Success\nTime: Yesterday 2:00 AM\n\nChanges included:\n- New payment gateway integration\n- Performance optimizations (30% faster API)\n- Bug fixes (12 issues resolved)\n- Security patches\n\nHealth Checks: All Passing\n\nDevOps Team",
        "labels": ["INBOX"],
        "received_at": yesterday_at(2, 30),
    },
    # ===== THIS WEEK (2-4 days ago) =====
    {
        "email_id": "email_acme_q4",
        "thread_id": "thread_acme_q4",
        "subject": "Acme Corp Q4 Budget Review",
        "sender": "sarah@acme-corp.com",
        "recipients": ["user@example.com", "finance@acme-corp.com"],
        "body_preview": "Hi team, please review the attached Q4 budget proposal. Key points: Marketing budget increased by 15%.",
        "body_full": "Hi team,\n\nPlease review the attached Q4 budget proposal.\n\nKey points:\n- Marketing budget increased by 15%\n- R&D allocation remains stable\n- Sales projections updated based on Q3 performance\n- New headcount requests included\n\nLet me know if you have any questions.\n\nBest,\nSarah",
        "labels": ["INBOX", "IMPORTANT"],
        "received_at": days_ago_at(2, 10, 30),
    },
    {
        "email_id": "email_sarah_budget",
        "thread_id": "thread_sarah_budget",
        "subject": "Re: Q4 Budget Discussion",
        "sender": "sarah@company.com",
        "recipients": ["user@example.com"],
        "body_preview": "Hi, following up on the budget discussion. I've reviewed the numbers and have some suggestions for the marketing allocation.",
        "body_full": "Hi,\n\nFollowing up on the budget discussion from our meeting.\n\nMy recommendations:\n1. Increase digital marketing by 20%\n2. Reduce print advertising by 30%\n3. Add $50K for influencer partnerships\n\nI've attached a detailed breakdown. Let me know if you want to discuss.\n\nThanks,\nSarah",
        "labels": ["INBOX"],
        "received_at": days_ago_at(2, 14, 30),
    },
    {
        "email_id": "email_acme_meeting_prep",
        "thread_id": "thread_acme_prep",
        "subject": "Prep for tomorrow's Acme Corp meeting",
        "sender": "pm@company.com",
        "recipients": ["user@example.com", "sales@company.com"],
        "body_preview": "Quick reminder about tomorrow's meeting with Acme Corp. Please review the attached deck and bring the Q4 projections.",
        "body_full": "Team,\n\nReminder about tomorrow's Acme Corp meeting:\n\nTime: 10:00 AM\nLocation: Conference Room A / Zoom\n\nAgenda:\n1. Q3 review and performance metrics\n2. Q4 partnership expansion proposal\n3. Contract renewal discussion\n\nPlease bring:\n- Q4 projections document\n- Updated pricing proposal\n- Customer success metrics\n\nLet me know if you have questions.\n\nThanks,\nProject Manager",
        "labels": ["INBOX", "IMPORTANT"],
        "received_at": today_at(7, 0),
    },
    {
        "email_id": "email_beta_launch",
        "thread_id": "thread_beta",
        "subject": "Project Beta - Launch Checklist",
        "sender": "pm@company.com",
        "recipients": ["user@example.com", "engineering@company.com"],
        "body_preview": "Project Beta launch is scheduled for next Friday. Please review the checklist.",
        "body_full": "Hi everyone,\n\nProject Beta launches next Friday! Here's the final checklist:\n\n- Code freeze: Complete\n- Security audit: Passed\n- Load testing: In progress\n- Documentation: 90% complete\n\nPlease confirm your items by Wednesday EOD.\n\nThanks,\nProject Manager",
        "labels": ["INBOX", "IMPORTANT"],
        "received_at": days_ago_at(3, 14, 0),
    },
    {
        "email_id": "email_invoice",
        "thread_id": "thread_invoice",
        "subject": "Invoice #12345 - Cloud Services",
        "sender": "billing@cloudprovider.com",
        "recipients": ["user@example.com"],
        "body_preview": "Your monthly invoice for cloud services is ready. Total amount: $1,234.56.",
        "body_full": "Dear Customer,\n\nYour monthly invoice for cloud services is ready.\n\nInvoice #: 12345\nPeriod: Last month\nTotal: $1,234.56\nDue: In 15 days\n\nServices:\n- Compute: $800.00\n- Storage: $300.00\n- Network: $134.56\n\nThank you,\nCloud Provider",
        "labels": ["INBOX", "CATEGORY_UPDATES"],
        "received_at": days_ago_at(3, 8, 0),
    },
    {
        "email_id": "email_saas_renewal",
        "thread_id": "thread_saas",
        "subject": "Your Subscription Renewal - Action Required",
        "sender": "billing@saasplatform.io",
        "recipients": ["user@example.com"],
        "body_preview": "Your annual subscription expires soon. Renew now to keep your Pro features.",
        "body_full": "Hi there,\n\nYour SaaS Platform Pro subscription expires at the end of this month.\n\nCurrent Plan: Pro ($99/month)\nRenewal Price: $79/month (20% early renewal discount)\n\nRenew within 15 days to lock in the discount.\n\nBest,\nSaaS Platform Team",
        "labels": ["INBOX", "CATEGORY_PROMOTIONS"],
        "received_at": days_ago_at(4, 9, 0),
    },
    # ===== LAST WEEK (5-10 days ago) =====
    {
        "email_id": "email_board_summary",
        "thread_id": "thread_board",
        "subject": "Board Meeting Summary",
        "sender": "ceo@company.com",
        "recipients": ["user@example.com", "executives@company.com"],
        "body_preview": "Summary of key decisions from last week's board meeting. Series B discussions progressing.",
        "body_full": "Executive Team,\n\nSummary of the Board Meeting:\n\n1. Funding:\n- Series B discussions with 3 VCs progressing well\n- Target: $50M at $300M valuation\n\n2. International Expansion:\n- EMEA expansion approved\n- London office opening next quarter\n\n3. Team:\n- VP of Sales search to begin\n- Engineering hiring plan approved\n\nBest,\nCEO",
        "labels": ["INBOX", "IMPORTANT", "STARRED"],
        "received_at": days_ago_at(7, 11, 0),
    },
    {
        "email_id": "email_quarterly_earnings",
        "thread_id": "thread_earnings",
        "subject": "Q3 Earnings Report - Company Performance Update",
        "sender": "cfo@company.com",
        "recipients": ["user@example.com", "all-hands@company.com"],
        "body_preview": "Q3 financial results are in. Revenue up 23% YoY, beating analyst expectations.",
        "body_full": "Team,\n\nI'm pleased to share our Q3 results:\n\nKey Metrics:\n- Revenue: $45.2M (+23% YoY)\n- Gross Margin: 68%\n- Customer Growth: 15% QoQ\n- Net Promoter Score: 72\n\nHighlights:\n- Enterprise segment grew 40%\n- New product launch exceeded targets\n\nBest,\nMichael Chen\nCFO",
        "labels": ["INBOX", "IMPORTANT", "STARRED"],
        "received_at": days_ago_at(8, 15, 0),
    },
    {
        "email_id": "email_welcome_new_hire",
        "thread_id": "thread_onboarding",
        "subject": "Welcome to the Team! - Onboarding Schedule",
        "sender": "hr@company.com",
        "recipients": ["user@example.com", "newemployee@company.com"],
        "body_preview": "Welcome aboard! Here's your first week schedule.",
        "body_full": "Hi and Welcome!\n\nWe're excited to have you join us. Here's your first week:\n\nDay 1:\n- 9:00 AM: IT setup\n- 11:00 AM: HR orientation\n- 2:00 PM: Meet your manager\n\nDay 2:\n- 10:00 AM: Team meet and greet\n- 2:00 PM: Product overview\n\nLooking forward to working with you!\n\nHR Team",
        "labels": ["INBOX"],
        "received_at": days_ago_at(6, 9, 0),
    },
    {
        "email_id": "email_security_alert",
        "thread_id": "thread_security",
        "subject": "Security Alert: Unusual Login Attempt",
        "sender": "security@company.com",
        "recipients": ["user@example.com"],
        "body_preview": "We detected a login attempt from an unusual location. If this was you, please ignore.",
        "body_full": "Security Alert\n\nWe detected a login attempt to your account:\n\nLocation: Berlin, Germany\nTime: Last week at 3:42 AM\nDevice: Chrome on Windows\n\nIf this wasn't you:\n1. Reset your password immediately\n2. Enable 2FA if not already active\n\nStay safe,\nSecurity Team",
        "labels": ["INBOX", "IMPORTANT"],
        "received_at": days_ago_at(5, 3, 42),
    },
    # ===== OLDER (2-4 weeks ago) =====
    {
        "email_id": "email_tk1234",
        "thread_id": "thread_tk1234",
        "subject": "Turkish Airlines Flight Confirmation - TK1234",
        "sender": "reservations@turkishairlines.com",
        "recipients": ["user@example.com"],
        "body_preview": "Your flight TK1234 Istanbul to NYC is confirmed. Booking reference: ABC123.",
        "body_full": "Dear Passenger,\n\nYour flight TK1234 Istanbul to NYC is confirmed.\n\nBooking reference: ABC123\nRoute: Istanbul (IST) -> New York (JFK)\nDeparture: In 2 weeks at 10:30 AM\n\nPlease arrive at the airport 3 hours before departure.\n\nBest regards,\nTurkish Airlines",
        "labels": ["INBOX", "CATEGORY_UPDATES"],
        "received_at": days_ago_at(20, 12, 0),
    },
    {
        "email_id": "email_hotel_hilton",
        "thread_id": "thread_hotel_hilton",
        "subject": "Hilton NYC - Reservation Confirmed",
        "sender": "reservations@hilton.com",
        "recipients": ["user@example.com"],
        "body_preview": "Your reservation at Hilton Midtown NYC is confirmed. Confirmation number: HLT789456.",
        "body_full": "Dear Guest,\n\nThank you for choosing Hilton Midtown NYC.\n\nReservation Details:\n- Confirmation: HLT789456\n- Check-in: In 2 weeks (3:00 PM)\n- Room Type: King Deluxe\n- Rate: $299/night\n\nWe look forward to welcoming you!\n\nHilton Hotels",
        "labels": ["INBOX", "CATEGORY_UPDATES"],
        "received_at": days_ago_at(18, 10, 0),
    },
    {
        "email_id": "email_pto_approval",
        "thread_id": "thread_pto",
        "subject": "PTO Request Approved",
        "sender": "hr@company.com",
        "recipients": ["user@example.com"],
        "body_preview": "Your PTO request has been approved. Remaining balance: 5 days.",
        "body_full": "Hi,\n\nYour PTO request has been approved!\n\nDetails:\n- Dates: Holiday break next month\n- Days: 5 business days\n- Remaining PTO balance: 5 days\n\nReminders:\n- Set up out-of-office message\n- Delegate urgent tasks\n\nEnjoy your time off!\n\nHR Team",
        "labels": ["INBOX"],
        "received_at": days_ago_at(14, 11, 0),
    },
    {
        "email_id": "email_vendor_contract",
        "thread_id": "thread_vendor",
        "subject": "Contract Review: DataAnalytics Inc Partnership",
        "sender": "legal@company.com",
        "recipients": ["user@example.com", "procurement@company.com"],
        "body_preview": "Please review the attached contract from DataAnalytics Inc.",
        "body_full": "Hi,\n\nPlease review the proposed contract with DataAnalytics Inc:\n\nKey Terms:\n- Duration: 2 years\n- Annual Fee: $50,000\n- SLA: 99.5% uptime guarantee\n\nConcerns flagged:\n1. IP ownership clause needs revision\n2. Liability cap too low\n\nPlease provide feedback by end of week.\n\nBest,\nLegal Team",
        "labels": ["INBOX"],
        "received_at": days_ago_at(12, 14, 0),
    },
]


# Mock calendar events with relative dates
# Mix of: PAST (already happened), TODAY, THIS WEEK, NEXT WEEK, FUTURE
MOCK_EVENTS = [
    # ===== PAST EVENTS (Already happened) =====
    {
        "event_id": "event_past_standup_1",
        "calendar_id": "primary",
        "title": "Daily Standup",
        "description": "Daily team sync. Updates, blockers, plans.",
        "start_time": yesterday_at(9, 0),
        "end_time": yesterday_at(9, 30),
        "attendees": ["team@company.com"],
        "location": None,
        "meeting_link": "https://meet.google.com/standup-123",
        "status": "confirmed",
    },
    {
        "event_id": "event_past_review",
        "calendar_id": "primary",
        "title": "Code Review Session",
        "description": "Review PR #456 for Project Alpha.",
        "start_time": yesterday_at(14, 0),
        "end_time": yesterday_at(15, 0),
        "attendees": ["john@company.com", "sarah@company.com"],
        "location": None,
        "meeting_link": "https://meet.google.com/code-review",
        "status": "confirmed",
    },
    {
        "event_id": "event_past_client",
        "calendar_id": "primary",
        "title": "Client Call - Acme Corp",
        "description": "Quarterly review with Acme Corp. Discuss Q4 budget.",
        "start_time": days_ago_at(2, 10, 0),
        "end_time": days_ago_at(2, 11, 0),
        "attendees": ["sarah@acme-corp.com", "user@example.com"],
        "location": None,
        "meeting_link": "https://zoom.us/j/acme-review",
        "status": "confirmed",
    },
    {
        "event_id": "event_past_sprint",
        "calendar_id": "primary",
        "title": "Sprint Planning - Sprint 23",
        "description": "Plan sprint 23. Review backlog, estimate stories.",
        "start_time": days_ago_at(7, 10, 0),
        "end_time": days_ago_at(7, 12, 0),
        "attendees": ["engineering@company.com", "pm@company.com"],
        "location": None,
        "meeting_link": "https://meet.google.com/sprint-planning",
        "status": "confirmed",
    },
    {
        "event_id": "event_past_retro",
        "calendar_id": "primary",
        "title": "Sprint Retrospective - Sprint 22",
        "description": "Sprint 22 retro. What went well, what to improve.",
        "start_time": days_ago_at(8, 15, 0),
        "end_time": days_ago_at(8, 16, 0),
        "attendees": ["engineering@company.com"],
        "location": None,
        "meeting_link": "https://meet.google.com/retro",
        "status": "confirmed",
    },
    # ===== TODAY'S EVENTS =====
    {
        "event_id": "event_today_standup",
        "calendar_id": "primary",
        "title": "Daily Standup",
        "description": "Daily team sync. Updates, blockers, plans.",
        "start_time": today_at(9, 0),
        "end_time": today_at(9, 30),
        "attendees": ["team@company.com"],
        "location": None,
        "meeting_link": "https://meet.google.com/standup-123",
        "status": "confirmed",
    },
    {
        "event_id": "event_today_1on1",
        "calendar_id": "primary",
        "title": "1:1 with Manager",
        "description": "Weekly sync with manager. Career discussion, project updates.",
        "start_time": today_at(11, 0),
        "end_time": today_at(11, 30),
        "attendees": ["manager@company.com"],
        "location": None,
        "meeting_link": "https://meet.google.com/manager-1on1",
        "status": "confirmed",
    },
    {
        "event_id": "event_today_lunch",
        "calendar_id": "primary",
        "title": "Team Lunch",
        "description": "Team lunch at Italian place downtown.",
        "start_time": today_at(12, 0),
        "end_time": today_at(13, 30),
        "attendees": ["team@company.com"],
        "location": "Pasta Paradise, 123 Main St",
        "meeting_link": None,
        "status": "confirmed",
    },
    {
        "event_id": "event_today_design",
        "calendar_id": "primary",
        "title": "Design Review - Dashboard v2",
        "description": "Review new dashboard designs with product team.",
        "start_time": today_at(15, 0),
        "end_time": today_at(16, 0),
        "attendees": ["design@company.com", "product@company.com"],
        "location": None,
        "meeting_link": "https://meet.google.com/design-review",
        "status": "confirmed",
    },
    # ===== TOMORROW'S EVENTS =====
    {
        "event_id": "event_tomorrow_standup",
        "calendar_id": "primary",
        "title": "Daily Standup",
        "description": "Daily team sync. Updates, blockers, plans.",
        "start_time": days_from_now_at(1, 9, 0),
        "end_time": days_from_now_at(1, 9, 30),
        "attendees": ["team@company.com"],
        "location": None,
        "meeting_link": "https://meet.google.com/standup-123",
        "status": "confirmed",
    },
    {
        "event_id": "event_tomorrow_acme",
        "calendar_id": "primary",
        "title": "Acme Corp Partnership Meeting",
        "description": "Quarterly partnership review with Acme Corp. Discuss Q4 budget, contract renewal, and expansion opportunities.",
        "start_time": days_from_now_at(1, 10, 0),
        "end_time": days_from_now_at(1, 11, 30),
        "attendees": ["sarah@acme-corp.com", "pm@company.com", "sales@company.com"],
        "location": "Conference Room A",
        "meeting_link": "https://zoom.us/j/acme-meeting",
        "status": "confirmed",
    },
    {
        "event_id": "event_tomorrow_client",
        "calendar_id": "primary",
        "title": "Client Demo - Project Alpha",
        "description": "Demo of Project Alpha progress to client. Show new features.",
        "start_time": days_from_now_at(1, 14, 0),
        "end_time": days_from_now_at(1, 15, 0),
        "attendees": ["client@bigcorp.com", "john@company.com"],
        "location": None,
        "meeting_link": "https://zoom.us/j/client-demo",
        "status": "confirmed",
    },
    # ===== THIS WEEK (2-5 days from now) =====
    {
        "event_id": "event_week_1on1_sarah",
        "calendar_id": "primary",
        "title": "1:1 with Sarah",
        "description": "Weekly sync with Sarah. Project updates, blockers.",
        "start_time": days_from_now_at(2, 10, 0),
        "end_time": days_from_now_at(2, 10, 30),
        "attendees": ["sarah@company.com"],
        "location": None,
        "meeting_link": "https://meet.google.com/1on1-sarah",
        "status": "confirmed",
    },
    {
        "event_id": "event_week_interview",
        "calendar_id": "primary",
        "title": "Interview: Senior Engineer Candidate",
        "description": "Technical interview with senior engineer candidate.",
        "start_time": days_from_now_at(3, 14, 0),
        "end_time": days_from_now_at(3, 15, 0),
        "attendees": ["hiring@company.com", "candidate@email.com"],
        "location": None,
        "meeting_link": "https://meet.google.com/interview-123",
        "status": "confirmed",
    },
    {
        "event_id": "event_week_doctor",
        "calendar_id": "primary",
        "title": "Doctor Appointment",
        "description": "Annual checkup.",
        "start_time": days_from_now_at(4, 10, 0),
        "end_time": days_from_now_at(4, 11, 0),
        "attendees": [],
        "location": "456 Medical Plaza, Suite 200",
        "meeting_link": None,
        "status": "confirmed",
    },
    {
        "event_id": "event_week_retro",
        "calendar_id": "primary",
        "title": "Sprint Retrospective - Sprint 23",
        "description": "Sprint 23 retro. What went well, what to improve.",
        "start_time": days_from_now_at(5, 15, 0),
        "end_time": days_from_now_at(5, 16, 0),
        "attendees": ["engineering@company.com"],
        "location": None,
        "meeting_link": "https://meet.google.com/retro-sprint23",
        "status": "confirmed",
    },
    # ===== NEXT WEEK (6-12 days from now) =====
    {
        "event_id": "event_next_sprint",
        "calendar_id": "primary",
        "title": "Sprint Planning - Sprint 24",
        "description": "Plan sprint 24. Review backlog, estimate stories.",
        "start_time": days_from_now_at(7, 10, 0),
        "end_time": days_from_now_at(7, 12, 0),
        "attendees": ["engineering@company.com", "pm@company.com"],
        "location": None,
        "meeting_link": "https://meet.google.com/sprint-planning",
        "status": "confirmed",
    },
    {
        "event_id": "event_next_all_hands",
        "calendar_id": "primary",
        "title": "All Hands Meeting",
        "description": "Monthly all-hands. Company updates, Q&A with leadership.",
        "start_time": days_from_now_at(10, 14, 0),
        "end_time": days_from_now_at(10, 15, 0),
        "attendees": ["all@company.com"],
        "location": None,
        "meeting_link": "https://meet.google.com/all-hands",
        "status": "confirmed",
    },
    {
        "event_id": "event_next_techcorp",
        "calendar_id": "primary",
        "title": "TechCorp Enterprise Kickoff",
        "description": "Kickoff meeting for TechCorp enterprise implementation.",
        "start_time": days_from_now_at(12, 10, 0),
        "end_time": days_from_now_at(12, 12, 0),
        "attendees": ["enterprise@techcorp.com", "pm@company.com"],
        "location": None,
        "meeting_link": "https://meet.google.com/techcorp-kickoff",
        "status": "confirmed",
    },
    # ===== FUTURE (2+ weeks) =====
    {
        "event_id": "event_future_offsite",
        "calendar_id": "primary",
        "title": "Team Offsite - Q1 Planning",
        "description": "Full day offsite for Q1 planning. Strategy and team building.",
        "start_time": days_from_now_at(21, 9, 0),
        "end_time": days_from_now_at(21, 17, 0),
        "attendees": ["team@company.com", "leadership@company.com"],
        "location": "WeWork, 555 Market St",
        "meeting_link": None,
        "status": "confirmed",
    },
    {
        "event_id": "event_future_flight",
        "calendar_id": "primary",
        "title": "Istanbul - NYC Flight TK1234",
        "description": "Turkish Airlines flight TK1234. Booking ref: ABC123.",
        "start_time": days_from_now_at(30, 10, 30),
        "end_time": days_from_now_at(30, 22, 45),
        "attendees": [],
        "location": "Istanbul Airport (IST)",
        "meeting_link": None,
        "status": "confirmed",
    },
    {
        "event_id": "event_future_pto",
        "calendar_id": "primary",
        "title": "PTO - Holiday Break",
        "description": "Out of office for holiday break.",
        "start_time": days_from_now_at(45, 0, 0),
        "end_time": days_from_now_at(52, 0, 0),
        "attendees": [],
        "location": None,
        "meeting_link": None,
        "status": "confirmed",
    },
]


# Mock Drive files with relative dates
# Mix of: TODAY, YESTERDAY, THIS WEEK, LAST WEEK, OLDER
MOCK_FILES = [
    # ===== TODAY =====
    {
        "file_id": "file_meeting_notes_today",
        "name": "Meeting_Notes_Today.docx",
        "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "content_preview": "Meeting Notes - Design Review\n\nAttendees: Design, Product, Engineering\n\nDecisions:\n- New dashboard approved\n- Color scheme updated\n- Launch date confirmed\n\nAction items listed below.",
        "parent_folder": "Meetings",
        "web_link": "https://drive.google.com/file/d/meeting-notes-today",
        "owners": ["user@example.com"],
        "shared_with": ["team@company.com"],
        "modified_at": today_at(10, 30),
    },
    {
        "file_id": "file_draft_proposal",
        "name": "Draft_Client_Proposal.docx",
        "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "content_preview": "Client Proposal - Enterprise Services\n\nDraft version for review.\n\nScope of work, timeline, and pricing outlined. Please review before client meeting tomorrow.",
        "parent_folder": "Sales",
        "web_link": "https://drive.google.com/file/d/draft-proposal",
        "owners": ["user@example.com"],
        "shared_with": ["sales@company.com"],
        "modified_at": today_at(9, 15),
    },
    # ===== YESTERDAY =====
    {
        "file_id": "file_meeting_deck",
        "name": "Acme_Corp_Meeting_Deck.pptx",
        "mime_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "content_preview": "Acme Corp Partnership Meeting\n\nAgenda:\n1. Q3 Review\n2. Q4 Proposal\n3. Next Steps\n\nQ3 Highlights:\n- Revenue up 25%\n- Customer satisfaction 4.5/5",
        "parent_folder": "Acme Corp",
        "web_link": "https://drive.google.com/file/d/acmedeck",
        "owners": ["user@example.com"],
        "shared_with": ["sarah@acme-corp.com"],
        "modified_at": yesterday_at(16, 0),
    },
    {
        "file_id": "file_acme_projections",
        "name": "Acme_Corp_Q4_Projections.xlsx",
        "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "content_preview": "Acme Corp Q4 Projections\n\nRevenue Forecast: $2.5M\nGrowth Rate: 18% YoY\n\nKey Metrics:\n- User adoption: +25%\n- Contract value: $500K/year",
        "parent_folder": "Acme Corp",
        "web_link": "https://drive.google.com/file/d/acme-projections",
        "owners": ["user@example.com"],
        "shared_with": ["sarah@acme-corp.com", "sales@company.com"],
        "modified_at": yesterday_at(11, 0),
    },
    {
        "file_id": "file_techcorp_contract",
        "name": "TechCorp_Enterprise_Agreement.pdf",
        "mime_type": "application/pdf",
        "content_preview": "Enterprise Service Agreement - TechCorp\n\nContract Value: $500,000 ARR\nTerm: 3 years\nSeats: 500 users\n\nIncludes enterprise tier features and dedicated support.",
        "parent_folder": "Contracts",
        "web_link": "https://drive.google.com/file/d/techcorp-contract",
        "owners": ["legal@company.com"],
        "shared_with": ["user@example.com", "sales@company.com"],
        "modified_at": yesterday_at(14, 30),
    },
    # ===== THIS WEEK (2-4 days ago) =====
    {
        "file_id": "file_q4_budget",
        "name": "Q4_Budget_Proposal.pdf",
        "mime_type": "application/pdf",
        "content_preview": "Q4 Budget Proposal\n\nExecutive Summary:\nBudget allocation for Q4 with focus on marketing expansion.\n\nKey Allocations:\n- Marketing: $500,000 (+15%)\n- R&D: $800,000\n- Operations: $300,000",
        "parent_folder": "Finance",
        "web_link": "https://drive.google.com/file/d/q4budget",
        "owners": ["user@example.com"],
        "shared_with": ["sarah@acme-corp.com", "finance@company.com"],
        "modified_at": days_ago_at(2, 11, 0),
    },
    {
        "file_id": "file_api_docs",
        "name": "API_Documentation_v2.5.pdf",
        "mime_type": "application/pdf",
        "content_preview": "API Documentation v2.5\n\nEndpoints:\n- POST /api/v1/users - Create user\n- GET /api/v1/users/{id} - Get user\n\nAuthentication:\n- Bearer token required\n- Tokens expire after 24 hours",
        "parent_folder": "Engineering",
        "web_link": "https://drive.google.com/file/d/api-docs",
        "owners": ["engineering@company.com"],
        "shared_with": ["user@example.com", "team@company.com"],
        "modified_at": days_ago_at(2, 15, 0),
    },
    {
        "file_id": "file_sales_pipeline",
        "name": "Sales_Pipeline_Current.xlsx",
        "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "content_preview": "Sales Pipeline\n\nTotal Pipeline: $2.5M\nWeighted: $1.8M\n\nTop Opportunities:\n1. TechCorp - $500K (90%)\n2. MegaCo - $300K (60%)\n3. StartupXYZ - $200K (40%)",
        "parent_folder": "Sales",
        "web_link": "https://drive.google.com/file/d/sales-pipeline",
        "owners": ["sales@company.com"],
        "shared_with": ["user@example.com", "leadership@company.com"],
        "modified_at": days_ago_at(3, 9, 0),
    },
    # ===== LAST WEEK (5-10 days ago) =====
    {
        "file_id": "file_project_alpha",
        "name": "Project_Alpha_Specs.docx",
        "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "content_preview": "Project Alpha Technical Specifications\n\nOverview:\nNext-generation platform for data analytics.\n\nArchitecture:\n- Microservices-based\n- Cloud-native deployment\n\nTimeline:\n- Phase 1: This quarter\n- Phase 2: Next quarter",
        "parent_folder": "Projects",
        "web_link": "https://drive.google.com/file/d/projectalpha",
        "owners": ["user@example.com"],
        "shared_with": ["john@company.com", "team@company.com"],
        "modified_at": days_ago_at(7, 14, 0),
    },
    {
        "file_id": "file_earnings_report",
        "name": "Q3_Earnings_Report.pdf",
        "mime_type": "application/pdf",
        "content_preview": "Q3 Earnings Report\n\nFinancial Highlights:\n- Revenue: $45.2M (+23% YoY)\n- Gross Margin: 68%\n- Net Income: $8.1M\n\nOperational Metrics:\n- Customer Count: 15,000\n- NPS: 72",
        "parent_folder": "Finance",
        "web_link": "https://drive.google.com/file/d/earnings-q3",
        "owners": ["cfo@company.com"],
        "shared_with": ["user@example.com", "executives@company.com"],
        "modified_at": days_ago_at(8, 10, 0),
    },
    {
        "file_id": "file_architecture_diagram",
        "name": "System_Architecture.png",
        "mime_type": "image/png",
        "content_preview": "System Architecture Diagram showing microservices layout, database connections, cache layers, and external integrations.",
        "parent_folder": "Engineering",
        "web_link": "https://drive.google.com/file/d/arch-diagram",
        "owners": ["user@example.com"],
        "shared_with": ["engineering@company.com"],
        "modified_at": days_ago_at(5, 16, 0),
    },
    {
        "file_id": "file_customer_analysis",
        "name": "Customer_Churn_Analysis.xlsx",
        "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "content_preview": "Customer Churn Analysis\n\nOverall Churn Rate: 2.1%\nNet Revenue Retention: 115%\n\nChurn Reasons:\n- Budget constraints: 35%\n- Switched to competitor: 25%",
        "parent_folder": "Analytics",
        "web_link": "https://drive.google.com/file/d/churn-analysis",
        "owners": ["analytics@company.com"],
        "shared_with": ["user@example.com", "cs@company.com"],
        "modified_at": days_ago_at(6, 11, 0),
    },
    # ===== OLDER (2-4 weeks ago) =====
    {
        "file_id": "file_product_roadmap",
        "name": "Product_Roadmap_2025.pdf",
        "mime_type": "application/pdf",
        "content_preview": "Product Roadmap 2025\n\nQ1: Enterprise Features\n- SSO integration\n- Advanced analytics\n\nQ2: AI & Automation\n- Smart suggestions\n- Automated workflows",
        "parent_folder": "Product",
        "web_link": "https://drive.google.com/file/d/roadmap-2025",
        "owners": ["product@company.com"],
        "shared_with": ["user@example.com", "leadership@company.com"],
        "modified_at": days_ago_at(14, 10, 0),
    },
    {
        "file_id": "file_design_system",
        "name": "Design_System_v3.0.pdf",
        "mime_type": "application/pdf",
        "content_preview": "Design System v3.0\n\nColors:\n- Primary: #2563EB\n- Secondary: #7C3AED\n\nTypography:\n- Headers: Inter Bold\n- Body: Inter Regular\n\nComponents: Buttons, Forms, Cards, Tables, Modals",
        "parent_folder": "Design",
        "web_link": "https://drive.google.com/file/d/design-system",
        "owners": ["design@company.com"],
        "shared_with": ["user@example.com", "engineering@company.com"],
        "modified_at": days_ago_at(15, 14, 0),
    },
    {
        "file_id": "file_ooo_schedule",
        "name": "Out_of_Office_Schedule.xlsx",
        "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "content_preview": "Out of Office Schedule\n\nUpcoming:\n- User: Holiday break next month\n- Sarah: Personal day next week\n- Team: Company holiday end of month",
        "parent_folder": "HR",
        "web_link": "https://drive.google.com/file/d/ooo",
        "owners": ["hr@company.com"],
        "shared_with": ["user@example.com", "team@company.com"],
        "modified_at": days_ago_at(14, 9, 0),
    },
    {
        "file_id": "file_org_chart",
        "name": "Company_Org_Chart.pdf",
        "mime_type": "application/pdf",
        "content_preview": "Company Organization Chart\n\nCEO: Jane Smith\n- CTO: Mike Johnson (Engineering, Product)\n- CFO: Michael Chen (Finance, Operations)\n- COO: Lisa Wang (Sales, Customer Success)",
        "parent_folder": "HR",
        "web_link": "https://drive.google.com/file/d/org-chart",
        "owners": ["hr@company.com"],
        "shared_with": ["all@company.com"],
        "modified_at": days_ago_at(20, 11, 0),
    },
    {
        "file_id": "file_onboarding",
        "name": "New_Employee_Onboarding_Guide.pdf",
        "mime_type": "application/pdf",
        "content_preview": "New Employee Onboarding Guide\n\nWeek 1:\n- IT setup and equipment\n- HR orientation\n- Meet your team\n\nWeek 2:\n- Product training\n- First project assignment",
        "parent_folder": "HR",
        "web_link": "https://drive.google.com/file/d/onboarding",
        "owners": ["hr@company.com"],
        "shared_with": ["user@example.com", "all@company.com"],
        "modified_at": days_ago_at(30, 10, 0),
    },
]


async def seed_data():
    """Seed the database with mock data.

    Creates demo user and populates Gmail, Calendar, and Drive cache tables
    with realistic test data including:
    - Vector embeddings for semantic search
    - TSVECTOR search_vector for BM25 keyword search
    """
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
            await db.flush()
            print("Created demo user")

        # Seed emails
        print("Seeding emails...")
        emails_added = 0
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
            text_for_embedding = f"{email_data['subject']} {email_data['body_preview']}"
            embedding = await embedding_service.embed_for_storage(text_for_embedding)

            # Use ORM model - search_vector is a generated column, handled by PostgreSQL
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
                is_read=False,
                has_attachments=False,
            )
            db.add(email)
            emails_added += 1
            print(f"  Added email: {email_data['subject'][:50]}...")

        # Seed calendar events
        print("Seeding calendar events...")
        events_added = 0
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

            text_for_embedding = f"{event_data['title']} {event_data['description']}"
            embedding = await embedding_service.embed_for_storage(text_for_embedding)

            # Use ORM model - search_vector is a generated column, handled by PostgreSQL
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
            events_added += 1
            print(f"  Added event: {event_data['title'][:50]}...")

        # Seed Drive files
        print("Seeding Drive files...")
        files_added = 0
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

            text_for_embedding = f"{file_data['name']} {file_data['content_preview']}"
            embedding = await embedding_service.embed_for_storage(text_for_embedding)

            # Use ORM model - search_vector is a generated column, handled by PostgreSQL
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
            files_added += 1
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
        print(f"  - {emails_added} emails added (out of {len(MOCK_EMAILS)} total)")
        print(f"  - {events_added} events added (out of {len(MOCK_EVENTS)} total)")
        print(f"  - {files_added} files added (out of {len(MOCK_FILES)} total)")


if __name__ == "__main__":
    asyncio.run(seed_data())
