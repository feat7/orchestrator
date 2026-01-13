# Sample Queries

Test queries with actual responses from the API.

> **Note:** Dates in responses are relative to today's date since seed data uses relative dates. The exact dates and LLM-generated text may vary slightly between runs.

---

## Single Service

### 1. Calendar Search
```
What's on my calendar next week?
```

**Response:**
```
Here's what's on your calendar next week:

- Tuesday, January 20 at 10:00 AM: Sprint Planning - Sprint 24
- Thursday, January 23 at 2:00 PM: All Hands Meeting
- Saturday, January 25 at 10:00 AM: TechCorp Enterprise Kickoff

Let me know if you'd like details about any of these events!
```

---

### 2. Email Search
```
Find emails from sarah@company.com about the budget
```

**Response:**
```
I found one email from sarah@company.com about the budget:

- Subject: "Re: Q4 Budget Discussion"
- Date: Saturday, January 11, 2026 at 2:30 PM

Would you like me to open it or provide a summary?
```

---

### 3. Drive Search
```
Show me PDFs in Drive from last month
```

**Response:**
```
I found several PDFs in your Google Drive from last month and early January 2026:

- TechCorp_Enterprise_Agreement.pdf - modified January 12 at 2:30 PM
- API_Documentation_v2.5.pdf - modified January 11 at 3:00 PM
- Q4_Budget_Proposal.pdf - modified January 11 at 11:00 AM
- Q3_Earnings_Report.pdf - modified January 5 at 10:00 AM

There is also one from December 30, 2025:
- Product_Roadmap_2025.pdf

Click any link to open the file directly in Google Drive. Let me know if you want
details on any of these!
```

---

## Multi-Service

### 4. Cancel Flight (Gmail + Calendar + Draft)
```
Cancel my Turkish Airlines flight
```

**Response:**
```
I've drafted an email for you and saved it to your Gmail Drafts:

**To:** support@turkishairlines.com
**Subject:** Flight Cancellation Request
**Draft ID:** draft_xxxxx

Hi Support,

I hope this message finds you well. I would like to cancel my upcoming flight
with Turkish Airlines. Please note that the booking reference will be provided
from the search results.

Thank you for your assistance.

Best regards,

---

This draft is saved in your Gmail. You can edit it here or directly in your mailbox.

Would you like me to send it? Reply 'yes' to send, or let me know what changes you'd like.
```

---

### 5. Meeting Prep (Calendar + Gmail + Drive)
```
Prepare for tomorrow's meeting with Acme Corp
```

**Response:**
```
Here's what I found to help you prepare for tomorrow's meeting with Acme Corp
on Wednesday, January 14 at 10:00 AM in Conference Room A:

- Calendar event: "Acme Corp Partnership Meeting"
- Relevant emails:
  - "Prep for tomorrow's Acme Corp meeting" from pm@company.com (Jan 13)
  - "Acme Corp Q4 Budget Review" from sarah@acme-corp.com (Jan 11)
- Important files:
  - Acme_Corp_Meeting_Deck.pptx (updated Jan 12)
  - Acme_Corp_Q4_Projections.xlsx (updated Jan 12)

Would you like me to help draft an agenda or summary email for the meeting?
```

---

### 6. Conflict Detection (Calendar + Drive)
```
Find events that conflict with my out-of-office doc
```

**Response:**
```
I found your out-of-office schedule document:
- Out_of_Office_Schedule.xlsx (last modified December 30, 2025)

Looking at your calendar, here are upcoming events that might conflict with your
out-of-office times:
- "1:1 with Manager" on Tuesday, January 13 at 11:00 AM
- "1:1 with Sarah" on Thursday, January 15 at 10:00 AM
- "Team Offsite - Q1 Planning" on Monday, February 3 at 9:00 AM
- "PTO - Holiday Break" starting Thursday, February 27 (all day)

Would you like me to check specific dates from your out-of-office file against
these events or help you manage any conflicts?
```

---

## Actions

### 7. Draft Email (with recipient resolution)
```
Draft an email to John about the project update
```

**Response:**
```
I've drafted an email for you and saved it to your Gmail Drafts:

**To:** john@company.com
**Subject:** Project Update
**Draft ID:** draft_xxxxx

Hi John,

I wanted to provide you with a quick update on the project. Please let me know
if you have any questions or need further details.

Best regards,

---

This draft is saved in your Gmail. You can edit it here or directly in your mailbox.

Would you like me to send it? Reply 'yes' to send, or let me know what changes you'd like.
```

---

### 8. Send Email (follow-up to draft)
```
Send it
```
*(After reviewing the draft from "Cancel my Turkish Airlines flight")*

**Response:**
```
Done! Your email has been sent.

**To:** support@turkishairlines.com
**Subject:** Flight Cancellation Request

The email is now in your Sent folder.
```

---

### 9. Create Meeting
```
Create a meeting with Sarah tomorrow at 2pm
```

**Response:**
```
I've created a calendar event for your meeting with Sarah tomorrow at 2:00 PM.

Additionally, I found recent emails from Sarah that might be helpful:
- "Team lunch today at noon?" from sarah@company.com on January 13, 2026
- "Re: Q4 Budget Discussion" from sarah@company.com on January 11, 2026
- "Acme Corp Q4 Budget Review" from sarah@acme-corp.com on January 11, 2026

Let me know if you'd like me to send a meeting invite or draft a message to Sarah.
```

---

## API Usage

### Intent Classification Only
```bash
curl -X POST http://localhost:8000/api/v1/intent \
  -H "Content-Type: application/json" \
  -b cookies.txt \
  -d '{"query": "What is on my calendar next week?"}'
```

### Full Query Execution
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -b cookies.txt \
  -d '{"query": "What is on my calendar next week?"}'
```

### With Conversation Context
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -b cookies.txt \
  -d '{"query": "Send it", "conversation_id": "abc123"}'
```
