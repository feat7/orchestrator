# Sample Queries

Test queries with actual responses from the API.

---

## Single Service

### 1. Calendar Search
```
What's on my calendar next week?
```

**Response:**
```
Here's what's on your calendar next week:

- Monday, January 19 at 10:00 AM: Sprint Planning - Sprint 24
- Wednesday, January 22 at 2:00 PM: All Hands Meeting
- Friday, January 24 at 10:00 AM: TechCorp Enterprise Kickoff

Let me know if you'd like more details about any of these events!
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
- Date: Friday, January 10, 2026 at 2:30 PM

Would you like me to open it or provide a summary?
```

---

### 3. Drive Search
```
Show me PDFs in Drive from last month
```

**Response:**
```
I found 5 PDF files in your Google Drive from last month (January 2026):

- TechCorp_Enterprise_Agreement.pdf - modified January 11 at 2:30 PM
- API_Documentation_v2.5.pdf - modified January 10 at 3:00 PM
- Q4_Budget_Proposal.pdf - modified January 10 at 11:00 AM
- Q3_Earnings_Report.pdf - modified January 4 at 10:00 AM
- Product_Roadmap_2025.pdf - modified December 29, 2025

Let me know if you want to open any of these or need more details!
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
**Draft ID:** draft_4060d18a

Hi Support,

I hope this message finds you well. I would like to request the cancellation
of my upcoming flight with Turkish Airlines. The booking reference will be
provided shortly from my search results.

Please let me know the next steps and any information you require from my
side to process this cancellation.

Thanks,

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
on Monday, January 13 at 10:00 AM in Conference Room A:

- Relevant emails:
  • "Prep for tomorrow's Acme Corp meeting" from pm@company.com (Jan 12)
  • "Acme Corp Q4 Budget Review" from sarah@acme-corp.com (Jan 10)
  • "Customer Feedback: Enterprise Deal Closed!" from sales@company.com (Jan 11)

- Important files in Google Drive:
  • Acme_Corp_Meeting_Deck.pptx (presentation, updated Jan 11)
  • Acme_Corp_Q4_Projections.xlsx (spreadsheet, updated Jan 11)

Would you like me to help draft an agenda or summary email for the meeting?
```

---

### 6. Conflict Detection (Calendar + Drive)
```
Find events that conflict with my out-of-office doc
```

**Response:**
```
I found your out-of-office schedule document titled "Out_of_Office_Schedule.xlsx,"
last modified on December 29, 2025.

Regarding your calendar, here are some upcoming events that might conflict
with your out-of-office times:

- "1:1 with Manager" on Monday, January 12 at 11:00 AM
- "1:1 with Sarah" on Wednesday, January 14 at 10:00 AM
- "Team Offsite - Q1 Planning" on Sunday, February 2 at 9:00 AM
- "PTO - Holiday Break" starting Wednesday, February 26

Would you like me to check specific dates from your out-of-office schedule
against these events or help you manage any conflicts?
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
**Draft ID:** draft_41d31c23

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
I tried to create a meeting with Sarah tomorrow at 2 PM, but there was
an error processing the date and time. Could you please confirm the exact
date you mean by "tomorrow" or provide the full date?
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
