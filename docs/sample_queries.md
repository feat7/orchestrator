# Sample Queries and Expected Outputs

This document contains 10+ sample queries with expected outputs, including edge cases.

All examples use the `/api/v1/intent` endpoint for intent classification and `/api/v1/query` for full execution.

---

## API Request/Response Examples

### Using the Intent Endpoint

```bash
# Login first (for demo mode)
curl -X POST http://localhost:8000/api/v1/auth/demo-login -c cookies.txt

# Classify intent only
curl -X POST http://localhost:8000/api/v1/intent \
  -H "Content-Type: application/json" \
  -b cookies.txt \
  -d '{"query": "What important things to do this week?"}'
```

### Using the Query Endpoint

```bash
# Execute full query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -b cookies.txt \
  -d '{"query": "What important things to do this week?"}'
```

---

## 1. Single Service: Calendar Search

**Query:**
```
What's on my calendar next week?
```

**API Request:**
```bash
curl -X POST http://localhost:8000/api/v1/intent \
  -H "Content-Type: application/json" \
  -b cookies.txt \
  -d '{"query": "What'\''s on my calendar next week?"}'
```

**Intent Response:**
```json
{
  "query": "What's on my calendar next week?",
  "intent": {
    "services": ["gcal"],
    "operation": "search",
    "steps": [
      {
        "step": "search_calendar",
        "params": {
          "search_query": "",
          "start_after": "2026-01-19T00:00:00",
          "start_before": "2026-01-25T23:59:59"
        }
      }
    ],
    "confidence": 0.95
  },
  "latency_ms": 1523
}
```

**Full Query Response:**
```
Here's what's on your calendar next week:

- Monday, January 19 at 10:00 AM: Sprint Planning - Sprint 24
- Wednesday, January 22 at 2:00 PM: All Hands Meeting
- Friday, January 24 at 10:00 AM: TechCorp Enterprise Kickoff

Would you like more details about any of these events?
```

---

## 2. Single Service: Email Search by Sender

**Query:**
```
Find emails from sarah@company.com about the budget
```

**Intent Response:**
```json
{
  "query": "Find emails from sarah@company.com about the budget",
  "intent": {
    "services": ["gmail"],
    "operation": "search",
    "steps": [
      {
        "step": "search_gmail",
        "params": {
          "search_query": "budget",
          "sender": "sarah@company.com"
        }
      }
    ],
    "confidence": 0.95
  },
  "latency_ms": 1245
}
```

**Full Query Response:**
```
I found one email from sarah@company.com about the budget:

- Subject: "Re: Q4 Budget Discussion"
- Date: Friday, January 10, 2026 at 2:30 PM

Would you like me to open it or provide a summary?
```

---

## 3. Single Service: Drive File Search

**Query:**
```
Show me PDFs in Drive from last month
```

**Intent Response:**
```json
{
  "query": "Show me PDFs in Drive from last month",
  "intent": {
    "services": ["gdrive"],
    "operation": "search",
    "steps": [
      {
        "step": "search_drive",
        "params": {
          "search_query": "",
          "modified_after": "2025-12-13",
          "mime_type": "application/pdf"
        }
      }
    ],
    "confidence": 0.9
  },
  "latency_ms": 1102
}
```

**Full Query Response:**
```
I found 5 PDF files in your Google Drive from last month:

- TechCorp_Enterprise_Agreement.pdf - modified January 11 at 2:30 PM
- API_Documentation_v2.5.pdf - modified January 10 at 3:00 PM
- Q4_Budget_Proposal.pdf - modified January 10 at 11:00 AM
- Q3_Earnings_Report.pdf - modified January 4 at 10:00 AM
- Product_Roadmap_2025.pdf - modified December 29 at 10:00 AM

Would you like me to open or share any of these files?
```

---

## 4. Multi-Service: Cancel Flight

**Query:**
```
Cancel my Turkish Airlines flight
```

**Intent Response:**
```json
{
  "query": "Cancel my Turkish Airlines flight",
  "intent": {
    "services": ["gmail", "gcal"],
    "operation": "action",
    "steps": [
      {
        "step": "search_gmail",
        "params": {
          "search_query": "Turkish Airlines flight booking confirmation"
        }
      },
      {
        "step": "search_calendar",
        "params": {
          "search_query": "Turkish Airlines flight"
        }
      },
      {
        "step": "draft_email",
        "params": {
          "to": "support@turkishairlines.com",
          "subject": "Flight Cancellation Request",
          "body": "I would like to cancel my upcoming flight."
        },
        "depends_on": [0]
      }
    ],
    "confidence": 0.9
  },
  "latency_ms": 2156
}
```

**Full Query Response:**
```
I've drafted an email for you and saved it to your Gmail Drafts:

**To:** support@turkishairlines.com
**Subject:** Flight Cancellation Request

I found your booking:
- Email: "Turkish Airlines Flight Confirmation - TK1234" (20 days ago)
- Calendar: "Istanbul - NYC Flight TK1234" on February 11

Would you like me to send the cancellation email?
```

---

## 5. Multi-Service: Meeting Preparation

**Query:**
```
Prepare for tomorrow's meeting with Acme Corp
```

**Intent Response:**
```json
{
  "query": "Prepare for tomorrow's meeting with Acme Corp",
  "intent": {
    "services": ["gcal", "gmail", "gdrive"],
    "operation": "search",
    "steps": [
      {
        "step": "search_calendar",
        "params": {
          "search_query": "Acme Corp",
          "start_after": "2026-01-13T00:00:00",
          "start_before": "2026-01-14T00:00:00"
        }
      },
      {
        "step": "search_gmail",
        "params": {
          "search_query": "Acme Corp"
        }
      },
      {
        "step": "search_drive",
        "params": {
          "search_query": "Acme Corp"
        }
      }
    ],
    "confidence": 0.9
  },
  "latency_ms": 1834
}
```

**Full Query Response:**
```
Here's what I found to help you prepare for tomorrow's meeting with Acme Corp:

**Meeting Details:**
- Acme Corp Partnership Meeting
- Tuesday, January 13 at 10:00 AM (Conference Room A)
- Attendees: sarah@acme-corp.com, pm@company.com, sales@company.com

**Related Emails:**
- "Prep for tomorrow's Acme Corp meeting" from pm@company.com (today)
- "Acme Corp Q4 Budget Review" from sarah@acme-corp.com (2 days ago)

**Related Files:**
- Acme_Corp_Meeting_Deck.pptx
- Acme_Corp_Q4_Projections.xlsx
- Q4_Budget_Proposal.pdf

Would you like me to open any of these documents?
```

---

## 6. Multi-Service: Conflict Detection

**Query:**
```
Find events next week that conflict with my out-of-office doc
```

**API Request:**
```bash
curl -X POST http://localhost:8000/api/v1/intent \
  -H "Content-Type: application/json" \
  -b cookies.txt \
  -d '{"query": "Find events next week that conflict with my out-of-office doc"}'
```

**Intent Response:**
```json
{
  "query": "Find events next week that conflict with my out-of-office doc",
  "intent": {
    "services": ["gcal", "gdrive"],
    "operation": "search",
    "steps": [
      {
        "step": "search_drive",
        "params": {
          "search_query": "out-of-office OOO vacation"
        }
      },
      {
        "step": "search_calendar",
        "params": {
          "search_query": "",
          "start_after": "2026-01-19T00:00:00",
          "start_before": "2026-01-25T23:59:59"
        }
      }
    ],
    "confidence": 0.8
  },
  "latency_ms": 1756
}
```

**Full Query Response:**
```
I found your out-of-office schedule and checked for conflicts:

**Your OOO Schedule:**
Nov 5-10: Travel to NYC

**Potential Conflicts:**
- Istanbul → NYC Flight TK1234 (Nov 5) - This is your travel
- Daily Standup (Nov 6-10) - You'll miss these during travel

Would you like me to update these events or notify attendees?
```

---

## 7. Edge Case: Ambiguous Query

**Query:**
```
Move the meeting with John
```

**API Request:**
```bash
curl -X POST http://localhost:8000/api/v1/intent \
  -H "Content-Type: application/json" \
  -b cookies.txt \
  -d '{"query": "Move the meeting with John"}'
```

**Intent Response:**
```json
{
  "query": "Move the meeting with John",
  "intent": {
    "services": ["gcal"],
    "operation": "update",
    "steps": [
      {
        "step": "search_calendar",
        "params": {
          "search_query": "John meeting"
        }
      }
    ],
    "confidence": 0.6
  },
  "latency_ms": 1234
}
```

**Full Query Response:**
```
I found multiple events that might be "the meeting with John":

1. "1:1 with Sarah" - john@company.com is not an attendee
2. "Client Call - Project Alpha Demo" - includes john@company.com

Could you clarify:
- Which meeting would you like to move?
- When would you like to reschedule it to?
```

---

## 8. Edge Case: Context Reference

**Query (first):**
```
Find emails from sarah@company.com
```

**Query (follow-up):**
```
That email about the proposal
```

**API Request (first query):**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -b cookies.txt \
  -d '{"query": "Find emails from sarah@company.com"}'
```

**API Request (follow-up with conversation context):**
```bash
curl -X POST http://localhost:8000/api/v1/intent \
  -H "Content-Type: application/json" \
  -b cookies.txt \
  -d '{"query": "That email about the proposal", "conversation_id": "conv_123"}'
```

**Intent Response:**
```json
{
  "query": "That email about the proposal",
  "intent": {
    "services": ["gmail"],
    "operation": "search",
    "steps": [
      {
        "step": "search_gmail",
        "params": {
          "search_query": "proposal",
          "sender": "sarah@company.com"
        }
      }
    ],
    "confidence": 0.75
  },
  "latency_ms": 1102
}
```

**Full Query Response:**
```
I couldn't find an email from sarah@company.com specifically about a proposal.

The most relevant email I found was:
- "Acme Corp Q4 Budget Review" which mentions the Q4 budget proposal

Is this what you were looking for, or would you like me to search more broadly?
```

---

## 9. Edge Case: Temporal Reasoning

**Query:**
```
What meetings do I have next Tuesday?
```

**API Request:**
```bash
curl -X POST http://localhost:8000/api/v1/intent \
  -H "Content-Type: application/json" \
  -b cookies.txt \
  -d '{"query": "What meetings do I have next Tuesday?"}'
```

**Intent Response:**
```json
{
  "query": "What meetings do I have next Tuesday?",
  "intent": {
    "services": ["gcal"],
    "operation": "search",
    "steps": [
      {
        "step": "search_calendar",
        "params": {
          "search_query": "meeting",
          "start_after": "2026-01-20T00:00:00",
          "start_before": "2026-01-20T23:59:59"
        }
      }
    ],
    "confidence": 0.95
  },
  "latency_ms": 1089
}
```

**Full Query Response:**
```
You have 1 meeting scheduled for next Tuesday (Jan 20):

- 2:00 PM: Client Call - Project Alpha Demo (1 hour)
  Attendees: client@bigcorp.com, john@company.com
  Meeting link: https://zoom.us/j/123456789

Would you like me to add anything to your schedule or send reminders?
```

---

## 10. Edge Case: No Results

**Query:**
```
Find emails about the Jupiter project
```

**API Request:**
```bash
curl -X POST http://localhost:8000/api/v1/intent \
  -H "Content-Type: application/json" \
  -b cookies.txt \
  -d '{"query": "Find emails about the Jupiter project"}'
```

**Intent Response:**
```json
{
  "query": "Find emails about the Jupiter project",
  "intent": {
    "services": ["gmail"],
    "operation": "search",
    "steps": [
      {
        "step": "search_gmail",
        "params": {
          "search_query": "Jupiter project"
        }
      }
    ],
    "confidence": 0.9
  },
  "latency_ms": 978
}
```

**Full Query Response:**
```
I couldn't find any emails about "Jupiter project" in your inbox.

Suggestions:
- Try different keywords (project name, people involved)
- Check if it might be under a different name
- The emails might have been archived or deleted

Would you like me to search Drive or Calendar for "Jupiter project" instead?
```

---

## 11. Edge Case: Multiple Actions

**Query:**
```
Draft an email to john@company.com about the meeting and add it to my calendar for Friday at 2pm
```

**API Request:**
```bash
curl -X POST http://localhost:8000/api/v1/intent \
  -H "Content-Type: application/json" \
  -b cookies.txt \
  -d '{"query": "Draft an email to john@company.com about the meeting and add it to my calendar for Friday at 2pm"}'
```

**Intent Response:**
```json
{
  "query": "Draft an email to john@company.com about the meeting and add it to my calendar for Friday at 2pm",
  "intent": {
    "services": ["gmail", "gcal"],
    "operation": "create",
    "steps": [
      {
        "step": "draft_email",
        "params": {
          "to": "john@company.com",
          "subject": "Meeting Discussion",
          "body": "Hi John,\n\nI wanted to discuss the upcoming meeting.\n\nBest regards"
        }
      },
      {
        "step": "create_event",
        "params": {
          "title": "Meeting with John",
          "start_time": "2026-01-16T14:00:00",
          "end_time": "2026-01-16T15:00:00",
          "attendees": ["john@company.com"]
        }
      }
    ],
    "confidence": 0.85
  },
  "latency_ms": 1567
}
```

**Full Query Response:**
```
I've prepared both items for you:

**Email Draft:**
- To: john@company.com
- Subject: Meeting Discussion
- Status: Draft saved

**Calendar Event:**
- Title: Meeting with John
- Time: Friday at 2:00 PM
- Status: Created

Would you like me to send the email or make any changes to the event?
```

---

## 12. Edge Case: Permission/Access Error

**Query:**
```
Share the contract file with external@competitor.com
```

**API Request:**
```bash
curl -X POST http://localhost:8000/api/v1/intent \
  -H "Content-Type: application/json" \
  -b cookies.txt \
  -d '{"query": "Share the contract file with external@competitor.com"}'
```

**Intent Response:**
```json
{
  "query": "Share the contract file with external@competitor.com",
  "intent": {
    "services": ["gdrive"],
    "operation": "update",
    "steps": [
      {
        "step": "search_drive",
        "params": {
          "search_query": "contract"
        }
      },
      {
        "step": "share_file",
        "params": {
          "share_with": "external@competitor.com",
          "permission": "reader"
        },
        "depends_on": [0]
      }
    ],
    "confidence": 0.8
  },
  "latency_ms": 1234
}
```

**Full Query Response:**
```
I found the file "Service_Agreement_2024.pdf" but I need to flag a concern:

- This appears to be a confidential contract document
- The email domain (competitor.com) is not in your organization
- Current access: Only legal@company.com and sarah@acme-corp.com

Are you sure you want to share this with an external party? If yes, please confirm and I'll proceed.
```

---

## Testing Notes

1. **Run all queries through the API:**
   ```bash
   for query in "query1" "query2"; do
     curl -X POST http://localhost:8000/api/v1/query \
       -H "Content-Type: application/json" \
       -d "{\"query\": \"$query\"}"
   done
   ```

2. **Check intent classification accuracy** by comparing `intent` field in response

3. **Verify semantic search relevance** by examining `actions_taken.data.results`

4. **Test conversation context** by using same `conversation_id` across queries
