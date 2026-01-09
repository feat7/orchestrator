# Sample Queries and Expected Outputs

This document contains 10+ sample queries with expected outputs, including edge cases.

---

## 1. Single Service: Calendar Search

**Query:**
```
What's on my calendar next week?
```

**Expected Intent:**
```json
{
  "services": ["gcal"],
  "operation": "search",
  "entities": {"time_range": "next_week"},
  "steps": ["search_calendar"],
  "confidence": 0.95
}
```

**Expected Response:**
```
Here's what's on your calendar next week:
- Monday 9:00 AM: Daily Standup (30 min)
- Tuesday 2:00 PM: Acme Corp Quarterly Review with sarah@acme-corp.com
- Thursday 2:00 PM: 1:1 with Sarah
- Friday 12:00 PM: Team Lunch at Pasta Paradise

Would you like more details about any of these events?
```

---

## 2. Single Service: Email Search by Sender

**Query:**
```
Find emails from sarah@acme-corp.com about the budget
```

**Expected Intent:**
```json
{
  "services": ["gmail"],
  "operation": "search",
  "entities": {"sender": "sarah@acme-corp.com", "topic": "budget"},
  "steps": ["search_gmail"],
  "confidence": 0.95
}
```

**Expected Response:**
```
I found 1 email from sarah@acme-corp.com about the budget:

- "Acme Corp Q4 Budget Review" (2 days ago)
  Preview: Hi team, please review the attached Q4 budget proposal...

Would you like me to show the full email?
```

---

## 3. Single Service: Drive File Search

**Query:**
```
Show me PDFs in Drive from last month
```

**Expected Intent:**
```json
{
  "services": ["gdrive"],
  "operation": "search",
  "entities": {"mime_type": "pdf", "time_range": "last_month"},
  "steps": ["search_drive"],
  "confidence": 0.9
}
```

**Expected Response:**
```
I found 2 PDF files from last month:

- Q4_Budget_Proposal.pdf (modified 3 days ago)
  In folder: Acme Corp

- Service_Agreement_2024.pdf (modified 60 days ago)
  In folder: Contracts

Would you like me to open or share any of these files?
```

---

## 4. Multi-Service: Cancel Flight

**Query:**
```
Cancel my Turkish Airlines flight
```

**Expected Intent:**
```json
{
  "services": ["gmail", "gcal"],
  "operation": "update",
  "entities": {"airline": "Turkish Airlines", "action": "cancel"},
  "steps": ["search_gmail", "search_calendar", "draft_email"],
  "confidence": 0.9
}
```

**Expected Response:**
```
I found your Turkish Airlines booking:

- Email: "Turkish Airlines Flight Confirmation - TK1234"
  From: reservations@turkishairlines.com (20 days ago)
  Booking ref: ABC123

- Calendar: "Istanbul → NYC Flight TK1234"
  Date: Nov 5, 2024 at 10:30 AM

I've drafted a cancellation email to support@turkishairlines.com with your booking reference.

Would you like me to send it?
```

---

## 5. Multi-Service: Meeting Preparation

**Query:**
```
Prepare for tomorrow's meeting with Acme Corp
```

**Expected Intent:**
```json
{
  "services": ["gcal", "gmail", "gdrive"],
  "operation": "search",
  "entities": {"company": "Acme Corp", "time": "tomorrow", "event_type": "meeting"},
  "steps": ["search_calendar", "search_gmail", "search_drive"],
  "confidence": 0.85
}
```

**Expected Response:**
```
Here's everything for your Acme Corp meeting tomorrow:

**Meeting Details:**
- Acme Corp Quarterly Review
- Tomorrow at 2:00 PM (1 hour)
- Location: Conference Room A
- Attendees: sarah@acme-corp.com, john@company.com

**Related Emails:**
- "Acme Corp Q4 Budget Review" from sarah@acme-corp.com (2 days ago)

**Related Files:**
- Q4_Budget_Proposal.pdf
- Acme_Corp_Meeting_Deck.pptx

Would you like me to open any of these documents?
```

---

## 6. Multi-Service: Conflict Detection

**Query:**
```
Find events next week that conflict with my out-of-office doc
```

**Expected Intent:**
```json
{
  "services": ["gcal", "gdrive"],
  "operation": "search",
  "entities": {"time_range": "next_week", "document_type": "out-of-office"},
  "steps": ["search_drive", "search_calendar"],
  "confidence": 0.8
}
```

**Expected Response:**
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

**Expected Intent:**
```json
{
  "services": ["gcal"],
  "operation": "update",
  "entities": {"attendee": "John", "action": "reschedule"},
  "steps": ["search_calendar"],
  "confidence": 0.6
}
```

**Expected Response:**
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

**Expected Intent:**
```json
{
  "services": ["gmail"],
  "operation": "search",
  "entities": {"sender": "sarah@company.com", "topic": "proposal"},
  "steps": ["get_email"],
  "confidence": 0.75
}
```

**Expected Response:**
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

**Expected Intent:**
```json
{
  "services": ["gcal"],
  "operation": "search",
  "entities": {"day": "next_tuesday", "event_type": "meeting"},
  "steps": ["search_calendar"],
  "confidence": 0.95
}
```

**Expected Response:**
```
You have 1 meeting scheduled for next Tuesday (Nov 12):

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

**Expected Intent:**
```json
{
  "services": ["gmail"],
  "operation": "search",
  "entities": {"topic": "Jupiter project"},
  "steps": ["search_gmail"],
  "confidence": 0.9
}
```

**Expected Response:**
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

**Expected Intent:**
```json
{
  "services": ["gmail", "gcal"],
  "operation": "create",
  "entities": {
    "recipient": "john@company.com",
    "topic": "meeting",
    "day": "Friday",
    "time": "2pm"
  },
  "steps": ["draft_email", "create_event"],
  "confidence": 0.85
}
```

**Expected Response:**
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

**Expected Intent:**
```json
{
  "services": ["gdrive"],
  "operation": "update",
  "entities": {"file_type": "contract", "share_with": "external@competitor.com"},
  "steps": ["search_drive", "share_file"],
  "confidence": 0.8
}
```

**Expected Response:**
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
