# API Documentation

## Base URL
```
http://localhost:8000/api/v1
```

## Authentication
Currently using demo mode. In production, include OAuth token:
```
Authorization: Bearer <token>
```

---

## Endpoints

### Query Processing

#### POST /query
Process a natural language query against Google Workspace.

**Request Body:**
```json
{
  "query": "What's on my calendar next week?",
  "conversation_id": "uuid-optional"
}
```

**Response:**
```json
{
  "response": "Here's what's on your calendar next week:\n- Monday 9 AM: Team standup\n- Tuesday 2 PM: Client call with Acme Corp\n...",
  "actions_taken": [
    {
      "step": "search_calendar",
      "success": true,
      "data": {
        "results": [
          {
            "id": "event_123",
            "title": "Team standup",
            "start_time": "2024-11-11T09:00:00",
            "end_time": "2024-11-11T09:30:00"
          }
        ]
      },
      "error": null
    }
  ],
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "intent": {
    "services": ["gcal"],
    "operation": "search",
    "entities": {"time_range": "next_week"},
    "steps": ["search_calendar"],
    "confidence": 0.95
  }
}
```

**Status Codes:**
- `200`: Success
- `422`: Validation error (missing query)
- `500`: Internal server error

---

### Sync Operations

#### POST /sync/trigger
Trigger a manual sync of Google services.

**Request Body:**
```json
{
  "service": "gmail"  // or "gcal", "gdrive", "all"
}
```

**Response:**
```json
{
  "status": "sync_triggered",
  "service": "gmail",
  "message": "Sync triggered for gmail. Check /sync/status for progress."
}
```

---

#### GET /sync/status
Get the current sync status for all services.

**Response:**
```json
{
  "gmail_last_sync": "2024-11-08T10:30:00",
  "gmail_status": "completed",
  "gcal_last_sync": "2024-11-08T10:30:00",
  "gcal_status": "completed",
  "gdrive_last_sync": "2024-11-08T10:25:00",
  "gdrive_status": "completed"
}
```

---

### Health Check

#### GET /health
Check service health.

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "redis": "connected"
}
```

---

### Conversations

#### GET /conversations/{conversation_id}/messages
Get messages from a conversation.

**Parameters:**
- `conversation_id` (path): UUID of the conversation
- `limit` (query): Max messages to return (default: 10)

**Response:**
```json
[
  {
    "id": "msg-uuid",
    "query": "What's on my calendar?",
    "response": "Here's your calendar...",
    "intent": {...},
    "created_at": "2024-11-08T10:30:00"
  }
]
```

---

## Example Queries

### Single Service Queries

**Calendar:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What'\''s on my calendar next week?"}'
```

**Gmail:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Find emails from sarah@company.com about the budget"}'
```

**Drive:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me PDFs from last month"}'
```

### Multi-Service Queries

**Cancel Flight:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Cancel my Turkish Airlines flight"}'
```

**Meeting Prep:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Prepare for tomorrow'\''s meeting with Acme Corp"}'
```

### With Conversation Context

```bash
# First query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Find emails from sarah@company.com"}'

# Follow-up using conversation_id
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me the latest one",
    "conversation_id": "550e8400-e29b-41d4-a716-446655440000"
  }'
```

---

## Error Responses

### Validation Error (422)
```json
{
  "detail": [
    {
      "loc": ["body", "query"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### Server Error (500)
```json
{
  "response": "I encountered an error processing your request. Please try again.",
  "actions_taken": [],
  "conversation_id": "uuid",
  "intent": null
}
```

---

### Search Quality Metrics

#### GET /metrics/precision
Run search quality benchmark and calculate Precision@5.

This endpoint runs predefined benchmark queries against the user's synced data
and calculates Precision@5 for search quality evaluation.

**Target:** Precision@5 > 0.8 (assignment requirement)

**Response:**
```json
{
  "overall_precision_at_5": 0.85,
  "target_precision": 0.8,
  "meets_target": true,
  "total_queries": 15,
  "queries_evaluated": 12,
  "queries_skipped": 3,
  "per_service": {
    "gmail": {
      "precision_at_5": 0.87,
      "queries_evaluated": 4,
      "total_queries": 5
    },
    "gcal": {
      "precision_at_5": 0.84,
      "queries_evaluated": 4,
      "total_queries": 5
    },
    "gdrive": {
      "precision_at_5": 0.82,
      "queries_evaluated": 4,
      "total_queries": 5
    }
  },
  "details": [
    {
      "query": "Turkish Airlines flight booking",
      "service": "gmail",
      "description": "Find emails about Turkish Airlines flights",
      "precision_at_5": 1.0,
      "relevant_count": 5,
      "total_results": 10,
      "top_5_results": [
        {
          "id": "msg123",
          "title": "Turkish Airlines Booking Confirmation",
          "is_relevant": true,
          "score": 0.92
        }
      ]
    }
  ]
}
```

**How Precision@5 is Calculated:**
1. Run each benchmark query through the search system
2. For each result in top-5, check if it matches relevance criteria (keywords in subject/body)
3. Precision@5 = (# relevant in top 5) / 5
4. Average across all queries for overall score

---

## OpenAPI Specification

Access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Postman Collection

A Postman collection is available at `docs/postman_collection.json`.

**To import:**
1. Open Postman
2. Click "Import" in the top left
3. Select the `postman_collection.json` file
4. The collection includes all endpoints with example requests and responses

**Collection includes:**
- Health & Status endpoints
- Query Processing (10 sample queries including edge cases)
- Sync Operations (trigger and status)
- Authentication (OAuth flow)
- Test Endpoints (intent classification, embeddings)
