# Google Workspace Orchestrator - System Design

## Overview

The Google Workspace Orchestrator is an intelligent system that processes natural language queries across Gmail, Google Calendar, and Google Drive services. It uses LLM-based intent classification, parallel execution, and semantic search to provide a unified interface for interacting with Google Workspace.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           User Query                                 │
│                    "Cancel my Turkish Airlines flight"               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Intent Classifier (LLM)                        │
│  • Parse query into structured intent                                │
│  • Identify services: [gmail, gcal]                                  │
│  • Extract entities: {airline: "Turkish Airlines"}                   │
│  • Determine steps: [search_gmail, search_calendar, draft_email]     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Query Planner (DAG)                          │
│  • Build execution graph with dependencies                           │
│  • Group parallel operations                                         │
│  • Plan: Group 1: [search_gmail, search_calendar] (parallel)         │
│          Group 2: [draft_email] (depends on Group 1)                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Service Orchestrator                             │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Gmail Agent  │  │ GCal Agent   │  │ GDrive Agent │               │
│  │  • search()  │  │  • search()  │  │  • search()  │               │
│  │  • execute() │  │  • execute() │  │  • execute() │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│         │                  │                  │                      │
│         └──────────────────┼──────────────────┘                      │
│                            ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Embedding & Search Layer (pgvector)             │    │
│  │  • Vector similarity search with cosine distance             │    │
│  │  • Metadata filtering (date, sender, type)                   │    │
│  │  • Hybrid search for best results                            │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Response Synthesizer (LLM)                      │
│  • Aggregate results from all steps                                  │
│  • Generate natural language response                                │
│  • Include actionable suggestions                                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Natural Response                             │
│  "I found your Turkish Airlines booking (TK1234)..."                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Intent Classifier (`app/core/intent.py`)
- Uses LLM with structured prompts to parse queries
- Returns `ParsedIntent` with services, operation, entities, steps
- Supports conversation context for follow-up queries
- Handles ambiguous queries with confidence scores

### 2. Query Planner (`app/core/planner.py`)
- Converts intent into execution DAG
- Identifies dependencies between steps
- Groups independent steps for parallel execution
- Handles fallback strategies for failures

### 3. Orchestrator (`app/core/orchestrator.py`)
- Main execution engine
- Runs parallel groups using `asyncio.gather`
- Passes results between dependent steps
- Handles errors gracefully per-step

### 4. Service Agents (`app/agents/`)
- **BaseAgent**: Abstract interface for all agents
- **GmailAgent**: Email search, read, draft, send
- **GcalAgent**: Event search, create, update, delete
- **GdriveAgent**: File search, read, share

### 5. Response Synthesizer (`app/core/synthesizer.py`)
- Aggregates results from all steps
- Uses LLM to generate natural language response
- Formats data for readability
- Suggests follow-up actions

## Data Flow

1. **Query Received** → API endpoint receives user query
2. **Classification** → LLM classifies intent into structured format
3. **Planning** → Planner creates execution DAG with parallel groups
4. **Execution** → Orchestrator runs steps, respecting dependencies
5. **Search** → Agents use embeddings for semantic search in pgvector
6. **Synthesis** → LLM generates natural response from results
7. **Response** → User receives conversational answer

## Database Schema

### Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ENTITY RELATIONSHIP DIAGRAM                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐
│        users         │
├──────────────────────┤
│ PK id           UUID │
│    email     VARCHAR │
│    google_access_    │
│      token      TEXT │
│    google_refresh_   │
│      token      TEXT │
│    token_expires_at  │
│              TIMESTAMP│
│    created_at        │
│              TIMESTAMP│
│    updated_at        │
│              TIMESTAMP│
└──────────┬───────────┘
           │
           │ 1
           │
           ├────────────────────┬────────────────────┬────────────────────┐
           │                    │                    │                    │
           │ *                  │ *                  │ *                  │ *
┌──────────▼───────────┐ ┌──────▼──────────────┐ ┌───▼─────────────────┐ ┌▼─────────────────────┐
│    conversations     │ │     gmail_cache     │ │     gcal_cache      │ │    gdrive_cache      │
├──────────────────────┤ ├─────────────────────┤ ├─────────────────────┤ ├──────────────────────┤
│ PK id           UUID │ │ PK id          UUID │ │ PK id          UUID │ │ PK id           UUID │
│ FK user_id      UUID │ │ FK user_id     UUID │ │ FK user_id     UUID │ │ FK user_id      UUID │
│    created_at        │ │    email_id VARCHAR │ │    event_id VARCHAR │ │    file_id   VARCHAR │
│              TIMESTAMP│ │    thread_id       │ │    calendar_id      │ │    name         TEXT │
└──────────┬───────────┘ │              VARCHAR │ │              VARCHAR │ │    mime_type VARCHAR │
           │             │    subject     TEXT │ │    title       TEXT │ │    content_preview   │
           │ 1           │    sender   VARCHAR │ │    description TEXT │ │                 TEXT │
           │             │    recipients JSONB │ │    start_time       │ │    parent_folder     │
           │ *           │    body_preview     │ │              TIMESTAMP│ │              VARCHAR │
┌──────────▼───────────┐ │                TEXT │ │    end_time         │ │    web_link     TEXT │
│       messages       │ │    body_full   TEXT │ │              TIMESTAMP│ │    owners      JSONB │
├──────────────────────┤ │    embedding        │ │    attendees  JSONB │ │    shared_with JSONB │
│ PK id           UUID │ │          VECTOR(1536)│ │    location    TEXT │ │    embedding         │
│ FK conversation_id   │ │    received_at      │ │    meeting_link     │ │          VECTOR(1536)│
│                 UUID │ │              TIMESTAMP│ │                TEXT │ │    created_at        │
│    query        TEXT │ │    labels     JSONB │ │    status   VARCHAR │ │              TIMESTAMP│
│    intent      JSONB │ │    is_read  BOOLEAN │ │    embedding        │ │    modified_at       │
│    response     TEXT │ │    has_attachments  │ │          VECTOR(1536)│ │              TIMESTAMP│
│    actions_taken     │ │             BOOLEAN │ │    synced_at        │ │    synced_at         │
│               JSONB  │ │    synced_at        │ │              TIMESTAMP│ │              TIMESTAMP│
│    created_at        │ │              TIMESTAMP│ └─────────────────────┘ └──────────────────────┘
│              TIMESTAMP│ └─────────────────────┘
└──────────────────────┘
                                                   ┌─────────────────────┐
                                                   │     sync_status     │
                                                   ├─────────────────────┤
                                                   │ PK id          UUID │
                                                   │ FK user_id     UUID │
                                                   │    service  VARCHAR │
                                                   │    last_sync_at     │
                                                   │              TIMESTAMP│
                                                   │    last_sync_token  │
                                                   │                TEXT │
                                                   │    status   VARCHAR │
                                                   │    error_message    │
                                                   │                TEXT │
                                                   └─────────────────────┘

RELATIONSHIPS:
─────────────────────────────────────────────────────────────────────────────
• users (1) ──────< (many) conversations
• users (1) ──────< (many) gmail_cache
• users (1) ──────< (many) gcal_cache
• users (1) ──────< (many) gdrive_cache
• users (1) ──────< (many) sync_status
• conversations (1) ──────< (many) messages

INDEXES:
─────────────────────────────────────────────────────────────────────────────
• gmail_cache.embedding    → ivfflat (vector_cosine_ops) for semantic search
• gcal_cache.embedding     → ivfflat (vector_cosine_ops) for semantic search
• gdrive_cache.embedding   → ivfflat (vector_cosine_ops) for semantic search
• gmail_cache(user_id, email_id)   → UNIQUE for upsert operations
• gcal_cache(user_id, event_id)    → UNIQUE for upsert operations
• gdrive_cache(user_id, file_id)   → UNIQUE for upsert operations
• sync_status(user_id, service)    → UNIQUE for per-service tracking
```

### Core Tables
- `users`: User accounts with OAuth tokens
- `conversations`: Conversation sessions
- `messages`: Individual query/response pairs

### Cache Tables (with vector indexes)
- `gmail_cache`: Cached emails with embeddings
- `gcal_cache`: Cached events with embeddings
- `gdrive_cache`: Cached files with embeddings

### Supporting Tables
- `sync_status`: Track sync status per service

## Scaling to 1M Users

### Architecture for Scale

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Load Balancer (nginx/ALB)                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │   API Pod 1  │ │   API Pod 2  │ │   API Pod N  │
            │   (FastAPI)  │ │   (FastAPI)  │ │   (FastAPI)  │
            └──────────────┘ └──────────────┘ └──────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
    ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
    │    Redis     │       │  PostgreSQL  │       │    Celery    │
    │   (Cache)    │       │  (pgvector)  │       │  (Workers)   │
    └──────────────┘       └──────────────┘       └──────────────┘
```

### Key Strategies

#### 1. Caching (Redis)
- **Embedding Cache**: 1hr TTL, MD5 hash as key
- **Intent Cache**: 5min TTL for repeated queries
- **Conversation Context**: Last 10 messages per conversation
- **Target**: >80% cache hit rate

#### 2. Rate Limiting
- 100 queries/user/hour
- Google API quota management (250 units/sec)
- Exponential backoff on failures

#### 3. Async Processing
- Celery workers for background sync
- Long-running orchestrations offloaded
- Real-time updates via WebSocket (bonus)

#### 4. Pre-computation
- Background sync every 15 minutes
- Index new emails/events/files automatically
- Embedding generation during sync

#### 5. Database Optimization
- Partition by `user_id` for large tables
- pgvector HNSW indexes for fast similarity search
- Connection pooling with asyncpg
- Read replicas for query load

#### 6. Multi-Region
- Deploy in US, EU, APAC
- Route to nearest region
- Regional data residency compliance

### Performance Targets
- P99 latency: <2s
- Query throughput: 10K queries/sec
- Embedding freshness: <15min lag
- Google API errors: <0.1%

## Security Considerations

### Authentication
- OAuth 2.0 with Google
- JWT tokens for API authentication
- Token refresh handling

### Data Protection
- User data isolation (all queries filtered by user_id)
- Encrypted OAuth tokens
- Audit logging for all operations

### Rate Limiting
- Per-user query limits
- API key quotas
- DDoS protection at load balancer

## Testing Strategy

### Unit Tests
- Intent classification accuracy
- Planner dependency resolution
- Agent operations

### Integration Tests
- Full query flow with mock services
- Database operations
- Cache behavior

### Performance Tests
- Load testing with locust
- Embedding query latency
- Concurrent user simulation

## Deployment

### Local Development
```bash
# Start services
docker-compose up -d

# Run migrations
alembic upgrade head

# Seed mock data
python -m scripts.seed_mock_data

# Start API
uvicorn app.main:app --reload
```

### Production
- Kubernetes deployment
- Helm charts for configuration
- GitOps with ArgoCD
- Monitoring with Prometheus/Grafana
