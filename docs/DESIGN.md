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

This section details how to scale the Google Workspace Orchestrator from a single-server deployment to a system capable of serving 1 million active users with high availability and low latency.

---

### 1. Requirements Analysis

#### Functional Requirements
- Process natural language queries across Gmail, Calendar, and Drive
- Execute multi-service orchestration with parallel operations
- Maintain conversation context across sessions
- Background sync of user data from Google APIs
- Real-time streaming responses

#### Non-Functional Requirements
| Metric | Target | Rationale |
|--------|--------|-----------|
| Availability | 99.9% (8.76h downtime/year) | Critical for enterprise users |
| P50 Latency | < 500ms | Interactive user experience |
| P99 Latency | < 2s | Acceptable for complex queries |
| Throughput | 10,000 QPS | Peak load capacity |
| Data Freshness | < 15 min | Acceptable sync lag |

#### Capacity Estimation

**Users & Traffic:**
```
Total Users:           1,000,000
Daily Active Users:    300,000 (30% DAU)
Queries/User/Day:      10 queries
Daily Queries:         3,000,000
Peak QPS:              3M / 86400 × 3 (peak multiplier) ≈ 105 QPS
Design Target:         500 QPS (5x headroom for growth)
```

**Storage:**
```
Emails/User:           5,000 (avg inbox size)
Events/User:           500 (2 years of events)
Files/User:            200 (Drive files)
Embedding Size:        1536 × 4 bytes = 6KB per item

Per-User Storage:
  - Gmail cache:       5,000 × (2KB metadata + 6KB embedding) = 40MB
  - Calendar cache:    500 × (1KB metadata + 6KB embedding) = 3.5MB
  - Drive cache:       200 × (1KB metadata + 6KB embedding) = 1.4MB
  - Total per user:    ~45MB

Total Storage:         1M users × 45MB = 45TB
With replication (3x): 135TB
```

**Bandwidth:**
```
Avg Response Size:     2KB
Peak Bandwidth:        500 QPS × 2KB = 1MB/s outbound
Google API Sync:       ~100MB/user/month incremental
Monthly Sync Traffic:  100TB (with delta sync optimization: ~10TB)
```

---

### 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                      CLIENTS                                             │
│                    (Web App, Mobile App, API Consumers)                                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           │ HTTPS
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               EDGE LAYER                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │   CloudFlare    │  │   AWS WAF       │  │  Rate Limiter   │  │   API Gateway   │    │
│  │   (CDN/DDoS)    │  │   (Security)    │  │  (per-user)     │  │   (Kong/AWS)    │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                        ┌──────────────────┼──────────────────┐
                        ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                            APPLICATION LAYER (Stateless)                                 │
│                                                                                          │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                        Kubernetes Cluster (Auto-scaling)                          │   │
│  │                                                                                   │   │
│  │   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │   │
│  │   │  API Pod    │ │  API Pod    │ │  API Pod    │ │  API Pod    │  ... (N pods) │   │
│  │   │  (FastAPI)  │ │  (FastAPI)  │ │  (FastAPI)  │ │  (FastAPI)  │               │   │
│  │   │             │ │             │ │             │ │             │               │   │
│  │   │ - Intent    │ │ - Intent    │ │ - Intent    │ │ - Intent    │               │   │
│  │   │ - Planner   │ │ - Planner   │ │ - Planner   │ │ - Planner   │               │   │
│  │   │ - Orchestr. │ │ - Orchestr. │ │ - Orchestr. │ │ - Orchestr. │               │   │
│  │   │ - Synth.    │ │ - Synth.    │ │ - Synth.    │ │ - Synth.    │               │   │
│  │   └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘               │   │
│  │                                                                                   │   │
│  │   HPA: Scale on CPU (70%) / Custom Metrics (QPS, Latency)                        │   │
│  │   Min: 3 pods | Max: 50 pods | Target: 100 QPS/pod                               │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                         Celery Workers (Background Jobs)                          │   │
│  │   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                                │   │
│  │   │ Sync Worker │ │ Sync Worker │ │ Sync Worker │  ... (M workers)               │   │
│  │   │ (Gmail)     │ │ (Calendar)  │ │ (Drive)     │                                │   │
│  │   └─────────────┘ └─────────────┘ └─────────────┘                                │   │
│  │   Concurrency: 4 tasks/worker | Auto-scale on queue depth                        │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                        │                  │                  │
          ┌─────────────┘                  │                  └─────────────┐
          ▼                                ▼                                ▼
┌───────────────────┐          ┌───────────────────┐          ┌───────────────────┐
│   CACHE LAYER     │          │    DATA LAYER     │          │  EXTERNAL APIS    │
│                   │          │                   │          │                   │
│  ┌─────────────┐  │          │  ┌─────────────┐  │          │  ┌─────────────┐  │
│  │   Redis     │  │          │  │ PostgreSQL  │  │          │  │ Google APIs │  │
│  │   Cluster   │  │          │  │  Primary    │  │          │  │  - Gmail    │  │
│  │             │  │          │  │ (pgvector)  │  │          │  │  - Calendar │  │
│  │ - Embeddings│  │          │  └──────┬──────┘  │          │  │  - Drive    │  │
│  │ - Intents   │  │          │         │         │          │  └─────────────┘  │
│  │ - Sessions  │  │          │  ┌──────┴──────┐  │          │                   │
│  │ - Rate Lim. │  │          │  │   Replicas  │  │          │  ┌─────────────┐  │
│  └─────────────┘  │          │  │  (2 read)   │  │          │  │  LLM APIs   │  │
│                   │          │  └─────────────┘  │          │  │  - OpenAI   │  │
│  6 nodes          │          │                   │          │  │  - Anthropic│  │
│  (3 primary +     │          │  Sharding:        │          │  └─────────────┘  │
│   3 replica)      │          │  user_id % 4      │          │                   │
└───────────────────┘          └───────────────────┘          └───────────────────┘
          │                                │                                │
          └────────────────────────────────┼────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              OBSERVABILITY LAYER                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │ Prometheus  │  │  Grafana    │  │   Jaeger    │  │    ELK      │  │  PagerDuty  │   │
│  │ (Metrics)   │  │ (Dashboard) │  │  (Tracing)  │  │  (Logging)  │  │  (Alerting) │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3. Component Deep Dives

#### 3.1 API Gateway & Load Balancing

```
                                    Internet
                                        │
                                        ▼
                            ┌───────────────────┐
                            │    CloudFlare     │
                            │  ─────────────────│
                            │  • DDoS Protection│
                            │  • SSL Termination│
                            │  • Geo-routing    │
                            │  • Edge Caching   │
                            └─────────┬─────────┘
                                      │
                                      ▼
                            ┌───────────────────┐
                            │   AWS ALB / NLB   │
                            │  ─────────────────│
                            │  • Health checks  │
                            │  • Sticky sessions│
                            │  • Cross-zone LB  │
                            └─────────┬─────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
              ┌───────────┐     ┌───────────┐     ┌───────────┐
              │  API Pod  │     │  API Pod  │     │  API Pod  │
              └───────────┘     └───────────┘     └───────────┘
```

**Rate Limiting Strategy:**
```python
# Token bucket per user (stored in Redis)
RATE_LIMITS = {
    "free_tier": {"requests": 100, "window": 3600},      # 100/hour
    "pro_tier": {"requests": 1000, "window": 3600},      # 1000/hour
    "enterprise": {"requests": 10000, "window": 3600},   # 10000/hour
}

# Redis key: rate_limit:{user_id}:{window_start}
# Sliding window counter with atomic INCR + EXPIRE
```

#### 3.2 Application Layer (Stateless Services)

Each API pod runs the complete orchestration pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Pod                                  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Request Handler                        │   │
│  │  • Authentication (JWT validation)                        │   │
│  │  • Request parsing & validation                           │   │
│  │  • Conversation context loading                           │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Intent Classifier                        │   │
│  │  • LLM call (cached for identical queries)               │   │
│  │  • Entity extraction                                      │   │
│  │  • Confidence scoring                                     │   │
│  │  • Cache: Redis (TTL: 5min)                              │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Query Planner                          │   │
│  │  • Build execution DAG                                    │   │
│  │  • Identify parallelizable steps                          │   │
│  │  • Dependency resolution                                  │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Orchestrator                           │   │
│  │  • Parallel execution (asyncio.gather)                    │   │
│  │  • Per-step error handling                                │   │
│  │  • Result aggregation                                     │   │
│  │                                                           │   │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐                  │   │
│  │   │ Gmail   │  │ GCal    │  │ GDrive  │   (run parallel) │   │
│  │   │ Agent   │  │ Agent   │  │ Agent   │                  │   │
│  │   └─────────┘  └─────────┘  └─────────┘                  │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 Response Synthesizer                      │   │
│  │  • LLM call for natural language generation              │   │
│  │  • Streaming response (SSE)                              │   │
│  │  • Action suggestions                                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Pod Specifications:**
```yaml
resources:
  requests:
    cpu: "500m"
    memory: "1Gi"
  limits:
    cpu: "2000m"
    memory: "4Gi"

autoscaling:
  minReplicas: 3
  maxReplicas: 50
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          averageValue: 100
```

#### 3.3 Database Architecture

**Sharding Strategy:**

For 1M users with 45TB total data, we use hash-based sharding by `user_id`:

```
                            ┌─────────────────┐
                            │  Shard Router   │
                            │  (in app layer) │
                            └────────┬────────┘
                                     │
           shard = hash(user_id) % 4 │
                                     │
        ┌────────────┬───────────────┼───────────────┬────────────┐
        ▼            ▼               ▼               ▼            ▼
   ┌─────────┐  ┌─────────┐    ┌─────────┐    ┌─────────┐  ┌─────────┐
   │ Shard 0 │  │ Shard 1 │    │ Shard 2 │    │ Shard 3 │  │ Global  │
   │ 250K    │  │ 250K    │    │ 250K    │    │ 250K    │  │ (users, │
   │ users   │  │ users   │    │ users   │    │ users   │  │  auth)  │
   │         │  │         │    │         │    │         │  │         │
   │ Primary │  │ Primary │    │ Primary │    │ Primary │  │ Primary │
   │    +    │  │    +    │    │    +    │    │    +    │  │    +    │
   │ 2 Read  │  │ 2 Read  │    │ 2 Read  │    │ 2 Read  │  │ 2 Read  │
   │ Replicas│  │ Replicas│    │ Replicas│    │ Replicas│  │ Replicas│
   └─────────┘  └─────────┘    └─────────┘    └─────────┘  └─────────┘

   Each shard: ~11TB (with replicas: 33TB)
   Total: 4 × 33TB + global = ~135TB
```

**Why 4 Shards?**
- Each shard handles ~250K users, ~11TB data
- PostgreSQL performs well up to 10-20TB per instance
- 4 shards provide good parallelism without operational complexity
- Easy to double (8 shards) if needed with consistent hashing

**Connection Pooling:**
```python
# Per-pod connection pool (using asyncpg)
DATABASE_POOL_CONFIG = {
    "min_size": 5,           # Min connections per pod
    "max_size": 20,          # Max connections per pod
    "max_inactive_time": 300, # Close idle connections after 5min
}

# With 50 API pods × 20 connections = 1000 connections/shard
# PostgreSQL max_connections = 1500 (with headroom)
```

**Read/Write Splitting:**
```python
# Writes go to primary
async def save_message(user_id: str, message: Message):
    shard = get_shard(user_id)
    async with shard.primary.acquire() as conn:
        await conn.execute(INSERT_QUERY, ...)

# Reads go to replicas (with failover to primary)
async def search_emails(user_id: str, query: str):
    shard = get_shard(user_id)
    async with shard.replica.acquire() as conn:
        return await conn.fetch(SEARCH_QUERY, ...)
```

**Vector Index Optimization:**
```sql
-- HNSW index for approximate nearest neighbor search
-- Tuned for 1M+ vectors per shard
CREATE INDEX ON gmail_cache
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Partial index for recent emails (hot data)
CREATE INDEX ON gmail_cache (user_id, received_at DESC)
WHERE received_at > NOW() - INTERVAL '30 days';
```

#### 3.4 Caching Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Redis Cluster (6 nodes)                           │
│                                                                              │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │                      Cache Hierarchy                               │     │
│   │                                                                    │     │
│   │   L1: Intent Cache                    L2: Embedding Cache          │     │
│   │   ─────────────────                   ────────────────────         │     │
│   │   Key: intent:{hash(query)}           Key: emb:{hash(text)}        │     │
│   │   TTL: 5 minutes                      TTL: 1 hour                  │     │
│   │   Size: ~500 bytes                    Size: 6KB (1536 floats)      │     │
│   │   Hit Rate Target: 60%                Hit Rate Target: 90%         │     │
│   │                                                                    │     │
│   │   L3: Session/Context Cache           L4: Rate Limit Counters      │     │
│   │   ───────────────────────             ───────────────────────      │     │
│   │   Key: conv:{conversation_id}         Key: rate:{user_id}:{window} │     │
│   │   TTL: 1 hour                         TTL: 1 hour (sliding)        │     │
│   │   Size: ~10KB (last 10 msgs)          Size: 16 bytes               │     │
│   │   Hit Rate Target: 95%                Always in cache              │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│   Memory Allocation:                                                         │
│   ─────────────────                                                          │
│   • Embedding cache: 80% (~48GB across cluster for 8M cached embeddings)     │
│   • Intent cache: 10% (~6GB for 12M cached intents)                          │
│   • Session cache: 8% (~5GB for 500K active sessions)                        │
│   • Rate limiters: 2% (~1GB for 1M user counters)                            │
│                                                                              │
│   Total: 60GB cluster (10GB per node)                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Cache Invalidation Strategy:**
```python
# Write-through for user data changes
async def on_email_sync(user_id: str, emails: List[Email]):
    # 1. Update database
    await db.upsert_emails(user_id, emails)

    # 2. Invalidate related caches
    await cache.delete_pattern(f"emb:*:{user_id}:*")  # Clear stale embeddings

    # 3. Pre-warm hot embeddings
    for email in emails[:10]:  # Most recent emails
        embedding = await embedding_service.embed(email.subject + email.body)
        await cache.set(f"emb:{hash(email.content)}", embedding, ttl=3600)
```

#### 3.5 Background Job Processing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Celery Architecture                                 │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         Redis (Message Broker)                       │    │
│  │                                                                      │    │
│  │   Queues:                                                            │    │
│  │   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │    │
│  │   │  gmail_sync │ │  gcal_sync  │ │ gdrive_sync │ │  priority   │   │    │
│  │   │  (default)  │ │  (default)  │ │  (default)  │ │   (high)    │   │    │
│  │   └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                         │                                    │
│               ┌─────────────────────────┼─────────────────────────┐         │
│               ▼                         ▼                         ▼         │
│  ┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐  │
│  │   Worker Pool 1   │     │   Worker Pool 2   │     │   Worker Pool 3   │  │
│  │   (Gmail Sync)    │     │   (GCal Sync)     │     │   (GDrive Sync)   │  │
│  │                   │     │                   │     │                   │  │
│  │   Concurrency: 8  │     │   Concurrency: 4  │     │   Concurrency: 4  │  │
│  │   Prefetch: 4     │     │   Prefetch: 2     │     │   Prefetch: 2     │  │
│  │                   │     │                   │     │                   │  │
│  │   Tasks:          │     │   Tasks:          │     │   Tasks:          │  │
│  │   • full_sync     │     │   • full_sync     │     │   • full_sync     │  │
│  │   • delta_sync    │     │   • delta_sync    │     │   • delta_sync    │  │
│  │   • embed_batch   │     │   • embed_batch   │     │   • embed_batch   │  │
│  └───────────────────┘     └───────────────────┘     └───────────────────┘  │
│                                                                              │
│  Scaling:                                                                    │
│  ─────────                                                                   │
│  • Auto-scale based on queue depth (Celery Flower + K8s HPA)                 │
│  • Min: 3 workers per pool | Max: 20 workers per pool                        │
│  • Scale up when queue > 1000 tasks | Scale down when queue < 100            │
│                                                                              │
│  Scheduling:                                                                 │
│  ──────────                                                                  │
│  • Delta sync: Every 15 minutes per user (staggered)                         │
│  • Full sync: Weekly (off-peak hours)                                        │
│  • Priority queue: User-triggered syncs (immediate)                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Sync Scheduling for 1M Users:**
```python
# Staggered sync to avoid thundering herd
# 1M users / 15 min = ~1,100 syncs/second (too high)
# Solution: Spread across 15-minute windows based on user_id hash

def get_sync_slot(user_id: str) -> int:
    """Returns minute offset (0-14) for user's sync slot."""
    return hash(user_id) % 15

# Each minute: 1M / 15 = ~66,666 users sync
# With delta sync (avg 10 API calls): 666,666 API calls/minute = 11,111/sec
# Google API quota: 250 units/sec → Need batching + multiple service accounts
```

#### 3.6 External API Management

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      External API Gateway                                    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Google API Client Pool                            │    │
│  │                                                                      │    │
│  │   ┌───────────────────────────────────────────────────────────┐     │    │
│  │   │                  Service Account Pool                      │     │    │
│  │   │                                                            │     │    │
│  │   │   Account 1        Account 2        Account 3    ... (N)   │     │    │
│  │   │   250 units/s      250 units/s      250 units/s            │     │    │
│  │   │                                                            │     │    │
│  │   │   Total Pool Capacity: N × 250 = 2,500 units/sec (N=10)   │     │    │
│  │   └───────────────────────────────────────────────────────────┘     │    │
│  │                                                                      │    │
│  │   ┌───────────────────────────────────────────────────────────┐     │    │
│  │   │                  Circuit Breaker                           │     │    │
│  │   │                                                            │     │    │
│  │   │   State: CLOSED → OPEN (on 5 failures) → HALF-OPEN → CLOSED│     │    │
│  │   │   Timeout: 30 seconds in OPEN state                        │     │    │
│  │   │   Failure threshold: 50% error rate in 10-second window    │     │    │
│  │   └───────────────────────────────────────────────────────────┘     │    │
│  │                                                                      │    │
│  │   ┌───────────────────────────────────────────────────────────┐     │    │
│  │   │                  Retry Strategy                            │     │    │
│  │   │                                                            │     │    │
│  │   │   • Exponential backoff: 1s, 2s, 4s, 8s, 16s (max 5 retries)│    │    │
│  │   │   • Jitter: ±25% to prevent thundering herd                │     │    │
│  │   │   • Retry on: 429 (rate limit), 500, 502, 503, 504        │     │    │
│  │   │   • No retry on: 400, 401, 403, 404                        │     │    │
│  │   └───────────────────────────────────────────────────────────┘     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      LLM API Client                                  │    │
│  │                                                                      │    │
│  │   Primary: OpenAI (GPT-4)         Fallback: Anthropic (Claude)       │    │
│  │                                                                      │    │
│  │   • Automatic failover on 5xx errors                                 │    │
│  │   • Response caching for identical prompts (5 min TTL)               │    │
│  │   • Streaming for long responses                                     │    │
│  │   • Cost tracking per request                                        │    │
│  │                                                                      │    │
│  │   Rate Limits:                                                       │    │
│  │   • OpenAI: 10,000 RPM (request/min), 2M TPM (tokens/min)           │    │
│  │   • Anthropic: 4,000 RPM, 400K TPM (as backup)                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 4. Request Flow (Detailed)

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                              REQUEST LIFECYCLE                                            │
└──────────────────────────────────────────────────────────────────────────────────────────┘

User: "Cancel my Turkish Airlines flight"
      │
      │ ① HTTPS Request
      ▼
┌─────────────┐
│  CloudFlare │ ──→ DDoS check, SSL termination, geo-routing
└──────┬──────┘
       │ ② Forwarded to origin
       ▼
┌─────────────┐
│  API Gateway│ ──→ Rate limit check (Redis): rate:{user_id}:{window}
└──────┬──────┘     └─→ If exceeded: HTTP 429 Too Many Requests
       │
       │ ③ Route to healthy pod
       ▼
┌─────────────┐
│  API Pod    │
│             │
│  ┌──────────┴───────────────────────────────────────────────────────────┐
│  │                                                                       │
│  │  ④ Authentication                                                     │
│  │     └─→ Validate JWT token                                           │
│  │     └─→ Load user from global DB: SELECT * FROM users WHERE id = ?   │
│  │                                                                       │
│  │  ⑤ Load Conversation Context                                          │
│  │     └─→ Redis: GET conv:{conversation_id}                             │
│  │     └─→ If miss: SELECT * FROM messages WHERE conversation_id = ?    │
│  │                                                                       │
│  │  ⑥ Intent Classification [~200ms]                                     │
│  │     └─→ Check cache: GET intent:{hash(query)}                         │
│  │     └─→ If miss: Call LLM API with system prompt + query              │
│  │     └─→ Parse response: {services: [gmail, gcal], steps: [...]}      │
│  │     └─→ Cache result: SET intent:{hash} ... EX 300                   │
│  │                                                                       │
│  │  ⑦ Query Planning [~10ms]                                             │
│  │     └─→ Build DAG from steps                                          │
│  │     └─→ Group parallel steps: [[search_gmail, search_calendar],       │
│  │                                [draft_email]]                         │
│  │                                                                       │
│  │  ⑧ Parallel Execution [~100ms total]                                  │
│  │     │                                                                 │
│  │     │  ┌─────────────────────────────────────────────────────────┐   │
│  │     │  │ asyncio.gather(                                          │   │
│  │     │  │   gmail_agent.search("Turkish Airlines"),  ──┐           │   │
│  │     │  │   gcal_agent.search("flight")              ──┼─ parallel │   │
│  │     │  │ )                                            ──┘           │   │
│  │     │  └─────────────────────────────────────────────────────────┘   │
│  │     │                                                                 │
│  │     │  Search Flow (per service):                                     │
│  │     │  └─→ Get shard: shard = hash(user_id) % 4                       │
│  │     │  └─→ Embed query: GET emb:{hash(query)} or call OpenAI          │
│  │     │  └─→ Hybrid search:                                             │
│  │     │      • BM25: ts_rank(search_vector, to_tsquery(?))              │
│  │     │      • Vector: embedding <=> ? ORDER BY distance LIMIT 10       │
│  │     │      • Filtered: Same + WHERE received_at > NOW() - 30 days     │
│  │     │  └─→ RRF fusion: Combine ranked results                         │
│  │     │  └─→ Return top 5 results                                       │
│  │     │                                                                 │
│  │     └─→ After parallel group completes:                               │
│  │         └─→ draft_email(booking_ref=results[0].booking_ref)           │
│  │                                                                       │
│  │  ⑨ Response Synthesis [~150ms]                                        │
│  │     └─→ Call LLM with results + system prompt                         │
│  │     └─→ Stream response chunks via SSE                                │
│  │                                                                       │
│  │  ⑩ Persist & Respond                                                  │
│  │     └─→ Save to messages table (async, non-blocking)                  │
│  │     └─→ Update conversation context in Redis                          │
│  │     └─→ Return SSE stream to client                                   │
│  │                                                                       │
│  └───────────────────────────────────────────────────────────────────────┘
│             │
└─────────────┘
       │
       ▼
┌─────────────┐
│   Client    │ ──→ Receives streamed response over ~500ms
└─────────────┘

Total Latency Breakdown:
──────────────────────────
• Network (CDN → API):     20ms
• Auth + Context:          30ms
• Intent Classification:  200ms (cached: 5ms)
• Planning:                10ms
• Parallel Search:        100ms
• Response Synthesis:     150ms (streaming)
──────────────────────────
Total:                   ~510ms (P50)
```

---

### 5. Failure Handling & Resilience

#### 5.1 Failure Scenarios & Mitigations

| Failure | Detection | Mitigation | Recovery |
|---------|-----------|------------|----------|
| **API Pod Crash** | K8s liveness probe (5s) | Load balancer removes pod | Auto-restart (30s) |
| **Redis Down** | Health check failure | Fall through to DB | Auto-reconnect |
| **DB Primary Down** | pg_isready timeout | Promote replica | Automatic failover (Patroni) |
| **Google API 429** | HTTP status code | Exponential backoff | Retry after delay |
| **Google API 5xx** | HTTP status code | Circuit breaker opens | Half-open after 30s |
| **LLM API Down** | Timeout (10s) | Failover to backup provider | Cache fallback for repeated queries |
| **Full Shard Down** | All replicas unreachable | Return partial results | Failover to DR region |

#### 5.2 Circuit Breaker Implementation

```
                    ┌─────────────────────────────────────────┐
                    │          Circuit Breaker FSM            │
                    └─────────────────────────────────────────┘

         ┌────────────────────────────────────────────────────────┐
         │                                                        │
         ▼                                                        │
    ┌─────────┐    failure_count >= 5    ┌─────────┐              │
    │ CLOSED  │ ──────────────────────→  │  OPEN   │              │
    │         │                          │         │              │
    │ Normal  │                          │ Reject  │              │
    │ traffic │                          │   all   │              │
    └─────────┘                          └────┬────┘              │
         ▲                                    │                   │
         │                              30s timeout               │
         │                                    │                   │
         │         success              ┌─────▼─────┐             │
         └───────────────────────────── │ HALF-OPEN │             │
                                        │           │             │
                                        │ Allow 1   │             │
                                        │ request   │ ──failure──→┘
                                        └───────────┘
```

#### 5.3 Graceful Degradation

```python
# Example: Multi-service query with partial failure
async def execute_query(intent: ParsedIntent) -> QueryResponse:
    results = {}
    errors = {}

    # Execute all services, collect results and errors
    tasks = [
        ("gmail", gmail_agent.search(intent.entities)),
        ("gcal", gcal_agent.search(intent.entities)),
        ("gdrive", gdrive_agent.search(intent.entities)),
    ]

    for service, task in tasks:
        try:
            results[service] = await asyncio.wait_for(task, timeout=5.0)
        except asyncio.TimeoutError:
            errors[service] = "Service timed out"
        except ServiceUnavailable:
            errors[service] = "Service temporarily unavailable"

    # Synthesize response with available data
    return await synthesizer.generate(
        results=results,
        errors=errors,
        partial=len(errors) > 0  # Flag for LLM to acknowledge partial results
    )

# Example response with partial failure:
# "I found your Turkish Airlines booking in Gmail (TK1234).
#  Note: Calendar search is temporarily unavailable, so I couldn't
#  check for related events. Would you like me to try again?"
```

---

### 6. Multi-Region Deployment

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              MULTI-REGION ARCHITECTURE                                   │
└─────────────────────────────────────────────────────────────────────────────────────────┘

                                  ┌───────────────┐
                                  │   Route 53    │
                                  │  (GeoDNS)     │
                                  └───────┬───────┘
                                          │
                   ┌──────────────────────┼──────────────────────┐
                   │                      │                      │
                   ▼                      ▼                      ▼
          ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
          │   US-EAST     │      │   EU-WEST     │      │   APAC        │
          │   (Primary)   │      │   (Primary)   │      │   (Primary)   │
          │               │      │               │      │               │
          │  ┌─────────┐  │      │  ┌─────────┐  │      │  ┌─────────┐  │
          │  │ API     │  │      │  │ API     │  │      │  │ API     │  │
          │  │ Cluster │  │      │  │ Cluster │  │      │  │ Cluster │  │
          │  └─────────┘  │      │  └─────────┘  │      │  └─────────┘  │
          │       │       │      │       │       │      │       │       │
          │  ┌─────────┐  │      │  ┌─────────┐  │      │  ┌─────────┐  │
          │  │ Redis   │  │      │  │ Redis   │  │      │  │ Redis   │  │
          │  │ Cluster │  │      │  │ Cluster │  │      │  │ Cluster │  │
          │  └─────────┘  │      │  └─────────┘  │      │  └─────────┘  │
          │       │       │      │       │       │      │       │       │
          │  ┌─────────┐  │      │  ┌─────────┐  │      │  ┌─────────┐  │
          │  │ PG      │  │◄────►│  │ PG      │  │◄────►│  │ PG      │  │
          │  │ Cluster │  │ sync │  │ Cluster │  │ sync │  │ Cluster │  │
          │  └─────────┘  │      │  └─────────┘  │      │  └─────────┘  │
          │               │      │               │      │               │
          └───────────────┘      └───────────────┘      └───────────────┘

Data Residency:
───────────────
• US users → US-EAST (data stays in US)
• EU users → EU-WEST (GDPR compliance)
• APAC users → APAC (data sovereignty)

Cross-Region Sync:
──────────────────
• User metadata: Async replication (eventual consistency, <1s lag)
• Cache data: No replication (region-local)
• Auth tokens: Replicated for cross-region login

Failover:
─────────
• If US-EAST down: Route US traffic to EU-WEST (with latency warning)
• RTO: 5 minutes (DNS TTL)
• RPO: 1 second (async replication lag)
```

---

### 7. Monitoring & Observability

#### 7.1 Key Metrics Dashboard

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OPERATIONS DASHBOARD                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  REQUEST METRICS                          LATENCY DISTRIBUTION               │
│  ────────────────                         ────────────────────               │
│  Current QPS: ████████████ 342/s          P50: ████████░░ 420ms              │
│  Peak QPS:    ████████████████ 512/s      P95: ████████████░ 890ms           │
│  Error Rate:  ██░░░░░░░░░░ 0.3%           P99: ██████████████ 1.8s           │
│                                                                              │
│  CACHE PERFORMANCE                        SERVICE HEALTH                     │
│  ─────────────────                        ──────────────                     │
│  Intent Hit:   ████████░░ 78%             API Pods:    ●●●●●○○○ 5/8         │
│  Embed Hit:    █████████░ 92%             Redis:       ●●●●●● 6/6            │
│  Session Hit:  ██████████ 97%             PostgreSQL:  ●●●●●● 6/6            │
│                                           Celery:      ●●●●●●●● 12/12        │
│                                                                              │
│  EXTERNAL APIS                            QUEUE DEPTH                        │
│  ─────────────                            ───────────                        │
│  Google API:   ██████████ 99.7% OK        Gmail Sync:  ████░░░░ 2,341        │
│  OpenAI:       █████████░ 98.2% OK        GCal Sync:   ██░░░░░░ 891          │
│  Anthropic:    ██████████ 99.9% OK        Drive Sync:  ███░░░░░ 1,203        │
│                                                                              │
│  PRECISION@5 (Search Quality)             USER ACTIVITY                      │
│  ────────────────────────────             ─────────────                      │
│  Gmail:    █████████░ 0.87                DAU:     312,456                   │
│  Calendar: ████████░░ 0.82                Queries: 2.1M today                │
│  Drive:    █████████░ 0.85                Syncs:   890K today                │
│  Overall:  █████████░ 0.85                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 7.2 Alerting Rules

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| High Error Rate | error_rate > 1% for 5m | P1 (Critical) | Page on-call |
| Latency Spike | P99 > 5s for 5m | P1 (Critical) | Page on-call |
| Cache Down | Redis unreachable | P1 (Critical) | Auto-failover + page |
| DB Connection Pool | connections > 80% | P2 (Warning) | Scale pods |
| Queue Backlog | depth > 10K for 15m | P2 (Warning) | Scale workers |
| API Quota Warning | usage > 80% | P3 (Info) | Alert Slack |
| Precision Drop | P@5 < 0.7 | P3 (Info) | Alert ML team |

#### 7.3 Distributed Tracing

```
Trace ID: abc123def456
──────────────────────────────────────────────────────────────────────────────

[API Gateway]─────────────[20ms]────────────────────────────────────────────►
     │
     └─[Auth]────[15ms]───►
     │
     └─[Intent Classifier]─────────────────[195ms]───────────────────────────►
     │         │
     │         └─[LLM Call (OpenAI)]───────[180ms]──────────────────────────►
     │
     └─[Orchestrator]──────────────────────────────────[250ms]───────────────►
               │
               ├─[Gmail Search]────────────[85ms]────────────────►
               │      │
               │      ├─[Embed Query]──[12ms]──►
               │      └─[Vector Search]──────[65ms]───►
               │
               ├─[GCal Search]─────────────[78ms]────────────────►  (parallel)
               │      │
               │      ├─[Embed Query]──[12ms]──► (cached)
               │      └─[Vector Search]──────[58ms]───►
               │
               └─[Draft Email]─────────────────────[45ms]────────►  (sequential)
     │
     └─[Synthesizer]───────────────────────────────────────[160ms]───────────►
               │
               └─[LLM Call (OpenAI)]───────────────────────[145ms]───────────►

Total: 510ms
```

---

### 8. Cost Optimization

#### 8.1 Monthly Cost Breakdown (1M Users)

| Component | Specification | Monthly Cost |
|-----------|---------------|--------------|
| **Compute** | | |
| API Pods (avg 15 pods) | c5.2xlarge × 15 | $2,500 |
| Celery Workers (avg 30) | c5.xlarge × 30 | $3,000 |
| **Database** | | |
| PostgreSQL (4 shards) | db.r5.4xlarge × 12 | $15,000 |
| Redis Cluster | cache.r5.xlarge × 6 | $2,400 |
| **Storage** | | |
| EBS (135TB with replicas) | gp3 @ $0.08/GB | $11,000 |
| S3 (backups) | 50TB @ $0.023/GB | $1,150 |
| **External APIs** | | |
| OpenAI (3M queries × 2 calls × 1K tokens) | @ $0.002/1K tokens | $12,000 |
| Google Workspace API | Per-user licensing | Included |
| **Network** | | |
| Data Transfer | ~10TB/month | $900 |
| CloudFlare Pro | Enterprise plan | $2,500 |
| **Observability** | | |
| Datadog / Grafana Cloud | Full stack | $3,000 |
| **Total** | | **~$53,450/month** |

**Cost per User:** $0.053/month (~$0.64/year)

#### 8.2 Optimization Strategies

1. **Reserved Instances**: 40% savings on compute → Save $2,200/month
2. **Spot Instances for Workers**: 70% savings on Celery → Save $2,100/month
3. **Tiered Storage**: Move cold data to S3 Glacier → Save $5,000/month
4. **LLM Caching**: 60% cache hit rate → Save $7,200/month on OpenAI
5. **Right-sizing**: Continuous optimization → Save $2,000/month

**Optimized Total: ~$35,000/month** ($0.035/user/month)

---

### 9. Trade-offs & Design Decisions

| Decision | Alternatives Considered | Why This Choice |
|----------|------------------------|-----------------|
| **PostgreSQL + pgvector** vs Pinecone/Weaviate | Managed vector DBs | Single DB for metadata + vectors, no vendor lock-in, lower cost |
| **Hash sharding** vs Range sharding | Range by user signup date | Even distribution, no hotspots, simple routing |
| **Redis Cluster** vs Memcached | Memcached is simpler | Redis persistence, pub/sub for cache invalidation, Lua scripting |
| **Celery** vs AWS SQS + Lambda | Serverless options | Control over execution, cost predictability, Python ecosystem |
| **HNSW index** vs IVFFlat | IVFFlat faster to build | HNSW doesn't require pre-training, better recall, works with incremental data |
| **SSE streaming** vs WebSocket | WebSocket for bidirectional | SSE simpler, HTTP-based, sufficient for server-push only |
| **Multi-region active-active** vs Active-passive | Simpler active-passive | Lower latency for global users, data residency compliance |
| **LLM fallback** vs Single provider | Rely on one provider | Higher availability, price competition, avoid vendor lock-in |

---

### 10. Future Improvements

1. **Semantic Caching**: Cache LLM responses based on semantic similarity of queries (not just exact match)
2. **Personalized Ranking**: Learn user preferences to re-rank search results
3. **Predictive Sync**: Predict which emails/events user will query, prioritize syncing those
4. **Edge Inference**: Run lightweight intent classification at CDN edge for <50ms latency
5. **Federated Search**: Allow users to connect additional data sources (Slack, Notion, etc.)

## External API Management

This section details how we handle external API integrations, including OAuth flows, rate limits, error handling, and user-facing error messages.

---

### 1. Google OAuth 2.0 Flow

#### 1.1 Authorization Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GOOGLE OAUTH 2.0 FLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

     User                    Our App                         Google
       │                        │                               │
       │  1. Click "Login"      │                               │
       │───────────────────────►│                               │
       │                        │                               │
       │                        │  2. Generate auth URL         │
       │                        │     with scopes + state       │
       │                        │                               │
       │  3. Redirect           │                               │
       │◄───────────────────────│                               │
       │                        │                               │
       │  4. User lands on Google consent screen                │
       │────────────────────────────────────────────────────────►
       │                        │                               │
       │                        │        CONSENT SCREEN         │
       │                        │   ┌───────────────────────┐   │
       │                        │   │ App wants to:         │   │
       │                        │   │ ☐ Read your emails    │   │
       │                        │   │ ☐ Manage calendar     │   │
       │                        │   │ ☐ Access Drive files  │   │
       │                        │   │                       │   │
       │                        │   │ [Allow]  [Deny]       │   │
       │                        │   └───────────────────────┘   │
       │                        │                               │
       │                        │  5a. User clicks "Allow"      │
       │◄────────────────────────────────────────────────────────
       │  Redirect with ?code=AUTH_CODE&state=STATE             │
       │                        │                               │
       │  6. Callback           │                               │
       │───────────────────────►│                               │
       │                        │                               │
       │                        │  7. Exchange code for tokens  │
       │                        │──────────────────────────────►│
       │                        │                               │
       │                        │  8. Return access_token,      │
       │                        │     refresh_token, expiry     │
       │                        │◄──────────────────────────────│
       │                        │                               │
       │                        │  9. Store tokens in DB        │
       │                        │     (encrypted)               │
       │                        │                               │
       │  10. Set session cookie│                               │
       │◄───────────────────────│                               │
       │                        │                               │
       │  11. Redirect to app   │                               │
       │◄───────────────────────│                               │
       │      /?auth=success    │                               │
```

#### 1.2 Required Scopes

```python
GOOGLE_SCOPES = [
    # Gmail - read emails, send drafts, manage labels
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.modify",

    # Calendar - full access to events
    "https://www.googleapis.com/auth/calendar",

    # Drive - read files and metadata
    "https://www.googleapis.com/auth/drive.readonly",

    # User info - email and profile
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]
```

#### 1.3 OAuth Error Handling

| Error Scenario | Detection | User Message | Recovery Action |
|----------------|-----------|--------------|-----------------|
| **User denies all permissions** | `error=access_denied` in callback | "You need to grant access to use this app. Please try again and click 'Allow' on the Google consent screen." | Redirect to `/auth/login` |
| **User denies specific scopes** | Check `granted_scopes` vs required | "Some permissions were not granted. The app needs access to [Gmail/Calendar/Drive] to work properly." | Show which features are disabled, offer to re-auth |
| **Authorization code expired** | `invalid_grant` error on token exchange | "The login session expired. Please try again." | Redirect to `/auth/login` |
| **Invalid client credentials** | `invalid_client` error | "There's a configuration issue. Please contact support." | Log error, alert ops team |
| **User revokes access later** | 401 error on API call | "Your Google access has been revoked. Please reconnect your account." | Clear tokens, redirect to `/auth/login` |
| **Refresh token expired/invalid** | `invalid_grant` on refresh | "Your session expired. Please log in again." | Clear tokens, redirect to `/auth/login` |
| **Rate limited during auth** | 429 error | "Too many login attempts. Please wait a moment and try again." | Exponential backoff |

#### 1.4 OAuth Callback Error Handling Implementation

```python
@router.get("/auth/callback")
async def google_callback(
    code: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
    state: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Handle Google OAuth callback with comprehensive error handling."""

    # Case 1: User denied access
    if error == "access_denied":
        return RedirectResponse(
            url="/?auth=denied&message=You need to grant permissions to use this app",
            status_code=302
        )

    # Case 2: Other OAuth errors
    if error:
        logger.error(f"OAuth error: {error} - {error_description}")
        return RedirectResponse(
            url=f"/?auth=error&message={error_description or error}",
            status_code=302
        )

    # Case 3: Missing authorization code
    if not code:
        return RedirectResponse(
            url="/?auth=error&message=No authorization code received",
            status_code=302
        )

    # Case 4: State mismatch (CSRF protection)
    expected_state = session.get("oauth_state")
    if state != expected_state:
        logger.warning(f"State mismatch: expected {expected_state}, got {state}")
        return RedirectResponse(
            url="/?auth=error&message=Security validation failed. Please try again.",
            status_code=302
        )

    try:
        auth_service = GoogleAuthService(db)
        result = await auth_service.handle_callback(code, state)

        # Case 5: Check granted scopes
        granted_scopes = result.get("granted_scopes", [])
        missing_scopes = set(REQUIRED_SCOPES) - set(granted_scopes)

        if missing_scopes:
            # Store partial auth, warn user
            return RedirectResponse(
                url=f"/?auth=partial&missing={','.join(missing_scopes)}",
                status_code=302
            )

        # Success
        response = RedirectResponse(url="/?auth=success", status_code=302)
        response.set_cookie("session_user_id", result["user_id"], httponly=True)
        return response

    except InvalidGrantError:
        # Authorization code expired or already used
        return RedirectResponse(
            url="/?auth=error&message=Login session expired. Please try again.",
            status_code=302
        )

    except Exception as e:
        logger.exception(f"OAuth callback failed: {e}")
        return RedirectResponse(
            url="/?auth=error&message=Login failed. Please try again.",
            status_code=302
        )
```

#### 1.5 Token Refresh Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TOKEN REFRESH FLOW                                   │
└─────────────────────────────────────────────────────────────────────────────┘

  API Request                 Token Check                   Google OAuth
       │                          │                              │
       │  1. Request with         │                              │
       │     user credentials     │                              │
       │─────────────────────────►│                              │
       │                          │                              │
       │                          │  2. Check token expiry       │
       │                          │     (stored in DB)           │
       │                          │                              │
       │                          │  If expired:                 │
       │                          │  3. Use refresh_token        │
       │                          │─────────────────────────────►│
       │                          │                              │
       │                          │  4. New access_token         │
       │                          │◄─────────────────────────────│
       │                          │                              │
       │                          │  5. Update DB with new token │
       │                          │                              │
       │  6. Continue with        │                              │
       │     valid token          │                              │
       │◄─────────────────────────│                              │


  Token Refresh Failure Scenarios:
  ─────────────────────────────────
  • Refresh token revoked → Clear tokens, return 401, user must re-auth
  • Refresh token expired → Clear tokens, return 401, user must re-auth
  • Rate limited → Retry with backoff, fail gracefully
  • Network error → Retry 3 times, then fail gracefully
```

---

### 2. Google API Rate Limits & Quotas

#### 2.1 API Quota Limits

| API | Default Quota | Our Usage Pattern | Strategy |
|-----|---------------|-------------------|----------|
| **Gmail API** | 250 quota units/sec | Sync: 5 units/email, Search: 5 units | Batch requests, delta sync |
| **Calendar API** | 500 queries/sec | Sync: 1 unit/event, Search: 1 unit | Incremental sync tokens |
| **Drive API** | 1000 queries/100 sec | Sync: 1 unit/file, Search: 1 unit | Metadata-only sync |
| **OAuth Token Refresh** | 50 req/sec | On-demand refresh | Cache tokens, refresh proactively |

#### 2.2 Rate Limit Handling Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RATE LIMIT HANDLING PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────────┘

  API Request
       │
       ▼
  ┌─────────────────┐
  │  Pre-flight     │
  │  Rate Check     │
  │  (Redis counter)│
  └────────┬────────┘
           │
    ┌──────┴──────┐
    │ Under limit?│
    └──────┬──────┘
           │
      ┌────┴────┐
     Yes        No
      │          │
      ▼          ▼
  ┌────────┐  ┌────────────────┐
  │ Execute│  │ Queue request  │
  │ Request│  │ (delay 100ms)  │
  └────┬───┘  └────────────────┘
       │
       ▼
  ┌─────────────────┐
  │ Check Response  │
  └────────┬────────┘
           │
    ┌──────┴──────┐
    │ HTTP Status │
    └──────┬──────┘
           │
    ┌──────┼──────┬──────────────┐
    │      │      │              │
   200    429    5xx          Other
    │      │      │              │
    ▼      ▼      ▼              ▼
 Success  Rate   Server       Return
         Limited Error         Error
    │      │      │
    │      ▼      ▼
    │  ┌────────────────────────────┐
    │  │   Exponential Backoff      │
    │  │   ──────────────────────   │
    │  │   Attempt 1: wait 1s       │
    │  │   Attempt 2: wait 2s       │
    │  │   Attempt 3: wait 4s       │
    │  │   Attempt 4: wait 8s       │
    │  │   Attempt 5: wait 16s      │
    │  │   (max 5 retries)          │
    │  │                            │
    │  │   + Jitter: ±25%           │
    │  │   (prevent thundering herd)│
    │  └────────────────────────────┘
    │              │
    │              ▼
    │      ┌──────────────┐
    │      │ Retry limit  │
    │      │ exceeded?    │
    │      └──────┬───────┘
    │             │
    │        ┌────┴────┐
    │       Yes        No
    │        │          │
    │        ▼          └──► Retry
    │   ┌────────────┐
    │   │ Circuit    │
    │   │ Breaker    │
    │   │ OPEN       │
    │   └────────────┘
    │        │
    ▼        ▼
  Return   Return
  Success  Graceful
           Failure
```

#### 2.3 Retry Implementation with Exponential Backoff

```python
import asyncio
import random
from typing import TypeVar, Callable
from functools import wraps

T = TypeVar('T')

class GoogleAPIError(Exception):
    """Base exception for Google API errors."""
    pass

class RateLimitError(GoogleAPIError):
    """Raised when rate limit is hit."""
    def __init__(self, retry_after: int = None):
        self.retry_after = retry_after

class QuotaExceededError(GoogleAPIError):
    """Raised when daily quota is exceeded."""
    pass

class TokenExpiredError(GoogleAPIError):
    """Raised when OAuth token is expired/revoked."""
    pass


def with_retry(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.25,
    retryable_errors: tuple = (RateLimitError, ConnectionError, TimeoutError),
):
    """Decorator for retrying Google API calls with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        jitter: Random jitter factor (±25% by default)
        retryable_errors: Tuple of exception types to retry
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except RateLimitError as e:
                    last_exception = e
                    # Use Retry-After header if provided
                    if e.retry_after:
                        delay = e.retry_after
                    else:
                        delay = min(base_delay * (2 ** attempt), max_delay)

                    # Add jitter
                    delay *= (1 + random.uniform(-jitter, jitter))

                    logger.warning(
                        f"Rate limited, attempt {attempt + 1}/{max_retries + 1}, "
                        f"waiting {delay:.2f}s"
                    )

                    if attempt < max_retries:
                        await asyncio.sleep(delay)

                except QuotaExceededError as e:
                    # Don't retry quota exceeded - it won't help
                    logger.error(f"Daily quota exceeded: {e}")
                    raise

                except TokenExpiredError as e:
                    # Don't retry - user needs to re-authenticate
                    logger.error(f"Token expired: {e}")
                    raise

                except retryable_errors as e:
                    last_exception = e
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    delay *= (1 + random.uniform(-jitter, jitter))

                    logger.warning(
                        f"Retryable error: {e}, attempt {attempt + 1}/{max_retries + 1}, "
                        f"waiting {delay:.2f}s"
                    )

                    if attempt < max_retries:
                        await asyncio.sleep(delay)

            # All retries exhausted
            raise last_exception

        return wrapper
    return decorator


# Usage example
class GmailService:
    @with_retry(max_retries=5, base_delay=1.0)
    async def search_emails(self, query: str) -> list:
        try:
            response = await self._execute_api_call("messages.list", q=query)
            return response.get("messages", [])

        except HttpError as e:
            if e.resp.status == 429:
                retry_after = int(e.resp.get("Retry-After", 0))
                raise RateLimitError(retry_after=retry_after)
            elif e.resp.status == 403 and "quotaExceeded" in str(e):
                raise QuotaExceededError("Daily API quota exceeded")
            elif e.resp.status == 401:
                raise TokenExpiredError("Access token expired or revoked")
            else:
                raise
```

#### 2.4 Circuit Breaker for External APIs

```python
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast, not calling API
    HALF_OPEN = "half_open"  # Testing if API recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for external API calls."""

    name: str
    failure_threshold: int = 5       # Failures before opening
    success_threshold: int = 2       # Successes in half-open before closing
    timeout: float = 30.0            # Seconds before trying half-open
    failure_window: float = 60.0     # Window to count failures

    def __post_init__(self):
        self.state = CircuitState.CLOSED
        self.failures: list[datetime] = []
        self.successes_in_half_open: int = 0
        self.last_failure_time: Optional[datetime] = None
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_try_half_open():
                    self.state = CircuitState.HALF_OPEN
                    self.successes_in_half_open = 0
                else:
                    raise CircuitOpenError(
                        f"Circuit breaker {self.name} is OPEN. "
                        f"Try again in {self._time_until_half_open():.0f}s"
                    )

        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result

        except Exception as e:
            await self._record_failure()
            raise

    async def _record_success(self):
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.successes_in_half_open += 1
                if self.successes_in_half_open >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failures = []

    async def _record_failure(self):
        async with self._lock:
            now = datetime.utcnow()
            self.failures.append(now)
            self.last_failure_time = now

            # Remove old failures outside the window
            cutoff = now - timedelta(seconds=self.failure_window)
            self.failures = [f for f in self.failures if f > cutoff]

            if self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self.state = CircuitState.OPEN

            elif len(self.failures) >= self.failure_threshold:
                self.state = CircuitState.OPEN

    def _should_try_half_open(self) -> bool:
        if not self.last_failure_time:
            return True
        elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout

    def _time_until_half_open(self) -> float:
        if not self.last_failure_time:
            return 0
        elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return max(0, self.timeout - elapsed)


# Per-service circuit breakers
circuit_breakers = {
    "gmail": CircuitBreaker(name="gmail", failure_threshold=5, timeout=30),
    "calendar": CircuitBreaker(name="calendar", failure_threshold=5, timeout=30),
    "drive": CircuitBreaker(name="drive", failure_threshold=5, timeout=30),
    "openai": CircuitBreaker(name="openai", failure_threshold=3, timeout=60),
}
```

---

### 3. Error Scenarios & User Messages

#### 3.1 Error Classification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ERROR CLASSIFICATION                                 │
└─────────────────────────────────────────────────────────────────────────────┘

  Error Type           Retryable?    User Action Required?    Message Style
  ─────────────────────────────────────────────────────────────────────────────
  Rate Limited         Yes           No                       "Moment, please..."
  Server Error (5xx)   Yes           No                       "Trying again..."
  Network Timeout      Yes           No                       "Connection issue..."
  Token Expired        No            Yes (re-auth)            "Please log in again"
  Permission Denied    No            Yes (grant access)       "Need permission to..."
  Quota Exceeded       No            No (wait until reset)    "Limit reached..."
  Invalid Request      No            Yes (fix query)          "I couldn't understand..."
  Not Found            No            No                       "Couldn't find..."
```

#### 3.2 User-Facing Error Messages

```python
ERROR_MESSAGES = {
    # Authentication errors
    "token_expired": {
        "title": "Session Expired",
        "message": "Your Google session has expired. Please log in again to continue.",
        "action": "Login Again",
        "action_url": "/api/v1/auth/login",
    },
    "permission_denied": {
        "title": "Permission Required",
        "message": "This action requires additional permissions. Please reconnect your Google account and grant access to {service}.",
        "action": "Reconnect Account",
        "action_url": "/api/v1/auth/login?prompt=consent",
    },
    "access_revoked": {
        "title": "Access Revoked",
        "message": "You've revoked access to your Google account. Please reconnect to continue using the app.",
        "action": "Reconnect Account",
        "action_url": "/api/v1/auth/login",
    },

    # Rate limit errors
    "rate_limited": {
        "title": "Please Wait",
        "message": "We're processing too many requests right now. Please wait a moment and try again.",
        "action": "Try Again",
        "retry_after": 5,  # seconds
    },
    "quota_exceeded": {
        "title": "Daily Limit Reached",
        "message": "We've reached our daily API limit. This will reset at midnight UTC. Sorry for the inconvenience!",
        "action": None,
    },

    # Service errors
    "gmail_unavailable": {
        "title": "Gmail Temporarily Unavailable",
        "message": "We couldn't connect to Gmail right now. Your calendar and drive searches still worked. Would you like to try the email search again?",
        "action": "Retry Email Search",
        "partial_results": True,
    },
    "calendar_unavailable": {
        "title": "Calendar Temporarily Unavailable",
        "message": "We couldn't connect to Google Calendar right now. Your email search still worked. Would you like to try the calendar search again?",
        "action": "Retry Calendar Search",
        "partial_results": True,
    },

    # LLM errors
    "llm_unavailable": {
        "title": "AI Processing Delayed",
        "message": "Our AI service is experiencing high load. We're processing your request, but it may take a bit longer than usual.",
        "action": None,
        "fallback": True,
    },

    # Generic errors
    "unknown_error": {
        "title": "Something Went Wrong",
        "message": "We encountered an unexpected error. Our team has been notified. Please try again in a few moments.",
        "action": "Try Again",
    },
}
```

#### 3.3 Graceful Degradation Examples

```python
async def execute_multi_service_query(query: str, services: list[str]) -> QueryResponse:
    """Execute query with graceful degradation on partial failures."""

    results = {}
    errors = {}
    partial_success = False

    for service in services:
        try:
            circuit = circuit_breakers[service]
            result = await circuit.call(
                execute_service_query,
                service=service,
                query=query
            )
            results[service] = result
            partial_success = True

        except CircuitOpenError as e:
            errors[service] = {
                "type": "circuit_open",
                "message": f"{service.title()} is temporarily unavailable",
                "retry_after": e.retry_after,
            }

        except TokenExpiredError:
            errors[service] = {
                "type": "auth_required",
                "message": "Please reconnect your Google account",
            }

        except RateLimitError as e:
            errors[service] = {
                "type": "rate_limited",
                "message": "Too many requests, please wait",
                "retry_after": e.retry_after,
            }

        except Exception as e:
            logger.exception(f"Error in {service}: {e}")
            errors[service] = {
                "type": "error",
                "message": f"Error searching {service}",
            }

    # Synthesize response with available results
    if partial_success:
        response = await synthesizer.generate(
            results=results,
            errors=errors,
            context=f"Note: {', '.join(errors.keys())} could not be searched."
        )
    else:
        response = synthesizer.generate_error_response(errors)

    return QueryResponse(
        response=response,
        partial=len(errors) > 0,
        services_failed=list(errors.keys()),
        retry_available=any(e.get("retry_after") for e in errors.values()),
    )
```

---

### 4. LLM API Management

#### 4.1 Provider Failover

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LLM PROVIDER FAILOVER                                │
└─────────────────────────────────────────────────────────────────────────────┘

                        ┌─────────────┐
                        │  LLM Request │
                        └──────┬──────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │   Primary:       │
                    │   OpenAI GPT-4   │
                    └────────┬─────────┘
                             │
                      ┌──────┴──────┐
                      │   Success?   │
                      └──────┬──────┘
                             │
                    ┌────────┴────────┐
                   Yes               No
                    │                 │
                    ▼                 ▼
               ┌────────┐      ┌─────────────────┐
               │ Return │      │   Check Error   │
               │ Result │      │   Type          │
               └────────┘      └────────┬────────┘
                                        │
                          ┌─────────────┼─────────────┐
                          │             │             │
                       Rate Limit    Server Error   Auth Error
                          │             │             │
                          ▼             ▼             ▼
                    ┌──────────┐  ┌──────────┐  ┌──────────┐
                    │ Retry w/ │  │ Fallback │  │ Return   │
                    │ Backoff  │  │ Provider │  │ Error    │
                    └──────────┘  └────┬─────┘  └──────────┘
                                       │
                                       ▼
                            ┌──────────────────┐
                            │   Fallback:      │
                            │   Anthropic      │
                            │   Claude         │
                            └────────┬─────────┘
                                     │
                              ┌──────┴──────┐
                              │   Success?   │
                              └──────┬──────┘
                                     │
                            ┌────────┴────────┐
                           Yes               No
                            │                 │
                            ▼                 ▼
                       ┌────────┐      ┌─────────────┐
                       │ Return │      │ Use Cached  │
                       │ Result │      │ Response or │
                       └────────┘      │ Error       │
                                       └─────────────┘
```

#### 4.2 LLM Error Handling

```python
class LLMService:
    """LLM service with provider failover and caching."""

    def __init__(self):
        self.primary = OpenAIProvider()
        self.fallback = AnthropicProvider()
        self.circuit = CircuitBreaker(name="llm", failure_threshold=3, timeout=60)
        self.cache = ResponseCache(ttl=300)  # 5 min cache

    @with_retry(max_retries=2, base_delay=1.0)
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response with failover."""

        # Check cache first
        cache_key = hash_prompt(prompt, **kwargs)
        cached = await self.cache.get(cache_key)
        if cached:
            return cached

        try:
            # Try primary provider
            response = await self.circuit.call(
                self.primary.generate,
                prompt=prompt,
                **kwargs
            )
            await self.cache.set(cache_key, response)
            return response

        except (CircuitOpenError, RateLimitError, ProviderError) as e:
            logger.warning(f"Primary LLM failed: {e}, trying fallback")

            try:
                # Try fallback provider
                response = await self.fallback.generate(prompt=prompt, **kwargs)
                await self.cache.set(cache_key, response)
                return response

            except Exception as fallback_error:
                logger.error(f"Fallback LLM also failed: {fallback_error}")

                # Last resort: return cached similar response or error
                similar = await self.cache.get_similar(prompt)
                if similar:
                    return similar + "\n\n(Note: This is a cached response)"

                raise LLMUnavailableError(
                    "AI service is temporarily unavailable. Please try again."
                )
```

---

### 5. Monitoring External API Health

#### 5.1 Health Check Dashboard

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EXTERNAL API HEALTH DASHBOARD                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  GOOGLE APIS                              LLM PROVIDERS                      │
│  ───────────                              ─────────────                      │
│                                                                              │
│  Gmail API                                OpenAI                             │
│  Status: ● Healthy                        Status: ● Healthy                  │
│  Latency: 85ms (p50)                      Latency: 180ms (p50)               │
│  Errors: 0.1%                             Errors: 0.2%                       │
│  Rate: 45 req/s                           Rate: 120 req/s                    │
│  Quota: 12% used                          Tokens: 850K/min                   │
│                                                                              │
│  Calendar API                             Anthropic (fallback)               │
│  Status: ● Healthy                        Status: ● Healthy                  │
│  Latency: 62ms (p50)                      Latency: 220ms (p50)               │
│  Errors: 0.0%                             Errors: 0.0%                       │
│  Rate: 28 req/s                           Rate: 5 req/s (fallback only)      │
│                                                                              │
│  Drive API                                                                   │
│  Status: ⚠ Degraded                       CIRCUIT BREAKERS                   │
│  Latency: 340ms (p50) ▲                   ────────────────                   │
│  Errors: 2.1% ▲                           gmail:    CLOSED ●                 │
│  Rate: 15 req/s                           calendar: CLOSED ●                 │
│  Quota: 8% used                           drive:    HALF-OPEN ◐             │
│                                           openai:   CLOSED ●                 │
│                                                                              │
│  RECENT INCIDENTS                                                            │
│  ────────────────                                                            │
│  • 14:32 UTC - Drive API latency spike (resolved)                           │
│  • 14:28 UTC - Drive circuit breaker opened (5 failures in 60s)             │
│  • 14:30 UTC - Drive circuit breaker half-open (testing recovery)           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 5.2 Alerting Rules for External APIs

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| API Error Rate High | >1% errors for 5 min | P2 | Investigate, check Google status |
| API Latency Spike | p95 > 2s for 5 min | P3 | Monitor, may indicate quota issues |
| Circuit Breaker Open | Any breaker opens | P2 | Check API status, may need to scale back |
| Quota Warning | >80% daily quota used | P3 | Alert team, consider request throttling |
| Token Refresh Failures | >10 failures/min | P2 | Check OAuth config, may affect users |
| LLM Provider Down | Primary + fallback down | P1 | Critical - core functionality impaired |
| All Google APIs Down | All 3 circuits open | P1 | Critical - check Google Cloud status |

## Security Considerations

### Authentication
- OAuth 2.0 with Google
- JWT tokens for API authentication
- Token refresh handling
- Secure token storage (encrypted at rest)
- Session management with httponly cookies

### Data Protection
- User data isolation (all queries filtered by user_id)
- Encrypted OAuth tokens in database
- Audit logging for all operations
- No storage of email body content (only previews)

### Rate Limiting
- Per-user query limits (100/hour free, 1000/hour pro)
- API key quotas for external APIs
- DDoS protection at load balancer
- Circuit breakers prevent cascade failures

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

### Production (AWS ECS with Spot Instances)

We use AWS ECS for production deployment - simpler than Kubernetes, cost-effective with Spot instances.

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         AWS Production Architecture                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Route 53 (DNS)                                                                │
│       │                                                                         │
│       ▼                                                                         │
│   CloudFront (CDN) ──► S3 (Static Assets)                                       │
│       │                                                                         │
│       ▼                                                                         │
│   Application Load Balancer                                                     │
│       │                                                                         │
│       ▼                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    ECS Cluster (Fargate Spot)                            │   │
│   │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐            │   │
│   │  │   API Service   │ │  Celery Worker  │ │  Celery Beat    │            │   │
│   │  │   (2-10 tasks)  │ │   (2-20 tasks)  │ │   (1 task)      │            │   │
│   │  │   Spot: 70%     │ │   Spot: 100%    │ │   On-Demand     │            │   │
│   │  │   On-Demand:30% │ │                 │ │                 │            │   │
│   │  └─────────────────┘ └─────────────────┘ └─────────────────┘            │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│       │                       │                                                 │
│       ▼                       ▼                                                 │
│   ┌───────────────┐   ┌───────────────┐                                         │
│   │  RDS Postgres │   │  ElastiCache  │                                         │
│   │  (pgvector)   │   │   (Redis)     │                                         │
│   │  Multi-AZ     │   │   Cluster     │                                         │
│   └───────────────┘   └───────────────┘                                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### ECS Task Definitions

**API Service** (mixed On-Demand + Spot for availability):
```json
{
  "family": "orchestrator-api",
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [{
    "name": "api",
    "image": "${ECR_REPO}/orchestrator:${VERSION}",
    "command": ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
    "portMappings": [{"containerPort": 8000}],
    "environment": [
      {"name": "DATABASE_URL", "value": "${DB_URL}"},
      {"name": "REDIS_URL", "value": "${REDIS_URL}"}
    ],
    "secrets": [
      {"name": "OPENAI_API_KEY", "valueFrom": "${SSM_OPENAI_KEY}"},
      {"name": "GOOGLE_CLIENT_SECRET", "valueFrom": "${SSM_GOOGLE_SECRET}"}
    ],
    "healthCheck": {
      "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
      "interval": 30,
      "timeout": 5,
      "retries": 3
    }
  }]
}
```

**Celery Worker** (100% Spot - interruption-tolerant):
```json
{
  "family": "orchestrator-worker",
  "cpu": "256",
  "memory": "512",
  "containerDefinitions": [{
    "name": "worker",
    "image": "${ECR_REPO}/orchestrator:${VERSION}",
    "command": ["celery", "-A", "app.celery_app", "worker",
                "--loglevel=info", "--concurrency=4"],
    "environment": [
      {"name": "DATABASE_URL", "value": "${DB_URL}"},
      {"name": "REDIS_URL", "value": "${REDIS_URL}"}
    ]
  }]
}
```

#### Auto-Scaling Configuration

```yaml
# API Service: Scale on CPU/Request count
ApiScalingTarget:
  Type: AWS::ApplicationAutoScaling::ScalableTarget
  Properties:
    ServiceNamespace: ecs
    ScalableDimension: ecs:service:DesiredCount
    MinCapacity: 2
    MaxCapacity: 10

ApiScalingPolicy:
  Type: AWS::ApplicationAutoScaling::ScalingPolicy
  Properties:
    PolicyType: TargetTrackingScaling
    TargetTrackingScalingPolicyConfiguration:
      TargetValue: 70  # 70% CPU utilization
      PredefinedMetricSpecification:
        PredefinedMetricType: ECSServiceAverageCPUUtilization
      ScaleInCooldown: 300
      ScaleOutCooldown: 60

# Worker Service: Scale on queue depth
WorkerScalingPolicy:
  Type: AWS::ApplicationAutoScaling::ScalingPolicy
  Properties:
    PolicyType: TargetTrackingScaling
    TargetTrackingScalingPolicyConfiguration:
      TargetValue: 100  # 100 tasks per worker
      CustomizedMetricSpecification:
        MetricName: CeleryQueueLength
        Namespace: Orchestrator
        Statistic: Average
      ScaleInCooldown: 300
      ScaleOutCooldown: 60
```

#### Spot Instance Strategy

**Why Spot for Celery Workers:**
- Workers are stateless - can be interrupted anytime
- Celery tasks auto-retry on worker death
- 60-90% cost savings vs On-Demand
- Redis queue persists tasks during interruption

**Capacity Provider Strategy:**
```yaml
CapacityProviderStrategy:
  # API: 70% Spot, 30% On-Demand for stability
  - CapacityProvider: FARGATE_SPOT
    Weight: 7
    Base: 0
  - CapacityProvider: FARGATE
    Weight: 3
    Base: 1  # Always have 1 On-Demand for availability

WorkerCapacityProviderStrategy:
  # Workers: 100% Spot
  - CapacityProvider: FARGATE_SPOT
    Weight: 1
    Base: 0
```

#### Cost Comparison (1M Users)

| Resource | On-Demand/Month | With Spot | Savings |
|----------|-----------------|-----------|---------|
| API (4 tasks avg) | $150 | $75 | 50% |
| Workers (10 tasks avg) | $200 | $40 | 80% |
| RDS db.r6g.large | $200 | $200 | - |
| ElastiCache | $100 | $100 | - |
| **Total** | **$650** | **$415** | **36%** |

#### Deployment Commands

```bash
# Build and push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REPO
docker build -t orchestrator:$VERSION .
docker tag orchestrator:$VERSION $ECR_REPO/orchestrator:$VERSION
docker push $ECR_REPO/orchestrator:$VERSION

# Update ECS service (rolling deployment)
aws ecs update-service \
  --cluster orchestrator-prod \
  --service api \
  --force-new-deployment

# Run migrations (one-off task)
aws ecs run-task \
  --cluster orchestrator-prod \
  --task-definition orchestrator-migrate \
  --launch-type FARGATE
```

#### Monitoring (CloudWatch)

```yaml
# Key metrics to monitor
Alarms:
  - Name: HighCPU
    Metric: CPUUtilization
    Threshold: 80
    Period: 300

  - Name: HighQueueDepth
    Metric: CeleryQueueLength
    Threshold: 1000
    Period: 60

  - Name: HighErrorRate
    Metric: 5xxErrors
    Threshold: 10
    Period: 60

  - Name: SpotInterruption
    Metric: SpotInstanceInterruptionWarning
    # Alert to monitor Spot interruption frequency
```

#### Infrastructure as Code

We use Terraform for infrastructure:

```hcl
# terraform/main.tf
module "ecs_cluster" {
  source = "./modules/ecs"

  cluster_name = "orchestrator-prod"

  services = {
    api = {
      cpu    = 512
      memory = 1024
      desired_count = 2
      spot_weight   = 7
      ondemand_weight = 3
    }
    worker = {
      cpu    = 256
      memory = 512
      desired_count = 4
      spot_weight   = 10
      ondemand_weight = 0
    }
    beat = {
      cpu    = 256
      memory = 512
      desired_count = 1
      spot_weight   = 0
      ondemand_weight = 1  # Beat must be On-Demand (singleton)
    }
  }
}

module "rds" {
  source = "./modules/rds"

  instance_class = "db.r6g.large"
  engine         = "postgres"
  engine_version = "15"
  multi_az       = true

  # Enable pgvector extension
  parameter_group_params = {
    shared_preload_libraries = "pg_stat_statements,pgvector"
  }
}

module "elasticache" {
  source = "./modules/elasticache"

  node_type       = "cache.r6g.large"
  num_cache_nodes = 2
  engine          = "redis"
}
```
