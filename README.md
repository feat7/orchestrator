# Google Workspace Orchestrator

An intelligent orchestrator that executes natural language queries across Gmail, Google Calendar, and Google Drive.

## Features

- **Intent Classification**: LLM-powered parsing of natural language into structured intents
- **Multi-Service Orchestration**: Execute queries across Gmail, Calendar, and Drive in parallel
- **Semantic Search**: Vector similarity search using pgvector for relevant results
- **Natural Language Responses**: Conversational responses synthesized by LLM

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- OpenAI API key (or Anthropic)

### Setup

1. **Clone and configure:**
   ```bash
   cd dalat-v1
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Start services:**
   ```bash
   docker-compose up -d db redis
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run migrations:**
   ```bash
   alembic upgrade head
   ```

5. **Seed mock data:**
   ```bash
   python -m scripts.seed_mock_data
   ```

6. **Start the API:**
   ```bash
   uvicorn app.main:app --reload
   ```

7. **Try a query:**
   ```bash
   curl -X POST http://localhost:8000/api/v1/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What'\''s on my calendar next week?"}'
   ```

## Project Structure

```
dalat-v1/
├── app/
│   ├── api/              # API routes and dependencies
│   ├── core/             # Core logic (intent, planner, orchestrator)
│   ├── agents/           # Service agents (gmail, gcal, gdrive)
│   ├── services/         # Google services and utilities
│   ├── db/               # Database models and connection
│   └── schemas/          # Pydantic schemas
├── alembic/              # Database migrations
├── tests/                # Test suite
├── docs/                 # Documentation
└── scripts/              # Utility scripts
```

## Sample Queries

**Single Service:**
- "What's on my calendar next week?"
- "Find emails from sarah@company.com about the budget"
- "Show me PDFs in Drive from last month"

**Multi-Service:**
- "Cancel my Turkish Airlines flight"
- "Prepare for tomorrow's meeting with Acme Corp"
- "Find events that conflict with my out-of-office doc"

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/query` | Process natural language query |
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/sync/trigger` | Trigger data sync |
| GET | `/api/v1/sync/status` | Get sync status |

See [API Documentation](docs/API.md) for details.

## Architecture

```
Query → Intent Classifier → Query Planner → Orchestrator → Response
                                    ↓
                    [Gmail, Calendar, Drive Agents]
                                    ↓
                         pgvector (Semantic Search)
```

See [Design Documentation](docs/DESIGN.md) for scaling strategy.

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | - |
| `REDIS_URL` | Redis connection string | - |
| `LLM_PROVIDER` | "openai" or "anthropic" | openai |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `USE_MOCK_GOOGLE` | Use mock Google services | true |

## Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_intent.py
```

## Development

```bash
# Format code
black app tests

# Type checking
mypy app

# Lint
ruff app
```

## Restrictions

Per assignment requirements:
- No LangChain, LlamaIndex, or agent frameworks
- No managed vector DBs (Pinecone, etc.)
- Orchestration built from scratch
- Using pgvector for vector search

## Documentation

- [System Design](docs/DESIGN.md) - Architecture and scaling
- [API Reference](docs/API.md) - Endpoint documentation
- [Sample Queries](docs/sample_queries.md) - Test cases with expected outputs

## License

MIT
