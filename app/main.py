"""Main FastAPI application for Google Workspace Orchestrator."""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

# Configure logging - must be done before importing other app modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
# Set specific loggers to INFO level
logging.getLogger("app").setLevel(logging.INFO)
logging.getLogger("app.core.intent").setLevel(logging.INFO)
logging.getLogger("app.api.routes").setLevel(logging.INFO)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.middleware.sessions import SessionMiddleware

from app.api.routes import router
from app.api.dependencies import get_cache_service
from app.config import settings

# Get the directory of this file
BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    # Startup
    yield
    # Shutdown
    cache = get_cache_service()
    await cache.close()


app = FastAPI(
    title="Google Workspace Orchestrator",
    description="""
    An intelligent orchestrator that executes natural language queries
    across Gmail, Google Calendar, and Google Drive.

    ## Features

    - **Intent Classification**: Parse natural language into structured intents
    - **Multi-Service Orchestration**: Execute queries across Gmail, Calendar, and Drive
    - **Semantic Search**: Find relevant emails, events, and files using embeddings
    - **Natural Language Responses**: Generate conversational responses

    ## Example Queries

    - "What's on my calendar next week?"
    - "Find emails from sarah@company.com about the budget"
    - "Cancel my Turkish Airlines flight"
    - "Prepare for tomorrow's meeting with Acme Corp"
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Add session middleware (must be added before CORS)
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.secret_key,
    session_cookie="session",
    max_age=86400,  # 24 hours
    same_site="lax",
    https_only=False,  # Set to True in production with HTTPS
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["orchestrator"])

# Mount static files
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", tags=["root"])
async def root():
    """Serve the main UI."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "name": "Google Workspace Orchestrator",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
