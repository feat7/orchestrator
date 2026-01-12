from pydantic_settings import BaseSettings
from typing import Literal, Optional


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://orchestrator:orchestrator@localhost:5432/orchestrator"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Session
    secret_key: str = "change-me-in-production-use-a-real-secret-key"

    # LLM Provider
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Embedding settings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Google API settings
    # Master switch - if True, all services use mock data
    use_mock_google: bool = True

    # Per-service mock settings (override master switch when set to False)
    # These allow using mock data for some services while using real APIs for others
    # Only effective when use_mock_google is False
    use_mock_gmail: Optional[bool] = None  # None = follow use_mock_google
    use_mock_gcal: Optional[bool] = None   # None = follow use_mock_google
    use_mock_gdrive: Optional[bool] = None # None = follow use_mock_google

    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8000/api/v1/auth/callback"

    @property
    def is_gmail_mock(self) -> bool:
        """Check if Gmail should use mock data."""
        if self.use_mock_google:
            return True
        return self.use_mock_gmail if self.use_mock_gmail is not None else False

    @property
    def is_gcal_mock(self) -> bool:
        """Check if Google Calendar should use mock data."""
        if self.use_mock_google:
            return True
        return self.use_mock_gcal if self.use_mock_gcal is not None else False

    @property
    def is_gdrive_mock(self) -> bool:
        """Check if Google Drive should use mock data."""
        if self.use_mock_google:
            return True
        return self.use_mock_gdrive if self.use_mock_gdrive is not None else False

    # Google OAuth scopes
    google_scopes: list = [
        "openid",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.compose",
        "https://www.googleapis.com/auth/gmail.modify",
        "https://www.googleapis.com/auth/calendar.readonly",
        "https://www.googleapis.com/auth/calendar.events",
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/drive.file",
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
