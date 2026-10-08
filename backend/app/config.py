"""Centralized Application Configuration.

Loads settings from environment variables or .env file with type validation
via Pydantic Settings.
"""

from functools import lru_cache
from pathlib import Path
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory of the repository
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings schema with default development values."""

    # Application settings
    APP_NAME: str = "MahaTraffic AI"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # OpenRouter / LLM Configuration
    OPENROUTER_API_KEY: str = Field(default="", description="OpenRouter API key for LLM agents")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_MODEL: str = "anthropic/claude-3.5-sonnet"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048

    # MongoDB Configuration
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "mahatraffic_ai"

    # Vector Storage & RAG
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    FAISS_INDEX_PATH: str = "rag/index/road_safety_faiss.index"
    DOCUMENTS_PATH: str = "rag/documents"

    # MCP Server
    MCP_SERVER_HOST: str = "127.0.0.1"
    MCP_SERVER_PORT: int = 8001
    MCP_ALLOWED_TOOLS: str = (
        "get_city_accident_statistics,"
        "get_monthly_accident_statistics,"
        "get_yearly_accident_statistics,"
        "calculate_risk_score,"
        "predict_risk,"
        "analyze_sentiment,"
        "get_social_trends,"
        "search_road_safety_documents"
    )

    # Big Data & Spark
    SPARK_MASTER: str = "local[*]"
    SPARK_APP_NAME: str = "MahaTrafficAI-SparkAnalytics"
    PARQUET_DATA_PATH: str = "data/processed/parquet"

    # Machine Learning
    ML_MODEL_PATH: str = "ml/models/random_forest_risk.joblib"

    # Reliability & Execution Tracing
    RELIABILITY_LOGS_PATH: str = "reliability/logs"
    EXECUTION_TIMEOUT_SECONDS: int = 60
    MAX_TOOL_RETRIES: int = 2

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @property
    def allowed_tools_list(self) -> List[str]:
        """Parse comma-separated MCP tools allowlist into a list."""
        return [tool.strip() for tool in self.MCP_ALLOWED_TOOLS.split(",") if tool.strip()]


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
