from pydantic_settings import BaseSettings
from pydantic import ConfigDict, field_validator
from typing import Optional, List
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings from environment variables"""

    # ================================
    # APPLICATION SETTINGS
    # ================================
    app_name: str = "Brevity API Server"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"

    # ================================
    # SERVER SETTINGS
    # ================================
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # ================================
    # API SETTINGS
    # ================================
    api_v1_prefix: str = "/api/v1"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"

    # ================================
    # NEWS API SETTINGS
    # ================================
    news_api_key: str
    news_api_base_url: str = "https://newsapi.org/v2"
    news_fetch_days: int = 5
    news_max_articles: int = 500
    news_batch_size: int = 5

    # ================================
    # AWS DYNAMODB SETTINGS
    # ================================
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_default_region: str = "us-east-2"  # Should be "us-east-2" for your setup
    dynamodb_table_name: str

    # ================================
    # AI SUMMARIZATION SETTINGS
    # ================================
    summarizer_model: str = "sshleifer/distilbart-cnn-12-6"
    summarizer_concurrency: int = 3
    summarizer_max_length: int = 250
    summarizer_min_length: int = 150
    summarizer_content_min_chars: int = 500
    summarizer_max_chars: int = 7000

    # ================================
    # ARTICLE PROCESSING SETTINGS
    # ================================
    article_ttl_days: int = 30
    content_extraction_timeout: int = 30
    batch_processing_delay: float = 1.0

    # ================================
    # CORS SETTINGS
    # ================================
    cors_origins: List[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]

    # ================================
    # REDIS SETTINGS
    # ================================
    redis_url: str = "redis://localhost:6379"
    session_ttl: int = 30 * 60  # 30 minutes in seconds
    # RAG SYSTEM SETTINGS
    # ================================
    ollama_host: str = "http://localhost:11434"

    #======================
    # Web Searcher Settings
    #======================
    tavily_api_key: str
    # ======================

    #======================
    # OpenAI Settings
    #======================
    openai_key: str
    # ======================

    # ================================
    # LOGGING SETTINGS
    # ================================
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None

    # Pydantic v2 configuration
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra environment variables
    )

    # Field validators for list parsing
    @field_validator('cors_origins', 'cors_allow_methods', 'cors_allow_headers')
    @classmethod
    def parse_cors_lists(cls, v):
        if isinstance(v, str):
            return [x.strip() for x in v.split(",")]
        return v

    def get_database_config(self) -> dict:
        """Get database configuration dictionary"""
        return {
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key,
            "region_name": self.aws_default_region,
            "table_name": self.dynamodb_table_name
        }

    def get_news_api_config(self) -> dict:
        """Get News API configuration dictionary"""
        return {
            "api_key": self.news_api_key,
            "base_url": self.news_api_base_url,
            "fetch_days": self.news_fetch_days,
            "max_articles": self.news_max_articles,
            "batch_size": self.news_batch_size
        }

    def get_summarizer_config(self) -> dict:
        """Get AI summarizer configuration dictionary"""
        return {
            "model_name": self.summarizer_model,
            "concurrency": self.summarizer_concurrency,
            "max_length": self.summarizer_max_length,
            "min_length": self.summarizer_min_length,
            "content_min_chars": self.summarizer_content_min_chars,
            "max_chars": self.summarizer_max_chars
        }


# Global settings instance
settings = Settings()


def validate_settings():
    """Validate critical settings are present"""
    required_fields = [
        "news_api_key",
        "aws_access_key_id",
        "aws_secret_access_key",
        "dynamodb_table_name",
        "tavily_api_key",
        "openai_key"
    ]

    missing_fields = []
    for field in required_fields:
        if not getattr(settings, field, None):
            missing_fields.append(field.upper())

    if missing_fields:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_fields)}")

    return True
