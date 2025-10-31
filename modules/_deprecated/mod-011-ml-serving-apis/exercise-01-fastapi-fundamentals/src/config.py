"""Application configuration."""

import os
from typing import List
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # API Configuration
    API_TITLE: str = "ML Prediction API"
    API_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"

    # Model Configuration
    MODEL_PATH: str = "models/model.pkl"
    MODEL_TYPE: str = "random_forest"
    MODEL_VERSION: str = "v1.0.0"

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4

    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
