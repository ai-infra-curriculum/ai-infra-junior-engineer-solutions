#!/usr/bin/env python3
"""
Configuration Management for Model API

Manages application configuration using environment variables with validation
and type safety.

Usage:
    from config import settings
    print(settings.model_name)
"""

import os
from typing import Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    """Application settings with validation"""

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    workers: int = 4

    # Model Configuration
    model_name: str = "resnet50"  # Options: resnet50, mobilenet_v2
    model_device: str = "cpu"  # Options: cpu, cuda
    batch_size: int = 1

    # API Configuration
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    request_timeout: int = 30
    max_concurrent_requests: int = 100

    # Logging Configuration
    log_level: str = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_format: str = "json"  # Options: json, text

    # Performance Configuration
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour

    def __post_init__(self):
        """Validate settings after initialization"""
        self._load_from_env()
        self._validate()

    def _load_from_env(self):
        """Load settings from environment variables"""
        # Server settings
        self.host = os.getenv("API_HOST", self.host)
        self.port = int(os.getenv("API_PORT", self.port))
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.workers = int(os.getenv("WORKERS", self.workers))

        # Model settings
        self.model_name = os.getenv("MODEL_NAME", self.model_name).lower()
        self.model_device = os.getenv("MODEL_DEVICE", self.model_device).lower()
        self.batch_size = int(os.getenv("BATCH_SIZE", self.batch_size))

        # API settings
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", self.max_file_size))
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", self.request_timeout))
        self.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", self.max_concurrent_requests))

        # Logging settings
        self.log_level = os.getenv("LOG_LEVEL", self.log_level).upper()
        self.log_format = os.getenv("LOG_FORMAT", self.log_format).lower()

        # Performance settings
        self.enable_caching = os.getenv("ENABLE_CACHING", "true").lower() == "true"
        self.cache_ttl = int(os.getenv("CACHE_TTL", self.cache_ttl))

    def _validate(self):
        """Validate settings"""
        # Validate model name
        valid_models = ["resnet50", "mobilenet_v2"]
        if self.model_name not in valid_models:
            raise ValueError(f"Invalid model_name: {self.model_name}. Must be one of {valid_models}")

        # Validate device
        valid_devices = ["cpu", "cuda"]
        if self.model_device not in valid_devices:
            raise ValueError(f"Invalid model_device: {self.model_device}. Must be one of {valid_devices}")

        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise ValueError(f"Invalid log_level: {self.log_level}. Must be one of {valid_log_levels}")

        # Validate port
        if not (1 <= self.port <= 65535):
            raise ValueError(f"Invalid port: {self.port}. Must be between 1 and 65535")

        # Validate batch size
        if self.batch_size < 1:
            raise ValueError(f"Invalid batch_size: {self.batch_size}. Must be >= 1")

        # Validate workers
        if self.workers < 1:
            raise ValueError(f"Invalid workers: {self.workers}. Must be >= 1")

    def to_dict(self) -> dict:
        """Convert settings to dictionary"""
        return {
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "workers": self.workers,
            "model_name": self.model_name,
            "model_device": self.model_device,
            "batch_size": self.batch_size,
            "max_file_size": self.max_file_size,
            "allowed_extensions": self.allowed_extensions,
            "request_timeout": self.request_timeout,
            "max_concurrent_requests": self.max_concurrent_requests,
            "log_level": self.log_level,
            "log_format": self.log_format,
            "enable_caching": self.enable_caching,
            "cache_ttl": self.cache_ttl,
        }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance"""
    return settings
