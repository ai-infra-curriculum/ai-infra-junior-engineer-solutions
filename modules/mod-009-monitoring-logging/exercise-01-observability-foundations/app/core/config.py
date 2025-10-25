"""
Configuration management using Pydantic Settings.

All configuration is loaded from environment variables with sensible defaults.
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="inference-gateway", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(default="development", description="Environment (development, staging, production)")
    debug: bool = Field(default=False, description="Debug mode")

    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=4, description="Number of worker processes")
    reload: bool = Field(default=False, description="Auto-reload on code changes")

    # Logging
    log_level: str = Field(default="INFO", description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    log_format: str = Field(default="json", description="Log format (json or console)")

    # Model Configuration
    model_name: str = Field(default="resnet50", description="PyTorch model name")
    model_warmup: bool = Field(default=True, description="Warmup model on startup")
    model_device: str = Field(default="cpu", description="Model device (cpu or cuda)")

    # Performance
    max_queue_size: int = Field(default=100, description="Maximum inference queue size")
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    inference_timeout: int = Field(default=10, description="Inference timeout in seconds")

    # Observability - Metrics
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=8000, description="Prometheus metrics port (same as app port)")

    # Observability - Tracing
    enable_tracing: bool = Field(default=True, description="Enable OpenTelemetry tracing")
    otel_service_name: str = Field(default="inference-gateway", description="OpenTelemetry service name")
    otel_exporter_otlp_endpoint: Optional[str] = Field(
        default="http://jaeger:4318",
        description="OTLP exporter endpoint"
    )
    otel_exporter_otlp_insecure: bool = Field(default=True, description="Use insecure OTLP connection")
    otel_traces_sampler: str = Field(default="always_on", description="Trace sampling strategy")

    # SLO Targets
    slo_availability_target: float = Field(default=99.5, description="Availability SLO target (%)")
    slo_latency_p99_target_ms: int = Field(default=300, description="P99 latency SLO target (ms)")
    slo_latency_p95_target_ms: int = Field(default=200, description="P95 latency SLO target (ms)")

    # CORS
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    cors_allow_credentials: bool = Field(default=True, description="Allow CORS credentials")
    cors_allow_methods: list[str] = Field(default=["*"], description="Allowed CORS methods")
    cors_allow_headers: list[str] = Field(default=["*"], description="Allowed CORS headers")


# Global settings instance
settings = Settings()
