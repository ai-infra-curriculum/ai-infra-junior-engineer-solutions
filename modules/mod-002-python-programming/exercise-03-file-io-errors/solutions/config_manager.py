#!/usr/bin/env python3
"""
Configuration Management System

A robust system for managing application configuration with validation,
environment support, hot-reloading, and comprehensive error handling.

Features:
- Load configs from YAML/JSON
- Validate configuration schema
- Environment-specific configs (dev, staging, prod)
- Configuration hot-reloading
- Default values and inheritance
- Comprehensive logging and error handling
"""

import yaml
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from dataclasses import dataclass, field
from threading import Thread, Lock
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pydantic import BaseModel, ValidationError, Field


# ===========================
# Configure Logging
# ===========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('config_manager.log')
    ]
)

logger = logging.getLogger(__name__)


# ===========================
# Custom Exceptions
# ===========================

class ConfigError(Exception):
    """Base exception for configuration errors."""

    def __init__(self, message: str, config_file: Optional[str] = None):
        super().__init__(message)
        self.config_file = config_file
        self.timestamp = datetime.now()

    def __str__(self):
        msg = f"{self.__class__.__name__}: {self.args[0]}"
        if self.config_file:
            msg += f"\nFile: {self.config_file}"
        msg += f"\nTime: {self.timestamp}"
        return msg


class ConfigNotFoundError(ConfigError):
    """Configuration file not found."""
    pass


class ConfigParseError(ConfigError):
    """Configuration parsing failed."""
    pass


class ConfigValidationError(ConfigError):
    """Configuration validation failed."""
    pass


class UnsupportedFormatError(ConfigError):
    """Unsupported configuration format."""
    pass


# ===========================
# Configuration Schema (Pydantic)
# ===========================

class DatabaseConfig(BaseModel):
    """Database configuration schema."""
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535)
    database: str
    username: str
    password: str
    pool_size: int = Field(default=10, ge=1)
    max_overflow: int = Field(default=20, ge=0)
    echo: bool = Field(default=False)


class APIConfig(BaseModel):
    """API configuration schema."""
    key: str
    secret: str
    endpoint: str = Field(default="https://api.example.com")
    timeout: int = Field(default=30, ge=1)
    retries: int = Field(default=3, ge=0)
    rate_limit: int = Field(default=100, ge=1)


class LoggingConfig(BaseModel):
    """Logging configuration schema."""
    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file: Optional[str] = None
    max_bytes: int = Field(default=10485760)  # 10MB
    backup_count: int = Field(default=5, ge=0)


class ModelConfig(BaseModel):
    """ML Model configuration schema."""
    name: str
    version: str
    path: str
    batch_size: int = Field(default=32, ge=1)
    device: str = Field(default="cpu")
    precision: str = Field(default="fp32")


class AppConfig(BaseModel):
    """Complete application configuration schema."""
    app_name: str
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    database: DatabaseConfig
    api: Optional[APIConfig] = None
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    models: List[ModelConfig] = Field(default_factory=list)


# ===========================
# Configuration Loader
# ===========================

class ConfigurationLoader:
    """Load configuration files with error handling."""

    SUPPORTED_FORMATS = {'.yaml', '.yml', '.json'}

    def __init__(self, config_dir: str = 'configs'):
        """
        Initialize configuration loader.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self._ensure_config_dir()

    def _ensure_config_dir(self) -> None:
        """Ensure configuration directory exists."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Configuration directory: {self.config_dir}")
        except Exception as e:
            logger.error(f"Failed to create config directory: {e}")
            raise ConfigError(f"Cannot create config directory: {e}")

    def load(self, filename: str) -> Dict[str, Any]:
        """
        Load configuration file.

        Args:
            filename: Configuration filename

        Returns:
            Configuration dictionary

        Raises:
            ConfigNotFoundError: If file not found
            ConfigParseError: If parsing fails
            UnsupportedFormatError: If format not supported
        """
        filepath = self.config_dir / filename

        # Check if file exists
        if not filepath.exists():
            logger.error(f"Configuration file not found: {filepath}")
            raise ConfigNotFoundError(
                f"Configuration file not found: {filename}",
                config_file=str(filepath)
            )

        # Check if it's a file
        if not filepath.is_file():
            logger.error(f"Not a file: {filepath}")
            raise ConfigError(
                f"Not a file: {filename}",
                config_file=str(filepath)
            )

        # Check file format
        suffix = filepath.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            logger.error(f"Unsupported format: {suffix}")
            raise UnsupportedFormatError(
                f"Unsupported format: {suffix}. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}",
                config_file=str(filepath)
            )

        # Load configuration
        try:
            with open(filepath, 'r') as f:
                if suffix in {'.yaml', '.yml'}:
                    config = yaml.safe_load(f)
                else:  # .json
                    config = json.load(f)

            logger.info(f"Loaded configuration from {filepath}")
            return config if config else {}

        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {filepath}: {e}")
            raise ConfigParseError(
                f"YAML parsing error: {e}",
                config_file=str(filepath)
            ) from e

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in {filepath}: {e}")
            raise ConfigParseError(
                f"JSON parsing error: {e}",
                config_file=str(filepath)
            ) from e

        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            raise ConfigError(
                f"Failed to load configuration: {e}",
                config_file=str(filepath)
            ) from e

    def save(self, filename: str, config: Dict[str, Any]) -> None:
        """
        Save configuration to file.

        Args:
            filename: Configuration filename
            config: Configuration dictionary
        """
        filepath = self.config_dir / filename
        suffix = filepath.suffix.lower()

        if suffix not in self.SUPPORTED_FORMATS:
            raise UnsupportedFormatError(
                f"Unsupported format: {suffix}",
                config_file=str(filepath)
            )

        try:
            with open(filepath, 'w') as f:
                if suffix in {'.yaml', '.yml'}:
                    yaml.dump(config, f, default_flow_style=False)
                else:  # .json
                    json.dump(config, f, indent=2)

            logger.info(f"Saved configuration to {filepath}")

        except Exception as e:
            logger.error(f"Error saving {filepath}: {e}")
            raise ConfigError(
                f"Failed to save configuration: {e}",
                config_file=str(filepath)
            ) from e


# ===========================
# Configuration Validator
# ===========================

class ConfigurationValidator:
    """Validate configuration against schema."""

    def validate(self, config: Dict[str, Any]) -> AppConfig:
        """
        Validate configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Validated AppConfig object

        Raises:
            ConfigValidationError: If validation fails
        """
        try:
            validated_config = AppConfig(**config)
            logger.info("Configuration validation successful")
            return validated_config

        except ValidationError as e:
            error_messages = []
            for error in e.errors():
                field = " -> ".join(str(x) for x in error['loc'])
                error_messages.append(f"{field}: {error['msg']}")

            error_msg = "Configuration validation failed:\n" + "\n".join(error_messages)
            logger.error(error_msg)
            raise ConfigValidationError(error_msg) from e


# ===========================
# Configuration Merger
# ===========================

class ConfigurationMerger:
    """Merge multiple configuration files."""

    def merge(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two configurations.

        Args:
            base_config: Base configuration
            override_config: Override configuration

        Returns:
            Merged configuration
        """
        merged = base_config.copy()

        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge(merged[key], value)
            else:
                merged[key] = value

        return merged

    def merge_multiple(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configurations.

        Args:
            *configs: Configuration dictionaries

        Returns:
            Merged configuration
        """
        if not configs:
            return {}

        result = configs[0].copy()
        for config in configs[1:]:
            result = self.merge(result, config)

        return result


# ===========================
# File Watcher for Hot Reload
# ===========================

class ConfigFileWatcher(FileSystemEventHandler):
    """Watch configuration files for changes."""

    def __init__(self, callback: Callable[[str], None]):
        """
        Initialize file watcher.

        Args:
            callback: Function to call when file changes
        """
        self.callback = callback
        self.last_modified: Dict[str, float] = {}
        self.debounce_seconds = 1.0

    def on_modified(self, event):
        """Handle file modification event."""
        if event.is_directory:
            return

        filepath = event.src_path

        # Debounce - ignore if modified recently
        now = time.time()
        if filepath in self.last_modified:
            if now - self.last_modified[filepath] < self.debounce_seconds:
                return

        self.last_modified[filepath] = now

        logger.info(f"Configuration file changed: {filepath}")
        self.callback(filepath)


# ===========================
# Main Configuration Manager
# ===========================

@dataclass
class ConfigurationManager:
    """
    Complete configuration management system.

    Features:
    - Load and validate configurations
    - Environment-specific configs
    - Hot-reloading
    - Thread-safe operations
    """

    config_dir: str = 'configs'
    default_environment: str = 'development'

    def __post_init__(self):
        """Initialize manager components."""
        self.loader = ConfigurationLoader(self.config_dir)
        self.validator = ConfigurationValidator()
        self.merger = ConfigurationMerger()

        self.current_config: Optional[AppConfig] = None
        self.config_lock = Lock()

        self.observer: Optional[Observer] = None
        self.watch_callbacks: List[Callable] = []

        logger.info("ConfigurationManager initialized")

    def load_config(
        self,
        filename: str,
        environment: Optional[str] = None
    ) -> AppConfig:
        """
        Load and validate configuration.

        Args:
            filename: Base configuration filename
            environment: Environment (dev, staging, prod)

        Returns:
            Validated configuration
        """
        env = environment or self.default_environment

        try:
            # Load base configuration
            base_config = self.loader.load(filename)
            logger.info(f"Loaded base config from {filename}")

            # Try to load environment-specific config
            name_parts = Path(filename).stem
            suffix = Path(filename).suffix
            env_filename = f"{name_parts}.{env}{suffix}"

            configs_to_merge = [base_config]

            try:
                env_config = self.loader.load(env_filename)
                configs_to_merge.append(env_config)
                logger.info(f"Loaded environment config from {env_filename}")
            except ConfigNotFoundError:
                logger.info(f"No environment-specific config found for {env}")

            # Merge configurations
            merged_config = self.merger.merge_multiple(*configs_to_merge)

            # Validate
            validated_config = self.validator.validate(merged_config)

            # Store current config
            with self.config_lock:
                self.current_config = validated_config

            logger.info(f"Configuration loaded successfully for {env} environment")
            return validated_config

        except ConfigError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}")
            raise ConfigError(f"Failed to load configuration: {e}") from e

    def get_config(self) -> Optional[AppConfig]:
        """Get current configuration thread-safely."""
        with self.config_lock:
            return self.current_config

    def watch_config(
        self,
        filename: str,
        callback: Optional[Callable] = None
    ) -> None:
        """
        Watch configuration file for changes and reload.

        Args:
            filename: Configuration filename to watch
            callback: Optional callback for config changes
        """
        if callback:
            self.watch_callbacks.append(callback)

        def on_change(filepath: str):
            """Handle configuration file change."""
            try:
                logger.info(f"Reloading configuration from {filepath}")
                self.load_config(filename)

                # Call callbacks
                for cb in self.watch_callbacks:
                    try:
                        cb(self.current_config)
                    except Exception as e:
                        logger.error(f"Error in watch callback: {e}")

            except Exception as e:
                logger.error(f"Error reloading configuration: {e}")

        # Set up file watcher
        event_handler = ConfigFileWatcher(on_change)
        self.observer = Observer()
        self.observer.schedule(
            event_handler,
            str(self.loader.config_dir),
            recursive=False
        )
        self.observer.start()

        logger.info(f"Watching {filename} for changes")

    def stop_watching(self) -> None:
        """Stop watching configuration files."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("Stopped watching configuration files")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_watching()
        return False


# ===========================
# Demo Function
# ===========================

def demo():
    """Demonstrate the Configuration Management System."""
    print("=" * 70)
    print("Configuration Management System Demo")
    print("=" * 70)

    # Create demo configuration
    demo_config = {
        "app_name": "MLPlatform",
        "environment": "development",
        "debug": True,
        "database": {
            "host": "localhost",
            "port": 5432,
            "database": "mlplatform",
            "username": "admin",
            "password": "secret123",
            "pool_size": 10
        },
        "api": {
            "key": "api_key_123",
            "secret": "api_secret_456",
            "endpoint": "https://api.example.com",
            "timeout": 30
        },
        "logging": {
            "level": "INFO",
            "file": "app.log"
        },
        "models": [
            {
                "name": "classifier",
                "version": "1.0.0",
                "path": "/models/classifier_v1.pkl",
                "batch_size": 32,
                "device": "cuda"
            }
        ]
    }

    # Initialize manager
    manager = ConfigurationManager(config_dir='demo_configs')

    # Save demo configuration
    print("\n1. Saving demo configuration...")
    manager.loader.save('app.yaml', demo_config)

    # Load configuration
    print("\n2. Loading configuration...")
    try:
        config = manager.load_config('app.yaml')
        print(f"   ✓ Configuration loaded: {config.app_name}")
        print(f"   ✓ Environment: {config.environment}")
        print(f"   ✓ Database: {config.database.host}:{config.database.port}")
        print(f"   ✓ Models: {len(config.models)} configured")

    except ConfigError as e:
        print(f"   ✗ Error: {e}")

    # Get current configuration
    print("\n3. Getting current configuration...")
    current = manager.get_config()
    if current:
        print(f"   ✓ App: {current.app_name}")
        print(f"   ✓ Debug mode: {current.debug}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo()
