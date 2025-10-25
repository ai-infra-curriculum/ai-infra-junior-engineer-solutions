#!/usr/bin/env python3
"""
Test suite for File I/O & Error Handling exercise.
"""

import pytest
import sys
import json
import yaml
from pathlib import Path

# Add solutions directory to path
solutions_dir = Path(__file__).parent.parent / "solutions"
sys.path.insert(0, str(solutions_dir))

from config_manager import (
    ConfigurationManager,
    ConfigurationLoader,
    ConfigurationValidator,
    ConfigurationMerger,
    ConfigError,
    ConfigNotFoundError,
    ConfigParseError,
    ConfigValidationError,
    UnsupportedFormatError,
    AppConfig
)


# ===========================
# Fixtures
# ===========================

@pytest.fixture
def temp_config_dir(tmp_path):
    """Create temporary configuration directory."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def sample_config():
    """Create sample configuration."""
    return {
        "app_name": "TestApp",
        "environment": "test",
        "debug": True,
        "database": {
            "host": "localhost",
            "port": 5432,
            "database": "testdb",
            "username": "test",
            "password": "test123"
        },
        "logging": {
            "level": "DEBUG",
            "file": "test.log"
        }
    }


@pytest.fixture
def config_loader(temp_config_dir):
    """Create configuration loader."""
    return ConfigurationLoader(str(temp_config_dir))


@pytest.fixture
def config_validator():
    """Create configuration validator."""
    return ConfigurationValidator()


@pytest.fixture
def config_merger():
    """Create configuration merger."""
    return ConfigurationMerger()


# ===========================
# Configuration Loader Tests
# ===========================

class TestConfigurationLoader:
    """Test ConfigurationLoader class."""

    def test_initialization(self, temp_config_dir):
        """Test loader initialization."""
        loader = ConfigurationLoader(str(temp_config_dir))
        assert loader.config_dir == temp_config_dir
        assert temp_config_dir.exists()

    def test_load_yaml(self, config_loader, temp_config_dir, sample_config):
        """Test loading YAML configuration."""
        # Save test config
        config_file = temp_config_dir / "test.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)

        # Load config
        loaded = config_loader.load("test.yaml")
        assert loaded["app_name"] == sample_config["app_name"]
        assert loaded["database"]["host"] == sample_config["database"]["host"]

    def test_load_json(self, config_loader, temp_config_dir, sample_config):
        """Test loading JSON configuration."""
        # Save test config
        config_file = temp_config_dir / "test.json"
        with open(config_file, 'w') as f:
            json.dump(sample_config, f)

        # Load config
        loaded = config_loader.load("test.json")
        assert loaded["app_name"] == sample_config["app_name"]

    def test_load_nonexistent_file(self, config_loader):
        """Test loading nonexistent file raises error."""
        with pytest.raises(ConfigNotFoundError):
            config_loader.load("nonexistent.yaml")

    def test_load_unsupported_format(self, config_loader, temp_config_dir):
        """Test loading unsupported format raises error."""
        # Create file with unsupported extension
        config_file = temp_config_dir / "test.txt"
        config_file.write_text("some content")

        with pytest.raises(UnsupportedFormatError):
            config_loader.load("test.txt")

    def test_load_invalid_yaml(self, config_loader, temp_config_dir):
        """Test loading invalid YAML raises error."""
        config_file = temp_config_dir / "invalid.yaml"
        config_file.write_text("invalid: yaml: content:\n  - bad")

        with pytest.raises(ConfigParseError):
            config_loader.load("invalid.yaml")

    def test_load_invalid_json(self, config_loader, temp_config_dir):
        """Test loading invalid JSON raises error."""
        config_file = temp_config_dir / "invalid.json"
        config_file.write_text('{"invalid": json content}')

        with pytest.raises(ConfigParseError):
            config_loader.load("invalid.json")

    def test_save_yaml(self, config_loader, temp_config_dir, sample_config):
        """Test saving YAML configuration."""
        config_loader.save("output.yaml", sample_config)

        config_file = temp_config_dir / "output.yaml"
        assert config_file.exists()

        # Load and verify
        with open(config_file, 'r') as f:
            loaded = yaml.safe_load(f)
        assert loaded == sample_config

    def test_save_json(self, config_loader, temp_config_dir, sample_config):
        """Test saving JSON configuration."""
        config_loader.save("output.json", sample_config)

        config_file = temp_config_dir / "output.json"
        assert config_file.exists()

        # Load and verify
        with open(config_file, 'r') as f:
            loaded = json.load(f)
        assert loaded == sample_config


# ===========================
# Configuration Validator Tests
# ===========================

class TestConfigurationValidator:
    """Test ConfigurationValidator class."""

    def test_validate_valid_config(self, config_validator, sample_config):
        """Test validating valid configuration."""
        validated = config_validator.validate(sample_config)
        assert isinstance(validated, AppConfig)
        assert validated.app_name == sample_config["app_name"]
        assert validated.database.host == sample_config["database"]["host"]

    def test_validate_missing_required_field(self, config_validator):
        """Test validation fails for missing required field."""
        invalid_config = {
            "app_name": "TestApp",
            # Missing database field
        }

        with pytest.raises(ConfigValidationError):
            config_validator.validate(invalid_config)

    def test_validate_invalid_port(self, config_validator, sample_config):
        """Test validation fails for invalid port."""
        sample_config["database"]["port"] = 99999  # Invalid port
        with pytest.raises(ConfigValidationError):
            config_validator.validate(sample_config)

    def test_validate_with_defaults(self, config_validator, sample_config):
        """Test validation applies default values."""
        # Remove optional fields
        sample_config.pop("logging", None)
        sample_config["database"].pop("pool_size", None)

        validated = config_validator.validate(sample_config)
        assert validated.logging is not None
        assert validated.database.pool_size == 10  # Default value


# ===========================
# Configuration Merger Tests
# ===========================

class TestConfigurationMerger:
    """Test ConfigurationMerger class."""

    def test_merge_two_configs(self, config_merger):
        """Test merging two configurations."""
        base = {
            "app_name": "BaseApp",
            "database": {
                "host": "localhost",
                "port": 5432
            }
        }

        override = {
            "app_name": "OverrideApp",
            "database": {
                "port": 3306
            }
        }

        merged = config_merger.merge(base, override)

        assert merged["app_name"] == "OverrideApp"
        assert merged["database"]["host"] == "localhost"
        assert merged["database"]["port"] == 3306

    def test_merge_multiple_configs(self, config_merger):
        """Test merging multiple configurations."""
        config1 = {"a": 1, "b": 2}
        config2 = {"b": 3, "c": 4}
        config3 = {"c": 5, "d": 6}

        merged = config_merger.merge_multiple(config1, config2, config3)

        assert merged["a"] == 1
        assert merged["b"] == 3
        assert merged["c"] == 5
        assert merged["d"] == 6

    def test_merge_nested_dicts(self, config_merger):
        """Test merging nested dictionaries."""
        base = {
            "settings": {
                "feature1": True,
                "feature2": False,
                "nested": {
                    "option1": "value1",
                    "option2": "value2"
                }
            }
        }

        override = {
            "settings": {
                "feature2": True,
                "nested": {
                    "option2": "new_value2",
                    "option3": "value3"
                }
            }
        }

        merged = config_merger.merge(base, override)

        assert merged["settings"]["feature1"] == True
        assert merged["settings"]["feature2"] == True
        assert merged["settings"]["nested"]["option1"] == "value1"
        assert merged["settings"]["nested"]["option2"] == "new_value2"
        assert merged["settings"]["nested"]["option3"] == "value3"


# ===========================
# Configuration Manager Tests
# ===========================

class TestConfigurationManager:
    """Test ConfigurationManager class."""

    def test_initialization(self, temp_config_dir):
        """Test manager initialization."""
        manager = ConfigurationManager(config_dir=str(temp_config_dir))
        assert manager.loader is not None
        assert manager.validator is not None
        assert manager.merger is not None

    def test_load_config(self, temp_config_dir, sample_config):
        """Test loading configuration."""
        # Save test config
        config_file = temp_config_dir / "app.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)

        # Load config
        manager = ConfigurationManager(config_dir=str(temp_config_dir))
        config = manager.load_config("app.yaml")

        assert config.app_name == sample_config["app_name"]
        assert config.environment == sample_config["environment"]

    def test_load_with_environment_override(self, temp_config_dir, sample_config):
        """Test loading with environment-specific override."""
        # Save base config
        base_config_file = temp_config_dir / "app.yaml"
        with open(base_config_file, 'w') as f:
            yaml.dump(sample_config, f)

        # Save environment-specific config
        env_config = {
            "environment": "production",
            "debug": False,
            "database": {
                "host": "prod.example.com"
            }
        }
        env_config_file = temp_config_dir / "app.production.yaml"
        with open(env_config_file, 'w') as f:
            yaml.dump(env_config, f)

        # Load with environment
        manager = ConfigurationManager(config_dir=str(temp_config_dir))
        config = manager.load_config("app.yaml", environment="production")

        assert config.environment == "production"
        assert config.debug == False
        assert config.database.host == "prod.example.com"
        # Base values should still be present
        assert config.database.port == sample_config["database"]["port"]

    def test_get_config(self, temp_config_dir, sample_config):
        """Test getting current configuration."""
        config_file = temp_config_dir / "app.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)

        manager = ConfigurationManager(config_dir=str(temp_config_dir))
        manager.load_config("app.yaml")

        current = manager.get_config()
        assert current is not None
        assert current.app_name == sample_config["app_name"]

    def test_context_manager(self, temp_config_dir, sample_config):
        """Test using manager as context manager."""
        config_file = temp_config_dir / "app.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)

        with ConfigurationManager(config_dir=str(temp_config_dir)) as manager:
            config = manager.load_config("app.yaml")
            assert config is not None


# ===========================
# Custom Exception Tests
# ===========================

class TestCustomExceptions:
    """Test custom exception classes."""

    def test_config_error_attributes(self):
        """Test ConfigError attributes."""
        error = ConfigError("Test error", config_file="test.yaml")
        assert error.config_file == "test.yaml"
        assert error.timestamp is not None

    def test_config_error_str(self):
        """Test ConfigError string representation."""
        error = ConfigError("Test error", config_file="test.yaml")
        error_str = str(error)
        assert "ConfigError" in error_str
        assert "Test error" in error_str
        assert "test.yaml" in error_str

    def test_exception_hierarchy(self):
        """Test exception inheritance."""
        assert issubclass(ConfigNotFoundError, ConfigError)
        assert issubclass(ConfigParseError, ConfigError)
        assert issubclass(ConfigValidationError, ConfigError)
        assert issubclass(UnsupportedFormatError, ConfigError)


# ===========================
# Integration Tests
# ===========================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow(self, temp_config_dir):
        """Test complete configuration workflow."""
        # Create base configuration
        base_config = {
            "app_name": "IntegrationTest",
            "environment": "development",
            "debug": True,
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "testdb",
                "username": "test",
                "password": "test123"
            }
        }

        # Save configuration
        config_file = temp_config_dir / "app.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(base_config, f)

        # Initialize manager and load
        manager = ConfigurationManager(config_dir=str(temp_config_dir))
        config = manager.load_config("app.yaml")

        # Verify configuration
        assert config.app_name == "IntegrationTest"
        assert config.database.host == "localhost"

        # Get current configuration
        current = manager.get_config()
        assert current.app_name == config.app_name

    def test_error_handling(self, temp_config_dir):
        """Test error handling in complete workflow."""
        manager = ConfigurationManager(config_dir=str(temp_config_dir))

        # Test loading nonexistent file
        with pytest.raises(ConfigNotFoundError):
            manager.load_config("nonexistent.yaml")

        # Test loading invalid configuration
        invalid_file = temp_config_dir / "invalid.yaml"
        invalid_file.write_text("app_name: Test\n# Missing required fields")

        with pytest.raises(ConfigValidationError):
            manager.load_config("invalid.yaml")


# ===========================
# Run Tests
# ===========================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
