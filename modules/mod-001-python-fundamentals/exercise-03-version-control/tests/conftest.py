#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for version control tests.
"""

import pytest
import subprocess
import shutil
from pathlib import Path


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


@pytest.fixture(scope="session")
def git_available():
    """Check if git is available."""
    try:
        subprocess.run(
            ["git", "--version"],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("Git is not available")
        return False


@pytest.fixture
def clean_git_config(tmp_path, monkeypatch):
    """
    Provide a clean Git configuration for tests.

    This ensures tests don't use the user's actual Git config.
    """
    git_config_home = tmp_path / "git_config"
    git_config_home.mkdir()

    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("HOME", str(git_config_home))

    return git_config_home
