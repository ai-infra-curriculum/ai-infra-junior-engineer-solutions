#!/usr/bin/env python3
"""
Tests for check_environment.py

Run with:
    pytest tests/test_check_environment.py -v
    pytest tests/test_check_environment.py -v --cov=../
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from check_environment import (
    EnvironmentChecker,
    CheckResult,
    CheckStatus
)


class TestEnvironmentChecker:
    """Test suite for EnvironmentChecker class."""

    @pytest.fixture
    def checker(self):
        """Create EnvironmentChecker instance."""
        return EnvironmentChecker(verbose=False)

    @pytest.fixture
    def verbose_checker(self):
        """Create verbose EnvironmentChecker instance."""
        return EnvironmentChecker(verbose=True)

    def test_initialization(self, checker):
        """Test EnvironmentChecker initialization."""
        assert checker.verbose == False
        assert checker.results == []
        assert 'os' in checker.system_info
        assert 'python_version' in checker.system_info

    def test_verbose_initialization(self, verbose_checker):
        """Test verbose mode initialization."""
        assert verbose_checker.verbose == True

    def test_system_info_collected(self, checker):
        """Test that system info is collected."""
        info = checker.system_info
        assert 'os' in info
        assert 'os_version' in info
        assert 'architecture' in info
        assert 'python_version' in info
        assert 'hostname' in info

    @patch('check_environment.subprocess.run')
    def test_run_command_success(self, mock_run, checker):
        """Test successful command execution."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        success, stdout, stderr = checker._run_command("test command")

        assert success == True
        assert stdout == "output"
        assert stderr == ""

    @patch('check_environment.subprocess.run')
    def test_run_command_failure(self, mock_run, checker):
        """Test failed command execution."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "error"
        mock_run.return_value = mock_result

        success, stdout, stderr = checker._run_command("test command")

        assert success == False
        assert stdout == ""
        assert stderr == "error"

    def test_check_python(self, checker):
        """Test Python version check."""
        result = checker.check_python()

        assert isinstance(result, CheckResult)
        assert result.name == "Python"
        assert result.version is not None
        assert result.status in [CheckStatus.PASS, CheckStatus.WARN, CheckStatus.FAIL]

    @patch('check_environment.EnvironmentChecker._run_command')
    def test_check_pip_installed(self, mock_command, checker):
        """Test pip check when installed."""
        mock_command.return_value = (True, "pip 23.0.1", "")

        result = checker.check_pip()

        assert result.status == CheckStatus.PASS
        assert result.name == "pip"
        assert "23.0.1" in result.version

    @patch('check_environment.EnvironmentChecker._run_command')
    def test_check_pip_not_installed(self, mock_command, checker):
        """Test pip check when not installed."""
        mock_command.return_value = (False, "", "command not found")

        result = checker.check_pip()

        assert result.status == CheckStatus.FAIL
        assert result.fix_command is not None

    @patch('check_environment.EnvironmentChecker._run_command')
    def test_check_git_not_installed(self, mock_command, checker):
        """Test Git check when not installed."""
        mock_command.return_value = (False, "", "command not found")

        result = checker.check_git()

        assert result.status == CheckStatus.FAIL
        assert "not installed" in result.message

    @patch('check_environment.EnvironmentChecker._run_command')
    def test_check_git_not_configured(self, mock_command, checker):
        """Test Git check when installed but not configured."""
        # First call for version check
        # Second call for user.name check
        # Third call for user.email check
        mock_command.side_effect = [
            (True, "git version 2.40.0", ""),
            (False, "", ""),  # user.name not set
            (False, "", "")   # user.email not set
        ]

        result = checker.check_git()

        assert result.status == CheckStatus.WARN
        assert result.fix_command is not None

    @patch('check_environment.EnvironmentChecker._run_command')
    def test_check_git_configured(self, mock_command, checker):
        """Test Git check when properly configured."""
        mock_command.side_effect = [
            (True, "git version 2.40.0", ""),
            (True, "John Doe", ""),
            (True, "john@example.com", "")
        ]

        result = checker.check_git()

        assert result.status == CheckStatus.PASS
        assert result.details is not None

    @patch('check_environment.EnvironmentChecker._run_command')
    def test_check_docker_not_installed(self, mock_command, checker):
        """Test Docker check when not installed."""
        mock_command.return_value = (False, "", "command not found")

        result = checker.check_docker()

        assert result.status == CheckStatus.FAIL
        assert "not installed" in result.message

    @patch('check_environment.EnvironmentChecker._run_command')
    def test_check_docker_daemon_not_running(self, mock_command, checker):
        """Test Docker check when daemon not running."""
        mock_command.side_effect = [
            (True, "Docker version 24.0.0", ""),
            (False, "", "Cannot connect to Docker daemon")
        ]

        result = checker.check_docker()

        assert result.status == CheckStatus.WARN
        assert "not running" in result.message

    @patch('check_environment.EnvironmentChecker._run_command')
    def test_check_kubectl_installed(self, mock_command, checker):
        """Test kubectl check when installed."""
        mock_command.return_value = (True, "Client Version: v1.28.0", "")

        result = checker.check_kubectl()

        assert result.status == CheckStatus.PASS
        assert result.version is not None

    def test_check_virtual_env_support(self, checker):
        """Test virtual environment support check."""
        result = checker.check_virtual_env_support()

        assert isinstance(result, CheckResult)
        assert result.name == "Virtual Environments"
        assert result.status in [CheckStatus.PASS, CheckStatus.FAIL]

    @patch('check_environment.shutil.disk_usage')
    def test_check_disk_space_sufficient(self, mock_disk, checker):
        """Test disk space check with sufficient space."""
        mock_stat = Mock()
        mock_stat.free = 50 * (1024**3)  # 50GB
        mock_stat.total = 100 * (1024**3)
        mock_stat.used = 50 * (1024**3)
        mock_disk.return_value = mock_stat

        result = checker.check_disk_space()

        assert result.status == CheckStatus.PASS

    @patch('check_environment.shutil.disk_usage')
    def test_check_disk_space_low(self, mock_disk, checker):
        """Test disk space check with low space."""
        mock_stat = Mock()
        mock_stat.free = 5 * (1024**3)  # 5GB
        mock_stat.total = 100 * (1024**3)
        mock_stat.used = 95 * (1024**3)
        mock_disk.return_value = mock_stat

        result = checker.check_disk_space()

        assert result.status == CheckStatus.FAIL

    def test_extract_version(self, checker):
        """Test version extraction from command output."""
        # Test different version patterns
        output1 = "Python 3.11.0"
        version1 = checker._extract_version(output1, [r'Python (\d+\.\d+\.\d+)'])
        assert version1 == "3.11.0"

        output2 = "git version 2.40.0"
        version2 = checker._extract_version(output2, [r'version (\d+\.\d+\.\d+)'])
        assert version2 == "2.40.0"

        output3 = "no version here"
        version3 = checker._extract_version(output3, [r'(\d+\.\d+\.\d+)'])
        assert version3 is None

    def test_run_all_checks(self, checker):
        """Test running all checks."""
        results = checker.run_all_checks()

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, CheckResult) for r in results)

    def test_export_json(self, checker):
        """Test JSON export."""
        checker.run_all_checks()
        json_output = checker.export_json()

        assert isinstance(json_output, str)
        assert "system_info" in json_output
        assert "checks" in json_output
        assert "summary" in json_output

        # Verify it's valid JSON
        import json
        data = json.loads(json_output)
        assert 'system_info' in data
        assert 'checks' in data
        assert 'summary' in data


class TestCheckResult:
    """Test CheckResult dataclass."""

    def test_check_result_creation(self):
        """Test creating CheckResult."""
        result = CheckResult(
            name="Test",
            status=CheckStatus.PASS,
            version="1.0.0",
            message="Test message"
        )

        assert result.name == "Test"
        assert result.status == CheckStatus.PASS
        assert result.version == "1.0.0"
        assert result.message == "Test message"

    def test_check_result_to_dict(self):
        """Test converting CheckResult to dictionary."""
        result = CheckResult(
            name="Test",
            status=CheckStatus.PASS,
            message="Test message",
            details={"key": "value"}
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict['name'] == "Test"
        assert result_dict['status'] == CheckStatus.PASS.value
        assert result_dict['details'] == {"key": "value"}


class TestCheckStatus:
    """Test CheckStatus enum."""

    def test_check_status_values(self):
        """Test CheckStatus enum values."""
        assert CheckStatus.PASS.value == "✅ PASS"
        assert CheckStatus.FAIL.value == "❌ FAIL"
        assert CheckStatus.WARN.value == "⚠️  WARN"
        assert CheckStatus.SKIP.value == "⏭️  SKIP"


class TestIntegration:
    """Integration tests."""

    def test_full_check_workflow(self):
        """Test complete workflow."""
        checker = EnvironmentChecker(verbose=False)

        # Run all checks
        results = checker.run_all_checks()

        # Verify we got results
        assert len(results) > 0

        # Export to JSON
        json_output = checker.export_json()
        assert len(json_output) > 0

        # Verify JSON structure
        import json
        data = json.loads(json_output)

        assert data['summary']['total'] == len(results)
        assert data['summary']['total'] == (
            data['summary']['passed'] +
            data['summary']['failed'] +
            data['summary']['warnings'] +
            data['summary']['skipped']
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=../"])
