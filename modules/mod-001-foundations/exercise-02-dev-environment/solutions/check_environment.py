#!/usr/bin/env python3
"""
Comprehensive Development Environment Checker for AI Infrastructure Engineers

This script validates that all required tools, configurations, and dependencies
are properly installed and configured for AI/ML infrastructure development.

Usage:
    python check_environment.py              # Check all components
    python check_environment.py --verbose    # Detailed output
    python check_environment.py --json       # JSON output for automation
    python check_environment.py --fix        # Attempt to fix issues

Author: AI Infrastructure Curriculum Team
License: MIT
"""

import subprocess
import sys
import os
import platform
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import argparse


class CheckStatus(Enum):
    """Status of environment checks."""
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    WARN = "⚠️  WARN"
    SKIP = "⏭️  SKIP"


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


@dataclass
class CheckResult:
    """Result of an individual environment check."""
    name: str
    status: CheckStatus
    version: Optional[str] = None
    message: Optional[str] = None
    fix_command: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        result = asdict(self)
        result['status'] = self.status.value
        return result


class EnvironmentChecker:
    """Main environment checker class."""

    def __init__(self, verbose: bool = False, json_output: bool = False):
        """
        Initialize environment checker.

        Args:
            verbose: Enable detailed output
            json_output: Output results as JSON
        """
        self.verbose = verbose
        self.json_output = json_output
        self.results: List[CheckResult] = []
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict[str, str]:
        """Get basic system information."""
        return {
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "hostname": platform.node()
        }

    def _run_command(self, command: str, shell: bool = True) -> Tuple[bool, str, str]:
        """
        Run a shell command and return result.

        Args:
            command: Command to execute
            shell: Whether to run in shell mode

        Returns:
            Tuple of (success, stdout, stderr)
        """
        try:
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=10
            )
            return (
                result.returncode == 0,
                result.stdout.strip(),
                result.stderr.strip()
            )
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)

    def _extract_version(self, output: str, patterns: List[str]) -> Optional[str]:
        """
        Extract version number from command output.

        Args:
            output: Command output text
            patterns: List of patterns to try

        Returns:
            Extracted version or None
        """
        import re
        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                return match.group(1) if match.groups() else match.group(0)
        return None

    def check_python(self) -> CheckResult:
        """Check Python installation and version."""
        try:
            version = platform.python_version()
            major, minor, patch = map(int, version.split('.'))

            if major >= 3 and minor >= 11:
                return CheckResult(
                    name="Python",
                    status=CheckStatus.PASS,
                    version=version,
                    message=f"Python {version} is installed (>= 3.11 required)"
                )
            elif major >= 3 and minor >= 8:
                return CheckResult(
                    name="Python",
                    status=CheckStatus.WARN,
                    version=version,
                    message=f"Python {version} works but 3.11+ recommended",
                    fix_command="Visit https://www.python.org/downloads/"
                )
            else:
                return CheckResult(
                    name="Python",
                    status=CheckStatus.FAIL,
                    version=version,
                    message=f"Python {version} is too old (need >= 3.8)",
                    fix_command="Visit https://www.python.org/downloads/"
                )
        except Exception as e:
            return CheckResult(
                name="Python",
                status=CheckStatus.FAIL,
                message=f"Python check failed: {e}"
            )

    def check_pip(self) -> CheckResult:
        """Check pip installation and version."""
        success, stdout, stderr = self._run_command("pip --version")

        if success:
            version = self._extract_version(stdout, [r'pip (\d+\.\d+\.\d+)'])
            return CheckResult(
                name="pip",
                status=CheckStatus.PASS,
                version=version,
                message="pip is installed"
            )
        else:
            return CheckResult(
                name="pip",
                status=CheckStatus.FAIL,
                message="pip is not installed",
                fix_command="python -m ensurepip --upgrade"
            )

    def check_git(self) -> CheckResult:
        """Check Git installation and configuration."""
        success, stdout, stderr = self._run_command("git --version")

        if not success:
            return CheckResult(
                name="Git",
                status=CheckStatus.FAIL,
                message="Git is not installed",
                fix_command="Visit https://git-scm.com/downloads"
            )

        version = self._extract_version(stdout, [r'git version (\d+\.\d+\.\d+)'])

        # Check Git config
        user_name_ok, name, _ = self._run_command("git config user.name")
        user_email_ok, email, _ = self._run_command("git config user.email")

        details = {
            "version": version,
            "user_name": name if user_name_ok else None,
            "user_email": email if user_email_ok else None
        }

        if not user_name_ok or not user_email_ok:
            return CheckResult(
                name="Git",
                status=CheckStatus.WARN,
                version=version,
                message="Git installed but not fully configured",
                fix_command="git config --global user.name 'Your Name' && git config --global user.email 'your@email.com'",
                details=details
            )

        return CheckResult(
            name="Git",
            status=CheckStatus.PASS,
            version=version,
            message="Git is properly configured",
            details=details
        )

    def check_docker(self) -> CheckResult:
        """Check Docker installation and daemon status."""
        success, stdout, stderr = self._run_command("docker --version")

        if not success:
            return CheckResult(
                name="Docker",
                status=CheckStatus.FAIL,
                message="Docker is not installed",
                fix_command="Visit https://docs.docker.com/get-docker/"
            )

        version = self._extract_version(stdout, [r'Docker version (\d+\.\d+\.\d+)'])

        # Check if Docker daemon is running
        daemon_ok, _, _ = self._run_command("docker ps")

        if not daemon_ok:
            return CheckResult(
                name="Docker",
                status=CheckStatus.WARN,
                version=version,
                message="Docker installed but daemon not running",
                fix_command="Start Docker Desktop or run: sudo systemctl start docker"
            )

        return CheckResult(
            name="Docker",
            status=CheckStatus.PASS,
            version=version,
            message="Docker is installed and running"
        )

    def check_docker_compose(self) -> CheckResult:
        """Check Docker Compose installation."""
        # Try docker compose (v2)
        success, stdout, stderr = self._run_command("docker compose version")

        if success:
            version = self._extract_version(stdout, [r'version (\d+\.\d+\.\d+)'])
            return CheckResult(
                name="Docker Compose",
                status=CheckStatus.PASS,
                version=version,
                message="Docker Compose V2 is installed"
            )

        # Try docker-compose (v1)
        success, stdout, stderr = self._run_command("docker-compose --version")

        if success:
            version = self._extract_version(stdout, [r'version (\d+\.\d+\.\d+)'])
            return CheckResult(
                name="Docker Compose",
                status=CheckStatus.WARN,
                version=version,
                message="Docker Compose V1 (deprecated) - upgrade to V2 recommended",
                fix_command="Visit https://docs.docker.com/compose/install/"
            )

        return CheckResult(
            name="Docker Compose",
            status=CheckStatus.FAIL,
            message="Docker Compose not installed",
            fix_command="Visit https://docs.docker.com/compose/install/"
        )

    def check_kubectl(self) -> CheckResult:
        """Check kubectl installation."""
        success, stdout, stderr = self._run_command("kubectl version --client --short 2>/dev/null || kubectl version --client")

        if not success:
            return CheckResult(
                name="kubectl",
                status=CheckStatus.FAIL,
                message="kubectl is not installed",
                fix_command="Visit https://kubernetes.io/docs/tasks/tools/"
            )

        version = self._extract_version(stdout, [r'Client Version: v(\d+\.\d+\.\d+)', r'v(\d+\.\d+\.\d+)'])

        return CheckResult(
            name="kubectl",
            status=CheckStatus.PASS,
            version=version,
            message="kubectl is installed"
        )

    def check_terraform(self) -> CheckResult:
        """Check Terraform installation."""
        success, stdout, stderr = self._run_command("terraform --version")

        if not success:
            return CheckResult(
                name="Terraform",
                status=CheckStatus.WARN,
                message="Terraform not installed (optional but recommended)",
                fix_command="Visit https://www.terraform.io/downloads"
            )

        version = self._extract_version(stdout, [r'Terraform v(\d+\.\d+\.\d+)'])

        return CheckResult(
            name="Terraform",
            status=CheckStatus.PASS,
            version=version,
            message="Terraform is installed"
        )

    def check_aws_cli(self) -> CheckResult:
        """Check AWS CLI installation."""
        success, stdout, stderr = self._run_command("aws --version")

        if not success:
            return CheckResult(
                name="AWS CLI",
                status=CheckStatus.WARN,
                message="AWS CLI not installed (optional)",
                fix_command="Visit https://aws.amazon.com/cli/"
            )

        version = self._extract_version(stdout, [r'aws-cli/(\d+\.\d+\.\d+)'])

        return CheckResult(
            name="AWS CLI",
            status=CheckStatus.PASS,
            version=version,
            message="AWS CLI is installed"
        )

    def check_gcloud(self) -> CheckResult:
        """Check Google Cloud SDK installation."""
        success, stdout, stderr = self._run_command("gcloud --version")

        if not success:
            return CheckResult(
                name="gcloud",
                status=CheckStatus.WARN,
                message="Google Cloud SDK not installed (optional)",
                fix_command="Visit https://cloud.google.com/sdk/docs/install"
            )

        version = self._extract_version(stdout, [r'Google Cloud SDK (\d+\.\d+\.\d+)'])

        return CheckResult(
            name="gcloud",
            status=CheckStatus.PASS,
            version=version,
            message="Google Cloud SDK is installed"
        )

    def check_vscode(self) -> CheckResult:
        """Check VS Code installation."""
        # Try different VS Code command names
        for cmd in ["code", "code-insiders"]:
            success, stdout, stderr = self._run_command(f"{cmd} --version")
            if success:
                lines = stdout.split('\n')
                version = lines[0] if lines else None
                return CheckResult(
                    name="VS Code",
                    status=CheckStatus.PASS,
                    version=version,
                    message=f"VS Code ({cmd}) is installed"
                )

        return CheckResult(
            name="VS Code",
            status=CheckStatus.WARN,
            message="VS Code not found (recommended IDE)",
            fix_command="Visit https://code.visualstudio.com/"
        )

    def check_virtual_env_support(self) -> CheckResult:
        """Check if virtual environment tools are available."""
        # Check for venv (built-in)
        try:
            import venv
            venv_available = True
        except ImportError:
            venv_available = False

        # Check for virtualenv
        virtualenv_ok, _, _ = self._run_command("virtualenv --version")

        if venv_available:
            return CheckResult(
                name="Virtual Environments",
                status=CheckStatus.PASS,
                message="venv module is available",
                details={"venv": True, "virtualenv": virtualenv_ok}
            )
        elif virtualenv_ok:
            return CheckResult(
                name="Virtual Environments",
                status=CheckStatus.PASS,
                message="virtualenv is installed",
                details={"venv": False, "virtualenv": True}
            )
        else:
            return CheckResult(
                name="Virtual Environments",
                status=CheckStatus.FAIL,
                message="No virtual environment tool found",
                fix_command="pip install virtualenv"
            )

    def check_ml_libraries(self) -> CheckResult:
        """Check if common ML libraries are installed."""
        libraries = {
            "numpy": "import numpy",
            "pandas": "import pandas",
            "scikit-learn": "import sklearn",
            "tensorflow": "import tensorflow",
            "torch": "import torch",
            "fastapi": "import fastapi"
        }

        installed = {}
        for lib_name, import_stmt in libraries.items():
            try:
                exec(import_stmt)
                installed[lib_name] = True
            except ImportError:
                installed[lib_name] = False

        installed_count = sum(installed.values())

        if installed_count >= 3:
            return CheckResult(
                name="ML Libraries",
                status=CheckStatus.PASS,
                message=f"{installed_count}/6 common ML libraries installed",
                details=installed
            )
        elif installed_count > 0:
            return CheckResult(
                name="ML Libraries",
                status=CheckStatus.WARN,
                message=f"Only {installed_count}/6 ML libraries installed",
                fix_command="pip install numpy pandas scikit-learn",
                details=installed
            )
        else:
            return CheckResult(
                name="ML Libraries",
                status=CheckStatus.WARN,
                message="No ML libraries installed yet",
                fix_command="pip install numpy pandas scikit-learn",
                details=installed
            )

    def check_disk_space(self) -> CheckResult:
        """Check available disk space."""
        try:
            home = Path.home()
            stat = shutil.disk_usage(home)
            free_gb = stat.free / (1024**3)
            total_gb = stat.total / (1024**3)
            used_percent = (stat.used / stat.total) * 100

            details = {
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
                "used_percent": round(used_percent, 2)
            }

            if free_gb < 10:
                return CheckResult(
                    name="Disk Space",
                    status=CheckStatus.FAIL,
                    message=f"Only {free_gb:.1f}GB free (need at least 10GB)",
                    details=details
                )
            elif free_gb < 20:
                return CheckResult(
                    name="Disk Space",
                    status=CheckStatus.WARN,
                    message=f"{free_gb:.1f}GB free (20GB+ recommended)",
                    details=details
                )
            else:
                return CheckResult(
                    name="Disk Space",
                    status=CheckStatus.PASS,
                    message=f"{free_gb:.1f}GB free",
                    details=details
                )
        except Exception as e:
            return CheckResult(
                name="Disk Space",
                status=CheckStatus.WARN,
                message=f"Could not check disk space: {e}"
            )

    def check_memory(self) -> CheckResult:
        """Check available RAM."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024**3)
            available_gb = mem.available / (1024**3)
            used_percent = mem.percent

            details = {
                "total_gb": round(total_gb, 2),
                "available_gb": round(available_gb, 2),
                "used_percent": round(used_percent, 2)
            }

            if total_gb < 8:
                return CheckResult(
                    name="RAM",
                    status=CheckStatus.WARN,
                    message=f"{total_gb:.1f}GB RAM (8GB+ recommended for ML)",
                    details=details
                )
            else:
                return CheckResult(
                    name="RAM",
                    status=CheckStatus.PASS,
                    message=f"{total_gb:.1f}GB RAM available",
                    details=details
                )
        except ImportError:
            return CheckResult(
                name="RAM",
                status=CheckStatus.SKIP,
                message="psutil not installed (run: pip install psutil)"
            )
        except Exception as e:
            return CheckResult(
                name="RAM",
                status=CheckStatus.WARN,
                message=f"Could not check RAM: {e}"
            )

    def run_all_checks(self) -> List[CheckResult]:
        """Run all environment checks."""
        checks = [
            self.check_python,
            self.check_pip,
            self.check_git,
            self.check_docker,
            self.check_docker_compose,
            self.check_kubectl,
            self.check_terraform,
            self.check_aws_cli,
            self.check_gcloud,
            self.check_vscode,
            self.check_virtual_env_support,
            self.check_ml_libraries,
            self.check_disk_space,
            self.check_memory
        ]

        self.results = []
        for check_func in checks:
            try:
                result = check_func()
                self.results.append(result)
            except Exception as e:
                self.results.append(CheckResult(
                    name=check_func.__name__.replace('check_', '').title(),
                    status=CheckStatus.FAIL,
                    message=f"Check failed with error: {e}"
                ))

        return self.results

    def print_results(self):
        """Print results in human-readable format."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}  AI Infrastructure Development Environment Check{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")

        # Print system info
        print(f"{Colors.BOLD}System Information:{Colors.END}")
        for key, value in self.system_info.items():
            print(f"  {key}: {value}")
        print()

        # Count statuses
        pass_count = sum(1 for r in self.results if r.status == CheckStatus.PASS)
        fail_count = sum(1 for r in self.results if r.status == CheckStatus.FAIL)
        warn_count = sum(1 for r in self.results if r.status == CheckStatus.WARN)

        # Print results
        for result in self.results:
            color = {
                CheckStatus.PASS: Colors.GREEN,
                CheckStatus.FAIL: Colors.RED,
                CheckStatus.WARN: Colors.YELLOW,
                CheckStatus.SKIP: Colors.BLUE
            }.get(result.status, Colors.END)

            version_str = f" ({result.version})" if result.version else ""
            print(f"{color}{result.status.value}{Colors.END} {Colors.BOLD}{result.name}{Colors.END}{version_str}")

            if self.verbose or result.status != CheckStatus.PASS:
                if result.message:
                    print(f"    {result.message}")
                if result.fix_command and result.status in [CheckStatus.FAIL, CheckStatus.WARN]:
                    print(f"    {Colors.CYAN}Fix: {result.fix_command}{Colors.END}")
                if self.verbose and result.details:
                    print(f"    Details: {json.dumps(result.details, indent=6)}")
            print()

        # Print summary
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}Summary:{Colors.END}")
        print(f"  {Colors.GREEN}✅ Passed: {pass_count}{Colors.END}")
        print(f"  {Colors.YELLOW}⚠️  Warnings: {warn_count}{Colors.END}")
        print(f"  {Colors.RED}❌ Failed: {fail_count}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")

        # Overall status
        if fail_count == 0 and warn_count == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}🎉 Your environment is fully configured!{Colors.END}\n")
            return 0
        elif fail_count == 0:
            print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  Your environment is usable but has some warnings.{Colors.END}\n")
            return 0
        else:
            print(f"{Colors.RED}{Colors.BOLD}❌ Please fix the failed checks before proceeding.{Colors.END}\n")
            return 1

    def export_json(self) -> str:
        """Export results as JSON."""
        output = {
            "system_info": self.system_info,
            "checks": [r.to_dict() for r in self.results],
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.status == CheckStatus.PASS),
                "failed": sum(1 for r in self.results if r.status == CheckStatus.FAIL),
                "warnings": sum(1 for r in self.results if r.status == CheckStatus.WARN),
                "skipped": sum(1 for r in self.results if r.status == CheckStatus.SKIP)
            }
        }
        return json.dumps(output, indent=2)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check AI Infrastructure development environment",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output for all checks"
    )
    parser.add_argument(
        "-j", "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to automatically fix issues (not yet implemented)"
    )

    args = parser.parse_args()

    checker = EnvironmentChecker(verbose=args.verbose, json_output=args.json)
    checker.run_all_checks()

    if args.json:
        print(checker.export_json())
        return 0
    else:
        return checker.print_results()


if __name__ == "__main__":
    sys.exit(main())
