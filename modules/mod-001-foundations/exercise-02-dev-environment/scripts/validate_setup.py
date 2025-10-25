#!/usr/bin/env python3
"""
Validation script for Exercise 02: Development Environment Setup

Checks if all required tools are installed and configured correctly.
"""

import subprocess
import sys
import os
import platform
from pathlib import Path
from typing import Tuple, Optional


class Colors:
    """ANSI color codes."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(message: str):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{message}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def print_success(message: str):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.RED}❌ {message}{Colors.END}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")


def run_command(command: str) -> Tuple[bool, Optional[str]]:
    """
    Run a shell command and return success status and output.

    Args:
        command: Command to run

    Returns:
        Tuple of (success, output)
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        return True, result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False, None


def check_command(command: str, name: str, version_flag: str = "--version") -> bool:
    """
    Check if a command exists and get its version.

    Args:
        command: Command to check
        name: Display name
        version_flag: Flag to get version

    Returns:
        True if command exists
    """
    success, output = run_command(f"{command} {version_flag}")
    if success:
        version = output.split('\n')[0] if output else "installed"
        print_success(f"{name} installed: {version}")
        return True
    else:
        print_error(f"{name} not found")
        return False


def check_zsh():
    """Check if Zsh is installed and set as default shell."""
    print(f"\n{Colors.BOLD}Checking Shell...{Colors.END}")

    # Check if Zsh exists
    success, output = run_command("which zsh")
    if not success:
        print_error("Zsh not installed")
        return False

    zsh_path = output
    print_success(f"Zsh installed: {zsh_path}")

    # Check if Zsh is default shell
    current_shell = os.environ.get('SHELL', '')
    if 'zsh' in current_shell:
        print_success(f"Zsh is default shell: {current_shell}")
    else:
        print_warning(f"Default shell is {current_shell}, not Zsh")
        print_warning("Run: chsh -s $(which zsh)")

    # Check Oh My Zsh
    oh_my_zsh_path = Path.home() / ".oh-my-zsh"
    if oh_my_zsh_path.exists():
        print_success("Oh My Zsh installed")
        return True
    else:
        print_warning("Oh My Zsh not found")
        return False


def check_python():
    """Check Python installation and pyenv."""
    print(f"\n{Colors.BOLD}Checking Python Environment...{Colors.END}")

    checks_passed = True

    # Check Python version
    success, output = run_command("python --version")
    if success and output:
        version = output.replace("Python ", "")
        if version.startswith("3.11") or version.startswith("3.10"):
            print_success(f"Python installed: {version}")
        else:
            print_warning(f"Python {version} found, but 3.11+ recommended")
    else:
        print_error("Python not found")
        checks_passed = False

    # Check pyenv
    if check_command("pyenv", "pyenv", "version"):
        # Check pyenv versions
        success, output = run_command("pyenv versions")
        if success and "3.11" in output:
            print_success("Python 3.11 available in pyenv")
        else:
            print_warning("Python 3.11 not installed in pyenv")
    else:
        checks_passed = False

    # Check pip
    if not check_command("pip", "pip", "--version"):
        checks_passed = False

    # Check poetry
    success, _ = run_command("poetry --version")
    if success:
        print_success("Poetry installed")
    else:
        print_warning("Poetry not installed (optional but recommended)")

    return checks_passed


def check_docker():
    """Check Docker installation."""
    print(f"\n{Colors.BOLD}Checking Docker...{Colors.END}")

    checks_passed = True

    # Check Docker
    if not check_command("docker", "Docker", "--version"):
        checks_passed = False
        return False

    # Check Docker Compose
    success, output = run_command("docker compose version")
    if success:
        print_success(f"Docker Compose installed: {output}")
    else:
        print_error("Docker Compose not found")
        checks_passed = False

    # Check if Docker daemon is running
    success, _ = run_command("docker ps")
    if success:
        print_success("Docker daemon is running")
    else:
        print_error("Docker daemon not running")
        print_warning("Start Docker Desktop or run: sudo systemctl start docker")
        checks_passed = False

    # Test Docker with hello-world
    success, _ = run_command("docker run --rm hello-world")
    if success:
        print_success("Docker hello-world test passed")
    else:
        print_warning("Docker hello-world test failed")

    return checks_passed


def check_cloud_cli():
    """Check cloud CLI tools."""
    print(f"\n{Colors.BOLD}Checking Cloud CLI Tools...{Colors.END}")

    # AWS CLI
    check_command("aws", "AWS CLI", "--version")

    # gcloud
    success, _ = run_command("gcloud version")
    if success:
        print_success("gcloud installed")
    else:
        print_warning("gcloud not installed (optional)")

    # Azure CLI
    success, _ = run_command("az --version")
    if success:
        print_success("Azure CLI installed")
    else:
        print_warning("Azure CLI not installed (optional)")

    return True  # Cloud tools are optional


def check_productivity_tools():
    """Check productivity tools."""
    print(f"\n{Colors.BOLD}Checking Productivity Tools...{Colors.END}")

    tools = [
        ("tmux", "tmux"),
        ("fzf", "fzf"),
        ("rg", "ripgrep"),
        ("jq", "jq"),
        ("htop", "htop"),
        ("tree", "tree"),
    ]

    for command, name in tools:
        success, _ = run_command(f"which {command}")
        if success:
            print_success(f"{name} installed")
        else:
            print_warning(f"{name} not installed (optional)")

    return True  # These are optional


def check_vs_code():
    """Check VS Code installation."""
    print(f"\n{Colors.BOLD}Checking VS Code...{Colors.END}")

    # Check if code command exists
    success, output = run_command("code --version")
    if success:
        version = output.split('\n')[0] if output else "installed"
        print_success(f"VS Code installed: {version}")

        # Check extensions
        success, output = run_command("code --list-extensions")
        if success and output:
            extensions = output.split('\n')
            print_success(f"{len(extensions)} VS Code extensions installed")

            # Check for important extensions
            important = [
                "ms-python.python",
                "ms-azuretools.vscode-docker",
                "eamodio.gitlens"
            ]
            for ext in important:
                if ext in extensions:
                    print_success(f"  - {ext} ✓")
                else:
                    print_warning(f"  - {ext} not installed (recommended)")

        return True
    else:
        print_error("VS Code not found or 'code' command not in PATH")
        print_warning("If VS Code is installed, add to PATH:")
        print_warning("  macOS: Open VS Code → Command Palette → 'Shell Command: Install code command in PATH'")
        return False


def check_infrastructure_tools():
    """Check infrastructure tools."""
    print(f"\n{Colors.BOLD}Checking Infrastructure Tools...{Colors.END}")

    # kubectl
    check_command("kubectl", "kubectl", "version --client --short")

    # Terraform
    check_command("terraform", "Terraform", "--version")

    return True  # These are optional for now


def check_git_config():
    """Check Git configuration."""
    print(f"\n{Colors.BOLD}Checking Git Configuration...{Colors.END}")

    checks_passed = True

    # Check Git installed
    if not check_command("git", "Git", "--version"):
        return False

    # Check user name
    success, output = run_command("git config --global user.name")
    if success and output:
        print_success(f"Git user name: {output}")
    else:
        print_warning("Git user name not configured")
        print_warning("Run: git config --global user.name 'Your Name'")
        checks_passed = False

    # Check user email
    success, output = run_command("git config --global user.email")
    if success and output:
        print_success(f"Git user email: {output}")
    else:
        print_warning("Git user email not configured")
        print_warning("Run: git config --global user.email 'your@email.com'")
        checks_passed = False

    return checks_passed


def check_dotfiles():
    """Check if dotfiles exist."""
    print(f"\n{Colors.BOLD}Checking Dotfiles...{Colors.END}")

    dotfiles = [".zshrc", ".gitconfig", ".vimrc"]
    found = 0

    for dotfile in dotfiles:
        path = Path.home() / dotfile
        if path.exists():
            print_success(f"{dotfile} exists")
            found += 1
        else:
            print_warning(f"{dotfile} not found")

    if found >= 2:
        return True
    else:
        print_warning("Create dotfiles repository (Exercise 02 deliverable)")
        return False


def main():
    """Run all validations."""
    print_header("Development Environment Validation")

    system = platform.system()
    print(f"Operating System: {system}")
    print(f"Python Version: {sys.version.split()[0]}")

    results = {
        "Shell (Zsh)": check_zsh(),
        "Git": check_git_config(),
        "Python Environment": check_python(),
        "VS Code": check_vs_code(),
        "Docker": check_docker(),
        "Cloud CLI": check_cloud_cli(),
        "Productivity Tools": check_productivity_tools(),
        "Infrastructure Tools": check_infrastructure_tools(),
        "Dotfiles": check_dotfiles(),
    }

    # Summary
    print_header("Validation Summary")

    required_checks = ["Shell (Zsh)", "Git", "Python Environment", "Docker"]
    optional_checks = [k for k in results.keys() if k not in required_checks]

    required_passed = sum(1 for k in required_checks if results[k])
    optional_passed = sum(1 for k in optional_checks if results[k])

    print(f"{Colors.BOLD}Required Tools:{Colors.END}")
    for check in required_checks:
        status = "✅" if results[check] else "❌"
        print(f"  {status} {check}")

    print(f"\n{Colors.BOLD}Optional Tools:{Colors.END}")
    for check in optional_checks:
        status = "✅" if results[check] else "⚠️"
        print(f"  {status} {check}")

    print(f"\n{Colors.BOLD}Score:{Colors.END}")
    print(f"  Required: {required_passed}/{len(required_checks)}")
    print(f"  Optional: {optional_passed}/{len(optional_checks)}")

    if required_passed == len(required_checks):
        print_success("\n🎉 All required tools installed!")

        if optional_passed == len(optional_checks):
            print_success("✨ All optional tools installed too!")
        else:
            print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
            print("1. Install missing optional tools (see README.md)")
            print("2. Configure Git if not done")
            print("3. Create dotfiles repository")
            print("4. Install VS Code extensions")
            print("5. Move to Exercise 03")

        return 0
    else:
        print_error(f"\n❌ {len(required_checks) - required_passed} required tool(s) missing")
        print(f"\n{Colors.BOLD}To Do:{Colors.END}")
        for check in required_checks:
            if not results[check]:
                print(f"  - Install {check}")
        print("\nSee README.md or STEP_BY_STEP.md for installation instructions")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Validation interrupted.{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Validation error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
