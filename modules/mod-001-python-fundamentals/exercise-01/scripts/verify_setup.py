#!/usr/bin/env python3
"""
Environment setup verification script.

Checks that the sentiment-classifier project is correctly set up:
- Virtual environment is activated
- Required packages are installed
- Environment variables are configured
- Project structure is correct

Usage:
    python scripts/verify_setup.py
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_header(text: str) -> None:
    """Print section header."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text.center(60)}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{GREEN}✓ {text}{RESET}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{RED}✗ {text}{RESET}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{YELLOW}⚠ {text}{RESET}")


def check_virtual_environment() -> bool:
    """Check if running inside a virtual environment."""
    print_header("Virtual Environment Check")

    in_venv = (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    )

    if in_venv:
        print_success(f"Virtual environment detected")
        print(f"  Python: {sys.executable}")
        print(f"  Prefix: {sys.prefix}")
        return True
    else:
        print_error("No virtual environment detected")
        print("  Activate your environment: source venv/bin/activate")
        return False


def check_python_version() -> bool:
    """Check Python version."""
    print_header("Python Version Check")

    version = sys.version_info
    required_major = 3
    required_minor = 11

    if version.major >= required_major and version.minor >= required_minor:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(
            f"Python {required_major}.{required_minor}+ required. "
            f"Found: {version.major}.{version.minor}.{version.micro}"
        )
        return False


def check_required_packages() -> bool:
    """Check if required packages are installed."""
    print_header("Package Installation Check")

    required_packages = [
        ("torch", "2.0.0"),
        ("transformers", "4.30.0"),
        ("pandas", "2.0.0"),
        ("numpy", "1.24.0"),
        ("scikit-learn", "1.3.0"),
        ("python-dotenv", "1.0.0"),
        ("pyyaml", "6.0.0"),
        ("pytest", "7.0.0"),
    ]

    all_installed = True

    for package_name, min_version in required_packages:
        try:
            if package_name == "scikit-learn":
                import sklearn

                package = sklearn
                import_name = "sklearn"
            elif package_name == "python-dotenv":
                import dotenv

                package = dotenv
                import_name = "dotenv"
            elif package_name == "pyyaml":
                import yaml

                package = yaml
                import_name = "yaml"
            else:
                package = __import__(package_name)
                import_name = package_name

            version = getattr(package, "__version__", "unknown")
            print_success(f"{package_name}=={version}")

        except ImportError:
            print_error(f"{package_name} not installed")
            all_installed = False

    return all_installed


def check_project_structure() -> bool:
    """Check if project structure is correct."""
    print_header("Project Structure Check")

    project_root = Path("sentiment-classifier")
    if not project_root.exists():
        project_root = Path(".")

    required_paths = [
        "src/__init__.py",
        "src/train.py",
        "src/evaluate.py",
        "src/utils/__init__.py",
        "src/utils/data_loader.py",
        "src/utils/metrics.py",
        "tests/__init__.py",
        "tests/test_data_loader.py",
        "tests/test_metrics.py",
        "configs/training_config.yaml",
        "configs/model_config.yaml",
        "requirements.txt",
        "requirements-dev.txt",
        "setup.sh",
        ".env.example",
        ".gitignore",
    ]

    all_exist = True

    for path_str in required_paths:
        path = project_root / path_str
        if path.exists():
            print_success(path_str)
        else:
            print_error(f"{path_str} not found")
            all_exist = False

    return all_exist


def check_environment_variables() -> bool:
    """Check if .env file exists and contains required variables."""
    print_header("Environment Variables Check")

    env_file = Path(".env")
    env_example = Path(".env.example")

    if not env_example.exists():
        print_warning(".env.example not found")
        return False

    if not env_file.exists():
        print_warning(".env file not found")
        print("  Create it: cp .env.example .env")
        return False

    print_success(".env file exists")

    # Try loading environment variables
    try:
        from dotenv import load_dotenv

        load_dotenv()

        required_vars = [
            "MODEL_NAME",
            "BATCH_SIZE",
            "LEARNING_RATE",
            "NUM_EPOCHS",
            "RANDOM_SEED",
        ]

        all_set = True
        for var in required_vars:
            value = os.getenv(var)
            if value:
                print_success(f"{var}={value}")
            else:
                print_warning(f"{var} not set")
                all_set = False

        return all_set

    except Exception as e:
        print_error(f"Error loading .env: {e}")
        return False


def check_directory_permissions() -> bool:
    """Check if directories are writable."""
    print_header("Directory Permissions Check")

    dirs_to_check = ["data", "models"]

    all_writable = True

    for dir_name in dirs_to_check:
        dir_path = Path(dir_name)
        if dir_path.exists():
            if os.access(dir_path, os.W_OK):
                print_success(f"{dir_name}/ is writable")
            else:
                print_error(f"{dir_name}/ is not writable")
                all_writable = False
        else:
            print_warning(f"{dir_name}/ does not exist")
            print(f"  Creating directory: {dir_name}/")
            dir_path.mkdir(parents=True, exist_ok=True)

    return all_writable


def main() -> int:
    """Run all verification checks."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{'Sentiment Classifier Setup Verification'.center(60)}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment),
        ("Required Packages", check_required_packages),
        ("Project Structure", check_project_structure),
        ("Environment Variables", check_environment_variables),
        ("Directory Permissions", check_directory_permissions),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"Error during {name} check: {e}")
            results.append((name, False))

    # Summary
    print_header("Verification Summary")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        if result:
            print_success(f"{name}: PASSED")
        else:
            print_error(f"{name}: FAILED")

    print(f"\n{BLUE}{'='*60}{RESET}")
    if passed == total:
        print(f"{GREEN}All checks passed! ({passed}/{total}){RESET}")
        print(f"{GREEN}Setup is complete and ready for training.{RESET}")
        return 0
    else:
        print(f"{RED}Some checks failed. ({passed}/{total} passed){RESET}")
        print(f"{YELLOW}Please fix the issues above and run this script again.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
