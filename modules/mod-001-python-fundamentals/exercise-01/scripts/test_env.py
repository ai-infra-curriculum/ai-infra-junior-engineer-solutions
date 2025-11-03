#!/usr/bin/env python3
"""
Test environment variable loading.

Verifies that environment variables are correctly loaded from .env file
and that python-dotenv is working properly.

Usage:
    python scripts/test_env.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv


def main():
    """Test environment variable loading."""
    print("="*60)
    print("Environment Variable Loading Test")
    print("="*60)
    print()

    # Check if .env exists
    env_file = Path(".env")
    if not env_file.exists():
        print("✗ .env file not found!")
        print("  Create it: cp .env.example .env")
        return 1

    print("✓ .env file found")
    print()

    # Load environment variables
    print("Loading environment variables from .env...")
    load_dotenv()
    print("✓ Environment variables loaded")
    print()

    # Test specific variables
    test_vars = [
        "MODEL_NAME",
        "BATCH_SIZE",
        "LEARNING_RATE",
        "NUM_EPOCHS",
        "RANDOM_SEED",
        "DEVICE",
        "DATA_PATH",
        "MODEL_OUTPUT_PATH",
    ]

    print("Environment Variables:")
    print("-" * 60)

    all_set = True
    for var in test_vars:
        value = os.getenv(var)
        if value:
            print(f"✓ {var}: {value}")
        else:
            print(f"✗ {var}: NOT SET")
            all_set = False

    print("-" * 60)
    print()

    # Additional environment info
    print("Python Environment:")
    print(f"  Python: {sys.version}")
    print(f"  Executable: {sys.executable}")
    print(f"  Prefix: {sys.prefix}")
    print()

    if all_set:
        print("="*60)
        print("✓ All environment variables are set!")
        print("="*60)
        return 0
    else:
        print("="*60)
        print("✗ Some environment variables are missing.")
        print("  Edit .env and set the missing variables.")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
