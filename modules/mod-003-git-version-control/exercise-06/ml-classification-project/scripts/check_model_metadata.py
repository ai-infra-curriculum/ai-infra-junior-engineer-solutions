#!/usr/bin/env python3
"""Validate model metadata files."""

import json
import sys
from pathlib import Path

REQUIRED_FIELDS = [
    "model_name",
    "version",
    "architecture",
    "framework",
    "created_at",
    "training_config",
    "metrics"
]

def validate_metadata(filepath):
    """Validate model metadata JSON."""
    with open(filepath) as f:
        metadata = json.load(f)

    missing = [field for field in REQUIRED_FIELDS if field not in metadata]

    if missing:
        print(f"Error in {filepath}: Missing fields: {missing}")
        return False

    # Check version format
    version = metadata.get("version", "")
    if not version or len(version.split(".")) != 3:
        print(f"Error in {filepath}: Invalid version format. Use X.Y.Z")
        return False

    print(f"✓ {filepath} is valid")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: check_model_metadata.py <metadata_file>")
        sys.exit(1)

    files = sys.argv[1:]
    all_valid = all(validate_metadata(f) for f in files)
    sys.exit(0 if all_valid else 1)
