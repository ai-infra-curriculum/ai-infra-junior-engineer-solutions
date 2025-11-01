#!/usr/bin/env python3
"""Validate experiment configuration files."""

import yaml
import sys

REQUIRED_SECTIONS = ["experiment", "model", "data", "training"]

def validate_experiment(filepath):
    """Validate experiment YAML."""
    with open(filepath) as f:
        config = yaml.safe_load(f)

    missing = [s for s in REQUIRED_SECTIONS if s not in config]

    if missing:
        print(f"Error in {filepath}: Missing sections: {missing}")
        return False

    # Check experiment ID format
    exp_id = config.get("experiment", {}).get("id")
    if not exp_id or not exp_id.startswith("exp-"):
        print(f"Error in {filepath}: Invalid experiment ID format. Use exp-XXX")
        return False

    print(f"✓ {filepath} is valid")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: validate_experiment.py <experiment_file>")
        sys.exit(1)

    files = sys.argv[1:]
    all_valid = all(validate_experiment(f) for f in files)
    sys.exit(0 if all_valid else 1)
