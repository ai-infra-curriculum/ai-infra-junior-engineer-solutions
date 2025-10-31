#!/usr/bin/env python3
"""
Docker secrets management for ML applications.
"""

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


class SecretsManager:
    """Manage Docker secrets securely."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _run_command(self, cmd: list) -> tuple:
        """Run command."""
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False, timeout=30
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            return 1, "", str(e)

    def create_secret_file(self, name: str, value: str, output_dir: str = ".") -> bool:
        """Create a secret file."""
        secret_file = Path(output_dir) / f"{name}.secret"
        secret_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            secret_file.write_text(value)
            secret_file.chmod(0o600)  # Readable only by owner
            print(f"✓ Secret file created: {secret_file}")
            return True
        except Exception as e:
            print(f"✗ Failed to create secret: {e}")
            return False

    def create_env_template(self, secrets: Dict[str, str], output_file: str) -> bool:
        """Create .env template with placeholders."""
        template = Path(output_file)

        try:
            with template.open('w') as f:
                f.write("# Environment variables template\n")
                f.write("# Replace <PLACEHOLDER> with actual values\n\n")
                for key in secrets:
                    f.write(f"{key}=<{key.upper()}_PLACEHOLDER>\n")

            print(f"✓ Template created: {template}")
            return True
        except Exception as e:
            print(f"✗ Failed to create template: {e}")
            return False

    def validate_env_file(self, env_file: str) -> bool:
        """Check env file for security issues."""
        env_path = Path(env_file)

        if not env_path.exists():
            print(f"✗ File not found: {env_file}")
            return False

        issues = []
        content = env_path.read_text()

        # Check for placeholder values
        if "PLACEHOLDER" in content:
            issues.append("Contains placeholder values")

        # Check for common secrets
        secret_patterns = ["password", "secret", "key", "token", "api_key"]
        for line in content.split('\n'):
            if '=' in line and not line.strip().startswith('#'):
                key, value = line.split('=', 1)
                if any(pattern in key.lower() for pattern in secret_patterns):
                    if len(value.strip()) > 0 and "changeme" not in value.lower():
                        # Good - has actual value
                        pass
                    else:
                        issues.append(f"Weak value for {key}")

        # Check file permissions
        mode = env_path.stat().st_mode & 0o777
        if mode & 0o044:  # World or group readable
            issues.append("File is readable by others")

        if issues:
            print(f"✗ Security issues in {env_file}:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print(f"✓ No security issues found")
            return True

    def generate_random_secret(self, length: int = 32) -> str:
        """Generate random secret."""
        import secrets
        import string

        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))

    def encode_secret(self, value: str) -> str:
        """Base64 encode secret."""
        return base64.b64encode(value.encode()).decode()

    def decode_secret(self, encoded: str) -> str:
        """Base64 decode secret."""
        return base64.b64decode(encoded.encode()).decode()


def main():
    parser = argparse.ArgumentParser(description="Manage Docker secrets")
    parser.add_argument(
        "action",
        choices=["create-file", "create-template", "validate", "generate"],
        help="Action to perform"
    )
    parser.add_argument("--name", help="Secret name")
    parser.add_argument("--value", help="Secret value")
    parser.add_argument("--env-file", help="Environment file")
    parser.add_argument("--output", default=".", help="Output directory/file")
    parser.add_argument("--length", type=int, default=32, help="Secret length")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    manager = SecretsManager(verbose=args.verbose)

    success = False

    if args.action == "create-file":
        if not args.name or not args.value:
            print("Error: --name and --value required")
            sys.exit(1)
        success = manager.create_secret_file(args.name, args.value, args.output)

    elif args.action == "create-template":
        if not args.output:
            print("Error: --output required")
            sys.exit(1)
        secrets = {
            "DATABASE_PASSWORD": "",
            "REDIS_PASSWORD": "",
            "SECRET_KEY": "",
            "JWT_SECRET": ""
        }
        success = manager.create_env_template(secrets, args.output)

    elif args.action == "validate":
        if not args.env_file:
            print("Error: --env-file required")
            sys.exit(1)
        success = manager.validate_env_file(args.env_file)

    elif args.action == "generate":
        secret = manager.generate_random_secret(args.length)
        print(f"Generated secret: {secret}")
        success = True

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
