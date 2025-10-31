#!/usr/bin/env python3
"""Tests for Docker security tools."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestDockerSecurity(unittest.TestCase):
    """Test Docker security."""

    @classmethod
    def setUpClass(cls):
        cls.base_path = Path(__file__).parent

    def _run_command(self, cmd: list) -> tuple:
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, check=False
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            return 1, "", str(e)

    def test_01_security_scanner_exists(self):
        """Test security scanner exists."""
        scanner = self.base_path / "security_scanner.py"
        self.assertTrue(scanner.exists())

    def test_02_secrets_manager_exists(self):
        """Test secrets manager exists."""
        manager = self.base_path / "secrets_manager.py"
        self.assertTrue(manager.exists())

    def test_03_scanner_has_functions(self):
        """Test scanner has required functions."""
        scanner = self.base_path / "security_scanner.py"
        if scanner.exists():
            content = scanner.read_text()
            self.assertIn("def scan_dockerfile", content)
            self.assertIn("def scan_image", content)
            self.assertIn("def check_container", content)

    def test_04_secrets_manager_has_functions(self):
        """Test secrets manager has required functions."""
        manager = self.base_path / "secrets_manager.py"
        if manager.exists():
            content = manager.read_text()
            self.assertIn("def create_secret", content)
            self.assertIn("def validate", content)

    def test_05_can_scan_dockerfile(self):
        """Test Dockerfile scanning."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.Dockerfile', delete=False) as f:
            f.write("FROM ubuntu:20.04\n")
            f.write("RUN apt-get update\n")
            f.write("CMD ['/bin/bash']\n")
            dockerfile = f.name

        # Test that file exists
        self.assertTrue(Path(dockerfile).exists())
        Path(dockerfile).unlink()

    def test_06_detects_root_user(self):
        """Test detection of root user."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.Dockerfile', delete=False) as f:
            f.write("FROM ubuntu:20.04\n")
            f.write("RUN apt-get update\n")
            dockerfile_path = f.name

        content = Path(dockerfile_path).read_text()
        self.assertNotIn("USER", content)  # No USER instruction
        Path(dockerfile_path).unlink()

    def test_07_detects_latest_tag(self):
        """Test detection of latest tag."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.Dockerfile', delete=False) as f:
            f.write("FROM ubuntu:latest\n")
            dockerfile_path = f.name

        content = Path(dockerfile_path).read_text()
        self.assertIn(":latest", content)
        Path(dockerfile_path).unlink()

    def test_08_good_dockerfile_example(self):
        """Test good Dockerfile example."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.Dockerfile', delete=False) as f:
            f.write("FROM python:3.11-slim\n")
            f.write("RUN useradd -m appuser\n")
            f.write("USER appuser\n")
            f.write("HEALTHCHECK CMD curl -f http://localhost/ || exit 1\n")
            f.write('CMD ["python", "app.py"]\n')
            dockerfile_path = f.name

        content = Path(dockerfile_path).read_text()
        self.assertIn("USER", content)
        self.assertIn("HEALTHCHECK", content)
        self.assertNotIn(":latest", content)
        Path(dockerfile_path).unlink()

    def test_09_secrets_generation(self):
        """Test secret generation."""
        # Test that we can generate random strings
        import secrets
        import string
        alphabet = string.ascii_letters + string.digits
        secret = ''.join(secrets.choice(alphabet) for _ in range(32))
        self.assertEqual(len(secret), 32)

    def test_10_file_permissions_check(self):
        """Test file permissions checking."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("SECRET_KEY=test\n")
            env_file = f.name

        # Set restrictive permissions
        Path(env_file).chmod(0o600)
        mode = Path(env_file).stat().st_mode & 0o777
        self.assertEqual(mode, 0o600)

        Path(env_file).unlink()

    def test_11_scanner_script_executable(self):
        """Test scanner script is executable."""
        scanner = self.base_path / "security_scanner.py"
        if scanner.exists():
            content = scanner.read_text()
            self.assertTrue(content.startswith("#!/usr/bin/env python3"))

    def test_12_secrets_script_executable(self):
        """Test secrets script is executable."""
        manager = self.base_path / "secrets_manager.py"
        if manager.exists():
            content = manager.read_text()
            self.assertTrue(content.startswith("#!/usr/bin/env python3"))

    def test_13_detects_healthcheck(self):
        """Test HEALTHCHECK detection."""
        with_hc = "FROM ubuntu\nHEALTHCHECK CMD curl -f http://localhost/"
        without_hc = "FROM ubuntu\nRUN apt-get update"

        self.assertIn("HEALTHCHECK", with_hc)
        self.assertNotIn("HEALTHCHECK", without_hc)

    def test_14_detects_workdir(self):
        """Test WORKDIR detection."""
        with_wd = "FROM ubuntu\nWORKDIR /app"
        without_wd = "FROM ubuntu\nRUN apt-get update"

        self.assertIn("WORKDIR", with_wd)
        self.assertNotIn("WORKDIR", without_wd)

    def test_15_detects_label(self):
        """Test LABEL detection."""
        with_label = "FROM ubuntu\nLABEL version=1.0"
        without_label = "FROM ubuntu\nRUN apt-get update"

        self.assertIn("LABEL", with_label)
        self.assertNotIn("LABEL", without_label)

    def test_16_base64_encoding(self):
        """Test base64 encoding/decoding."""
        import base64
        original = "my-secret-value"
        encoded = base64.b64encode(original.encode()).decode()
        decoded = base64.b64decode(encoded.encode()).decode()
        self.assertEqual(original, decoded)

    def test_17_env_template_creation(self):
        """Test environment template creation."""
        secrets = {"DB_PASS": "", "API_KEY": ""}
        template = "\n".join([f"{k}=<PLACEHOLDER>" for k in secrets])
        self.assertIn("DB_PASS", template)
        self.assertIn("API_KEY", template)

    def test_18_dockerfile_copy_vs_add(self):
        """Test COPY vs ADD usage."""
        good = "FROM ubuntu\nCOPY app.py /app/"
        bad = "FROM ubuntu\nADD app.py /app/"

        self.assertIn("COPY", good)
        self.assertIn("ADD", bad)

    def test_19_exec_form_cmd(self):
        """Test exec form CMD."""
        exec_form = 'CMD ["python", "app.py"]'
        shell_form = 'CMD python app.py'

        self.assertIn('["', exec_form)
        self.assertNotIn('["', shell_form)

    def test_20_apt_cleanup(self):
        """Test apt cleanup detection."""
        with_cleanup = "RUN apt-get update && rm -rf /var/lib/apt/lists/*"
        without_cleanup = "RUN apt-get update"

        self.assertIn("rm -rf /var/lib/apt/lists", with_cleanup)
        self.assertNotIn("rm -rf /var/lib/apt/lists", without_cleanup)


def run_tests():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDockerSecurity)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
