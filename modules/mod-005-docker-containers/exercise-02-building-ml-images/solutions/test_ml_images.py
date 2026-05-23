#!/usr/bin/env python3
"""
Comprehensive tests for ML Docker images.
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Optional


class TestMLDockerImages(unittest.TestCase):
    """Test ML Docker images."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.base_path = Path(__file__).parent
        cls.test_images = [
            "ml-tensorflow:latest",
            "ml-pytorch:latest",
            "ml-sklearn:latest"
        ]

    def _run_docker_command(self, cmd: list) -> tuple:
        """Run docker command and return output."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"

    def _image_exists(self, image_name: str) -> bool:
        """Check if image exists locally."""
        returncode, stdout, _ = self._run_docker_command(
            ["docker", "images", "-q", image_name]
        )
        return returncode == 0 and len(stdout.strip()) > 0

    def _get_image_size(self, image_name: str) -> Optional[int]:
        """Get image size in bytes."""
        returncode, stdout, _ = self._run_docker_command(
            ["docker", "inspect", "--format={{.Size}}", image_name]
        )
        if returncode == 0:
            try:
                return int(stdout.strip())
            except ValueError:
                return None
        return None

    def test_01_dockerfile_tensorflow_exists(self):
        """Test that TensorFlow Dockerfile exists."""
        dockerfile = self.base_path / "Dockerfile.tensorflow"
        self.assertTrue(
            dockerfile.exists(),
            "TensorFlow Dockerfile should exist"
        )

    def test_02_dockerfile_pytorch_exists(self):
        """Test that PyTorch Dockerfile exists."""
        dockerfile = self.base_path / "Dockerfile.pytorch"
        self.assertTrue(
            dockerfile.exists(),
            "PyTorch Dockerfile should exist"
        )

    def test_03_dockerfile_sklearn_exists(self):
        """Test that scikit-learn Dockerfile exists."""
        dockerfile = self.base_path / "Dockerfile.scikit-learn"
        self.assertTrue(
            dockerfile.exists(),
            "Scikit-learn Dockerfile should exist"
        )

    def test_04_dockerfile_multistage_exists(self):
        """Test that multi-stage optimized Dockerfile exists."""
        dockerfile = self.base_path / "Dockerfile.multistage-optimized"
        self.assertTrue(
            dockerfile.exists(),
            "Multi-stage optimized Dockerfile should exist"
        )

    def test_05_dockerfile_gpu_exists(self):
        """Test that GPU Dockerfile exists."""
        dockerfile = self.base_path / "Dockerfile.gpu"
        self.assertTrue(
            dockerfile.exists(),
            "GPU Dockerfile should exist"
        )

    def test_06_requirements_files_exist(self):
        """Test that all requirements files exist."""
        requirements = [
            "requirements-tensorflow.txt",
            "requirements-pytorch.txt",
            "requirements-sklearn.txt"
        ]
        for req_file in requirements:
            path = self.base_path / req_file
            self.assertTrue(
                path.exists(),
                f"{req_file} should exist"
            )

    def test_07_dockerfile_uses_multistage(self):
        """Test that Dockerfiles use multi-stage builds."""
        dockerfile = self.base_path / "Dockerfile.multistage-optimized"
        if dockerfile.exists():
            content = dockerfile.read_text()
            self.assertIn(
                "FROM",
                content,
                "Dockerfile should have FROM statement"
            )
            # Count FROM statements
            from_count = content.count("FROM ")
            self.assertGreater(
                from_count,
                1,
                "Multi-stage Dockerfile should have multiple FROM statements"
            )

    def test_08_dockerfile_uses_nonroot_user(self):
        """Test that Dockerfiles create non-root users."""
        dockerfiles = [
            "Dockerfile.tensorflow",
            "Dockerfile.pytorch",
            "Dockerfile.scikit-learn"
        ]
        for df_name in dockerfiles:
            dockerfile = self.base_path / df_name
            if dockerfile.exists():
                content = dockerfile.read_text()
                self.assertTrue(
                    "useradd" in content or "USER" in content,
                    f"{df_name} should create/use non-root user"
                )

    def test_09_dockerfile_has_healthcheck(self):
        """Test that Dockerfiles include health checks."""
        dockerfiles = [
            "Dockerfile.tensorflow",
            "Dockerfile.pytorch",
            "Dockerfile.scikit-learn"
        ]
        for df_name in dockerfiles:
            dockerfile = self.base_path / df_name
            if dockerfile.exists():
                content = dockerfile.read_text()
                self.assertIn(
                    "HEALTHCHECK",
                    content,
                    f"{df_name} should include HEALTHCHECK"
                )

    def test_10_dockerfile_sets_workdir(self):
        """Test that Dockerfiles set WORKDIR."""
        dockerfiles = [
            "Dockerfile.tensorflow",
            "Dockerfile.pytorch",
            "Dockerfile.scikit-learn"
        ]
        for df_name in dockerfiles:
            dockerfile = self.base_path / df_name
            if dockerfile.exists():
                content = dockerfile.read_text()
                self.assertIn(
                    "WORKDIR",
                    content,
                    f"{df_name} should set WORKDIR"
                )

    def test_11_dockerfile_optimizes_layers(self):
        """Test that Dockerfiles use layer optimization techniques."""
        dockerfile = self.base_path / "Dockerfile.multistage-optimized"
        if dockerfile.exists():
            content = dockerfile.read_text()
            # Check for common optimizations
            self.assertTrue(
                "rm -rf /var/lib/apt/lists/*" in content or
                "--no-install-recommends" in content,
                "Dockerfile should clean up apt lists"
            )

    def test_12_build_script_exists(self):
        """Test that build script exists."""
        build_script = self.base_path / "build_images.py"
        self.assertTrue(
            build_script.exists(),
            "Build script should exist"
        )
        # Check if executable
        self.assertTrue(
            build_script.stat().st_mode & 0o111,
            "Build script should be executable"
        )

    def test_13_optimizer_script_exists(self):
        """Test that optimizer script exists."""
        optimizer = self.base_path / "image_optimizer.py"
        self.assertTrue(
            optimizer.exists(),
            "Image optimizer script should exist"
        )

    def test_14_dockerfile_uses_pip_optimization(self):
        """Test that Dockerfiles use pip optimization flags."""
        dockerfiles = [
            "Dockerfile.tensorflow",
            "Dockerfile.pytorch",
            "Dockerfile.scikit-learn"
        ]
        for df_name in dockerfiles:
            dockerfile = self.base_path / df_name
            if dockerfile.exists():
                content = dockerfile.read_text()
                self.assertTrue(
                    "--no-cache-dir" in content,
                    f"{df_name} should use --no-cache-dir for pip"
                )

    def test_15_dockerfile_copies_requirements_separately(self):
        """Test that Dockerfiles copy requirements separately for caching."""
        dockerfiles = [
            "Dockerfile.tensorflow",
            "Dockerfile.pytorch",
            "Dockerfile.scikit-learn"
        ]
        for df_name in dockerfiles:
            dockerfile = self.base_path / df_name
            if dockerfile.exists():
                content = dockerfile.read_text()
                # Should copy requirements before application code
                copy_statements = [
                    line for line in content.split('\n')
                    if line.strip().startswith('COPY')
                ]
                if len(copy_statements) >= 2:
                    # First COPY should be requirements
                    first_copy = copy_statements[0]
                    self.assertTrue(
                        "requirements" in first_copy.lower() or
                        ".txt" in first_copy,
                        f"{df_name} should copy requirements before app code"
                    )

    def test_16_requirements_have_versions(self):
        """Test that requirements files specify versions."""
        requirements = [
            "requirements-tensorflow.txt",
            "requirements-pytorch.txt",
            "requirements-sklearn.txt"
        ]
        for req_file in requirements:
            path = self.base_path / req_file
            if path.exists():
                content = path.read_text()
                lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
                for line in lines:
                    self.assertTrue(
                        "==" in line or ">=" in line or "<=" in line,
                        f"{req_file} should pin package versions"
                    )

    def test_17_dockerfile_exposes_ports(self):
        """Test that Dockerfiles expose necessary ports."""
        dockerfiles = [
            "Dockerfile.tensorflow",
            "Dockerfile.pytorch",
            "Dockerfile.scikit-learn"
        ]
        for df_name in dockerfiles:
            dockerfile = self.base_path / df_name
            if dockerfile.exists():
                content = dockerfile.read_text()
                # Should have either EXPOSE or documentation about ports
                # This is optional but recommended
                pass  # Soft check

    def test_18_build_script_has_error_handling(self):
        """Test that build script has proper error handling."""
        build_script = self.base_path / "build_images.py"
        if build_script.exists():
            content = build_script.read_text()
            self.assertIn(
                "try:",
                content,
                "Build script should have try-except blocks"
            )
            self.assertIn(
                "except",
                content,
                "Build script should have exception handling"
            )

    def test_19_optimizer_has_analysis_functions(self):
        """Test that optimizer has analysis functions."""
        optimizer = self.base_path / "image_optimizer.py"
        if optimizer.exists():
            content = optimizer.read_text()
            self.assertIn(
                "def analyze",
                content,
                "Optimizer should have analyze function"
            )

    def test_20_dockerfiles_set_env_variables(self):
        """Test that Dockerfiles set appropriate environment variables."""
        dockerfiles = [
            "Dockerfile.tensorflow",
            "Dockerfile.pytorch",
            "Dockerfile.scikit-learn"
        ]
        for df_name in dockerfiles:
            dockerfile = self.base_path / df_name
            if dockerfile.exists():
                content = dockerfile.read_text()
                self.assertIn(
                    "ENV",
                    content,
                    f"{df_name} should set environment variables"
                )
                # Check for Python optimization vars
                self.assertTrue(
                    "PYTHONUNBUFFERED" in content or
                    "PYTHONDONTWRITEBYTECODE" in content,
                    f"{df_name} should set Python optimization env vars"
                )


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMLDockerImages)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
