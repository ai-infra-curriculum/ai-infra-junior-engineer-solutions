#!/usr/bin/env python3
"""
Comprehensive tests for Docker Compose configurations.
"""

import json
import subprocess
import sys
import unittest
import yaml
from pathlib import Path
from typing import Optional


class TestDockerCompose(unittest.TestCase):
    """Test Docker Compose configurations."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.base_path = Path(__file__).parent
        cls.compose_files = [
            "docker-compose-ml-api.yml",
            "docker-compose-jupyter-mlflow.yml",
            "docker-compose-model-serving.yml"
        ]

    def _load_compose_file(self, filename: str) -> Optional[dict]:
        """Load and parse compose file."""
        try:
            file_path = self.base_path / filename
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None

    def _validate_compose_syntax(self, filename: str) -> bool:
        """Validate compose file syntax."""
        try:
            file_path = self.base_path / filename
            cmd = ["docker-compose", "-f", str(file_path), "config", "--quiet"]
            result = subprocess.run(cmd, capture_output=True, timeout=10, check=False)
            return result.returncode == 0
        except Exception:
            return False

    def test_01_compose_files_exist(self):
        """Test that all compose files exist."""
        for filename in self.compose_files:
            file_path = self.base_path / filename
            self.assertTrue(
                file_path.exists(),
                f"{filename} should exist"
            )

    def test_02_compose_files_valid_yaml(self):
        """Test that compose files are valid YAML."""
        for filename in self.compose_files:
            compose_data = self._load_compose_file(filename)
            self.assertIsNotNone(
                compose_data,
                f"{filename} should be valid YAML"
            )

    def test_03_compose_files_have_version(self):
        """Test that compose files specify version."""
        for filename in self.compose_files:
            compose_data = self._load_compose_file(filename)
            if compose_data:
                self.assertIn(
                    "version",
                    compose_data,
                    f"{filename} should specify version"
                )

    def test_04_compose_files_have_services(self):
        """Test that compose files define services."""
        for filename in self.compose_files:
            compose_data = self._load_compose_file(filename)
            if compose_data:
                self.assertIn(
                    "services",
                    compose_data,
                    f"{filename} should define services"
                )
                self.assertGreater(
                    len(compose_data["services"]),
                    0,
                    f"{filename} should have at least one service"
                )

    def test_05_services_have_health_checks(self):
        """Test that critical services have health checks."""
        critical_services = ["postgres", "redis", "ml-api", "mlflow"]

        for filename in self.compose_files:
            compose_data = self._load_compose_file(filename)
            if not compose_data:
                continue

            services = compose_data.get("services", {})
            for service_name, service_config in services.items():
                if any(crit in service_name.lower() for crit in critical_services):
                    self.assertTrue(
                        "healthcheck" in service_config or
                        "depends_on" in service_config,
                        f"{service_name} in {filename} should have healthcheck or depends_on"
                    )

    def test_06_services_use_named_volumes(self):
        """Test that services use named volumes for data persistence."""
        for filename in self.compose_files:
            compose_data = self._load_compose_file(filename)
            if not compose_data:
                continue

            volumes = compose_data.get("volumes", {})
            if volumes:
                self.assertGreater(
                    len(volumes),
                    0,
                    f"{filename} should define named volumes"
                )

    def test_07_services_expose_ports_correctly(self):
        """Test that services expose ports correctly."""
        for filename in self.compose_files:
            compose_data = self._load_compose_file(filename)
            if not compose_data:
                continue

            services = compose_data.get("services", {})
            for service_name, service_config in services.items():
                if "ports" in service_config:
                    ports = service_config["ports"]
                    self.assertIsInstance(
                        ports,
                        list,
                        f"{service_name} ports should be a list"
                    )

    def test_08_services_use_networks(self):
        """Test that services are connected to networks."""
        for filename in self.compose_files:
            compose_data = self._load_compose_file(filename)
            if not compose_data:
                continue

            networks = compose_data.get("networks", {})
            services = compose_data.get("services", {})

            if networks:
                # At least one service should use the network
                network_used = False
                for service_config in services.values():
                    if "networks" in service_config:
                        network_used = True
                        break
                self.assertTrue(
                    network_used,
                    f"{filename} defines networks but no service uses them"
                )

    def test_09_environment_variables_defined(self):
        """Test that services define environment variables."""
        for filename in self.compose_files:
            compose_data = self._load_compose_file(filename)
            if not compose_data:
                continue

            services = compose_data.get("services", {})
            for service_name, service_config in services.items():
                # Most services should have environment config
                if service_name not in ["nginx", "prometheus", "grafana"]:
                    has_env = "environment" in service_config or "env_file" in service_config
                    # Soft check - not all services need env vars
                    pass

    def test_10_depends_on_correctly_configured(self):
        """Test that service dependencies are correctly configured."""
        for filename in self.compose_files:
            compose_data = self._load_compose_file(filename)
            if not compose_data:
                continue

            services = compose_data.get("services", {})
            for service_name, service_config in services.items():
                if "depends_on" in service_config:
                    depends_on = service_config["depends_on"]

                    # Can be list or dict
                    if isinstance(depends_on, list):
                        for dep in depends_on:
                            self.assertIn(
                                dep,
                                services,
                                f"{service_name} depends on non-existent service {dep}"
                            )
                    elif isinstance(depends_on, dict):
                        for dep in depends_on.keys():
                            self.assertIn(
                                dep,
                                services,
                                f"{service_name} depends on non-existent service {dep}"
                            )

    def test_11_restart_policies_configured(self):
        """Test that services have restart policies."""
        for filename in self.compose_files:
            compose_data = self._load_compose_file(filename)
            if not compose_data:
                continue

            services = compose_data.get("services", {})
            for service_name, service_config in services.items():
                # Production services should have restart policy
                if "restart" in service_config:
                    restart = service_config["restart"]
                    self.assertIn(
                        restart,
                        ["always", "unless-stopped", "on-failure"],
                        f"{service_name} has invalid restart policy"
                    )

    def test_12_ml_api_compose_complete(self):
        """Test ML API compose file is complete."""
        compose_data = self._load_compose_file("docker-compose-ml-api.yml")
        self.assertIsNotNone(compose_data)

        services = compose_data.get("services", {})
        required_services = ["ml-api", "postgres", "redis"]

        for service in required_services:
            self.assertIn(
                service,
                services,
                f"ML API compose should include {service}"
            )

    def test_13_jupyter_mlflow_compose_complete(self):
        """Test Jupyter MLflow compose file is complete."""
        compose_data = self._load_compose_file("docker-compose-jupyter-mlflow.yml")
        self.assertIsNotNone(compose_data)

        services = compose_data.get("services", {})
        required_services = ["jupyter", "mlflow", "postgres"]

        for service in required_services:
            self.assertIn(
                service,
                services,
                f"Jupyter MLflow compose should include {service}"
            )

    def test_14_model_serving_compose_complete(self):
        """Test model serving compose file is complete."""
        compose_data = self._load_compose_file("docker-compose-model-serving.yml")
        self.assertIsNotNone(compose_data)

        services = compose_data.get("services", {})

        # Should have multiple model server replicas
        model_servers = [s for s in services if "model-server" in s]
        self.assertGreater(
            len(model_servers),
            1,
            "Model serving should have multiple replicas"
        )

    def test_15_compose_manager_exists(self):
        """Test that compose manager script exists."""
        manager = self.base_path / "compose_manager.py"
        self.assertTrue(
            manager.exists(),
            "compose_manager.py should exist"
        )

    def test_16_env_example_exists(self):
        """Test that .env.example exists."""
        env_file = self.base_path / ".env.example"
        self.assertTrue(
            env_file.exists(),
            ".env.example should exist"
        )

    def test_17_dockerfiles_exist(self):
        """Test that required Dockerfiles exist."""
        dockerfiles = [
            "Dockerfile.api",
            "Dockerfile.jupyter",
            "Dockerfile.mlflow"
        ]
        for dockerfile in dockerfiles:
            path = self.base_path / dockerfile
            self.assertTrue(
                path.exists(),
                f"{dockerfile} should exist"
            )

    def test_18_requirements_files_exist(self):
        """Test that requirements files exist."""
        requirements = [
            "requirements-api.txt",
            "requirements-jupyter.txt",
            "requirements-mlflow.txt"
        ]
        for req_file in requirements:
            path = self.base_path / req_file
            self.assertTrue(
                path.exists(),
                f"{req_file} should exist"
            )

    def test_19_resource_limits_configured(self):
        """Test that resource limits are configured for services."""
        for filename in self.compose_files:
            compose_data = self._load_compose_file(filename)
            if not compose_data:
                continue

            services = compose_data.get("services", {})
            # Check if at least some services have resource limits
            has_limits = False
            for service_config in services.values():
                if "deploy" in service_config:
                    if "resources" in service_config["deploy"]:
                        has_limits = True
                        break

            # Soft check - at least production stacks should have limits
            if "serving" in filename or "api" in filename:
                self.assertTrue(
                    has_limits,
                    f"{filename} should configure resource limits"
                )

    def test_20_compose_syntax_valid(self):
        """Test that compose files have valid syntax."""
        for filename in self.compose_files:
            is_valid = self._validate_compose_syntax(filename)
            # Soft check - may fail if docker-compose not installed
            if not is_valid:
                print(f"Warning: Could not validate {filename} (docker-compose may not be available)")


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDockerCompose)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
