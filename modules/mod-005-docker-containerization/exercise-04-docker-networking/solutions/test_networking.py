#!/usr/bin/env python3
"""
Tests for Docker networking functionality.
"""

import subprocess
import sys
import unittest
from pathlib import Path


class TestDockerNetworking(unittest.TestCase):
    """Test Docker networking."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.base_path = Path(__file__).parent
        cls.test_network = "test-ml-network"

    def _run_command(self, cmd: list) -> tuple:
        """Run command and return result."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            return 1, "", str(e)

    def test_01_network_manager_exists(self):
        """Test that network manager script exists."""
        manager = self.base_path / "network_manager.py"
        self.assertTrue(manager.exists())

    def test_02_network_demos_exist(self):
        """Test that network demos script exists."""
        demos = self.base_path / "network_demos.sh"
        self.assertTrue(demos.exists())

    def test_03_can_create_bridge_network(self):
        """Test creating a bridge network."""
        returncode, _, _ = self._run_command([
            "docker", "network", "create",
            "--driver", "bridge",
            self.test_network
        ])
        self.assertEqual(returncode, 0)

        # Cleanup
        self._run_command(["docker", "network", "rm", self.test_network])

    def test_04_can_list_networks(self):
        """Test listing networks."""
        returncode, stdout, _ = self._run_command([
            "docker", "network", "ls"
        ])
        self.assertEqual(returncode, 0)
        self.assertIn("bridge", stdout)

    def test_05_can_inspect_network(self):
        """Test inspecting network."""
        returncode, stdout, _ = self._run_command([
            "docker", "network", "inspect", "bridge"
        ])
        self.assertEqual(returncode, 0)
        self.assertIn("bridge", stdout)

    def test_06_default_networks_exist(self):
        """Test that default networks exist."""
        returncode, stdout, _ = self._run_command([
            "docker", "network", "ls", "--format", "{{.Name}}"
        ])
        self.assertEqual(returncode, 0)
        networks = stdout.strip().split('\n')
        self.assertIn("bridge", networks)
        self.assertIn("host", networks)
        self.assertIn("none", networks)

    def test_07_network_has_subnet(self):
        """Test that custom network has subnet."""
        net_name = "test-subnet-net"
        self._run_command([
            "docker", "network", "create",
            "--subnet", "172.30.0.0/16",
            net_name
        ])

        returncode, stdout, _ = self._run_command([
            "docker", "network", "inspect", net_name
        ])

        self.assertEqual(returncode, 0)
        self.assertIn("172.30.0.0/16", stdout)

        # Cleanup
        self._run_command(["docker", "network", "rm", net_name])

    def test_08_network_manager_has_functions(self):
        """Test that network manager has required functions."""
        manager = self.base_path / "network_manager.py"
        if manager.exists():
            content = manager.read_text()
            self.assertIn("def create_network", content)
            self.assertIn("def remove_network", content)
            self.assertIn("def list_networks", content)

    def test_09_can_create_internal_network(self):
        """Test creating internal network."""
        net_name = "test-internal"
        returncode, _, _ = self._run_command([
            "docker", "network", "create",
            "--internal",
            net_name
        ])
        self.assertEqual(returncode, 0)

        # Cleanup
        self._run_command(["docker", "network", "rm", net_name])

    def test_10_network_driver_types(self):
        """Test different network driver types."""
        drivers = ["bridge"]  # Only test bridge in unit tests
        for driver in drivers:
            net_name = f"test-{driver}"
            returncode, _, _ = self._run_command([
                "docker", "network", "create",
                "--driver", driver,
                net_name
            ])
            self.assertEqual(returncode, 0)
            self._run_command(["docker", "network", "rm", net_name])

    def test_11_network_labels(self):
        """Test adding labels to network."""
        net_name = "test-labels"
        returncode, _, _ = self._run_command([
            "docker", "network", "create",
            "--label", "env=test",
            "--label", "purpose=ml",
            net_name
        ])
        self.assertEqual(returncode, 0)

        returncode, stdout, _ = self._run_command([
            "docker", "network", "inspect", net_name
        ])
        self.assertIn("env", stdout)

        # Cleanup
        self._run_command(["docker", "network", "rm", net_name])

    def test_12_network_gateway(self):
        """Test custom gateway configuration."""
        net_name = "test-gateway"
        returncode, _, _ = self._run_command([
            "docker", "network", "create",
            "--subnet", "172.31.0.0/16",
            "--gateway", "172.31.0.1",
            net_name
        ])
        self.assertEqual(returncode, 0)

        returncode, stdout, _ = self._run_command([
            "docker", "network", "inspect", net_name
        ])
        self.assertIn("172.31.0.1", stdout)

        # Cleanup
        self._run_command(["docker", "network", "rm", net_name])

    def test_13_can_remove_network(self):
        """Test removing network."""
        net_name = "test-remove"
        self._run_command([
            "docker", "network", "create", net_name
        ])

        returncode, _, _ = self._run_command([
            "docker", "network", "rm", net_name
        ])
        self.assertEqual(returncode, 0)

    def test_14_network_json_format(self):
        """Test network output in JSON format."""
        returncode, stdout, _ = self._run_command([
            "docker", "network", "ls", "--format", "{{json .}}"
        ])
        self.assertEqual(returncode, 0)
        self.assertIn('"Name"', stdout)

    def test_15_network_filter(self):
        """Test filtering networks."""
        returncode, stdout, _ = self._run_command([
            "docker", "network", "ls",
            "--filter", "driver=bridge"
        ])
        self.assertEqual(returncode, 0)

    def test_16_demos_script_executable(self):
        """Test that demos script is executable."""
        demos = self.base_path / "network_demos.sh"
        if demos.exists():
            self.assertTrue(demos.stat().st_mode & 0o111)

    def test_17_manager_script_executable(self):
        """Test that manager script is executable."""
        manager = self.base_path / "network_manager.py"
        if manager.exists():
            # Should be executable or have shebang
            content = manager.read_text()
            self.assertTrue(
                content.startswith("#!/usr/bin/env python3") or
                manager.stat().st_mode & 0o111
            )

    def test_18_network_prune_command(self):
        """Test network prune (dry run)."""
        # Just verify command exists
        returncode, _, _ = self._run_command([
            "docker", "network", "prune", "--help"
        ])
        self.assertEqual(returncode, 0)

    def test_19_network_connect_command(self):
        """Test network connect command format."""
        # Just verify command exists
        returncode, _, _ = self._run_command([
            "docker", "network", "connect", "--help"
        ])
        self.assertEqual(returncode, 0)

    def test_20_network_disconnect_command(self):
        """Test network disconnect command format."""
        # Just verify command exists
        returncode, _, _ = self._run_command([
            "docker", "network", "disconnect", "--help"
        ])
        self.assertEqual(returncode, 0)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDockerNetworking)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
