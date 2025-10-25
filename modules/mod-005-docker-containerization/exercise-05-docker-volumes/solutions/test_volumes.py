#!/usr/bin/env python3
"""Tests for Docker volume management."""

import subprocess
import sys
import unittest
from pathlib import Path


class TestDockerVolumes(unittest.TestCase):
    """Test Docker volumes."""

    @classmethod
    def setUpClass(cls):
        cls.base_path = Path(__file__).parent
        cls.test_volume = "test-ml-volume"

    def _run_command(self, cmd: list) -> tuple:
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, check=False
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            return 1, "", str(e)

    def test_01_volume_manager_exists(self):
        """Test volume manager script exists."""
        manager = self.base_path / "volume_manager.py"
        self.assertTrue(manager.exists())

    def test_02_can_create_volume(self):
        """Test creating a volume."""
        returncode, _, _ = self._run_command(
            ["docker", "volume", "create", self.test_volume]
        )
        self.assertEqual(returncode, 0)
        self._run_command(["docker", "volume", "rm", self.test_volume])

    def test_03_can_list_volumes(self):
        """Test listing volumes."""
        returncode, stdout, _ = self._run_command(["docker", "volume", "ls"])
        self.assertEqual(returncode, 0)

    def test_04_can_inspect_volume(self):
        """Test inspecting volume."""
        vol = "test-inspect"
        self._run_command(["docker", "volume", "create", vol])
        returncode, stdout, _ = self._run_command(
            ["docker", "volume", "inspect", vol]
        )
        self.assertEqual(returncode, 0)
        self.assertIn("Mountpoint", stdout)
        self._run_command(["docker", "volume", "rm", vol])

    def test_05_can_remove_volume(self):
        """Test removing volume."""
        vol = "test-remove"
        self._run_command(["docker", "volume", "create", vol])
        returncode, _, _ = self._run_command(["docker", "volume", "rm", vol])
        self.assertEqual(returncode, 0)

    def test_06_volume_with_labels(self):
        """Test creating volume with labels."""
        vol = "test-labels"
        returncode, _, _ = self._run_command([
            "docker", "volume", "create",
            "--label", "env=test",
            "--label", "purpose=ml",
            vol
        ])
        self.assertEqual(returncode, 0)
        self._run_command(["docker", "volume", "rm", vol])

    def test_07_volume_drivers(self):
        """Test volume drivers."""
        vol = "test-driver"
        returncode, _, _ = self._run_command([
            "docker", "volume", "create",
            "--driver", "local",
            vol
        ])
        self.assertEqual(returncode, 0)
        self._run_command(["docker", "volume", "rm", vol])

    def test_08_volume_manager_has_functions(self):
        """Test volume manager has required functions."""
        manager = self.base_path / "volume_manager.py"
        if manager.exists():
            content = manager.read_text()
            self.assertIn("def create_volume", content)
            self.assertIn("def remove_volume", content)
            self.assertIn("def backup_volume", content)
            self.assertIn("def restore_volume", content)

    def test_09_volume_json_format(self):
        """Test volume JSON output."""
        returncode, stdout, _ = self._run_command([
            "docker", "volume", "ls", "--format", "{{json .}}"
        ])
        self.assertEqual(returncode, 0)

    def test_10_volume_filter(self):
        """Test filtering volumes."""
        returncode, _, _ = self._run_command([
            "docker", "volume", "ls", "--filter", "driver=local"
        ])
        self.assertEqual(returncode, 0)

    def test_11_manager_script_executable(self):
        """Test manager script is executable."""
        manager = self.base_path / "volume_manager.py"
        if manager.exists():
            content = manager.read_text()
            self.assertTrue(content.startswith("#!/usr/bin/env python3"))

    def test_12_prune_command_exists(self):
        """Test prune command exists."""
        returncode, _, _ = self._run_command(["docker", "volume", "prune", "--help"])
        self.assertEqual(returncode, 0)

    def test_13_volume_mountpoint(self):
        """Test volume has mountpoint."""
        vol = "test-mount"
        self._run_command(["docker", "volume", "create", vol])
        returncode, stdout, _ = self._run_command([
            "docker", "volume", "inspect", vol,
            "--format", "{{.Mountpoint}}"
        ])
        self.assertEqual(returncode, 0)
        self.assertTrue(len(stdout.strip()) > 0)
        self._run_command(["docker", "volume", "rm", vol])

    def test_14_volume_in_container(self):
        """Test using volume in container."""
        vol = "test-container"
        self._run_command(["docker", "volume", "create", vol])

        returncode, _, _ = self._run_command([
            "docker", "run", "--rm",
            "-v", f"{vol}:/data",
            "alpine", "touch", "/data/test.txt"
        ])
        self.assertEqual(returncode, 0)

        self._run_command(["docker", "volume", "rm", vol])

    def test_15_volume_persistence(self):
        """Test volume data persistence."""
        vol = "test-persist"
        self._run_command(["docker", "volume", "create", vol])

        # Write data
        self._run_command([
            "docker", "run", "--rm",
            "-v", f"{vol}:/data",
            "alpine", "sh", "-c", "echo 'test' > /data/file.txt"
        ])

        # Read data
        returncode, stdout, _ = self._run_command([
            "docker", "run", "--rm",
            "-v", f"{vol}:/data",
            "alpine", "cat", "/data/file.txt"
        ])
        self.assertEqual(returncode, 0)
        self.assertIn("test", stdout)

        self._run_command(["docker", "volume", "rm", vol])

    def test_16_readonly_volume(self):
        """Test readonly volume mount."""
        vol = "test-ro"
        self._run_command(["docker", "volume", "create", vol])

        # Try to write to readonly volume (should fail)
        returncode, _, _ = self._run_command([
            "docker", "run", "--rm",
            "-v", f"{vol}:/data:ro",
            "alpine", "touch", "/data/test.txt"
        ])
        self.assertNotEqual(returncode, 0)  # Should fail

        self._run_command(["docker", "volume", "rm", vol])

    def test_17_volume_shared_between_containers(self):
        """Test sharing volume between containers."""
        vol = "test-shared"
        self._run_command(["docker", "volume", "create", vol])

        # Container 1 writes
        self._run_command([
            "docker", "run", "--rm",
            "-v", f"{vol}:/data",
            "alpine", "sh", "-c", "echo 'shared' > /data/shared.txt"
        ])

        # Container 2 reads
        returncode, stdout, _ = self._run_command([
            "docker", "run", "--rm",
            "-v", f"{vol}:/data",
            "alpine", "cat", "/data/shared.txt"
        ])
        self.assertEqual(returncode, 0)
        self.assertIn("shared", stdout)

        self._run_command(["docker", "volume", "rm", vol])

    def test_18_anonymous_volume(self):
        """Test anonymous volume creation."""
        returncode, stdout, _ = self._run_command([
            "docker", "run", "--rm",
            "-v", "/data",
            "alpine", "ls", "/data"
        ])
        self.assertEqual(returncode, 0)

    def test_19_volume_backup_capability(self):
        """Test volume backup functionality exists."""
        manager = self.base_path / "volume_manager.py"
        if manager.exists():
            content = manager.read_text()
            self.assertIn("backup", content.lower())

    def test_20_volume_restore_capability(self):
        """Test volume restore functionality exists."""
        manager = self.base_path / "volume_manager.py"
        if manager.exists():
            content = manager.read_text()
            self.assertIn("restore", content.lower())


def run_tests():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDockerVolumes)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
