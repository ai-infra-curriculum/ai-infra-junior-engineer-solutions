#!/usr/bin/env python3
"""
Docker volume management tool for ML data persistence.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class VolumeInfo:
    """Information about a Docker volume."""
    name: str
    driver: str
    mountpoint: str
    scope: str
    labels: Dict[str, str]
    size: Optional[str] = None


class VolumeManager:
    """Manage Docker volumes."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _run_command(self, cmd: List[str]) -> tuple:
        """Run docker command."""
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False, timeout=60
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            return 1, "", str(e)

    def create_volume(
        self,
        name: str,
        driver: str = "local",
        labels: Optional[Dict[str, str]] = None,
        opts: Optional[Dict[str, str]] = None
    ) -> bool:
        """Create a Docker volume."""
        cmd = ["docker", "volume", "create", "--driver", driver]

        if labels:
            for key, value in labels.items():
                cmd.extend(["--label", f"{key}={value}"])

        if opts:
            for key, value in opts.items():
                cmd.extend(["--opt", f"{key}={value}"])

        cmd.append(name)

        if self.verbose:
            print(f"Creating volume: {' '.join(cmd)}")

        returncode, stdout, stderr = self._run_command(cmd)

        if returncode == 0:
            print(f"✓ Volume '{name}' created")
            return True
        else:
            print(f"✗ Failed to create volume '{name}': {stderr}")
            return False

    def remove_volume(self, name: str, force: bool = False) -> bool:
        """Remove a Docker volume."""
        cmd = ["docker", "volume", "rm"]
        if force:
            cmd.append("--force")
        cmd.append(name)

        returncode, stdout, stderr = self._run_command(cmd)

        if returncode == 0:
            print(f"✓ Volume '{name}' removed")
            return True
        else:
            print(f"✗ Failed to remove volume '{name}': {stderr}")
            return False

    def list_volumes(self) -> List[VolumeInfo]:
        """List all Docker volumes."""
        cmd = ["docker", "volume", "ls", "--format", "{{json .}}"]
        returncode, stdout, stderr = self._run_command(cmd)

        volumes = []
        if returncode == 0 and stdout:
            for line in stdout.strip().split('\n'):
                if line:
                    try:
                        data = json.loads(line)
                        detailed = self.inspect_volume(data['Name'])
                        if detailed:
                            volumes.append(detailed)
                    except json.JSONDecodeError:
                        pass

        return volumes

    def inspect_volume(self, name: str) -> Optional[VolumeInfo]:
        """Get detailed volume information."""
        cmd = ["docker", "volume", "inspect", name]
        returncode, stdout, stderr = self._run_command(cmd)

        if returncode == 0 and stdout:
            try:
                data = json.loads(stdout)[0]
                return VolumeInfo(
                    name=data['Name'],
                    driver=data['Driver'],
                    mountpoint=data['Mountpoint'],
                    scope=data.get('Scope', 'local'),
                    labels=data.get('Labels', {}) or {}
                )
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

        return None

    def backup_volume(self, volume_name: str, backup_path: str) -> bool:
        """Backup volume to tar archive."""
        backup_file = Path(backup_path)
        backup_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{volume_name}:/source:ro",
            "-v", f"{backup_file.parent}:/backup",
            "alpine",
            "tar", "czf", f"/backup/{backup_file.name}", "-C", "/source", "."
        ]

        if self.verbose:
            print(f"Backing up volume '{volume_name}' to {backup_path}")

        returncode, stdout, stderr = self._run_command(cmd)

        if returncode == 0:
            print(f"✓ Volume backed up to {backup_path}")
            return True
        else:
            print(f"✗ Backup failed: {stderr}")
            return False

    def restore_volume(self, volume_name: str, backup_path: str) -> bool:
        """Restore volume from tar archive."""
        backup_file = Path(backup_path)
        if not backup_file.exists():
            print(f"✗ Backup file not found: {backup_path}")
            return False

        # Create volume if it doesn't exist
        self.create_volume(volume_name)

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{volume_name}:/target",
            "-v", f"{backup_file.parent}:/backup",
            "alpine",
            "tar", "xzf", f"/backup/{backup_file.name}", "-C", "/target"
        ]

        if self.verbose:
            print(f"Restoring volume '{volume_name}' from {backup_path}")

        returncode, stdout, stderr = self._run_command(cmd)

        if returncode == 0:
            print(f"✓ Volume restored from {backup_path}")
            return True
        else:
            print(f"✗ Restore failed: {stderr}")
            return False

    def prune_volumes(self, force: bool = False) -> bool:
        """Remove unused volumes."""
        cmd = ["docker", "volume", "prune"]
        if force:
            cmd.append("--force")

        returncode, stdout, stderr = self._run_command(cmd)

        if returncode == 0:
            print("✓ Unused volumes removed")
            return True
        else:
            print("✗ Prune failed")
            return False

    def create_ml_volumes(self) -> bool:
        """Create standard ML volumes."""
        volumes = {
            "ml-models": {"purpose": "ml", "type": "models"},
            "ml-data": {"purpose": "ml", "type": "data"},
            "ml-logs": {"purpose": "ml", "type": "logs"},
            "postgres-data": {"purpose": "database", "type": "postgres"},
            "redis-data": {"purpose": "cache", "type": "redis"}
        }

        success_count = 0
        for vol_name, labels in volumes.items():
            if self.create_volume(vol_name, labels=labels):
                success_count += 1

        print(f"\nCreated {success_count}/{len(volumes)} volumes")
        return success_count == len(volumes)


def main():
    parser = argparse.ArgumentParser(description="Manage Docker volumes")
    parser.add_argument(
        "action",
        choices=["create", "remove", "list", "inspect", "backup", "restore", "prune", "setup-ml"],
        help="Action to perform"
    )
    parser.add_argument("--name", help="Volume name")
    parser.add_argument("--driver", default="local", help="Volume driver")
    parser.add_argument("--backup-path", help="Backup file path")
    parser.add_argument("--force", action="store_true", help="Force operation")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    manager = VolumeManager(verbose=args.verbose)

    success = False

    if args.action == "create":
        if not args.name:
            print("Error: --name required")
            sys.exit(1)
        success = manager.create_volume(args.name, driver=args.driver)

    elif args.action == "remove":
        if not args.name:
            print("Error: --name required")
            sys.exit(1)
        success = manager.remove_volume(args.name, force=args.force)

    elif args.action == "list":
        volumes = manager.list_volumes()
        for vol in volumes:
            print(f"{vol.name} ({vol.driver})")
        success = True

    elif args.action == "inspect":
        if not args.name:
            print("Error: --name required")
            sys.exit(1)
        vol = manager.inspect_volume(args.name)
        if vol:
            print(json.dumps(vol.__dict__, indent=2))
            success = True

    elif args.action == "backup":
        if not args.name or not args.backup_path:
            print("Error: --name and --backup-path required")
            sys.exit(1)
        success = manager.backup_volume(args.name, args.backup_path)

    elif args.action == "restore":
        if not args.name or not args.backup_path:
            print("Error: --name and --backup-path required")
            sys.exit(1)
        success = manager.restore_volume(args.name, args.backup_path)

    elif args.action == "prune":
        success = manager.prune_volumes(force=args.force)

    elif args.action == "setup-ml":
        success = manager.create_ml_volumes()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
