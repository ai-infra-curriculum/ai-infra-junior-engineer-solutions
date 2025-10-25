#!/usr/bin/env python3
"""
Docker Compose stack manager for ML applications.
Manages multiple compose configurations and provides utilities.
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ComposeStack:
    """Represents a Docker Compose stack."""
    name: str
    file: str
    description: str
    services: List[str]


class ComposeManager:
    """Manage Docker Compose stacks."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.base_path = Path(__file__).parent

    def _run_compose_command(
        self,
        compose_file: str,
        command: List[str],
        capture_output: bool = True
    ) -> tuple:
        """Run docker-compose command."""
        cmd = ["docker-compose", "-f", compose_file] + command

        if self.verbose:
            print(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                check=False
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            print(f"Error running command: {e}")
            return 1, "", str(e)

    def up(
        self,
        stack_name: str,
        detach: bool = True,
        build: bool = False,
        scale: Optional[Dict[str, int]] = None
    ) -> bool:
        """
        Start a compose stack.

        Args:
            stack_name: Name of the stack to start
            detach: Run in detached mode
            build: Build images before starting
            scale: Dictionary of service_name: replica_count

        Returns:
            True if successful
        """
        compose_file = self._get_compose_file(stack_name)
        if not compose_file:
            print(f"Stack '{stack_name}' not found")
            return False

        cmd = ["up"]
        if detach:
            cmd.append("-d")
        if build:
            cmd.append("--build")

        # Add scaling options
        if scale:
            for service, count in scale.items():
                cmd.extend(["--scale", f"{service}={count}"])

        print(f"Starting stack: {stack_name}")
        returncode, stdout, stderr = self._run_compose_command(
            compose_file,
            cmd,
            capture_output=not self.verbose
        )

        if returncode == 0:
            print(f"✓ Stack '{stack_name}' started successfully")
            return True
        else:
            print(f"✗ Failed to start stack '{stack_name}'")
            if stderr:
                print(f"Error: {stderr}")
            return False

    def down(
        self,
        stack_name: str,
        volumes: bool = False,
        remove_orphans: bool = True
    ) -> bool:
        """
        Stop and remove a compose stack.

        Args:
            stack_name: Name of the stack to stop
            volumes: Remove named volumes
            remove_orphans: Remove orphaned containers

        Returns:
            True if successful
        """
        compose_file = self._get_compose_file(stack_name)
        if not compose_file:
            print(f"Stack '{stack_name}' not found")
            return False

        cmd = ["down"]
        if volumes:
            cmd.append("-v")
        if remove_orphans:
            cmd.append("--remove-orphans")

        print(f"Stopping stack: {stack_name}")
        returncode, stdout, stderr = self._run_compose_command(
            compose_file,
            cmd,
            capture_output=not self.verbose
        )

        if returncode == 0:
            print(f"✓ Stack '{stack_name}' stopped successfully")
            return True
        else:
            print(f"✗ Failed to stop stack '{stack_name}'")
            if stderr:
                print(f"Error: {stderr}")
            return False

    def restart(self, stack_name: str, services: Optional[List[str]] = None) -> bool:
        """
        Restart services in a stack.

        Args:
            stack_name: Name of the stack
            services: List of services to restart (None for all)

        Returns:
            True if successful
        """
        compose_file = self._get_compose_file(stack_name)
        if not compose_file:
            print(f"Stack '{stack_name}' not found")
            return False

        cmd = ["restart"]
        if services:
            cmd.extend(services)

        print(f"Restarting services in stack: {stack_name}")
        returncode, stdout, stderr = self._run_compose_command(
            compose_file,
            cmd,
            capture_output=not self.verbose
        )

        if returncode == 0:
            print(f"✓ Services restarted successfully")
            return True
        else:
            print(f"✗ Failed to restart services")
            return False

    def logs(
        self,
        stack_name: str,
        service: Optional[str] = None,
        follow: bool = False,
        tail: int = 100
    ) -> bool:
        """
        Show logs from stack services.

        Args:
            stack_name: Name of the stack
            service: Specific service (None for all)
            follow: Follow log output
            tail: Number of lines to show

        Returns:
            True if successful
        """
        compose_file = self._get_compose_file(stack_name)
        if not compose_file:
            print(f"Stack '{stack_name}' not found")
            return False

        cmd = ["logs", f"--tail={tail}"]
        if follow:
            cmd.append("-f")
        if service:
            cmd.append(service)

        returncode, stdout, stderr = self._run_compose_command(
            compose_file,
            cmd,
            capture_output=False
        )

        return returncode == 0

    def ps(self, stack_name: str) -> bool:
        """
        Show status of stack services.

        Args:
            stack_name: Name of the stack

        Returns:
            True if successful
        """
        compose_file = self._get_compose_file(stack_name)
        if not compose_file:
            print(f"Stack '{stack_name}' not found")
            return False

        returncode, stdout, stderr = self._run_compose_command(
            compose_file,
            ["ps"],
            capture_output=False
        )

        return returncode == 0

    def health_check(self, stack_name: str) -> Dict:
        """
        Check health of all services in a stack.

        Args:
            stack_name: Name of the stack

        Returns:
            Dictionary of service health statuses
        """
        compose_file = self._get_compose_file(stack_name)
        if not compose_file:
            return {}

        returncode, stdout, stderr = self._run_compose_command(
            compose_file,
            ["ps", "--format", "json"]
        )

        health_status = {}
        if returncode == 0 and stdout:
            try:
                services = json.loads(stdout)
                for service in services:
                    name = service.get("Service", "unknown")
                    state = service.get("State", "unknown")
                    health = service.get("Health", "none")
                    health_status[name] = {
                        "state": state,
                        "health": health
                    }
            except json.JSONDecodeError:
                pass

        return health_status

    def exec_command(
        self,
        stack_name: str,
        service: str,
        command: List[str],
        interactive: bool = False
    ) -> bool:
        """
        Execute command in a service container.

        Args:
            stack_name: Name of the stack
            service: Service to run command in
            command: Command to execute
            interactive: Run in interactive mode

        Returns:
            True if successful
        """
        compose_file = self._get_compose_file(stack_name)
        if not compose_file:
            print(f"Stack '{stack_name}' not found")
            return False

        cmd = ["exec"]
        if not interactive:
            cmd.append("-T")
        cmd.append(service)
        cmd.extend(command)

        returncode, stdout, stderr = self._run_compose_command(
            compose_file,
            cmd,
            capture_output=not interactive
        )

        if interactive:
            return returncode == 0
        else:
            if returncode == 0:
                print(stdout)
                return True
            else:
                print(f"Error: {stderr}")
                return False

    def _get_compose_file(self, stack_name: str) -> Optional[str]:
        """Get compose file path for stack name."""
        stacks = {
            "ml-api": "docker-compose-ml-api.yml",
            "jupyter-mlflow": "docker-compose-jupyter-mlflow.yml",
            "model-serving": "docker-compose-model-serving.yml"
        }

        filename = stacks.get(stack_name)
        if not filename:
            return None

        file_path = self.base_path / filename
        if not file_path.exists():
            return None

        return str(file_path)

    def list_stacks(self) -> List[ComposeStack]:
        """List all available stacks."""
        stacks = [
            ComposeStack(
                name="ml-api",
                file="docker-compose-ml-api.yml",
                description="ML API with PostgreSQL, Redis, and monitoring",
                services=["ml-api", "postgres", "redis", "prometheus", "grafana"]
            ),
            ComposeStack(
                name="jupyter-mlflow",
                file="docker-compose-jupyter-mlflow.yml",
                description="Jupyter Lab with MLflow tracking server",
                services=["jupyter", "mlflow", "postgres", "minio"]
            ),
            ComposeStack(
                name="model-serving",
                file="docker-compose-model-serving.yml",
                description="Load-balanced model serving with replicas",
                services=["nginx", "model-server-1", "model-server-2", "model-server-3", "redis"]
            )
        ]
        return stacks

    def validate_stack(self, stack_name: str) -> bool:
        """
        Validate compose file configuration.

        Args:
            stack_name: Name of the stack to validate

        Returns:
            True if valid
        """
        compose_file = self._get_compose_file(stack_name)
        if not compose_file:
            print(f"Stack '{stack_name}' not found")
            return False

        print(f"Validating stack: {stack_name}")
        returncode, stdout, stderr = self._run_compose_command(
            compose_file,
            ["config", "--quiet"]
        )

        if returncode == 0:
            print(f"✓ Stack '{stack_name}' configuration is valid")
            return True
        else:
            print(f"✗ Stack '{stack_name}' configuration is invalid")
            if stderr:
                print(f"Error: {stderr}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Manage Docker Compose stacks for ML applications"
    )
    parser.add_argument(
        "action",
        choices=["up", "down", "restart", "logs", "ps", "health", "exec", "list", "validate"],
        help="Action to perform"
    )
    parser.add_argument(
        "--stack",
        help="Stack name (ml-api, jupyter-mlflow, model-serving)"
    )
    parser.add_argument(
        "--service",
        help="Service name (for logs, exec)"
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build images before starting"
    )
    parser.add_argument(
        "--volumes",
        action="store_true",
        help="Remove volumes when stopping"
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Follow log output"
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=100,
        help="Number of log lines to show"
    )
    parser.add_argument(
        "--command",
        nargs="+",
        help="Command to execute (for exec action)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    manager = ComposeManager(verbose=args.verbose)

    # Handle list action
    if args.action == "list":
        stacks = manager.list_stacks()
        print("\nAvailable stacks:\n")
        for stack in stacks:
            print(f"  {stack.name}")
            print(f"    File: {stack.file}")
            print(f"    Description: {stack.description}")
            print(f"    Services: {', '.join(stack.services)}")
            print()
        sys.exit(0)

    # All other actions require --stack
    if not args.stack:
        print("Error: --stack is required for this action")
        sys.exit(1)

    # Execute action
    success = False

    if args.action == "up":
        success = manager.up(args.stack, build=args.build)
    elif args.action == "down":
        success = manager.down(args.stack, volumes=args.volumes)
    elif args.action == "restart":
        success = manager.restart(args.stack)
    elif args.action == "logs":
        success = manager.logs(
            args.stack,
            service=args.service,
            follow=args.follow,
            tail=args.tail
        )
    elif args.action == "ps":
        success = manager.ps(args.stack)
    elif args.action == "health":
        health = manager.health_check(args.stack)
        if health:
            print("\nService Health Status:\n")
            for service, status in health.items():
                print(f"  {service}: {status['state']} (health: {status['health']})")
            success = True
        else:
            print("Failed to get health status")
    elif args.action == "exec":
        if not args.service or not args.command:
            print("Error: --service and --command are required for exec")
            sys.exit(1)
        success = manager.exec_command(
            args.stack,
            args.service,
            args.command
        )
    elif args.action == "validate":
        success = manager.validate_stack(args.stack)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
