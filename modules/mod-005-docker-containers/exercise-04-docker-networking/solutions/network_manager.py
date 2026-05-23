#!/usr/bin/env python3
"""
Docker network management tool for ML infrastructure.
Create, manage, and troubleshoot Docker networks.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class NetworkInfo:
    """Information about a Docker network."""
    id: str
    name: str
    driver: str
    scope: str
    subnet: Optional[str]
    gateway: Optional[str]
    containers: List[str]


class NetworkManager:
    """Manage Docker networks."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _run_command(self, cmd: List[str]) -> tuple:
        """Run docker command and return result."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=30
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            return 1, "", str(e)

    def create_network(
        self,
        name: str,
        driver: str = "bridge",
        subnet: Optional[str] = None,
        gateway: Optional[str] = None,
        internal: bool = False,
        attachable: bool = True,
        labels: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Create a Docker network.

        Args:
            name: Network name
            driver: Network driver (bridge, overlay, macvlan, host, none)
            subnet: Subnet in CIDR format
            gateway: Gateway IP address
            internal: Restrict external access
            attachable: Enable manual container attachment
            labels: Metadata labels

        Returns:
            True if successful
        """
        cmd = ["docker", "network", "create"]

        # Add driver
        cmd.extend(["--driver", driver])

        # Add subnet if provided
        if subnet:
            cmd.extend(["--subnet", subnet])

        # Add gateway if provided
        if gateway:
            cmd.extend(["--gateway", gateway])

        # Add internal flag
        if internal:
            cmd.append("--internal")

        # Add attachable flag
        if attachable and driver == "overlay":
            cmd.append("--attachable")

        # Add labels
        if labels:
            for key, value in labels.items():
                cmd.extend(["--label", f"{key}={value}"])

        cmd.append(name)

        if self.verbose:
            print(f"Creating network: {' '.join(cmd)}")

        returncode, stdout, stderr = self._run_command(cmd)

        if returncode == 0:
            print(f"✓ Network '{name}' created successfully")
            return True
        else:
            print(f"✗ Failed to create network '{name}'")
            if stderr:
                print(f"Error: {stderr}")
            return False

    def remove_network(self, name: str) -> bool:
        """Remove a Docker network."""
        cmd = ["docker", "network", "rm", name]

        if self.verbose:
            print(f"Removing network: {name}")

        returncode, stdout, stderr = self._run_command(cmd)

        if returncode == 0:
            print(f"✓ Network '{name}' removed successfully")
            return True
        else:
            print(f"✗ Failed to remove network '{name}'")
            if stderr:
                print(f"Error: {stderr}")
            return False

    def list_networks(self, filters: Optional[Dict[str, str]] = None) -> List[NetworkInfo]:
        """
        List Docker networks.

        Args:
            filters: Filter networks (e.g., {"driver": "bridge"})

        Returns:
            List of NetworkInfo objects
        """
        cmd = ["docker", "network", "ls", "--format", "{{json .}}"]

        # Add filters
        if filters:
            for key, value in filters.items():
                cmd.extend(["--filter", f"{key}={value}"])

        returncode, stdout, stderr = self._run_command(cmd)

        networks = []
        if returncode == 0 and stdout:
            for line in stdout.strip().split('\n'):
                if line:
                    try:
                        data = json.loads(line)
                        # Get detailed info
                        detailed = self.inspect_network(data['Name'])
                        if detailed:
                            networks.append(detailed)
                    except json.JSONDecodeError:
                        pass

        return networks

    def inspect_network(self, name: str) -> Optional[NetworkInfo]:
        """Get detailed information about a network."""
        cmd = ["docker", "network", "inspect", name]

        returncode, stdout, stderr = self._run_command(cmd)

        if returncode == 0 and stdout:
            try:
                data = json.loads(stdout)[0]

                # Extract subnet and gateway
                subnet = None
                gateway = None
                if data.get('IPAM', {}).get('Config'):
                    config = data['IPAM']['Config'][0] if data['IPAM']['Config'] else {}
                    subnet = config.get('Subnet')
                    gateway = config.get('Gateway')

                # Extract containers
                containers = list(data.get('Containers', {}).keys())

                return NetworkInfo(
                    id=data['Id'],
                    name=data['Name'],
                    driver=data['Driver'],
                    scope=data['Scope'],
                    subnet=subnet,
                    gateway=gateway,
                    containers=containers
                )
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

        return None

    def connect_container(
        self,
        network: str,
        container: str,
        alias: Optional[str] = None,
        ip: Optional[str] = None
    ) -> bool:
        """
        Connect a container to a network.

        Args:
            network: Network name
            container: Container name or ID
            alias: Network alias
            ip: Static IP address

        Returns:
            True if successful
        """
        cmd = ["docker", "network", "connect"]

        if alias:
            cmd.extend(["--alias", alias])

        if ip:
            cmd.extend(["--ip", ip])

        cmd.extend([network, container])

        if self.verbose:
            print(f"Connecting {container} to {network}")

        returncode, stdout, stderr = self._run_command(cmd)

        if returncode == 0:
            print(f"✓ Connected {container} to {network}")
            return True
        else:
            print(f"✗ Failed to connect {container} to {network}")
            if stderr:
                print(f"Error: {stderr}")
            return False

    def disconnect_container(self, network: str, container: str, force: bool = False) -> bool:
        """Disconnect a container from a network."""
        cmd = ["docker", "network", "disconnect"]

        if force:
            cmd.append("--force")

        cmd.extend([network, container])

        if self.verbose:
            print(f"Disconnecting {container} from {network}")

        returncode, stdout, stderr = self._run_command(cmd)

        if returncode == 0:
            print(f"✓ Disconnected {container} from {network}")
            return True
        else:
            print(f"✗ Failed to disconnect {container} from {network}")
            if stderr:
                print(f"Error: {stderr}")
            return False

    def prune_networks(self, force: bool = False) -> bool:
        """Remove unused networks."""
        cmd = ["docker", "network", "prune"]

        if force:
            cmd.append("--force")

        returncode, stdout, stderr = self._run_command(cmd)

        if returncode == 0:
            print("✓ Unused networks removed")
            return True
        else:
            print("✗ Failed to prune networks")
            return False

    def test_connectivity(
        self,
        source_container: str,
        target: str,
        network: Optional[str] = None
    ) -> bool:
        """
        Test network connectivity between containers.

        Args:
            source_container: Source container name
            target: Target hostname/IP or container name
            network: Network to test (optional)

        Returns:
            True if connectivity successful
        """
        # Ping test
        cmd = [
            "docker", "exec", source_container,
            "ping", "-c", "3", "-W", "2", target
        ]

        if self.verbose:
            print(f"Testing connectivity from {source_container} to {target}")

        returncode, stdout, stderr = self._run_command(cmd)

        if returncode == 0:
            print(f"✓ Connectivity OK: {source_container} -> {target}")
            return True
        else:
            print(f"✗ Connectivity FAILED: {source_container} -> {target}")
            return False

    def create_ml_networks(self) -> bool:
        """Create standard ML infrastructure networks."""
        networks = [
            {
                "name": "ml-frontend",
                "driver": "bridge",
                "subnet": "172.20.0.0/16",
                "gateway": "172.20.0.1",
                "internal": False,
                "labels": {"purpose": "ml-api", "tier": "frontend"}
            },
            {
                "name": "ml-backend",
                "driver": "bridge",
                "subnet": "172.21.0.0/16",
                "gateway": "172.21.0.1",
                "internal": True,
                "labels": {"purpose": "ml-data", "tier": "backend"}
            },
            {
                "name": "ml-monitoring",
                "driver": "bridge",
                "subnet": "172.22.0.0/16",
                "gateway": "172.22.0.1",
                "internal": False,
                "labels": {"purpose": "monitoring", "tier": "observability"}
            }
        ]

        success_count = 0
        for net_config in networks:
            if self.create_network(**net_config):
                success_count += 1

        print(f"\nCreated {success_count}/{len(networks)} networks")
        return success_count == len(networks)

    def print_network_info(self, network_info: NetworkInfo) -> None:
        """Print network information in a formatted way."""
        print(f"\nNetwork: {network_info.name}")
        print(f"  ID: {network_info.id[:12]}")
        print(f"  Driver: {network_info.driver}")
        print(f"  Scope: {network_info.scope}")
        if network_info.subnet:
            print(f"  Subnet: {network_info.subnet}")
        if network_info.gateway:
            print(f"  Gateway: {network_info.gateway}")
        if network_info.containers:
            print(f"  Containers: {len(network_info.containers)}")
            for container in network_info.containers[:5]:
                print(f"    - {container[:12]}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage Docker networks for ML infrastructure"
    )
    parser.add_argument(
        "action",
        choices=["create", "remove", "list", "inspect", "connect", "disconnect", "prune", "setup-ml", "test"],
        help="Action to perform"
    )
    parser.add_argument(
        "--name",
        help="Network name"
    )
    parser.add_argument(
        "--driver",
        default="bridge",
        choices=["bridge", "overlay", "macvlan", "host", "none"],
        help="Network driver"
    )
    parser.add_argument(
        "--subnet",
        help="Subnet in CIDR format (e.g., 172.20.0.0/16)"
    )
    parser.add_argument(
        "--gateway",
        help="Gateway IP address"
    )
    parser.add_argument(
        "--internal",
        action="store_true",
        help="Create internal network (no external access)"
    )
    parser.add_argument(
        "--container",
        help="Container name (for connect/disconnect/test)"
    )
    parser.add_argument(
        "--target",
        help="Target container/host (for test)"
    )
    parser.add_argument(
        "--alias",
        help="Network alias (for connect)"
    )
    parser.add_argument(
        "--ip",
        help="Static IP address (for connect)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force operation"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    manager = NetworkManager(verbose=args.verbose)

    success = False

    if args.action == "create":
        if not args.name:
            print("Error: --name is required for create")
            sys.exit(1)
        success = manager.create_network(
            name=args.name,
            driver=args.driver,
            subnet=args.subnet,
            gateway=args.gateway,
            internal=args.internal
        )

    elif args.action == "remove":
        if not args.name:
            print("Error: --name is required for remove")
            sys.exit(1)
        success = manager.remove_network(args.name)

    elif args.action == "list":
        networks = manager.list_networks()
        if networks:
            for net in networks:
                manager.print_network_info(net)
            success = True
        else:
            print("No networks found")

    elif args.action == "inspect":
        if not args.name:
            print("Error: --name is required for inspect")
            sys.exit(1)
        net_info = manager.inspect_network(args.name)
        if net_info:
            manager.print_network_info(net_info)
            success = True
        else:
            print(f"Network '{args.name}' not found")

    elif args.action == "connect":
        if not args.name or not args.container:
            print("Error: --name and --container are required")
            sys.exit(1)
        success = manager.connect_container(
            network=args.name,
            container=args.container,
            alias=args.alias,
            ip=args.ip
        )

    elif args.action == "disconnect":
        if not args.name or not args.container:
            print("Error: --name and --container are required")
            sys.exit(1)
        success = manager.disconnect_container(
            network=args.name,
            container=args.container,
            force=args.force
        )

    elif args.action == "prune":
        success = manager.prune_networks(force=args.force)

    elif args.action == "setup-ml":
        success = manager.create_ml_networks()

    elif args.action == "test":
        if not args.container or not args.target:
            print("Error: --container and --target are required for test")
            sys.exit(1)
        success = manager.test_connectivity(
            source_container=args.container,
            target=args.target,
            network=args.name
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
