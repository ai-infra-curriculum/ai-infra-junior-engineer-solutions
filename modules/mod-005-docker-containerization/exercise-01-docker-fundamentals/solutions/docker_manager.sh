#!/usr/bin/env python3
"""
docker_manager.py - Comprehensive Docker container management tool

Description:
    Manage Docker containers with advanced features including lifecycle
    management, health monitoring, resource tracking, and automation.

Usage:
    python docker_manager.py [COMMAND] [OPTIONS]

Commands:
    list                List containers
    start NAME          Start container
    stop NAME           Stop container
    restart NAME        Restart container
    logs NAME           View container logs
    exec NAME CMD       Execute command in container
    stats               Show resource statistics
    health              Check container health
    cleanup             Remove stopped containers
    monitor             Real-time monitoring

Options:
    --all               Include stopped containers
    --format FORMAT     Output format (table, json)
    --filter KEY=VALUE  Filter containers
    --verbose           Verbose output
    --help              Display help
"""

import docker
import argparse
import json
import sys
import time
from typing import List, Dict, Optional
from datetime import datetime
from tabulate import tabulate

# Colors
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

class DockerManager:
    """Docker container management"""

    def __init__(self):
        """Initialize Docker client"""
        try:
            self.client = docker.from_env()
            self.client.ping()
        except Exception as e:
            print(f"{Colors.RED}Error connecting to Docker: {e}{Colors.RESET}")
            sys.exit(1)

    def list_containers(self, all_containers: bool = False,
                       output_format: str = 'table',
                       filters: Optional[Dict] = None) -> None:
        """List containers"""
        try:
            containers = self.client.containers.list(all=all_containers, filters=filters)

            if not containers:
                print(f"{Colors.YELLOW}No containers found{Colors.RESET}")
                return

            if output_format == 'json':
                data = []
                for container in containers:
                    data.append({
                        'id': container.short_id,
                        'name': container.name,
                        'image': container.image.tags[0] if container.image.tags else container.image.short_id,
                        'status': container.status,
                        'created': container.attrs['Created']
                    })
                print(json.dumps(data, indent=2))
            else:
                # Table format
                headers = ['ID', 'Name', 'Image', 'Status', 'Ports']
                rows = []

                for container in containers:
                    # Format ports
                    ports = []
                    if container.ports:
                        for container_port, bindings in container.ports.items():
                            if bindings:
                                for binding in bindings:
                                    ports.append(f"{binding['HostPort']}->{container_port}")

                    # Status color
                    status = container.status
                    if status == 'running':
                        status = f"{Colors.GREEN}{status}{Colors.RESET}"
                    elif status == 'exited':
                        status = f"{Colors.RED}{status}{Colors.RESET}"
                    else:
                        status = f"{Colors.YELLOW}{status}{Colors.RESET}"

                    rows.append([
                        container.short_id,
                        container.name,
                        container.image.tags[0] if container.image.tags else container.image.short_id[:12],
                        status,
                        ', '.join(ports) if ports else '-'
                    ])

                print(tabulate(rows, headers=headers, tablefmt='grid'))

        except Exception as e:
            print(f"{Colors.RED}Error listing containers: {e}{Colors.RESET}")

    def start_container(self, name: str) -> None:
        """Start container"""
        try:
            container = self.client.containers.get(name)
            container.start()
            print(f"{Colors.GREEN}✓ Started container: {name}{Colors.RESET}")
        except docker.errors.NotFound:
            print(f"{Colors.RED}Container not found: {name}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}Error starting container: {e}{Colors.RESET}")

    def stop_container(self, name: str, timeout: int = 10) -> None:
        """Stop container"""
        try:
            container = self.client.containers.get(name)
            container.stop(timeout=timeout)
            print(f"{Colors.GREEN}✓ Stopped container: {name}{Colors.RESET}")
        except docker.errors.NotFound:
            print(f"{Colors.RED}Container not found: {name}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}Error stopping container: {e}{Colors.RESET}")

    def restart_container(self, name: str) -> None:
        """Restart container"""
        try:
            container = self.client.containers.get(name)
            container.restart()
            print(f"{Colors.GREEN}✓ Restarted container: {name}{Colors.RESET}")
        except docker.errors.NotFound:
            print(f"{Colors.RED}Container not found: {name}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}Error restarting container: {e}{Colors.RESET}")

    def view_logs(self, name: str, follow: bool = False, tail: int = 100) -> None:
        """View container logs"""
        try:
            container = self.client.containers.get(name)

            if follow:
                for line in container.logs(stream=True, follow=True, tail=tail):
                    print(line.decode('utf-8').strip())
            else:
                logs = container.logs(tail=tail).decode('utf-8')
                print(logs)

        except docker.errors.NotFound:
            print(f"{Colors.RED}Container not found: {name}{Colors.RESET}")
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Stopped following logs{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}Error viewing logs: {e}{Colors.RESET}")

    def exec_command(self, name: str, command: str) -> None:
        """Execute command in container"""
        try:
            container = self.client.containers.get(name)
            result = container.exec_run(command)
            print(result.output.decode('utf-8'))
        except docker.errors.NotFound:
            print(f"{Colors.RED}Container not found: {name}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}Error executing command: {e}{Colors.RESET}")

    def show_stats(self) -> None:
        """Show resource statistics"""
        try:
            containers = self.client.containers.list()

            if not containers:
                print(f"{Colors.YELLOW}No running containers{Colors.RESET}")
                return

            headers = ['Container', 'CPU %', 'Memory', 'Net I/O', 'Block I/O']
            rows = []

            for container in containers:
                stats = container.stats(stream=False)

                # CPU percentage
                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                           stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                              stats['precpu_stats']['system_cpu_usage']
                cpu_percent = (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100

                # Memory
                mem_usage = stats['memory_stats']['usage'] / (1024 * 1024)  # MB
                mem_limit = stats['memory_stats']['limit'] / (1024 * 1024)  # MB
                mem_percent = (mem_usage / mem_limit) * 100

                # Network I/O
                net_rx = stats['networks']['eth0']['rx_bytes'] / (1024 * 1024)  # MB
                net_tx = stats['networks']['eth0']['tx_bytes'] / (1024 * 1024)  # MB

                # Block I/O
                blk_read = sum([x['value'] for x in stats['blkio_stats']['io_service_bytes_recursive'] if x['op'] == 'Read']) / (1024 * 1024)
                blk_write = sum([x['value'] for x in stats['blkio_stats']['io_service_bytes_recursive'] if x['op'] == 'Write']) / (1024 * 1024)

                rows.append([
                    container.name,
                    f"{cpu_percent:.2f}%",
                    f"{mem_usage:.1f}MB / {mem_limit:.1f}MB ({mem_percent:.1f}%)",
                    f"↓ {net_rx:.2f}MB ↑ {net_tx:.2f}MB",
                    f"↓ {blk_read:.2f}MB ↑ {blk_write:.2f}MB"
                ])

            print(tabulate(rows, headers=headers, tablefmt='grid'))

        except Exception as e:
            print(f"{Colors.RED}Error getting stats: {e}{Colors.RESET}")

    def check_health(self) -> None:
        """Check container health"""
        try:
            containers = self.client.containers.list()

            headers = ['Container', 'Status', 'Health', 'Restarts']
            rows = []

            for container in containers:
                # Health status
                health_status = container.attrs.get('State', {}).get('Health', {}).get('Status', 'N/A')

                if health_status == 'healthy':
                    health = f"{Colors.GREEN}{health_status}{Colors.RESET}"
                elif health_status == 'unhealthy':
                    health = f"{Colors.RED}{health_status}{Colors.RESET}"
                else:
                    health = f"{Colors.YELLOW}{health_status}{Colors.RESET}"

                # Restart count
                restart_count = container.attrs['RestartCount']

                rows.append([
                    container.name,
                    container.status,
                    health,
                    restart_count
                ])

            print(tabulate(rows, headers=headers, tablefmt='grid'))

        except Exception as e:
            print(f"{Colors.RED}Error checking health: {e}{Colors.RESET}")

    def cleanup(self, dry_run: bool = False) -> None:
        """Remove stopped containers"""
        try:
            containers = self.client.containers.list(all=True, filters={'status': 'exited'})

            if not containers:
                print(f"{Colors.YELLOW}No stopped containers to remove{Colors.RESET}")
                return

            print(f"Found {len(containers)} stopped container(s):")
            for container in containers:
                print(f"  - {container.name} ({container.short_id})")

            if dry_run:
                print(f"\n{Colors.YELLOW}Dry run - no containers removed{Colors.RESET}")
                return

            confirm = input("\nRemove these containers? (y/N): ")
            if confirm.lower() == 'y':
                for container in containers:
                    container.remove()
                    print(f"{Colors.GREEN}✓ Removed: {container.name}{Colors.RESET}")
            else:
                print(f"{Colors.YELLOW}Cancelled{Colors.RESET}")

        except Exception as e:
            print(f"{Colors.RED}Error during cleanup: {e}{Colors.RESET}")

    def monitor(self, interval: int = 2) -> None:
        """Real-time monitoring"""
        try:
            while True:
                # Clear screen
                print('\033[2J\033[H', end='')

                print(f"{Colors.BOLD}{Colors.CYAN}Docker Container Monitor{Colors.RESET}")
                print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 80 + "\n")

                self.show_stats()

                print(f"\n{Colors.CYAN}Press Ctrl+C to stop{Colors.RESET}")

                time.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Monitoring stopped{Colors.RESET}")

def main():
    parser = argparse.ArgumentParser(
        description='Docker container management tool',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # List command
    list_parser = subparsers.add_parser('list', help='List containers')
    list_parser.add_argument('--all', action='store_true', help='Include stopped containers')
    list_parser.add_argument('--format', choices=['table', 'json'], default='table')

    # Start command
    start_parser = subparsers.add_parser('start', help='Start container')
    start_parser.add_argument('name', help='Container name')

    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop container')
    stop_parser.add_argument('name', help='Container name')
    stop_parser.add_argument('--timeout', type=int, default=10)

    # Restart command
    restart_parser = subparsers.add_parser('restart', help='Restart container')
    restart_parser.add_argument('name', help='Container name')

    # Logs command
    logs_parser = subparsers.add_parser('logs', help='View logs')
    logs_parser.add_argument('name', help='Container name')
    logs_parser.add_argument('-f', '--follow', action='store_true')
    logs_parser.add_argument('--tail', type=int, default=100)

    # Exec command
    exec_parser = subparsers.add_parser('exec', help='Execute command')
    exec_parser.add_argument('name', help='Container name')
    exec_parser.add_argument('command', nargs='+', help='Command to execute')

    # Stats command
    subparsers.add_parser('stats', help='Show resource statistics')

    # Health command
    subparsers.add_parser('health', help='Check container health')

    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Remove stopped containers')
    cleanup_parser.add_argument('--dry-run', action='store_true')

    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Real-time monitoring')
    monitor_parser.add_argument('--interval', type=int, default=2)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    manager = DockerManager()

    # Execute command
    if args.command == 'list':
        manager.list_containers(all_containers=args.all, output_format=args.format)
    elif args.command == 'start':
        manager.start_container(args.name)
    elif args.command == 'stop':
        manager.stop_container(args.name, timeout=args.timeout)
    elif args.command == 'restart':
        manager.restart_container(args.name)
    elif args.command == 'logs':
        manager.view_logs(args.name, follow=args.follow, tail=args.tail)
    elif args.command == 'exec':
        manager.exec_command(args.name, ' '.join(args.command))
    elif args.command == 'stats':
        manager.show_stats()
    elif args.command == 'health':
        manager.check_health()
    elif args.command == 'cleanup':
        manager.cleanup(dry_run=args.dry_run)
    elif args.command == 'monitor':
        manager.monitor(interval=args.interval)

if __name__ == '__main__':
    main()
