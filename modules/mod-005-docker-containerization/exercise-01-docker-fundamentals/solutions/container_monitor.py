#!/usr/bin/env python3
"""
container_monitor.py - Container health and performance monitoring

Description:
    Real-time monitoring of Docker containers with health checks,
    resource tracking, alerting, and dashboard visualization.

Usage:
    python container_monitor.py [OPTIONS]

Options:
    --interval SECONDS  Monitoring interval (default: 5)
    --once              Run once and exit
    --alerts            Enable alerts
    --threshold FILE    Load thresholds from file
    --export FILE       Export metrics to file
    --verbose           Verbose output
    --help              Display help
"""

import docker
import argparse
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque

# Colors
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class ContainerMonitor:
    """Monitor container health and performance"""

    def __init__(self, interval: float = 5.0, alerts_enabled: bool = False,
                 thresholds: Optional[Dict] = None):
        """Initialize monitor"""
        try:
            self.client = docker.from_env()
            self.client.ping()
        except Exception as e:
            print(f"{Colors.RED}Error connecting to Docker: {e}{Colors.RESET}")
            sys.exit(1)

        self.interval = interval
        self.alerts_enabled = alerts_enabled
        self.thresholds = thresholds or {
            'cpu_percent': 80.0,
            'memory_percent': 80.0,
            'restart_count': 5
        }

        # Metrics history
        self.metrics_history = {}
        self.alerts = []
        self.start_time = time.time()

    def get_container_stats(self, container) -> Optional[Dict]:
        """Get container statistics"""
        try:
            stats = container.stats(stream=False)

            # CPU percentage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']

            cpu_count = len(stats['cpu_stats']['cpu_usage'].get('percpu_usage', [1]))
            cpu_percent = 0.0
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * cpu_count * 100.0

            # Memory
            mem_usage = stats['memory_stats'].get('usage', 0)
            mem_limit = stats['memory_stats'].get('limit', 1)
            mem_percent = (mem_usage / mem_limit) * 100.0 if mem_limit > 0 else 0.0

            # Network I/O
            net_stats = stats.get('networks', {}).get('eth0', {})
            net_rx_bytes = net_stats.get('rx_bytes', 0)
            net_tx_bytes = net_stats.get('tx_bytes', 0)

            # Block I/O
            blkio_stats = stats.get('blkio_stats', {}).get('io_service_bytes_recursive', [])
            blk_read = sum(x['value'] for x in blkio_stats if x.get('op') == 'Read')
            blk_write = sum(x['value'] for x in blkio_stats if x.get('op') == 'Write')

            return {
                'cpu_percent': cpu_percent,
                'memory_usage': mem_usage,
                'memory_limit': mem_limit,
                'memory_percent': mem_percent,
                'network_rx': net_rx_bytes,
                'network_tx': net_tx_bytes,
                'block_read': blk_read,
                'block_write': blk_write
            }

        except Exception as e:
            return None

    def get_container_health(self, container) -> Dict:
        """Get container health status"""
        try:
            attrs = container.attrs
            state = attrs.get('State', {})

            health_status = state.get('Health', {}).get('Status', 'none')
            restart_count = state.get('RestartCount', 0)
            started_at = state.get('StartedAt', '')
            status = container.status

            return {
                'status': status,
                'health': health_status,
                'restart_count': restart_count,
                'started_at': started_at
            }

        except Exception as e:
            return {
                'status': 'error',
                'health': 'unknown',
                'restart_count': 0,
                'started_at': ''
            }

    def check_alerts(self, container_name: str, stats: Dict, health: Dict) -> None:
        """Check for alert conditions"""
        if not self.alerts_enabled:
            return

        alerts = []

        # CPU alert
        if stats['cpu_percent'] > self.thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {stats['cpu_percent']:.1f}%")

        # Memory alert
        if stats['memory_percent'] > self.thresholds['memory_percent']:
            alerts.append(f"High memory usage: {stats['memory_percent']:.1f}%")

        # Restart alert
        if health['restart_count'] > self.thresholds['restart_count']:
            alerts.append(f"High restart count: {health['restart_count']}")

        # Health alert
        if health['health'] == 'unhealthy':
            alerts.append("Container unhealthy")

        # Store alerts
        for alert in alerts:
            alert_msg = f"{container_name}: {alert}"
            if alert_msg not in self.alerts:
                self.alerts.append(alert_msg)

    def monitor_once(self) -> Dict[str, Dict]:
        """Monitor all containers once"""
        containers = self.client.containers.list()
        results = {}

        for container in containers:
            stats = self.get_container_stats(container)
            health = self.get_container_health(container)

            if stats:
                # Check alerts
                self.check_alerts(container.name, stats, health)

                # Store metrics
                results[container.name] = {
                    'stats': stats,
                    'health': health,
                    'timestamp': datetime.now().isoformat()
                }

                # Update history
                if container.name not in self.metrics_history:
                    self.metrics_history[container.name] = {
                        'cpu': deque(maxlen=60),
                        'memory': deque(maxlen=60)
                    }

                self.metrics_history[container.name]['cpu'].append(stats['cpu_percent'])
                self.metrics_history[container.name]['memory'].append(stats['memory_percent'])

        return results

    def display_dashboard(self, metrics: Dict[str, Dict]) -> None:
        """Display monitoring dashboard"""
        # Clear screen
        print('\033[2J\033[H', end='')

        # Header
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 100}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}Container Health Monitor{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 100}{Colors.RESET}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
              f"Monitoring {len(metrics)} container(s) | "
              f"Interval: {self.interval}s\n")

        if not metrics:
            print(f"{Colors.YELLOW}No running containers to monitor{Colors.RESET}")
            return

        # Container details
        for name, data in sorted(metrics.items()):
            stats = data['stats']
            health = data['health']

            # Status color
            if health['status'] == 'running':
                status_color = Colors.GREEN
            else:
                status_color = Colors.RED

            # Health color
            if health['health'] == 'healthy':
                health_color = Colors.GREEN
            elif health['health'] == 'unhealthy':
                health_color = Colors.RED
            else:
                health_color = Colors.YELLOW

            print(f"{Colors.BOLD}{name}{Colors.RESET}")
            print(f"  Status: {status_color}{health['status']}{Colors.RESET} | "
                  f"Health: {health_color}{health['health']}{Colors.RESET} | "
                  f"Restarts: {health['restart_count']}")

            # CPU
            cpu_percent = stats['cpu_percent']
            cpu_color = Colors.GREEN if cpu_percent < 50 else Colors.YELLOW if cpu_percent < 80 else Colors.RED
            cpu_bar = self._create_bar(cpu_percent, 100)
            print(f"  CPU:    {cpu_color}{cpu_bar} {cpu_percent:6.2f}%{Colors.RESET}")

            # Memory
            mem_percent = stats['memory_percent']
            mem_color = Colors.GREEN if mem_percent < 50 else Colors.YELLOW if mem_percent < 80 else Colors.RED
            mem_bar = self._create_bar(mem_percent, 100)
            mem_mb = stats['memory_usage'] / (1024 * 1024)
            mem_limit_mb = stats['memory_limit'] / (1024 * 1024)
            print(f"  Memory: {mem_color}{mem_bar} {mem_percent:6.2f}%{Colors.RESET} "
                  f"({mem_mb:.0f}MB / {mem_limit_mb:.0f}MB)")

            # Network
            net_rx_mb = stats['network_rx'] / (1024 * 1024)
            net_tx_mb = stats['network_tx'] / (1024 * 1024)
            print(f"  Network: ↓ {net_rx_mb:8.2f}MB  ↑ {net_tx_mb:8.2f}MB")

            # Disk
            blk_read_mb = stats['block_read'] / (1024 * 1024)
            blk_write_mb = stats['block_write'] / (1024 * 1024)
            print(f"  Disk:    ↓ {blk_read_mb:8.2f}MB  ↑ {blk_write_mb:8.2f}MB")

            # Trend indicators if history available
            if name in self.metrics_history:
                cpu_history = list(self.metrics_history[name]['cpu'])
                mem_history = list(self.metrics_history[name]['memory'])

                if len(cpu_history) > 1:
                    cpu_trend = self._get_trend(cpu_history)
                    mem_trend = self._get_trend(mem_history)
                    print(f"  Trend:   CPU {cpu_trend}  Memory {mem_trend}")

            print()

        # Alerts
        if self.alerts:
            print(f"{Colors.BOLD}{Colors.RED}Active Alerts:{Colors.RESET}")
            for alert in self.alerts[-10:]:  # Show last 10
                print(f"  ⚠️  {alert}")
            print()

        # Footer
        print(f"{Colors.CYAN}{'=' * 100}{Colors.RESET}")
        print(f"Press Ctrl+C to stop monitoring")

    def _create_bar(self, value: float, max_value: float, width: int = 20) -> str:
        """Create a text-based progress bar"""
        filled = int((value / max_value) * width)
        bar = '█' * filled + '░' * (width - filled)
        return bar

    def _get_trend(self, values: List[float]) -> str:
        """Get trend indicator from values"""
        if len(values) < 2:
            return '→'

        recent_avg = sum(values[-5:]) / min(5, len(values[-5:]))
        older_avg = sum(values[-10:-5]) / min(5, len(values[-10:-5])) if len(values) > 5 else recent_avg

        if recent_avg > older_avg * 1.1:
            return '↗ Increasing'
        elif recent_avg < older_avg * 0.9:
            return '↘ Decreasing'
        else:
            return '→ Stable'

    def export_metrics(self, filename: str, metrics: Dict[str, Dict]) -> None:
        """Export metrics to JSON file"""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'alerts': self.alerts
            }

            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)

            print(f"\n{Colors.GREEN}Metrics exported to: {filename}{Colors.RESET}")

        except Exception as e:
            print(f"{Colors.RED}Error exporting metrics: {e}{Colors.RESET}")

    def monitor_continuous(self) -> None:
        """Continuous monitoring loop"""
        try:
            while True:
                # Get metrics
                metrics = self.monitor_once()

                # Display dashboard
                self.display_dashboard(metrics)

                # Wait for next interval
                time.sleep(self.interval)

        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Monitoring stopped by user{Colors.RESET}")

            # Summary
            uptime = time.time() - self.start_time
            print(f"\nMonitoring Summary:")
            print(f"  Duration: {uptime / 60:.1f} minutes")
            print(f"  Containers monitored: {len(self.metrics_history)}")
            if self.alerts:
                print(f"  Total alerts: {len(self.alerts)}")


def main():
    parser = argparse.ArgumentParser(
        description='Container health and performance monitoring',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--interval', type=float, default=5.0,
                       help='Monitoring interval in seconds (default: 5.0)')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit')
    parser.add_argument('--alerts', action='store_true',
                       help='Enable alerts')
    parser.add_argument('--threshold', type=str, default=None,
                       help='Load thresholds from JSON file')
    parser.add_argument('--export', type=str, default=None,
                       help='Export metrics to JSON file')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Load thresholds if provided
    thresholds = None
    if args.threshold:
        try:
            with open(args.threshold) as f:
                thresholds = json.load(f)
        except Exception as e:
            print(f"{Colors.RED}Error loading thresholds: {e}{Colors.RESET}")
            sys.exit(1)

    # Create monitor
    monitor = ContainerMonitor(
        interval=args.interval,
        alerts_enabled=args.alerts,
        thresholds=thresholds
    )

    if args.once:
        # Run once
        metrics = monitor.monitor_once()
        monitor.display_dashboard(metrics)

        if args.export:
            monitor.export_metrics(args.export, metrics)
    else:
        # Continuous monitoring
        monitor.monitor_continuous()


if __name__ == '__main__':
    main()
