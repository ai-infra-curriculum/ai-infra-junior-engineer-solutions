#!/usr/bin/env python3
"""
Production Health Monitor

Real-time monitoring dashboard for production deployments.

Features:
- Service health monitoring
- Response time tracking
- Error rate monitoring
- Resource usage tracking
- Alert system
- Grafana-style terminal dashboard
"""

import argparse
import docker
import requests
import sys
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Deque
import statistics


class Colors:
    """ANSI color codes"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ServiceHealthTracker:
    """Track health metrics for a service"""

    def __init__(self, name: str, health_url: str, max_history: int = 100):
        self.name = name
        self.health_url = health_url
        self.max_history = max_history

        # Metrics history
        self.response_times: Deque[float] = deque(maxlen=max_history)
        self.status_codes: Deque[int] = deque(maxlen=max_history)
        self.errors: Deque[str] = deque(maxlen=max_history)

        # Current state
        self.is_healthy = False
        self.last_check_time: Optional[datetime] = None
        self.consecutive_failures = 0

    def check_health(self, timeout: int = 5) -> Dict:
        """Perform health check"""
        self.last_check_time = datetime.now()

        try:
            start_time = time.time()
            response = requests.get(self.health_url, timeout=timeout)
            response_time = time.time() - start_time

            self.response_times.append(response_time)
            self.status_codes.append(response.status_code)

            if response.status_code == 200:
                self.is_healthy = True
                self.consecutive_failures = 0
                return {
                    'healthy': True,
                    'status_code': response.status_code,
                    'response_time': response_time,
                    'body': response.json() if 'application/json' in response.headers.get('content-type', '') else None
                }
            else:
                self.is_healthy = False
                self.consecutive_failures += 1
                error_msg = f"HTTP {response.status_code}"
                self.errors.append(error_msg)
                return {
                    'healthy': False,
                    'status_code': response.status_code,
                    'response_time': response_time,
                    'error': error_msg
                }

        except requests.exceptions.RequestException as e:
            self.is_healthy = False
            self.consecutive_failures += 1
            error_msg = str(e)
            self.errors.append(error_msg)
            return {
                'healthy': False,
                'error': error_msg
            }

    def get_stats(self) -> Dict:
        """Get statistics"""
        if not self.response_times:
            return {
                'avg_response_time': 0,
                'min_response_time': 0,
                'max_response_time': 0,
                'p95_response_time': 0,
                'success_rate': 0,
                'total_requests': 0
            }

        response_times_list = list(self.response_times)
        status_codes_list = list(self.status_codes)

        successful = sum(1 for code in status_codes_list if code == 200)
        total = len(status_codes_list)

        sorted_times = sorted(response_times_list)
        p95_index = int(len(sorted_times) * 0.95)

        return {
            'avg_response_time': statistics.mean(response_times_list),
            'min_response_time': min(response_times_list),
            'max_response_time': max(response_times_list),
            'p95_response_time': sorted_times[p95_index] if sorted_times else 0,
            'success_rate': (successful / total * 100) if total > 0 else 0,
            'total_requests': total,
            'error_count': len(self.errors)
        }


class ResourceMonitor:
    """Monitor container resource usage"""

    def __init__(self, client: docker.DockerClient):
        self.client = client

    def get_container_stats(self, container_id: str) -> Dict:
        """Get container resource stats"""
        try:
            container = self.client.containers.get(container_id)
            stats = container.stats(stream=False)

            # Calculate CPU percentage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            cpu_count = stats['cpu_stats']['online_cpus']

            cpu_percent = 0.0
            if system_delta > 0 and cpu_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * cpu_count * 100.0

            # Calculate memory
            mem_usage = stats['memory_stats'].get('usage', 0)
            mem_limit = stats['memory_stats'].get('limit', 1)
            mem_percent = (mem_usage / mem_limit) * 100.0 if mem_limit > 0 else 0

            # Network
            networks = stats.get('networks', {})
            rx_bytes = sum(net['rx_bytes'] for net in networks.values())
            tx_bytes = sum(net['tx_bytes'] for net in networks.values())

            return {
                'cpu_percent': cpu_percent,
                'memory_usage': mem_usage,
                'memory_limit': mem_limit,
                'memory_percent': mem_percent,
                'network_rx_bytes': rx_bytes,
                'network_tx_bytes': tx_bytes,
                'status': container.status
            }

        except docker.errors.NotFound:
            return None
        except Exception as e:
            return {'error': str(e)}


class AlertManager:
    """Manage alerts based on thresholds"""

    def __init__(self):
        self.alerts: List[Dict] = []
        self.thresholds = {
            'response_time': 1.0,  # seconds
            'error_rate': 5.0,  # percentage
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'consecutive_failures': 3
        }

    def check_alerts(self, service_name: str, health_stats: Dict, resource_stats: Optional[Dict] = None) -> List[str]:
        """Check for alert conditions"""
        alerts = []

        # Response time alert
        if health_stats.get('avg_response_time', 0) > self.thresholds['response_time']:
            alerts.append(f"High response time: {health_stats['avg_response_time']:.3f}s")

        # Error rate alert
        success_rate = health_stats.get('success_rate', 100)
        if success_rate < (100 - self.thresholds['error_rate']):
            alerts.append(f"High error rate: {100 - success_rate:.1f}%")

        # Resource alerts
        if resource_stats:
            if resource_stats.get('cpu_percent', 0) > self.thresholds['cpu_percent']:
                alerts.append(f"High CPU usage: {resource_stats['cpu_percent']:.1f}%")

            if resource_stats.get('memory_percent', 0) > self.thresholds['memory_percent']:
                alerts.append(f"High memory usage: {resource_stats['memory_percent']:.1f}%")

        if alerts:
            for alert in alerts:
                self.alerts.append({
                    'service': service_name,
                    'time': datetime.now(),
                    'message': alert,
                    'severity': 'warning'
                })

        return alerts


class HealthMonitorDashboard:
    """Real-time health monitoring dashboard"""

    def __init__(self, interval: int = 5):
        self.interval = interval

        try:
            self.client = docker.from_env()
        except docker.errors.DockerException as e:
            print(f"{Colors.RED}Error: Unable to connect to Docker daemon: {e}{Colors.END}")
            sys.exit(1)

        self.trackers: Dict[str, ServiceHealthTracker] = {}
        self.resource_monitor = ResourceMonitor(self.client)
        self.alert_manager = AlertManager()

    def add_service(self, name: str, health_url: str):
        """Add service to monitor"""
        self.trackers[name] = ServiceHealthTracker(name, health_url)

    def discover_services(self, label: str = "monitoring=enabled"):
        """Auto-discover services with monitoring label"""
        containers = self.client.containers.list(filters={'label': label})

        for container in containers:
            name = container.name
            labels = container.labels

            # Try to find health endpoint
            port = labels.get('health.port', '8000')
            path = labels.get('health.path', '/health')

            container.reload()
            networks = container.attrs['NetworkSettings']['Networks']

            # Get IP from first network
            if networks:
                ip = list(networks.values())[0]['IPAddress']
                health_url = f"http://{ip}:{port}{path}"
                self.add_service(name, health_url)
                print(f"{Colors.GREEN}✓ Discovered service: {name} at {health_url}{Colors.END}")

    def render_dashboard(self):
        """Render dashboard"""
        # Clear screen
        print('\033[2J\033[H', end='')

        # Header
        print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
        print(f"{Colors.BOLD}Production Health Monitor{Colors.END}".center(80))
        print(f"{Colors.BOLD}{'='*80}{Colors.END}\n")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Monitoring {len(self.trackers)} services\n")

        # Service health
        for name, tracker in self.trackers.items():
            stats = tracker.get_stats()

            # Status indicator
            if tracker.is_healthy:
                status_icon = f"{Colors.GREEN}●{Colors.END}"
                status_text = f"{Colors.GREEN}HEALTHY{Colors.END}"
            else:
                status_icon = f"{Colors.RED}●{Colors.END}"
                status_text = f"{Colors.RED}DOWN{Colors.END}"

            print(f"{status_icon} {Colors.BOLD}{name}{Colors.END} - {status_text}")
            print(f"  URL: {tracker.health_url}")

            if tracker.last_check_time:
                print(f"  Last Check: {tracker.last_check_time.strftime('%H:%M:%S')}")

            if stats['total_requests'] > 0:
                print(f"  Response Time: avg={stats['avg_response_time']*1000:.0f}ms "
                      f"p95={stats['p95_response_time']*1000:.0f}ms "
                      f"min={stats['min_response_time']*1000:.0f}ms "
                      f"max={stats['max_response_time']*1000:.0f}ms")
                print(f"  Success Rate: {stats['success_rate']:.1f}% ({stats['total_requests']} requests)")

                if stats['error_count'] > 0:
                    print(f"  {Colors.RED}Errors: {stats['error_count']}{Colors.END}")

            # Check for container resources
            containers = self.client.containers.list(filters={'name': name})
            if containers:
                container = containers[0]
                resource_stats = self.resource_monitor.get_container_stats(container.id)

                if resource_stats and 'error' not in resource_stats:
                    print(f"  CPU: {resource_stats['cpu_percent']:.1f}% | "
                          f"Memory: {resource_stats['memory_percent']:.1f}% "
                          f"({resource_stats['memory_usage']/(1024**2):.0f}MB / "
                          f"{resource_stats['memory_limit']/(1024**2):.0f}MB)")

                    # Check alerts
                    alerts = self.alert_manager.check_alerts(name, stats, resource_stats)
                    if alerts:
                        for alert in alerts:
                            print(f"  {Colors.YELLOW}⚠ ALERT: {alert}{Colors.END}")

            print()

        # Recent alerts
        recent_alerts = [a for a in self.alert_manager.alerts if (datetime.now() - a['time']).seconds < 300]
        if recent_alerts:
            print(f"\n{Colors.BOLD}Recent Alerts (last 5 minutes):{Colors.END}")
            for alert in recent_alerts[-5:]:
                time_str = alert['time'].strftime('%H:%M:%S')
                print(f"  {Colors.YELLOW}[{time_str}] {alert['service']}: {alert['message']}{Colors.END}")

        print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
        print(f"Refresh interval: {self.interval}s | Press Ctrl+C to exit")

    def monitor(self):
        """Start monitoring loop"""
        print(f"{Colors.CYAN}Starting health monitor...{Colors.END}\n")

        if not self.trackers:
            print(f"{Colors.YELLOW}No services to monitor. Use --discover or --service{Colors.END}")
            return

        try:
            while True:
                # Check health for all services
                for tracker in self.trackers.values():
                    tracker.check_health()

                # Render dashboard
                self.render_dashboard()

                # Wait
                time.sleep(self.interval)

        except KeyboardInterrupt:
            print(f"\n\n{Colors.CYAN}Monitoring stopped{Colors.END}")

    def generate_report(self):
        """Generate health report"""
        print(f"\n{Colors.BOLD}Health Monitor Report{Colors.END}")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        for name, tracker in self.trackers.items():
            stats = tracker.get_stats()

            print(f"{Colors.BOLD}{name}{Colors.END}")
            print(f"  Status: {'HEALTHY' if tracker.is_healthy else 'DOWN'}")
            print(f"  Total Requests: {stats['total_requests']}")
            print(f"  Success Rate: {stats['success_rate']:.2f}%")
            print(f"  Avg Response Time: {stats['avg_response_time']*1000:.1f}ms")
            print(f"  P95 Response Time: {stats['p95_response_time']*1000:.1f}ms")
            print(f"  Error Count: {stats['error_count']}")
            print()

        # Alerts summary
        if self.alert_manager.alerts:
            print(f"{Colors.BOLD}Alerts Summary:{Colors.END}")
            print(f"  Total Alerts: {len(self.alert_manager.alerts)}")

            # Group by service
            alerts_by_service = {}
            for alert in self.alert_manager.alerts:
                service = alert['service']
                if service not in alerts_by_service:
                    alerts_by_service[service] = 0
                alerts_by_service[service] += 1

            for service, count in alerts_by_service.items():
                print(f"  {service}: {count} alerts")


def main():
    parser = argparse.ArgumentParser(
        description='Production health monitor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor specific services
  health_monitor.py --service ml-api http://localhost:8000/health \\
                    --service ml-api-2 http://localhost:8001/health

  # Auto-discover services with label
  health_monitor.py --discover monitoring=enabled

  # Custom refresh interval
  health_monitor.py --discover --interval 10

  # Generate one-time report
  health_monitor.py --service ml-api http://localhost:8000/health --report
        """
    )

    parser.add_argument(
        '--service',
        nargs=2,
        action='append',
        metavar=('NAME', 'URL'),
        help='Add service to monitor (name and health URL)'
    )
    parser.add_argument(
        '--discover',
        nargs='?',
        const='monitoring=enabled',
        metavar='LABEL',
        help='Auto-discover services with label (default: monitoring=enabled)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Refresh interval in seconds (default: 5)'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate one-time report instead of continuous monitoring'
    )

    args = parser.parse_args()

    monitor = HealthMonitorDashboard(interval=args.interval)

    # Add services
    if args.service:
        for name, url in args.service:
            monitor.add_service(name, url)

    # Discover services
    if args.discover:
        monitor.discover_services(label=args.discover)

    # Run monitor or generate report
    if args.report:
        # Check health once for each service
        for tracker in monitor.trackers.values():
            tracker.check_health()
        monitor.generate_report()
    else:
        monitor.monitor()


if __name__ == '__main__':
    main()
