#!/usr/bin/env python3
"""
monitor_deployment.py - Deployment health monitoring

Description:
    Monitor deployed ML models health, performance, and availability
    with real-time dashboard and alerting.

Usage:
    python monitor_deployment.py --url http://localhost:8000 --interval 5

Options:
    --url URL             API base URL
    --interval SECONDS    Check interval (default: 5)
    --namespace NS        Kubernetes namespace
    --alert-email EMAIL   Email for alerts
    --thresholds FILE     Thresholds configuration file
    --verbose             Verbose output
    --help                Display this help message
"""

import argparse
import requests
import time
import sys
from typing import Dict, Optional
from datetime import datetime
from collections import deque
import json

# Colors
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class DeploymentMonitor:
    """Monitor deployment health and performance"""

    def __init__(self, url: str, interval: float = 5.0,
                 namespace: str = 'default',
                 thresholds: Optional[Dict] = None):
        """
        Initialize deployment monitor

        Args:
            url: API base URL
            interval: Check interval in seconds
            namespace: Kubernetes namespace
            thresholds: Alert thresholds
        """
        self.url = url.rstrip('/')
        self.interval = interval
        self.namespace = namespace
        self.thresholds = thresholds or {
            'response_time_ms': 1000,
            'error_rate': 0.05,
            'cpu_usage': 80,
            'memory_usage': 80
        }

        # Metrics history
        self.latency_history = deque(maxlen=100)
        self.error_count = 0
        self.total_requests = 0
        self.alerts = []
        self.start_time = time.time()

    def check_health(self) -> Dict:
        """Check API health endpoint"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.url}/health", timeout=10)
            latency = (time.time() - start_time) * 1000  # ms

            self.latency_history.append(latency)
            self.total_requests += 1

            if response.status_code == 200:
                return {
                    'healthy': True,
                    'status_code': response.status_code,
                    'latency_ms': latency,
                    'data': response.json()
                }
            else:
                self.error_count += 1
                return {
                    'healthy': False,
                    'status_code': response.status_code,
                    'latency_ms': latency,
                    'error': f"Unexpected status code: {response.status_code}"
                }

        except Exception as e:
            self.error_count += 1
            self.total_requests += 1
            return {
                'healthy': False,
                'status_code': 0,
                'latency_ms': 0,
                'error': str(e)
            }

    def check_alerts(self, health_data: Dict):
        """Check for alert conditions"""
        # Check latency
        if health_data.get('latency_ms', 0) > self.thresholds['response_time_ms']:
            alert = f"High latency: {health_data['latency_ms']:.2f}ms"
            if alert not in self.alerts:
                self.alerts.append(alert)

        # Check error rate
        error_rate = self.error_count / self.total_requests if self.total_requests > 0 else 0
        if error_rate > self.thresholds['error_rate']:
            alert = f"High error rate: {error_rate * 100:.2f}%"
            if alert not in self.alerts:
                self.alerts.append(alert)

    def get_statistics(self) -> Dict:
        """Calculate statistics"""
        stats = {}

        if self.latency_history:
            stats['avg_latency_ms'] = sum(self.latency_history) / len(self.latency_history)
            stats['min_latency_ms'] = min(self.latency_history)
            stats['max_latency_ms'] = max(self.latency_history)

        stats['total_requests'] = self.total_requests
        stats['error_count'] = self.error_count
        stats['error_rate'] = self.error_count / self.total_requests if self.total_requests > 0 else 0
        stats['uptime_seconds'] = time.time() - self.start_time

        return stats

    def display_dashboard(self, health_data: Dict):
        """Display monitoring dashboard"""
        # Clear screen
        print('\033[2J\033[H', end='')

        # Header
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}Deployment Monitoring Dashboard{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.RESET}\n")

        # Status
        print(f"{Colors.BOLD}Status:{Colors.RESET}")
        print(f"  URL: {self.url}")
        print(f"  Last Check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if health_data.get('healthy'):
            print(f"  Health: {Colors.GREEN}✓ Healthy{Colors.RESET}")
        else:
            print(f"  Health: {Colors.RED}✗ Unhealthy{Colors.RESET}")
            if 'error' in health_data:
                print(f"  Error: {health_data['error']}")

        print()

        # API Info
        if health_data.get('healthy') and 'data' in health_data:
            data = health_data['data']
            print(f"{Colors.BOLD}API Information:{Colors.RESET}")
            print(f"  Models Loaded: {data.get('models_loaded', 0)}")
            print(f"  GPU Available: {data.get('gpu_available', False)}")
            if data.get('gpu_available'):
                print(f"  GPU Count: {data.get('gpu_count', 0)}")
            print(f"  Uptime: {data.get('uptime_seconds', 0) / 3600:.2f} hours")
            print()

        # Statistics
        stats = self.get_statistics()
        print(f"{Colors.BOLD}Statistics:{Colors.RESET}")

        if stats.get('avg_latency_ms'):
            latency_color = Colors.GREEN if stats['avg_latency_ms'] < 100 else Colors.YELLOW if stats['avg_latency_ms'] < 500 else Colors.RED
            print(f"  Avg Latency: {latency_color}{stats['avg_latency_ms']:.2f}ms{Colors.RESET}")
            print(f"  Min/Max Latency: {stats['min_latency_ms']:.2f}ms / {stats['max_latency_ms']:.2f}ms")

        print(f"  Total Requests: {stats['total_requests']}")
        print(f"  Errors: {Colors.RED if stats['error_count'] > 0 else Colors.GREEN}{stats['error_count']}{Colors.RESET}")

        error_rate_pct = stats['error_rate'] * 100
        error_color = Colors.GREEN if error_rate_pct < 1 else Colors.YELLOW if error_rate_pct < 5 else Colors.RED
        print(f"  Error Rate: {error_color}{error_rate_pct:.2f}%{Colors.RESET}")

        print(f"  Monitor Uptime: {stats['uptime_seconds'] / 60:.2f} minutes")
        print()

        # Alerts
        if self.alerts:
            print(f"{Colors.BOLD}{Colors.RED}Active Alerts:{Colors.RESET}")
            for alert in self.alerts[-5:]:  # Show last 5 alerts
                print(f"  ⚠️  {alert}")
            print()

        # Latency trend (simple text visualization)
        if len(self.latency_history) > 10:
            print(f"{Colors.BOLD}Latency Trend (last 10 checks):{Colors.RESET}")
            recent_latencies = list(self.latency_history)[-10:]
            max_latency = max(recent_latencies)
            for i, latency in enumerate(recent_latencies):
                # Normalize to 0-50 range for visualization
                normalized = int((latency / max_latency) * 50) if max_latency > 0 else 0
                bar = '█' * normalized
                print(f"  {i + 1:2d}: {bar} {latency:.2f}ms")
            print()

        # Footer
        print(f"{Colors.CYAN}{'=' * 80}{Colors.RESET}")
        print(f"Refresh interval: {self.interval}s | Press Ctrl+C to stop")

    def monitor(self):
        """Main monitoring loop"""
        print(f"{Colors.CYAN}Starting deployment monitoring...{Colors.RESET}")
        print(f"URL: {self.url}")
        print(f"Interval: {self.interval}s\n")

        try:
            while True:
                # Check health
                health_data = self.check_health()

                # Check for alerts
                self.check_alerts(health_data)

                # Display dashboard
                self.display_dashboard(health_data)

                # Wait for next check
                time.sleep(self.interval)

        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Monitoring stopped by user{Colors.RESET}")

            # Print final summary
            stats = self.get_statistics()
            print(f"\n{Colors.BOLD}Final Summary:{Colors.RESET}")
            print(f"  Total Checks: {stats['total_requests']}")
            print(f"  Errors: {stats['error_count']}")
            print(f"  Error Rate: {stats['error_rate'] * 100:.2f}%")
            if stats.get('avg_latency_ms'):
                print(f"  Average Latency: {stats['avg_latency_ms']:.2f}ms")
            print(f"  Monitoring Duration: {stats['uptime_seconds'] / 60:.2f} minutes")

            if self.alerts:
                print(f"\n  Total Alerts: {len(self.alerts)}")


def main():
    parser = argparse.ArgumentParser(
        description='Deployment health monitoring',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--url', type=str, required=True,
                       help='API base URL')
    parser.add_argument('--interval', type=float, default=5.0,
                       help='Check interval in seconds (default: 5.0)')
    parser.add_argument('--namespace', type=str, default='default',
                       help='Kubernetes namespace (default: default)')
    parser.add_argument('--alert-email', type=str, default=None,
                       help='Email for alerts')
    parser.add_argument('--thresholds', type=str, default=None,
                       help='Thresholds configuration file')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Load thresholds if provided
    thresholds = None
    if args.thresholds:
        with open(args.thresholds) as f:
            thresholds = json.load(f)

    # Create monitor
    monitor = DeploymentMonitor(
        url=args.url,
        interval=args.interval,
        namespace=args.namespace,
        thresholds=thresholds
    )

    # Start monitoring
    monitor.monitor()


if __name__ == '__main__':
    main()
