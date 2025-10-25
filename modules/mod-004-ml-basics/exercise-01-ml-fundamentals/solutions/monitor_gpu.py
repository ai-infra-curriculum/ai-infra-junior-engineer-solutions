#!/usr/bin/env python3
"""
monitor_gpu.py - Real-time GPU monitoring for ML workloads

Description:
    Monitor GPU utilization, memory usage, temperature, and processes
    during ML training with real-time dashboard and logging.

Usage:
    python monitor_gpu.py [OPTIONS]

Options:
    --interval SECONDS     Update interval (default: 1)
    --duration SECONDS     Monitoring duration (0 = infinite)
    --gpu ID              Monitor specific GPU (default: all)
    --log FILE            Log metrics to file
    --plot                Generate plots after monitoring
    --quiet               Minimal output
    --help                Display this help message
"""

import subprocess
import time
import sys
import argparse
import json
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class GPUMonitor:
    """GPU monitoring class"""

    def __init__(self, interval: float = 1.0, log_file: Optional[str] = None,
                 gpu_id: Optional[int] = None, quiet: bool = False):
        self.interval = interval
        self.log_file = log_file
        self.gpu_id = gpu_id
        self.quiet = quiet

        # Metrics storage
        self.metrics_history = defaultdict(list)
        self.process_history = []

        # Check if nvidia-smi is available
        if not self._check_nvidia_smi():
            print(f"{Colors.RED}Error: nvidia-smi not found{Colors.RESET}")
            print("NVIDIA drivers may not be installed or no GPU detected")
            sys.exit(1)

    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available"""
        try:
            subprocess.run(['nvidia-smi'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def get_gpu_count(self) -> int:
        """Get number of GPUs"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=count', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                check=True
            )
            return int(result.stdout.strip())
        except Exception:
            return 0

    def get_gpu_metrics(self) -> List[Dict]:
        """Get current GPU metrics"""
        query = [
            'index',
            'name',
            'temperature.gpu',
            'utilization.gpu',
            'utilization.memory',
            'memory.used',
            'memory.total',
            'power.draw',
            'power.limit',
            'clocks.current.graphics',
            'clocks.current.memory'
        ]

        cmd = [
            'nvidia-smi',
            f'--query-gpu={",".join(query)}',
            '--format=csv,noheader,nounits'
        ]

        if self.gpu_id is not None:
            cmd.insert(1, f'--id={self.gpu_id}')

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')

            metrics = []
            for line in lines:
                values = [v.strip() for v in line.split(',')]
                metrics.append({
                    'index': int(values[0]),
                    'name': values[1],
                    'temperature': int(values[2]),
                    'gpu_util': int(values[3]),
                    'mem_util': int(values[4]),
                    'mem_used': int(values[5]),
                    'mem_total': int(values[6]),
                    'power_draw': float(values[7]),
                    'power_limit': float(values[8]),
                    'clock_gpu': int(values[9]),
                    'clock_mem': int(values[10])
                })

            return metrics

        except Exception as e:
            print(f"{Colors.RED}Error getting GPU metrics: {e}{Colors.RESET}")
            return []

    def get_gpu_processes(self) -> List[Dict]:
        """Get processes using GPU"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,name,used_memory',
                 '--format=csv,noheader'],
                capture_output=True,
                text=True,
                check=True
            )

            processes = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        processes.append({
                            'pid': int(parts[0].strip()),
                            'name': parts[1].strip(),
                            'memory': int(parts[2].strip().split()[0])
                        })

            return processes

        except Exception:
            return []

    def display_metrics(self, metrics: List[Dict], processes: List[Dict]):
        """Display current metrics"""
        if self.quiet:
            return

        # Clear screen
        print('\033[2J\033[H', end='')

        # Header
        print(f"{Colors.BOLD}{Colors.CYAN}GPU Monitoring Dashboard{Colors.RESET}")
        print("=" * 80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # GPU Metrics
        for gpu in metrics:
            idx = gpu['index']
            name = gpu['name']

            # Color code based on utilization
            gpu_util = gpu['gpu_util']
            if gpu_util >= 90:
                util_color = Colors.GREEN
            elif gpu_util >= 70:
                util_color = Colors.YELLOW
            else:
                util_color = Colors.RED

            # Color code based on temperature
            temp = gpu['temperature']
            if temp >= 85:
                temp_color = Colors.RED
            elif temp >= 75:
                temp_color = Colors.YELLOW
            else:
                temp_color = Colors.GREEN

            # Memory percentage
            mem_pct = (gpu['mem_used'] / gpu['mem_total']) * 100

            print(f"{Colors.BOLD}GPU {idx}:{Colors.RESET} {name}")
            print(f"  {util_color}GPU Utilization: {gpu_util:3d}%{Colors.RESET}")
            print(f"  Memory: {gpu['mem_used']:5d} MB / {gpu['mem_total']:5d} MB "
                  f"({mem_pct:.1f}%)")
            print(f"  {temp_color}Temperature: {temp:3d}°C{Colors.RESET}")
            print(f"  Power: {gpu['power_draw']:6.1f} W / {gpu['power_limit']:6.1f} W")
            print(f"  Clocks: GPU {gpu['clock_gpu']:4d} MHz | Memory {gpu['clock_mem']:4d} MHz")
            print()

        # Processes
        if processes:
            print(f"{Colors.BOLD}GPU Processes:{Colors.RESET}")
            print("-" * 80)
            print(f"{'PID':<10} {'NAME':<40} {'MEMORY':>10}")
            print("-" * 80)
            for proc in processes:
                print(f"{proc['pid']:<10} {proc['name']:<40} {proc['memory']:>8} MB")
            print()
        else:
            print(f"{Colors.YELLOW}No GPU processes running{Colors.RESET}")
            print()

        # Statistics
        if self.metrics_history:
            print(f"{Colors.BOLD}Statistics:{Colors.RESET}")
            gpu0_utils = self.metrics_history['gpu_0_util']
            if gpu0_utils:
                avg_util = sum(gpu0_utils) / len(gpu0_utils)
                max_util = max(gpu0_utils)
                print(f"  Average GPU Utilization: {avg_util:.1f}%")
                print(f"  Peak GPU Utilization: {max_util}%")
            print()

    def log_metrics(self, metrics: List[Dict], timestamp: float):
        """Log metrics to file"""
        if not self.log_file:
            return

        try:
            with open(self.log_file, 'a') as f:
                for gpu in metrics:
                    log_entry = {
                        'timestamp': timestamp,
                        'gpu_index': gpu['index'],
                        'gpu_util': gpu['gpu_util'],
                        'mem_util': gpu['mem_util'],
                        'mem_used': gpu['mem_used'],
                        'temperature': gpu['temperature'],
                        'power_draw': gpu['power_draw']
                    }
                    f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"{Colors.RED}Error logging metrics: {e}{Colors.RESET}")

    def store_metrics(self, metrics: List[Dict]):
        """Store metrics in history"""
        for gpu in metrics:
            idx = gpu['index']
            self.metrics_history[f'gpu_{idx}_util'].append(gpu['gpu_util'])
            self.metrics_history[f'gpu_{idx}_mem'].append(gpu['mem_used'])
            self.metrics_history[f'gpu_{idx}_temp'].append(gpu['temperature'])
            self.metrics_history[f'gpu_{idx}_power'].append(gpu['power_draw'])

    def generate_plots(self):
        """Generate plots from collected metrics"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print(f"{Colors.YELLOW}matplotlib not installed, skipping plots{Colors.RESET}")
            return

        if not self.metrics_history:
            print(f"{Colors.YELLOW}No metrics to plot{Colors.RESET}")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GPU Monitoring Report', fontsize=16, fontweight='bold')

        # GPU Utilization
        ax = axes[0, 0]
        for key in self.metrics_history:
            if 'util' in key:
                ax.plot(self.metrics_history[key], label=key)
        ax.set_title('GPU Utilization (%)')
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('Utilization (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Memory Usage
        ax = axes[0, 1]
        for key in self.metrics_history:
            if 'mem' in key:
                ax.plot(self.metrics_history[key], label=key)
        ax.set_title('Memory Usage (MB)')
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('Memory (MB)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Temperature
        ax = axes[1, 0]
        for key in self.metrics_history:
            if 'temp' in key:
                ax.plot(self.metrics_history[key], label=key)
        ax.set_title('Temperature (°C)')
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('Temperature (°C)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Power Draw
        ax = axes[1, 1]
        for key in self.metrics_history:
            if 'power' in key:
                ax.plot(self.metrics_history[key], label=key)
        ax.set_title('Power Draw (W)')
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('Power (W)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_file = f'gpu_monitoring_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_file, dpi=150)
        print(f"\n{Colors.GREEN}Plot saved to: {plot_file}{Colors.RESET}")

    def monitor(self, duration: float = 0):
        """Main monitoring loop"""
        start_time = time.time()
        iteration = 0

        print(f"{Colors.CYAN}Starting GPU monitoring...{Colors.RESET}")
        if duration > 0:
            print(f"Duration: {duration} seconds")
        else:
            print("Press Ctrl+C to stop")
        print()

        try:
            while True:
                # Get metrics
                metrics = self.get_gpu_metrics()
                processes = self.get_gpu_processes()

                # Store metrics
                self.store_metrics(metrics)

                # Display
                self.display_metrics(metrics, processes)

                # Log
                if self.log_file:
                    self.log_metrics(metrics, time.time())

                iteration += 1

                # Check duration
                if duration > 0 and (time.time() - start_time) >= duration:
                    break

                # Sleep
                time.sleep(self.interval)

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Monitoring stopped by user{Colors.RESET}")

        # Print summary
        elapsed = time.time() - start_time
        print(f"\n{Colors.BOLD}Monitoring Summary:{Colors.RESET}")
        print(f"Duration: {elapsed:.1f} seconds")
        print(f"Samples: {iteration}")
        if self.log_file:
            print(f"Log file: {self.log_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Real-time GPU monitoring for ML workloads',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--interval', type=float, default=1.0,
                        help='Update interval in seconds (default: 1.0)')
    parser.add_argument('--duration', type=float, default=0,
                        help='Monitoring duration in seconds (0 = infinite)')
    parser.add_argument('--gpu', type=int, default=None,
                        help='Monitor specific GPU ID')
    parser.add_argument('--log', type=str, default=None,
                        help='Log metrics to file')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots after monitoring')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')

    args = parser.parse_args()

    # Create monitor
    monitor = GPUMonitor(
        interval=args.interval,
        log_file=args.log,
        gpu_id=args.gpu,
        quiet=args.quiet
    )

    # Start monitoring
    monitor.monitor(duration=args.duration)

    # Generate plots
    if args.plot:
        monitor.generate_plots()


if __name__ == '__main__':
    main()
