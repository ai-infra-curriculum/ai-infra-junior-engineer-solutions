#!/usr/bin/env python3
"""
monitor_training.py - Real-time training monitor with dashboard

Description:
    Monitor training progress in real-time with metrics visualization,
    GPU utilization tracking, and alert system for training issues.

Usage:
    python monitor_training.py [OPTIONS]

Options:
    --log-dir DIR          Training log directory to monitor
    --metrics-file FILE    Metrics JSON file to monitor
    --refresh SECONDS      Refresh interval (default: 5)
    --alert-email EMAIL    Email for alerts
    --gpu-threshold PCT    GPU utilization alert threshold (default: 20)
    --loss-stall N         Loss stall alert after N epochs (default: 10)
    --verbose              Verbose output
    --help                 Display this help message
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import deque
import subprocess

import torch

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class TrainingMonitor:
    """Monitor training progress"""

    def __init__(self, log_dir: str, metrics_file: str,
                 refresh_interval: float = 5.0,
                 gpu_threshold: int = 20,
                 loss_stall_threshold: int = 10):
        """
        Initialize training monitor

        Args:
            log_dir: Directory containing training logs
            metrics_file: Path to metrics JSON file
            refresh_interval: Refresh interval in seconds
            gpu_threshold: Minimum GPU utilization threshold
            loss_stall_threshold: Number of epochs before loss stall alert
        """
        self.log_dir = Path(log_dir)
        self.metrics_file = Path(metrics_file)
        self.refresh_interval = refresh_interval
        self.gpu_threshold = gpu_threshold
        self.loss_stall_threshold = loss_stall_threshold

        # State
        self.metrics_history = {}
        self.last_epoch = -1
        self.start_time = time.time()
        self.alerts = []

        # GPU tracking
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0

    def load_metrics(self) -> bool:
        """Load metrics from file"""
        if not self.metrics_file.exists():
            return False

        try:
            with open(self.metrics_file) as f:
                self.metrics_history = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading metrics: {e}")
            return False

    def get_gpu_info(self) -> Dict:
        """Get current GPU information"""
        if not self.gpu_available:
            return {}

        try:
            result = subprocess.run(
                ['nvidia-smi',
                 '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )

            gpu_info = []
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 8:
                    gpu_info.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'utilization': int(parts[2]),
                        'memory_utilization': int(parts[3]),
                        'memory_used': int(parts[4]),
                        'memory_total': int(parts[5]),
                        'temperature': int(parts[6]),
                        'power_draw': float(parts[7])
                    })

            return {'gpus': gpu_info}

        except Exception as e:
            return {}

    def check_for_issues(self):
        """Check for training issues"""
        if not self.metrics_history:
            return

        # Check for loss stall
        if 'train_loss' in self.metrics_history:
            losses = self.metrics_history['train_loss']
            if len(losses) >= self.loss_stall_threshold:
                recent_losses = losses[-self.loss_stall_threshold:]
                loss_change = abs(recent_losses[-1] - recent_losses[0])

                if loss_change < 0.001:
                    alert = f"⚠️  Loss stalled for {self.loss_stall_threshold} epochs"
                    if alert not in self.alerts:
                        self.alerts.append(alert)

        # Check for NaN/Inf losses
        if 'train_loss' in self.metrics_history:
            latest_loss = self.metrics_history['train_loss'][-1]
            if not (0 <= latest_loss < float('inf')):
                alert = f"🔥 Invalid loss detected: {latest_loss}"
                if alert not in self.alerts:
                    self.alerts.append(alert)

        # Check GPU utilization
        gpu_info = self.get_gpu_info()
        if gpu_info and 'gpus' in gpu_info:
            for gpu in gpu_info['gpus']:
                if gpu['utilization'] < self.gpu_threshold:
                    alert = f"⚠️  Low GPU {gpu['index']} utilization: {gpu['utilization']}%"
                    if alert not in self.alerts:
                        self.alerts.append(alert)

    def get_training_statistics(self) -> Dict:
        """Calculate training statistics"""
        if not self.metrics_history:
            return {}

        stats = {}

        # Loss statistics
        if 'train_loss' in self.metrics_history:
            train_losses = self.metrics_history['train_loss']
            stats['train_loss'] = {
                'current': train_losses[-1] if train_losses else 0,
                'min': min(train_losses) if train_losses else 0,
                'max': max(train_losses) if train_losses else 0,
                'mean': sum(train_losses) / len(train_losses) if train_losses else 0
            }

        if 'val_loss' in self.metrics_history:
            val_losses = self.metrics_history['val_loss']
            stats['val_loss'] = {
                'current': val_losses[-1] if val_losses else 0,
                'min': min(val_losses) if val_losses else 0,
                'best_epoch': val_losses.index(min(val_losses)) if val_losses else 0
            }

        # Accuracy statistics
        if 'train_acc' in self.metrics_history:
            train_accs = self.metrics_history['train_acc']
            stats['train_acc'] = {
                'current': train_accs[-1] if train_accs else 0,
                'max': max(train_accs) if train_accs else 0
            }

        if 'val_acc' in self.metrics_history:
            val_accs = self.metrics_history['val_acc']
            stats['val_acc'] = {
                'current': val_accs[-1] if val_accs else 0,
                'max': max(val_accs) if val_accs else 0,
                'best_epoch': val_accs.index(max(val_accs)) if val_accs else 0
            }

        return stats

    def display_dashboard(self):
        """Display monitoring dashboard"""
        # Clear screen
        print('\033[2J\033[H', end='')

        # Header
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}Training Monitor Dashboard{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.RESET}")
        print()

        # Training info
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        print(f"{Colors.BOLD}Training Status:{Colors.RESET}")
        print(f"  Log Directory: {self.log_dir}")
        print(f"  Metrics File: {self.metrics_file}")
        print(f"  Monitoring Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"  Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Metrics
        if self.metrics_history:
            current_epoch = len(self.metrics_history.get('train_loss', [])) - 1

            print(f"{Colors.BOLD}Current Epoch: {current_epoch + 1}{Colors.RESET}")
            print()

            # Training metrics
            if 'train_loss' in self.metrics_history and self.metrics_history['train_loss']:
                train_loss = self.metrics_history['train_loss'][-1]
                print(f"  {Colors.GREEN}Train Loss:{Colors.RESET} {train_loss:.4f}")

            if 'train_acc' in self.metrics_history and self.metrics_history['train_acc']:
                train_acc = self.metrics_history['train_acc'][-1]
                print(f"  {Colors.GREEN}Train Accuracy:{Colors.RESET} {train_acc:.2f}%")

            # Validation metrics
            if 'val_loss' in self.metrics_history and self.metrics_history['val_loss']:
                val_loss = self.metrics_history['val_loss'][-1]
                print(f"  {Colors.CYAN}Val Loss:{Colors.RESET} {val_loss:.4f}")

            if 'val_acc' in self.metrics_history and self.metrics_history['val_acc']:
                val_acc = self.metrics_history['val_acc'][-1]
                print(f"  {Colors.CYAN}Val Accuracy:{Colors.RESET} {val_acc:.2f}%")

            # Learning rate
            if 'learning_rate' in self.metrics_history and self.metrics_history['learning_rate']:
                lr = self.metrics_history['learning_rate'][-1]
                print(f"  {Colors.YELLOW}Learning Rate:{Colors.RESET} {lr:.6f}")

            print()

            # Statistics
            stats = self.get_training_statistics()
            if stats:
                print(f"{Colors.BOLD}Statistics:{Colors.RESET}")

                if 'val_loss' in stats:
                    print(f"  Best Val Loss: {stats['val_loss']['min']:.4f} "
                          f"(epoch {stats['val_loss']['best_epoch'] + 1})")

                if 'val_acc' in stats:
                    print(f"  Best Val Accuracy: {stats['val_acc']['max']:.2f}% "
                          f"(epoch {stats['val_acc']['best_epoch'] + 1})")

                print()

        else:
            print(f"{Colors.YELLOW}No metrics available yet...{Colors.RESET}")
            print()

        # GPU info
        if self.gpu_available:
            gpu_info = self.get_gpu_info()
            if gpu_info and 'gpus' in gpu_info:
                print(f"{Colors.BOLD}GPU Status:{Colors.RESET}")

                for gpu in gpu_info['gpus']:
                    # Color code utilization
                    util = gpu['utilization']
                    if util >= 80:
                        util_color = Colors.GREEN
                    elif util >= 50:
                        util_color = Colors.YELLOW
                    else:
                        util_color = Colors.RED

                    mem_pct = (gpu['memory_used'] / gpu['memory_total']) * 100

                    print(f"  GPU {gpu['index']}: {gpu['name']}")
                    print(f"    {util_color}Utilization: {util}%{Colors.RESET}")
                    print(f"    Memory: {gpu['memory_used']} MB / {gpu['memory_total']} MB "
                          f"({mem_pct:.1f}%)")
                    print(f"    Temperature: {gpu['temperature']}°C")
                    print(f"    Power: {gpu['power_draw']:.1f} W")
                    print()
        else:
            print(f"{Colors.YELLOW}No GPU available{Colors.RESET}")
            print()

        # Alerts
        if self.alerts:
            print(f"{Colors.BOLD}{Colors.RED}Alerts:{Colors.RESET}")
            for alert in self.alerts[-5:]:  # Show last 5 alerts
                print(f"  {alert}")
            print()

        # Progress visualization (simple text-based)
        if 'train_loss' in self.metrics_history and len(self.metrics_history['train_loss']) > 1:
            print(f"{Colors.BOLD}Loss Trend (last 10 epochs):{Colors.RESET}")

            recent_losses = self.metrics_history['train_loss'][-10:]
            max_loss = max(recent_losses)
            min_loss = min(recent_losses)

            for i, loss in enumerate(recent_losses):
                # Normalize to 0-50 range for visualization
                if max_loss != min_loss:
                    normalized = int((loss - min_loss) / (max_loss - min_loss) * 50)
                else:
                    normalized = 25

                bar = '█' * normalized
                epoch_num = len(self.metrics_history['train_loss']) - len(recent_losses) + i + 1
                print(f"  Epoch {epoch_num:3d}: {bar} {loss:.4f}")

            print()

        # Footer
        print(f"{Colors.CYAN}{'=' * 80}{Colors.RESET}")
        print(f"Refresh interval: {self.refresh_interval}s | Press Ctrl+C to stop")

    def monitor(self):
        """Main monitoring loop"""
        print(f"{Colors.CYAN}Starting training monitor...{Colors.RESET}")
        print(f"Monitoring: {self.metrics_file}")
        print(f"Refresh interval: {self.refresh_interval}s")
        print()

        try:
            while True:
                # Load latest metrics
                self.load_metrics()

                # Check for issues
                self.check_for_issues()

                # Display dashboard
                self.display_dashboard()

                # Wait for next refresh
                time.sleep(self.refresh_interval)

        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Monitoring stopped by user{Colors.RESET}")

            # Print final summary
            if self.metrics_history:
                stats = self.get_training_statistics()
                print(f"\n{Colors.BOLD}Final Summary:{Colors.RESET}")

                if 'val_loss' in stats:
                    print(f"  Best Val Loss: {stats['val_loss']['min']:.4f} "
                          f"(epoch {stats['val_loss']['best_epoch'] + 1})")

                if 'val_acc' in stats:
                    print(f"  Best Val Accuracy: {stats['val_acc']['max']:.2f}% "
                          f"(epoch {stats['val_acc']['best_epoch'] + 1})")

                if self.alerts:
                    print(f"\n  Total Alerts: {len(self.alerts)}")


def main():
    parser = argparse.ArgumentParser(
        description='Real-time training monitor',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--log-dir', type=str, required=True,
                       help='Training log directory to monitor')
    parser.add_argument('--metrics-file', type=str, default=None,
                       help='Metrics JSON file to monitor (default: log-dir/metrics.json)')
    parser.add_argument('--refresh', type=float, default=5.0,
                       help='Refresh interval in seconds (default: 5.0)')
    parser.add_argument('--alert-email', type=str, default=None,
                       help='Email for alerts')
    parser.add_argument('--gpu-threshold', type=int, default=20,
                       help='GPU utilization alert threshold (default: 20)')
    parser.add_argument('--loss-stall', type=int, default=10,
                       help='Loss stall alert after N epochs (default: 10)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Determine metrics file
    if args.metrics_file is None:
        args.metrics_file = str(Path(args.log_dir) / 'metrics.json')

    # Check if metrics file exists
    if not Path(args.metrics_file).exists():
        print(f"{Colors.YELLOW}Warning: Metrics file not found: {args.metrics_file}{Colors.RESET}")
        print(f"Waiting for training to start...")

    # Create monitor
    monitor = TrainingMonitor(
        log_dir=args.log_dir,
        metrics_file=args.metrics_file,
        refresh_interval=args.refresh,
        gpu_threshold=args.gpu_threshold,
        loss_stall_threshold=args.loss_stall
    )

    # Start monitoring
    monitor.monitor()


if __name__ == '__main__':
    main()
