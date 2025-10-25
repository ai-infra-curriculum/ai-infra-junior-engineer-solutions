#!/usr/bin/env python3
"""
benchmark_gpu.py - GPU benchmark tool for ML workloads

Description:
    Benchmark GPU performance for ML training with different configurations
    including batch sizes, model sizes, and data types.

Usage:
    python benchmark_gpu.py [OPTIONS]

Options:
    --framework FRAMEWORK   Framework to test (pytorch, tensorflow, both)
    --batch-sizes SIZES     Comma-separated batch sizes (default: 8,16,32,64)
    --iterations N          Number of iterations per test (default: 100)
    --warmup N             Warmup iterations (default: 10)
    --precision PRECISION   Precision (fp32, fp16, mixed)
    --report FILE          Save report to file
    --verbose              Verbose output
    --help                 Display this help message
"""

import argparse
import time
import json
from datetime import datetime
from typing import Dict, List, Optional
import sys


class Colors:
    """ANSI color codes"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class GPUBenchmark:
    """GPU benchmark class"""

    def __init__(self, framework: str = 'both', batch_sizes: List[int] = None,
                 iterations: int = 100, warmup: int = 10,
                 precision: str = 'fp32', verbose: bool = False):
        self.framework = framework
        self.batch_sizes = batch_sizes or [8, 16, 32, 64]
        self.iterations = iterations
        self.warmup = warmup
        self.precision = precision
        self.verbose = verbose

        # Results storage
        self.results = []

        # Check framework availability
        self.pytorch_available = self._check_pytorch()
        self.tensorflow_available = self._check_tensorflow()

    def _check_pytorch(self) -> bool:
        """Check if PyTorch is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _check_tensorflow(self) -> bool:
        """Check if TensorFlow is available"""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            return len(gpus) > 0
        except ImportError:
            return False

    def print_system_info(self):
        """Print system information"""
        print(f"{Colors.BOLD}{Colors.CYAN}System Information{Colors.RESET}")
        print("=" * 80)

        if self.pytorch_available:
            import torch
            print(f"\n{Colors.BOLD}PyTorch:{Colors.RESET}")
            print(f"  Version: {torch.__version__}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  GPU Count: {torch.cuda.device_count()}")
            if torch.cuda.is_available():
                print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
                props = torch.cuda.get_device_properties(0)
                print(f"  GPU Memory: {props.total_memory / 1e9:.2f} GB")
                print(f"  Compute Capability: {props.major}.{props.minor}")

        if self.tensorflow_available:
            import tensorflow as tf
            print(f"\n{Colors.BOLD}TensorFlow:{Colors.RESET}")
            print(f"  Version: {tf.__version__}")
            gpus = tf.config.list_physical_devices('GPU')
            print(f"  GPU Count: {len(gpus)}")
            for gpu in gpus:
                print(f"  GPU: {gpu.name}")

        print("\n" + "=" * 80 + "\n")

    def benchmark_pytorch(self, batch_size: int) -> Dict:
        """Benchmark PyTorch"""
        import torch
        import torch.nn as nn

        # Create model
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
                self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
                self.fc = nn.Linear(256 * 28 * 28, 1000)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.relu(self.conv3(x))
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        device = torch.device('cuda')
        model = SimpleModel().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        # Prepare data
        if self.precision == 'fp16':
            dtype = torch.float16
        else:
            dtype = torch.float32

        inputs = torch.randn(batch_size, 3, 224, 224, dtype=dtype, device=device)
        targets = torch.randint(0, 1000, (batch_size,), device=device)

        # Mixed precision setup
        scaler = None
        if self.precision == 'mixed':
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()

        # Warmup
        model.train()
        for _ in range(self.warmup):
            optimizer.zero_grad()
            if self.precision == 'mixed':
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()
        start_mem = torch.cuda.memory_allocated()

        for _ in range(self.iterations):
            optimizer.zero_grad()
            if self.precision == 'mixed':
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        torch.cuda.synchronize()
        end_time = time.time()
        end_mem = torch.cuda.memory_allocated()

        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / self.iterations
        throughput = batch_size * self.iterations / total_time
        memory_used = (end_mem - start_mem) / 1e9

        return {
            'framework': 'PyTorch',
            'batch_size': batch_size,
            'precision': self.precision,
            'total_time': total_time,
            'avg_time_per_batch': avg_time,
            'throughput': throughput,
            'memory_used_gb': memory_used
        }

    def benchmark_tensorflow(self, batch_size: int) -> Dict:
        """Benchmark TensorFlow"""
        import tensorflow as tf
        from tensorflow import keras

        # Create model
        model = keras.Sequential([
            keras.layers.Conv2D(64, 3, padding='same', activation='relu', input_shape=(224, 224, 3)),
            keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(1000, activation='softmax')
        ])

        # Compile model
        if self.precision == 'mixed':
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Prepare data
        import numpy as np
        inputs = np.random.randn(batch_size, 224, 224, 3).astype(np.float32)
        targets = np.random.randint(0, 1000, batch_size)

        # Warmup
        for _ in range(self.warmup):
            model.train_on_batch(inputs, targets)

        # Benchmark
        start_time = time.time()

        for _ in range(self.iterations):
            model.train_on_batch(inputs, targets)

        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / self.iterations
        throughput = batch_size * self.iterations / total_time

        return {
            'framework': 'TensorFlow',
            'batch_size': batch_size,
            'precision': self.precision,
            'total_time': total_time,
            'avg_time_per_batch': avg_time,
            'throughput': throughput,
            'memory_used_gb': 'N/A'
        }

    def run_benchmarks(self):
        """Run all benchmarks"""
        print(f"{Colors.BOLD}{Colors.CYAN}Running GPU Benchmarks{Colors.RESET}")
        print("=" * 80)
        print(f"Batch sizes: {self.batch_sizes}")
        print(f"Iterations: {self.iterations} (warmup: {self.warmup})")
        print(f"Precision: {self.precision}")
        print("=" * 80 + "\n")

        frameworks = []
        if self.framework in ['pytorch', 'both'] and self.pytorch_available:
            frameworks.append('pytorch')
        if self.framework in ['tensorflow', 'both'] and self.tensorflow_available:
            frameworks.append('tensorflow')

        if not frameworks:
            print(f"{Colors.RED}No frameworks available for benchmarking{Colors.RESET}")
            return

        for fw in frameworks:
            print(f"\n{Colors.BOLD}Benchmarking {fw.upper()}{Colors.RESET}")
            print("-" * 80)

            for batch_size in self.batch_sizes:
                print(f"\nBatch size: {batch_size}")

                try:
                    if fw == 'pytorch':
                        result = self.benchmark_pytorch(batch_size)
                    else:
                        result = self.benchmark_tensorflow(batch_size)

                    self.results.append(result)

                    # Print results
                    print(f"  {Colors.GREEN}✓{Colors.RESET} Time: {result['avg_time_per_batch']*1000:.2f} ms/batch")
                    print(f"  {Colors.GREEN}✓{Colors.RESET} Throughput: {result['throughput']:.1f} samples/sec")
                    if result['memory_used_gb'] != 'N/A':
                        print(f"  {Colors.GREEN}✓{Colors.RESET} Memory: {result['memory_used_gb']:.2f} GB")

                except Exception as e:
                    print(f"  {Colors.RED}✗{Colors.RESET} Error: {e}")
                    if self.verbose:
                        import traceback
                        traceback.print_exc()

    def print_summary(self):
        """Print benchmark summary"""
        if not self.results:
            return

        print(f"\n{Colors.BOLD}{Colors.CYAN}Benchmark Summary{Colors.RESET}")
        print("=" * 80)

        # Group by framework
        pytorch_results = [r for r in self.results if r['framework'] == 'PyTorch']
        tf_results = [r for r in self.results if r['framework'] == 'TensorFlow']

        for fw_results, fw_name in [(pytorch_results, 'PyTorch'), (tf_results, 'TensorFlow')]:
            if not fw_results:
                continue

            print(f"\n{Colors.BOLD}{fw_name}:{Colors.RESET}")
            print(f"{'Batch Size':<15} {'Avg Time (ms)':<20} {'Throughput (samples/s)':<25} {'Memory (GB)':<15}")
            print("-" * 80)

            for result in fw_results:
                mem_str = f"{result['memory_used_gb']:.2f}" if result['memory_used_gb'] != 'N/A' else 'N/A'
                print(f"{result['batch_size']:<15} "
                      f"{result['avg_time_per_batch']*1000:<20.2f} "
                      f"{result['throughput']:<25.1f} "
                      f"{mem_str:<15}")

        # Recommendations
        print(f"\n{Colors.BOLD}Recommendations:{Colors.RESET}")
        if pytorch_results:
            # Find optimal batch size (highest throughput)
            best = max(pytorch_results, key=lambda x: x['throughput'])
            print(f"  • Optimal batch size for PyTorch: {best['batch_size']} "
                  f"({best['throughput']:.1f} samples/sec)")

        if tf_results:
            best = max(tf_results, key=lambda x: x['throughput'])
            print(f"  • Optimal batch size for TensorFlow: {best['batch_size']} "
                  f"({best['throughput']:.1f} samples/sec)")

    def save_report(self, filename: str):
        """Save benchmark report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'framework': self.framework,
                'batch_sizes': self.batch_sizes,
                'iterations': self.iterations,
                'warmup': self.warmup,
                'precision': self.precision
            },
            'results': self.results
        }

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n{Colors.GREEN}Report saved to: {filename}{Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(
        description='GPU benchmark tool for ML workloads',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--framework', type=str, default='both',
                        choices=['pytorch', 'tensorflow', 'both'],
                        help='Framework to test (default: both)')
    parser.add_argument('--batch-sizes', type=str, default='8,16,32,64',
                        help='Comma-separated batch sizes (default: 8,16,32,64)')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of iterations per test (default: 100)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Warmup iterations (default: 10)')
    parser.add_argument('--precision', type=str, default='fp32',
                        choices=['fp32', 'fp16', 'mixed'],
                        help='Precision (default: fp32)')
    parser.add_argument('--report', type=str, default=None,
                        help='Save report to file')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(',')]

    # Create benchmark
    benchmark = GPUBenchmark(
        framework=args.framework,
        batch_sizes=batch_sizes,
        iterations=args.iterations,
        warmup=args.warmup,
        precision=args.precision,
        verbose=args.verbose
    )

    # Print system info
    benchmark.print_system_info()

    # Run benchmarks
    benchmark.run_benchmarks()

    # Print summary
    benchmark.print_summary()

    # Save report
    if args.report:
        benchmark.save_report(args.report)


if __name__ == '__main__':
    main()
