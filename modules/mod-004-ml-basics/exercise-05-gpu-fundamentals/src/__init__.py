"""
GPU Fundamentals Exercise - Source Package

This package contains modules for GPU detection, benchmarking,
and memory management demonstrations.
"""

__version__ = "1.0.0"
__author__ = "AI Infrastructure Junior Engineer Training"

from .check_gpu import check_gpu_availability, get_gpu_info
from .cpu_vs_gpu_benchmark import benchmark_matmul, run_all_benchmarks
from .memory_management import track_gpu_memory, cleanup_gpu_memory
from .model_inference_comparison import benchmark_model_inference

__all__ = [
    "check_gpu_availability",
    "get_gpu_info",
    "benchmark_matmul",
    "run_all_benchmarks",
    "track_gpu_memory",
    "cleanup_gpu_memory",
    "benchmark_model_inference",
]
