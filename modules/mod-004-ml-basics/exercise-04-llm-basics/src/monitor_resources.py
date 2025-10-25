"""Resource Monitoring

This script monitors CPU, memory, and performance metrics when running LLM inference.
It helps understand the infrastructure requirements for deploying LLMs.
"""

import logging
import time
import psutil
from typing import Dict, Any
from transformers import pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Monitor system resources during LLM operations."""

    def __init__(self):
        """Initialize the resource monitor."""
        self.process = psutil.Process()

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB.

        Returns:
            Memory usage in megabytes
        """
        return self.process.memory_info().rss / 1024 / 1024

    def get_cpu_percent(self) -> float:
        """Get CPU usage percentage.

        Returns:
            CPU usage percentage
        """
        return self.process.cpu_percent(interval=1.0)

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information.

        Returns:
            Dictionary with system information
        """
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_physical": psutil.cpu_count(logical=False),
            "total_memory_gb": psutil.virtual_memory().total / (1024 ** 3),
            "available_memory_gb": psutil.virtual_memory().available / (1024 ** 3),
            "memory_percent": psutil.virtual_memory().percent
        }


def monitor_model_loading(model_name: str = 'gpt2', device: int = -1) -> Dict[str, Any]:
    """Monitor resources during model loading.

    Args:
        model_name: Name of model to load
        device: Device to use (-1 for CPU, 0+ for GPU)

    Returns:
        Dictionary with loading metrics
    """
    monitor = ResourceMonitor()

    print("=" * 80)
    print("MODEL LOADING MONITORING")
    print("=" * 80)

    # Get baseline
    mem_before = monitor.get_memory_usage()
    print(f"\nMemory before loading model: {mem_before:.2f} MB")

    # Load model and measure
    print(f"Loading model: {model_name}...")
    start_time = time.time()

    generator = pipeline('text-generation', model=model_name, device=device)

    load_time = time.time() - start_time
    mem_after = monitor.get_memory_usage()
    mem_delta = mem_after - mem_before

    print(f"Memory after loading model: {mem_after:.2f} MB")
    print(f"Model memory usage: {mem_delta:.2f} MB")
    print(f"Load time: {load_time:.2f} seconds")

    return {
        "model_name": model_name,
        "memory_before_mb": mem_before,
        "memory_after_mb": mem_after,
        "memory_delta_mb": mem_delta,
        "load_time_seconds": load_time,
        "generator": generator
    }


def monitor_inference(
    generator,
    prompt: str,
    max_length: int = 50,
    num_runs: int = 5
) -> Dict[str, Any]:
    """Monitor resources during inference.

    Args:
        generator: Text generation pipeline
        prompt: Input prompt
        max_length: Maximum generation length
        num_runs: Number of inference runs to average

    Returns:
        Dictionary with inference metrics
    """
    monitor = ResourceMonitor()

    print("\n" + "=" * 80)
    print("INFERENCE MONITORING")
    print("=" * 80)
    print(f"Prompt: '{prompt}'")
    print(f"Max Length: {max_length}")
    print(f"Number of runs: {num_runs}")
    print("-" * 80)

    inference_times = []
    memory_readings = []

    # Run multiple inferences
    for i in range(num_runs):
        logger.info(f"Running inference {i+1}/{num_runs}")

        mem_before = monitor.get_memory_usage()
        start_time = time.time()

        result = generator(
            prompt,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True
        )

        inference_time = time.time() - start_time
        mem_after = monitor.get_memory_usage()

        inference_times.append(inference_time)
        memory_readings.append(mem_after)

        if i == 0:  # Print first result
            print(f"\nFirst generation result:")
            print(f"{result[0]['generated_text']}")
            print("-" * 80)

    # Calculate statistics
    avg_inference_time = sum(inference_times) / len(inference_times)
    min_inference_time = min(inference_times)
    max_inference_time = max(inference_times)
    avg_memory = sum(memory_readings) / len(memory_readings)

    # Estimate tokens per second
    estimated_tokens = max_length - len(prompt.split())
    tokens_per_second = estimated_tokens / avg_inference_time if avg_inference_time > 0 else 0

    print(f"\nInference Statistics ({num_runs} runs):")
    print(f"  Average inference time: {avg_inference_time:.3f}s")
    print(f"  Min inference time: {min_inference_time:.3f}s")
    print(f"  Max inference time: {max_inference_time:.3f}s")
    print(f"  Average memory: {avg_memory:.2f} MB")
    print(f"  Estimated tokens/second: {tokens_per_second:.1f}")

    return {
        "prompt": prompt,
        "max_length": max_length,
        "num_runs": num_runs,
        "avg_inference_time": avg_inference_time,
        "min_inference_time": min_inference_time,
        "max_inference_time": max_inference_time,
        "avg_memory_mb": avg_memory,
        "tokens_per_second": tokens_per_second,
        "all_inference_times": inference_times
    }


def generate_resource_report(
    loading_metrics: Dict[str, Any],
    inference_metrics: Dict[str, Any]
):
    """Generate a comprehensive resource usage report.

    Args:
        loading_metrics: Metrics from model loading
        inference_metrics: Metrics from inference runs
    """
    monitor = ResourceMonitor()
    system_info = monitor.get_system_info()

    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESOURCE REPORT")
    print("=" * 80)

    print("\nSystem Information:")
    print(f"  CPU Cores (logical): {system_info['cpu_count']}")
    print(f"  CPU Cores (physical): {system_info['cpu_physical']}")
    print(f"  Total Memory: {system_info['total_memory_gb']:.2f} GB")
    print(f"  Available Memory: {system_info['available_memory_gb']:.2f} GB")
    print(f"  Memory Usage: {system_info['memory_percent']:.1f}%")

    print("\nModel Loading:")
    print(f"  Model: {loading_metrics['model_name']}")
    print(f"  Memory Required: {loading_metrics['memory_delta_mb']:.2f} MB")
    print(f"  Load Time: {loading_metrics['load_time_seconds']:.2f}s")

    print("\nInference Performance:")
    print(f"  Average Latency: {inference_metrics['avg_inference_time']:.3f}s")
    print(f"  Throughput: {inference_metrics['tokens_per_second']:.1f} tokens/second")
    print(f"  Memory During Inference: {inference_metrics['avg_memory_mb']:.2f} MB")

    print("\nRecommendations:")
    mem_required_gb = loading_metrics['memory_delta_mb'] / 1024
    if mem_required_gb < 1:
        print("  - Model is lightweight, suitable for small instances")
    elif mem_required_gb < 2:
        print("  - Model requires ~2GB RAM, use medium instances")
    else:
        print("  - Model is memory-intensive, use large instances")

    if inference_metrics['tokens_per_second'] < 10:
        print("  - Inference is slow on CPU, consider GPU acceleration")
    elif inference_metrics['tokens_per_second'] < 30:
        print("  - Inference speed is moderate, GPU recommended for production")
    else:
        print("  - Inference speed is good for CPU-based deployment")

    print("\nProduction Considerations:")
    print("  - Load model once at startup, not per-request")
    print("  - Implement request queuing for high concurrency")
    print("  - Set appropriate timeout values based on observed latency")
    print("  - Monitor memory usage to prevent OOM errors")
    print("  - Consider model quantization to reduce memory footprint")


def main():
    """Main execution function."""
    print("=" * 80)
    print("LLM RESOURCE MONITORING")
    print("=" * 80)

    # Monitor model loading
    loading_metrics = monitor_model_loading(model_name='gpt2', device=-1)

    # Monitor inference
    inference_metrics = monitor_inference(
        generator=loading_metrics['generator'],
        prompt="Machine learning is",
        max_length=50,
        num_runs=5
    )

    # Generate comprehensive report
    generate_resource_report(loading_metrics, inference_metrics)

    print("\n" + "=" * 80)
    print("Monitoring Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
