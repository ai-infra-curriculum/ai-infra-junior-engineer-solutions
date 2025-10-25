#!/usr/bin/env python3
"""
Model Inference Comparison Module

This module compares CPU vs GPU performance for real ML model inference
using transformer models from Hugging Face.
"""

import time
from typing import Dict, List, Optional
import torch
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. Install with: pip install transformers")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil library not available. Install with: pip install psutil")


def get_memory_usage_mb() -> float:
    """
    Get current process memory usage in MB.

    Returns:
        Memory usage in MB, or 0 if psutil not available
    """
    if not PSUTIL_AVAILABLE:
        return 0.0

    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def benchmark_model_inference(
    model_name: str,
    device: torch.device,
    prompt: str = "Machine learning infrastructure requires",
    max_length: int = 50,
    num_runs: int = 10
) -> Dict[str, any]:
    """
    Benchmark model inference on specified device.

    Args:
        model_name: Hugging Face model name
        device: Device to run on (CPU or CUDA)
        prompt: Text prompt for generation
        max_length: Maximum length of generated text
        num_runs: Number of inference runs for averaging

    Returns:
        Dictionary with benchmark results
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library required. Install with: pip install transformers")

    print(f"\n{'=' * 60}")
    print(f"Benchmarking on {device}")
    print(f"{'=' * 60}")

    # Memory before loading
    mem_before = get_memory_usage_mb()

    # Load model and tokenizer
    print(f"Loading model: {model_name}...")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    load_time = time.time() - start

    # Memory after loading
    mem_after = get_memory_usage_mb()
    model_memory = mem_after - mem_before

    print(f"✓ Model loaded in {load_time:.2f}s")
    print(f"  RAM usage: {model_memory:.2f} MB")

    # GPU memory if applicable
    gpu_memory = 0
    if device.type == 'cuda':
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"  GPU memory: {gpu_memory:.2f} MB")

    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Warmup runs
    print("Warming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = model.generate(**inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)

    # Benchmark runs
    print(f"Running {num_runs} inference iterations...")
    inference_times = []
    sample_output = None

    for i in range(num_runs):
        # Synchronize if GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                pad_token_id=tokenizer.eos_token_id
            )

        # Synchronize if GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start
        inference_times.append(elapsed)

        # Save first output as sample
        if i == 0:
            sample_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nSample output:\n{sample_output}\n")

    # Calculate statistics
    avg_time = sum(inference_times) / len(inference_times)
    min_time = min(inference_times)
    max_time = max(inference_times)
    tokens_per_sec = max_length / avg_time

    print(f"Inference Statistics:")
    print(f"  Average: {avg_time:.3f}s")
    print(f"  Min:     {min_time:.3f}s")
    print(f"  Max:     {max_time:.3f}s")
    print(f"  Tokens/second: ~{tokens_per_sec:.1f}")

    # Cleanup
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return {
        'device': str(device),
        'model_name': model_name,
        'load_time': load_time,
        'ram_memory_mb': model_memory,
        'gpu_memory_mb': gpu_memory,
        'avg_inference_time': avg_time,
        'min_inference_time': min_time,
        'max_inference_time': max_time,
        'tokens_per_sec': tokens_per_sec,
        'sample_output': sample_output,
        'num_runs': num_runs
    }


def compare_cpu_gpu_inference(
    model_name: str = "gpt2",
    prompt: str = "Machine learning infrastructure requires",
    max_length: int = 50,
    num_runs: int = 10
) -> Dict[str, any]:
    """
    Compare CPU and GPU inference performance.

    Args:
        model_name: Hugging Face model name
        prompt: Text prompt
        max_length: Maximum generation length
        num_runs: Number of runs per device

    Returns:
        Dictionary with comparison results
    """
    if not TRANSFORMERS_AVAILABLE:
        print("Error: transformers library not installed")
        print("Install with: pip install transformers")
        return {}

    print("=" * 60)
    print(f"Model Inference Benchmark: {model_name}")
    print("=" * 60)

    results = {}

    # CPU benchmark
    print("\n" + "=" * 60)
    print("CPU Inference Benchmark")
    print("=" * 60)

    cpu_result = benchmark_model_inference(
        model_name,
        torch.device('cpu'),
        prompt,
        max_length,
        num_runs
    )
    results['cpu'] = cpu_result

    # GPU benchmark (if available)
    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("GPU Inference Benchmark")
        print("=" * 60)

        gpu_result = benchmark_model_inference(
            model_name,
            torch.device('cuda'),
            prompt,
            max_length,
            num_runs
        )
        results['gpu'] = gpu_result

        # Print comparison
        print_comparison_summary(cpu_result, gpu_result)
    else:
        print("\n⚠ No GPU available for comparison")
        print("CPU-only results:")
        print(f"  Average inference time: {cpu_result['avg_inference_time']:.3f}s")
        print(f"  Tokens/second: {cpu_result['tokens_per_sec']:.1f}")

    return results


def print_comparison_summary(
    cpu_result: Dict[str, any],
    gpu_result: Dict[str, any]
) -> None:
    """
    Print formatted comparison summary.

    Args:
        cpu_result: CPU benchmark results
        gpu_result: GPU benchmark results
    """
    print(f"\n{'=' * 60}")
    print("Performance Comparison Summary")
    print(f"{'=' * 60}")

    # Inference speed
    speedup = cpu_result['avg_inference_time'] / gpu_result['avg_inference_time']
    print(f"\nInference Speedup: {speedup:.1f}x faster on GPU")
    print(f"  CPU: {cpu_result['avg_inference_time']:.3f}s per inference")
    print(f"  GPU: {gpu_result['avg_inference_time']:.3f}s per inference")

    # Throughput
    throughput_increase = (
        (gpu_result['tokens_per_sec'] - cpu_result['tokens_per_sec']) /
        cpu_result['tokens_per_sec'] * 100
    )
    print(f"\nThroughput:")
    print(f"  CPU: {cpu_result['tokens_per_sec']:.1f} tokens/s")
    print(f"  GPU: {gpu_result['tokens_per_sec']:.1f} tokens/s")
    print(f"  Increase: {throughput_increase:.1f}%")

    # Loading time
    print(f"\nModel Loading Time:")
    print(f"  CPU: {cpu_result['load_time']:.2f}s")
    print(f"  GPU: {gpu_result['load_time']:.2f}s")

    # Memory usage
    print(f"\nMemory Usage:")
    print(f"  CPU RAM: {cpu_result['ram_memory_mb']:.2f} MB")
    print(f"  GPU RAM: {gpu_result['ram_memory_mb']:.2f} MB")
    print(f"  GPU VRAM: {gpu_result['gpu_memory_mb']:.2f} MB")

    # Recommendations
    print(f"\n{'=' * 60}")
    print("Recommendations:")
    print(f"{'=' * 60}")

    if speedup > 2:
        print(f"✓ GPU provides {speedup:.1f}x speedup - recommended for production")
        print("  Use GPU for:")
        print("    - High-throughput inference")
        print("    - Real-time applications")
        print("    - Large batch processing")
    elif speedup > 1:
        print(f"○ GPU provides {speedup:.1f}x speedup - moderate improvement")
        print("  Consider:")
        print("    - Cost vs performance trade-off")
        print("    - Request volume")
        print("    - Latency requirements")
    else:
        print(f"⚠ Limited GPU benefit ({speedup:.1f}x) for this model size")
        print("  CPU may be more cost-effective for:")
        print("    - Small models")
        print("    - Low request volume")
        print("    - Development/testing")


def benchmark_batch_processing(
    model_name: str = "gpt2",
    batch_sizes: List[int] = [1, 4, 8, 16]
) -> None:
    """
    Benchmark different batch sizes to show GPU efficiency.

    Args:
        model_name: Hugging Face model name
        batch_sizes: List of batch sizes to test
    """
    if not torch.cuda.is_available():
        print("\nGPU not available - skipping batch processing benchmark")
        return

    if not TRANSFORMERS_AVAILABLE:
        print("\ntransformers library not available - skipping batch benchmark")
        return

    print("\n" + "=" * 60)
    print("Batch Processing Benchmark (GPU only)")
    print("=" * 60)

    # Load model once
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to('cuda')
    model.eval()

    prompts_base = ["Machine learning is", "GPU acceleration enables"]

    print(f"\n{'Batch Size':<12} {'Time/Batch (s)':<15} {'Time/Item (s)':<15} {'Throughput':<15}")
    print("-" * 60)

    for batch_size in batch_sizes:
        # Create batch
        prompts = prompts_base * (batch_size // 2)
        if len(prompts) < batch_size:
            prompts.extend(prompts_base[:batch_size - len(prompts)])

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

        # Warmup
        with torch.no_grad():
            _ = model.generate(**inputs, max_length=30, pad_token_id=tokenizer.eos_token_id)

        # Benchmark
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            _ = model.generate(**inputs, max_length=30, pad_token_id=tokenizer.eos_token_id)

        torch.cuda.synchronize()
        batch_time = time.time() - start

        time_per_item = batch_time / batch_size
        throughput = batch_size / batch_time

        print(
            f"{batch_size:<12} {batch_time:>13.3f}  "
            f"{time_per_item:>13.3f}  {throughput:>13.2f}/s"
        )

    # Cleanup
    del model
    torch.cuda.empty_cache()

    print("\nObservation: Larger batches improve GPU utilization and throughput")


def main():
    """Main function to run model inference comparisons."""
    if not TRANSFORMERS_AVAILABLE:
        print("Error: transformers library not installed")
        print("Install with: pip install transformers")
        return

    # Run main comparison
    results = compare_cpu_gpu_inference(
        model_name="gpt2",
        prompt="Artificial intelligence and machine learning",
        max_length=50,
        num_runs=10
    )

    # Run batch processing benchmark if GPU available
    if torch.cuda.is_available():
        benchmark_batch_processing()

    print("\n" + "=" * 60)
    print("Model Inference Benchmark Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
