"""ONNX model conversion and optimization.

Converts PyTorch/TensorFlow models to ONNX format and applies
graph optimizations for faster inference.
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
import time
from pathlib import Path


class ExampleModel(nn.Module):
    """Example PyTorch model."""

    def __init__(self, input_size=10, hidden_size=128, num_classes=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def convert_pytorch_to_onnx(
    model,
    input_shape,
    output_path="models/model.onnx",
    opset_version=13,
    dynamic_axes=None,
):
    """
    Convert PyTorch model to ONNX format.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (tuple)
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
        dynamic_axes: Dict of dynamic axes for variable batch size

    Returns:
        Path to saved ONNX model
    """
    print("=" * 60)
    print("CONVERTING PYTORCH MODEL TO ONNX")
    print("=" * 60)

    model.eval()
    dummy_input = torch.randn(*input_shape)

    # Default dynamic axes for variable batch size
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }

    print(f"\nInput shape: {input_shape}")
    print(f"Opset version: {opset_version}")
    print(f"Dynamic axes: {dynamic_axes}")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,  # Optimization
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
    )

    # Verify the model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # Get model size
    model_size_mb = Path(output_path).stat().st_size / (1024 * 1024)

    print(f"\n✓ Model exported to ONNX: {output_path}")
    print(f"✓ Model size: {model_size_mb:.2f} MB")
    print(f"✓ Model validated successfully")

    return output_path


def optimize_onnx_model(input_path, output_path="models/model_optimized.onnx"):
    """
    Apply graph optimizations to ONNX model.

    Optimizations include:
    - Constant folding
    - Dead code elimination
    - Operator fusion
    - Graph simplification

    Args:
        input_path: Path to input ONNX model
        output_path: Path to save optimized model

    Returns:
        Path to optimized model
    """
    print("\n" + "=" * 60)
    print("OPTIMIZING ONNX MODEL")
    print("=" * 60)

    import onnx
    from onnx import optimizer

    # Load model
    model = onnx.load(input_path)

    # Available optimization passes
    optimization_passes = [
        'eliminate_deadend',
        'eliminate_identity',
        'eliminate_nop_dropout',
        'eliminate_nop_monotone_argmax',
        'eliminate_nop_pad',
        'extract_constant_to_initializer',
        'eliminate_unused_initializer',
        'fuse_add_bias_into_conv',
        'fuse_bn_into_conv',
        'fuse_consecutive_concats',
        'fuse_consecutive_reduce_unsqueeze',
        'fuse_consecutive_squeezes',
        'fuse_consecutive_transposes',
        'fuse_matmul_add_bias_into_gemm',
        'fuse_pad_into_conv',
        'fuse_transpose_into_gemm',
    ]

    print(f"\nApplying {len(optimization_passes)} optimization passes...")

    # Apply optimizations
    optimized_model = optimizer.optimize(model, optimization_passes)

    # Save optimized model
    onnx.save(optimized_model, output_path)

    # Compare sizes
    original_size = Path(input_path).stat().st_size / (1024 * 1024)
    optimized_size = Path(output_path).stat().st_size / (1024 * 1024)
    reduction = (1 - optimized_size / original_size) * 100

    print(f"\n✓ Optimizations applied")
    print(f"✓ Original size:  {original_size:.2f} MB")
    print(f"✓ Optimized size: {optimized_size:.2f} MB")
    print(f"✓ Size reduction: {reduction:.1f}%")
    print(f"✓ Saved to: {output_path}")

    return output_path


def quantize_onnx_model(input_path, output_path="models/model_quantized.onnx"):
    """
    Apply dynamic quantization to ONNX model.

    Args:
        input_path: Path to input ONNX model
        output_path: Path to save quantized model

    Returns:
        Path to quantized model
    """
    print("\n" + "=" * 60)
    print("QUANTIZING ONNX MODEL")
    print("=" * 60)

    # Quantize model (int8)
    quantize_dynamic(
        input_path,
        output_path,
        weight_type=QuantType.QInt8,
    )

    # Compare sizes
    original_size = Path(input_path).stat().st_size / (1024 * 1024)
    quantized_size = Path(output_path).stat().st_size / (1024 * 1024)
    reduction = (1 - quantized_size / original_size) * 100

    print(f"\n✓ Model quantized to int8")
    print(f"✓ Original size:   {original_size:.2f} MB")
    print(f"✓ Quantized size:  {quantized_size:.2f} MB")
    print(f"✓ Size reduction:  {reduction:.1f}%")
    print(f"✓ Saved to: {output_path}")

    return output_path


class ONNXInferenceSession:
    """Wrapper for ONNX Runtime inference."""

    def __init__(self, model_path, use_gpu=False):
        """
        Initialize ONNX inference session.

        Args:
            model_path: Path to ONNX model
            use_gpu: Whether to use GPU acceleration
        """
        # Set execution providers
        providers = []
        if use_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')

        # Create session
        self.session = ort.InferenceSession(model_path, providers=providers)

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"✓ ONNX Runtime session created")
        print(f"  Model: {model_path}")
        print(f"  Providers: {self.session.get_providers()}")
        print(f"  Input name: {self.input_name}")
        print(f"  Output name: {self.output_name}")

    def predict(self, input_data):
        """
        Run inference.

        Args:
            input_data: numpy array

        Returns:
            Prediction output
        """
        # Ensure numpy array
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.cpu().numpy()

        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data.astype(np.float32)}
        )

        return outputs[0]

    def benchmark(self, input_data, num_iterations=1000):
        """
        Benchmark inference performance.

        Args:
            input_data: Test input
            num_iterations: Number of iterations

        Returns:
            Performance metrics
        """
        # Warm up
        for _ in range(10):
            self.predict(input_data)

        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self.predict(input_data)
            latencies.append((time.perf_counter() - start) * 1000)  # ms

        latencies = np.array(latencies)

        return {
            "mean_ms": float(np.mean(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "throughput_rps": float(1000 / np.mean(latencies)),
        }


def compare_pytorch_vs_onnx(pytorch_model, onnx_path, test_input, num_iterations=1000):
    """
    Compare PyTorch vs ONNX inference performance.

    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to ONNX model
        test_input: Test input tensor
        num_iterations: Benchmark iterations

    Returns:
        Comparison metrics
    """
    print("\n" + "=" * 60)
    print("PYTORCH VS ONNX PERFORMANCE COMPARISON")
    print("=" * 60)

    pytorch_model.eval()

    # Benchmark PyTorch
    print("\nBenchmarking PyTorch model...")
    pytorch_times = []
    with torch.no_grad():
        # Warm up
        for _ in range(10):
            _ = pytorch_model(test_input)

        # Benchmark
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = pytorch_model(test_input)
            pytorch_times.append((time.perf_counter() - start) * 1000)

    pytorch_times = np.array(pytorch_times)

    # Benchmark ONNX
    print("Benchmarking ONNX model...")
    onnx_session = ONNXInferenceSession(onnx_path)
    onnx_metrics = onnx_session.benchmark(test_input, num_iterations)

    # Calculate speedup
    speedup = np.mean(pytorch_times) / onnx_metrics['mean_ms']

    # Print results
    print(f"\n{'Metric':<20} {'PyTorch':<15} {'ONNX':<15} {'Speedup':<10}")
    print("-" * 60)
    print(f"{'Mean latency (ms)':<20} {np.mean(pytorch_times):>12.2f}    {onnx_metrics['mean_ms']:>12.2f}    {speedup:>6.2f}x")
    print(f"{'P95 latency (ms)':<20} {np.percentile(pytorch_times, 95):>12.2f}    {onnx_metrics['p95_ms']:>12.2f}")
    print(f"{'P99 latency (ms)':<20} {np.percentile(pytorch_times, 99):>12.2f}    {onnx_metrics['p99_ms']:>12.2f}")
    print(f"{'Throughput (RPS)':<20} {1000/np.mean(pytorch_times):>12.0f}    {onnx_metrics['throughput_rps']:>12.0f}")

    return {
        "pytorch_mean_ms": float(np.mean(pytorch_times)),
        "onnx_mean_ms": float(onnx_metrics['mean_ms']),
        "speedup": float(speedup),
    }


# Example usage
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ONNX MODEL OPTIMIZATION PIPELINE")
    print("=" * 60)

    # Create output directory
    Path("models").mkdir(exist_ok=True)

    # 1. Create PyTorch model
    input_size = 10
    batch_size = 32
    model = ExampleModel(input_size=input_size, hidden_size=128, num_classes=3)

    # Create dummy input
    test_input = torch.randn(batch_size, input_size)

    print(f"\nModel: {model.__class__.__name__}")
    print(f"Input shape: {test_input.shape}")

    # 2. Convert to ONNX
    onnx_path = convert_pytorch_to_onnx(
        model,
        input_shape=(batch_size, input_size),
        output_path="models/model.onnx"
    )

    # 3. Optimize ONNX model
    optimized_path = optimize_onnx_model(
        onnx_path,
        output_path="models/model_optimized.onnx"
    )

    # 4. Quantize ONNX model
    quantized_path = quantize_onnx_model(
        optimized_path,
        output_path="models/model_quantized.onnx"
    )

    # 5. Compare performance
    print("\n\n" + "=" * 60)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 60)

    print("\n--- Original ONNX ---")
    onnx_session = ONNXInferenceSession(onnx_path)
    original_metrics = onnx_session.benchmark(test_input)

    print("\n--- Optimized ONNX ---")
    optimized_session = ONNXInferenceSession(optimized_path)
    optimized_metrics = optimized_session.benchmark(test_input)

    print("\n--- Quantized ONNX ---")
    quantized_session = ONNXInferenceSession(quantized_path)
    quantized_metrics = quantized_session.benchmark(test_input)

    # 6. PyTorch vs ONNX comparison
    pytorch_vs_onnx = compare_pytorch_vs_onnx(
        model,
        quantized_path,
        test_input
    )

    # 7. Summary
    print("\n\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)

    print(f"\n{'Version':<20} {'Latency (ms)':<15} {'Throughput (RPS)':<20} {'Speedup':<10}")
    print("-" * 65)

    baseline = original_metrics['mean_ms']
    print(f"{'Original ONNX':<20} {original_metrics['mean_ms']:>12.2f}    {original_metrics['throughput_rps']:>15.0f}       {1.0:>6.2f}x")
    print(f"{'Optimized ONNX':<20} {optimized_metrics['mean_ms']:>12.2f}    {optimized_metrics['throughput_rps']:>15.0f}       {baseline/optimized_metrics['mean_ms']:>6.2f}x")
    print(f"{'Quantized ONNX':<20} {quantized_metrics['mean_ms']:>12.2f}    {quantized_metrics['throughput_rps']:>15.0f}       {baseline/quantized_metrics['mean_ms']:>6.2f}x")

    print("\n\nRECOMMENDATIONS:")
    print("-" * 60)
    print("✓ ONNX provides cross-platform compatibility")
    print("✓ Graph optimizations improve performance 10-30%")
    print("✓ Quantization provides 2-4x speedup with minimal accuracy loss")
    print("✓ Use quantized ONNX for production CPU inference")
    print("✓ Use TensorRT (NVIDIA) or OpenVINO (Intel) for hardware-specific optimization")

    print("\n" + "=" * 60)
    print("ONNX OPTIMIZATION COMPLETE")
    print("=" * 60)
