"""Model quantization for faster inference.

This module provides utilities for quantizing PyTorch models using:
- Dynamic quantization (no calibration needed)
- Static quantization (requires calibration dataset)
- Quantization-aware training (train with quantization in mind)
"""

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, QuantStub, DeQuantStub
from torch.quantization import prepare, convert
import time
import numpy as np
from pathlib import Path


class SimpleMLModel(nn.Module):
    """Example ML model for demonstration."""

    def __init__(self, input_size=10, hidden_size=64, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class QuantizableModel(nn.Module):
    """Model with QuantStub/DeQuantStub for static quantization."""

    def __init__(self, model):
        super().__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


def dynamic_quantization(model, dtype=torch.qint8):
    """
    Apply dynamic quantization to model.

    Best for:
    - Models with Linear/LSTM layers
    - Quick optimization without calibration
    - CPU inference

    Args:
        model: PyTorch model
        dtype: Quantization dtype (qint8, float16)

    Returns:
        Quantized model
    """
    print("=" * 50)
    print("DYNAMIC QUANTIZATION")
    print("=" * 50)

    # Quantize Linear layers
    quantized_model = quantize_dynamic(
        model,
        {nn.Linear},  # Layers to quantize
        dtype=dtype
    )

    print(f"✓ Model quantized to {dtype}")
    print(f"✓ Quantized layers: Linear")

    return quantized_model


def static_quantization(model, calibration_data):
    """
    Apply static quantization with calibration.

    Best for:
    - Maximum accuracy
    - When calibration data available
    - CNN models

    Args:
        model: PyTorch model
        calibration_data: Representative dataset for calibration

    Returns:
        Statically quantized model
    """
    print("=" * 50)
    print("STATIC QUANTIZATION")
    print("=" * 50)

    # Wrap model with quant/dequant stubs
    quantizable_model = QuantizableModel(model)
    quantizable_model.eval()

    # Set quantization configuration
    quantizable_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Prepare for quantization (insert observers)
    prepared_model = prepare(quantizable_model)

    print("Calibrating model...")
    # Calibrate with representative data
    with torch.no_grad():
        for data in calibration_data:
            prepared_model(data)

    print("Converting to quantized model...")
    # Convert to quantized model
    quantized_model = convert(prepared_model)

    print("✓ Model statically quantized")
    print("✓ Calibration completed")

    return quantized_model


def compare_models(original_model, quantized_model, test_data, num_iterations=1000):
    """
    Compare performance of original vs quantized model.

    Args:
        original_model: Original float32 model
        quantized_model: Quantized model
        test_data: Test input tensor
        num_iterations: Number of benchmark iterations

    Returns:
        Dictionary with performance metrics
    """
    print("\n" + "=" * 50)
    print("PERFORMANCE COMPARISON")
    print("=" * 50)

    original_model.eval()
    quantized_model.eval()

    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = original_model(test_data)
            _ = quantized_model(test_data)

    # Benchmark original model
    original_times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = original_model(test_data)
            original_times.append(time.perf_counter() - start)

    # Benchmark quantized model
    quantized_times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = quantized_model(test_data)
            quantized_times.append(time.perf_counter() - start)

    # Calculate metrics
    original_times = np.array(original_times) * 1000  # Convert to ms
    quantized_times = np.array(quantized_times) * 1000

    speedup = np.mean(original_times) / np.mean(quantized_times)

    # Model sizes
    def get_model_size(model):
        torch.save(model.state_dict(), "/tmp/temp_model.pth")
        size_mb = Path("/tmp/temp_model.pth").stat().st_size / (1024 * 1024)
        return size_mb

    original_size = get_model_size(original_model)
    quantized_size = get_model_size(quantized_model)
    size_reduction = (1 - quantized_size / original_size) * 100

    # Print results
    print(f"\nOriginal Model:")
    print(f"  Mean latency: {np.mean(original_times):.2f} ms")
    print(f"  P95 latency:  {np.percentile(original_times, 95):.2f} ms")
    print(f"  P99 latency:  {np.percentile(original_times, 99):.2f} ms")
    print(f"  Model size:   {original_size:.2f} MB")

    print(f"\nQuantized Model:")
    print(f"  Mean latency: {np.mean(quantized_times):.2f} ms")
    print(f"  P95 latency:  {np.percentile(quantized_times, 95):.2f} ms")
    print(f"  P99 latency:  {np.percentile(quantized_times, 99):.2f} ms")
    print(f"  Model size:   {quantized_size:.2f} MB")

    print(f"\nImprovement:")
    print(f"  Speedup:          {speedup:.2f}x")
    print(f"  Size reduction:   {size_reduction:.1f}%")
    print(f"  Throughput gain:  {speedup:.2f}x")

    return {
        "original_latency_ms": float(np.mean(original_times)),
        "quantized_latency_ms": float(np.mean(quantized_times)),
        "speedup": float(speedup),
        "original_size_mb": float(original_size),
        "quantized_size_mb": float(quantized_size),
        "size_reduction_pct": float(size_reduction),
    }


def save_quantized_model(model, output_path):
    """
    Save quantized model for inference.

    Args:
        model: Quantized PyTorch model
        output_path: Path to save model
    """
    # Save using TorchScript for deployment
    model.eval()
    scripted_model = torch.jit.script(model)
    scripted_model.save(output_path)
    print(f"\n✓ Quantized model saved to: {output_path}")


def load_quantized_model(model_path):
    """
    Load quantized model for inference.

    Args:
        model_path: Path to saved quantized model

    Returns:
        Loaded quantized model
    """
    model = torch.jit.load(model_path)
    model.eval()
    print(f"✓ Quantized model loaded from: {model_path}")
    return model


# Example usage
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("MODEL QUANTIZATION DEMONSTRATION")
    print("=" * 50)

    # Create example model
    input_size = 10
    model = SimpleMLModel(input_size=input_size, hidden_size=128, num_classes=3)

    # Create sample data
    batch_size = 32
    test_input = torch.randn(batch_size, input_size)

    # Create calibration dataset (for static quantization)
    calibration_data = [
        torch.randn(batch_size, input_size) for _ in range(100)
    ]

    print(f"\nModel architecture:")
    print(model)
    print(f"\nInput shape: {test_input.shape}")

    # 1. Dynamic Quantization
    print("\n\n1. TESTING DYNAMIC QUANTIZATION")
    print("-" * 50)
    dynamic_quant_model = dynamic_quantization(model)
    dynamic_metrics = compare_models(model, dynamic_quant_model, test_input)

    # Save dynamic quantized model
    save_quantized_model(
        dynamic_quant_model,
        "models/model_dynamic_quantized.pt"
    )

    # 2. Static Quantization
    print("\n\n2. TESTING STATIC QUANTIZATION")
    print("-" * 50)
    static_quant_model = static_quantization(model, calibration_data)
    static_metrics = compare_models(model, static_quant_model, test_input)

    # Save static quantized model
    save_quantized_model(
        static_quant_model,
        "models/model_static_quantized.pt"
    )

    # Compare both quantization methods
    print("\n\n" + "=" * 50)
    print("QUANTIZATION METHOD COMPARISON")
    print("=" * 50)
    print(f"\n{'Metric':<25} {'Dynamic':<15} {'Static':<15}")
    print("-" * 55)
    print(f"{'Speedup':<25} {dynamic_metrics['speedup']:.2f}x{'':<10} {static_metrics['speedup']:.2f}x")
    print(f"{'Size reduction':<25} {dynamic_metrics['size_reduction_pct']:.1f}%{'':<10} {static_metrics['size_reduction_pct']:.1f}%")
    print(f"{'Latency (ms)':<25} {dynamic_metrics['quantized_latency_ms']:.2f}{'':<12} {static_metrics['quantized_latency_ms']:.2f}")
    print(f"{'Model size (MB)':<25} {dynamic_metrics['quantized_size_mb']:.2f}{'':<12} {static_metrics['quantized_size_mb']:.2f}")

    print("\n\nRECOMMENDATIONS:")
    print("-" * 50)
    print("✓ Dynamic quantization: Fast, no calibration needed")
    print("  Best for: LSTM/RNN models, quick optimization")
    print("\n✓ Static quantization: Better accuracy, requires calibration")
    print("  Best for: CNN models, maximum performance")

    print("\n" + "=" * 50)
    print("QUANTIZATION COMPLETE")
    print("=" * 50)
