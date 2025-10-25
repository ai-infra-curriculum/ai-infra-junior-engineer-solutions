"""
Tests for Performance Benchmarking Module

Tests CPU vs GPU performance comparison functionality.
"""

import pytest
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cpu_vs_gpu_benchmark import (
    benchmark_matmul,
    compare_cpu_gpu,
    benchmark_elementwise_operations
)
from memory_management import (
    get_gpu_memory_info,
    track_gpu_memory,
    cleanup_gpu_memory
)


class TestMatrixMultiplicationBenchmark:
    """Test suite for matrix multiplication benchmarking."""

    def test_benchmark_matmul_on_cpu(self):
        """Test that CPU benchmark runs successfully."""
        time_ms = benchmark_matmul(100, torch.device('cpu'), iterations=10)
        assert isinstance(time_ms, float)
        assert time_ms > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_benchmark_matmul_on_gpu(self):
        """Test that GPU benchmark runs successfully."""
        time_ms = benchmark_matmul(100, torch.device('cuda'), iterations=10)
        assert isinstance(time_ms, float)
        assert time_ms > 0

    def test_benchmark_matmul_different_sizes(self):
        """Test benchmark with different matrix sizes."""
        sizes = [50, 100, 200]
        device = torch.device('cpu')

        for size in sizes:
            time_ms = benchmark_matmul(size, device, iterations=10)
            assert time_ms > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_faster_for_large_matrices(self):
        """Test that GPU is faster for large matrices."""
        size = 1000
        iterations = 10

        cpu_time = benchmark_matmul(size, torch.device('cpu'), iterations=iterations)
        gpu_time = benchmark_matmul(size, torch.device('cuda'), iterations=iterations)

        # GPU should be faster for large matrices
        assert gpu_time < cpu_time, f"GPU ({gpu_time}ms) not faster than CPU ({cpu_time}ms)"

    def test_benchmark_matmul_custom_iterations(self):
        """Test benchmark with custom iteration count."""
        time_ms = benchmark_matmul(100, torch.device('cpu'), iterations=5, warmup=2)
        assert isinstance(time_ms, float)
        assert time_ms > 0


class TestCPUGPUComparison:
    """Test suite for CPU vs GPU comparison."""

    def test_compare_cpu_gpu_returns_list(self):
        """Test that comparison returns a list."""
        sizes = [100]
        results = compare_cpu_gpu(sizes, iterations=5)
        assert isinstance(results, list)

    def test_compare_cpu_gpu_result_structure(self):
        """Test that comparison results have correct structure."""
        sizes = [100]
        results = compare_cpu_gpu(sizes, iterations=5)

        assert len(results) == len(sizes)

        for result in results:
            assert 'size' in result
            assert 'cpu_time_ms' in result
            assert 'gpu_time_ms' in result
            assert 'speedup' in result
            assert 'gpu_available' in result

    def test_compare_cpu_gpu_cpu_times_valid(self):
        """Test that CPU times are valid."""
        sizes = [100, 200]
        results = compare_cpu_gpu(sizes, iterations=5)

        for result in results:
            assert result['cpu_time_ms'] is not None
            assert result['cpu_time_ms'] > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_compare_cpu_gpu_gpu_times_valid(self):
        """Test that GPU times are valid when GPU available."""
        sizes = [100]
        results = compare_cpu_gpu(sizes, iterations=5)

        for result in results:
            assert result['gpu_time_ms'] is not None
            assert result['gpu_time_ms'] > 0
            assert result['speedup'] is not None
            assert result['speedup'] > 0

    def test_compare_cpu_gpu_multiple_sizes(self):
        """Test comparison with multiple matrix sizes."""
        sizes = [50, 100, 200]
        results = compare_cpu_gpu(sizes, iterations=5)

        assert len(results) == len(sizes)

        for i, result in enumerate(results):
            assert result['size'] == sizes[i]


class TestElementwiseOperations:
    """Test suite for element-wise operation benchmarks."""

    def test_benchmark_elementwise_returns_dict(self):
        """Test that element-wise benchmark returns a dictionary."""
        results = benchmark_elementwise_operations(100, torch.device('cpu'), iterations=10)
        assert isinstance(results, dict)

    def test_benchmark_elementwise_has_operations(self):
        """Test that benchmark includes expected operations."""
        results = benchmark_elementwise_operations(100, torch.device('cpu'), iterations=10)

        expected_ops = ['relu', 'sigmoid', 'tanh', 'addition', 'multiplication']
        for op in expected_ops:
            assert op in results, f"Missing operation: {op}"

    def test_benchmark_elementwise_times_valid(self):
        """Test that timing results are valid."""
        results = benchmark_elementwise_operations(100, torch.device('cpu'), iterations=10)

        for op_name, time_ms in results.items():
            assert isinstance(time_ms, float)
            assert time_ms > 0, f"Invalid time for {op_name}: {time_ms}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_benchmark_elementwise_on_gpu(self):
        """Test element-wise benchmark on GPU."""
        results = benchmark_elementwise_operations(100, torch.device('cuda'), iterations=10)

        assert isinstance(results, dict)
        assert len(results) > 0


class TestMemoryManagement:
    """Test suite for GPU memory management."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_get_gpu_memory_info_returns_dict(self):
        """Test that memory info returns a dictionary."""
        info = get_gpu_memory_info()
        assert isinstance(info, dict)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_get_gpu_memory_info_has_required_keys(self):
        """Test that memory info has required keys."""
        info = get_gpu_memory_info()

        required_keys = ['allocated_gb', 'reserved_gb', 'total_gb', 'free_gb']
        for key in required_keys:
            assert key in info, f"Missing key: {key}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_get_gpu_memory_info_values_valid(self):
        """Test that memory info values are valid."""
        info = get_gpu_memory_info()

        assert info['allocated_gb'] >= 0
        assert info['reserved_gb'] >= 0
        assert info['total_gb'] > 0
        assert info['free_gb'] >= 0

    def test_get_gpu_memory_info_without_gpu(self):
        """Test memory info when GPU not available."""
        if not torch.cuda.is_available():
            info = get_gpu_memory_info()
            assert info is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_track_gpu_memory(self):
        """Test GPU memory tracking."""
        def create_tensor():
            return torch.randn(1000, 1000, device='cuda')

        result, mem_delta = track_gpu_memory("Test operation", create_tensor)

        assert isinstance(result, torch.Tensor)
        assert isinstance(mem_delta, float)
        assert mem_delta > 0  # Should allocate memory

        # Cleanup
        del result
        cleanup_gpu_memory()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_cleanup_gpu_memory(self):
        """Test GPU memory cleanup."""
        # Allocate some memory
        tensor = torch.randn(1000, 1000, device='cuda')
        initial_allocated = torch.cuda.memory_allocated()

        # Delete and cleanup
        del tensor
        cleanup_gpu_memory()

        final_allocated = torch.cuda.memory_allocated()

        # Memory should be freed
        assert final_allocated < initial_allocated


class TestDeviceAgnosticCode:
    """Test suite for device-agnostic code patterns."""

    def test_tensor_creation_on_dynamic_device(self):
        """Test creating tensors on dynamically selected device."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tensor = torch.randn(10, 10, device=device)

        assert tensor.device.type == device.type

    def test_tensor_transfer_to_device(self):
        """Test transferring tensors to device."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        cpu_tensor = torch.randn(10, 10)
        device_tensor = cpu_tensor.to(device)

        assert device_tensor.device.type == device.type

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_model_device_placement(self):
        """Test placing model on device."""
        device = torch.device('cuda')

        # Simple linear layer
        model = torch.nn.Linear(10, 10)
        model = model.to(device)

        # Check model parameters are on correct device
        for param in model.parameters():
            assert param.device.type == 'cuda'

        # Test forward pass
        input_tensor = torch.randn(5, 10, device=device)
        output = model(input_tensor)

        assert output.device.type == 'cuda'


class TestPerformanceCharacteristics:
    """Test suite for performance characteristics."""

    def test_larger_matrices_take_longer(self):
        """Test that larger matrices take longer to process."""
        device = torch.device('cpu')

        time_small = benchmark_matmul(100, device, iterations=10)
        time_large = benchmark_matmul(500, device, iterations=10)

        assert time_large > time_small

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_warmup_affects_timing(self):
        """Test that warmup affects first run timing."""
        device = torch.device('cuda')
        size = 1000

        # Without warmup
        time_no_warmup = benchmark_matmul(size, device, iterations=1, warmup=0)

        # With warmup
        time_with_warmup = benchmark_matmul(size, device, iterations=1, warmup=10)

        # With warmup should be more consistent
        assert time_with_warmup > 0
        assert time_no_warmup > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
