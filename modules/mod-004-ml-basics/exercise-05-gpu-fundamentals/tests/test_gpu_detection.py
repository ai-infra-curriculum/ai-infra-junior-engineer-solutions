"""
Tests for GPU Detection Module

Tests GPU availability checking, device information retrieval,
and device creation functionality.
"""

import pytest
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from check_gpu import (
    check_gpu_availability,
    get_gpu_info,
    test_device_creation,
    get_device
)


class TestGPUDetection:
    """Test suite for GPU detection functionality."""

    def test_check_gpu_availability_returns_bool(self):
        """Test that GPU availability check returns a boolean."""
        result = check_gpu_availability()
        assert isinstance(result, bool)

    def test_check_gpu_availability_matches_torch(self):
        """Test that our function matches PyTorch's CUDA check."""
        result = check_gpu_availability()
        expected = torch.cuda.is_available()
        assert result == expected

    def test_get_gpu_info_returns_dict(self):
        """Test that GPU info returns a dictionary."""
        info = get_gpu_info()
        assert isinstance(info, dict)

    def test_get_gpu_info_has_required_keys(self):
        """Test that GPU info contains required keys."""
        info = get_gpu_info()
        required_keys = [
            'cuda_available',
            'gpu_count',
            'gpus',
            'pytorch_version',
            'python_version'
        ]
        for key in required_keys:
            assert key in info, f"Missing required key: {key}"

    def test_get_gpu_info_cuda_available_is_bool(self):
        """Test that cuda_available is a boolean."""
        info = get_gpu_info()
        assert isinstance(info['cuda_available'], bool)

    def test_get_gpu_info_gpu_count_is_int(self):
        """Test that gpu_count is an integer."""
        info = get_gpu_info()
        assert isinstance(info['gpu_count'], int)
        assert info['gpu_count'] >= 0

    def test_get_gpu_info_gpus_is_list(self):
        """Test that gpus is a list."""
        info = get_gpu_info()
        assert isinstance(info['gpus'], list)

    def test_get_gpu_info_gpu_count_matches_list_length(self):
        """Test that GPU count matches the length of GPU list."""
        info = get_gpu_info()
        assert info['gpu_count'] == len(info['gpus'])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_get_gpu_info_with_gpu(self):
        """Test GPU info when GPU is available."""
        info = get_gpu_info()
        assert info['cuda_available'] is True
        assert info['gpu_count'] > 0
        assert len(info['gpus']) > 0
        assert info['cuda_version'] is not None
        assert info['cudnn_version'] is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_get_gpu_info_gpu_properties(self):
        """Test that GPU properties contain expected fields."""
        info = get_gpu_info()
        if info['gpu_count'] > 0:
            gpu = info['gpus'][0]
            required_fields = [
                'id', 'name', 'compute_capability',
                'total_memory_gb', 'allocated_memory_gb',
                'reserved_memory_gb', 'multi_processor_count'
            ]
            for field in required_fields:
                assert field in gpu, f"Missing GPU field: {field}"

    def test_test_device_creation_returns_dict(self):
        """Test that device creation test returns a dictionary."""
        result = test_device_creation()
        assert isinstance(result, dict)

    def test_test_device_creation_has_required_keys(self):
        """Test that device creation test has required keys."""
        result = test_device_creation()
        required_keys = [
            'cpu_tensor_created',
            'gpu_tensor_created',
            'cpu_device',
            'gpu_device'
        ]
        for key in required_keys:
            assert key in result

    def test_cpu_tensor_creation(self):
        """Test that CPU tensor creation works."""
        result = test_device_creation()
        assert result['cpu_tensor_created'] is True
        assert result['cpu_device'] == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_tensor_creation(self):
        """Test that GPU tensor creation works when GPU available."""
        result = test_device_creation()
        assert result['gpu_tensor_created'] is True
        assert 'cuda' in result['gpu_device']

    def test_get_device_returns_torch_device(self):
        """Test that get_device returns a torch.device."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_get_device_cpu_fallback(self):
        """Test that get_device returns CPU when no GPU available."""
        device = get_device()
        if not torch.cuda.is_available():
            assert device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_get_device_uses_gpu_when_available(self):
        """Test that get_device returns CUDA when GPU available."""
        device = get_device()
        assert device.type == 'cuda'


class TestGPUInfoConsistency:
    """Test suite for consistency between different GPU info methods."""

    def test_cuda_availability_consistency(self):
        """Test consistency between different CUDA availability checks."""
        method1 = check_gpu_availability()
        method2 = get_gpu_info()['cuda_available']
        method3 = torch.cuda.is_available()

        assert method1 == method2 == method3

    def test_gpu_count_consistency(self):
        """Test consistency of GPU count."""
        info_count = get_gpu_info()['gpu_count']
        torch_count = torch.cuda.device_count()

        assert info_count == torch_count

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_name_consistency(self):
        """Test that GPU name matches PyTorch's report."""
        info = get_gpu_info()
        if info['gpu_count'] > 0:
            our_name = info['gpus'][0]['name']
            torch_name = torch.cuda.get_device_name(0)
            assert our_name == torch_name


class TestDeviceCreation:
    """Test suite for tensor creation on different devices."""

    def test_create_tensor_on_cpu(self):
        """Test creating tensor on CPU."""
        tensor = torch.randn(10, 10)
        assert tensor.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_create_tensor_on_gpu(self):
        """Test creating tensor on GPU."""
        tensor = torch.randn(10, 10, device='cuda')
        assert tensor.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_move_tensor_to_gpu(self):
        """Test moving tensor from CPU to GPU."""
        cpu_tensor = torch.randn(10, 10)
        gpu_tensor = cpu_tensor.to('cuda')

        assert cpu_tensor.device.type == 'cpu'
        assert gpu_tensor.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_move_tensor_to_cpu(self):
        """Test moving tensor from GPU to CPU."""
        gpu_tensor = torch.randn(10, 10, device='cuda')
        cpu_tensor = gpu_tensor.to('cpu')

        assert gpu_tensor.device.type == 'cuda'
        assert cpu_tensor.device.type == 'cpu'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
