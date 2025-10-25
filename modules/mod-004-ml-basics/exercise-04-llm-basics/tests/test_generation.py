"""Tests for Text Generation

These tests verify the text generation functionality and utilities.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestBasicGeneration:
    """Tests for basic text generation module."""

    @patch('src.basic_generation.pipeline')
    def test_initialize_generator(self, mock_pipeline):
        """Test generator initialization."""
        from src.basic_generation import initialize_generator

        mock_pipeline.return_value = Mock()
        generator = initialize_generator('gpt2', -1)

        assert generator is not None
        mock_pipeline.assert_called_once()

    @patch('src.basic_generation.pipeline')
    def test_generate_text(self, mock_pipeline):
        """Test text generation function."""
        from src.basic_generation import generate_text

        # Mock the pipeline
        mock_gen = Mock()
        mock_gen.return_value = [
            {'generated_text': 'Test output 1'},
            {'generated_text': 'Test output 2'}
        ]

        results = generate_text(
            mock_gen,
            prompt='Test',
            max_length=50,
            num_sequences=2
        )

        assert len(results) == 2
        assert results[0]['generated_text'] == 'Test output 1'
        mock_gen.assert_called_once()


class TestParameterExploration:
    """Tests for parameter exploration module."""

    @patch('src.parameter_exploration.pipeline')
    def test_parameter_explorer_init(self, mock_pipeline):
        """Test ParameterExplorer initialization."""
        from src.parameter_exploration import ParameterExplorer

        mock_pipeline.return_value = Mock()
        explorer = ParameterExplorer('gpt2', -1)

        assert explorer is not None
        assert explorer.model_name == 'gpt2'

    @patch('src.parameter_exploration.pipeline')
    def test_explore_temperature(self, mock_pipeline):
        """Test temperature exploration."""
        from src.parameter_exploration import ParameterExplorer

        mock_gen = Mock()
        mock_gen.return_value = [{'generated_text': 'Test output'}]
        mock_pipeline.return_value = mock_gen

        explorer = ParameterExplorer('gpt2', -1)
        results = explorer.explore_temperature('Test prompt')

        assert isinstance(results, dict)
        assert len(results) > 0

    @patch('src.parameter_exploration.pipeline')
    def test_explore_max_length(self, mock_pipeline):
        """Test max length exploration."""
        from src.parameter_exploration import ParameterExplorer

        mock_gen = Mock()
        mock_gen.return_value = [{'generated_text': 'Test output'}]
        mock_pipeline.return_value = mock_gen

        explorer = ParameterExplorer('gpt2', -1)
        results = explorer.explore_max_length('Test prompt')

        assert isinstance(results, dict)
        assert len(results) > 0


class TestResourceMonitor:
    """Tests for resource monitoring module."""

    def test_resource_monitor_init(self):
        """Test ResourceMonitor initialization."""
        from src.monitor_resources import ResourceMonitor

        monitor = ResourceMonitor()
        assert monitor is not None

    def test_get_memory_usage(self):
        """Test memory usage retrieval."""
        from src.monitor_resources import ResourceMonitor

        monitor = ResourceMonitor()
        memory = monitor.get_memory_usage()

        assert isinstance(memory, float)
        assert memory > 0

    def test_get_system_info(self):
        """Test system info retrieval."""
        from src.monitor_resources import ResourceMonitor

        monitor = ResourceMonitor()
        info = monitor.get_system_info()

        assert isinstance(info, dict)
        assert 'cpu_count' in info
        assert 'total_memory_gb' in info

    @patch('src.monitor_resources.pipeline')
    def test_monitor_model_loading(self, mock_pipeline):
        """Test model loading monitoring."""
        from src.monitor_resources import monitor_model_loading

        mock_gen = Mock()
        mock_pipeline.return_value = mock_gen

        metrics = monitor_model_loading('gpt2', -1)

        assert isinstance(metrics, dict)
        assert 'model_name' in metrics
        assert 'load_time_seconds' in metrics
        assert 'memory_delta_mb' in metrics


class TestModelComparator:
    """Tests for model comparison module."""

    def test_model_comparator_init(self):
        """Test ModelComparator initialization."""
        from src.compare_models import ModelComparator

        comparator = ModelComparator(-1)
        assert comparator is not None
        assert comparator.device == -1

    def test_get_memory_usage(self):
        """Test memory usage method."""
        from src.compare_models import ModelComparator

        comparator = ModelComparator(-1)
        memory = comparator.get_memory_usage()

        assert isinstance(memory, float)
        assert memory > 0

    @patch('src.compare_models.pipeline')
    def test_test_model_success(self, mock_pipeline):
        """Test successful model testing."""
        from src.compare_models import ModelComparator

        mock_gen = Mock()
        mock_gen.return_value = [{'generated_text': 'Test output'}]
        mock_pipeline.return_value = mock_gen

        comparator = ModelComparator(-1)
        metrics = comparator.test_model('gpt2')

        assert metrics['success'] is True
        assert 'load_time_seconds' in metrics
        assert 'memory_mb' in metrics
        assert 'inference_time_seconds' in metrics

    @patch('src.compare_models.pipeline')
    def test_test_model_failure(self, mock_pipeline):
        """Test model testing with failure."""
        from src.compare_models import ModelComparator

        mock_pipeline.side_effect = Exception('Model load failed')

        comparator = ModelComparator(-1)
        metrics = comparator.test_model('nonexistent-model')

        assert metrics['success'] is False
        assert 'error' in metrics


class TestIntegration:
    """Integration tests for complete workflows."""

    @patch('src.basic_generation.pipeline')
    def test_basic_generation_workflow(self, mock_pipeline):
        """Test complete basic generation workflow."""
        from src.basic_generation import initialize_generator, generate_text

        mock_gen = Mock()
        mock_gen.return_value = [
            {'generated_text': 'Generated text 1'},
            {'generated_text': 'Generated text 2'}
        ]
        mock_pipeline.return_value = mock_gen

        # Initialize
        generator = initialize_generator()
        assert generator is not None

        # Generate
        results = generate_text(generator, 'Test prompt', num_sequences=2)
        assert len(results) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
