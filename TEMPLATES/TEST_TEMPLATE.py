"""
Test Template for AI Infrastructure Exercises

This template provides a standard structure for writing tests for exercise solutions.
Follow pytest conventions and best practices.
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add src directory to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import your modules here
# from src.module_name import ClassName, function_name


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {
        'key1': 'value1',
        'key2': 'value2'
    }


@pytest.fixture
def mock_config():
    """Mock configuration object."""
    config = Mock()
    config.get.return_value = 'default_value'
    return config


@pytest.fixture(scope="module")
def setup_environment():
    """Set up test environment (runs once per module)."""
    # Setup code
    print("\nSetting up test environment...")

    yield  # Tests run here

    # Teardown code
    print("\nTearing down test environment...")


# ============================================================================
# UNIT TESTS
# ============================================================================

class TestClassName:
    """Test suite for ClassName"""

    def test_initialization(self):
        """Test object initialization."""
        # Arrange
        expected_value = "test"

        # Act
        # obj = ClassName(expected_value)

        # Assert
        # assert obj.value == expected_value
        pass

    def test_method_with_valid_input(self):
        """Test method with valid input."""
        # Arrange
        input_data = "valid input"
        expected_output = "expected result"

        # Act
        # result = method(input_data)

        # Assert
        # assert result == expected_output
        pass

    def test_method_with_invalid_input(self):
        """Test method handles invalid input correctly."""
        # Arrange
        invalid_input = None

        # Act & Assert
        # with pytest.raises(ValueError):
        #     method(invalid_input)
        pass

    def test_method_with_edge_case(self):
        """Test method with edge case."""
        # Arrange
        edge_case = ""  # Empty string, zero, etc.

        # Act
        # result = method(edge_case)

        # Assert
        # assert result is not None
        pass


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for multiple components working together"""

    def test_end_to_end_workflow(self, setup_environment):
        """Test complete workflow from start to finish."""
        # Arrange
        initial_state = {}

        # Act
        # Step 1: ...
        # Step 2: ...
        # Step 3: ...

        # Assert
        # assert final_state == expected_state
        pass

    def test_component_interaction(self):
        """Test interaction between components."""
        # Arrange
        # component_a = ComponentA()
        # component_b = ComponentB()

        # Act
        # result = component_a.interact_with(component_b)

        # Assert
        # assert result is not None
        pass


# ============================================================================
# MOCKING TESTS
# ============================================================================

class TestWithMocks:
    """Tests using mocks for external dependencies"""

    @patch('module.external_dependency')
    def test_with_mocked_dependency(self, mock_dependency):
        """Test with mocked external dependency."""
        # Arrange
        mock_dependency.return_value = "mocked value"

        # Act
        # result = function_that_uses_dependency()

        # Assert
        # assert result == "expected based on mock"
        # mock_dependency.assert_called_once()
        pass

    def test_with_environment_variables(self, monkeypatch):
        """Test with mocked environment variables."""
        # Arrange
        monkeypatch.setenv("TEST_VAR", "test_value")

        # Act
        # result = function_that_uses_env()

        # Assert
        # assert result == "test_value"
        pass


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

@pytest.mark.parametrize("input_value,expected_output", [
    ("input1", "output1"),
    ("input2", "output2"),
    ("input3", "output3"),
])
def test_multiple_inputs(input_value, expected_output):
    """Test function with multiple input/output pairs."""
    # Act
    # result = function(input_value)

    # Assert
    # assert result == expected_output
    pass


# ============================================================================
# ASYNC TESTS (if using async code)
# ============================================================================

@pytest.mark.asyncio
async def test_async_function():
    """Test asynchronous function."""
    # Arrange
    input_data = "test"

    # Act
    # result = await async_function(input_data)

    # Assert
    # assert result is not None
    pass


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

def test_function_performance():
    """Test that function completes within acceptable time."""
    import time

    # Arrange
    start_time = time.time()
    max_duration = 1.0  # seconds

    # Act
    # function_to_test()

    # Assert
    duration = time.time() - start_time
    assert duration < max_duration, f"Function took {duration}s, expected < {max_duration}s"


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Tests for error handling and edge cases"""

    def test_handles_file_not_found(self):
        """Test handling of file not found error."""
        # Act & Assert
        # with pytest.raises(FileNotFoundError):
        #     function_that_reads_file("nonexistent.txt")
        pass

    def test_handles_network_error(self):
        """Test handling of network errors."""
        # Mock network failure and verify graceful handling
        pass

    def test_handles_invalid_json(self):
        """Test handling of malformed JSON."""
        # Arrange
        invalid_json = "{ invalid }"

        # Act & Assert
        # with pytest.raises(json.JSONDecodeError):
        #     parse_json(invalid_json)
        pass


# ============================================================================
# CLEANUP
# ============================================================================

def teardown_module():
    """Clean up after all tests in module complete."""
    # Clean up test files, connections, etc.
    pass


# ============================================================================
# TEST MARKERS (for selective test running)
# ============================================================================

@pytest.mark.slow
def test_slow_operation():
    """Test that takes a long time to run."""
    # Run with: pytest -m slow
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    """Test for feature not yet implemented."""
    pass


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Requires Python 3.11+")
def test_python_311_feature():
    """Test for Python 3.11+ specific feature."""
    pass


# ============================================================================
# USAGE NOTES
# ============================================================================

"""
To run these tests:

# Run all tests
pytest test_module.py -v

# Run specific test class
pytest test_module.py::TestClassName -v

# Run specific test function
pytest test_module.py::test_function_name -v

# Run with coverage
pytest test_module.py --cov=src --cov-report=html

# Run only fast tests
pytest test_module.py -m "not slow"

# Run with output
pytest test_module.py -v -s

# Run and stop on first failure
pytest test_module.py -x

# Run and show local variables on failure
pytest test_module.py -l
"""
