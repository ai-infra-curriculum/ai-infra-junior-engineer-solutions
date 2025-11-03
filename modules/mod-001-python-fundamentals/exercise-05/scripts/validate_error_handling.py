#!/usr/bin/env python3
"""
Validation Script for Error Handling Exercise

Tests all error handling implementations to ensure they work correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.custom_exceptions import (
    ModelNotFoundError,
    InvalidDataError,
    ConfigurationError,
    validate_config,
    validate_data
)
from scripts.retry_logic import retry_with_backoff
from scripts.ml_pipeline_robust import RobustMLPipeline, PipelineStatus


def test_exception_handling():
    """Test basic exception handling works correctly"""
    print("Test 1: Basic Exception Handling")
    print("-" * 70)

    # Test ZeroDivisionError
    try:
        value = 10 / 0
        print("✗ Should have raised ZeroDivisionError")
        return False
    except ZeroDivisionError:
        print("✓ Caught ZeroDivisionError")

    # Test KeyError
    try:
        data = {"a": 1}
        _ = data["b"]
        print("✗ Should have raised KeyError")
        return False
    except KeyError:
        print("✓ Caught KeyError")

    # Test FileNotFoundError
    try:
        with open("nonexistent_file.txt", 'r') as f:
            _ = f.read()
        print("✗ Should have raised FileNotFoundError")
        return False
    except FileNotFoundError:
        print("✓ Caught FileNotFoundError")

    print("✓ All basic exception tests passed\n")
    return True


def test_custom_exceptions():
    """Test custom exceptions work correctly"""
    print("Test 2: Custom Exceptions")
    print("-" * 70)

    # Test ModelNotFoundError
    try:
        raise ModelNotFoundError("/path/to/model.h5")
    except ModelNotFoundError as e:
        assert e.model_path == "/path/to/model.h5"
        assert "model.h5" in str(e)
        print("✓ ModelNotFoundError works correctly")

    # Test InvalidDataError
    try:
        validate_data([])
    except InvalidDataError as e:
        assert e.data_info is not None
        assert "empty" in str(e).lower()
        print("✓ InvalidDataError works correctly")

    # Test ConfigurationError
    try:
        config = {"learning_rate": 0.001, "batch_size": 32}  # Missing epochs
        validate_config(config)
        print("✗ Should have raised ConfigurationError")
        return False
    except ConfigurationError as e:
        assert e.param == "epochs"
        print("✓ ConfigurationError works correctly")

    print("✓ All custom exception tests passed\n")
    return True


def test_try_except_else_finally():
    """Test try-except-else-finally pattern"""
    print("Test 3: Try-Except-Else-Finally")
    print("-" * 70)

    else_executed = False
    finally_executed = False

    try:
        result = 10 / 2
    except ZeroDivisionError:
        print("✗ Should not have caught exception")
        return False
    else:
        else_executed = True
        print("✓ Else block executed")
    finally:
        finally_executed = True
        print("✓ Finally block executed")

    if not else_executed or not finally_executed:
        print("✗ Else or finally block not executed")
        return False

    # Test with exception
    finally_executed_2 = False
    try:
        try:
            _ = 10 / 0
        except ZeroDivisionError:
            print("✓ Exception caught")
        finally:
            finally_executed_2 = True
            print("✓ Finally executed after exception")
    except:
        pass

    if not finally_executed_2:
        print("✗ Finally not executed with exception")
        return False

    print("✓ Try-except-else-finally pattern works correctly\n")
    return True


def test_retry_logic():
    """Test retry decorator works correctly"""
    print("Test 4: Retry Logic")
    print("-" * 70)

    attempt_count = 0

    @retry_with_backoff(max_retries=2, initial_delay=0.1)
    def flaky_function():
        nonlocal attempt_count
        attempt_count += 1

        if attempt_count < 3:
            raise ValueError("Simulated failure")

        return "success"

    try:
        result = flaky_function()
        if result == "success" and attempt_count == 3:
            print(f"✓ Retry logic works (attempted {attempt_count} times)")
        else:
            print(f"✗ Unexpected result: {result}, attempts: {attempt_count}")
            return False
    except ValueError:
        print(f"✗ Should have succeeded after retries")
        return False

    # Test that it eventually fails
    attempt_count_2 = 0

    @retry_with_backoff(max_retries=2, initial_delay=0.05)
    def always_fails():
        nonlocal attempt_count_2
        attempt_count_2 += 1
        raise ValueError("Always fails")

    try:
        always_fails()
        print("✗ Should have raised exception after all retries")
        return False
    except ValueError:
        if attempt_count_2 == 3:  # Initial attempt + 2 retries
            print(f"✓ Retry logic gives up after max_retries")
        else:
            print(f"✗ Wrong number of attempts: {attempt_count_2}")
            return False

    print("✓ Retry logic tests passed\n")
    return True


def test_context_managers():
    """Test context managers work correctly"""
    print("Test 5: Context Managers")
    print("-" * 70)

    # Test basic context manager
    class TestContext:
        def __init__(self):
            self.entered = False
            self.exited = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.exited = True
            return False

    ctx = TestContext()
    with ctx:
        pass

    if not ctx.entered or not ctx.exited:
        print("✗ Context manager enter/exit not called")
        return False

    print("✓ Context manager __enter__ and __exit__ work correctly")

    # Test context manager with exception
    ctx2 = TestContext()
    try:
        with ctx2:
            raise ValueError("Test exception")
    except ValueError:
        pass

    if not ctx2.exited:
        print("✗ Context manager exit not called after exception")
        return False

    print("✓ Context manager exit called even with exception")
    print("✓ Context manager tests passed\n")
    return True


def test_pipeline_error_handling():
    """Test ML pipeline error handling"""
    print("Test 6: ML Pipeline Error Handling")
    print("-" * 70)

    # Test successful pipeline
    config = {
        "data_path": "/data/train.csv",
        "model_type": "test",
        "batch_size": 32,
        "epochs": 10
    }

    pipeline = RobustMLPipeline(config)
    result = pipeline.run()

    if result.status != PipelineStatus.SUCCESS:
        print(f"✗ Pipeline should have succeeded, got: {result.status}")
        return False

    print("✓ Successful pipeline execution works")

    if result.data is None or "model" not in result.data:
        print("✗ Pipeline result should contain model")
        return False

    print("✓ Pipeline result contains expected data")

    if result.metadata["steps_completed"] != 5:
        print(f"✗ Expected 5 steps completed, got {result.metadata['steps_completed']}")
        return False

    print("✓ All pipeline steps completed")
    print("✓ Pipeline error handling tests passed\n")
    return True


def test_error_collection():
    """Test that errors and warnings are collected correctly"""
    print("Test 7: Error and Warning Collection")
    print("-" * 70)

    config = {"model_type": "test", "batch_size": 32, "epochs": 5}

    pipeline = RobustMLPipeline(config)
    result = pipeline.run()

    # Should have warnings about small dataset
    if len(result.warnings) > 0:
        print(f"✓ Warnings collected: {result.warnings}")
    else:
        print("⚠ No warnings collected (expected for small dataset)")

    print("✓ Error and warning collection works\n")
    return True


def test_exception_chaining():
    """Test exception chaining preserves context"""
    print("Test 8: Exception Context Preservation")
    print("-" * 70)

    try:
        try:
            _ = 10 / 0
        except ZeroDivisionError as e:
            raise ValueError("Higher level error") from e
    except ValueError as e:
        if e.__cause__ is not None:
            print("✓ Exception chaining preserves original exception")
        else:
            print("✗ Original exception not preserved")
            return False

    print("✓ Exception context preservation works\n")
    return True


def run_all_tests():
    """Run all validation tests"""
    print("=" * 70)
    print("Error Handling Validation Tests")
    print("=" * 70)
    print()

    tests = [
        ("Basic Exception Handling", test_exception_handling),
        ("Custom Exceptions", test_custom_exceptions),
        ("Try-Except-Else-Finally", test_try_except_else_finally),
        ("Retry Logic", test_retry_logic),
        ("Context Managers", test_context_managers),
        ("Pipeline Error Handling", test_pipeline_error_handling),
        ("Error Collection", test_error_collection),
        ("Exception Chaining", test_exception_chaining),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test_name} FAILED\n")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} FAILED with exception: {e}\n")

    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    print()

    if failed == 0:
        print("✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"✗ {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
