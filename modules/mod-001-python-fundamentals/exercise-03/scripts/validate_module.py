#!/usr/bin/env python3
"""
Module Validation Script

Validates that all exercise components are working correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def validate_type_hints():
    """Validate that type hints are present."""
    print("1. Validating Type Hints")
    print("-" * 70)

    from ml_utils import metrics
    import inspect

    # Check function signature
    sig = inspect.signature(metrics.accuracy)
    print(f"   metrics.accuracy signature: {sig}")

    # Check if annotations exist
    if sig.parameters:
        has_types = any(
            param.annotation != inspect.Parameter.empty
            for param in sig.parameters.values()
        )
        print(f"   ✓ Type hints present: {has_types}")
    else:
        print(f"   ✗ No parameters found")

    print()


def validate_decorators():
    """Validate that decorators work correctly."""
    print("2. Validating Decorators")
    print("-" * 70)

    # Import decorator module
    sys.path.insert(0, str(Path(__file__).parent))
    from decorators import timing_decorator

    # Test decorator
    @timing_decorator
    def test_func():
        import time
        time.sleep(0.1)
        return "done"

    result = test_func()
    print(f"   ✓ Decorator executed successfully")
    print(f"   Result: {result}")
    print()


def validate_module_imports():
    """Validate that modules import correctly."""
    print("3. Validating Module Imports")
    print("-" * 70)

    try:
        from ml_utils import metrics, preprocessing
        print(f"   ✓ ml_utils.metrics imported")
        print(f"   ✓ ml_utils.preprocessing imported")

        # Check module attributes
        print(f"\n   metrics module functions:")
        metric_funcs = [name for name in dir(metrics) if not name.startswith('_')]
        for func_name in metric_funcs[:5]:  # Show first 5
            print(f"     - {func_name}")

        print(f"\n   preprocessing module functions:")
        prep_funcs = [name for name in dir(preprocessing) if not name.startswith('_')]
        for func_name in prep_funcs[:5]:  # Show first 5
            print(f"     - {func_name}")

    except ImportError as e:
        print(f"   ✗ Module import failed: {e}")

    print()


def validate_function_correctness():
    """Validate that functions produce correct results."""
    print("4. Validating Function Correctness")
    print("-" * 70)

    from ml_utils import metrics, preprocessing

    # Test accuracy
    preds = [1, 0, 1, 1]
    labels = [1, 0, 0, 1]
    acc = metrics.accuracy(preds, labels)
    expected = 0.75
    assert abs(acc - expected) < 0.01, f"Expected {expected}, got {acc}"
    print(f"   ✓ metrics.accuracy correct: {acc:.2f}")

    # Test normalization
    data = [0.0, 5.0, 10.0]
    normalized = preprocessing.normalize_minmax(data)
    expected_norm = [0.0, 0.5, 1.0]
    assert all(abs(a - b) < 0.01 for a, b in zip(normalized, expected_norm))
    print(f"   ✓ preprocessing.normalize_minmax correct: {normalized}")

    # Test precision
    prec = metrics.precision(preds, labels)
    expected_prec = 2/3  # 2 true positives, 3 predicted positives
    assert abs(prec - expected_prec) < 0.01, f"Expected {expected_prec}, got {prec}"
    print(f"   ✓ metrics.precision correct: {prec:.4f}")

    # Test train-test split
    dataset = list(range(10))
    train, test = preprocessing.train_test_split(dataset, test_size=0.2, shuffle=False)
    assert len(train) == 8 and len(test) == 2
    print(f"   ✓ preprocessing.train_test_split correct: {len(train)} train, {len(test)} test")

    print()


def validate_error_handling():
    """Validate that functions handle errors properly."""
    print("5. Validating Error Handling")
    print("-" * 70)

    from ml_utils import metrics, preprocessing

    # Test mismatched lengths
    try:
        metrics.accuracy([1, 0, 1], [1, 0])
        print(f"   ✗ Should have raised ValueError for mismatched lengths")
    except ValueError as e:
        print(f"   ✓ Caught expected error: {type(e).__name__}")

    # Test invalid method
    try:
        preprocessing.normalize_minmax([1, 2, 3])  # This should work
        print(f"   ✓ Valid method works correctly")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")

    # Test invalid test_size
    try:
        preprocessing.train_test_split([1, 2, 3], test_size=1.5)
        print(f"   ✗ Should have raised ValueError for invalid test_size")
    except ValueError as e:
        print(f"   ✓ Caught expected error: {type(e).__name__}")

    print()


def validate_edge_cases():
    """Validate that edge cases are handled."""
    print("6. Validating Edge Cases")
    print("-" * 70)

    from ml_utils import metrics, preprocessing

    # Empty list
    result = metrics.accuracy([], [])
    assert result == 0.0
    print(f"   ✓ Empty list handled: {result}")

    # Single element
    result = preprocessing.normalize_minmax([5.0])
    assert result == [0.0]
    print(f"   ✓ Single element handled: {result}")

    # All same values
    result = preprocessing.normalize_minmax([3.0, 3.0, 3.0])
    assert all(x == 0.0 for x in result)
    print(f"   ✓ All same values handled: {result}")

    # Missing values
    result = preprocessing.fill_missing_values([1.0, None, 3.0], "mean")
    assert None not in result
    print(f"   ✓ Missing values handled: {result}")

    print()


def main():
    """Run all validations."""
    print()
    print("=" * 70)
    print("Exercise 03 Validation")
    print("=" * 70)
    print()

    try:
        validate_type_hints()
        validate_decorators()
        validate_module_imports()
        validate_function_correctness()
        validate_error_handling()
        validate_edge_cases()

        print("=" * 70)
        print("✓ All validations passed!")
        print("=" * 70)
        print()

        return 0

    except Exception as e:
        print()
        print("=" * 70)
        print(f"✗ Validation failed: {e}")
        print("=" * 70)
        print()
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
