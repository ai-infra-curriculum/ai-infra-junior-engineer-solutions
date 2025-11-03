#!/usr/bin/env python3
"""
Exception Basics for ML Workflows

Demonstrates common exception types and when they occur in ML applications.
"""


def demonstrate_exceptions():
    """Demonstrate common exception types in ML workflows"""
    print("=" * 70)
    print("Common Exception Types in ML Workflows")
    print("=" * 70)
    print()

    # ValueError: Invalid value for operation
    print("1. ValueError - Invalid Value")
    print("-" * 70)
    try:
        batch_size = int("invalid")
    except ValueError as e:
        print(f"✗ ValueError: {e}")
        print(f"   Context: Attempting to convert invalid string to integer")
    print()

    # TypeError: Operation with incompatible types
    print("2. TypeError - Type Mismatch")
    print("-" * 70)
    try:
        result = "text" + 123
    except TypeError as e:
        print(f"✗ TypeError: {e}")
        print(f"   Context: Cannot concatenate string and integer")
    print()

    # KeyError: Missing dictionary key
    print("3. KeyError - Missing Configuration")
    print("-" * 70)
    try:
        config = {"learning_rate": 0.001}
        batch_size = config["batch_size"]
    except KeyError as e:
        print(f"✗ KeyError: Missing key {e}")
        print(f"   Context: Configuration parameter not found")
        print(f"   Available keys: {list(config.keys())}")
    print()

    # IndexError: Invalid list/array index
    print("4. IndexError - Out of Bounds")
    print("-" * 70)
    try:
        data = [1, 2, 3]
        value = data[10]
    except IndexError as e:
        print(f"✗ IndexError: {e}")
        print(f"   Context: Index 10 out of range for list of length {len(data)}")
    print()

    # FileNotFoundError: Missing file
    print("5. FileNotFoundError - Missing Model")
    print("-" * 70)
    try:
        with open("nonexistent_model.h5", 'r') as f:
            content = f.read()
    except FileNotFoundError as e:
        print(f"✗ FileNotFoundError: {e}")
        print(f"   Context: Model file not found at specified path")
    print()

    # ZeroDivisionError: Division by zero
    print("6. ZeroDivisionError - Invalid Calculation")
    print("-" * 70)
    try:
        correct_predictions = 10
        total_predictions = 0
        accuracy = correct_predictions / total_predictions
    except ZeroDivisionError as e:
        print(f"✗ ZeroDivisionError: {e}")
        print(f"   Context: Cannot divide by zero when calculating accuracy")
    print()

    # AttributeError: Missing attribute/method
    print("7. AttributeError - Missing Method")
    print("-" * 70)
    try:
        model = None
        predictions = model.predict([1, 2, 3])
    except AttributeError as e:
        print(f"✗ AttributeError: {e}")
        print(f"   Context: Model is None, cannot call predict method")
    print()

    # ImportError: Missing module
    print("8. ImportError - Missing Dependency")
    print("-" * 70)
    try:
        import nonexistent_library
    except ImportError as e:
        print(f"✗ ImportError: {e}")
        print(f"   Context: Required ML library not installed")
    print()

    # MemoryError: Out of memory
    print("9. MemoryError Example (not triggered)")
    print("-" * 70)
    print("   MemoryError occurs when trying to allocate memory that exceeds available RAM")
    print("   Common in ML: Large batch sizes, huge models, insufficient GPU memory")
    print("   Example: batch_size = 10000, model_size = 1GB → GPU OOM")
    print()

    # RuntimeError: Generic runtime error
    print("10. RuntimeError - General Execution Error")
    print("-" * 70)
    try:
        raise RuntimeError("GPU not available for training")
    except RuntimeError as e:
        print(f"✗ RuntimeError: {e}")
        print(f"   Context: Runtime condition prevents execution")
    print()


def ml_error_scenarios():
    """Demonstrate ML-specific error scenarios"""
    print("=" * 70)
    print("ML-Specific Error Scenarios")
    print("=" * 70)
    print()

    scenarios = [
        {
            "name": "Dataset Loading Failure",
            "exception": "FileNotFoundError",
            "cause": "Data file moved or deleted",
            "impact": "Cannot start training",
            "solution": "Verify file path, check permissions"
        },
        {
            "name": "GPU Out of Memory",
            "exception": "RuntimeError/CUDA OOM",
            "cause": "Batch size too large for GPU",
            "impact": "Training crashes",
            "solution": "Reduce batch_size, use gradient accumulation"
        },
        {
            "name": "Invalid Configuration",
            "exception": "ValueError/KeyError",
            "cause": "Missing or invalid config parameter",
            "impact": "Training fails to start",
            "solution": "Validate config, provide defaults"
        },
        {
            "name": "Data Corruption",
            "exception": "ValueError/RuntimeError",
            "cause": "Corrupted data samples",
            "impact": "Training unstable or fails",
            "solution": "Validate data, skip corrupted samples"
        },
        {
            "name": "Model Checkpoint Failure",
            "exception": "IOError/OSError",
            "cause": "Disk full, permission denied",
            "impact": "Cannot save model",
            "solution": "Check disk space, verify permissions"
        },
        {
            "name": "Network Request Timeout",
            "exception": "TimeoutError/ConnectionError",
            "cause": "API unreachable, slow connection",
            "impact": "Cannot download model/data",
            "solution": "Retry with backoff, use local cache"
        },
        {
            "name": "Incompatible Data Shapes",
            "exception": "ValueError/RuntimeError",
            "cause": "Input shape mismatch",
            "impact": "Forward pass fails",
            "solution": "Validate input dimensions, reshape data"
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['name']}")
        print(f"   Exception: {scenario['exception']}")
        print(f"   Cause: {scenario['cause']}")
        print(f"   Impact: {scenario['impact']}")
        print(f"   Solution: {scenario['solution']}")
        print()


def exception_hierarchy():
    """Show Python exception hierarchy relevant to ML"""
    print("=" * 70)
    print("Python Exception Hierarchy (ML-Relevant Subset)")
    print("=" * 70)
    print()
    print("BaseException")
    print("└── Exception")
    print("    ├── ArithmeticError")
    print("    │   ├── ZeroDivisionError")
    print("    │   └── OverflowError")
    print("    ├── LookupError")
    print("    │   ├── IndexError")
    print("    │   └── KeyError")
    print("    ├── ValueError")
    print("    ├── TypeError")
    print("    ├── AttributeError")
    print("    ├── ImportError")
    print("    │   └── ModuleNotFoundError")
    print("    ├── OSError")
    print("    │   ├── FileNotFoundError")
    print("    │   ├── PermissionError")
    print("    │   └── IOError")
    print("    ├── RuntimeError")
    print("    │   └── RecursionError")
    print("    ├── MemoryError")
    print("    └── AssertionError")
    print()
    print("Best Practice: Catch specific exceptions, not generic Exception")
    print()


def main():
    """Run all demonstrations"""
    demonstrate_exceptions()
    print()
    ml_error_scenarios()
    print()
    exception_hierarchy()

    print("=" * 70)
    print("✓ Exception basics demonstration complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
