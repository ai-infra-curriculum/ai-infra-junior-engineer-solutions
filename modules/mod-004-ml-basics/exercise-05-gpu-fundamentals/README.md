# Exercise 05: GPU Fundamentals - Solution

Complete solution for learning GPU basics and comparing CPU vs GPU performance for ML workloads.

## Overview

This solution demonstrates:
- GPU detection and capability checking
- CPU vs GPU performance benchmarking
- GPU memory management
- Model inference comparison
- Device-agnostic code patterns

## Quick Start

### Prerequisites

- Python 3.8+
- pip package manager
- (Optional) NVIDIA GPU with CUDA support

### Installation

```bash
# Clone or navigate to this directory
cd exercise-05-gpu-fundamentals

# Make setup script executable and run it
chmod +x scripts/setup.sh
./scripts/setup.sh

# Or manually install dependencies
pip install -r requirements.txt
```

### Basic Usage

1. **Check GPU availability:**
```bash
python src/check_gpu.py
```

2. **Run performance benchmarks:**
```bash
python src/cpu_vs_gpu_benchmark.py
```

3. **Test memory management:**
```bash
python src/memory_management.py
```

4. **Compare model inference:**
```bash
python src/model_inference_comparison.py
```

5. **Run all benchmarks:**
```bash
chmod +x scripts/run_benchmarks.sh
./scripts/run_benchmarks.sh
```

### Run Tests

```bash
# Make test script executable
chmod +x scripts/test.sh
./scripts/test.sh

# Or run with pytest directly
pytest tests/ -v
```

## Features

### 1. GPU Detection (`check_gpu.py`)
- Detects CUDA availability
- Reports GPU specifications
- Shows memory information
- Tests device creation

### 2. Performance Benchmarking (`cpu_vs_gpu_benchmark.py`)
- Matrix multiplication benchmarks
- Multiple size comparisons
- Speedup calculations
- Warmup and synchronization handling

### 3. Memory Management (`memory_management.py`)
- GPU memory tracking
- Allocation and cleanup
- Cache management
- Out-of-memory handling

### 4. Model Inference (`model_inference_comparison.py`)
- Real model comparison (GPT-2)
- CPU vs GPU inference timing
- Memory usage tracking
- Throughput calculations

## Project Structure

```
exercise-05-gpu-fundamentals/
├── README.md                          # This file
├── STEP_BY_STEP.md                    # Detailed implementation guide
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── check_gpu.py                   # GPU detection and info
│   ├── cpu_vs_gpu_benchmark.py        # Performance comparison
│   ├── memory_management.py           # GPU memory management
│   └── model_inference_comparison.py  # Model inference benchmarking
├── tests/
│   ├── __init__.py                    # Test package initialization
│   ├── test_gpu_detection.py          # GPU detection tests
│   └── test_performance.py            # Performance tests
├── scripts/
│   ├── setup.sh                       # Environment setup
│   ├── run_benchmarks.sh              # Run all benchmarks
│   └── test.sh                        # Run tests
├── requirements.txt                   # Python dependencies
└── .gitignore                         # Git ignore file
```

## Expected Output

### With GPU Available

```
============================================================
GPU Detection and Information
============================================================

CUDA Available: True
Number of GPUs: 1

--- GPU 0 ---
Name: NVIDIA GeForce RTX 3060
Total Memory: 12.00 GB
Currently Allocated: 0.00 GB

Matrix Multiplication Benchmark:
Matrix size: 2000x2000
  CPU: 421.56 ms
  GPU: 5.34 ms
  Speedup: 78.9x faster
```

### CPU-Only Mode

```
============================================================
GPU Detection and Information
============================================================

CUDA Available: False

No CUDA-capable GPU detected.
This is fine for learning! We can still run everything on CPU.

Matrix Multiplication Benchmark:
Matrix size: 2000x2000
  CPU: 421.56 ms
  GPU: Not available
```

## Key Concepts Demonstrated

1. **Device Management**: Proper tensor and model placement
2. **Performance Measurement**: Accurate timing with synchronization
3. **Memory Tracking**: Monitoring GPU memory usage
4. **Graceful Fallback**: Works with or without GPU
5. **Best Practices**: Warmup, cleanup, error handling

## Common Issues and Solutions

### Issue: "CUDA not available"
- **Solution**: This is expected if you don't have an NVIDIA GPU. The code works in CPU-only mode.

### Issue: "Out of memory"
- **Solution**: Reduce batch size or matrix sizes in benchmarks. Clear GPU cache with `torch.cuda.empty_cache()`.

### Issue: "Slow first run"
- **Solution**: This is expected - first run includes model loading and compilation. Subsequent runs are faster.

## Performance Tips

1. **Always warmup GPU** before benchmarking
2. **Use `torch.cuda.synchronize()`** for accurate timing
3. **Batch operations** when possible for better GPU utilization
4. **Clean up memory** when done with large tensors
5. **Monitor memory usage** to avoid OOM errors

## Next Steps

1. Review `STEP_BY_STEP.md` for detailed implementation explanations
2. Experiment with different matrix sizes
3. Try different models for inference comparison
4. Implement the challenges in the learning exercise
5. Add your own benchmarking scenarios

## Resources

- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [GPU Performance Best Practices](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [NVIDIA Developer Resources](https://developer.nvidia.com/)

## License

Educational use only. Part of AI Infrastructure Junior Engineer training program.
