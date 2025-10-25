# Step-by-Step Implementation Guide: GPU Fundamentals

This guide explains the implementation details and GPU concepts used in this solution.

## Table of Contents

1. [Understanding GPU Architecture](#understanding-gpu-architecture)
2. [GPU Detection Implementation](#gpu-detection-implementation)
3. [Device Management Patterns](#device-management-patterns)
4. [Performance Benchmarking](#performance-benchmarking)
5. [Memory Management](#memory-management)
6. [Model Inference Optimization](#model-inference-optimization)
7. [Testing Strategies](#testing-strategies)

## Understanding GPU Architecture

### CPU vs GPU: Key Differences

**CPU (Central Processing Unit)**
- Few powerful cores (4-64 typically)
- High clock speed (3-5 GHz)
- Large cache per core
- Optimized for sequential operations
- Good for complex logic and branching

**GPU (Graphics Processing Unit)**
- Thousands of simpler cores (2000-10000+)
- Lower clock speed per core (1-2 GHz)
- Smaller cache, more compute units
- Optimized for parallel operations
- Excellent for matrix math

### Why ML Benefits from GPUs

Machine learning operations are fundamentally:

```python
# Neural network forward pass
output = input @ weights + bias  # Matrix multiplication
output = activation(output)       # Element-wise operation
```

Both operations are **highly parallelizable**:
- Matrix multiplication: multiply thousands of elements simultaneously
- Activation functions: apply to each element independently

**Example Speedup**:
- 1000x1000 matrix multiplication
  - CPU: Process ~1000 elements sequentially
  - GPU: Process thousands in parallel
  - Result: 20-100x speedup

## GPU Detection Implementation

### Step 1: Check CUDA Availability

```python
import torch

cuda_available = torch.cuda.is_available()
```

**What this checks**:
1. PyTorch was compiled with CUDA support
2. NVIDIA GPU drivers are installed
3. CUDA toolkit is available
4. Compatible GPU is present

### Step 2: Get GPU Information

```python
if cuda_available:
    # Number of GPUs
    gpu_count = torch.cuda.device_count()

    # GPU name and capabilities
    gpu_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)

    # Memory information
    total_memory = torch.cuda.get_device_properties(0).total_memory
```

**Key Properties**:
- **Compute Capability**: Version of GPU architecture (e.g., 8.6 for RTX 3060)
- **Total Memory**: Available VRAM for models and data
- **CUDA Version**: Compatibility level with PyTorch

### Step 3: Test Device Creation

```python
# Create tensors on specific devices
cpu_tensor = torch.randn(1000, 1000)  # Default: CPU
gpu_tensor = torch.randn(1000, 1000, device='cuda')  # Explicit: GPU

# Check device placement
print(cpu_tensor.device)  # cpu
print(gpu_tensor.device)  # cuda:0
```

## Device Management Patterns

### Pattern 1: Device-Agnostic Code

**Bad**: Hardcoded device
```python
model = MyModel()
model.cuda()  # Fails if no GPU!
```

**Good**: Conditional device selection
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel()
model.to(device)  # Works with or without GPU
```

### Pattern 2: Moving Data Efficiently

**Inefficient**: Multiple transfers
```python
# DON'T DO THIS
for batch in dataloader:
    x = batch['input'].to(device)
    y = batch['target'].to(device)
    # Process...
```

**Efficient**: Batch transfer
```python
# DO THIS
for batch in dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    # Process...
```

### Pattern 3: Keeping Model and Data on Same Device

```python
# Setup: Move model once
model = model.to(device)

# Inference: Move data to match model
inputs = inputs.to(device)
outputs = model(inputs)  # Both on same device

# Post-processing: Move results back if needed
outputs = outputs.cpu().numpy()  # Back to CPU for numpy
```

## Performance Benchmarking

### Critical Concept: GPU Asynchrony

GPU operations are **asynchronous** - Python continues while GPU works.

**Wrong timing**:
```python
start = time.time()
result = torch.matmul(a, b)  # GPU starts, Python continues
elapsed = time.time() - start  # Too fast! GPU still working
```

**Correct timing**:
```python
if device.type == 'cuda':
    torch.cuda.synchronize()  # Wait for GPU to finish previous ops

start = time.time()
result = torch.matmul(a, b)

if device.type == 'cuda':
    torch.cuda.synchronize()  # Wait for operation to complete

elapsed = time.time() - start  # Accurate!
```

### Warmup Importance

**First run is always slower** due to:
- CUDA kernel compilation (JIT)
- Memory allocation
- Cache warming

```python
# Warmup runs
for _ in range(10):
    _ = torch.matmul(a, b)

# Now benchmark
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    result = torch.matmul(a, b)
torch.cuda.synchronize()
elapsed = time.time() - start
```

### Understanding Speedup Results

**Small matrices (100x100)**:
- CPU: 0.12 ms
- GPU: 0.25 ms
- **GPU slower!** Why?

Overhead costs:
1. Transfer data to GPU: ~0.1 ms
2. Launch kernel: ~0.05 ms
3. Transfer result back: ~0.1 ms
4. Computation: ~0.01 ms

Total overhead >> computation time

**Large matrices (2000x2000)**:
- CPU: 421 ms
- GPU: 5.34 ms
- **GPU 79x faster!** Why?

Overhead costs same, but:
- Computation: 421 ms on CPU vs 5 ms on GPU
- Overhead becomes negligible
- Massive parallel advantage shows

**Rule of thumb**: GPU beneficial when matrix size > 500x500

## Memory Management

### GPU Memory Types

1. **Allocated Memory**: Actually used by tensors
2. **Reserved Memory**: Cached by PyTorch for efficiency
3. **Free Memory**: Available for allocation

```python
allocated = torch.cuda.memory_allocated()  # In use
reserved = torch.cuda.memory_reserved()    # Cached
total = torch.cuda.get_device_properties(0).total_memory
free = total - allocated
```

### Memory Lifecycle

```python
# 1. Allocate
tensor = torch.randn(1000, 1000, device='cuda')
# Memory: Allocated + Reserved both increase

# 2. Delete
del tensor
# Memory: Allocated decreases, Reserved stays (cached)

# 3. Clear cache
torch.cuda.empty_cache()
# Memory: Reserved decreases (returned to GPU)
```

### Handling Out of Memory (OOM)

**Prevention strategies**:

```python
try:
    # Attempt operation
    result = large_operation()
except RuntimeError as e:
    if "out of memory" in str(e):
        # Clear cache
        torch.cuda.empty_cache()

        # Try with smaller batch
        result = smaller_operation()
    else:
        raise
```

**Best practices**:
1. Monitor memory usage regularly
2. Use `torch.cuda.empty_cache()` after large operations
3. Delete unused tensors explicitly
4. Use smaller batch sizes if needed
5. Consider gradient checkpointing for training

## Model Inference Optimization

### Loading Models Efficiently

```python
# Load model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move to device ONCE at startup
model = model.to(device)

# Set to eval mode (disables dropout, etc.)
model.eval()
```

### Inference Best Practices

```python
# Disable gradient computation (saves memory and time)
with torch.no_grad():
    outputs = model(inputs)
```

**Why `no_grad()`?**
- Training: Need gradients for backpropagation
- Inference: Don't need gradients
- Savings: ~2x memory, ~20% faster

### Batching for Throughput

**Single inference** (low throughput):
```python
for text in texts:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    # GPU underutilized - only processing 1 sequence
```

**Batched inference** (high throughput):
```python
# Process multiple at once
inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
outputs = model.generate(**inputs)
# GPU fully utilized - processing N sequences in parallel
```

**Trade-offs**:
- Batch size 1: Low latency, low throughput
- Batch size 32: Higher latency, high throughput
- Choose based on requirements

## Testing Strategies

### Testing Without GPU

All code should work in CPU-only mode:

```python
@pytest.fixture
def device():
    """Provide appropriate device for testing"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model_loading(device):
    """Test works on both CPU and GPU"""
    model = MyModel().to(device)
    assert model is not None
```

### Testing Performance

```python
def test_gpu_faster_than_cpu():
    """Verify GPU speedup for large operations"""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")

    size = 2000

    # CPU timing
    cpu_time = benchmark_matmul(size, torch.device('cpu'))

    # GPU timing
    gpu_time = benchmark_matmul(size, torch.device('cuda'))

    # GPU should be faster for large matrices
    assert gpu_time < cpu_time
```

### Testing Memory Management

```python
def test_memory_cleanup():
    """Verify memory is freed after deletion"""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")

    # Initial memory
    initial = torch.cuda.memory_allocated()

    # Allocate
    tensor = torch.randn(1000, 1000, device='cuda')
    allocated = torch.cuda.memory_allocated()
    assert allocated > initial

    # Cleanup
    del tensor
    torch.cuda.empty_cache()

    # Verify cleanup
    final = torch.cuda.memory_allocated()
    assert final == initial
```

## Implementation Checklist

- [ ] GPU detection works correctly
- [ ] Code runs in both CPU and GPU modes
- [ ] Proper synchronization for timing
- [ ] Warmup before benchmarking
- [ ] Memory tracking and cleanup
- [ ] Error handling for OOM
- [ ] Type hints for all functions
- [ ] Docstrings for all modules
- [ ] Tests for critical functionality
- [ ] Logging for debugging
- [ ] Clear output formatting

## Common Pitfalls

### 1. Missing Synchronization
```python
# WRONG
start = time.time()
result = gpu_operation()
elapsed = time.time() - start  # Inaccurate!

# RIGHT
torch.cuda.synchronize()
start = time.time()
result = gpu_operation()
torch.cuda.synchronize()
elapsed = time.time() - start  # Accurate
```

### 2. Device Mismatch
```python
# WRONG
model = model.to('cuda')
inputs = torch.randn(10, 10)  # Still on CPU!
outputs = model(inputs)  # ERROR: device mismatch

# RIGHT
model = model.to('cuda')
inputs = torch.randn(10, 10).to('cuda')
outputs = model(inputs)  # Works!
```

### 3. No Warmup
```python
# WRONG
start = time.time()
result = gpu_operation()  # First run - includes compilation
elapsed = time.time() - start  # Misleading!

# RIGHT
for _ in range(10):
    _ = gpu_operation()  # Warmup
torch.cuda.synchronize()
start = time.time()
result = gpu_operation()  # Now benchmark
torch.cuda.synchronize()
elapsed = time.time() - start  # Accurate
```

### 4. Memory Leaks
```python
# WRONG
for i in range(1000):
    tensor = torch.randn(1000, 1000, device='cuda')
    # Never deleted - memory leak!

# RIGHT
for i in range(1000):
    tensor = torch.randn(1000, 1000, device='cuda')
    # Use tensor...
    del tensor
    if i % 100 == 0:
        torch.cuda.empty_cache()
```

## Performance Optimization Tips

1. **Use larger batches** when possible
2. **Enable cuDNN benchmarking** for CNNs: `torch.backends.cudnn.benchmark = True`
3. **Use mixed precision** (FP16): Faster and less memory
4. **Profile your code**: Use PyTorch profiler to find bottlenecks
5. **Minimize CPU-GPU transfers**: Keep data on GPU when possible
6. **Use pinned memory** for faster transfers: `tensor.pin_memory()`
7. **Compile models** with `torch.compile()` (PyTorch 2.0+)

## Debugging Tips

1. **Check device placement**: Print `tensor.device` regularly
2. **Monitor memory**: Use `torch.cuda.memory_summary()`
3. **Enable anomaly detection**: `torch.autograd.set_detect_anomaly(True)`
4. **Use CUDA_LAUNCH_BLOCKING**: Set env var for synchronous execution during debugging
5. **Check CUDA errors**: Errors are async, might appear later - sync to catch immediately

## Next Steps

1. Implement mixed precision inference
2. Add batch processing support
3. Try different models and sizes
4. Profile and optimize bottlenecks
5. Implement multi-GPU support
6. Add monitoring and alerting
7. Optimize for production deployment

## Resources

- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [GPU Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
