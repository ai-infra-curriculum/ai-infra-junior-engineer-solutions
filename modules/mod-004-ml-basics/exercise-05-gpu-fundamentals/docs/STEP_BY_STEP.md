# Step-by-Step Implementation Guide: GPU Fundamentals

## Overview

Master GPU programming and optimization for ML workloads. Learn to detect GPUs, benchmark CPU vs GPU performance, manage GPU memory, compare inference strategies, and optimize model performance.

**Time**: 3-4 hours | **Difficulty**: Intermediate
**Files**: `check_gpu.py`, `cpu_vs_gpu_benchmark.py`, `memory_management.py`, `model_inference_comparison.py`

---

## Learning Objectives

✅ Detect and query GPU properties
✅ Benchmark CPU vs GPU performance
✅ Manage GPU memory efficiently
✅ Compare different inference strategies
✅ Optimize model execution on GPU
✅ Handle multi-GPU scenarios
✅ Debug GPU-related issues

---

## Quick Start

```bash
# Check GPU availability
python src/check_gpu.py

# Run CPU vs GPU benchmark
python src/cpu_vs_gpu_benchmark.py \
    --matrix-size 8192 \
    --iterations 100

# Memory management demo
python src/memory_management.py \
    --batch-sizes 16,32,64,128

# Inference comparison
python src/model_inference_comparison.py \
    --model-name "resnet50" \
    --batch-sizes 1,8,16,32

# Run all benchmarks
bash scripts/run_benchmarks.sh
```

---

## Implementation Guide

### Phase 1: GPU Detection

```python
import torch

def check_gpu():
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.version.cuda}")
        print(f"✓ GPU count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Multi-processors: {props.multi_processor_count}")
    else:
        print("✗ No CUDA GPUs available")
```

### Phase 2: CPU vs GPU Benchmarking

```python
import time
import torch

def benchmark_matmul(size=8192, device='cuda'):
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    c = torch.matmul(a, b)

    if device == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start
    gflops = (2 * size ** 3) / (elapsed * 1e9)

    return elapsed, gflops
```

### Phase 3: Memory Management

**Key concepts**:
- Allocate tensors on GPU explicitly
- Clear cache when needed
- Monitor memory usage
- Use gradient checkpointing for large models

```python
# Move to GPU
tensor = tensor.to('cuda')

# Clear cache
torch.cuda.empty_cache()

# Monitor memory
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### Phase 4: Inference Optimization

**Strategies**:
1. **Batch processing**: Process multiple inputs together
2. **Mixed precision**: Use FP16/BF16 for faster inference
3. **TorchScript**: Optimize model graph
4. **ONNX Runtime**: Cross-platform optimization
5. **TensorRT**: NVIDIA-specific optimization

```python
# Mixed precision
with torch.cuda.amp.autocast():
    output = model(input)

# TorchScript
scripted_model = torch.jit.script(model)

# ONNX export
torch.onnx.export(model, dummy_input, "model.onnx")
```

### Phase 5: Multi-GPU Usage

```python
# DataParallel (simple)
model = nn.DataParallel(model)

# DistributedDataParallel (recommended)
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model, device_ids=[local_rank])
```

---

## Benchmark Results

**Expected Speedups**:
- Matrix Multiplication: 50-100x
- Convolution: 30-60x
- Model Training: 10-50x (depending on model size)

---

## Memory Optimization Tips

1. **Reduce batch size**: Most common solution
2. **Gradient checkpointing**: Trade compute for memory
3. **Mixed precision training**: Half the memory usage
4. **Accumulate gradients**: Simulate larger batches
5. **Model parallelism**: Split model across GPUs

---

## Troubleshooting

**CUDA Out of Memory**:
```python
# Clear cache
torch.cuda.empty_cache()

# Reduce batch size
batch_size = batch_size // 2

# Use gradient accumulation
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Slow GPU Performance**:
- Check for CPU-GPU data transfers in loops
- Ensure data is on GPU before operations
- Use `torch.cuda.synchronize()` for accurate timing
- Profile with `torch.profiler`

---

## Profiling

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

## Best Practices

✅ Always check `torch.cuda.is_available()` before GPU operations
✅ Move data to GPU once, not repeatedly in loops
✅ Use `torch.cuda.synchronize()` for accurate timing
✅ Monitor GPU memory with `nvidia-smi`
✅ Clear cache between experiments
✅ Use mixed precision when possible
✅ Profile code to find bottlenecks

---

**GPU optimization mastered!** ⚡
