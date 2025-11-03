# Exercise 02: Reflection Question Answers

Comprehensive answers to all reflection questions.

## Question 1: When should you use a list vs. a tuple for ML data?

### Use Lists When:
- Data needs to be modified (adding/removing samples during preprocessing)
- Building datasets incrementally
- Batch processing where order matters but contents change
- Collecting results during training epochs

### Use Tuples When:
- Data is immutable (model metadata, configuration)
- Need to use as dictionary keys (e.g., (model, dataset) pairs)
- Returning multiple values from functions
- Fixed training history records
- Memory efficiency is critical

### Example:
```python
# List - mutable training data
training_samples = ["sample1.jpg", "sample2.jpg"]
training_samples.append("sample3.jpg")  # ✓ Can modify

# Tuple - immutable model metadata
model_info = ("ResNet50", "v1.0", 0.92, "2024-10-30")
# model_info[0] = "VGG16"  # ✗ TypeError: immutable
```

---

## Question 2: Why are dictionaries useful for storing configuration?

### Key Benefits:

1. **Named Parameters:** Clear key-value mapping
2. **Flexible Schema:** Easy to add/remove options
3. **Default Values:** Use `.get(key, default)` for safe access
4. **Nested Structures:** Hierarchical configs
5. **JSON Compatible:** Easy serialization
6. **Fast Lookups:** O(1) average case

### Example:
```python
config = {
    "model": {
        "architecture": "resnet50",
        "layers": 50
    },
    "training": {
        "batch_size": 32,
        "learning_rate": 0.001
    }
}

# Safe access with defaults
dropout = config.get("dropout", 0.5)
lr = config["training"].get("learning_rate", 0.01)
```

---

## Question 3: How can sets help prevent data leakage?

### Data Leakage Prevention:

Sets provide:
- **Fast membership testing:** O(1) to check if sample is in set
- **Overlap detection:** Use intersection (`&`) to find duplicates
- **Validation:** Ensure disjoint splits with `.isdisjoint()`
- **Deduplication:** Automatically removes duplicates

### Example:
```python
train_ids = {1, 2, 3, 4, 5}
test_ids = {6, 7, 8, 9, 10}

# Check for data leakage
overlap = train_ids & test_ids
if overlap:
    raise ValueError(f"Data leakage: {overlap}")

# Verify disjoint
assert train_ids.isdisjoint(test_ids), "Splits must not overlap"

# Union to verify all samples accounted for
all_ids = train_ids | val_ids | test_ids
assert len(all_ids) == total_samples
```

---

## Question 4: Performance implications of comprehensions vs. loops?

### List Comprehensions:
**Pros:**
- 20-30% faster (optimized in CPython)
- More readable for simple operations
- Single expression

**Cons:**
- Less flexible (single expression only)
- Harder to debug
- Creates entire list in memory

### For Loops:
**Pros:**
- More flexible (multiple statements)
- Better for complex logic
- Can modify existing structures
- Easier to debug

**Cons:**
- Slightly slower
- More verbose

### Benchmark:
```python
import timeit

# Comprehension: ~0.05s
t1 = timeit.timeit('[x**2 for x in range(1000)]', number=10000)

# Loop: ~0.08s (60% slower)
def loop():
    result = []
    for x in range(1000):
        result.append(x**2)
    return result

t2 = timeit.timeit('loop()', number=10000, globals={'loop': loop})
```

**Guideline:** Use comprehensions for simple transformations, loops for complex logic.

---

## Question 5: How to handle millions of samples efficiently?

### Problem:
Lists of millions of items consume excessive memory.

### Solutions:

**1. Generators (Lazy Evaluation):**
```python
def batch_generator(data_path, batch_size):
    """Load data on-demand."""
    for file in os.listdir(data_path):
        batch = load_batch(file, batch_size)
        yield batch

# Memory efficient - generates on-the-fly
for batch in batch_generator("/data", 32):
    train(batch)
```

**2. Store IDs Only:**
```python
# Store IDs (cheap), load data when needed
sample_ids = list(range(1_000_000))  # Just integers

def load_sample(sample_id):
    return np.load(f"data/{sample_id}.npy")
```

**3. Memory-Mapped Files:**
```python
import numpy as np
data = np.memmap('huge_dataset.dat', dtype='float32',
                 mode='r', shape=(1_000_000, 256))
```

**4. Chunked Processing:**
```python
chunk_size = 10_000
for i in range(0, total_samples, chunk_size):
    chunk = load_chunk(i, chunk_size)
    process(chunk)
```

**5. Database/HDF5:**
```python
import h5py
with h5py.File('dataset.h5', 'r') as f:
    # Access data without loading all into memory
    batch = f['images'][0:32]
```

---

## Question 6: Named tuple vs. dictionary?

### Use Named Tuples When:
- Fixed schema (known fields at design time)
- Immutable data
- Memory efficiency critical (3-4x smaller)
- Need attribute access (dot notation)
- Type safety desired

### Use Dictionaries When:
- Dynamic schema (variable fields)
- Need to modify data
- Unknown fields at creation
- JSON-like data
- Flexible structure needed

### Comparison:
```python
from collections import namedtuple
import sys

# Named Tuple
ModelConfig = namedtuple('ModelConfig', ['name', 'layers', 'params'])
nt = ModelConfig('ResNet50', 50, 25_000_000)

# Dictionary
dt = {'name': 'ResNet50', 'layers': 50, 'params': 25_000_000}

# Memory usage
sys.getsizeof(nt)  # ~64 bytes
sys.getsizeof(dt)  # ~232 bytes (3.6x larger!)

# Access
nt.name          # ✓ Attribute access
dt['name']       # ✓ Key access

# Mutability
# nt.layers = 100  # ✗ TypeError
dt['layers'] = 100  # ✓ Can modify
```

---

## Question 7: Ensuring data structure choices don't impact performance?

### Best Practices:

**1. Profile to Find Bottlenecks:**
```python
import cProfile

profiler = cProfile.Profile()
profiler.enable()
train_model()
profiler.disable()

profiler.print_stats(sort='cumulative')
```

**2. Preprocess Once:**
```python
# ✗ BAD: Convert every batch
for epoch in range(epochs):
    for data in dataset:
        batch = list(data)  # Unnecessary!
        train(batch)

# ✓ GOOD: Preprocess once
preprocessed = [preprocess(x) for x in dataset]
for epoch in range(epochs):
    for batch in get_batches(preprocessed):
        train(batch)
```

**3. Use Generators for Large Data:**
```python
def data_generator():
    while True:
        for sample in dataset:
            yield preprocess(sample)

for batch in batch_generator(data_generator(), 32):
    train(batch)
```

**4. Choose Right Structure:**
- Lists: Sequential access, ordered data
- Dicts/Sets: Lookups, membership tests
- Tuples: Immutable, memory-efficient
- Arrays/NumPy: Numerical operations

**5. Benchmark Critical Paths:**
```python
import timeit

# Test different approaches
t1 = timeit.timeit('approach1()', number=1000, globals=globals())
t2 = timeit.timeit('approach2()', number=1000, globals=globals())

if t1 < t2:
    print("Approach 1 is faster")
```

**6. Monitor Memory:**
```python
import tracemalloc

tracemalloc.start()
# Your code here
current, peak = tracemalloc.get_traced_memory()
print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
tracemalloc.stop()
```

---

## Summary

Key takeaways:
1. **Lists:** Mutable, ordered, use for changing data
2. **Tuples:** Immutable, memory-efficient, use for fixed data
3. **Dictionaries:** Key-value pairs, flexible configs
4. **Sets:** Fast membership, deduplication, no-overlap validation
5. **Performance:** Profile first, optimize bottlenecks
6. **Large data:** Use generators, memory-mapping, chunking

**Golden Rule:** Choose the simplest structure that meets your needs, then optimize if profiling shows it's a bottleneck.

---

**Last Updated:** 2025-10-30
