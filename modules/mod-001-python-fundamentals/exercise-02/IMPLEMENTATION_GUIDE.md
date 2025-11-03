# Implementation Guide - Exercise 02: Data Structures

Step-by-step guide for implementing this exercise from scratch.

## Prerequisites

- Python 3.11+ installed
- Completed Exercise 01 (Environment Setup)
- Text editor or IDE

## Time Estimate

90-120 minutes total

## Implementation Steps

### Part 1: Lists (30 minutes)

1. **Create list_operations.py**
   - Basic list operations (append, extend, insert, remove)
   - Slicing for batch creation
   - Searching and indexing
   - Run: `python scripts/list_operations.py`

2. **Create list_comprehensions.py**
   - Filter data with comprehensions
   - Transform and normalize data
   - Create batches with nested comprehensions
   - Parse model filenames
   - Run: `python scripts/list_comprehensions.py`

3. **Create batch_processor.py**
   - `DataBatchProcessor` class
   - `StratifiedBatchProcessor` class
   - Batch generation methods
   - Run: `python scripts/batch_processor.py`

### Part 2: Dictionaries (25 minutes)

4. **Create dict_operations.py**
   - Basic dictionary operations
   - Nested dictionaries
   - Safe access with `.get()`
   - Dictionary merging
   - Run: `python scripts/dict_operations.py`

5. **Create dict_comprehensions.py**
   - Create metrics dictionaries
   - Filter and transform values
   - Analyze experiment results
   - Group data by keys
   - Run: `python scripts/dict_comprehensions.py`

6. **Create feature_manager.py**
   - `FeatureManager` class
   - Add/update/query features
   - Import/export to JSON
   - Run: `python scripts/feature_manager.py`

### Part 3: Sets (15 minutes)

7. **Create set_operations.py**
   - Set intersections for overlap detection
   - Union, difference operations
   - Data leakage validation
   - Run: `python scripts/set_operations.py`

8. **Create deduplication.py**
   - Remove duplicate samples
   - Find duplicate files
   - Validate data consistency
   - Run: `python scripts/deduplication.py`

### Part 4: Tuples (10 minutes)

9. **Create tuple_operations.py**
   - Immutable data with tuples
   - Named tuples for readability
   - Multiple return values
   - Tuples as dict keys
   - Run: `python scripts/tuple_operations.py`

### Part 5: Comprehensive Manager (30 minutes)

10. **Create dataset_manager.py**
    - `MLDatasetManager` class
    - Add/remove samples
    - Split dataset (random and stratified)
    - Validate splits
    - Class distribution analysis
    - Batch retrieval
    - Run: `python scripts/dataset_manager.py`

### Testing (10 minutes)

11. **Create test files**
    - `tests/test_dataset_manager.py`
    - Other test files as needed
    - Run: `pytest tests/ -v`

## Quick Validation

```bash
# Run all scripts
for script in scripts/*.py; do
    echo "Running $script..."
    python "$script"
    echo ""
done

# Run tests
pytest tests/ -v
```

## Key Concepts Checklist

- [ ] List operations (append, extend, slice)
- [ ] List comprehensions with filtering
- [ ] Batch processing with stride
- [ ] Dictionary operations (get, update, merge)
- [ ] Dict comprehensions
- [ ] Nested data structures
- [ ] Set operations (intersection, union, difference)
- [ ] Deduplication with sets
- [ ] Tuple immutability
- [ ] Named tuples
- [ ] Combined data structures in classes

## Common Issues

**Issue:** Import errors when running tests
**Solution:** Ensure `__init__.py` exists in `scripts/` and `tests/`

**Issue:** Stratified split unbalanced
**Solution:** Ensure sufficient samples per class (at least 3-5)

**Issue:** Set operations confusing
**Solution:** Draw Venn diagrams to visualize intersections/unions

## Next Steps

- Proceed to Exercise 03 (Functions & Modules)
- Apply patterns to real ML projects
- Experiment with larger datasets

---

**Last Updated:** 2025-10-30
