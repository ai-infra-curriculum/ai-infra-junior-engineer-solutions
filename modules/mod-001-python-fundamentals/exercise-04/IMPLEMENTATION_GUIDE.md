# Implementation Guide - Exercise 04: File I/O

Step-by-step guide for implementing file I/O operations for ML workflows.

## Prerequisites

- Python 3.11+ installed
- PyYAML installed: `pip install pyyaml`
- Completed Exercises 01-03
- Understanding of Python data structures

## Time Estimate

90-120 minutes total

## Implementation Steps

### Part 1: CSV Operations (30 minutes)

**Step 1: Create csv_operations.py**
- Implement `read_csv_basic()` - Read CSV into list of rows
- Implement `read_csv_with_headers()` - Separate headers and data
- Implement `read_csv_as_dicts()` - Read as dictionaries (most useful)
- Implement `read_csv_filtered()` - Read with filtering condition
- Implement `create_sample_dataset()` - Generate sample data
- Run: `python scripts/csv_operations.py`

**Key concepts:**
- Context managers: `with open(file, 'r') as f:`
- Encoding: Always use `encoding='utf-8'`
- csv.reader vs csv.DictReader
- Newline handling: `newline=''` for writing

**Step 2: Create csv_writer.py**
- Implement `write_csv_from_lists()` - Write from list of lists
- Implement `write_csv_from_dicts()` - Write from dictionaries
- Implement `append_to_csv()` - Append single row
- Implement `write_predictions()` - Write model predictions
- Run: `python scripts/csv_writer.py`

### Part 2: JSON Operations (20 minutes)

**Step 3: Create json_operations.py**
- Implement `save_model_metadata()` - Save model info as JSON
- Implement `load_model_metadata()` - Load model info
- Implement `save_training_config()` - Save config with formatting
- Implement `load_training_config()` - Load with error handling
- Implement `update_experiment_log()` - Append to experiment log
- Implement `save_metrics_history()` - Save training metrics
- Run: `python scripts/json_operations.py`

**Key concepts:**
- `json.dump()` vs `json.dumps()` (file vs string)
- `indent=2` for readability
- `sort_keys=True` for consistent diffs
- Error handling: FileNotFoundError, JSONDecodeError

### Part 3: YAML Configuration (15 minutes)

**Step 4: Create yaml_operations.py**
- Implement `save_yaml_config()` - Save to YAML
- Implement `load_yaml_config()` - Load from YAML
- Implement `merge_configs()` - Merge defaults with user config
- Create sample ML pipeline configuration
- Run: `python scripts/yaml_operations.py`

**Key concepts:**
- `yaml.safe_load()` for security (vs `yaml.load()`)
- `default_flow_style=False` for readable output
- `sort_keys=False` to preserve order
- YAML advantages: comments, cleaner syntax

### Part 4: Pickle Serialization (15 minutes)

**Step 5: Create pickle_operations.py**
- Implement `save_object()` - Save Python object
- Implement `load_object()` - Load Python object
- Create `ModelCheckpoint` class
- Demonstrate saving/loading complex objects
- Run: `python scripts/pickle_operations.py`

**Key concepts:**
- Binary mode: `'wb'` for writing, `'rb'` for reading
- Security warning: Never unpickle untrusted data
- When to use: model checkpoints, preprocessing pipelines
- When not to use: configuration, cross-platform data

### Part 5: Large File Processing (20 minutes)

**Step 6: Create large_file_processing.py**
- Implement `read_csv_chunks()` - Generator for chunked reading
- Implement `process_large_dataset()` - Process in chunks
- Implement `read_file_lines()` - Line-by-line reading
- Demonstrate memory-efficient processing
- Run: `python scripts/large_file_processing.py`

**Key concepts:**
- Generators with `yield` for lazy evaluation
- Constant memory usage
- Processing files larger than RAM
- Iterator pattern

### Part 6: Unified File Manager (30 minutes)

**Step 7: Create file_manager.py**
- Implement `MLFileManager` class
- `save()` method with auto-format detection
- `load()` method with format detection
- Private methods: `_save_json()`, `_load_json()`, etc.
- `list_files()` for file discovery
- `file_exists()`, `delete_file()`, `get_file_size()`
- Run: `python scripts/file_manager.py`

**Key concepts:**
- Unified API for multiple formats
- pathlib.Path for cross-platform paths
- Auto-format detection from extension
- Consistent error handling

### Part 7: Testing (15 minutes)

**Step 8: Create pytest tests**
- Create `tests/test_file_manager.py`
  - Test saving/loading each format
  - Test auto-format detection
  - Test error handling
  - Test file operations (exists, delete, size)
- Create `tests/test_csv_operations.py`
- Create `tests/test_json_operations.py`

**Run tests:**
```bash
pytest tests/ -v
pytest tests/ --cov=scripts --cov-report=term-missing
```

### Part 8: Validation (10 minutes)

**Step 9: Create validate_file_io.py**
- Test CSV operations
- Test JSON operations
- Test file existence checking
- Test error handling
- Cleanup temporary files
- Run: `python scripts/validate_file_io.py`

## Quick Validation

```bash
# Create all sample data
python scripts/csv_operations.py
python scripts/csv_writer.py
python scripts/json_operations.py
python scripts/yaml_operations.py
python scripts/pickle_operations.py
python scripts/large_file_processing.py

# Test unified manager
python scripts/file_manager.py

# Validate
python scripts/validate_file_io.py

# Run tests
pytest tests/ -v
```

## Key Concepts Checklist

- [ ] Context managers with `with` statement
- [ ] Encoding specification (`utf-8`)
- [ ] CSV reading methods (reader, DictReader)
- [ ] CSV writing (writer, DictWriter)
- [ ] JSON serialization with formatting
- [ ] YAML for human-readable configs
- [ ] Pickle for Python objects (with security awareness)
- [ ] Generator pattern for large files
- [ ] pathlib.Path for cross-platform paths
- [ ] Error handling (FileNotFoundError, JSONDecodeError)
- [ ] Auto-format detection
- [ ] Memory-efficient processing

## Common Issues

**Issue:** Extra blank lines in CSV on Windows
**Solution:** Use `newline=''` when opening files:
```python
with open('file.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
```

**Issue:** UnicodeDecodeError
**Solution:** Always specify encoding:
```python
with open('file.txt', 'r', encoding='utf-8') as f:
    data = f.read()
```

**Issue:** JSON decode error
**Solution:** Add error handling:
```python
try:
    data = json.load(f)
except json.JSONDecodeError as e:
    print(f"Invalid JSON at line {e.lineno}: {e.msg}")
```

**Issue:** Memory error with large CSV
**Solution:** Use chunked reading:
```python
for chunk in read_csv_chunks(filepath, chunk_size=1000):
    process(chunk)
```

**Issue:** Path separator issues across OSes
**Solution:** Use pathlib:
```python
from pathlib import Path
filepath = Path('data') / 'train.csv'  # Works on all platforms
```

## Best Practices

1. **Always use context managers:**
   ```python
   with open(file, 'r') as f:
       data = f.read()
   ```

2. **Always specify encoding:**
   ```python
   open(file, 'r', encoding='utf-8')
   ```

3. **Use pathlib for paths:**
   ```python
   from pathlib import Path
   filepath = Path('data') / 'file.csv'
   ```

4. **Handle errors gracefully:**
   ```python
   try:
       with open(file, 'r') as f:
           data = f.read()
   except FileNotFoundError:
       # Handle missing file
   ```

5. **Use appropriate format:**
   - CSV: Tabular data, datasets
   - JSON: Structured data, metadata
   - YAML: Configuration files
   - Pickle: Python objects (trusted sources only)

## Format Decision Matrix

| Use Case | Best Format | Why |
|----------|-------------|-----|
| Dataset | CSV | Tabular, widely compatible |
| Model metadata | JSON | Structured, version-control friendly |
| Configuration | YAML | Human-readable, supports comments |
| Model checkpoint | Pickle | Fast, preserves Python objects |
| Experiment log | JSON/CSV | Easy to parse, append-friendly |
| Feature definitions | JSON/YAML | Structured, readable |

## Next Steps

After completing this exercise:

1. **Exercise 05: Error Handling** - Robust error management
2. **Exercise 06: Async Programming** - Concurrent I/O
3. **Apply to projects:**
   - Build data pipeline with file I/O
   - Create configuration management system
   - Implement experiment tracking

## Resources

- [CSV Module](https://docs.python.org/3/library/csv.html)
- [JSON Module](https://docs.python.org/3/library/json.html)
- [PyYAML](https://pyyaml.org/)
- [Pickle Security](https://docs.python.org/3/library/pickle.html#module-pickle)
- [Pathlib](https://docs.python.org/3/library/pathlib.html)

---

**Last Updated:** 2025-10-30
