# Step-by-Step Implementation Guide: File I/O & Error Handling

## Overview

Master file operations and robust error handling in Python for ML infrastructure. Learn to read/write files, handle exceptions, work with JSON/CSV/YAML, manage resources, and implement production-ready error handling patterns.

**Time**: 2-3 hours | **Difficulty**: Beginner to Intermediate

---

## Learning Objectives

✅ Perform file operations (read, write, append)
✅ Handle exceptions with try/except/finally
✅ Work with JSON, CSV, YAML formats
✅ Implement context managers
✅ Use logging instead of print statements
✅ Create custom exceptions
✅ Handle errors in production code

---

## Phase 1: File Operations

### Reading Files

```python
# Basic file reading
with open('data.txt', 'r') as f:
    content = f.read()
    print(content)

# Read lines
with open('data.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        print(line.strip())

# Read line by line (memory efficient)
with open('large_file.txt', 'r') as f:
    for line in f:
        process(line.strip())

# Read binary file
with open('model.pkl', 'rb') as f:
    data = f.read()
```

### Writing Files

```python
# Write (overwrite)
with open('output.txt', 'w') as f:
    f.write('Hello, World!\n')
    f.write('Another line\n')

# Append
with open('log.txt', 'a') as f:
    f.write(f'{timestamp}: Log entry\n')

# Write lines
lines = ['line1\n', 'line2\n', 'line3\n']
with open('output.txt', 'w') as f:
    f.writelines(lines)

# Write binary
with open('model.pkl', 'wb') as f:
    f.write(binary_data)
```

---

## Phase 2: Working with Data Formats

### JSON

```python
import json

# Read JSON
with open('config.json', 'r') as f:
    config = json.load(f)

print(config['model_path'])
print(config['hyperparameters']['learning_rate'])

# Write JSON
data = {
    'model': 'bert-base',
    'accuracy': 0.95,
    'metrics': {
        'precision': 0.94,
        'recall': 0.96
    }
}

with open('results.json', 'w') as f:
    json.dump(data, f, indent=2)

# Pretty print JSON
print(json.dumps(data, indent=2))

# Handle JSON errors
try:
    with open('data.json', 'r') as f:
        data = json.load(f)
except json.JSONDecodeError as e:
    print(f"Invalid JSON: {e}")
```

### CSV

```python
import csv

# Read CSV
with open('data.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)  # Skip header
    for row in reader:
        print(row)

# Read CSV with DictReader
with open('data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row['name'], row['age'])

# Write CSV
data = [
    ['name', 'age', 'city'],
    ['Alice', 30, 'NYC'],
    ['Bob', 25, 'SF']
]

with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

# Write CSV with DictWriter
rows = [
    {'name': 'Alice', 'age': 30},
    {'name': 'Bob', 'age': 25}
]

with open('output.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['name', 'age'])
    writer.writeheader()
    writer.writerows(rows)
```

### YAML

```python
import yaml

# Read YAML
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Write YAML
data = {
    'model': {
        'name': 'resnet50',
        'layers': 50,
        'pretrained': True
    },
    'training': {
        'epochs': 100,
        'batch_size': 32
    }
}

with open('config.yaml', 'w') as f:
    yaml.dump(data, f, default_flow_style=False)
```

---

## Phase 3: Error Handling

### Try/Except/Finally

```python
# Basic exception handling
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Multiple exceptions
try:
    with open('file.txt', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print("File not found")
except json.JSONDecodeError:
    print("Invalid JSON")
except Exception as e:
    print(f"Unexpected error: {e}")

# Finally block (always executes)
try:
    file = open('data.txt', 'r')
    data = file.read()
except FileNotFoundError:
    print("File not found")
finally:
    if 'file' in locals():
        file.close()

# Better: Use context manager
try:
    with open('data.txt', 'r') as f:
        data = f.read()
except FileNotFoundError:
    print("File not found")
```

### Exception Hierarchy

```python
# Catch specific exceptions first, general ones last
try:
    # Some operation
    pass
except FileNotFoundError:
    # Handle missing file
    pass
except PermissionError:
    # Handle permission issues
    pass
except OSError:
    # Handle other OS errors
    pass
except Exception as e:
    # Catch-all for unexpected errors
    print(f"Unexpected: {e}")
```

### Custom Exceptions

```python
class ModelNotFoundError(Exception):
    """Raised when model file is not found"""
    pass

class InvalidModelError(Exception):
    """Raised when model format is invalid"""
    pass

# Usage
def load_model(path):
    if not os.path.exists(path):
        raise ModelNotFoundError(f"Model not found: {path}")

    try:
        model = torch.load(path)
    except Exception:
        raise InvalidModelError(f"Cannot load model: {path}")

    return model

# Catching custom exceptions
try:
    model = load_model('model.pth')
except ModelNotFoundError as e:
    print(f"Error: {e}")
    # Use default model
    model = load_default_model()
except InvalidModelError as e:
    print(f"Error: {e}")
    sys.exit(1)
```

---

## Phase 4: Context Managers

### Using Context Managers

```python
# Automatic resource cleanup
with open('file.txt', 'r') as f:
    data = f.read()
# File automatically closed

# Multiple context managers
with open('input.txt', 'r') as fin, open('output.txt', 'w') as fout:
    for line in fin:
        fout.write(line.upper())

# Database connection example
from contextlib import contextmanager

@contextmanager
def database_connection(db_url):
    conn = connect(db_url)
    try:
        yield conn
    finally:
        conn.close()

# Usage
with database_connection('postgresql://...') as conn:
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM models')
```

### Creating Context Managers

```python
class Timer:
    """Context manager for timing code blocks"""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.elapsed = self.end - self.start
        print(f"Elapsed: {self.elapsed:.2f}s")
        return False  # Don't suppress exceptions

# Usage
with Timer():
    # Time this code block
    train_model()
```

---

## Phase 5: Logging

### Basic Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Log messages
logger.debug("Debug info")
logger.info("Training started")
logger.warning("Low memory")
logger.error("Failed to load model")
logger.critical("System crash")

# Log with variables
logger.info(f"Epoch {epoch}/{total_epochs}, Loss: {loss:.4f}")

# Log exceptions
try:
    result = risky_operation()
except Exception:
    logger.exception("Operation failed")  # Includes traceback
```

### Production Logging

```python
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'json': {
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
            'level': 'INFO'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'json',
            'level': 'DEBUG'
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console', 'file']
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)
```

---

## Best Practices

✅ Always use context managers (`with`) for files
✅ Handle specific exceptions before general ones
✅ Use logging instead of print statements
✅ Create custom exceptions for domain errors
✅ Never catch exceptions silently
✅ Clean up resources in finally blocks
✅ Log exceptions with traceback
✅ Validate file existence before operations
✅ Use pathlib for path operations
✅ Close files even if exceptions occur

---

## Common Patterns

### Safe File Loading

```python
import os
from pathlib import Path

def safe_load_json(filepath):
    """Safely load JSON with error handling"""
    path = Path(filepath)

    if not path.exists():
        logger.error(f"File not found: {filepath}")
        return None

    try:
        with path.open('r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filepath}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error loading {filepath}")
        return None
```

### Retry Logic

```python
import time
from functools import wraps

def retry(max_attempts=3, delay=1):
    """Retry decorator for functions that might fail"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=2)
def download_model(url):
    # May fail due to network issues
    return requests.get(url)
```

---

**File I/O and error handling mastered!** 📁
