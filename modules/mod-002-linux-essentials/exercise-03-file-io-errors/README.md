# Exercise 03: File I/O & Error Handling

## Overview

Master file operations, configuration management, logging, and exception handling for production-ready Python applications. Learn to write defensive code that handles errors gracefully.

## Learning Objectives

- ✅ Read and write files (text, binary, CSV, JSON, YAML)
- ✅ Use context managers for resource management
- ✅ Handle exceptions with try/except/finally
- ✅ Create custom exceptions
- ✅ Implement logging best practices
- ✅ Work with configuration files
- ✅ Build robust error handling strategies

## Topics Covered

### 1. File Operations

#### Reading Files

```python
# Read entire file
with open('file.txt', 'r') as f:
    content = f.read()

# Read line by line
with open('file.txt', 'r') as f:
    for line in f:
        print(line.strip())

# Read all lines into list
with open('file.txt', 'r') as f:
    lines = f.readlines()

# Read with specific encoding
with open('file.txt', 'r', encoding='utf-8') as f:
    content = f.read()
```

#### Writing Files

```python
# Write (overwrite)
with open('file.txt', 'w') as f:
    f.write('Hello, World!\n')

# Append
with open('file.txt', 'a') as f:
    f.write('Additional line\n')

# Write multiple lines
lines = ['Line 1\n', 'Line 2\n', 'Line 3\n']
with open('file.txt', 'w') as f:
    f.writelines(lines)
```

#### Binary Files

```python
# Read binary file
with open('image.png', 'rb') as f:
    data = f.read()

# Write binary file
with open('output.bin', 'wb') as f:
    f.write(binary_data)
```

### 2. Working with Different Formats

#### JSON

```python
import json

# Read JSON
with open('config.json', 'r') as f:
    config = json.load(f)

# Write JSON
data = {'name': 'Model', 'version': '1.0.0'}
with open('output.json', 'w') as f:
    json.dump(data, f, indent=2)

# Parse JSON string
json_string = '{"key": "value"}'
data = json.loads(json_string)

# Convert to JSON string
json_string = json.dumps(data, indent=2)
```

#### YAML

```python
import yaml

# Read YAML
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Write YAML
data = {'database': {'host': 'localhost', 'port': 5432}}
with open('output.yaml', 'w') as f:
    yaml.dump(data, f, default_flow_style=False)

# Multiple YAML documents
with open('multi.yaml', 'r') as f:
    for doc in yaml.safe_load_all(f):
        print(doc)
```

#### CSV

```python
import csv

# Read CSV
with open('data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

# Read CSV as dictionaries
with open('data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row['column_name'])

# Write CSV
data = [
    ['Name', 'Age', 'City'],
    ['Alice', '30', 'NYC'],
    ['Bob', '25', 'LA']
]
with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

# Write CSV from dictionaries
fieldnames = ['name', 'age', 'city']
data = [
    {'name': 'Alice', 'age': 30, 'city': 'NYC'},
    {'name': 'Bob', 'age': 25, 'city': 'LA'}
]
with open('output.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)
```

### 3. Context Managers

#### Built-in Context Managers

```python
# File handling
with open('file.txt', 'r') as f:
    content = f.read()
# File automatically closed

# Multiple context managers
with open('input.txt', 'r') as input_file, \
     open('output.txt', 'w') as output_file:
    content = input_file.read()
    output_file.write(content.upper())
```

#### Custom Context Managers

```python
from contextlib import contextmanager
import time

@contextmanager
def timer(name):
    """Context manager for timing operations."""
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        print(f"{name} took {duration:.4f}s")

# Usage
with timer("Data Processing"):
    # Your code here
    time.sleep(1)
```

#### Context Manager Class

```python
class DatabaseConnection:
    """Context manager for database connections."""

    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = None

    def __enter__(self):
        """Open connection."""
        print(f"Connecting to {self.connection_string}")
        self.connection = "connection_object"  # Simulate connection
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close connection."""
        print("Closing connection")
        if exc_type is not None:
            print(f"Error occurred: {exc_val}")
        # Return False to propagate exception
        return False

# Usage
with DatabaseConnection("postgresql://localhost/db") as conn:
    # Use connection
    print("Working with connection")
```

### 4. Exception Handling

#### Basic Try/Except

```python
# Basic exception handling
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")

# Multiple exception types
try:
    value = int(input("Enter a number: "))
    result = 10 / value
except ValueError:
    print("Invalid number")
except ZeroDivisionError:
    print("Cannot divide by zero")

# Catch multiple exceptions
try:
    # risky operation
    pass
except (ValueError, TypeError) as e:
    print(f"Error: {e}")

# Catch all exceptions
try:
    # risky operation
    pass
except Exception as e:
    print(f"Unexpected error: {e}")
```

#### Try/Except/Else/Finally

```python
try:
    file = open('data.txt', 'r')
    data = file.read()
except FileNotFoundError:
    print("File not found")
except PermissionError:
    print("Permission denied")
else:
    # Executed if no exception
    print("File read successfully")
    print(f"Read {len(data)} characters")
finally:
    # Always executed
    if 'file' in locals():
        file.close()
    print("Cleanup complete")
```

#### Re-raising Exceptions

```python
def process_file(filename):
    """Process file with error handling."""
    try:
        with open(filename, 'r') as f:
            data = f.read()
        return process_data(data)
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        raise  # Re-raise the exception
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise ValueError(f"Failed to process {filename}") from e
```

### 5. Custom Exceptions

#### Basic Custom Exception

```python
class ValidationError(Exception):
    """Raised when validation fails."""
    pass

def validate_age(age):
    """Validate age."""
    if age < 0:
        raise ValidationError("Age cannot be negative")
    if age > 150:
        raise ValidationError("Age is unrealistic")
    return True

# Usage
try:
    validate_age(-5)
except ValidationError as e:
    print(f"Validation failed: {e}")
```

#### Custom Exception with Attributes

```python
class ModelError(Exception):
    """Base exception for model errors."""

    def __init__(self, message, model_name=None, error_code=None):
        super().__init__(message)
        self.model_name = model_name
        self.error_code = error_code
        self.timestamp = datetime.now()

    def __str__(self):
        return (
            f"{self.__class__.__name__}: {self.args[0]}\n"
            f"Model: {self.model_name}\n"
            f"Code: {self.error_code}\n"
            f"Time: {self.timestamp}"
        )


class ModelNotFoundError(ModelError):
    """Raised when model is not found."""
    pass


class ModelLoadError(ModelError):
    """Raised when model fails to load."""
    pass


class PredictionError(ModelError):
    """Raised when prediction fails."""
    pass


# Usage
try:
    raise ModelNotFoundError(
        "Model file not found",
        model_name="classifier_v1",
        error_code="E001"
    )
except ModelError as e:
    print(e)
    print(f"Error code: {e.error_code}")
```

#### Exception Hierarchy

```python
class ConfigError(Exception):
    """Base exception for configuration errors."""
    pass


class ConfigNotFoundError(ConfigError):
    """Configuration file not found."""
    pass


class ConfigParseError(ConfigError):
    """Configuration parsing failed."""
    pass


class ConfigValidationError(ConfigError):
    """Configuration validation failed."""
    pass


# Catch specific or general error
try:
    load_config("config.yaml")
except ConfigValidationError:
    # Handle validation error specifically
    pass
except ConfigError:
    # Handle any config error
    pass
```

### 6. Logging

#### Basic Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create logger
logger = logging.getLogger(__name__)

# Log messages
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")
```

#### Advanced Logging Configuration

```python
import logging
import logging.handlers

# Create logger
logger = logging.getLogger('my_app')
logger.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# File handler
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Rotating file handler
rotating_handler = logging.handlers.RotatingFileHandler(
    'app.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
rotating_handler.setLevel(logging.DEBUG)
rotating_handler.setFormatter(formatter)

# Add handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.addHandler(rotating_handler)

# Use logger
logger.info("Application started")
logger.debug("Debug information")
```

#### Logging with Context

```python
# Log with exception info
try:
    result = 10 / 0
except ZeroDivisionError:
    logger.error("Division by zero occurred", exc_info=True)

# Log with extra context
logger.info(
    "User logged in",
    extra={'user_id': 123, 'ip': '192.168.1.1'}
)

# Structured logging
def log_model_prediction(model_name, input_data, prediction, duration):
    """Log model prediction with context."""
    logger.info(
        "Model prediction completed",
        extra={
            'model': model_name,
            'input_size': len(input_data),
            'prediction': prediction,
            'duration_ms': duration * 1000
        }
    )
```

#### Logger Hierarchy

```python
# Module-level loggers
logger = logging.getLogger(__name__)

# In main.py
main_logger = logging.getLogger('myapp.main')

# In utils.py
utils_logger = logging.getLogger('myapp.utils')

# In models.py
models_logger = logging.getLogger('myapp.models')

# Configure root logger
logging.getLogger('myapp').setLevel(logging.INFO)
```

### 7. Path Operations

#### Using pathlib

```python
from pathlib import Path

# Create path object
path = Path('/home/user/data')

# Join paths
config_path = path / 'config' / 'settings.yaml'

# Check if exists
if path.exists():
    print("Path exists")

# Check if file or directory
if path.is_file():
    print("It's a file")
if path.is_dir():
    print("It's a directory")

# Get file parts
print(path.name)        # 'settings.yaml'
print(path.stem)        # 'settings'
print(path.suffix)      # '.yaml'
print(path.parent)      # '/home/user/data/config'

# Create directories
path.mkdir(parents=True, exist_ok=True)

# List directory contents
for item in path.iterdir():
    print(item)

# Find files
for yaml_file in path.glob('*.yaml'):
    print(yaml_file)

# Recursive search
for py_file in path.rglob('*.py'):
    print(py_file)

# Read/write with pathlib
content = path.read_text()
path.write_text("New content")
```

---

## Project: Configuration Management System

Build a robust configuration management system for ML applications.

### Requirements

**Features:**
1. Load configuration from YAML/JSON files
2. Validate configuration schema
3. Support environment-specific configs (dev, staging, prod)
4. Handle missing/invalid configurations gracefully
5. Log all configuration operations
6. Support configuration hot-reloading
7. Provide default values

**Technical Requirements:**
- Use Pydantic for validation
- Implement custom exceptions
- Comprehensive logging
- Context managers for file operations
- Type hints throughout
- Handle all edge cases

### Implementation

See `solutions/config_manager.py` for complete implementation.

### Example Usage

```python
from config_manager import ConfigManager, ConfigError

# Initialize configuration
config_manager = ConfigManager('configs')

try:
    # Load configuration
    config = config_manager.load_config('app.yaml', environment='production')

    # Access configuration
    db_host = config.database.host
    api_key = config.api.key

    # Validate configuration
    config_manager.validate_config(config)

    # Watch for changes
    config_manager.watch_config('app.yaml')

except ConfigError as e:
    logger.error(f"Configuration error: {e}")
```

---

## Practice Problems

### Problem 1: File Reader with Error Handling

```python
def safe_read_file(filepath: str, encoding: str = 'utf-8') -> Optional[str]:
    """
    Safely read file with comprehensive error handling.

    Args:
        filepath: Path to file
        encoding: File encoding

    Returns:
        File contents or None if error

    Example:
        >>> content = safe_read_file('data.txt')
        >>> if content:
        ...     print(content)
    """
    # Your implementation here
    pass
```

### Problem 2: JSON Validator

```python
def validate_json_file(filepath: str, schema: dict) -> Tuple[bool, Optional[str]]:
    """
    Validate JSON file against schema.

    Args:
        filepath: Path to JSON file
        schema: JSON schema for validation

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> schema = {'type': 'object', 'required': ['name', 'version']}
        >>> is_valid, error = validate_json_file('config.json', schema)
    """
    # Your implementation here
    pass
```

### Problem 3: Log Analyzer

```python
def analyze_log_file(filepath: str) -> Dict[str, int]:
    """
    Analyze log file and count log levels.

    Args:
        filepath: Path to log file

    Returns:
        Dictionary with counts for each log level

    Example:
        >>> counts = analyze_log_file('app.log')
        >>> print(counts)
        {'INFO': 150, 'WARNING': 25, 'ERROR': 5}
    """
    # Your implementation here
    pass
```

### Problem 4: Configuration Merger

```python
def merge_configs(*config_files: str) -> dict:
    """
    Merge multiple configuration files.

    Later files override earlier ones.

    Args:
        *config_files: Paths to configuration files

    Returns:
        Merged configuration dictionary

    Example:
        >>> config = merge_configs('base.yaml', 'dev.yaml', 'local.yaml')
    """
    # Your implementation here
    pass
```

### Problem 5: File Backup System

```python
def create_backup(filepath: str, backup_dir: str = 'backups') -> str:
    """
    Create timestamped backup of file.

    Args:
        filepath: File to backup
        backup_dir: Directory for backups

    Returns:
        Path to backup file

    Example:
        >>> backup_path = create_backup('important.db')
        >>> print(backup_path)
        'backups/important_20250124_143022.db'
    """
    # Your implementation here
    pass
```

---

## Code Examples

### Example 1: Robust File Reader

```python
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FileReadError(Exception):
    """Custom exception for file reading errors."""
    pass


def read_file_safely(
    filepath: str,
    encoding: str = 'utf-8',
    max_size_mb: int = 100
) -> Optional[str]:
    """
    Read file with comprehensive error handling.

    Args:
        filepath: Path to file
        encoding: File encoding
        max_size_mb: Maximum file size in MB

    Returns:
        File contents or None

    Raises:
        FileReadError: If file cannot be read
    """
    path = Path(filepath)

    # Check if file exists
    if not path.exists():
        logger.error(f"File not found: {filepath}")
        raise FileReadError(f"File not found: {filepath}")

    # Check if it's a file
    if not path.is_file():
        logger.error(f"Not a file: {filepath}")
        raise FileReadError(f"Not a file: {filepath}")

    # Check file size
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        logger.warning(f"File too large: {size_mb:.2f}MB > {max_size_mb}MB")
        raise FileReadError(f"File too large: {filepath}")

    # Read file
    try:
        with open(path, 'r', encoding=encoding) as f:
            content = f.read()
        logger.info(f"Successfully read {filepath} ({len(content)} chars)")
        return content

    except UnicodeDecodeError as e:
        logger.error(f"Encoding error reading {filepath}: {e}")
        raise FileReadError(f"Encoding error: {e}") from e

    except PermissionError as e:
        logger.error(f"Permission denied: {filepath}")
        raise FileReadError(f"Permission denied: {filepath}") from e

    except Exception as e:
        logger.error(f"Unexpected error reading {filepath}: {e}")
        raise FileReadError(f"Failed to read file: {e}") from e
```

### Example 2: Configuration Loader

```python
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigurationLoader:
    """Load and validate configuration files."""

    SUPPORTED_FORMATS = ['.yaml', '.yml', '.json']

    def __init__(self, config_dir: str = 'configs'):
        """
        Initialize configuration loader.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_cache: Dict[str, Dict[str, Any]] = {}

    def load(
        self,
        filename: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Load configuration file.

        Args:
            filename: Configuration filename
            use_cache: Whether to use cached config

        Returns:
            Configuration dictionary

        Raises:
            ValueError: If file format not supported
            FileNotFoundError: If file doesn't exist
        """
        # Check cache
        if use_cache and filename in self.config_cache:
            logger.debug(f"Returning cached config: {filename}")
            return self.config_cache[filename]

        # Build full path
        filepath = self.config_dir / filename

        # Check if file exists
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        # Check file format
        suffix = filepath.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {suffix}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )

        # Load configuration
        try:
            with open(filepath, 'r') as f:
                if suffix in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                else:  # .json
                    config = json.load(f)

            logger.info(f"Loaded configuration from {filepath}")

            # Cache configuration
            self.config_cache[filename] = config

            return config

        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {filepath}: {e}")
            raise ValueError(f"Invalid YAML: {e}") from e

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in {filepath}: {e}")
            raise ValueError(f"Invalid JSON: {e}") from e

        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            raise

    def merge_configs(self, *filenames: str) -> Dict[str, Any]:
        """
        Merge multiple configuration files.

        Later configs override earlier ones.

        Args:
            *filenames: Configuration filenames to merge

        Returns:
            Merged configuration
        """
        merged = {}

        for filename in filenames:
            config = self.load(filename)
            merged.update(config)
            logger.debug(f"Merged config from {filename}")

        return merged

    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self.config_cache.clear()
        logger.debug("Configuration cache cleared")
```

---

## Best Practices

### 1. Always Use Context Managers

```python
# Good - file automatically closed
with open('file.txt', 'r') as f:
    content = f.read()

# Bad - file might not be closed
f = open('file.txt', 'r')
content = f.read()
f.close()
```

### 2. Handle Specific Exceptions

```python
# Good - specific exception handling
try:
    with open('file.txt', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    logger.error("File not found")
except json.JSONDecodeError:
    logger.error("Invalid JSON")
except PermissionError:
    logger.error("Permission denied")

# Bad - catch all
try:
    with open('file.txt', 'r') as f:
        data = json.load(f)
except Exception:
    logger.error("Something went wrong")
```

### 3. Use Pathlib

```python
# Good - pathlib
from pathlib import Path

config_path = Path('configs') / 'app.yaml'
if config_path.exists():
    content = config_path.read_text()

# Less ideal - os.path
import os

config_path = os.path.join('configs', 'app.yaml')
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        content = f.read()
```

### 4. Structured Logging

```python
# Good - structured logging
logger.info(
    "Model prediction completed",
    extra={
        'model': 'classifier_v1',
        'duration_ms': 45.2,
        'input_size': 1000
    }
)

# Less ideal - string formatting
logger.info(f"Model classifier_v1 completed in 45.2ms with input size 1000")
```

### 5. Custom Exceptions

```python
# Good - specific custom exceptions
class ConfigError(Exception):
    """Base configuration error."""
    pass

class ConfigNotFoundError(ConfigError):
    """Configuration file not found."""
    pass

raise ConfigNotFoundError(f"Config not found: {path}")

# Bad - generic exceptions
raise Exception("Config not found")
```

---

## Common Pitfalls

### 1. Not Closing Files

```python
# Bad - file not closed on error
f = open('file.txt', 'r')
data = process(f.read())  # If this fails, file stays open
f.close()

# Good - always closed
with open('file.txt', 'r') as f:
    data = process(f.read())
```

### 2. Catching Too Broad

```python
# Bad - hides all errors
try:
    critical_operation()
except:
    pass

# Good - handle specific errors
try:
    critical_operation()
except ValueError as e:
    logger.error(f"Value error: {e}")
    raise
```

### 3. Not Logging Exceptions

```python
# Bad - silent failure
try:
    risky_operation()
except Exception:
    pass

# Good - log the error
try:
    risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    raise
```

---

## Validation

Run the validation script:

```bash
python tests/test_file_io.py
```

Expected output:
```
✅ File operations tests passed
✅ JSON/YAML handling correct
✅ Exception handling proper
✅ Logging configured correctly
✅ Configuration system working

🎉 Exercise 03 Complete!
```

---

## Resources

### Documentation
- [Python File I/O](https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files)
- [Context Managers](https://docs.python.org/3/library/contextlib.html)
- [Logging](https://docs.python.org/3/library/logging.html)
- [Pathlib](https://docs.python.org/3/library/pathlib.html)
- [PyYAML Documentation](https://pyyaml.org/wiki/PyYAMLDocumentation)

### Articles
- [Python Logging Best Practices](https://realpython.com/python-logging/)
- [Context Managers and Python's with Statement](https://realpython.com/python-with-statement/)
- [Working with Files in Python](https://realpython.com/working-with-files-in-python/)

---

## Next Steps

After completing this exercise:

1. **Exercise 04: Testing with pytest** - Write comprehensive tests
2. Practice file operations in your projects
3. Implement proper error handling in all code
4. Set up logging for your applications

---

**Handle errors gracefully and write production-ready code! 🛡️**
