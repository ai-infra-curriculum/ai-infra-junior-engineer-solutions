#!/usr/bin/env python3
"""
File I/O Operations Examples

Practical examples of file operations with proper error handling.
"""

import json
import yaml
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import time


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===========================
# 1. Basic File Operations
# ===========================

def read_text_file(filepath: str) -> Optional[str]:
    """
    Read text file with error handling.

    Args:
        filepath: Path to text file

    Returns:
        File contents or None if error
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Read {len(content)} characters from {filepath}")
        return content

    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return None

    except PermissionError:
        logger.error(f"Permission denied: {filepath}")
        return None

    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return None


def write_text_file(filepath: str, content: str) -> bool:
    """
    Write text file with error handling.

    Args:
        filepath: Path to text file
        content: Content to write

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Wrote {len(content)} characters to {filepath}")
        return True

    except PermissionError:
        logger.error(f"Permission denied: {filepath}")
        return False

    except Exception as e:
        logger.error(f"Error writing {filepath}: {e}")
        return False


# ===========================
# 2. JSON Operations
# ===========================

def read_json(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Read JSON file with error handling.

    Args:
        filepath: Path to JSON file

    Returns:
        Parsed JSON or None if error
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON from {filepath}")
        return data

    except FileNotFoundError:
        logger.error(f"JSON file not found: {filepath}")
        return None

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filepath}: {e}")
        return None

    except Exception as e:
        logger.error(f"Error reading JSON {filepath}: {e}")
        return None


def write_json(filepath: str, data: Dict[str, Any], indent: int = 2) -> bool:
    """
    Write JSON file with error handling.

    Args:
        filepath: Path to JSON file
        data: Data to write
        indent: JSON indentation

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)
        logger.info(f"Wrote JSON to {filepath}")
        return True

    except TypeError as e:
        logger.error(f"Data not JSON serializable: {e}")
        return False

    except Exception as e:
        logger.error(f"Error writing JSON {filepath}: {e}")
        return False


# ===========================
# 3. YAML Operations
# ===========================

def read_yaml(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Read YAML file with error handling.

    Args:
        filepath: Path to YAML file

    Returns:
        Parsed YAML or None if error
    """
    try:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        logger.info(f"Loaded YAML from {filepath}")
        return data if data else {}

    except FileNotFoundError:
        logger.error(f"YAML file not found: {filepath}")
        return None

    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in {filepath}: {e}")
        return None

    except Exception as e:
        logger.error(f"Error reading YAML {filepath}: {e}")
        return None


def write_yaml(filepath: str, data: Dict[str, Any]) -> bool:
    """
    Write YAML file with error handling.

    Args:
        filepath: Path to YAML file
        data: Data to write

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        logger.info(f"Wrote YAML to {filepath}")
        return True

    except Exception as e:
        logger.error(f"Error writing YAML {filepath}: {e}")
        return False


# ===========================
# 4. CSV Operations
# ===========================

def read_csv(filepath: str) -> Optional[List[Dict[str, str]]]:
    """
    Read CSV file as list of dictionaries.

    Args:
        filepath: Path to CSV file

    Returns:
        List of dictionaries or None if error
    """
    try:
        with open(filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        logger.info(f"Read {len(data)} rows from {filepath}")
        return data

    except FileNotFoundError:
        logger.error(f"CSV file not found: {filepath}")
        return None

    except Exception as e:
        logger.error(f"Error reading CSV {filepath}: {e}")
        return None


def write_csv(
    filepath: str,
    data: List[Dict[str, Any]],
    fieldnames: Optional[List[str]] = None
) -> bool:
    """
    Write CSV file from list of dictionaries.

    Args:
        filepath: Path to CSV file
        data: List of dictionaries
        fieldnames: Field names (inferred from first row if not provided)

    Returns:
        True if successful, False otherwise
    """
    if not data:
        logger.warning("No data to write")
        return False

    try:
        if fieldnames is None:
            fieldnames = list(data[0].keys())

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        logger.info(f"Wrote {len(data)} rows to {filepath}")
        return True

    except Exception as e:
        logger.error(f"Error writing CSV {filepath}: {e}")
        return False


# ===========================
# 5. Context Managers
# ===========================

@contextmanager
def timer(operation: str):
    """
    Context manager for timing operations.

    Args:
        operation: Operation description
    """
    start = time.time()
    logger.info(f"Starting: {operation}")

    try:
        yield
    finally:
        duration = time.time() - start
        logger.info(f"Completed: {operation} in {duration:.4f}s")


@contextmanager
def safe_file_operation(filepath: str, mode: str = 'r'):
    """
    Context manager for safe file operations.

    Args:
        filepath: Path to file
        mode: File mode ('r', 'w', 'a', etc.)

    Yields:
        File object
    """
    file_obj = None
    try:
        file_obj = open(filepath, mode)
        yield file_obj

    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise

    except PermissionError:
        logger.error(f"Permission denied: {filepath}")
        raise

    except Exception as e:
        logger.error(f"Error with file {filepath}: {e}")
        raise

    finally:
        if file_obj:
            file_obj.close()
            logger.debug(f"Closed file: {filepath}")


# ===========================
# 6. Batch File Operations
# ===========================

def read_multiple_files(filepaths: List[str]) -> Dict[str, Optional[str]]:
    """
    Read multiple files.

    Args:
        filepaths: List of file paths

    Returns:
        Dictionary mapping filepath to content
    """
    results = {}

    for filepath in filepaths:
        content = read_text_file(filepath)
        results[filepath] = content

    success_count = sum(1 for v in results.values() if v is not None)
    logger.info(f"Successfully read {success_count}/{len(filepaths)} files")

    return results


def find_and_read_files(directory: str, pattern: str) -> Dict[str, str]:
    """
    Find and read files matching pattern.

    Args:
        directory: Directory to search
        pattern: Glob pattern (e.g., '*.txt')

    Returns:
        Dictionary mapping filepath to content
    """
    dir_path = Path(directory)
    results = {}

    try:
        for filepath in dir_path.glob(pattern):
            if filepath.is_file():
                content = read_text_file(str(filepath))
                if content:
                    results[str(filepath)] = content

        logger.info(f"Found and read {len(results)} files")
        return results

    except Exception as e:
        logger.error(f"Error finding files: {e}")
        return {}


# ===========================
# 7. File Information
# ===========================

def get_file_info(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Get file information.

    Args:
        filepath: Path to file

    Returns:
        Dictionary with file information
    """
    try:
        path = Path(filepath)

        if not path.exists():
            logger.error(f"File does not exist: {filepath}")
            return None

        stat = path.stat()

        info = {
            'name': path.name,
            'stem': path.stem,
            'suffix': path.suffix,
            'size_bytes': stat.st_size,
            'size_kb': stat.st_size / 1024,
            'size_mb': stat.st_size / (1024 * 1024),
            'is_file': path.is_file(),
            'is_dir': path.is_dir(),
            'parent': str(path.parent),
            'absolute': str(path.absolute())
        }

        logger.info(f"Retrieved info for {filepath}")
        return info

    except Exception as e:
        logger.error(f"Error getting file info for {filepath}: {e}")
        return None


# ===========================
# 8. Demo Functions
# ===========================

def demo_basic_operations():
    """Demonstrate basic file operations."""
    print("\n" + "=" * 60)
    print("Basic File Operations Demo")
    print("=" * 60)

    # Write text file
    print("\n1. Writing text file...")
    success = write_text_file("demo.txt", "Hello, File I/O!\nThis is a test.")
    print(f"   Result: {'Success' if success else 'Failed'}")

    # Read text file
    print("\n2. Reading text file...")
    content = read_text_file("demo.txt")
    if content:
        print(f"   Content: {content[:50]}...")

    # Clean up
    Path("demo.txt").unlink(missing_ok=True)


def demo_json_operations():
    """Demonstrate JSON operations."""
    print("\n" + "=" * 60)
    print("JSON Operations Demo")
    print("=" * 60)

    # Write JSON
    print("\n1. Writing JSON file...")
    data = {
        "name": "ML Model",
        "version": "1.0.0",
        "metrics": {
            "accuracy": 0.95,
            "loss": 0.05
        }
    }
    write_json("demo.json", data)

    # Read JSON
    print("\n2. Reading JSON file...")
    loaded = read_json("demo.json")
    if loaded:
        print(f"   Name: {loaded.get('name')}")
        print(f"   Metrics: {loaded.get('metrics')}")

    # Clean up
    Path("demo.json").unlink(missing_ok=True)


def demo_csv_operations():
    """Demonstrate CSV operations."""
    print("\n" + "=" * 60)
    print("CSV Operations Demo")
    print("=" * 60)

    # Write CSV
    print("\n1. Writing CSV file...")
    data = [
        {"model": "Model A", "accuracy": 0.95, "f1_score": 0.93},
        {"model": "Model B", "accuracy": 0.92, "f1_score": 0.91},
        {"model": "Model C", "accuracy": 0.97, "f1_score": 0.96}
    ]
    write_csv("demo.csv", data)

    # Read CSV
    print("\n2. Reading CSV file...")
    loaded = read_csv("demo.csv")
    if loaded:
        print(f"   Read {len(loaded)} rows")
        for row in loaded:
            print(f"   {row['model']}: {row['accuracy']}")

    # Clean up
    Path("demo.csv").unlink(missing_ok=True)


def demo_context_managers():
    """Demonstrate context managers."""
    print("\n" + "=" * 60)
    print("Context Managers Demo")
    print("=" * 60)

    # Timer context manager
    print("\n1. Using timer context manager...")
    with timer("Sleep operation"):
        time.sleep(0.1)

    # Safe file operation
    print("\n2. Using safe file operation...")
    write_text_file("demo.txt", "Test content")

    with safe_file_operation("demo.txt", 'r') as f:
        content = f.read()
        print(f"   Read: {content}")

    # Clean up
    Path("demo.txt").unlink(missing_ok=True)


def main():
    """Run all demos."""
    print("=" * 70)
    print("File I/O Operations Examples")
    print("=" * 70)

    demo_basic_operations()
    demo_json_operations()
    demo_csv_operations()
    demo_context_managers()

    print("\n" + "=" * 70)
    print("All demos complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
