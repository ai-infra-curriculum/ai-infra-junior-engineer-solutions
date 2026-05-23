#!/usr/bin/env python3
"""
Context Managers and Resource Management

Comprehensive examples of context managers, the with statement, and proper resource handling.
"""

from contextlib import contextmanager, suppress, redirect_stdout, ExitStack
from typing import Any, Optional, List, Dict
import time
import io
import tempfile
import os


# ============= Basic Context Manager Class =============

class Timer:
    """
    Context manager to measure execution time.

    Example:
        >>> with Timer("operation"):
        ...     # code to time
        ...     time.sleep(0.1)
    """

    def __init__(self, name: str = "Operation"):
        """
        Initialize timer.

        Args:
            name: Name of operation being timed
        """
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        """Start timer."""
        self.start_time = time.time()
        print(f"[{self.name}] Starting...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and print elapsed time."""
        self.elapsed = time.time() - self.start_time
        print(f"[{self.name}] Completed in {self.elapsed:.4f} seconds")
        return False  # Don't suppress exceptions


class DatabaseConnection:
    """
    Context manager for database connection simulation.

    Example:
        >>> with DatabaseConnection("mydb") as conn:
        ...     conn.execute("SELECT * FROM users")
    """

    def __init__(self, db_name: str):
        """
        Initialize connection.

        Args:
            db_name: Database name
        """
        self.db_name = db_name
        self.connected = False

    def __enter__(self):
        """Establish connection."""
        print(f"Connecting to database: {self.db_name}")
        self.connected = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close connection."""
        if self.connected:
            print(f"Closing connection to: {self.db_name}")
            self.connected = False

        if exc_type is not None:
            print(f"Exception occurred: {exc_type.__name__}: {exc_val}")

        return False  # Don't suppress exceptions

    def execute(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute query.

        Args:
            query: SQL query

        Returns:
            Query results
        """
        if not self.connected:
            raise RuntimeError("Not connected to database")

        print(f"Executing: {query}")
        return [{"id": 1, "name": "Sample"}]


class FileWriter:
    """
    Context manager for safe file writing.

    Example:
        >>> with FileWriter("output.txt") as writer:
        ...     writer.write("Hello, World!")
    """

    def __init__(self, filepath: str, mode: str = 'w'):
        """
        Initialize file writer.

        Args:
            filepath: Path to file
            mode: Write mode
        """
        self.filepath = filepath
        self.mode = mode
        self.file = None

    def __enter__(self):
        """Open file."""
        self.file = open(self.filepath, self.mode, encoding='utf-8')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close file."""
        if self.file:
            self.file.close()

        if exc_type is not None:
            # If error occurred, try to clean up partial file
            if os.path.exists(self.filepath):
                print(f"Error occurred, cleaning up: {self.filepath}")

        return False

    def write(self, content: str):
        """Write content to file."""
        if not self.file:
            raise RuntimeError("File not open")
        self.file.write(content)


# ============= Context Manager Generators =============

@contextmanager
def temporary_directory(prefix: str = "tmp"):
    """
    Create temporary directory that's automatically cleaned up.

    Args:
        prefix: Directory name prefix

    Yields:
        Path to temporary directory

    Example:
        >>> with temporary_directory() as tmpdir:
        ...     # use tmpdir
        ...     pass
    """
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    print(f"Created temporary directory: {tmpdir}")

    try:
        yield tmpdir
    finally:
        import shutil
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
            print(f"Cleaned up temporary directory: {tmpdir}")


@contextmanager
def change_directory(path: str):
    """
    Temporarily change current directory.

    Args:
        path: Directory to change to

    Yields:
        None

    Example:
        >>> with change_directory("/tmp"):
        ...     # operations in /tmp
        ...     pass
    """
    original_dir = os.getcwd()
    os.chdir(path)
    print(f"Changed directory to: {path}")

    try:
        yield
    finally:
        os.chdir(original_dir)
        print(f"Restored directory to: {original_dir}")


@contextmanager
def capture_output():
    """
    Capture stdout output.

    Yields:
        StringIO object containing captured output

    Example:
        >>> with capture_output() as output:
        ...     print("Hello")
        >>> output.getvalue()
        'Hello\\n'
    """
    output = io.StringIO()

    try:
        with redirect_stdout(output):
            yield output
    finally:
        pass


@contextmanager
def error_handling(error_message: str, default: Any = None):
    """
    Handle errors with context manager.

    Args:
        error_message: Message to print on error
        default: Default value to return

    Yields:
        None

    Example:
        >>> with error_handling("Failed"):
        ...     # risky operation
        ...     pass
    """
    try:
        yield
    except Exception as e:
        print(f"{error_message}: {type(e).__name__}: {e}")
        return default


@contextmanager
def managed_resource(resource_name: str):
    """
    Manage generic resource with setup and teardown.

    Args:
        resource_name: Name of resource

    Yields:
        Resource object

    Example:
        >>> with managed_resource("api_client") as resource:
        ...     # use resource
        ...     pass
    """
    # Setup
    print(f"Acquiring resource: {resource_name}")
    resource = {"name": resource_name, "active": True}

    try:
        yield resource
    finally:
        # Teardown
        resource["active"] = False
        print(f"Released resource: {resource_name}")


# ============= Advanced Context Managers =============

class TransactionManager:
    """
    Context manager for transaction-like operations.

    Example:
        >>> with TransactionManager() as txn:
        ...     txn.add_operation("op1")
        ...     txn.add_operation("op2")
    """

    def __init__(self):
        """Initialize transaction manager."""
        self.operations = []
        self.committed = False

    def __enter__(self):
        """Start transaction."""
        print("Transaction started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Commit or rollback transaction."""
        if exc_type is None:
            self.commit()
        else:
            self.rollback()

        return False

    def add_operation(self, operation: str):
        """Add operation to transaction."""
        self.operations.append(operation)
        print(f"Added operation: {operation}")

    def commit(self):
        """Commit transaction."""
        print(f"Committing {len(self.operations)} operations")
        self.committed = True
        self.operations.clear()

    def rollback(self):
        """Rollback transaction."""
        print(f"Rolling back {len(self.operations)} operations")
        self.operations.clear()


class LockManager:
    """
    Context manager for lock/mutex simulation.

    Example:
        >>> with LockManager("critical_section") as lock:
        ...     # protected code
        ...     pass
    """

    def __init__(self, name: str):
        """
        Initialize lock manager.

        Args:
            name: Lock name
        """
        self.name = name
        self.locked = False

    def __enter__(self):
        """Acquire lock."""
        print(f"Acquiring lock: {self.name}")
        self.locked = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release lock."""
        if self.locked:
            print(f"Releasing lock: {self.name}")
            self.locked = False

        return False


# ============= Nested Context Managers =============

@contextmanager
def nested_resources(*resource_names):
    """
    Manage multiple resources.

    Args:
        *resource_names: Names of resources to manage

    Yields:
        List of resources

    Example:
        >>> with nested_resources("res1", "res2") as resources:
        ...     # use resources
        ...     pass
    """
    resources = []

    try:
        # Acquire all resources
        for name in resource_names:
            print(f"Acquiring: {name}")
            resources.append({"name": name, "active": True})

        yield resources

    finally:
        # Release in reverse order
        for resource in reversed(resources):
            print(f"Releasing: {resource['name']}")
            resource["active"] = False


class ContextStack:
    """
    Manage multiple context managers with ExitStack.

    Example:
        >>> stack = ContextStack()
        >>> stack.add_timer("operation")
        >>> stack.add_lock("mutex")
        >>> with stack:
        ...     # all contexts active
        ...     pass
    """

    def __init__(self):
        """Initialize context stack."""
        self.stack = ExitStack()
        self.contexts = []

    def add_timer(self, name: str):
        """Add timer to stack."""
        self.contexts.append(('timer', name))

    def add_lock(self, name: str):
        """Add lock to stack."""
        self.contexts.append(('lock', name))

    def __enter__(self):
        """Enter all contexts."""
        for context_type, name in self.contexts:
            if context_type == 'timer':
                self.stack.enter_context(Timer(name))
            elif context_type == 'lock':
                self.stack.enter_context(LockManager(name))

        return self.stack.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit all contexts."""
        return self.stack.__exit__(exc_type, exc_val, exc_tb)


# ============= Practical Examples =============

def process_file_safely(input_path: str, output_path: str):
    """
    Process file with proper resource management.

    Args:
        input_path: Input file path
        output_path: Output file path

    Example:
        >>> # process_file_safely("input.txt", "output.txt")
        >>> pass
    """
    with Timer("File Processing"):
        # Both files automatically closed even if error occurs
        with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
            for line in infile:
                processed = line.upper()
                outfile.write(processed)


def database_transaction_example():
    """
    Demonstrate database transaction with context manager.

    Example:
        >>> # database_transaction_example()
        >>> pass
    """
    with TransactionManager() as txn:
        txn.add_operation("INSERT INTO users VALUES (1, 'Alice')")
        txn.add_operation("INSERT INTO users VALUES (2, 'Bob')")
        # If exception occurs, transaction rolls back
        # Otherwise, commits on exit


if __name__ == '__main__':
    print("=== Context Manager Examples ===\n")

    # Timer
    print("1. Timer:")
    with Timer("Sleep Operation"):
        time.sleep(0.1)

    print("\n2. Database Connection:")
    with DatabaseConnection("testdb") as conn:
        results = conn.execute("SELECT * FROM users")
        print(f"   Results: {results}")

    print("\n3. Temporary Directory:")
    with temporary_directory(prefix="demo_") as tmpdir:
        print(f"   Using: {tmpdir}")

    print("\n4. Transaction Manager:")
    with TransactionManager() as txn:
        txn.add_operation("operation1")
        txn.add_operation("operation2")

    print("\n5. Lock Manager:")
    with LockManager("critical_section"):
        print("   Inside critical section")

    print("\n6. Nested Resources:")
    with nested_resources("db", "cache", "api") as resources:
        print(f"   Managing {len(resources)} resources")

    print("\n7. Suppress Exceptions:")
    with suppress(FileNotFoundError):
        # This won't raise an error
        open("nonexistent_file.txt")
        print("   (File not found, but suppressed)")

    print("\n8. Capture Output:")
    with capture_output() as output:
        print("This will be captured")
    print(f"   Captured: {repr(output.getvalue())}")

    print("\n✓ All context manager examples completed")
