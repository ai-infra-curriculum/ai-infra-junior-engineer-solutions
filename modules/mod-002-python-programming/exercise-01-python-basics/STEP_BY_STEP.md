# Step-by-Step Implementation Guide: Python Basics

## Overview

Master Python fundamentals through hands-on practice: data types, control flow, functions, and modules. Build a system monitoring CLI tool.

**Time**: 4-6 hours | **Difficulty**: Beginner

---

## Phase 1: Setup Python Environment (30 minutes)

### Step 1: Install Python and Verify

```bash
# Check Python version (3.11+ recommended)
python3 --version

# Check pip
pip3 --version

# If not installed:
# macOS: brew install python3
# Ubuntu: sudo apt install python3 python3-pip
# Windows: Download from python.org
```

### Step 2: Create Virtual Environment

```bash
# Create project directory
mkdir python-basics-project
cd python-basics-project

# Create virtual environment
python3 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Verify activation (should show (venv) prefix)
which python  # Should show venv/bin/python
```

### Step 3: Install Dependencies

```bash
# Install required packages
pip install psutil click rich

# Create requirements.txt
cat > requirements.txt <<EOF
psutil==5.9.6
click==8.1.7
rich==13.7.0
EOF

# Install from requirements
pip install -r requirements.txt

# Verify installations
python -c "import psutil, click, rich; print('✅ All packages installed')"
```

---

## Phase 2: Python Basics Practice (1-1.5 hours)

### Step 4: Data Types and Variables

Create `basics/01_data_types.py`:

```python
"""Practice with Python data types."""

# Numbers
integer_example = 42
float_example = 3.14159
complex_example = 2 + 3j

print(f"Integer: {integer_example}, Type: {type(integer_example)}")
print(f"Float: {float_example}, Type: {type(float_example)}")
print(f"Complex: {complex_example}, Type: {type(complex_example)}")

# Strings
name = "AI Infrastructure"
multiline = """This is a
multiline string
for documentation"""

print(f"String: {name}")
print(f"Length: {len(name)}")
print(f"Uppercase: {name.upper()}")
print(f"Split: {name.split()}")

# Collections
my_list = [1, 2, 3, 4, 5]
my_tuple = (1, 2, 3)  # Immutable
my_dict = {"model": "bert", "version": "1.0", "accuracy": 0.95}
my_set = {1, 2, 3, 2, 1}  # Duplicates removed

print(f"List: {my_list}")
print(f"Tuple: {my_tuple}")
print(f"Dict: {my_dict}")
print(f"Set: {my_set}")  # {1, 2, 3}

# Type conversion
str_num = "42"
converted = int(str_num)
print(f"Converted '{str_num}' to {converted}")
```

**Run and verify**:
```bash
python basics/01_data_types.py
# Expected: Output showing all data types
```

### Step 5: Control Flow

Create `basics/02_control_flow.py`:

```python
"""Practice control flow structures."""

# If/Else
def get_grade(score: int) -> str:
    """Return letter grade based on score."""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"

# Test
print(f"Score 95: Grade {get_grade(95)}")
print(f"Score 82: Grade {get_grade(82)}")

# For loops
print("\nCounting with for loop:")
for i in range(5):
    print(f"Count: {i}")

# Loop over list
names = ["Alice", "Bob", "Charlie"]
for name in names:
    print(f"Hello, {name}!")

# While loop
print("\nCounting with while loop:")
count = 0
while count < 5:
    print(f"Count: {count}")
    count += 1

# Break and Continue
print("\nBreak and Continue:")
for i in range(10):
    if i == 3:
        continue  # Skip 3
    if i == 7:
        break  # Stop at 7
    print(i)  # Prints: 0, 1, 2, 4, 5, 6
```

**Run**:
```bash
python basics/02_control_flow.py
```

### Step 6: Functions

Create `basics/03_functions.py`:

```python
"""Practice writing functions."""

from typing import List, Tuple, Dict

# Basic function
def greet(name: str) -> str:
    """Return greeting message."""
    return f"Hello, {name}!"

# Function with default arguments
def power(base: float, exponent: float = 2) -> float:
    """Calculate base raised to exponent (default 2)."""
    return base ** exponent

# Multiple return values
def get_stats(numbers: List[float]) -> Tuple[float, float, float]:
    """Return min, max, and average."""
    return min(numbers), max(numbers), sum(numbers) / len(numbers)

# Type hints
def calculate_accuracy(correct: int, total: int) -> float:
    """Calculate accuracy percentage."""
    if total == 0:
        return 0.0
    return (correct / total) * 100

# *args and **kwargs
def log_message(*args, **kwargs):
    """Log message with flexible arguments."""
    level = kwargs.get('level', 'INFO')
    message = ' '.join(str(arg) for arg in args)
    print(f"[{level}] {message}")

# Lambda functions
square = lambda x: x ** 2
add = lambda x, y: x + y

# Test functions
if __name__ == "__main__":
    print(greet("World"))
    print(f"2^3 = {power(2, 3)}")
    print(f"2^2 (default) = {power(2)}")

    stats = get_stats([1, 2, 3, 4, 5])
    print(f"Stats: min={stats[0]}, max={stats[1]}, avg={stats[2]}")

    print(f"Accuracy: {calculate_accuracy(85, 100)}%")

    log_message("Server started", "on port 8000", level="INFO")
    log_message("Error occurred", level="ERROR")

    print(f"Square of 5: {square(5)}")
    print(f"Add 3 + 7: {add(3, 7)}")
```

**Run**:
```bash
python basics/03_functions.py
```

### Step 7: List Comprehensions

Create `basics/04_comprehensions.py`:

```python
"""Practice list comprehensions and generators."""

# List comprehension
squares = [x ** 2 for x in range(10)]
print(f"Squares: {squares}")

# With condition
evens = [x for x in range(20) if x % 2 == 0]
print(f"Even numbers: {evens}")

# Nested comprehension
matrix = [[i + j for j in range(3)] for i in range(3)]
print(f"Matrix:\n{matrix}")

# Dict comprehension
squared_dict = {x: x**2 for x in range(5)}
print(f"Squared dict: {squared_dict}")

# Set comprehension
words = ["hello", "world", "hi", "python", "code"]
unique_lengths = {len(word) for word in words}
print(f"Unique word lengths: {unique_lengths}")

# Generator function
def fibonacci(n: int):
    """Generate first n Fibonacci numbers."""
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1

# Use generator
print("Fibonacci sequence:")
for num in fibonacci(10):
    print(num, end=' ')
print()

# Generator expression (lazy evaluation)
squared_gen = (x**2 for x in range(1000000))
print(f"First 5 squares: {[next(squared_gen) for _ in range(5)]}")
```

**Run**:
```bash
python basics/04_comprehensions.py
```

---

## Phase 3: Practice Problems (1.5-2 hours)

### Step 8: Solve FizzBuzz

Create `solutions/fizzbuzz.py`:

```python
"""FizzBuzz implementation."""

def fizzbuzz(n: int) -> list:
    """
    Return FizzBuzz sequence from 1 to n.

    Rules:
    - Multiples of 3: "Fizz"
    - Multiples of 5: "Buzz"
    - Multiples of both: "FizzBuzz"
    - Otherwise: number as string
    """
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:  # Multiple of both 3 and 5
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result

# Test
if __name__ == "__main__":
    result = fizzbuzz(15)
    print(result)
    # Expected: ['1', '2', 'Fizz', '4', 'Buzz', 'Fizz', '7', '8',
    #            'Fizz', 'Buzz', '11', 'Fizz', '13', '14', 'FizzBuzz']
```

### Step 9: Find Duplicates

Create `solutions/duplicates.py`:

```python
"""Find duplicate numbers."""

def find_duplicates(numbers: list) -> list:
    """Find all duplicate numbers in a list."""
    seen = set()
    duplicates = set()

    for num in numbers:
        if num in seen:
            duplicates.add(num)
        else:
            seen.add(num)

    return sorted(list(duplicates))

# Test
if __name__ == "__main__":
    result = find_duplicates([1, 2, 3, 2, 4, 5, 3, 6, 7, 3])
    print(f"Duplicates: {result}")
    # Expected: [2, 3]
```

### Step 10: Word Counter

Create `solutions/word_counter.py`:

```python
"""Count word frequencies."""

from collections import Counter

def count_words(text: str) -> dict:
    """Count frequency of each word in text."""
    # Clean and split
    words = text.lower().split()

    # Method 1: Manual counting
    # word_count = {}
    # for word in words:
    #     word_count[word] = word_count.get(word, 0) + 1

    # Method 2: Using Counter (better)
    word_count = Counter(words)

    return dict(word_count)

# Test
if __name__ == "__main__":
    text = "hello world hello python world python python"
    result = count_words(text)
    print(f"Word counts: {result}")
    # Expected: {'hello': 2, 'world': 2, 'python': 3}
```

### Step 11: Prime Numbers

Create `solutions/primes.py`:

```python
"""Prime number operations."""

def is_prime(n: int) -> bool:
    """Check if number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    # Check odd divisors up to sqrt(n)
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2

    return True

def get_primes(limit: int) -> list:
    """Get all prime numbers up to limit."""
    return [n for n in range(2, limit + 1) if is_prime(n)]

# Test
if __name__ == "__main__":
    print(f"Is 17 prime? {is_prime(17)}")
    print(f"Is 20 prime? {is_prime(20)}")

    primes = get_primes(30)
    print(f"Primes up to 30: {primes}")
    # Expected: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
```

---

## Phase 4: System Monitor CLI Project (2-2.5 hours)

### Step 12: Create System Monitor

Create `system_monitor.py`:

```python
"""System resource monitoring CLI tool."""

import psutil
import click
import json
from datetime import datetime
from rich.console import Console
from rich.table import Table
import time

console = Console()

@click.group()
def cli():
    """System Monitor CLI - Monitor system resources."""
    pass

@cli.command()
def cpu():
    """Display CPU usage."""
    usage = psutil.cpu_percent(interval=1)
    count = psutil.cpu_count()

    table = Table(title="CPU Information")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("CPU Usage", f"{usage}%")
    table.add_row("CPU Count", str(count))
    table.add_row("CPU Frequency", f"{psutil.cpu_freq().current:.2f} MHz")

    console.print(table)

@cli.command()
def memory():
    """Display memory usage."""
    mem = psutil.virtual_memory()

    table = Table(title="Memory Information")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total", f"{mem.total / (1024**3):.2f} GB")
    table.add_row("Available", f"{mem.available / (1024**3):.2f} GB")
    table.add_row("Used", f"{mem.used / (1024**3):.2f} GB")
    table.add_row("Percent", f"{mem.percent}%")

    console.print(table)

@cli.command()
def disk():
    """Display disk usage."""
    disk = psutil.disk_usage('/')

    table = Table(title="Disk Information")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total", f"{disk.total / (1024**3):.2f} GB")
    table.add_row("Used", f"{disk.used / (1024**3):.2f} GB")
    table.add_row("Free", f"{disk.free / (1024**3):.2f} GB")
    table.add_row("Percent", f"{disk.percent}%")

    console.print(table)

@cli.command()
@click.option('--limit', default=10, help='Number of processes to show')
def processes(limit):
    """Display running processes."""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Sort by CPU usage
    processes.sort(key=lambda x: x['cpu_percent'], reverse=True)

    table = Table(title=f"Top {limit} Processes by CPU")
    table.add_column("PID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("CPU %", style="magenta")
    table.add_column("Memory %", style="yellow")

    for proc in processes[:limit]:
        table.add_row(
            str(proc['pid']),
            proc['name'],
            f"{proc['cpu_percent']:.1f}",
            f"{proc['memory_percent']:.1f}"
        )

    console.print(table)

@cli.command()
@click.option('--output', default=None, help='Output file (JSON)')
def all_metrics(output):
    """Display all metrics."""
    cpu_usage = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "cpu": {
            "usage_percent": cpu_usage,
            "count": psutil.cpu_count()
        },
        "memory": {
            "total_gb": mem.total / (1024**3),
            "used_gb": mem.used / (1024**3),
            "percent": mem.percent
        },
        "disk": {
            "total_gb": disk.total / (1024**3),
            "used_gb": disk.used / (1024**3),
            "percent": disk.percent
        }
    }

    # Display
    console.print("[bold cyan]System Metrics[/bold cyan]")
    console.print(f"CPU Usage: {cpu_usage}%")
    console.print(f"Memory Usage: {mem.percent}%")
    console.print(f"Disk Usage: {disk.percent}%")

    # Save to file if requested
    if output:
        with open(output, 'w') as f:
            json.dump(metrics, f, indent=2)
        console.print(f"\n[green]✅ Metrics saved to {output}[/green]")

@cli.command()
@click.option('--interval', default=2, help='Refresh interval in seconds')
def watch(interval):
    """Watch mode - continuously refresh metrics."""
    try:
        while True:
            console.clear()
            cpu_usage = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()

            console.print(f"[bold cyan]System Monitor[/bold cyan] - {datetime.now().strftime('%H:%M:%S')}")
            console.print(f"CPU: {cpu_usage}%")
            console.print(f"Memory: {mem.percent}%")
            console.print(f"\nRefreshing every {interval}s... (Press Ctrl+C to stop)")

            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped[/yellow]")

if __name__ == "__main__":
    cli()
```

### Step 13: Test System Monitor

```bash
# Show CPU info
python system_monitor.py cpu

# Show memory info
python system_monitor.py memory

# Show disk info
python system_monitor.py disk

# Show top processes
python system_monitor.py processes --limit 10

# Show all metrics
python system_monitor.py all-metrics

# Save to file
python system_monitor.py all-metrics --output metrics.json

# Watch mode
python system_monitor.py watch --interval 2
# Press Ctrl+C to stop
```

---

## Phase 5: Testing and Validation (30 minutes)

### Step 14: Create Tests

Create `tests/test_solutions.py`:

```python
"""Tests for practice problems."""

import pytest
from solutions.fizzbuzz import fizzbuzz
from solutions.duplicates import find_duplicates
from solutions.word_counter import count_words
from solutions.primes import is_prime, get_primes

def test_fizzbuzz():
    """Test FizzBuzz implementation."""
    result = fizzbuzz(15)
    assert result[0] == "1"
    assert result[2] == "Fizz"  # 3
    assert result[4] == "Buzz"  # 5
    assert result[14] == "FizzBuzz"  # 15
    assert len(result) == 15

def test_find_duplicates():
    """Test duplicate finder."""
    assert find_duplicates([1, 2, 3, 2, 4, 5, 3]) == [2, 3]
    assert find_duplicates([1, 2, 3]) == []
    assert find_duplicates([1, 1, 1]) == [1]

def test_word_counter():
    """Test word counter."""
    result = count_words("hello world hello")
    assert result['hello'] == 2
    assert result['world'] == 1

def test_primes():
    """Test prime functions."""
    assert is_prime(17) == True
    assert is_prime(20) == False
    assert is_prime(2) == True
    assert is_prime(1) == False

    primes = get_primes(10)
    assert primes == [2, 3, 5, 7]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Run tests**:
```bash
# Install pytest
pip install pytest

# Run tests
pytest tests/test_solutions.py -v

# Expected: All tests pass
```

---

## Summary

**What You Built**:
- ✅ Mastered Python data types (int, float, str, list, dict, set, tuple)
- ✅ Control flow (if/else, for, while, break, continue)
- ✅ Functions (def, lambda, type hints, *args, **kwargs)
- ✅ List comprehensions and generators
- ✅ Solved classic programming problems (FizzBuzz, duplicates, primes)
- ✅ Built complete CLI tool for system monitoring
- ✅ Wrote tests with pytest

**Key Python Concepts**:
```python
# Type hints
def calculate(x: int, y: int) -> int:
    return x + y

# List comprehension
squares = [x**2 for x in range(10) if x % 2 == 0]

# Generator
def fibonacci(n):
    a, b = 0, 1
    while a < n:
        yield a
        a, b = b, a + b

# Context manager
with open('file.txt', 'r') as f:
    data = f.read()
```

**Next Steps**:
- Exercise 02: Object-Oriented Programming
- Practice on coding platforms (LeetCode, HackerRank)
- Build automation scripts with Python
- Read "Python Crash Course" or "Automate the Boring Stuff"
