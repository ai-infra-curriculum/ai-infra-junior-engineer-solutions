# Exercise 01: Python Basics

## Overview

Master Python fundamentals: data types, control flow, functions, and modules. Build a solid foundation for AI Infrastructure development.

## Learning Objectives

- ✅ Understand Python data types (str, int, float, list, dict, set, tuple)
- ✅ Use control structures (if/else, for, while)
- ✅ Write functions with proper documentation
- ✅ Work with list comprehensions and generators
- ✅ Create and use modules and packages
- ✅ Manage virtual environments

## Topics Covered

### 1. Variables & Data Types

```python
# Numbers
integer_num = 42
float_num = 3.14
complex_num = 1 + 2j

# Strings
name = "AI Infrastructure"
multiline = """This is a
multiline string"""

# Boolean
is_active = True

# Collections
my_list = [1, 2, 3, 4, 5]
my_tuple = (1, 2, 3)  # Immutable
my_dict = {"name": "Model", "version": "1.0"}
my_set = {1, 2, 3}  # Unique elements

# Type checking
print(type(integer_num))  # <class 'int'>
```

### 2. Control Flow

```python
# If/Else
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
else:
    grade = "C"

# For loops
for i in range(5):
    print(i)

for name in ["Alice", "Bob", "Charlie"]:
    print(f"Hello, {name}!")

# While loops
count = 0
while count < 5:
    print(count)
    count += 1

# Break and Continue
for i in range(10):
    if i == 3:
        continue  # Skip 3
    if i == 7:
        break  # Stop at 7
    print(i)
```

### 3. Functions

```python
# Basic function
def greet(name):
    """Greet a person by name."""
    return f"Hello, {name}!"

# Function with default arguments
def power(base, exponent=2):
    """Calculate base raised to exponent."""
    return base ** exponent

# Function with multiple return values
def get_stats(numbers):
    """Return min, max, and average of numbers."""
    return min(numbers), max(numbers), sum(numbers) / len(numbers)

# Lambda functions (anonymous)
square = lambda x: x ** 2
add = lambda x, y: x + y

# Type hints
def calculate_accuracy(correct: int, total: int) -> float:
    """Calculate accuracy percentage."""
    return (correct / total) * 100

# *args and **kwargs
def flexible_function(*args, **kwargs):
    """Accept any number of positional and keyword arguments."""
    print(f"Args: {args}")
    print(f"Kwargs: {kwargs}")
```

### 4. List Comprehensions

```python
# Basic list comprehension
squares = [x ** 2 for x in range(10)]

# With condition
evens = [x for x in range(20) if x % 2 == 0]

# Nested
matrix = [[i + j for j in range(3)] for i in range(3)]

# Dict comprehension
squared_dict = {x: x**2 for x in range(5)}

# Set comprehension
unique_lengths = {len(word) for word in ["hello", "world", "hi"]}
```

### 5. Generators

```python
# Generator function
def fibonacci(n):
    """Generate Fibonacci sequence up to n terms."""
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1

# Usage
for num in fibonacci(10):
    print(num)

# Generator expression
squared_gen = (x**2 for x in range(10))
print(next(squared_gen))  # Get next value
```

### 6. Modules & Packages

```python
# Importing modules
import math
import os
from pathlib import Path
from datetime import datetime, timedelta

# Custom imports
from my_module import my_function
from my_package.my_module import MyClass

# Import with alias
import numpy as np
import pandas as pd
```

---

## Project: System Monitor CLI

Build a command-line tool to monitor system resources.

### Requirements

**Features:**
1. Display CPU usage
2. Display memory usage
3. Display disk usage
4. Display running processes
5. Save metrics to file
6. Command-line interface

**Technical Requirements:**
- Use `psutil` library for system metrics
- Use `click` for CLI
- Format output with `rich`
- Follow PEP 8 style guide
- Include docstrings

### Implementation

See `solutions/system_monitor.py` for complete implementation.

### Example Usage

```bash
# Show all metrics
python system_monitor.py --all

# Show specific metrics
python system_monitor.py --cpu --memory

# Save to file
python system_monitor.py --all --output metrics.json

# Watch mode (refresh every 2 seconds)
python system_monitor.py --all --watch 2
```

---

## Practice Problems

### Problem 1: FizzBuzz

```python
def fizzbuzz(n: int) -> list:
    """
    Return FizzBuzz sequence from 1 to n.

    Rules:
    - Multiples of 3: "Fizz"
    - Multiples of 5: "Buzz"
    - Multiples of both: "FizzBuzz"
    - Otherwise: number as string

    Example:
        >>> fizzbuzz(15)
        ['1', '2', 'Fizz', '4', 'Buzz', 'Fizz', '7', '8',
         'Fizz', 'Buzz', '11', 'Fizz', '13', '14', 'FizzBuzz']
    """
    # Your implementation here
    pass
```

### Problem 2: Find Duplicates

```python
def find_duplicates(numbers: list) -> list:
    """
    Find all duplicate numbers in a list.

    Example:
        >>> find_duplicates([1, 2, 3, 2, 4, 5, 3])
        [2, 3]
    """
    # Your implementation here
    pass
```

### Problem 3: Word Counter

```python
def count_words(text: str) -> dict:
    """
    Count frequency of each word in text.

    Example:
        >>> count_words("hello world hello")
        {'hello': 2, 'world': 1}
    """
    # Your implementation here
    pass
```

### Problem 4: Prime Numbers

```python
def is_prime(n: int) -> bool:
    """Check if number is prime."""
    # Your implementation here
    pass

def get_primes(limit: int) -> list:
    """Get all prime numbers up to limit."""
    # Your implementation here
    pass
```

### Problem 5: List Operations

```python
def merge_sorted_lists(list1: list, list2: list) -> list:
    """
    Merge two sorted lists into one sorted list.

    Example:
        >>> merge_sorted_lists([1, 3, 5], [2, 4, 6])
        [1, 2, 3, 4, 5, 6]
    """
    # Your implementation here
    pass
```

---

## Code Examples

### Example 1: File Reader

```python
def read_file_lines(filepath: str) -> list:
    """
    Read file and return list of lines.

    Args:
        filepath: Path to file

    Returns:
        List of lines (strings)

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    try:
        with open(filepath, 'r') as f:
            return [line.strip() for line in f]
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
```

### Example 2: Data Processing

```python
def process_data(data: list[dict]) -> list[dict]:
    """
    Process and clean data.

    Removes entries with missing required fields.
    Converts numeric strings to numbers.
    """
    required_fields = ['id', 'name', 'value']

    processed = []
    for item in data:
        # Check required fields
        if not all(field in item for field in required_fields):
            continue

        # Convert types
        if isinstance(item['value'], str):
            item['value'] = float(item['value'])

        processed.append(item)

    return processed
```

### Example 3: Simple Calculator

```python
def calculator(operation: str, a: float, b: float) -> float:
    """
    Perform calculation based on operation.

    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        a: First number
        b: Second number

    Returns:
        Result of calculation

    Raises:
        ValueError: If operation is invalid
        ZeroDivisionError: If dividing by zero
    """
    operations = {
        'add': lambda x, y: x + y,
        'subtract': lambda x, y: x - y,
        'multiply': lambda x, y: x * y,
        'divide': lambda x, y: x / y if y != 0 else None,
    }

    if operation not in operations:
        raise ValueError(f"Invalid operation: {operation}")

    result = operations[operation](a, b)

    if result is None:
        raise ZeroDivisionError("Cannot divide by zero")

    return result
```

---

## Best Practices

### 1. PEP 8 Style Guide

```python
# Good
def calculate_average(numbers: list) -> float:
    """Calculate average of numbers."""
    return sum(numbers) / len(numbers)


# Bad
def calcAvg(nums):
    return sum(nums)/len(nums)
```

### 2. Meaningful Names

```python
# Good
user_count = 100
is_valid = True
calculate_total_price()

# Bad
uc = 100
flag = True
calc()
```

### 3. Constants

```python
# Good - uppercase constants
MAX_CONNECTIONS = 100
DEFAULT_TIMEOUT = 30
API_BASE_URL = "https://api.example.com"

# Usage
if connections > MAX_CONNECTIONS:
    print("Too many connections")
```

### 4. List vs Generator

```python
# List - loads all into memory
squares_list = [x**2 for x in range(1000000)]

# Generator - lazy evaluation
squares_gen = (x**2 for x in range(1000000))

# Use generator for large datasets
total = sum(x**2 for x in range(1000000))
```

---

## Common Pitfalls

### 1. Mutable Default Arguments

```python
# Bad
def append_to_list(item, my_list=[]):
    my_list.append(item)
    return my_list

# Good
def append_to_list(item, my_list=None):
    if my_list is None:
        my_list = []
    my_list.append(item)
    return my_list
```

### 2. Variable Scope

```python
# Bad - modifying global
count = 0

def increment():
    global count
    count += 1

# Good - return new value
def increment(count):
    return count + 1

# Usage
count = increment(count)
```

### 3. String Concatenation

```python
# Bad - inefficient for many strings
result = ""
for item in items:
    result += str(item)

# Good - use join
result = "".join(str(item) for item in items)

# Or f-strings
name = "Alice"
age = 30
message = f"{name} is {age} years old"
```

---

## Testing Your Knowledge

### Quiz

1. What's the difference between a list and a tuple?
2. When would you use a set instead of a list?
3. What's the difference between `==` and `is`?
4. What's a lambda function?
5. What's the purpose of `__name__ == "__main__"`?

### Coding Challenges

Complete these challenges in `examples/challenges.py`:

1. **Palindrome Checker**: Check if string is palindrome
2. **Anagram Detector**: Check if two words are anagrams
3. **List Flattener**: Flatten nested lists
4. **Dictionary Merger**: Merge multiple dictionaries
5. **Recursive Factorial**: Calculate factorial recursively

---

## Validation

Run the validation script:

```bash
python tests/test_basics.py
```

Expected output:
```
✅ All basic Python tests passed
✅ FizzBuzz implementation correct
✅ List operations working
✅ Function definitions proper
✅ Code follows PEP 8

🎉 Exercise 01 Complete!
```

---

## Resources

### Documentation
- [Python Tutorial](https://docs.python.org/3/tutorial/)
- [Python Standard Library](https://docs.python.org/3/library/)
- [PEP 8 Style Guide](https://pep8.org/)

### Practice
- [Python Exercises on HackerRank](https://www.hackerrank.com/domains/python)
- [Python Track on Exercism](https://exercism.org/tracks/python)
- [LeetCode Easy Problems](https://leetcode.com/problemset/all/?difficulty=EASY)

### Books
- "Python Crash Course" - Eric Matthes
- "Automate the Boring Stuff" - Al Sweigart

---

## Next Steps

After completing this exercise:

1. **Exercise 02: Object-Oriented Programming**
2. Practice daily on coding platforms
3. Build small automation scripts
4. Read other people's Python code

---

**Time to build a strong Python foundation! 🐍**
