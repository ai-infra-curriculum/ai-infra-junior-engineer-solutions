# Module 002: Python Programming

## Overview

Master Python programming fundamentals essential for AI Infrastructure Engineering. This module covers core Python concepts, object-oriented programming, error handling, testing, and data processing - all critical skills for building robust ML infrastructure.

## Learning Objectives

By completing this module, you will:

- ✅ Write clean, efficient Python code following PEP 8 standards
- ✅ Understand and apply object-oriented programming principles
- ✅ Handle errors gracefully and write defensive code
- ✅ Write comprehensive tests using pytest
- ✅ Process and manipulate data with pandas and numpy
- ✅ Use Python for automation and scripting
- ✅ Apply best practices for production Python code

## Module Structure

### Exercise 01: Python Basics (6-8 hours)
**Complexity:** ⭐ Easy

Master Python fundamentals: data types, control flow, functions, and modules.

**Topics:**
- Variables and data types
- Control structures (if/else, loops)
- Functions and lambda expressions
- List comprehensions and generators
- Modules and packages
- Virtual environments

**Project:** CLI tool for system monitoring

---

### Exercise 02: Object-Oriented Programming (6-8 hours)
**Complexity:** ⭐⭐ Medium

Learn OOP principles and design patterns for building scalable applications.

**Topics:**
- Classes and objects
- Inheritance and polymorphism
- Encapsulation and abstraction
- Magic methods (dunder methods)
- Decorators and property
- Design patterns

**Project:** ML model manager with caching

---

### Exercise 03: File I/O & Error Handling (4-6 hours)
**Complexity:** ⭐⭐ Medium

Handle files, configuration, logging, and exceptions professionally.

**Topics:**
- File operations (read, write, append)
- Working with JSON, YAML, CSV
- Context managers (with statements)
- Exception handling (try/except/finally)
- Custom exceptions
- Logging best practices

**Project:** Configuration management system

---

### Exercise 04: Testing with pytest (6-8 hours)
**Complexity:** ⭐⭐⭐ Hard

Write comprehensive tests to ensure code quality and reliability.

**Topics:**
- Unit testing fundamentals
- pytest framework
- Fixtures and mocking
- Parameterized tests
- Test coverage
- Test-driven development (TDD)

**Project:** Tested ML pipeline components

---

### Exercise 05: Python for Data Processing (8-10 hours)
**Complexity:** ⭐⭐⭐ Hard

Process and analyze data efficiently for ML infrastructure tasks.

**Topics:**
- NumPy arrays and operations
- Pandas DataFrames
- Data cleaning and transformation
- Reading/writing various formats
- Performance optimization
- Memory management

**Project:** ML metrics aggregation tool

---

## Prerequisites

**From Module 001:**
- Development environment set up
- Python 3.11+ installed via pyenv
- VS Code configured with Python extensions
- Git fundamentals

**Knowledge Requirements:**
- Basic programming concepts (helpful but not required)
- Willingness to practice and debug
- Growth mindset for learning

---

## Success Criteria

After completing this module, you should be able to:

✅ Write production-quality Python code with proper structure
✅ Design and implement object-oriented systems
✅ Handle errors and edge cases gracefully
✅ Write comprehensive test suites with pytest
✅ Process large datasets efficiently
✅ Debug Python code effectively
✅ Follow Python best practices (PEP 8, type hints)
✅ Use Python for ML infrastructure automation

---

## Estimated Time

**Total Module Time:** 30-40 hours

- Exercise 01: Python Basics (6-8 hours)
- Exercise 02: OOP (6-8 hours)
- Exercise 03: File I/O & Error Handling (4-6 hours)
- Exercise 04: Testing with pytest (6-8 hours)
- Exercise 05: Data Processing (8-10 hours)

---

## Key Python Concepts for AI Infrastructure

### 1. Clean Code Principles

```python
# Good: Clear, readable, documented
def calculate_model_accuracy(predictions: list, labels: list) -> float:
    """
    Calculate accuracy metric for model predictions.

    Args:
        predictions: Model prediction outputs
        labels: Ground truth labels

    Returns:
        Accuracy score between 0 and 1
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")

    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(labels)


# Bad: Unclear, no documentation
def calc(p, l):
    return sum(p[i] == l[i] for i in range(len(p))) / len(p)
```

### 2. Type Hints

```python
from typing import List, Dict, Optional, Union

def process_data(
    data: List[Dict[str, Union[int, float]]],
    threshold: float = 0.5,
    output_path: Optional[str] = None
) -> List[Dict[str, float]]:
    """Process data with type hints for clarity."""
    # Implementation
    pass
```

### 3. Error Handling

```python
class ModelNotFoundError(Exception):
    """Raised when ML model file is not found."""
    pass

def load_model(model_path: str):
    """Load model with proper error handling."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        raise ModelNotFoundError(f"Model not found: {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
```

### 4. Context Managers

```python
from contextlib import contextmanager

@contextmanager
def gpu_memory_manager():
    """Manage GPU memory allocation."""
    try:
        torch.cuda.empty_cache()
        yield
    finally:
        torch.cuda.empty_cache()

# Usage
with gpu_memory_manager():
    model.train()
```

---

## Python Tools & Libraries

### Essential Tools
- **pip** - Package installer
- **poetry** - Dependency management
- **black** - Code formatter
- **flake8** - Linter
- **mypy** - Type checker
- **pytest** - Testing framework

### Key Libraries
- **numpy** - Numerical computing
- **pandas** - Data manipulation
- **requests** - HTTP library
- **pyyaml** - YAML parsing
- **click** - CLI creation
- **rich** - Terminal formatting

### ML Infrastructure Libraries
- **fastapi** - Web APIs
- **pydantic** - Data validation
- **sqlalchemy** - Database ORM
- **redis** - Caching
- **celery** - Task queues

---

## Best Practices

### 1. Code Organization

```
project/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifier.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   └── config.py
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   └── test_utils.py
├── requirements.txt
├── setup.py
└── README.md
```

### 2. Documentation

```python
def train_model(
    data: pd.DataFrame,
    target: str,
    model_type: str = "random_forest"
) -> object:
    """
    Train machine learning model.

    Args:
        data: Training dataset
        target: Target column name
        model_type: Type of model ('random_forest', 'xgboost')

    Returns:
        Trained model object

    Raises:
        ValueError: If target column not in data

    Example:
        >>> df = pd.read_csv('data.csv')
        >>> model = train_model(df, 'price', 'xgboost')
    """
    pass
```

### 3. Error Messages

```python
# Good: Specific, actionable
if not os.path.exists(config_path):
    raise FileNotFoundError(
        f"Configuration file not found: {config_path}\n"
        f"Please create config.yaml or specify --config-path"
    )

# Bad: Vague
raise Exception("Error")
```

---

## Testing Strategy

### Test Pyramid

```
       /\
      /  \
     / E2E \ (Few)
    /------\
   /  Integ \
  /----------\
 /    Unit    \ (Many)
/--------------\
```

**Unit Tests (70%):**
- Test individual functions
- Fast execution
- Easy to debug

**Integration Tests (20%):**
- Test component interactions
- Database, API calls
- Slower but valuable

**End-to-End Tests (10%):**
- Test complete workflows
- Expensive but critical

---

## Resources

### Official Documentation
- [Python Documentation](https://docs.python.org/3/)
- [PEP 8 Style Guide](https://pep8.org/)
- [pytest Documentation](https://docs.pytest.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

### Books
- **"Python Crash Course"** by Eric Matthes
- **"Fluent Python"** by Luciano Ramalho
- **"Effective Python"** by Brett Slatkin
- **"Python Testing with pytest"** by Brian Okken

### Online Courses
- [Real Python](https://realpython.com/)
- [Python for Everybody](https://www.py4e.com/)
- [Talk Python](https://training.talkpython.fm/)

### Practice Platforms
- [LeetCode](https://leetcode.com/) - Algorithms
- [HackerRank](https://www.hackerrank.com/) - Python challenges
- [Exercism](https://exercism.org/tracks/python) - Mentored practice
- [Project Euler](https://projecteuler.net/) - Math problems

---

## Module Projects

### Project 1: System Monitor CLI
Build a command-line tool for monitoring system resources.

**Skills:** Python basics, file I/O, CLI arguments

### Project 2: ML Model Manager
Object-oriented system for managing ML models with caching.

**Skills:** OOP, design patterns, caching

### Project 3: Config Manager
Configuration management system with validation.

**Skills:** File I/O, YAML/JSON, error handling, logging

### Project 4: Tested ML Pipeline
Complete ML pipeline with comprehensive test coverage.

**Skills:** Testing, pytest, fixtures, mocking

### Project 5: Metrics Aggregator
Process and aggregate ML experiment metrics.

**Skills:** pandas, numpy, data processing, performance

---

## Getting Started

### 1. Activate Python Environment

```bash
# Using pyenv (recommended)
pyenv local 3.11.7

# Or create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
```

### 2. Install Requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start with Exercise 01

```bash
cd exercise-01-python-basics
cat README.md
```

### 4. Complete Exercises in Order

Each exercise builds on previous knowledge.

---

## Assessment

### Self-Assessment Questions

After each exercise, ask yourself:
- [ ] Can I explain the concept to someone else?
- [ ] Can I write code without looking at examples?
- [ ] Do I understand why this approach is best practice?
- [ ] Have I completed all practice problems?
- [ ] Did my code pass all tests?

### Code Review Checklist

- [ ] Follows PEP 8 style guide
- [ ] Has type hints for function signatures
- [ ] Includes docstrings for all functions
- [ ] Handles errors appropriately
- [ ] Has comprehensive tests
- [ ] Is performant and readable

---

## Next Steps

After completing this module:

1. **Module 003: Linux & Command Line** - System administration
2. Apply Python skills to real projects
3. Contribute to open-source Python projects
4. Build automation scripts for daily tasks
5. Start LeetCode Python challenges

---

## Module Completion Checklist

- [ ] Exercise 01: Python Basics completed
- [ ] Exercise 02: OOP completed
- [ ] Exercise 03: File I/O & Error Handling completed
- [ ] Exercise 04: Testing with pytest completed
- [ ] Exercise 05: Data Processing completed
- [ ] All projects functional and tested
- [ ] Code follows Python best practices
- [ ] Ready for Module 003

---

**Let's master Python for AI Infrastructure! 🐍🚀**
