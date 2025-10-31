#!/usr/bin/env python3
"""
Comprehensive Test Suite for Exercise 01: Python Basics

Tests for all Python basics modules including data structures, functions,
strings, file operations, error handling, and context managers.
"""

import pytest
import sys
import os
import tempfile
import json
from pathlib import Path

# Add solutions directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solutions'))

from data_structures import (
    DataStructureExamples,
    AdvancedCollections,
    ListComprehensions,
    merge_dicts,
    flatten_list,
    find_duplicates,
    remove_duplicates_preserve_order,
    chunk_list,
    transpose_matrix,
)

from functions_decorators import (
    calculate_stats,
    fibonacci,
    fibonacci_generator,
    apply_operation,
    filter_data,
    timing_decorator,
    cache_decorator,
    make_multiplier,
    make_counter,
    compose,
)

from string_manipulation import (
    StringOperations,
    StringFormatting,
    TextProcessing,
    StringValidation,
    parse_key_value_pairs,
)

from file_operations import (
    FileReader,
    FileWriter,
    StructuredFileHandler,
    FileUtilities,
    TextFileProcessor,
)

from error_handling import (
    ValidationError,
    safe_divide,
    safe_int_conversion,
    safe_list_access,
    validate_age,
    validate_email,
    ResourceManager,
)

from context_managers import (
    Timer,
    TransactionManager,
    managed_resource,
)


# ============= Data Structures Tests =============

class TestDataStructures:
    """Test data structure operations."""

    def test_list_operations(self):
        """Test list operations."""
        ops = DataStructureExamples()
        result = ops.list_operations([5, 2, 8, 1, 9])

        assert result['length'] == 5
        assert result['first'] == 5
        assert result['last'] == 9
        assert result['reverse'] == [9, 1, 8, 2, 5]

    def test_dict_operations(self):
        """Test dictionary operations."""
        ops = DataStructureExamples()
        result = ops.dict_operations({'a': 1, 'b': 2, 'c': 3})

        assert result['key_count'] == 3
        assert 'a' in result['keys']
        assert 1 in result['values']

    def test_set_operations(self):
        """Test set operations."""
        ops = DataStructureExamples()
        result = ops.set_operations({1, 2, 3}, {2, 3, 4})

        assert result['union'] == {1, 2, 3, 4}
        assert result['intersection'] == {2, 3}
        assert result['difference'] == {1}

    def test_count_frequencies(self):
        """Test frequency counting."""
        result = AdvancedCollections.count_frequencies([1, 2, 2, 3, 3, 3])

        assert result[1] == 1
        assert result[2] == 2
        assert result[3] == 3

    def test_group_by_key(self):
        """Test grouping by key."""
        items = [
            {'type': 'a', 'value': 1},
            {'type': 'b', 'value': 2},
            {'type': 'a', 'value': 3},
        ]
        result = AdvancedCollections.group_by_key(items, 'type')

        assert len(result['a']) == 2
        assert len(result['b']) == 1

    def test_sliding_window(self):
        """Test sliding window operation."""
        result = AdvancedCollections.sliding_window([1, 2, 3, 4, 5], 3)

        assert len(result) == 3
        assert result[0] == [1, 2, 3]
        assert result[2] == [3, 4, 5]

    def test_list_comprehensions(self):
        """Test list comprehensions."""
        result = ListComprehensions.basic_comprehensions(5)

        assert result['squares'] == [0, 1, 4, 9, 16]
        assert result['evens'] == [0, 2, 4]

    def test_merge_dicts(self):
        """Test dictionary merging."""
        result = merge_dicts({'a': 1}, {'b': 2}, {'a': 3})

        assert result['a'] == 3  # Last value wins
        assert result['b'] == 2

    def test_flatten_list(self):
        """Test list flattening."""
        result = flatten_list([1, [2, 3], [4, [5, 6]]])

        assert result == [1, 2, 3, 4, 5, 6]

    def test_find_duplicates(self):
        """Test finding duplicates."""
        result = find_duplicates([1, 2, 3, 2, 4, 3, 5])

        assert set(result) == {2, 3}

    def test_remove_duplicates_preserve_order(self):
        """Test removing duplicates while preserving order."""
        result = remove_duplicates_preserve_order([1, 2, 3, 2, 4, 1, 5])

        assert result == [1, 2, 3, 4, 5]

    def test_chunk_list(self):
        """Test list chunking."""
        result = chunk_list([1, 2, 3, 4, 5, 6, 7], 3)

        assert len(result) == 3
        assert result[0] == [1, 2, 3]
        assert result[2] == [7]

    def test_transpose_matrix(self):
        """Test matrix transposition."""
        result = transpose_matrix([[1, 2, 3], [4, 5, 6]])

        assert result == [[1, 4], [2, 5], [3, 6]]


# ============= Functions and Decorators Tests =============

class TestFunctionsDecorators:
    """Test functions and decorators."""

    def test_calculate_stats(self):
        """Test statistics calculation."""
        result = calculate_stats([1, 2, 3, 4, 5])

        assert result['min'] == 1
        assert result['max'] == 5
        assert result['mean'] == 3.0
        assert result['median'] == 3.0

    def test_calculate_stats_empty_list(self):
        """Test statistics with empty list."""
        with pytest.raises(ValueError):
            calculate_stats([])

    def test_fibonacci(self):
        """Test Fibonacci sequence."""
        result = fibonacci(7)

        assert result == [0, 1, 1, 2, 3, 5, 8]

    def test_fibonacci_generator(self):
        """Test Fibonacci generator."""
        result = list(fibonacci_generator(5))

        assert result == [0, 1, 1, 2, 3]

    def test_apply_operation(self):
        """Test applying operation to list."""
        result = apply_operation([1, 2, 3], lambda x: x ** 2)

        assert result == [1, 4, 9]

    def test_filter_data(self):
        """Test filtering data."""
        result = filter_data([1, 2, 3, 4, 5], lambda x: x % 2 == 0)

        assert result == [2, 4]

    def test_cache_decorator(self):
        """Test caching decorator."""
        call_count = [0]

        @cache_decorator
        def expensive_func(n):
            call_count[0] += 1
            return n ** 2

        # First call
        result1 = expensive_func(5)
        assert result1 == 25
        assert call_count[0] == 1

        # Second call (should be cached)
        result2 = expensive_func(5)
        assert result2 == 25
        assert call_count[0] == 1  # Not called again

    def test_make_multiplier(self):
        """Test multiplier closure."""
        times_three = make_multiplier(3)

        assert times_three(4) == 12
        assert times_three(5) == 15

    def test_make_counter(self):
        """Test counter closure."""
        counter = make_counter(10)

        assert counter() == 11
        assert counter() == 12
        assert counter() == 13

    def test_compose(self):
        """Test function composition."""
        add_one = lambda x: x + 1
        double = lambda x: x * 2

        f = compose(add_one, double)
        result = f(5)

        assert result == 11  # (5 * 2) + 1


# ============= String Manipulation Tests =============

class TestStringManipulation:
    """Test string manipulation operations."""

    def test_basic_operations(self):
        """Test basic string operations."""
        ops = StringOperations()
        result = ops.basic_operations("  Hello World  ")

        assert result['stripped'] == "Hello World"
        assert result['uppercase'] == "  HELLO WORLD  "
        assert result['length'] == 15

    def test_split_and_join(self):
        """Test split and join operations."""
        ops = StringOperations()
        result = ops.split_and_join("a,b,c", ",")

        assert result['split'] == ['a', 'b', 'c']
        assert result['join_with_dash'] == 'a-b-c'

    def test_word_frequency(self):
        """Test word frequency counting."""
        result = TextProcessing.word_frequency("hello world hello python")

        assert result['hello'] == 2
        assert result['world'] == 1
        assert result['python'] == 1

    def test_extract_emails(self):
        """Test email extraction."""
        text = "Contact alice@example.com or bob@test.org"
        result = TextProcessing.extract_emails(text)

        assert len(result) == 2
        assert "alice@example.com" in result

    def test_is_palindrome(self):
        """Test palindrome detection."""
        assert TextProcessing.is_palindrome("A man a plan a canal Panama")
        assert not TextProcessing.is_palindrome("hello")

    def test_reverse_words(self):
        """Test word reversal."""
        result = TextProcessing.reverse_words("Hello world from Python")

        assert result == "Python from world Hello"

    def test_truncate_text(self):
        """Test text truncation."""
        result = TextProcessing.truncate_text("This is a long sentence", 10)

        assert len(result) <= 10
        assert result.endswith('...')

    def test_validate_email(self):
        """Test email validation."""
        validator = StringValidation()

        assert validator.validate_email("test@example.com")
        assert not validator.validate_email("invalid")
        assert not validator.validate_email("no-at-sign.com")

    def test_validate_ip(self):
        """Test IP validation."""
        validator = StringValidation()

        assert validator.is_valid_ip("192.168.1.1")
        assert validator.is_valid_ip("8.8.8.8")
        assert not validator.is_valid_ip("256.1.1.1")
        assert not validator.is_valid_ip("invalid")

    def test_parse_key_value_pairs(self):
        """Test key-value pair parsing."""
        result = parse_key_value_pairs("name=Alice,age=30,city=NYC")

        assert result['name'] == 'Alice'
        assert result['age'] == '30'
        assert result['city'] == 'NYC'


# ============= File Operations Tests =============

class TestFileOperations:
    """Test file operations."""

    def test_write_and_read_text(self):
        """Test writing and reading text files."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_path = f.name

        try:
            # Write
            FileWriter.write_text(temp_path, "Hello, World!")

            # Read
            content = FileReader.read_entire_file(temp_path)
            assert content == "Hello, World!"

        finally:
            os.unlink(temp_path)

    def test_write_and_read_lines(self):
        """Test writing and reading lines."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_path = f.name

        try:
            lines = ["Line 1", "Line 2", "Line 3"]
            FileWriter.write_lines(temp_path, lines)

            read_lines = FileReader.read_lines(temp_path)
            assert read_lines == lines

        finally:
            os.unlink(temp_path)

    def test_json_operations(self):
        """Test JSON read/write."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            data = {"name": "Alice", "age": 30, "skills": ["Python", "ML"]}

            StructuredFileHandler.write_json(temp_path, data)
            loaded = StructuredFileHandler.read_json(temp_path)

            assert loaded == data

        finally:
            os.unlink(temp_path)

    def test_csv_operations(self):
        """Test CSV read/write."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_path = f.name

        try:
            data = [
                {"name": "Alice", "age": "30"},
                {"name": "Bob", "age": "25"},
            ]

            StructuredFileHandler.write_csv(temp_path, data)
            loaded = StructuredFileHandler.read_csv(temp_path)

            assert len(loaded) == 2
            assert loaded[0]['name'] == 'Alice'

        finally:
            os.unlink(temp_path)

    def test_file_utilities(self):
        """Test file utility functions."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_path = f.name
            f.write("test content")

        try:
            assert FileUtilities.file_exists(temp_path)
            assert FileUtilities.get_file_size(temp_path) > 0
            assert FileUtilities.get_file_extension(temp_path) == '.txt'

        finally:
            os.unlink(temp_path)


# ============= Error Handling Tests =============

class TestErrorHandling:
    """Test error handling functionality."""

    def test_safe_divide(self):
        """Test safe division."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) is None

    def test_safe_int_conversion(self):
        """Test safe integer conversion."""
        assert safe_int_conversion("123") == 123
        assert safe_int_conversion("abc") is None

    def test_safe_list_access(self):
        """Test safe list access."""
        lst = [1, 2, 3]

        assert safe_list_access(lst, 1) == 2
        assert safe_list_access(lst, 10, default=-1) == -1

    def test_validate_age(self):
        """Test age validation."""
        assert validate_age(25) == 25

        with pytest.raises(ValidationError):
            validate_age(-5)

        with pytest.raises(ValidationError):
            validate_age(200)

    def test_validate_email(self):
        """Test email validation."""
        assert validate_email("test@example.com") == "test@example.com"

        with pytest.raises(ValidationError):
            validate_email("invalid")

    def test_resource_manager(self):
        """Test resource management."""
        manager = ResourceManager("test")
        result = manager.use_resource()

        assert "Resource_test" in result
        assert manager.resource is None  # Should be released


# ============= Context Managers Tests =============

class TestContextManagers:
    """Test context managers."""

    def test_timer(self):
        """Test timer context manager."""
        import time

        with Timer("test") as timer:
            time.sleep(0.01)

        assert timer.elapsed > 0

    def test_transaction_manager(self):
        """Test transaction manager."""
        with TransactionManager() as txn:
            txn.add_operation("op1")
            txn.add_operation("op2")

        assert txn.committed
        assert len(txn.operations) == 0

    def test_transaction_rollback(self):
        """Test transaction rollback on error."""
        txn = TransactionManager()

        try:
            with txn:
                txn.add_operation("op1")
                raise ValueError("Test error")
        except ValueError:
            pass

        assert not txn.committed
        assert len(txn.operations) == 0

    def test_managed_resource(self):
        """Test managed resource."""
        with managed_resource("test_resource") as resource:
            assert resource["active"] is True
            assert resource["name"] == "test_resource"

        assert resource["active"] is False


# ============= Integration Tests =============

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_data_pipeline(self):
        """Test complete data processing pipeline."""
        # Generate data
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Filter evens
        evens = filter_data(data, lambda x: x % 2 == 0)
        assert evens == [2, 4, 6, 8, 10]

        # Square them
        squares = apply_operation(evens, lambda x: x ** 2)
        assert squares == [4, 16, 36, 64, 100]

        # Calculate stats
        stats = calculate_stats(squares)
        assert stats['min'] == 4
        assert stats['max'] == 100

    def test_file_processing_pipeline(self):
        """Test file processing with error handling."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_path = f.name

        try:
            # Write data
            data = ["hello world", "python programming", "data processing"]
            FileWriter.write_lines(temp_path, data)

            # Read and process
            lines = FileReader.read_lines(temp_path)
            assert len(lines) == 3

            # Count words
            total_words = sum(len(line.split()) for line in lines)
            assert total_words == 6

        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
