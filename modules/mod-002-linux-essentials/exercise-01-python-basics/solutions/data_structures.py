#!/usr/bin/env python3
"""
Data Structures Examples and Operations

Comprehensive examples of Python data structures: lists, tuples, sets, dictionaries.
Includes common operations, transformations, and best practices.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, Counter, deque
import json


class DataStructureExamples:
    """Examples and operations for Python data structures."""

    @staticmethod
    def list_operations(items: List[Any]) -> Dict[str, Any]:
        """
        Demonstrate various list operations.

        Args:
            items: List of items to operate on

        Returns:
            Dictionary containing results of various operations

        Example:
            >>> ops = DataStructureExamples()
            >>> result = ops.list_operations([1, 2, 3, 4, 5])
            >>> result['length']
            5
        """
        if not items:
            return {'length': 0, 'operations': []}

        results = {
            'length': len(items),
            'first': items[0],
            'last': items[-1],
            'slice_first_three': items[:3],
            'reverse': items[::-1],
            'sorted': sorted(items) if all(isinstance(x, (int, float, str)) for x in items) else items,
            'unique_count': len(set(items)),
            'sum': sum(items) if all(isinstance(x, (int, float)) for x in items) else None,
        }

        return results

    @staticmethod
    def dict_operations(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Demonstrate dictionary operations.

        Args:
            data: Dictionary to operate on

        Returns:
            Dictionary containing operation results

        Example:
            >>> ops = DataStructureExamples()
            >>> result = ops.dict_operations({'a': 1, 'b': 2})
            >>> result['keys']
            ['a', 'b']
        """
        return {
            'keys': list(data.keys()),
            'values': list(data.values()),
            'items': list(data.items()),
            'key_count': len(data),
            'has_key_a': 'a' in data,
            'get_with_default': data.get('missing', 'default_value'),
        }

    @staticmethod
    def set_operations(set1: Set[Any], set2: Set[Any]) -> Dict[str, Set[Any]]:
        """
        Demonstrate set operations.

        Args:
            set1: First set
            set2: Second set

        Returns:
            Dictionary with results of set operations

        Example:
            >>> ops = DataStructureExamples()
            >>> result = ops.set_operations({1, 2, 3}, {2, 3, 4})
            >>> result['union']
            {1, 2, 3, 4}
        """
        return {
            'union': set1 | set2,
            'intersection': set1 & set2,
            'difference': set1 - set2,
            'symmetric_difference': set1 ^ set2,
            'is_subset': set1.issubset(set2),
            'is_superset': set1.issuperset(set2),
        }

    @staticmethod
    def tuple_operations(data: Tuple[Any, ...]) -> Dict[str, Any]:
        """
        Demonstrate tuple operations (immutable sequences).

        Args:
            data: Tuple to operate on

        Returns:
            Dictionary with tuple operation results

        Example:
            >>> ops = DataStructureExamples()
            >>> result = ops.tuple_operations((1, 2, 3, 2))
            >>> result['count_of_2']
            2
        """
        return {
            'length': len(data),
            'count_of_2': data.count(2) if 2 in data else 0,
            'index_of_first': data.index(data[0]) if data else None,
            'can_be_dict_key': True,  # Tuples are hashable
            'unpacking_example': data if len(data) == 3 else None,
        }


class AdvancedCollections:
    """Advanced collection operations using collections module."""

    @staticmethod
    def count_frequencies(items: List[Any]) -> Dict[Any, int]:
        """
        Count frequency of items using Counter.

        Args:
            items: List of items to count

        Returns:
            Dictionary mapping items to their counts

        Example:
            >>> AdvancedCollections.count_frequencies([1, 2, 2, 3, 3, 3])
            {3: 3, 2: 2, 1: 1}
        """
        counter = Counter(items)
        return dict(counter.most_common())

    @staticmethod
    def group_by_key(items: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group items by a specific key using defaultdict.

        Args:
            items: List of dictionaries
            key: Key to group by

        Returns:
            Dictionary mapping key values to lists of items

        Example:
            >>> items = [{'type': 'a', 'val': 1}, {'type': 'a', 'val': 2}]
            >>> AdvancedCollections.group_by_key(items, 'type')
            {'a': [{'type': 'a', 'val': 1}, {'type': 'a', 'val': 2}]}
        """
        grouped = defaultdict(list)
        for item in items:
            if key in item:
                grouped[item[key]].append(item)
        return dict(grouped)

    @staticmethod
    def sliding_window(items: List[Any], window_size: int) -> List[List[Any]]:
        """
        Create sliding windows over a list using deque.

        Args:
            items: List of items
            window_size: Size of sliding window

        Returns:
            List of windows

        Example:
            >>> AdvancedCollections.sliding_window([1, 2, 3, 4, 5], 3)
            [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        """
        if window_size > len(items):
            return []

        windows = []
        window = deque(maxlen=window_size)

        for item in items:
            window.append(item)
            if len(window) == window_size:
                windows.append(list(window))

        return windows


class ListComprehensions:
    """Examples of list comprehensions and generator expressions."""

    @staticmethod
    def basic_comprehensions(n: int) -> Dict[str, List[int]]:
        """
        Demonstrate basic list comprehensions.

        Args:
            n: Upper limit for range

        Returns:
            Dictionary with various comprehension results

        Example:
            >>> ListComprehensions.basic_comprehensions(5)['squares']
            [0, 1, 4, 9, 16]
        """
        return {
            'squares': [x**2 for x in range(n)],
            'evens': [x for x in range(n) if x % 2 == 0],
            'tuples': [(x, x**2) for x in range(n)],
            'flattened': [item for sublist in [[1, 2], [3, 4]] for item in sublist],
        }

    @staticmethod
    def dict_comprehensions(keys: List[str], values: List[int]) -> Dict[str, Any]:
        """
        Demonstrate dictionary comprehensions.

        Args:
            keys: List of keys
            values: List of values

        Returns:
            Dictionary of comprehension examples

        Example:
            >>> ListComprehensions.dict_comprehensions(['a', 'b'], [1, 2])
            {'key_value_pairs': {'a': 1, 'b': 2}, ...}
        """
        return {
            'key_value_pairs': {k: v for k, v in zip(keys, values)},
            'squared_dict': {x: x**2 for x in range(5)},
            'filtered_dict': {k: v for k, v in zip(keys, values) if v > 0},
        }

    @staticmethod
    def set_comprehensions(items: List[int]) -> Set[int]:
        """
        Demonstrate set comprehensions.

        Args:
            items: List of items

        Returns:
            Set of unique squared values

        Example:
            >>> ListComprehensions.set_comprehensions([1, 2, 2, 3])
            {1, 4, 9}
        """
        return {x**2 for x in items}

    @staticmethod
    def generator_expression(n: int) -> int:
        """
        Demonstrate generator expressions for memory efficiency.

        Args:
            n: Upper limit

        Returns:
            Sum of squares (computed lazily)

        Example:
            >>> ListComprehensions.generator_expression(5)
            30
        """
        # Generator expression - memory efficient
        return sum(x**2 for x in range(n))


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries.

    Args:
        *dicts: Variable number of dictionaries to merge

    Returns:
        Merged dictionary (later values override earlier ones)

    Example:
        >>> merge_dicts({'a': 1}, {'b': 2}, {'a': 3})
        {'a': 3, 'b': 2}
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def flatten_list(nested_list: List[Any]) -> List[Any]:
    """
    Flatten a nested list structure.

    Args:
        nested_list: List that may contain nested lists

    Returns:
        Flattened list

    Example:
        >>> flatten_list([1, [2, 3], [4, [5, 6]]])
        [1, 2, 3, 4, 5, 6]
    """
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def find_duplicates(items: List[Any]) -> List[Any]:
    """
    Find duplicate items in a list.

    Args:
        items: List to check for duplicates

    Returns:
        List of duplicate items (unique)

    Example:
        >>> find_duplicates([1, 2, 3, 2, 4, 3, 5])
        [2, 3]
    """
    seen = set()
    duplicates = set()

    for item in items:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)

    return list(duplicates)


def remove_duplicates_preserve_order(items: List[Any]) -> List[Any]:
    """
    Remove duplicates while preserving order.

    Args:
        items: List with potential duplicates

    Returns:
        List with duplicates removed, order preserved

    Example:
        >>> remove_duplicates_preserve_order([1, 2, 3, 2, 4, 1])
        [1, 2, 3, 4]
    """
    seen = set()
    result = []

    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.

    Args:
        items: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks

    Example:
        >>> chunk_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def transpose_matrix(matrix: List[List[Any]]) -> List[List[Any]]:
    """
    Transpose a matrix (swap rows and columns).

    Args:
        matrix: 2D list representing a matrix

    Returns:
        Transposed matrix

    Example:
        >>> transpose_matrix([[1, 2, 3], [4, 5, 6]])
        [[1, 4], [2, 5], [3, 6]]
    """
    if not matrix or not matrix[0]:
        return []

    return [list(row) for row in zip(*matrix)]


if __name__ == '__main__':
    # Demonstrate usage
    print("=== Data Structure Examples ===\n")

    # List operations
    ops = DataStructureExamples()
    list_result = ops.list_operations([5, 2, 8, 1, 9, 3])
    print("List Operations:")
    print(json.dumps(list_result, indent=2, default=str))

    # Set operations
    set_result = ops.set_operations({1, 2, 3, 4}, {3, 4, 5, 6})
    print("\nSet Operations:")
    print(json.dumps({k: list(v) if isinstance(v, set) else v for k, v in set_result.items()}, indent=2))

    # Advanced collections
    print("\nFrequency Counting:")
    freq = AdvancedCollections.count_frequencies(['a', 'b', 'a', 'c', 'b', 'a'])
    print(json.dumps(freq, indent=2))

    # List comprehensions
    print("\nList Comprehensions:")
    comp_result = ListComprehensions.basic_comprehensions(10)
    print(json.dumps(comp_result, indent=2))

    # Utility functions
    print("\nUtility Functions:")
    print(f"Flatten: {flatten_list([1, [2, 3], [4, [5, 6]]])}")
    print(f"Find duplicates: {find_duplicates([1, 2, 3, 2, 4, 3, 5])}")
    print(f"Chunk list: {chunk_list(list(range(10)), 3)}")
