#!/usr/bin/env python3
"""
JSON and YAML Processing

Tools for reading, writing, validating, and transforming JSON and YAML data.
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JSONProcessor:
    """Process JSON data."""

    @staticmethod
    def read(filepath: str, encoding: str = 'utf-8') -> Any:
        """
        Read JSON file.

        Args:
            filepath: Path to JSON file
            encoding: File encoding

        Returns:
            Parsed JSON data

        Example:
            >>> data = JSONProcessor.read('config.json')
        """
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                data = json.load(f)
            logger.info(f"Read JSON from {filepath}")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filepath}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            raise

    @staticmethod
    def write(
        filepath: str,
        data: Any,
        indent: int = 2,
        sort_keys: bool = False,
        encoding: str = 'utf-8'
    ) -> None:
        """
        Write data to JSON file.

        Args:
            filepath: Output file path
            data: Data to write
            indent: Indentation spaces
            sort_keys: Sort dictionary keys
            encoding: File encoding

        Example:
            >>> JSONProcessor.write('output.json', {'key': 'value'})
        """
        try:
            with open(filepath, 'w', encoding=encoding) as f:
                json.dump(data, f, indent=indent, sort_keys=sort_keys, ensure_ascii=False)
            logger.info(f"Wrote JSON to {filepath}")
        except Exception as e:
            logger.error(f"Failed to write {filepath}: {e}")
            raise

    @staticmethod
    def read_jsonl(filepath: str, encoding: str = 'utf-8') -> List[Dict[str, Any]]:
        """
        Read JSON Lines file (one JSON object per line).

        Args:
            filepath: Path to JSONL file
            encoding: File encoding

        Returns:
            List of JSON objects

        Example:
            >>> data = JSONProcessor.read_jsonl('data.jsonl')
        """
        data = []
        with open(filepath, 'r', encoding=encoding) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON on line {line_num}: {e}")

        logger.info(f"Read {len(data)} records from {filepath}")
        return data

    @staticmethod
    def write_jsonl(
        filepath: str,
        data: List[Dict[str, Any]],
        encoding: str = 'utf-8'
    ) -> None:
        """
        Write data to JSON Lines file.

        Args:
            filepath: Output file path
            data: List of dictionaries to write
            encoding: File encoding

        Example:
            >>> JSONProcessor.write_jsonl('output.jsonl', [{'id': 1}, {'id': 2}])
        """
        with open(filepath, 'w', encoding=encoding) as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"Wrote {len(data)} records to {filepath}")

    @staticmethod
    def pretty_print(data: Any, indent: int = 2) -> None:
        """
        Pretty print JSON data.

        Args:
            data: Data to print
            indent: Indentation spaces

        Example:
            >>> JSONProcessor.pretty_print({'key': 'value'})
        """
        print(json.dumps(data, indent=indent, ensure_ascii=False))

    @staticmethod
    def validate_structure(
        data: Dict[str, Any],
        required_keys: List[str],
        optional_keys: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate JSON structure.

        Args:
            data: Data to validate
            required_keys: Required keys
            optional_keys: Optional keys

        Returns:
            Tuple of (is_valid, list of errors)

        Example:
            >>> valid, errors = JSONProcessor.validate_structure(
            ...     {'name': 'Alice', 'age': 30},
            ...     required_keys=['name', 'age']
            ... )
        """
        errors = []

        # Check required keys
        for key in required_keys:
            if key not in data:
                errors.append(f"Missing required key: {key}")

        # Check for unexpected keys
        if optional_keys is not None:
            allowed_keys = set(required_keys + optional_keys)
            for key in data.keys():
                if key not in allowed_keys:
                    errors.append(f"Unexpected key: {key}")

        return len(errors) == 0, errors

    @staticmethod
    def merge(
        *json_dicts: Dict[str, Any],
        deep: bool = True
    ) -> Dict[str, Any]:
        """
        Merge multiple JSON dictionaries.

        Args:
            *json_dicts: Dictionaries to merge
            deep: Perform deep merge

        Returns:
            Merged dictionary

        Example:
            >>> merged = JSONProcessor.merge({'a': 1}, {'b': 2}, {'c': 3})
        """
        if not deep:
            result = {}
            for d in json_dicts:
                result.update(d)
            return result

        def deep_merge(dict1, dict2):
            result = dict1.copy()
            for key, value in dict2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        result = {}
        for d in json_dicts:
            result = deep_merge(result, d)

        return result

    @staticmethod
    def flatten(
        data: Dict[str, Any],
        separator: str = '.',
        prefix: str = ''
    ) -> Dict[str, Any]:
        """
        Flatten nested JSON structure.

        Args:
            data: Nested dictionary
            separator: Key separator
            prefix: Key prefix

        Returns:
            Flattened dictionary

        Example:
            >>> flat = JSONProcessor.flatten({'a': {'b': {'c': 1}}})
            >>> # Returns: {'a.b.c': 1}
        """
        result = {}

        for key, value in data.items():
            new_key = f"{prefix}{separator}{key}" if prefix else key

            if isinstance(value, dict):
                result.update(JSONProcessor.flatten(value, separator, new_key))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        result.update(JSONProcessor.flatten(item, separator, f"{new_key}[{i}]"))
                    else:
                        result[f"{new_key}[{i}]"] = item
            else:
                result[new_key] = value

        return result


class YAMLProcessor:
    """Process YAML data."""

    @staticmethod
    def read(filepath: str, encoding: str = 'utf-8') -> Any:
        """
        Read YAML file.

        Args:
            filepath: Path to YAML file
            encoding: File encoding

        Returns:
            Parsed YAML data

        Example:
            >>> data = YAMLProcessor.read('config.yaml')
        """
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                data = yaml.safe_load(f)
            logger.info(f"Read YAML from {filepath}")
            return data
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in {filepath}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            raise

    @staticmethod
    def write(
        filepath: str,
        data: Any,
        encoding: str = 'utf-8',
        default_flow_style: bool = False
    ) -> None:
        """
        Write data to YAML file.

        Args:
            filepath: Output file path
            data: Data to write
            encoding: File encoding
            default_flow_style: Use flow style (inline)

        Example:
            >>> YAMLProcessor.write('output.yaml', {'key': 'value'})
        """
        try:
            with open(filepath, 'w', encoding=encoding) as f:
                yaml.dump(data, f, default_flow_style=default_flow_style, allow_unicode=True)
            logger.info(f"Wrote YAML to {filepath}")
        except Exception as e:
            logger.error(f"Failed to write {filepath}: {e}")
            raise

    @staticmethod
    def read_all(filepath: str, encoding: str = 'utf-8') -> List[Any]:
        """
        Read multi-document YAML file.

        Args:
            filepath: Path to YAML file
            encoding: File encoding

        Returns:
            List of documents

        Example:
            >>> docs = YAMLProcessor.read_all('multi.yaml')
        """
        with open(filepath, 'r', encoding=encoding) as f:
            docs = list(yaml.safe_load_all(f))

        logger.info(f"Read {len(docs)} documents from {filepath}")
        return docs

    @staticmethod
    def write_all(
        filepath: str,
        documents: List[Any],
        encoding: str = 'utf-8'
    ) -> None:
        """
        Write multiple documents to YAML file.

        Args:
            filepath: Output file path
            documents: List of documents to write
            encoding: File encoding

        Example:
            >>> YAMLProcessor.write_all('multi.yaml', [{'doc': 1}, {'doc': 2}])
        """
        with open(filepath, 'w', encoding=encoding) as f:
            yaml.dump_all(documents, f, allow_unicode=True)

        logger.info(f"Wrote {len(documents)} documents to {filepath}")


class DataConverter:
    """Convert between JSON and YAML formats."""

    @staticmethod
    def json_to_yaml(json_path: str, yaml_path: str) -> None:
        """
        Convert JSON file to YAML.

        Args:
            json_path: Input JSON file
            yaml_path: Output YAML file

        Example:
            >>> DataConverter.json_to_yaml('config.json', 'config.yaml')
        """
        data = JSONProcessor.read(json_path)
        YAMLProcessor.write(yaml_path, data)
        logger.info(f"Converted {json_path} to {yaml_path}")

    @staticmethod
    def yaml_to_json(yaml_path: str, json_path: str, indent: int = 2) -> None:
        """
        Convert YAML file to JSON.

        Args:
            yaml_path: Input YAML file
            json_path: Output JSON file
            indent: JSON indentation

        Example:
            >>> DataConverter.yaml_to_json('config.yaml', 'config.json')
        """
        data = YAMLProcessor.read(yaml_path)
        JSONProcessor.write(json_path, data, indent=indent)
        logger.info(f"Converted {yaml_path} to {json_path}")

    @staticmethod
    def json_string_to_dict(json_string: str) -> Dict[str, Any]:
        """
        Parse JSON string to dictionary.

        Args:
            json_string: JSON string

        Returns:
            Parsed dictionary

        Example:
            >>> data = DataConverter.json_string_to_dict('{"key": "value"}')
        """
        return json.loads(json_string)

    @staticmethod
    def dict_to_json_string(data: Dict[str, Any], indent: Optional[int] = None) -> str:
        """
        Convert dictionary to JSON string.

        Args:
            data: Dictionary to convert
            indent: Indentation (None for compact)

        Returns:
            JSON string

        Example:
            >>> json_str = DataConverter.dict_to_json_string({'key': 'value'})
        """
        return json.dumps(data, indent=indent, ensure_ascii=False)


if __name__ == '__main__':
    print("=== JSON/YAML Processing Examples ===\n")

    import tempfile
    import os

    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = os.path.join(tmpdir, 'test.json')
        yaml_file = os.path.join(tmpdir, 'test.yaml')
        jsonl_file = os.path.join(tmpdir, 'test.jsonl')

        # Test data
        data = {
            'name': 'AI Infrastructure',
            'version': '1.0',
            'config': {
                'timeout': 30,
                'retries': 3,
                'features': ['feature1', 'feature2']
            }
        }

        print("1. JSON Processing:")
        JSONProcessor.write(json_file, data)
        loaded_json = JSONProcessor.read(json_file)
        print(f"   Written and read JSON successfully")

        print("\n2. YAML Processing:")
        YAMLProcessor.write(yaml_file, data)
        loaded_yaml = YAMLProcessor.read(yaml_file)
        print(f"   Written and read YAML successfully")

        print("\n3. JSON Lines:")
        jsonl_data = [
            {'id': 1, 'name': 'Alice'},
            {'id': 2, 'name': 'Bob'},
            {'id': 3, 'name': 'Charlie'}
        ]
        JSONProcessor.write_jsonl(jsonl_file, jsonl_data)
        loaded_jsonl = JSONProcessor.read_jsonl(jsonl_file)
        print(f"   Written and read {len(loaded_jsonl)} JSONL records")

        print("\n4. Flatten JSON:")
        flat = JSONProcessor.flatten(data)
        print(f"   Flattened keys: {list(flat.keys())}")

        print("\n5. Merge JSON:")
        dict1 = {'a': 1, 'b': {'c': 2}}
        dict2 = {'b': {'d': 3}, 'e': 4}
        merged = JSONProcessor.merge(dict1, dict2)
        print(f"   Merged: {merged}")

        print("\n6. Validate JSON:")
        valid, errors = JSONProcessor.validate_structure(
            data,
            required_keys=['name', 'version'],
            optional_keys=['config']
        )
        print(f"   Valid: {valid}, Errors: {errors}")

        print("\n7. Convert JSON to YAML:")
        yaml_output = os.path.join(tmpdir, 'converted.yaml')
        DataConverter.json_to_yaml(json_file, yaml_output)
        print(f"   Converted successfully")

    print("\n✓ JSON/YAML processing examples completed")
