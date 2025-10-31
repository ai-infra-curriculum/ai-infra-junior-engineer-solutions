#!/usr/bin/env python3
"""
File Operations and I/O

Comprehensive file handling, reading, writing, and processing operations.
"""

import os
import json
import csv
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from contextlib import contextmanager
import shutil


class FileReader:
    """File reading operations."""

    @staticmethod
    def read_entire_file(filepath: str) -> str:
        """
        Read entire file as string.

        Args:
            filepath: Path to file

        Returns:
            File contents as string

        Raises:
            FileNotFoundError: If file doesn't exist

        Example:
            >>> # content = FileReader.read_entire_file("test.txt")
            >>> pass
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def read_lines(filepath: str) -> List[str]:
        """
        Read file as list of lines.

        Args:
            filepath: Path to file

        Returns:
            List of lines (stripped of newlines)

        Example:
            >>> # lines = FileReader.read_lines("test.txt")
            >>> pass
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.rstrip('\n') for line in f]

    @staticmethod
    def read_lines_generator(filepath: str) -> Iterator[str]:
        """
        Read file line by line using generator (memory efficient).

        Args:
            filepath: Path to file

        Yields:
            Each line from file

        Example:
            >>> # for line in FileReader.read_lines_generator("large.txt"):
            >>> #     process(line)
            >>> pass
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.rstrip('\n')

    @staticmethod
    def read_binary(filepath: str) -> bytes:
        """
        Read file in binary mode.

        Args:
            filepath: Path to file

        Returns:
            File contents as bytes

        Example:
            >>> # data = FileReader.read_binary("image.png")
            >>> pass
        """
        with open(filepath, 'rb') as f:
            return f.read()


class FileWriter:
    """File writing operations."""

    @staticmethod
    def write_text(filepath: str, content: str, mode: str = 'w') -> None:
        """
        Write text to file.

        Args:
            filepath: Path to file
            content: Text to write
            mode: Write mode ('w' for overwrite, 'a' for append)

        Example:
            >>> # FileWriter.write_text("output.txt", "Hello, World!")
            >>> pass
        """
        with open(filepath, mode, encoding='utf-8') as f:
            f.write(content)

    @staticmethod
    def write_lines(filepath: str, lines: List[str], mode: str = 'w') -> None:
        """
        Write list of lines to file.

        Args:
            filepath: Path to file
            lines: List of lines to write
            mode: Write mode ('w' or 'a')

        Example:
            >>> # FileWriter.write_lines("output.txt", ["line1", "line2", "line3"])
            >>> pass
        """
        with open(filepath, mode, encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')

    @staticmethod
    def write_binary(filepath: str, data: bytes) -> None:
        """
        Write binary data to file.

        Args:
            filepath: Path to file
            data: Binary data to write

        Example:
            >>> # FileWriter.write_binary("output.bin", b"binary data")
            >>> pass
        """
        with open(filepath, 'wb') as f:
            f.write(data)


class StructuredFileHandler:
    """Handle structured file formats (JSON, CSV)."""

    @staticmethod
    def read_json(filepath: str) -> Any:
        """
        Read JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Parsed JSON data

        Example:
            >>> # data = StructuredFileHandler.read_json("config.json")
            >>> pass
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def write_json(filepath: str, data: Any, indent: int = 2) -> None:
        """
        Write data to JSON file.

        Args:
            filepath: Path to output file
            data: Data to write
            indent: Indentation level

        Example:
            >>> # StructuredFileHandler.write_json("output.json", {"key": "value"})
            >>> pass
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

    @staticmethod
    def read_csv(filepath: str, has_header: bool = True) -> List[Dict[str, str]]:
        """
        Read CSV file as list of dictionaries.

        Args:
            filepath: Path to CSV file
            has_header: Whether CSV has header row

        Returns:
            List of dictionaries (one per row)

        Example:
            >>> # data = StructuredFileHandler.read_csv("data.csv")
            >>> pass
        """
        with open(filepath, 'r', encoding='utf-8', newline='') as f:
            if has_header:
                reader = csv.DictReader(f)
                return list(reader)
            else:
                reader = csv.reader(f)
                return [{'col' + str(i): val for i, val in enumerate(row)} for row in reader]

    @staticmethod
    def write_csv(filepath: str, data: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
        """
        Write data to CSV file.

        Args:
            filepath: Path to output file
            data: List of dictionaries to write
            fieldnames: Column names (auto-detected if None)

        Example:
            >>> # StructuredFileHandler.write_csv("output.csv", [{"name": "Alice", "age": 30}])
            >>> pass
        """
        if not data:
            return

        if fieldnames is None:
            fieldnames = list(data[0].keys())

        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)


class FileUtilities:
    """File system utilities."""

    @staticmethod
    def file_exists(filepath: str) -> bool:
        """Check if file exists."""
        return Path(filepath).exists()

    @staticmethod
    def get_file_size(filepath: str) -> int:
        """
        Get file size in bytes.

        Args:
            filepath: Path to file

        Returns:
            File size in bytes

        Example:
            >>> # size = FileUtilities.get_file_size("test.txt")
            >>> pass
        """
        return Path(filepath).stat().st_size

    @staticmethod
    def get_file_extension(filepath: str) -> str:
        """
        Get file extension.

        Args:
            filepath: Path to file

        Returns:
            File extension (including dot)

        Example:
            >>> FileUtilities.get_file_extension("test.txt")
            '.txt'
        """
        return Path(filepath).suffix

    @staticmethod
    def list_directory(dirpath: str, pattern: str = '*') -> List[str]:
        """
        List files in directory.

        Args:
            dirpath: Path to directory
            pattern: Glob pattern for filtering

        Returns:
            List of file paths

        Example:
            >>> # files = FileUtilities.list_directory(".", "*.py")
            >>> pass
        """
        return [str(p) for p in Path(dirpath).glob(pattern)]

    @staticmethod
    def create_directory(dirpath: str, exist_ok: bool = True) -> None:
        """
        Create directory (and parents if needed).

        Args:
            dirpath: Path to directory
            exist_ok: Don't raise error if directory exists

        Example:
            >>> # FileUtilities.create_directory("output/results")
            >>> pass
        """
        Path(dirpath).mkdir(parents=True, exist_ok=exist_ok)

    @staticmethod
    def copy_file(src: str, dst: str) -> None:
        """
        Copy file from source to destination.

        Args:
            src: Source file path
            dst: Destination file path

        Example:
            >>> # FileUtilities.copy_file("source.txt", "backup.txt")
            >>> pass
        """
        shutil.copy2(src, dst)

    @staticmethod
    def move_file(src: str, dst: str) -> None:
        """
        Move file from source to destination.

        Args:
            src: Source file path
            dst: Destination file path

        Example:
            >>> # FileUtilities.move_file("temp.txt", "archive/temp.txt")
            >>> pass
        """
        shutil.move(src, dst)

    @staticmethod
    def delete_file(filepath: str) -> None:
        """
        Delete file.

        Args:
            filepath: Path to file

        Example:
            >>> # FileUtilities.delete_file("temp.txt")
            >>> pass
        """
        Path(filepath).unlink(missing_ok=True)


class TextFileProcessor:
    """Text file processing utilities."""

    @staticmethod
    def count_lines(filepath: str) -> int:
        """
        Count lines in file.

        Args:
            filepath: Path to file

        Returns:
            Number of lines

        Example:
            >>> # count = TextFileProcessor.count_lines("test.txt")
            >>> pass
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    @staticmethod
    def count_words(filepath: str) -> int:
        """
        Count words in file.

        Args:
            filepath: Path to file

        Returns:
            Number of words

        Example:
            >>> # count = TextFileProcessor.count_words("test.txt")
            >>> pass
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(len(line.split()) for line in f)

    @staticmethod
    def grep(filepath: str, pattern: str, case_sensitive: bool = True) -> List[str]:
        """
        Search for pattern in file (like grep).

        Args:
            filepath: Path to file
            pattern: Pattern to search for
            case_sensitive: Whether search is case-sensitive

        Returns:
            List of matching lines

        Example:
            >>> # matches = TextFileProcessor.grep("log.txt", "ERROR")
            >>> pass
        """
        matches = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                search_line = line if case_sensitive else line.lower()
                search_pattern = pattern if case_sensitive else pattern.lower()

                if search_pattern in search_line:
                    matches.append(line.rstrip('\n'))

        return matches

    @staticmethod
    def replace_in_file(filepath: str, old: str, new: str, output_path: Optional[str] = None) -> int:
        """
        Replace text in file.

        Args:
            filepath: Path to input file
            old: Text to replace
            new: Replacement text
            output_path: Output file (overwrites input if None)

        Returns:
            Number of replacements made

        Example:
            >>> # count = TextFileProcessor.replace_in_file("test.txt", "old", "new")
            >>> pass
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        new_content = content.replace(old, new)
        count = content.count(old)

        output = output_path or filepath
        with open(output, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return count


@contextmanager
def temporary_file(suffix: str = '', prefix: str = 'tmp', text: bool = True):
    """
    Context manager for temporary file.

    Args:
        suffix: File suffix
        prefix: File prefix
        text: Whether to open in text mode

    Yields:
        Temporary file path

    Example:
        >>> with temporary_file(suffix='.txt') as temp_path:
        ...     # Use temp_path
        ...     pass
    """
    mode = 'w+' if text else 'w+b'
    temp = tempfile.NamedTemporaryFile(mode=mode, suffix=suffix, prefix=prefix, delete=False)
    temp_path = temp.name
    temp.close()

    try:
        yield temp_path
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@contextmanager
def atomic_write(filepath: str, mode: str = 'w', encoding: str = 'utf-8'):
    """
    Context manager for atomic file writes.

    Writes to temporary file, then moves to target on success.

    Args:
        filepath: Target file path
        mode: Write mode
        encoding: Text encoding

    Yields:
        File object to write to

    Example:
        >>> with atomic_write('config.json') as f:
        ...     json.dump(data, f)
    """
    temp_path = filepath + '.tmp'

    try:
        with open(temp_path, mode, encoding=encoding) as f:
            yield f

        # Atomic move on success
        shutil.move(temp_path, filepath)
    except:
        # Clean up on failure
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


if __name__ == '__main__':
    print("=== File Operations Examples ===\n")

    # Create a temporary directory for demonstrations
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Write and read text file
        print("Text File Operations:")
        test_file = temp_path / "test.txt"
        FileWriter.write_lines(str(test_file), ["Line 1", "Line 2", "Line 3"])
        lines = FileReader.read_lines(str(test_file))
        print(f"  Written and read {len(lines)} lines")

        # JSON operations
        print("\nJSON Operations:")
        json_file = temp_path / "data.json"
        data = {"name": "Alice", "age": 30, "skills": ["Python", "ML"]}
        StructuredFileHandler.write_json(str(json_file), data)
        loaded_data = StructuredFileHandler.read_json(str(json_file))
        print(f"  Written and read: {loaded_data}")

        # CSV operations
        print("\nCSV Operations:")
        csv_file = temp_path / "data.csv"
        csv_data = [
            {"name": "Alice", "age": "30", "city": "NYC"},
            {"name": "Bob", "age": "25", "city": "LA"},
        ]
        StructuredFileHandler.write_csv(str(csv_file), csv_data)
        loaded_csv = StructuredFileHandler.read_csv(str(csv_file))
        print(f"  Written and read {len(loaded_csv)} rows")

        # File utilities
        print("\nFile Utilities:")
        print(f"  File exists: {FileUtilities.file_exists(str(test_file))}")
        print(f"  File size: {FileUtilities.get_file_size(str(test_file))} bytes")
        print(f"  Extension: {FileUtilities.get_file_extension(str(test_file))}")

        # Text processing
        print("\nText Processing:")
        print(f"  Line count: {TextFileProcessor.count_lines(str(test_file))}")
        print(f"  Word count: {TextFileProcessor.count_words(str(test_file))}")

    print("\n✓ All file operations completed successfully")
