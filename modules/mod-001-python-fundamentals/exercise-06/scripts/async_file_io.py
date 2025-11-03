#!/usr/bin/env python3
"""
Async File I/O Operations

Demonstrates non-blocking file operations with aiofiles.
"""

import asyncio
import time
from pathlib import Path
from typing import List, Dict
import csv
import io

try:
    import aiofiles
except ImportError:
    print("Error: aiofiles not installed")
    print("Install with: pip install aiofiles")
    exit(1)


async def read_file_async(filepath: str) -> str:
    """
    Read file asynchronously.

    Args:
        filepath: Path to file

    Returns:
        File contents
    """
    async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
        content = await f.read()
    return content


async def write_file_async(filepath: str, content: str) -> None:
    """
    Write file asynchronously.

    Args:
        filepath: Path to file
        content: Content to write
    """
    async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
        await f.write(content)


async def read_multiple_files(filepaths: List[str]) -> Dict[str, str]:
    """
    Read multiple files concurrently.

    Args:
        filepaths: List of file paths

    Returns:
        Dictionary mapping filepath to content
    """
    async def read_one(path: str) -> tuple:
        try:
            content = await read_file_async(path)
            return path, content
        except FileNotFoundError:
            return path, None
        except Exception as e:
            return path, f"Error: {e}"

    tasks = [read_one(path) for path in filepaths]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter successful reads
    successful = {}
    for result in results:
        if isinstance(result, tuple):
            path, content = result
            if content is not None and not str(content).startswith("Error:"):
                successful[path] = content

    return successful


async def write_multiple_files(file_data: Dict[str, str]) -> None:
    """
    Write multiple files concurrently.

    Args:
        file_data: Dictionary mapping filepath to content
    """
    tasks = [write_file_async(path, content) for path, content in file_data.items()]
    await asyncio.gather(*tasks)


async def process_csv_async(filepath: str) -> List[Dict]:
    """
    Process CSV file asynchronously.

    Args:
        filepath: Path to CSV file

    Returns:
        List of dictionaries (one per row)
    """
    async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
        content = await f.read()

    # Parse CSV
    if not content.strip():
        return []

    reader = csv.DictReader(io.StringIO(content))
    return list(reader)


async def save_predictions_async(filepath: str, predictions: List[Dict]) -> None:
    """
    Save predictions asynchronously as CSV.

    Args:
        filepath: Output file path
        predictions: List of prediction dictionaries
    """
    if not predictions:
        return

    # Convert to CSV string
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=predictions[0].keys())
    writer.writeheader()
    writer.writerows(predictions)

    # Write asynchronously
    async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
        await f.write(output.getvalue())


async def append_to_file_async(filepath: str, line: str) -> None:
    """
    Append line to file asynchronously.

    Args:
        filepath: Path to file
        line: Line to append
    """
    async with aiofiles.open(filepath, 'a', encoding='utf-8') as f:
        await f.write(line + '\n')


async def read_large_file_chunks(filepath: str, chunk_size: int = 1024) -> List[str]:
    """
    Read large file in chunks.

    Args:
        filepath: Path to file
        chunk_size: Size of each chunk in bytes

    Returns:
        List of chunks
    """
    chunks = []
    async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
        while True:
            chunk = await f.read(chunk_size)
            if not chunk:
                break
            chunks.append(chunk)
    return chunks


async def main():
    """Run all demonstrations"""
    print("\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  Async File I/O Operations".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print("\n")

    # Create temp directory
    temp_dir = Path("temp_async_files")
    temp_dir.mkdir(exist_ok=True)

    # Example 1: Write multiple files
    print("=" * 70)
    print("Example 1: Writing Multiple Files Concurrently")
    print("=" * 70)
    print()

    file_data = {
        str(temp_dir / f"file_{i}.txt"): f"Content of file {i}\n" * 100
        for i in range(10)
    }

    print(f"Writing {len(file_data)} files...")
    start = time.time()
    await write_multiple_files(file_data)
    elapsed = time.time() - start

    print(f"✓ Wrote {len(file_data)} files in {elapsed:.3f}s")
    print(f"  Average: {elapsed/len(file_data):.3f}s per file")
    print()

    # Example 2: Read multiple files
    print("=" * 70)
    print("Example 2: Reading Multiple Files Concurrently")
    print("=" * 70)
    print()

    filepaths = list(file_data.keys())
    print(f"Reading {len(filepaths)} files...")

    start = time.time()
    contents = await read_multiple_files(filepaths)
    elapsed = time.time() - start

    print(f"✓ Read {len(contents)} files in {elapsed:.3f}s")
    print(f"  Average: {elapsed/len(contents):.3f}s per file")

    # Show file sizes
    total_size = sum(len(content) for content in contents.values())
    print(f"  Total data read: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    print()

    # Example 3: CSV operations
    print("=" * 70)
    print("Example 3: CSV Operations")
    print("=" * 70)
    print()

    # Create sample CSV
    csv_path = temp_dir / "predictions.csv"
    predictions = [
        {"sample_id": i, "prediction": 0.9 - (i * 0.01), "label": i % 2}
        for i in range(100)
    ]

    print(f"Saving {len(predictions)} predictions to CSV...")
    await save_predictions_async(str(csv_path), predictions)
    print(f"✓ Saved to {csv_path}")

    print(f"Reading CSV...")
    loaded = await process_csv_async(str(csv_path))
    print(f"✓ Loaded {len(loaded)} rows")
    print(f"  First row: {loaded[0]}")
    print()

    # Example 4: Append operations
    print("=" * 70)
    print("Example 4: Appending to Log File")
    print("=" * 70)
    print()

    log_path = temp_dir / "experiment.log"
    print(f"Appending 20 log entries...")

    start = time.time()
    tasks = [
        append_to_file_async(str(log_path), f"Epoch {i}: loss=0.{95-i}, acc=0.{80+i}")
        for i in range(20)
    ]
    await asyncio.gather(*tasks)
    elapsed = time.time() - start

    print(f"✓ Appended 20 entries in {elapsed:.3f}s")

    # Read log
    log_content = await read_file_async(str(log_path))
    log_lines = log_content.strip().split('\n')
    print(f"  Log has {len(log_lines)} lines")
    print(f"  First: {log_lines[0]}")
    print(f"  Last: {log_lines[-1]}")
    print()

    # Example 5: Large file chunked reading
    print("=" * 70)
    print("Example 5: Reading Large File in Chunks")
    print("=" * 70)
    print()

    # Create large file
    large_file = temp_dir / "large_dataset.txt"
    large_content = "Sample data line\n" * 10000
    await write_file_async(str(large_file), large_content)

    print(f"Reading large file ({len(large_content)} bytes) in chunks...")
    start = time.time()
    chunks = await read_large_file_chunks(str(large_file), chunk_size=4096)
    elapsed = time.time() - start

    print(f"✓ Read {len(chunks)} chunks in {elapsed:.3f}s")
    total_read = sum(len(chunk) for chunk in chunks)
    print(f"  Total bytes read: {total_read:,}")
    print()

    # Example 6: Concurrent file operations mix
    print("=" * 70)
    print("Example 6: Mixed Concurrent Operations")
    print("=" * 70)
    print()

    print("Performing mixed file operations concurrently...")
    start = time.time()

    # Do multiple different operations at once
    results = await asyncio.gather(
        write_file_async(str(temp_dir / "new1.txt"), "New file 1"),
        write_file_async(str(temp_dir / "new2.txt"), "New file 2"),
        read_file_async(str(temp_dir / "file_0.txt")),
        read_file_async(str(temp_dir / "file_1.txt")),
        append_to_file_async(str(log_path), "Final log entry")
    )

    elapsed = time.time() - start
    print(f"✓ Completed 5 mixed operations in {elapsed:.3f}s")
    print()

    # Cleanup
    print("=" * 70)
    print("Cleanup")
    print("=" * 70)
    print()
    print(f"Cleaning up {len(list(temp_dir.iterdir()))} files...")

    # Delete all files
    for file_path in temp_dir.iterdir():
        file_path.unlink()
    temp_dir.rmdir()

    print("✓ Cleanup complete")
    print()

    # Summary
    print("=" * 70)
    print("Key Takeaways")
    print("=" * 70)
    print("1. Use 'async with aiofiles.open()' for async file I/O")
    print("2. Use 'await f.read()' and 'await f.write()' for operations")
    print("3. Read/write multiple files concurrently for speedup")
    print("4. Convert CSV operations to async with StringIO")
    print("5. Use chunked reading for large files")
    print("6. Async file I/O is especially beneficial for many files")
    print("=" * 70)
    print()


if __name__ == "__main__":
    asyncio.run(main())
