#!/usr/bin/env python3
"""
Comprehensive File Manager for ML Workflows

Unified interface for reading and writing multiple file formats (CSV, JSON, YAML, Pickle).
"""

import json
import csv
import yaml
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union, Optional


class MLFileManager:
    """
    Comprehensive file manager for ML workflows.

    Provides a unified interface for saving and loading data in multiple formats
    with automatic format detection and consistent error handling.
    """

    def __init__(self, base_dir: str = "."):
        """
        Initialize file manager.

        Args:
            base_dir: Base directory for file operations
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, data: Any, filename: str, format: str = 'auto') -> None:
        """
        Save data in specified format.

        Args:
            data: Data to save
            filename: Output filename
            format: Format ('json', 'yaml', 'csv', 'pickle', or 'auto')

        Raises:
            ValueError: If format is unsupported
        """
        filepath = self.base_dir / filename

        # Auto-detect format from extension
        if format == 'auto':
            format = filepath.suffix[1:]  # Remove dot

        if format == 'json':
            self._save_json(filepath, data)
        elif format in ('yaml', 'yml'):
            self._save_yaml(filepath, data)
        elif format == 'csv':
            self._save_csv(filepath, data)
        elif format in ('pkl', 'pickle'):
            self._save_pickle(filepath, data)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"✓ Saved {filename}")

    def load(self, filename: str, format: str = 'auto') -> Any:
        """
        Load data from file.

        Args:
            filename: Input filename
            format: Format ('json', 'yaml', 'csv', 'pickle', or 'auto')

        Returns:
            Loaded data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format is unsupported
        """
        filepath = self.base_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if format == 'auto':
            format = filepath.suffix[1:]

        if format == 'json':
            return self._load_json(filepath)
        elif format in ('yaml', 'yml'):
            return self._load_yaml(filepath)
        elif format == 'csv':
            return self._load_csv(filepath)
        elif format in ('pkl', 'pickle'):
            return self._load_pickle(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _save_json(self, filepath: Path, data: Any) -> None:
        """Save data as JSON."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, sort_keys=True)

    def _load_json(self, filepath: Path) -> Any:
        """Load data from JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_yaml(self, filepath: Path, data: Any) -> None:
        """Save data as YAML."""
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def _load_yaml(self, filepath: Path) -> Any:
        """Load data from YAML."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _save_csv(self, filepath: Path, data: List[Dict]) -> None:
        """Save data as CSV."""
        if not data:
            return

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    def _load_csv(self, filepath: Path) -> List[Dict]:
        """Load data from CSV."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))

    def _save_pickle(self, filepath: Path, data: Any) -> None:
        """Save data as pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def _load_pickle(self, filepath: Path) -> Any:
        """Load data from pickle."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def list_files(self, pattern: str = '*') -> List[str]:
        """
        List files matching pattern.

        Args:
            pattern: Glob pattern (default: '*' for all files)

        Returns:
            List of matching filenames
        """
        return [f.name for f in self.base_dir.glob(pattern)]

    def file_exists(self, filename: str) -> bool:
        """Check if file exists."""
        return (self.base_dir / filename).exists()

    def delete_file(self, filename: str) -> None:
        """Delete file if it exists."""
        filepath = self.base_dir / filename
        if filepath.exists():
            filepath.unlink()
            print(f"✓ Deleted {filename}")
        else:
            print(f"✗ File not found: {filename}")

    def get_file_size(self, filename: str) -> int:
        """
        Get file size in bytes.

        Returns:
            File size in bytes, or -1 if file doesn't exist
        """
        filepath = self.base_dir / filename
        if filepath.exists():
            return filepath.stat().st_size
        return -1


def main():
    """Demonstrate file manager usage."""
    print("=" * 70)
    print("ML File Manager Demonstration")
    print("=" * 70)
    print()

    # Create manager
    manager = MLFileManager(base_dir="ml_data")
    print(f"Base directory: {manager.base_dir.resolve()}")
    print()

    # Example 1: Save/load JSON
    print("Example 1: JSON Operations")
    print("-" * 70)
    config = {
        'model': 'resnet50',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100
    }
    manager.save(config, 'config.json')
    loaded_config = manager.load('config.json')
    print(f"Loaded config: {loaded_config}")
    print()

    # Example 2: Save/load YAML
    print("Example 2: YAML Operations")
    print("-" * 70)
    pipeline_config = {
        'preprocessing': {
            'normalize': True,
            'augment': False
        },
        'model': {
            'type': 'cnn',
            'layers': [64, 128, 256]
        }
    }
    manager.save(pipeline_config, 'pipeline.yaml')
    loaded_pipeline = manager.load('pipeline.yaml')
    print(f"Loaded pipeline: {loaded_pipeline}")
    print()

    # Example 3: Save/load CSV
    print("Example 3: CSV Operations")
    print("-" * 70)
    results = [
        {'experiment': 'exp001', 'accuracy': 0.92, 'loss': 0.15},
        {'experiment': 'exp002', 'accuracy': 0.88, 'loss': 0.22},
        {'experiment': 'exp003', 'accuracy': 0.95, 'loss': 0.10}
    ]
    manager.save(results, 'results.csv')
    loaded_results = manager.load('results.csv')
    print(f"Loaded {len(loaded_results)} results")
    print(f"First result: {loaded_results[0]}")
    print()

    # Example 4: Save/load Pickle
    print("Example 4: Pickle Operations")
    print("-" * 70)
    model_checkpoint = {
        'epoch': 50,
        'model_state': {'layer1.weight': [0.1, 0.2, 0.3]},
        'optimizer_state': {'lr': 0.001},
        'metrics': {'accuracy': 0.92, 'loss': 0.15}
    }
    manager.save(model_checkpoint, 'checkpoint.pkl')
    loaded_checkpoint = manager.load('checkpoint.pkl')
    print(f"Loaded checkpoint: epoch {loaded_checkpoint['epoch']}")
    print(f"Metrics: {loaded_checkpoint['metrics']}")
    print()

    # Example 5: List files
    print("Example 5: File Listing")
    print("-" * 70)
    all_files = manager.list_files()
    print(f"All files: {all_files}")

    json_files = manager.list_files('*.json')
    print(f"JSON files: {json_files}")

    csv_files = manager.list_files('*.csv')
    print(f"CSV files: {csv_files}")
    print()

    # Example 6: File operations
    print("Example 6: File Operations")
    print("-" * 70)
    print(f"config.json exists: {manager.file_exists('config.json')}")
    print(f"nonexistent.json exists: {manager.file_exists('nonexistent.json')}")

    size = manager.get_file_size('config.json')
    print(f"config.json size: {size} bytes")
    print()

    # Example 7: Error handling
    print("Example 7: Error Handling")
    print("-" * 70)
    try:
        manager.load('nonexistent.json')
    except FileNotFoundError as e:
        print(f"✓ Caught expected error: {type(e).__name__}")

    try:
        manager.save({}, 'file.xyz', format='unsupported')
    except ValueError as e:
        print(f"✓ Caught expected error: {type(e).__name__}")
    print()

    print("=" * 70)
    print("✓ File manager demonstration complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
