#!/usr/bin/env python3
"""
Feature metadata manager for ML features.

Manages features and their metadata including importance, data types, and usage tracking.
"""

from typing import Dict, List, Any, Tuple
import json
from pathlib import Path


class FeatureManager:
    """Manage ML features and their metadata."""

    def __init__(self):
        """Initialize feature manager."""
        self.features: Dict[str, Dict[str, Any]] = {}

    def add_feature(self, name: str, dtype: str,
                   importance: float = 0.0, description: str = ""):
        """
        Add a feature with metadata.

        Args:
            name: Feature name
            dtype: Data type (int, float, str, bool)
            importance: Feature importance score (0-1)
            description: Feature description
        """
        self.features[name] = {
            "dtype": dtype,
            "importance": importance,
            "description": description,
            "used_count": 0
        }

    def get_feature(self, name: str) -> Dict[str, Any]:
        """
        Get feature metadata.

        Args:
            name: Feature name

        Returns:
            Feature metadata dictionary
        """
        return self.features.get(name, {})

    def update_importance(self, name: str, importance: float):
        """
        Update feature importance.

        Args:
            name: Feature name
            importance: New importance score
        """
        if name in self.features:
            self.features[name]["importance"] = importance

    def increment_usage(self, name: str):
        """
        Track feature usage.

        Args:
            name: Feature name
        """
        if name in self.features:
            self.features[name]["used_count"] += 1

    def get_top_features(self, n: int = 5) -> List[Tuple[str, Dict]]:
        """
        Get top N features by importance.

        Args:
            n: Number of top features to return

        Returns:
            List of (feature_name, metadata) tuples
        """
        sorted_features = sorted(
            self.features.items(),
            key=lambda x: x[1]["importance"],
            reverse=True
        )
        return sorted_features[:n]

    def filter_by_dtype(self, dtype: str) -> Dict[str, Dict]:
        """
        Get all features of specific data type.

        Args:
            dtype: Data type to filter by

        Returns:
            Dictionary of matching features
        """
        return {
            name: meta for name, meta in self.features.items()
            if meta["dtype"] == dtype
        }

    def get_unused_features(self) -> List[str]:
        """
        Get list of unused features.

        Returns:
            List of feature names with used_count == 0
        """
        return [name for name, meta in self.features.items()
                if meta["used_count"] == 0]

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Dictionary with summary information
        """
        if not self.features:
            return {"total_features": 0}

        importances = [m["importance"] for m in self.features.values()]

        return {
            "total_features": len(self.features),
            "avg_importance": sum(importances) / len(importances),
            "max_importance": max(importances),
            "unused_features": len(self.get_unused_features()),
            "dtypes": list(set(m["dtype"] for m in self.features.values()))
        }

    def export_config(self, filepath: str):
        """
        Export features to JSON.

        Args:
            filepath: Path to save JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(self.features, f, indent=2)

    def import_config(self, filepath: str):
        """
        Import features from JSON.

        Args:
            filepath: Path to JSON file
        """
        with open(filepath, 'r') as f:
            self.features = json.load(f)


def main():
    """Demonstrate feature manager usage."""
    print("=" * 60)
    print("Feature Manager for ML")
    print("=" * 60)
    print()

    manager = FeatureManager()

    # Add features
    manager.add_feature("age", "int", 0.85, "User age in years")
    manager.add_feature("income", "float", 0.92, "Annual income")
    manager.add_feature("location", "str", 0.65, "City name")
    manager.add_feature("clicks", "int", 0.78, "Number of clicks")
    manager.add_feature("conversion", "bool", 0.95, "Converted or not")
    manager.add_feature("gender", "str", 0.45, "User gender")
    manager.add_feature("signup_date", "str", 0.55, "Account creation date")

    print(f"Added {len(manager.features)} features")
    print()

    # Update and use features
    manager.update_importance("age", 0.88)
    manager.increment_usage("age")
    manager.increment_usage("income")
    manager.increment_usage("conversion")

    # Get top features
    print("Top 3 features by importance:")
    top_features = manager.get_top_features(3)
    for i, (name, meta) in enumerate(top_features, 1):
        print(f"  {i}. {name}:")
        print(f"     - Importance: {meta['importance']:.2f}")
        print(f"     - Type: {meta['dtype']}")
        print(f"     - Description: {meta['description']}")
        print(f"     - Used: {meta['used_count']} times")
    print()

    # Filter by type
    numeric_features = manager.filter_by_dtype("int")
    print(f"Integer features ({len(numeric_features)}):")
    for name in numeric_features.keys():
        print(f"  - {name}")
    print()

    # Get unused features
    unused = manager.get_unused_features()
    print(f"Unused features ({len(unused)}):")
    for name in unused:
        print(f"  - {name}")
    print()

    # Get summary
    summary = manager.get_summary()
    print("Summary statistics:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    print()

    # Export/import
    output_path = Path("features.json")
    manager.export_config(str(output_path))
    print(f"✓ Features exported to {output_path}")

    # Test import
    manager2 = FeatureManager()
    manager2.import_config(str(output_path))
    print(f"✓ Features imported: {len(manager2.features)} features")

    # Cleanup
    if output_path.exists():
        output_path.unlink()

    print()
    print("✓ Feature manager demonstration complete")


if __name__ == "__main__":
    main()
