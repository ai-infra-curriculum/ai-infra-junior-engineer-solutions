#!/usr/bin/env python3
"""
Docker image optimization and analysis tool.
Analyzes images for size, layers, and efficiency.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class LayerInfo:
    """Information about a Docker layer."""
    id: str
    size: int
    command: str
    created_by: str


@dataclass
class ImageAnalysis:
    """Analysis results for a Docker image."""
    name: str
    total_size: int
    layer_count: int
    layers: List[LayerInfo]
    wasted_space: int
    efficiency_score: float
    recommendations: List[str]


class ImageOptimizer:
    """Analyze and optimize Docker images."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def analyze_image(self, image_name: str) -> Optional[ImageAnalysis]:
        """
        Analyze a Docker image.

        Args:
            image_name: Name/tag of the image to analyze

        Returns:
            ImageAnalysis object or None if analysis fails
        """
        try:
            # Get image details
            inspect_data = self._inspect_image(image_name)
            if not inspect_data:
                return None

            # Parse layers
            layers = self._parse_layers(inspect_data)

            # Calculate metrics
            total_size = sum(layer.size for layer in layers)
            wasted_space = self._calculate_wasted_space(layers)
            efficiency_score = self._calculate_efficiency(total_size, wasted_space)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                inspect_data, layers, efficiency_score
            )

            return ImageAnalysis(
                name=image_name,
                total_size=total_size,
                layer_count=len(layers),
                layers=layers,
                wasted_space=wasted_space,
                efficiency_score=efficiency_score,
                recommendations=recommendations
            )

        except Exception as e:
            print(f"Error analyzing image {image_name}: {e}")
            return None

    def _inspect_image(self, image_name: str) -> Optional[Dict]:
        """Inspect image and return metadata."""
        try:
            cmd = ["docker", "inspect", image_name]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            return data[0] if data else None
        except (subprocess.CalledProcessError, json.JSONDecodeError, IndexError):
            return None

    def _parse_layers(self, inspect_data: Dict) -> List[LayerInfo]:
        """Parse layer information from inspect data."""
        layers = []

        # Get layer history
        history = inspect_data.get("RootFS", {}).get("Layers", [])

        for idx, layer_id in enumerate(history):
            # Try to get size from metadata
            size = 0  # Would need more detailed inspection
            command = f"Layer {idx}"
            created_by = "unknown"

            layers.append(LayerInfo(
                id=layer_id,
                size=size,
                command=command,
                created_by=created_by
            ))

        return layers

    def _calculate_wasted_space(self, layers: List[LayerInfo]) -> int:
        """Calculate wasted space in image."""
        # This is a simplified calculation
        # In reality, would need to analyze layer contents
        return 0

    def _calculate_efficiency(self, total_size: int, wasted_space: int) -> float:
        """Calculate efficiency score (0-100)."""
        if total_size == 0:
            return 100.0

        efficiency = ((total_size - wasted_space) / total_size) * 100
        return round(efficiency, 2)

    def _generate_recommendations(
        self,
        inspect_data: Dict,
        layers: List[LayerInfo],
        efficiency_score: float
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Check user
        config = inspect_data.get("Config", {})
        user = config.get("User", "")
        if not user or user == "root":
            recommendations.append(
                "Use a non-root user for better security"
            )

        # Check for health check
        if not config.get("Healthcheck"):
            recommendations.append(
                "Add a HEALTHCHECK instruction for better container monitoring"
            )

        # Check layer count
        if len(layers) > 20:
            recommendations.append(
                f"Image has {len(layers)} layers. Consider combining RUN commands "
                "to reduce layer count"
            )

        # Check for common optimizations
        labels = config.get("Labels", {})
        if not labels:
            recommendations.append(
                "Add labels for better image metadata and tracking"
            )

        # Check efficiency score
        if efficiency_score < 80:
            recommendations.append(
                f"Low efficiency score ({efficiency_score}%). "
                "Consider using multi-stage builds and removing unnecessary files"
            )

        # Check base image
        if "ubuntu" in inspect_data.get("RepoTags", [""])[0].lower():
            recommendations.append(
                "Consider using Alpine or slim variants for smaller base images"
            )

        return recommendations

    def get_image_size(self, image_name: str) -> Tuple[int, str]:
        """
        Get image size in bytes and human-readable format.

        Returns:
            Tuple of (size_bytes, size_human_readable)
        """
        try:
            # Get size in bytes
            cmd = ["docker", "inspect", "--format={{.Size}}", image_name]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            size_bytes = int(result.stdout.strip())

            # Convert to human-readable
            size_human = self._bytes_to_human(size_bytes)

            return size_bytes, size_human

        except (subprocess.CalledProcessError, ValueError):
            return 0, "unknown"

    def _bytes_to_human(self, size_bytes: int) -> str:
        """Convert bytes to human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f}PB"

    def compare_images(self, image1: str, image2: str) -> Dict:
        """
        Compare two images.

        Args:
            image1: First image name
            image2: Second image name

        Returns:
            Comparison dictionary
        """
        size1, human1 = self.get_image_size(image1)
        size2, human2 = self.get_image_size(image2)

        analysis1 = self.analyze_image(image1)
        analysis2 = self.analyze_image(image2)

        comparison = {
            "image1": {
                "name": image1,
                "size_bytes": size1,
                "size_human": human1,
                "layers": analysis1.layer_count if analysis1 else 0,
                "efficiency": analysis1.efficiency_score if analysis1 else 0
            },
            "image2": {
                "name": image2,
                "size_bytes": size2,
                "size_human": human2,
                "layers": analysis2.layer_count if analysis2 else 0,
                "efficiency": analysis2.efficiency_score if analysis2 else 0
            },
            "differences": {
                "size_diff_bytes": size2 - size1,
                "size_diff_percent": ((size2 - size1) / size1 * 100) if size1 > 0 else 0,
                "layer_diff": (analysis2.layer_count - analysis1.layer_count) if (analysis1 and analysis2) else 0
            }
        }

        return comparison

    def export_sbom(self, image_name: str, output_file: str) -> bool:
        """
        Export Software Bill of Materials (SBOM) for an image.

        Args:
            image_name: Image to analyze
            output_file: Output file path

        Returns:
            True if successful
        """
        try:
            # Use syft to generate SBOM
            cmd = ["syft", image_name, "-o", "json", "--file", output_file]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode == 0:
                print(f"✓ SBOM exported to {output_file}")
                return True
            else:
                print(f"✗ Failed to export SBOM")
                return False

        except FileNotFoundError:
            print("✗ syft not found. Install it for SBOM generation.")
            return False

    def print_analysis(self, analysis: ImageAnalysis) -> None:
        """Print analysis results."""
        print(f"\n{'='*60}")
        print(f"Image Analysis: {analysis.name}")
        print(f"{'='*60}")
        print(f"Total Size: {self._bytes_to_human(analysis.total_size)}")
        print(f"Layer Count: {analysis.layer_count}")
        print(f"Wasted Space: {self._bytes_to_human(analysis.wasted_space)}")
        print(f"Efficiency Score: {analysis.efficiency_score}%")

        if analysis.recommendations:
            print(f"\nRecommendations:")
            for idx, rec in enumerate(analysis.recommendations, 1):
                print(f"  {idx}. {rec}")

        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and optimize Docker images"
    )
    parser.add_argument(
        "image",
        help="Image name to analyze"
    )
    parser.add_argument(
        "--compare",
        help="Compare with another image"
    )
    parser.add_argument(
        "--sbom",
        help="Export SBOM to file"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    optimizer = ImageOptimizer(verbose=args.verbose)

    # Analyze image
    analysis = optimizer.analyze_image(args.image)
    if not analysis:
        print(f"Failed to analyze image: {args.image}")
        sys.exit(1)

    # Compare if requested
    if args.compare:
        comparison = optimizer.compare_images(args.image, args.compare)
        if args.json:
            print(json.dumps(comparison, indent=2))
        else:
            print(f"\nComparison:")
            print(f"  {args.image}: {comparison['image1']['size_human']}")
            print(f"  {args.compare}: {comparison['image2']['size_human']}")
            diff_pct = comparison['differences']['size_diff_percent']
            print(f"  Difference: {diff_pct:+.2f}%")

    # Export SBOM if requested
    if args.sbom:
        optimizer.export_sbom(args.image, args.sbom)

    # Print or output results
    if args.json:
        output = {
            "name": analysis.name,
            "total_size": analysis.total_size,
            "layer_count": analysis.layer_count,
            "efficiency_score": analysis.efficiency_score,
            "recommendations": analysis.recommendations
        }
        print(json.dumps(output, indent=2))
    else:
        optimizer.print_analysis(analysis)


if __name__ == "__main__":
    main()
