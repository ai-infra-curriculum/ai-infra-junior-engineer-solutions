#!/usr/bin/env python3
"""
Advanced Docker image builder for ML frameworks.
Supports building, tagging, pushing, and optimizing ML Docker images.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class BuildConfig:
    """Configuration for Docker build."""
    name: str
    dockerfile: str
    context: str
    tags: List[str]
    build_args: Dict[str, str]
    cache_from: List[str]
    target: Optional[str] = None
    platform: Optional[str] = None


class DockerImageBuilder:
    """Build and manage Docker images for ML frameworks."""

    def __init__(self, registry: Optional[str] = None, verbose: bool = False):
        self.registry = registry
        self.verbose = verbose
        self.build_history: List[Dict] = []

    def build_image(
        self,
        config: BuildConfig,
        no_cache: bool = False,
        push: bool = False,
        squash: bool = False
    ) -> bool:
        """
        Build a Docker image with the given configuration.

        Args:
            config: Build configuration
            no_cache: Disable build cache
            push: Push image after building
            squash: Squash image layers (experimental)

        Returns:
            True if build successful, False otherwise
        """
        build_start = datetime.now()

        try:
            # Construct build command
            cmd = ["docker", "build"]

            # Add build arguments
            for key, value in config.build_args.items():
                cmd.extend(["--build-arg", f"{key}={value}"])

            # Add tags
            for tag in config.tags:
                full_tag = f"{self.registry}/{tag}" if self.registry else tag
                cmd.extend(["-t", full_tag])

            # Add cache options
            if not no_cache:
                for cache_img in config.cache_from:
                    cmd.extend(["--cache-from", cache_img])
            else:
                cmd.append("--no-cache")

            # Add target stage if specified
            if config.target:
                cmd.extend(["--target", config.target])

            # Add platform if specified
            if config.platform:
                cmd.extend(["--platform", config.platform])

            # Add experimental features
            if squash:
                cmd.append("--squash")

            # Add context and dockerfile
            cmd.extend(["-f", config.dockerfile, config.context])

            if self.verbose:
                print(f"Building {config.name}...")
                print(f"Command: {' '.join(cmd)}")

            # Execute build
            result = subprocess.run(
                cmd,
                capture_output=not self.verbose,
                text=True,
                check=False
            )

            build_end = datetime.now()
            build_time = (build_end - build_start).total_seconds()

            if result.returncode == 0:
                print(f"✓ Successfully built {config.name} in {build_time:.2f}s")

                # Get image size
                size = self._get_image_size(config.tags[0])

                # Record build history
                self.build_history.append({
                    "name": config.name,
                    "tags": config.tags,
                    "build_time": build_time,
                    "size": size,
                    "timestamp": build_start.isoformat(),
                    "success": True
                })

                # Push if requested
                if push:
                    self.push_image(config.tags)

                return True
            else:
                print(f"✗ Failed to build {config.name}")
                if not self.verbose and result.stderr:
                    print(f"Error: {result.stderr}")

                self.build_history.append({
                    "name": config.name,
                    "tags": config.tags,
                    "build_time": build_time,
                    "timestamp": build_start.isoformat(),
                    "success": False,
                    "error": result.stderr
                })

                return False

        except Exception as e:
            print(f"✗ Exception while building {config.name}: {e}")
            return False

    def _get_image_size(self, tag: str) -> str:
        """Get the size of a Docker image."""
        try:
            cmd = ["docker", "images", "--format", "{{.Size}}", tag]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"

    def push_image(self, tags: List[str]) -> bool:
        """
        Push image to registry.

        Args:
            tags: List of image tags to push

        Returns:
            True if all pushes successful
        """
        all_success = True

        for tag in tags:
            full_tag = f"{self.registry}/{tag}" if self.registry else tag

            try:
                if self.verbose:
                    print(f"Pushing {full_tag}...")

                result = subprocess.run(
                    ["docker", "push", full_tag],
                    capture_output=not self.verbose,
                    text=True,
                    check=False
                )

                if result.returncode == 0:
                    print(f"✓ Successfully pushed {full_tag}")
                else:
                    print(f"✗ Failed to push {full_tag}")
                    all_success = False

            except Exception as e:
                print(f"✗ Exception while pushing {full_tag}: {e}")
                all_success = False

        return all_success

    def scan_image(self, tag: str, severity: str = "HIGH,CRITICAL") -> Dict:
        """
        Scan image for vulnerabilities using Trivy.

        Args:
            tag: Image tag to scan
            severity: Severity levels to report

        Returns:
            Scan results dictionary
        """
        try:
            cmd = [
                "trivy", "image",
                "--severity", severity,
                "--format", "json",
                tag
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)

        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to scan {tag}: {e}")
            return {}
        except FileNotFoundError:
            print("✗ Trivy not found. Install it for vulnerability scanning.")
            return {}

    def optimize_image(self, tag: str) -> bool:
        """
        Optimize image using dive or similar tools.

        Args:
            tag: Image tag to optimize

        Returns:
            True if optimization successful
        """
        try:
            # Use dive to analyze image efficiency
            cmd = ["dive", "--ci", tag]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode == 0:
                print(f"✓ Image {tag} passes efficiency checks")
                return True
            else:
                print(f"✗ Image {tag} failed efficiency checks")
                if result.stdout:
                    print(result.stdout)
                return False

        except FileNotFoundError:
            print("✗ dive not found. Install it for image optimization analysis.")
            return False

    def export_build_report(self, output_file: str) -> None:
        """Export build history to JSON file."""
        report = {
            "build_date": datetime.now().isoformat(),
            "total_builds": len(self.build_history),
            "successful_builds": sum(1 for b in self.build_history if b["success"]),
            "builds": self.build_history
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✓ Build report exported to {output_file}")


def get_ml_build_configs(base_path: Path) -> List[BuildConfig]:
    """Get build configurations for all ML frameworks."""
    configs = []

    # TensorFlow configuration
    configs.append(BuildConfig(
        name="tensorflow",
        dockerfile=str(base_path / "Dockerfile.tensorflow"),
        context=str(base_path),
        tags=["ml-tensorflow:latest", "ml-tensorflow:2.15.0"],
        build_args={"PYTHON_VERSION": "3.11"},
        cache_from=["ml-tensorflow:latest"]
    ))

    # PyTorch configuration
    configs.append(BuildConfig(
        name="pytorch",
        dockerfile=str(base_path / "Dockerfile.pytorch"),
        context=str(base_path),
        tags=["ml-pytorch:latest", "ml-pytorch:2.1.1"],
        build_args={"PYTHON_VERSION": "3.11"},
        cache_from=["ml-pytorch:latest"]
    ))

    # Scikit-learn configuration
    configs.append(BuildConfig(
        name="scikit-learn",
        dockerfile=str(base_path / "Dockerfile.scikit-learn"),
        context=str(base_path),
        tags=["ml-sklearn:latest", "ml-sklearn:1.3.2"],
        build_args={"PYTHON_VERSION": "3.11"},
        cache_from=["ml-sklearn:latest"]
    ))

    # Optimized multi-stage configuration
    configs.append(BuildConfig(
        name="optimized",
        dockerfile=str(base_path / "Dockerfile.multistage-optimized"),
        context=str(base_path),
        tags=["ml-optimized:latest"],
        build_args={"PYTHON_VERSION": "3.11"},
        cache_from=["ml-optimized:latest"],
        target="runtime"
    ))

    # GPU configuration
    configs.append(BuildConfig(
        name="pytorch-gpu",
        dockerfile=str(base_path / "Dockerfile.gpu"),
        context=str(base_path),
        tags=["ml-pytorch-gpu:latest", "ml-pytorch-gpu:2.1.1-cuda12.1"],
        build_args={"CUDA_VERSION": "12.1"},
        cache_from=["ml-pytorch-gpu:latest"]
    ))

    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Build ML Docker images with optimization"
    )
    parser.add_argument(
        "--framework",
        choices=["tensorflow", "pytorch", "scikit-learn", "optimized", "pytorch-gpu", "all"],
        default="all",
        help="Framework to build"
    )
    parser.add_argument(
        "--registry",
        help="Docker registry to use for tagging/pushing"
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push images after building"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Build without cache"
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan images for vulnerabilities"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run optimization checks"
    )
    parser.add_argument(
        "--report",
        help="Export build report to file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Initialize builder
    builder = DockerImageBuilder(registry=args.registry, verbose=args.verbose)

    # Get build configurations
    base_path = Path(__file__).parent
    all_configs = get_ml_build_configs(base_path)

    # Filter configurations
    if args.framework == "all":
        configs = all_configs
    else:
        configs = [c for c in all_configs if c.name == args.framework]

    # Build images
    print(f"Building {len(configs)} image(s)...")
    success_count = 0

    for config in configs:
        success = builder.build_image(
            config,
            no_cache=args.no_cache,
            push=args.push
        )

        if success:
            success_count += 1

            # Scan if requested
            if args.scan:
                print(f"Scanning {config.tags[0]}...")
                results = builder.scan_image(config.tags[0])
                if results:
                    print(f"Scan completed for {config.tags[0]}")

            # Optimize if requested
            if args.optimize:
                print(f"Optimizing {config.tags[0]}...")
                builder.optimize_image(config.tags[0])

    # Print summary
    print(f"\n{'='*60}")
    print(f"Build Summary: {success_count}/{len(configs)} successful")
    print(f"{'='*60}")

    # Export report if requested
    if args.report:
        builder.export_build_report(args.report)

    # Exit with appropriate code
    sys.exit(0 if success_count == len(configs) else 1)


if __name__ == "__main__":
    main()
