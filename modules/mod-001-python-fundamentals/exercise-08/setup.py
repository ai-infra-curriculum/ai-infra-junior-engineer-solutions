"""Setup configuration for ml-infra-utils package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read version from package
about = {}
with open("src/ml_infra_utils/__init__.py", encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            about["__version__"] = line.split('"')[1]
            break

setup(
    name="ml-infra-utils",
    version=about["__version__"],
    author="ML Infrastructure Team",
    author_email="ml-team@example.com",
    description="Reusable ML infrastructure utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourorg/ml-infra-utils",
    project_urls={
        "Bug Tracker": "https://github.com/yourorg/ml-infra-utils/issues",
        "Documentation": "https://ml-infra-utils.readthedocs.io",
        "Source Code": "https://github.com/yourorg/ml-infra-utils",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        # No runtime dependencies for this simple package
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "isort>=5.10.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Add CLI tools if needed
            # "ml-utils=ml_infra_utils.cli:main",
        ],
    },
)
