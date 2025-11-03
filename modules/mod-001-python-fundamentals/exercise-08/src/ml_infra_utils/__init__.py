"""ML Infrastructure Utilities Package.

A collection of reusable utilities for ML infrastructure projects.
"""

__version__ = "0.1.0"
__author__ = "ML Infrastructure Team"
__email__ = "ml-team@example.com"

# Import commonly used functions for convenience
from ml_infra_utils.preprocessing.normalization import normalize, standardize
from ml_infra_utils.metrics.classification import accuracy, precision, recall, f1_score
from ml_infra_utils.decorators.timing import timer
from ml_infra_utils.decorators.retry import retry
from ml_infra_utils.logging.structured import StructuredLogger

__all__ = [
    "normalize",
    "standardize",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "timer",
    "retry",
    "StructuredLogger",
    "__version__",
]
