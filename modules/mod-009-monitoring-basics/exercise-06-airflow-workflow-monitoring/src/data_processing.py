"""
Data Processing Tasks

Contains all data processing related tasks:
- Download data from storage
- Validate data quality
- Preprocess and clean data
"""

import time
import random
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def download_data(**context):
    """
    Download training dataset from cloud storage.

    Simulates downloading data from S3/GCS and performs basic checks.

    Returns:
        dict: Dataset metadata including size, samples, and download time
    """
    logger.info("=" * 60)
    logger.info("Starting data download...")
    logger.info("=" * 60)

    # Get execution date from context
    ds = context.get('ds', datetime.now().strftime('%Y-%m-%d'))
    logger.info(f"Execution date: {ds}")

    # Simulate download process
    logger.info("Connecting to cloud storage...")
    time.sleep(1)

    logger.info("Downloading dataset...")
    # Simulate variable download time
    download_time = random.uniform(2, 5)
    time.sleep(download_time)

    # Generate dataset metadata
    dataset_size_gb = round(random.uniform(1.0, 2.5), 2)
    num_samples = random.randint(8000, 12000)

    metadata = {
        'dataset_size_gb': dataset_size_gb,
        'num_samples': num_samples,
        'download_time_seconds': round(download_time, 2),
        'source': f's3://ml-data/datasets/{ds}',
        'timestamp': datetime.now().isoformat()
    }

    logger.info(f"Successfully downloaded {dataset_size_gb}GB dataset")
    logger.info(f"Total samples: {num_samples:,}")
    logger.info(f"Download time: {download_time:.2f}s")
    logger.info("=" * 60)

    # Return metadata (stored in XCom)
    return metadata


def validate_data(**context):
    """
    Validate data quality and schema.

    Checks:
    - Missing values
    - Schema validity
    - Duplicate records
    - Data range validity

    Raises:
        ValueError: If data quality checks fail

    Returns:
        dict: Validation results
    """
    logger.info("=" * 60)
    logger.info("Starting data validation...")
    logger.info("=" * 60)

    # Get metadata from previous task
    ti = context['ti']
    dataset_metadata = ti.xcom_pull(task_ids='download_data')

    if not dataset_metadata:
        raise ValueError("No dataset metadata found from download task")

    num_samples = dataset_metadata['num_samples']
    logger.info(f"Validating {num_samples:,} samples...")

    # Simulate validation checks
    time.sleep(2)

    # Check for nulls
    null_percentage = random.uniform(0, 0.08)
    logger.info(f"Null values: {null_percentage:.2%}")

    if null_percentage > 0.05:  # 5% threshold
        raise ValueError(
            f"Too many null values: {null_percentage:.2%} (threshold: 5%)"
        )

    # Check for duplicates
    duplicate_percentage = random.uniform(0, 0.03)
    logger.info(f"Duplicate records: {duplicate_percentage:.2%}")

    if duplicate_percentage > 0.01:  # 1% threshold
        raise ValueError(
            f"Too many duplicates: {duplicate_percentage:.2%} (threshold: 1%)"
        )

    # Validate schema
    logger.info("Validating schema...")
    time.sleep(1)

    expected_columns = [
        'id', 'features', 'label', 'timestamp',
        'category', 'value', 'metadata'
    ]
    logger.info(f"Expected columns: {', '.join(expected_columns)}")
    logger.info("Schema validation: PASSED")

    # Check data ranges
    logger.info("Validating data ranges...")
    outlier_percentage = random.uniform(0, 0.04)
    logger.info(f"Outliers detected: {outlier_percentage:.2%}")

    validation_results = {
        'null_percentage': round(null_percentage, 4),
        'duplicate_percentage': round(duplicate_percentage, 4),
        'outlier_percentage': round(outlier_percentage, 4),
        'schema_valid': True,
        'total_samples': num_samples,
        'valid_samples': int(num_samples * (1 - null_percentage - duplicate_percentage)),
        'validation_passed': True
    }

    logger.info("=" * 60)
    logger.info("Validation PASSED")
    logger.info(f"Valid samples: {validation_results['valid_samples']:,}")
    logger.info("=" * 60)

    return validation_results


def preprocess_data(**context):
    """
    Preprocess and clean data.

    Steps:
    - Handle missing values
    - Normalize numerical features
    - Encode categorical features
    - Split into train/validation sets

    Returns:
        dict: Preprocessing results
    """
    logger.info("=" * 60)
    logger.info("Starting data preprocessing...")
    logger.info("=" * 60)

    # Get data from previous tasks
    ti = context['ti']
    validation_results = ti.xcom_pull(task_ids='validate_data')

    if not validation_results:
        raise ValueError("No validation results found")

    valid_samples = validation_results['valid_samples']
    logger.info(f"Preprocessing {valid_samples:,} valid samples...")

    # Step 1: Handle missing values
    logger.info("Step 1: Handling missing values...")
    time.sleep(1)
    logger.info("  - Numerical: Imputed with median")
    logger.info("  - Categorical: Imputed with mode")

    # Step 2: Normalize numerical features
    logger.info("Step 2: Normalizing numerical features...")
    time.sleep(1)
    logger.info("  - Applied StandardScaler")
    logger.info("  - Mean: 0, Std: 1")

    # Step 3: Encode categorical features
    logger.info("Step 3: Encoding categorical features...")
    time.sleep(1)
    num_categories = random.randint(5, 15)
    logger.info(f"  - One-hot encoded {num_categories} categories")

    # Step 4: Train/validation split
    logger.info("Step 4: Splitting data...")
    validation_split = 0.2
    train_samples = int(valid_samples * (1 - validation_split))
    val_samples = valid_samples - train_samples

    logger.info(f"  - Training samples: {train_samples:,}")
    logger.info(f"  - Validation samples: {val_samples:,}")
    logger.info(f"  - Split ratio: {1-validation_split:.0%}/{validation_split:.0%}")

    preprocessing_results = {
        'total_samples': valid_samples,
        'train_samples': train_samples,
        'validation_samples': val_samples,
        'validation_split': validation_split,
        'num_features': 50 + num_categories,
        'num_numerical_features': 50,
        'num_categorical_features': num_categories,
        'normalization': 'StandardScaler',
        'encoding': 'OneHotEncoder'
    }

    logger.info("=" * 60)
    logger.info("Preprocessing completed successfully")
    logger.info(f"Total features: {preprocessing_results['num_features']}")
    logger.info("=" * 60)

    return preprocessing_results
