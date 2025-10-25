#!/usr/bin/env python3
"""
prepare_dataset.py - Data validation and preprocessing pipeline

Description:
    Comprehensive data preparation pipeline for ML training including
    validation, preprocessing, feature engineering, and train/val/test splits.

Usage:
    python prepare_dataset.py [OPTIONS]

Options:
    --data-dir DIR          Input data directory
    --output-dir DIR        Output directory for processed data
    --validate              Run validation only
    --preprocess            Run preprocessing
    --split                 Create train/val/test splits
    --split-ratio RATIO     Train/val/test split ratio (default: 0.7,0.15,0.15)
    --schema FILE           Data schema file for validation
    --config FILE           Configuration file
    --workers N             Number of worker processes (default: 4)
    --verbose               Verbose output
    --help                  Display this help message
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class DataValidator:
    """Validate data against schema"""

    def __init__(self, schema_path: str):
        """Initialize validator with schema"""
        with open(schema_path) as f:
            self.schema = json.load(f)

    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate DataFrame against schema

        Args:
            df: DataFrame to validate

        Returns:
            Validation result dictionary
        """
        issues = []
        warnings = []

        # Check required columns
        required_cols = self.schema.get('required_columns', [])
        missing = set(required_cols) - set(df.columns)
        if missing:
            issues.append(f"Missing required columns: {missing}")

        # Check data types
        for col, expected_type in self.schema.get('dtypes', {}).items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if not self._types_compatible(actual_type, expected_type):
                    issues.append(f"Column '{col}' type mismatch: "
                                f"expected {expected_type}, got {actual_type}")

        # Check for nulls
        null_counts = df.isnull().sum()
        for col in self.schema.get('non_nullable', []):
            if col in df.columns and null_counts[col] > 0:
                issues.append(f"Column '{col}' has {null_counts[col]} null values")

        # Check value ranges
        for col, constraints in self.schema.get('constraints', {}).items():
            if col not in df.columns:
                continue

            if 'min' in constraints:
                violations = (df[col] < constraints['min']).sum()
                if violations > 0:
                    issues.append(f"Column '{col}' has {violations} values "
                                f"below minimum {constraints['min']}")

            if 'max' in constraints:
                violations = (df[col] > constraints['max']).sum()
                if violations > 0:
                    issues.append(f"Column '{col}' has {violations} values "
                                f"above maximum {constraints['max']}")

            if 'values' in constraints:
                allowed = set(constraints['values'])
                actual = set(df[col].unique())
                invalid = actual - allowed
                if invalid:
                    issues.append(f"Column '{col}' has invalid values: {invalid}")

        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            warnings.append(f"Found {duplicates} duplicate rows")

        # Check data balance (for classification)
        if 'target_column' in self.schema:
            target_col = self.schema['target_column']
            if target_col in df.columns:
                balance = df[target_col].value_counts()
                imbalance_ratio = balance.max() / balance.min()
                if imbalance_ratio > 10:
                    warnings.append(f"Severe class imbalance detected: "
                                  f"ratio {imbalance_ratio:.1f}:1")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'row_count': len(df),
            'column_count': len(df.columns),
            'null_counts': null_counts.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1e6  # MB
        }

    def _types_compatible(self, actual: str, expected: str) -> bool:
        """Check if actual type is compatible with expected type"""
        type_mapping = {
            'int': ['int64', 'int32', 'int16', 'int8'],
            'float': ['float64', 'float32', 'float16'],
            'string': ['object', 'string'],
            'bool': ['bool'],
            'datetime': ['datetime64']
        }

        for expected_base, compatible_types in type_mapping.items():
            if expected == expected_base:
                return any(ct in actual for ct in compatible_types)

        return actual == expected


class DataPreprocessor:
    """Preprocess data for ML training"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize preprocessor with configuration"""
        self.config = config
        self.scalers = {}
        self.encoders = {}

    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Preprocess DataFrame

        Args:
            df: Input DataFrame
            fit: Whether to fit transformers

        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()

        # Handle missing values
        df = self._handle_missing_values(df)

        # Encode categorical variables
        df = self._encode_categorical(df, fit=fit)

        # Scale numerical features
        df = self._scale_numerical(df, fit=fit)

        # Feature engineering
        df = self._engineer_features(df)

        # Remove outliers
        if self.config.get('remove_outliers', False):
            df = self._remove_outliers(df)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        strategy = self.config.get('missing_value_strategy', 'drop')

        if strategy == 'drop':
            return df.dropna()
        elif strategy == 'mean':
            return df.fillna(df.mean())
        elif strategy == 'median':
            return df.fillna(df.median())
        elif strategy == 'mode':
            return df.fillna(df.mode().iloc[0])
        elif strategy == 'forward_fill':
            return df.fillna(method='ffill')
        else:
            return df

    def _encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            if col not in self.encoders:
                if fit:
                    self.encoders[col] = LabelEncoder()
                    df[col] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Skip if encoder doesn't exist
                    continue
            else:
                df[col] = self.encoders[col].transform(df[col].astype(str))

        return df

    def _scale_numerical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features"""
        if not self.config.get('scale_features', True):
            return df

        numerical_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = self.config.get('exclude_from_scaling', [])
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

        if not numerical_cols:
            return df

        if 'scaler' not in self.scalers:
            if fit:
                self.scalers['scaler'] = StandardScaler()
                df[numerical_cols] = self.scalers['scaler'].fit_transform(df[numerical_cols])
            else:
                # Skip if scaler doesn't exist
                return df
        else:
            df[numerical_cols] = self.scalers['scaler'].transform(df[numerical_cols])

        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features"""
        # Custom feature engineering based on config
        feature_engineering = self.config.get('feature_engineering', {})

        # Polynomial features
        if feature_engineering.get('polynomial', False):
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols[:3]:  # Limit to first 3 columns
                df[f'{col}_squared'] = df[col] ** 2

        # Interaction features
        if feature_engineering.get('interactions', False):
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) >= 2:
                df[f'{numerical_cols[0]}_x_{numerical_cols[1]}'] = \
                    df[numerical_cols[0]] * df[numerical_cols[1]]

        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        return df

    def save_transformers(self, output_dir: Path):
        """Save fitted transformers"""
        import pickle

        transformers = {
            'scalers': self.scalers,
            'encoders': self.encoders
        }

        output_path = output_dir / 'transformers.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(transformers, f)

        logger.info(f"Saved transformers to {output_path}")

    def load_transformers(self, input_dir: Path):
        """Load fitted transformers"""
        import pickle

        input_path = input_dir / 'transformers.pkl'
        with open(input_path, 'rb') as f:
            transformers = pickle.load(f)

        self.scalers = transformers['scalers']
        self.encoders = transformers['encoders']

        logger.info(f"Loaded transformers from {input_path}")


class DatasetPreparer:
    """Main dataset preparation orchestrator"""

    def __init__(self, data_dir: str, output_dir: str,
                 schema_path: Optional[str] = None,
                 config: Optional[Dict] = None,
                 workers: int = 4):
        """Initialize dataset preparer"""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.schema_path = schema_path
        self.config = config or {}
        self.workers = workers

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.validator = DataValidator(schema_path) if schema_path else None
        self.preprocessor = DataPreprocessor(self.config.get('preprocessing', {}))

    def validate_data(self) -> bool:
        """Validate all data files"""
        logger.info(f"{Colors.BOLD}{Colors.CYAN}Validating Data{Colors.RESET}")
        logger.info("=" * 80)

        if not self.validator:
            logger.warning("No schema provided, skipping validation")
            return True

        # Find all CSV files
        data_files = list(self.data_dir.glob('**/*.csv'))
        if not data_files:
            logger.error(f"No CSV files found in {self.data_dir}")
            return False

        all_valid = True

        for data_file in data_files:
            logger.info(f"\nValidating: {data_file.name}")

            try:
                df = pd.read_csv(data_file)
                result = self.validator.validate_dataframe(df)

                if result['valid']:
                    logger.info(f"{Colors.GREEN}✓ Validation passed{Colors.RESET}")
                    logger.info(f"  Rows: {result['row_count']}")
                    logger.info(f"  Columns: {result['column_count']}")
                    logger.info(f"  Memory: {result['memory_usage']:.2f} MB")
                else:
                    logger.error(f"{Colors.RED}✗ Validation failed{Colors.RESET}")
                    for issue in result['issues']:
                        logger.error(f"  - {issue}")
                    all_valid = False

                if result['warnings']:
                    for warning in result['warnings']:
                        logger.warning(f"  ⚠ {warning}")

            except Exception as e:
                logger.error(f"{Colors.RED}Error validating {data_file.name}: {e}{Colors.RESET}")
                all_valid = False

        return all_valid

    def preprocess_data(self) -> bool:
        """Preprocess all data files"""
        logger.info(f"\n{Colors.BOLD}{Colors.CYAN}Preprocessing Data{Colors.RESET}")
        logger.info("=" * 80)

        # Find all CSV files
        data_files = list(self.data_dir.glob('**/*.csv'))
        if not data_files:
            logger.error(f"No CSV files found in {self.data_dir}")
            return False

        for data_file in data_files:
            logger.info(f"\nProcessing: {data_file.name}")

            try:
                # Load data
                df = pd.read_csv(data_file)
                logger.info(f"  Loaded {len(df)} rows")

                # Preprocess
                is_train = 'train' in data_file.name.lower()
                df_processed = self.preprocessor.preprocess(df, fit=is_train)
                logger.info(f"  Processed to {len(df_processed)} rows")

                # Save processed data
                output_path = self.output_dir / 'processed' / data_file.name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df_processed.to_csv(output_path, index=False)

                logger.info(f"{Colors.GREEN}✓ Saved to {output_path}{Colors.RESET}")

            except Exception as e:
                logger.error(f"{Colors.RED}Error processing {data_file.name}: {e}{Colors.RESET}")
                return False

        # Save transformers
        self.preprocessor.save_transformers(self.output_dir / 'processed')

        return True

    def create_splits(self, split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> bool:
        """Create train/val/test splits"""
        logger.info(f"\n{Colors.BOLD}{Colors.CYAN}Creating Data Splits{Colors.RESET}")
        logger.info("=" * 80)
        logger.info(f"Split ratio: {split_ratio[0]:.0%} train, "
                   f"{split_ratio[1]:.0%} val, {split_ratio[2]:.0%} test")

        # Find processed data
        processed_dir = self.output_dir / 'processed'
        data_files = list(processed_dir.glob('*.csv'))

        if not data_files:
            logger.error("No processed data found")
            return False

        # Load all data
        dfs = []
        for data_file in data_files:
            df = pd.read_csv(data_file)
            dfs.append(df)

        df_all = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total samples: {len(df_all)}")

        # Get target column
        target_col = self.config.get('target_column', 'target')
        if target_col not in df_all.columns:
            logger.error(f"Target column '{target_col}' not found")
            return False

        # First split: train and temp (val + test)
        train_ratio = split_ratio[0]
        temp_ratio = 1 - train_ratio

        df_train, df_temp = train_test_split(
            df_all,
            train_size=train_ratio,
            random_state=42,
            stratify=df_all[target_col] if df_all[target_col].dtype in ['object', 'int64'] else None
        )

        # Second split: val and test
        val_ratio = split_ratio[1] / temp_ratio

        df_val, df_test = train_test_split(
            df_temp,
            train_size=val_ratio,
            random_state=42,
            stratify=df_temp[target_col] if df_temp[target_col].dtype in ['object', 'int64'] else None
        )

        # Save splits
        splits_dir = self.output_dir / 'splits'
        splits_dir.mkdir(parents=True, exist_ok=True)

        df_train.to_csv(splits_dir / 'train.csv', index=False)
        df_val.to_csv(splits_dir / 'val.csv', index=False)
        df_test.to_csv(splits_dir / 'test.csv', index=False)

        logger.info(f"\n{Colors.GREEN}✓ Created splits:{Colors.RESET}")
        logger.info(f"  Train: {len(df_train)} samples ({len(df_train)/len(df_all):.1%})")
        logger.info(f"  Val: {len(df_val)} samples ({len(df_val)/len(df_all):.1%})")
        logger.info(f"  Test: {len(df_test)} samples ({len(df_test)/len(df_all):.1%})")

        # Save split info
        split_info = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(df_all),
            'train_samples': len(df_train),
            'val_samples': len(df_val),
            'test_samples': len(df_test),
            'split_ratio': split_ratio,
            'target_column': target_col
        }

        with open(splits_dir / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)

        return True

    def generate_statistics(self):
        """Generate dataset statistics"""
        logger.info(f"\n{Colors.BOLD}{Colors.CYAN}Generating Statistics{Colors.RESET}")
        logger.info("=" * 80)

        stats = {}

        # Check each split
        splits_dir = self.output_dir / 'splits'
        if splits_dir.exists():
            for split_file in ['train.csv', 'val.csv', 'test.csv']:
                split_path = splits_dir / split_file
                if not split_path.exists():
                    continue

                df = pd.read_csv(split_path)
                split_name = split_file.replace('.csv', '')

                stats[split_name] = {
                    'num_samples': len(df),
                    'num_features': len(df.columns),
                    'feature_types': df.dtypes.value_counts().to_dict(),
                    'missing_values': df.isnull().sum().to_dict(),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1e6
                }

                # Numerical statistics
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                if len(numerical_cols) > 0:
                    stats[split_name]['numerical_stats'] = df[numerical_cols].describe().to_dict()

                logger.info(f"\n{split_name.capitalize()} Set:")
                logger.info(f"  Samples: {stats[split_name]['num_samples']}")
                logger.info(f"  Features: {stats[split_name]['num_features']}")
                logger.info(f"  Memory: {stats[split_name]['memory_usage_mb']:.2f} MB")

        # Save statistics
        stats_path = self.output_dir / 'statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)

        logger.info(f"\n{Colors.GREEN}✓ Statistics saved to {stats_path}{Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(
        description='Data validation and preprocessing pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--data-dir', type=str, required=True,
                       help='Input data directory')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for processed data')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation only')
    parser.add_argument('--preprocess', action='store_true',
                       help='Run preprocessing')
    parser.add_argument('--split', action='store_true',
                       help='Create train/val/test splits')
    parser.add_argument('--split-ratio', type=str, default='0.7,0.15,0.15',
                       help='Train/val/test split ratio (default: 0.7,0.15,0.15)')
    parser.add_argument('--schema', type=str, default=None,
                       help='Data schema file for validation')
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration file')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of worker processes')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Load config
    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Parse split ratio
    split_ratio = tuple(float(x) for x in args.split_ratio.split(','))
    if len(split_ratio) != 3 or sum(split_ratio) != 1.0:
        logger.error("Split ratio must be three numbers that sum to 1.0")
        sys.exit(1)

    # Create preparer
    preparer = DatasetPreparer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        schema_path=args.schema,
        config=config,
        workers=args.workers
    )

    # Run operations
    success = True

    if args.validate or not (args.preprocess or args.split):
        success = preparer.validate_data()
        if not success:
            logger.error("\nValidation failed!")
            sys.exit(1)

    if args.preprocess:
        success = preparer.preprocess_data()
        if not success:
            logger.error("\nPreprocessing failed!")
            sys.exit(1)

    if args.split:
        success = preparer.create_splits(split_ratio)
        if not success:
            logger.error("\nSplit creation failed!")
            sys.exit(1)

        # Generate statistics
        preparer.generate_statistics()

    if success:
        logger.info(f"\n{Colors.GREEN}{Colors.BOLD}✓ Data preparation completed successfully!{Colors.RESET}")
    else:
        logger.error(f"\n{Colors.RED}{Colors.BOLD}✗ Data preparation failed!{Colors.RESET}")
        sys.exit(1)


if __name__ == '__main__':
    main()
