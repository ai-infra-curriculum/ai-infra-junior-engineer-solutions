"""Data validation module for ensuring data quality."""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a data validation issue."""

    severity: str  # "error", "warning"
    category: str  # "missing_values", "duplicates", etc.
    message: str
    details: Optional[Dict[str, Any]] = None


class DataValidator:
    """Validate data quality and schema."""

    def __init__(
        self,
        max_missing_ratio: float = 0.3,
        max_duplicate_ratio: float = 0.1,
        required_columns: Optional[List[str]] = None
    ):
        """Initialize data validator.

        Args:
            max_missing_ratio: Maximum allowed ratio of missing values
            max_duplicate_ratio: Maximum allowed ratio of duplicate rows
            required_columns: List of required column names
        """
        self.max_missing_ratio = max_missing_ratio
        self.max_duplicate_ratio = max_duplicate_ratio
        self.required_columns = required_columns or []
        logger.info("DataValidator initialized")

    def check_schema(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check if required columns exist.

        Args:
            df: DataFrame to validate

        Returns:
            List of validation issues
        """
        issues = []
        missing_cols = set(self.required_columns) - set(df.columns)

        if missing_cols:
            issues.append(ValidationIssue(
                severity="error",
                category="schema",
                message=f"Missing required columns: {missing_cols}",
                details={"missing_columns": list(missing_cols)}
            ))

        return issues

    def check_missing_values(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check for excessive missing values.

        Args:
            df: DataFrame to validate

        Returns:
            List of validation issues
        """
        issues = []
        missing_ratios = df.isnull().sum() / len(df)

        for col, ratio in missing_ratios.items():
            if ratio > self.max_missing_ratio:
                issues.append(ValidationIssue(
                    severity="error",
                    category="missing_values",
                    message=f"Column '{col}' has {ratio:.1%} missing values",
                    details={"column": col, "ratio": float(ratio)}
                ))
            elif ratio > 0:
                issues.append(ValidationIssue(
                    severity="warning",
                    category="missing_values",
                    message=f"Column '{col}' has {ratio:.1%} missing values",
                    details={"column": col, "ratio": float(ratio)}
                ))

        return issues

    def check_duplicates(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check for duplicate rows.

        Args:
            df: DataFrame to validate

        Returns:
            List of validation issues
        """
        issues = []
        n_duplicates = df.duplicated().sum()
        dup_ratio = n_duplicates / len(df)

        if dup_ratio > self.max_duplicate_ratio:
            issues.append(ValidationIssue(
                severity="error",
                category="duplicates",
                message=f"Found {n_duplicates} duplicate rows ({dup_ratio:.1%})",
                details={"count": int(n_duplicates), "ratio": float(dup_ratio)}
            ))
        elif n_duplicates > 0:
            issues.append(ValidationIssue(
                severity="warning",
                category="duplicates",
                message=f"Found {n_duplicates} duplicate rows ({dup_ratio:.1%})",
                details={"count": int(n_duplicates), "ratio": float(dup_ratio)}
            ))

        return issues

    def check_data_types(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check for unexpected data types.

        Args:
            df: DataFrame to validate

        Returns:
            List of validation issues
        """
        issues = []

        # Check for object columns that might be numeric
        for col in df.select_dtypes(include=['object']).columns:
            try:
                pd.to_numeric(df[col], errors='coerce')
                non_numeric = df[col].apply(
                    lambda x: pd.isna(x) or not str(x).replace('.', '').replace('-', '').isdigit()
                ).sum()

                if non_numeric == 0:
                    issues.append(ValidationIssue(
                        severity="warning",
                        category="data_types",
                        message=f"Column '{col}' is object but appears numeric",
                        details={"column": col, "current_type": "object"}
                    ))
            except (ValueError, TypeError):
                pass

        return issues

    def check_outliers(self, df: pd.DataFrame, z_threshold: float = 3.0) -> List[ValidationIssue]:
        """Check for statistical outliers in numeric columns.

        Args:
            df: DataFrame to validate
            z_threshold: Z-score threshold for outlier detection

        Returns:
            List of validation issues
        """
        issues = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            # Calculate z-scores
            mean = df[col].mean()
            std = df[col].std()

            if std > 0:
                z_scores = np.abs((df[col] - mean) / std)
                n_outliers = (z_scores > z_threshold).sum()

                if n_outliers > 0:
                    outlier_ratio = n_outliers / len(df)
                    issues.append(ValidationIssue(
                        severity="warning",
                        category="outliers",
                        message=f"Column '{col}' has {n_outliers} outliers ({outlier_ratio:.1%})",
                        details={
                            "column": col,
                            "count": int(n_outliers),
                            "ratio": float(outlier_ratio)
                        }
                    ))

        return issues

    def validate(
        self,
        df: pd.DataFrame,
        check_outliers: bool = True
    ) -> Tuple[bool, List[ValidationIssue]]:
        """Run all validation checks.

        Args:
            df: DataFrame to validate
            check_outliers: Whether to check for outliers

        Returns:
            Tuple of (is_valid, list of issues)
        """
        logger.info(f"Validating data: {len(df)} rows, {len(df.columns)} columns")

        all_issues = []

        # Run all checks
        all_issues.extend(self.check_schema(df))
        all_issues.extend(self.check_missing_values(df))
        all_issues.extend(self.check_duplicates(df))
        all_issues.extend(self.check_data_types(df))

        if check_outliers:
            all_issues.extend(self.check_outliers(df))

        # Count errors
        n_errors = sum(1 for issue in all_issues if issue.severity == "error")
        is_valid = n_errors == 0

        logger.info(f"Validation complete: {n_errors} errors, {len(all_issues) - n_errors} warnings")

        return is_valid, all_issues

    def get_validation_report(self, issues: List[ValidationIssue]) -> str:
        """Generate human-readable validation report.

        Args:
            issues: List of validation issues

        Returns:
            Formatted report string
        """
        if not issues:
            return "✓ All validation checks passed"

        errors = [i for i in issues if i.severity == "error"]
        warnings = [i for i in issues if i.severity == "warning"]

        report = []
        report.append("=" * 60)
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 60)

        if errors:
            report.append(f"\nERRORS ({len(errors)}):")
            for i, issue in enumerate(errors, 1):
                report.append(f"  {i}. [{issue.category}] {issue.message}")

        if warnings:
            report.append(f"\nWARNINGS ({len(warnings)}):")
            for i, issue in enumerate(warnings, 1):
                report.append(f"  {i}. [{issue.category}] {issue.message}")

        report.append("=" * 60)

        return "\n".join(report)
