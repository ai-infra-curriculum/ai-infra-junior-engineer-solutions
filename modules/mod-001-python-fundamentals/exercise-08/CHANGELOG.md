# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nothing yet

## [0.1.0] - 2025-10-30

### Added
- Initial release
- Preprocessing utilities
  - `normalize()`: Min-max normalization to [0, 1] range
  - `standardize()`: Z-score standardization (mean=0, std=1)
  - `clip_outliers()`: Clip outliers based on percentiles
  - `check_missing_values()`: Check for None and NaN values
  - `validate_range()`: Validate values within specified range
  - `check_data_types()`: Validate data types consistency
- Classification metrics
  - `accuracy()`: Calculate accuracy score
  - `precision()`: Calculate precision score
  - `recall()`: Calculate recall score
  - `f1_score()`: Calculate F1 score
- Decorators
  - `@timer`: Measure and print function execution time
  - `@timer_with_units()`: Timer with custom time units
  - `@retry()`: Retry function with exponential backoff
  - `@retry_on_condition()`: Retry based on result condition
- Structured logging
  - `StructuredLogger`: JSON-based structured logging
  - `JsonFormatter`: Custom JSON log formatter
  - `log_ml_event()`: ML-specific event logging helper
- Comprehensive test suite with 100% coverage
- Complete documentation and examples

### Changed
- Nothing yet

### Deprecated
- Nothing yet

### Removed
- Nothing yet

### Fixed
- Nothing yet

### Security
- Nothing yet

[Unreleased]: https://github.com/yourorg/ml-infra-utils/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourorg/ml-infra-utils/releases/tag/v0.1.0
