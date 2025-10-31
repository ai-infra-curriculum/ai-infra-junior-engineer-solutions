# Exercise 05: Python for Data Processing

## Overview

Master data processing with NumPy and Pandas for ML infrastructure tasks. Learn efficient data manipulation, aggregation, transformation, and performance optimization techniques.

## Learning Objectives

- ✅ Work with NumPy arrays and operations
- ✅ Manipulate data with Pandas DataFrames
- ✅ Clean and transform data efficiently
- ✅ Aggregate and group data
- ✅ Handle time series data
- ✅ Optimize performance for large datasets
- ✅ Build ML metrics aggregation systems

## Topics Covered

### 1. NumPy Fundamentals

#### Array Creation

```python
import numpy as np

# From list
arr = np.array([1, 2, 3, 4, 5])

# Zeros and ones
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))

# Range
arr = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

# Linspace
arr = np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]

# Random arrays
random_arr = np.random.rand(3, 4)  # Uniform [0, 1)
random_normal = np.random.randn(3, 4)  # Normal distribution
random_int = np.random.randint(0, 100, (3, 4))  # Random integers

# Identity matrix
identity = np.eye(3)

# Empty array
empty = np.empty((2, 3))
```

#### Array Operations

```python
# Element-wise operations
arr = np.array([1, 2, 3, 4])
print(arr + 5)      # [6, 7, 8, 9]
print(arr * 2)      # [2, 4, 6, 8]
print(arr ** 2)     # [1, 4, 9, 16]
print(np.sqrt(arr)) # [1, 1.414, 1.732, 2]

# Array operations
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(arr1 + arr2)  # [5, 7, 9]
print(arr1 * arr2)  # [4, 10, 18]

# Broadcasting
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr + 10)  # Add 10 to all elements

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(A @ B)        # Matrix multiplication
print(A.T)          # Transpose
print(np.linalg.inv(A))  # Inverse
print(np.linalg.det(A))  # Determinant
```

#### Array Indexing and Slicing

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Basic indexing
print(arr[0])       # 1
print(arr[-1])      # 10
print(arr[2:5])     # [3, 4, 5]
print(arr[::2])     # [1, 3, 5, 7, 9]

# Boolean indexing
print(arr[arr > 5])  # [6, 7, 8, 9, 10]
print(arr[(arr > 3) & (arr < 8)])  # [4, 5, 6, 7]

# Fancy indexing
indices = [1, 3, 5]
print(arr[indices])  # [2, 4, 6]

# 2D arrays
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr_2d[0, 1])      # 2
print(arr_2d[1:, :2])    # [[4, 5], [7, 8]]
print(arr_2d[:, 1])      # [2, 5, 8]
```

#### Aggregations and Statistics

```python
arr = np.array([1, 2, 3, 4, 5])

# Basic statistics
print(np.sum(arr))      # 15
print(np.mean(arr))     # 3.0
print(np.median(arr))   # 3.0
print(np.std(arr))      # 1.414
print(np.var(arr))      # 2.0
print(np.min(arr))      # 1
print(np.max(arr))      # 5

# Along axis
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(np.sum(arr_2d, axis=0))  # [5, 7, 9] (column sums)
print(np.sum(arr_2d, axis=1))  # [6, 15] (row sums)

# Cumulative operations
print(np.cumsum(arr))   # [1, 3, 6, 10, 15]
print(np.cumprod(arr))  # [1, 2, 6, 24, 120]

# Percentiles
print(np.percentile(arr, 50))  # 3.0 (median)
print(np.percentile(arr, 25))  # 2.0
print(np.percentile(arr, 75))  # 4.0
```

#### Array Reshaping

```python
arr = np.arange(12)

# Reshape
arr_2d = arr.reshape(3, 4)
arr_3d = arr.reshape(2, 3, 2)

# Flatten
flat = arr_2d.flatten()
flat = arr_2d.ravel()  # View, not copy

# Transpose
transposed = arr_2d.T

# Add dimension
expanded = arr[:, np.newaxis]  # (12,) -> (12, 1)

# Squeeze
squeezed = expanded.squeeze()  # Remove single dimensions
```

### 2. Pandas Fundamentals

#### Series

```python
import pandas as pd

# Create Series
s = pd.Series([1, 2, 3, 4, 5])
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s = pd.Series({'a': 1, 'b': 2, 'c': 3})

# Accessing elements
print(s[0])       # By position
print(s['a'])     # By label
print(s.iloc[0])  # Explicit position
print(s.loc['a']) # Explicit label

# Operations
print(s * 2)
print(s + 10)
print(s[s > 2])
```

#### DataFrame Creation

```python
# From dictionary
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
})

# From list of dictionaries
data = [
    {'name': 'Alice', 'age': 25},
    {'name': 'Bob', 'age': 30}
]
df = pd.DataFrame(data)

# From NumPy array
arr = np.random.rand(3, 4)
df = pd.DataFrame(arr, columns=['A', 'B', 'C', 'D'])

# From CSV
df = pd.read_csv('data.csv')

# From JSON
df = pd.read_json('data.json')
```

#### DataFrame Inspection

```python
# Basic info
print(df.head())        # First 5 rows
print(df.tail(3))       # Last 3 rows
print(df.shape)         # (rows, columns)
print(df.columns)       # Column names
print(df.dtypes)        # Data types
print(df.info())        # Summary
print(df.describe())    # Statistics

# Check for missing values
print(df.isnull().sum())
print(df.isna().sum())

# Memory usage
print(df.memory_usage())
```

#### Selecting Data

```python
# Select columns
print(df['name'])              # Single column (Series)
print(df[['name', 'age']])     # Multiple columns (DataFrame)

# Select rows
print(df.loc[0])               # By label
print(df.iloc[0])              # By position
print(df.loc[0:2])             # Slice by label
print(df.iloc[0:2])            # Slice by position

# Select specific cells
print(df.loc[0, 'name'])       # Row 0, column 'name'
print(df.iloc[0, 1])           # Row 0, column 1

# Boolean indexing
print(df[df['age'] > 25])
print(df[(df['age'] > 25) & (df['city'] == 'NYC')])

# Query method
print(df.query('age > 25'))
print(df.query('age > 25 and city == "NYC"'))
```

#### Data Cleaning

```python
# Handle missing values
df = df.dropna()                    # Drop rows with any NaN
df = df.dropna(axis=1)              # Drop columns with any NaN
df = df.dropna(thresh=2)            # Keep rows with at least 2 non-NaN
df = df.fillna(0)                   # Fill NaN with 0
df = df.fillna(df.mean())           # Fill with mean
df = df.fillna(method='ffill')      # Forward fill
df = df.fillna(method='bfill')      # Backward fill

# Handle duplicates
df = df.drop_duplicates()
df = df.drop_duplicates(subset=['name'])
df = df.drop_duplicates(keep='last')

# Replace values
df = df.replace('old_value', 'new_value')
df = df.replace({'old1': 'new1', 'old2': 'new2'})

# Type conversion
df['age'] = df['age'].astype(int)
df['date'] = pd.to_datetime(df['date'])
df['category'] = df['category'].astype('category')

# Rename columns
df = df.rename(columns={'old_name': 'new_name'})
df.columns = ['col1', 'col2', 'col3']
```

### 3. Data Transformation

#### Adding and Modifying Columns

```python
# Add new column
df['new_col'] = 0
df['sum'] = df['col1'] + df['col2']
df['ratio'] = df['col1'] / df['col2']

# Apply function
df['squared'] = df['value'].apply(lambda x: x ** 2)

# Apply function to DataFrame
def calculate_bmi(row):
    return row['weight'] / (row['height'] ** 2)

df['bmi'] = df.apply(calculate_bmi, axis=1)

# Map values
df['category'] = df['value'].map({1: 'low', 2: 'medium', 3: 'high'})

# Conditional assignment
df['status'] = np.where(df['score'] > 80, 'pass', 'fail')

# Multiple conditions
conditions = [
    df['score'] > 90,
    df['score'] > 80,
    df['score'] > 70
]
choices = ['A', 'B', 'C']
df['grade'] = np.select(conditions, choices, default='F')
```

#### Sorting and Ranking

```python
# Sort by values
df = df.sort_values('age')
df = df.sort_values('age', ascending=False)
df = df.sort_values(['city', 'age'])

# Sort by index
df = df.sort_index()

# Ranking
df['rank'] = df['score'].rank()
df['rank'] = df['score'].rank(method='dense')
df['rank'] = df['score'].rank(ascending=False)
```

#### Binning and Discretization

```python
# Cut into bins
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 18, 30, 50, 100],
    labels=['child', 'young', 'adult', 'senior']
)

# Equal-width bins
df['score_bin'] = pd.cut(df['score'], bins=5)

# Quantile-based bins
df['score_quantile'] = pd.qcut(df['score'], q=4)
```

### 4. Grouping and Aggregation

#### GroupBy Operations

```python
# Group by single column
grouped = df.groupby('city')

# Get group
print(grouped.get_group('NYC'))

# Aggregate
print(grouped['age'].mean())
print(grouped['age'].sum())
print(grouped.agg({'age': 'mean', 'salary': 'sum'}))

# Multiple aggregations
print(grouped['age'].agg(['mean', 'min', 'max', 'std']))

# Custom aggregation
def range_func(x):
    return x.max() - x.min()

print(grouped['age'].agg(range_func))

# Group by multiple columns
grouped = df.groupby(['city', 'department'])
print(grouped['salary'].mean())

# Transform (return same shape)
df['age_normalized'] = grouped['age'].transform(lambda x: (x - x.mean()) / x.std())

# Filter groups
filtered = grouped.filter(lambda x: len(x) > 5)
```

#### Pivot Tables

```python
# Create pivot table
pivot = pd.pivot_table(
    df,
    values='salary',
    index='city',
    columns='department',
    aggfunc='mean'
)

# Multiple aggregations
pivot = pd.pivot_table(
    df,
    values='salary',
    index='city',
    columns='department',
    aggfunc=['mean', 'sum', 'count']
)

# With margins
pivot = pd.pivot_table(
    df,
    values='salary',
    index='city',
    columns='department',
    aggfunc='mean',
    margins=True  # Add totals
)
```

#### Crosstab

```python
# Frequency table
crosstab = pd.crosstab(df['city'], df['department'])

# With percentages
crosstab = pd.crosstab(
    df['city'],
    df['department'],
    normalize='index'  # Row percentages
)
```

### 5. Merging and Joining

#### Concatenation

```python
# Concatenate DataFrames
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

# Vertical concatenation (stack rows)
result = pd.concat([df1, df2], axis=0, ignore_index=True)

# Horizontal concatenation (stack columns)
result = pd.concat([df1, df2], axis=1)
```

#### Merging (SQL-style joins)

```python
# Inner join
merged = pd.merge(df1, df2, on='key')

# Left join
merged = pd.merge(df1, df2, on='key', how='left')

# Right join
merged = pd.merge(df1, df2, on='key', how='right')

# Outer join
merged = pd.merge(df1, df2, on='key', how='outer')

# Multiple keys
merged = pd.merge(df1, df2, on=['key1', 'key2'])

# Different column names
merged = pd.merge(df1, df2, left_on='id', right_on='user_id')
```

### 6. Time Series

#### DateTime Operations

```python
# Create datetime
df['date'] = pd.to_datetime(df['date_string'])

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter

# Date arithmetic
df['tomorrow'] = df['date'] + pd.Timedelta(days=1)
df['last_week'] = df['date'] - pd.Timedelta(weeks=1)

# Time delta
df['days_since'] = (pd.Timestamp.now() - df['date']).dt.days
```

#### Time Series Indexing

```python
# Set datetime index
df = df.set_index('date')

# Select by date
print(df['2024-01'])           # All January 2024
print(df['2024-01-15'])        # Specific date
print(df['2024-01':'2024-03']) # Date range

# Resample
daily_df = df.resample('D').mean()    # Daily average
weekly_df = df.resample('W').sum()    # Weekly sum
monthly_df = df.resample('M').mean()  # Monthly average

# Rolling window
df['rolling_mean'] = df['value'].rolling(window=7).mean()
df['rolling_std'] = df['value'].rolling(window=7).std()

# Expanding window
df['expanding_mean'] = df['value'].expanding().mean()

# Shift (lag)
df['prev_value'] = df['value'].shift(1)
df['next_value'] = df['value'].shift(-1)
df['change'] = df['value'] - df['value'].shift(1)
```

### 7. Performance Optimization

#### Memory Optimization

```python
# Check memory usage
print(df.memory_usage(deep=True))

# Optimize dtypes
df['category_col'] = df['category_col'].astype('category')
df['int_col'] = pd.to_numeric(df['int_col'], downcast='integer')
df['float_col'] = pd.to_numeric(df['float_col'], downcast='float')

# Read CSV with specific dtypes
df = pd.read_csv(
    'data.csv',
    dtype={'col1': 'int32', 'col2': 'category'}
)

# Read in chunks for large files
chunk_size = 10000
chunks = []
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    processed = process_chunk(chunk)
    chunks.append(processed)
df = pd.concat(chunks, ignore_index=True)
```

#### Vectorization

```python
# Avoid loops - use vectorization

# Bad - loop
result = []
for value in df['col']:
    result.append(value * 2)
df['result'] = result

# Good - vectorized
df['result'] = df['col'] * 2

# Use apply only when necessary
# Vectorized operations are much faster
df['result'] = df['col1'] + df['col2']  # Fast
df['result'] = df.apply(lambda row: row['col1'] + row['col2'], axis=1)  # Slower
```

#### Efficient Operations

```python
# Use query for filtering (faster for large DataFrames)
df.query('age > 25 and city == "NYC"')

# Use eval for complex expressions
df.eval('new_col = col1 + col2 * col3')

# Use categorical for repeated strings
df['category'] = df['category'].astype('category')

# Use appropriate join types
# Inner joins are faster than outer joins
```

---

## Project: ML Metrics Aggregation Tool

Build a system to aggregate, analyze, and visualize ML experiment metrics.

### Requirements

**Features:**
1. Load metrics from multiple experiments
2. Aggregate metrics by model, dataset, or time period
3. Calculate statistical summaries
4. Compare model performance
5. Identify best performing models
6. Export results to various formats
7. Generate reports and visualizations

**Technical Requirements:**
- Use pandas for data manipulation
- NumPy for numerical computations
- Efficient processing of large metric files
- Memory-optimized operations
- Comprehensive test coverage
- Type hints throughout

### Implementation

See `solutions/metrics_aggregator.py` for complete implementation.

### Example Usage

```python
from metrics_aggregator import MetricsAggregator

# Initialize aggregator
aggregator = MetricsAggregator()

# Load metrics
aggregator.load_metrics('experiments/results.csv')

# Aggregate by model
model_stats = aggregator.aggregate_by_model()
print(model_stats)

# Compare models
comparison = aggregator.compare_models(['model_a', 'model_b'])

# Find best model
best = aggregator.get_best_model(metric='f1_score')
print(f"Best model: {best}")

# Export report
aggregator.export_report('report.csv')
```

---

## Practice Problems

### Problem 1: Array Operations

```python
def calculate_statistics(arr: np.ndarray) -> dict:
    """
    Calculate comprehensive statistics for array.

    Args:
        arr: NumPy array

    Returns:
        Dictionary with statistics

    Example:
        >>> stats = calculate_statistics(np.array([1, 2, 3, 4, 5]))
        >>> print(stats['mean'])
        3.0
    """
    # Your implementation here
    pass
```

### Problem 2: Data Cleaning

```python
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset: remove duplicates, handle missing values.

    Args:
        df: Input DataFrame

    Returns:
        Cleaned DataFrame
    """
    # Your implementation here
    pass
```

### Problem 3: GroupBy Aggregation

```python
def aggregate_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sales data by product and region.

    Args:
        df: Sales DataFrame with columns: product, region, sales, date

    Returns:
        Aggregated DataFrame with total and average sales
    """
    # Your implementation here
    pass
```

### Problem 4: Time Series Analysis

```python
def calculate_moving_average(
    df: pd.DataFrame,
    column: str,
    window: int
) -> pd.Series:
    """
    Calculate moving average for time series.

    Args:
        df: DataFrame with datetime index
        column: Column to calculate MA for
        window: Window size

    Returns:
        Series with moving averages
    """
    # Your implementation here
    pass
```

### Problem 5: Performance Optimization

```python
def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage.

    Args:
        df: Input DataFrame

    Returns:
        Memory-optimized DataFrame
    """
    # Your implementation here
    pass
```

---

## Best Practices

### 1. Use Vectorized Operations

```python
# Good - vectorized
df['result'] = df['col1'] * df['col2'] + df['col3']

# Bad - loop
result = []
for i in range(len(df)):
    result.append(df['col1'].iloc[i] * df['col2'].iloc[i] + df['col3'].iloc[i])
```

### 2. Chain Operations

```python
# Good - chained
result = (
    df
    .dropna()
    .query('age > 25')
    .groupby('city')
    ['salary']
    .mean()
    .sort_values(ascending=False)
)

# Less ideal - multiple steps
df = df.dropna()
df = df.query('age > 25')
grouped = df.groupby('city')
result = grouped['salary'].mean()
result = result.sort_values(ascending=False)
```

### 3. Use Categories for Repeated Strings

```python
# Good - categorical
df['category'] = df['category'].astype('category')

# Less efficient for repeated values
df['category'] = df['category'].astype('string')
```

### 4. Use appropriate dtypes

```python
# Good - specific dtypes
df = pd.read_csv('data.csv', dtype={
    'id': 'int32',
    'value': 'float32',
    'category': 'category'
})

# Less efficient
df = pd.read_csv('data.csv')  # Infers types
```

---

## Common Pitfalls

### 1. SettingWithCopyWarning

```python
# Bad - chained assignment
df[df['age'] > 25]['salary'] = 50000  # Warning!

# Good - use loc
df.loc[df['age'] > 25, 'salary'] = 50000
```

### 2. Inefficient Loops

```python
# Bad - iterating over DataFrame
for i in range(len(df)):
    df.loc[i, 'result'] = df.loc[i, 'col1'] * 2

# Good - vectorized
df['result'] = df['col1'] * 2
```

### 3. Loading Entire Large Files

```python
# Bad - load entire file
df = pd.read_csv('huge_file.csv')

# Good - use chunks
for chunk in pd.read_csv('huge_file.csv', chunksize=10000):
    process(chunk)
```

---

## Validation

Run the validation script:

```bash
python tests/test_metrics_aggregator.py
```

Expected output:
```
✅ NumPy operations correct
✅ Pandas manipulations working
✅ Aggregations accurate
✅ Performance optimized
✅ Memory usage efficient

🎉 Exercise 05 Complete!
```

---

## Resources

### Documentation
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)

### Books
- "Python for Data Analysis" by Wes McKinney
- "NumPy Essentials" by Leo (Liang-Huan) Chin
- "Effective Pandas" by Matt Harrison

### Tutorials
- [NumPy Quickstart Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [10 Minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Pandas Cookbook](https://pandas.pydata.org/docs/user_guide/cookbook.html)

---

## Next Steps

After completing this exercise:

1. **Module 003: Linux & Command Line** - System administration
2. Practice data processing daily
3. Work with real-world datasets
4. Optimize for performance
5. Learn data visualization (Matplotlib, Seaborn)

---

**Process data efficiently and power your ML infrastructure! 📊**
