# Step-by-Step Implementation Guide: Data Processing

## Overview

Master data processing techniques for AI infrastructure: pandas, data cleaning, aggregation, and metrics processing.

**Time**: 3-4 hours | **Difficulty**: Intermediate

---

## Phase 1: pandas Basics (1 hour)

### Step 1: Setup and Basic Operations

```python
# data_processing/pandas_basics.py
import pandas as pd
import numpy as np

# Create DataFrame
data = {
    'model': ['BERT', 'GPT-3', 'T5', 'BERT', 'GPT-3'],
    'accuracy': [0.95, 0.93, 0.94, 0.96, 0.92],
    'latency_ms': [45, 120, 80, 42, 115],
    'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02', '2024-01-03'])
}

df = pd.DataFrame(data)
print(df)

# Basic info
print(df.info())
print(df.describe())

# Selection
print(df['model'])  # Column
print(df[df['accuracy'] > 0.94])  # Filter rows
print(df.loc[0])  # Row by index
print(df.iloc[0:2])  # Slice

# Add column
df['efficiency'] = df['accuracy'] / (df['latency_ms'] / 1000)
print(df)
```

---

## Phase 2: Data Cleaning (1 hour)

### Step 2: Handle Missing Data

```python
# data_processing/cleaning.py
import pandas as pd
import numpy as np

# Create data with missing values
data = {
    'model': ['BERT', 'GPT-3', None, 'T5'],
    'accuracy': [0.95, None, 0.94, 0.92],
    'latency': [45, 120, 80, None]
}
df = pd.DataFrame(data)

print("Missing values:")
print(df.isnull().sum())

# Drop rows with any missing values
df_dropped = df.dropna()

# Fill missing values
df_filled = df.fillna({
    'model': 'Unknown',
    'accuracy': df['accuracy'].mean(),
    'latency': df['latency'].median()
})

print("\nAfter filling:")
print(df_filled)

# Forward fill / backward fill
df['accuracy'].fillna(method='ffill', inplace=True)
```

### Step 3: Data Validation and Cleaning

```python
# data_processing/validation.py
import pandas as pd

def clean_metrics_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate metrics data."""

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove invalid values
    df = df[df['accuracy'] >= 0]
    df = df[df['accuracy'] <= 1]
    df = df[df['latency_ms'] > 0]

    # Standardize model names
    df['model'] = df['model'].str.upper().str.strip()

    # Convert types
    df['accuracy'] = df['accuracy'].astype(float)
    df['latency_ms'] = df['latency_ms'].astype(int)

    return df

# Test
dirty_data = pd.DataFrame({
    'model': [' bert ', 'GPT-3', 'BERT', 't5  '],
    'accuracy': [0.95, 1.2, 0.95, -0.1],
    'latency_ms': [45, 120, 45, -10]
})

clean_data = clean_metrics_data(dirty_data)
print(clean_data)
```

---

## Phase 3: Data Aggregation (1 hour)

### Step 4: GroupBy Operations

```python
# data_processing/aggregation.py
import pandas as pd

# Sample data
data = {
    'model': ['BERT', 'BERT', 'GPT-3', 'GPT-3', 'T5', 'T5'],
    'date': pd.to_datetime(['2024-01-01'] * 6),
    'accuracy': [0.95, 0.96, 0.93, 0.92, 0.94, 0.95],
    'requests': [1000, 1200, 800, 900, 1100, 1050]
}
df = pd.DataFrame(data)

# Group by model
grouped = df.groupby('model').agg({
    'accuracy': ['mean', 'std', 'min', 'max'],
    'requests': ['sum', 'mean']
})

print(grouped)

# Multiple grouping
df['hour'] = pd.to_datetime(df['date']).dt.hour
multi_group = df.groupby(['model', 'hour'])['requests'].sum()
print(multi_group)
```

### Step 5: Pivot Tables

```python
# data_processing/pivot.py
import pandas as pd

# Sample data
data = {
    'date': pd.date_range('2024-01-01', periods=6, freq='D').repeat(2),
    'model': ['BERT', 'GPT-3'] * 6,
    'accuracy': [0.95, 0.93, 0.96, 0.92, 0.94, 0.91, 0.95, 0.93, 0.96, 0.92, 0.94, 0.93]
}
df = pd.DataFrame(data)

# Create pivot table
pivot = df.pivot_table(
    values='accuracy',
    index='date',
    columns='model',
    aggfunc='mean'
)

print(pivot)
```

---

## Phase 4: Time Series Processing (1 hour)

### Step 6: Time Series Operations

```python
# data_processing/timeseries.py
import pandas as pd
import numpy as np

# Generate time series data
dates = pd.date_range('2024-01-01', periods=100, freq='H')
df = pd.DataFrame({
    'timestamp': dates,
    'requests': np.random.randint(100, 1000, 100),
    'latency': np.random.uniform(10, 100, 100)
})

df.set_index('timestamp', inplace=True)

# Resample to daily
daily = df.resample('D').agg({
    'requests': 'sum',
    'latency': 'mean'
})

print("Daily aggregation:")
print(daily)

# Rolling average
df['latency_ma'] = df['latency'].rolling(window=24).mean()

# Detect anomalies (simple method)
mean = df['latency'].mean()
std = df['latency'].std()
df['is_anomaly'] = np.abs(df['latency'] - mean) > 3 * std

print(f"\nAnomalies detected: {df['is_anomaly'].sum()}")
```

---

## Phase 5: Metrics Aggregator Project (30-45 minutes)

### Step 7: Build Metrics Aggregator

```python
# metrics_aggregator.py
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

class MetricsAggregator:
    """Aggregate and analyze ML model metrics."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.metrics = None

    def load_metrics(self, pattern: str = "*.csv"):
        """Load metrics from CSV files."""
        dfs = []
        for file in self.data_dir.glob(pattern):
            df = pd.read_csv(file)
            df['source_file'] = file.name
            dfs.append(df)

        if dfs:
            self.metrics = pd.concat(dfs, ignore_index=True)
            self.metrics['timestamp'] = pd.to_datetime(self.metrics['timestamp'])
            return self.metrics
        return pd.DataFrame()

    def calculate_daily_stats(self):
        """Calculate daily statistics."""
        if self.metrics is None or self.metrics.empty:
            return pd.DataFrame()

        daily = self.metrics.groupby([
            pd.Grouper(key='timestamp', freq='D'),
            'model'
        ]).agg({
            'accuracy': ['mean', 'std', 'min', 'max'],
            'latency_ms': ['mean', 'p95', 'p99'],
            'requests': 'sum'
        }).reset_index()

        return daily

    def detect_degradation(self, threshold: float = 0.05):
        """Detect model performance degradation."""
        if self.metrics is None or self.metrics.empty:
            return []

        # Calculate 7-day rolling average
        self.metrics = self.metrics.sort_values('timestamp')
        self.metrics['accuracy_ma'] = (
            self.metrics.groupby('model')['accuracy']
            .rolling(window=7, min_periods=1)
            .mean()
            .reset_index(drop=True)
        )

        # Detect drops
        degraded = []
        for model in self.metrics['model'].unique():
            model_data = self.metrics[self.metrics['model'] == model]
            recent_avg = model_data.tail(7)['accuracy'].mean()
            baseline_avg = model_data.head(30)['accuracy'].mean()

            if baseline_avg - recent_avg > threshold:
                degraded.append({
                    'model': model,
                    'baseline': baseline_avg,
                    'recent': recent_avg,
                    'drop': baseline_avg - recent_avg
                })

        return degraded

    def export_report(self, output_path: str):
        """Export aggregated metrics report."""
        if self.metrics is None or self.metrics.empty:
            return

        report = {
            'summary': self.metrics.groupby('model').agg({
                'accuracy': ['mean', 'std'],
                'latency_ms': ['mean', 'p95'],
                'requests': 'sum'
            }),
            'daily_stats': self.calculate_daily_stats(),
            'degradation': self.detect_degradation()
        }

        # Save to Excel
        with pd.ExcelWriter(output_path) as writer:
            report['summary'].to_excel(writer, sheet_name='Summary')
            report['daily_stats'].to_excel(writer, sheet_name='Daily Stats')
            pd.DataFrame(report['degradation']).to_excel(writer, sheet_name='Degradation')

# Usage
aggregator = MetricsAggregator('data/metrics/')
aggregator.load_metrics()
aggregator.export_report('report.xlsx')
```

---

## Summary

**What You Built**:
- ✅ pandas data manipulation
- ✅ Data cleaning and validation
- ✅ GroupBy aggregations
- ✅ Time series processing
- ✅ Complete metrics aggregator
- ✅ Anomaly detection
- ✅ Report generation

**Key pandas Operations**:
```python
# Selection
df[df['col'] > 10]

# GroupBy
df.groupby('category').agg({'value': 'mean'})

# Time series
df.resample('D').mean()

# Pivot
df.pivot_table(values='metric', index='date', columns='model')
```

**Next Steps**:
- Study NumPy for numerical computing
- Learn advanced pandas techniques
- Practice with real datasets
- Explore data visualization (matplotlib, seaborn)
