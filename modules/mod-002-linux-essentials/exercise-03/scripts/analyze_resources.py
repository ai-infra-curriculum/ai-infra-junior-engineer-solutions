#!/usr/bin/env python3
"""
Resource Usage Analysis Tool

Analyzes resource usage data collected by monitor_resources.sh
"""

import csv
import sys
import os
from datetime import datetime


def analyze_resource_usage(csv_file):
    """Analyze resource usage from CSV file"""

    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}", file=sys.stderr)
        return 1

    cpu_values = []
    mem_values = []
    rss_values = []
    vsz_values = []
    timestamps = []

    # Read CSV data
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    cpu_values.append(float(row['cpu_percent']))
                    mem_values.append(float(row['mem_percent']))
                    rss_values.append(float(row['mem_rss_mb']))
                    vsz_values.append(float(row['mem_vsz_mb']))
                    timestamps.append(row['timestamp'])
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping invalid row: {e}", file=sys.stderr)
                    continue
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        return 1

    if not cpu_values:
        print("Error: No valid data to analyze", file=sys.stderr)
        return 1

    # Calculate statistics
    num_samples = len(cpu_values)
    duration_seconds = num_samples * 2  # Assuming 2-second intervals

    # CPU statistics
    cpu_avg = sum(cpu_values) / len(cpu_values)
    cpu_max = max(cpu_values)
    cpu_min = min(cpu_values)

    # Memory statistics
    mem_avg = sum(mem_values) / len(mem_values)
    mem_max = max(mem_values)
    mem_min = min(mem_values)

    # RSS statistics
    rss_avg = sum(rss_values) / len(rss_values)
    rss_max = max(rss_values)
    rss_min = min(rss_values)
    rss_growth = rss_values[-1] - rss_values[0] if len(rss_values) > 1 else 0

    # VSZ statistics
    vsz_avg = sum(vsz_values) / len(vsz_values)
    vsz_max = max(vsz_values)
    vsz_min = min(vsz_values)

    # Print analysis
    print("=" * 70)
    print("Resource Usage Analysis")
    print("=" * 70)
    print()

    print(f"File: {csv_file}")
    print(f"Samples: {num_samples}")
    print(f"Duration: {duration_seconds} seconds ({duration_seconds/60:.1f} minutes)")
    print(f"Start time: {timestamps[0] if timestamps else 'N/A'}")
    print(f"End time: {timestamps[-1] if timestamps else 'N/A'}")
    print()

    print("-" * 70)
    print("CPU Usage")
    print("-" * 70)
    print(f"  Average:  {cpu_avg:6.2f}%")
    print(f"  Maximum:  {cpu_max:6.2f}%")
    print(f"  Minimum:  {cpu_min:6.2f}%")
    print()

    print("-" * 70)
    print("Memory Usage (Percent)")
    print("-" * 70)
    print(f"  Average:  {mem_avg:6.2f}%")
    print(f"  Maximum:  {mem_max:6.2f}%")
    print(f"  Minimum:  {mem_min:6.2f}%")
    print()

    print("-" * 70)
    print("Memory (RSS - Resident Set Size)")
    print("-" * 70)
    print(f"  Average:  {rss_avg:7.2f} MB")
    print(f"  Maximum:  {rss_max:7.2f} MB")
    print(f"  Minimum:  {rss_min:7.2f} MB")
    print(f"  Growth:   {rss_growth:+7.2f} MB")
    print()

    print("-" * 70)
    print("Memory (VSZ - Virtual Memory Size)")
    print("-" * 70)
    print(f"  Average:  {vsz_avg:7.2f} MB")
    print(f"  Maximum:  {vsz_max:7.2f} MB")
    print(f"  Minimum:  {vsz_min:7.2f} MB")
    print()

    # Recommendations
    print("=" * 70)
    print("Analysis & Recommendations")
    print("=" * 70)
    print()

    if cpu_avg > 80:
        print("⚠  CPU: High average CPU usage (>80%)")
        print("   → Consider: CPU optimization, reducing batch size")
    elif cpu_avg > 50:
        print("✓  CPU: Moderate CPU usage (50-80%)")
        print("   → Good CPU utilization")
    else:
        print("ℹ  CPU: Low CPU usage (<50%)")
        print("   → May indicate I/O bottleneck or inefficient code")

    print()

    if mem_avg > 80:
        print("⚠  Memory: High memory usage (>80%)")
        print("   → Risk of OOM killer")
        print("   → Consider: Reducing batch size, adding swap")
    elif mem_avg > 50:
        print("✓  Memory: Moderate memory usage (50-80%)")
    else:
        print("ℹ  Memory: Low memory usage (<50%)")
        print("   → Underutilizing available memory")

    print()

    if rss_growth > 100:
        print("⚠  Memory Growth: Significant RSS growth (>100MB)")
        print("   → Possible memory leak")
        print("   → Monitor over longer period")
    elif rss_growth > 50:
        print("ℹ  Memory Growth: Moderate RSS growth (50-100MB)")
        print("   → Normal for ML training with accumulating state")
    else:
        print("✓  Memory Growth: Stable RSS (<50MB growth)")
        print("   → Good memory management")

    print()
    print("=" * 70)

    return 0


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: analyze_resources.py <csv_file>", file=sys.stderr)
        print("\nAnalyzes resource usage data from monitor_resources.sh", file=sys.stderr)
        print("\nExample:", file=sys.stderr)
        print("  ./analyze_resources.py resource_usage.csv", file=sys.stderr)
        return 1

    csv_file = sys.argv[1]
    return analyze_resource_usage(csv_file)


if __name__ == "__main__":
    sys.exit(main())
