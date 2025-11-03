#!/usr/bin/env python3
"""
S3 Storage Optimization Tool

Analyze S3 storage usage and apply optimization strategies:
- Lifecycle policies for automated tiering
- Compression recommendations
- Duplicate detection
- Cost analysis by storage class

Usage:
    python optimize_storage.py --analyze
    python optimize_storage.py --apply-lifecycle --bucket my-bucket
    python optimize_storage.py --find-duplicates --bucket my-bucket
    python optimize_storage.py --compress --bucket my-bucket
"""

import argparse
import sys
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List
from collections import defaultdict
import boto3


# S3 storage class pricing (per GB/month, us-east-1)
STORAGE_PRICING = {
    'STANDARD': 0.023,
    'INTELLIGENT_TIERING': 0.0125,  # Average
    'STANDARD_IA': 0.0125,
    'ONEZONE_IA': 0.01,
    'GLACIER': 0.004,
    'GLACIER_IR': 0.004,
    'DEEP_ARCHIVE': 0.00099
}


def get_bucket_size_by_storage_class(bucket_name: str) -> Dict:
    """
    Get bucket size broken down by storage class.

    Args:
        bucket_name: S3 bucket name

    Returns:
        Dict with storage class -> size in bytes
    """
    cloudwatch = boto3.client('cloudwatch')

    storage_classes = [
        'StandardStorage',
        'IntelligentTieringStorage',
        'StandardIAStorage',
        'OneZoneIAStorage',
        'GlacierStorage',
        'GlacierInstantRetrievalStorage',
        'DeepArchiveStorage'
    ]

    sizes = {}
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=1)

    for storage_class in storage_classes:
        try:
            response = cloudwatch.get_metric_statistics(
                Namespace='AWS/S3',
                MetricName='BucketSizeBytes',
                Dimensions=[
                    {'Name': 'BucketName', 'Value': bucket_name},
                    {'Name': 'StorageType', 'Value': storage_class}
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=86400,  # 1 day
                Statistics=['Average']
            )

            if response['Datapoints']:
                size_bytes = response['Datapoints'][0]['Average']
                sizes[storage_class] = size_bytes
        except Exception as e:
            print(f"Warning: Could not fetch {storage_class} for {bucket_name}: {e}")

    return sizes


def get_object_count(bucket_name: str) -> int:
    """Get total object count in bucket."""
    cloudwatch = boto3.client('cloudwatch')

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=1)

    response = cloudwatch.get_metric_statistics(
        Namespace='AWS/S3',
        MetricName='NumberOfObjects',
        Dimensions=[
            {'Name': 'BucketName', 'Value': bucket_name},
            {'Name': 'StorageType', 'Value': 'AllStorageTypes'}
        ],
        StartTime=start_time,
        EndTime=end_time,
        Period=86400,
        Statistics=['Average']
    )

    if response['Datapoints']:
        return int(response['Datapoints'][0]['Average'])
    return 0


def calculate_storage_cost(sizes: Dict) -> Dict:
    """Calculate monthly cost for each storage class."""
    costs = {}
    total_cost = 0

    for storage_class, size_bytes in sizes.items():
        size_gb = size_bytes / (1024**3)

        # Map CloudWatch storage type to pricing key
        pricing_key = storage_class.replace('Storage', '').replace('StandardIA', 'STANDARD_IA').replace('OneZoneIA', 'ONEZONE_IA').replace('Glacier', 'GLACIER').replace('DeepArchive', 'DEEP_ARCHIVE').replace('IntelligentTiering', 'INTELLIGENT_TIERING').upper()

        if pricing_key == 'GLACIERINSTANTRETRIEVAL':
            pricing_key = 'GLACIER_IR'

        price_per_gb = STORAGE_PRICING.get(pricing_key, 0.023)  # Default to STANDARD
        monthly_cost = size_gb * price_per_gb

        costs[storage_class] = {
            'size_gb': size_gb,
            'price_per_gb': price_per_gb,
            'monthly_cost': monthly_cost
        }

        total_cost += monthly_cost

    return costs, total_cost


def analyze_bucket(bucket_name: str) -> Dict:
    """Analyze a single bucket."""
    print(f"Analyzing bucket: {bucket_name}...", end=' ')

    # Get sizes by storage class
    sizes = get_bucket_size_by_storage_class(bucket_name)

    if not sizes:
        print("No data")
        return None

    # Calculate costs
    costs, total_cost = calculate_storage_cost(sizes)

    # Get object count
    object_count = get_object_count(bucket_name)

    print(f"${total_cost:,.2f}/month")

    return {
        'bucket_name': bucket_name,
        'sizes': sizes,
        'costs': costs,
        'total_cost': total_cost,
        'object_count': object_count
    }


def recommend_lifecycle_policy(bucket_name: str, prefix: str = '') -> Dict:
    """
    Recommend lifecycle policy based on bucket contents and naming.

    Args:
        bucket_name: S3 bucket name
        prefix: Object prefix to analyze

    Returns:
        Dict with lifecycle policy recommendation
    """
    # Heuristics based on bucket name/prefix
    bucket_lower = bucket_name.lower()
    prefix_lower = prefix.lower()

    # Training data: transition after 30 days, archive after 90 days
    if 'training' in bucket_lower or 'train' in prefix_lower:
        return {
            'name': 'Training Data Lifecycle',
            'policy': {
                'Rules': [{
                    'Id': 'training-data-lifecycle',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': prefix},
                    'Transitions': [
                        {'Days': 30, 'StorageClass': 'INTELLIGENT_TIERING'},
                        {'Days': 90, 'StorageClass': 'GLACIER'}
                    ],
                    'Expiration': {'Days': 365}
                }]
            },
            'savings_estimate': '40-50% after 90 days'
        }

    # Models: transition after 7 days to IT, archive after 90 days
    elif 'model' in bucket_lower or 'model' in prefix_lower:
        return {
            'name': 'Model Artifacts Lifecycle',
            'policy': {
                'Rules': [{
                    'Id': 'model-lifecycle',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': prefix},
                    'Transitions': [
                        {'Days': 7, 'StorageClass': 'INTELLIGENT_TIERING'},
                        {'Days': 90, 'StorageClass': 'GLACIER'}
                    ]
                }]
            },
            'savings_estimate': '45-55% after 90 days'
        }

    # Logs: transition quickly, expire after 90 days
    elif 'log' in bucket_lower or 'log' in prefix_lower:
        return {
            'name': 'Logs Lifecycle',
            'policy': {
                'Rules': [{
                    'Id': 'logs-lifecycle',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': prefix},
                    'Transitions': [
                        {'Days': 7, 'StorageClass': 'GLACIER'}
                    ],
                    'Expiration': {'Days': 90}
                }]
            },
            'savings_estimate': '80% after 7 days'
        }

    # Backups: immediate archive
    elif 'backup' in bucket_lower or 'backup' in prefix_lower:
        return {
            'name': 'Backup Lifecycle',
            'policy': {
                'Rules': [{
                    'Id': 'backup-lifecycle',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': prefix},
                    'Transitions': [
                        {'Days': 0, 'StorageClass': 'GLACIER_IR'},
                        {'Days': 90, 'StorageClass': 'DEEP_ARCHIVE'}
                    ]
                }]
            },
            'savings_estimate': '80-90%'
        }

    # Default: conservative tiering
    else:
        return {
            'name': 'Default Lifecycle',
            'policy': {
                'Rules': [{
                    'Id': 'default-lifecycle',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': prefix},
                    'Transitions': [
                        {'Days': 30, 'StorageClass': 'INTELLIGENT_TIERING'}
                    ]
                }]
            },
            'savings_estimate': '30-40% after 30 days'
        }


def apply_lifecycle_policy(bucket_name: str, policy: Dict, dry_run: bool = True):
    """
    Apply lifecycle policy to bucket.

    Args:
        bucket_name: S3 bucket name
        policy: Lifecycle policy dict
        dry_run: If True, only print policy without applying
    """
    if dry_run:
        print(f"[DRY RUN] Would apply lifecycle policy to {bucket_name}:")
        print(json.dumps(policy, indent=2))
        return

    s3 = boto3.client('s3')

    try:
        s3.put_bucket_lifecycle_configuration(
            Bucket=bucket_name,
            LifecycleConfiguration=policy
        )
        print(f"✓ Applied lifecycle policy to {bucket_name}")
    except Exception as e:
        print(f"✗ Failed to apply policy to {bucket_name}: {e}")


def find_large_objects(bucket_name: str, min_size_mb: int = 100, limit: int = 100) -> List[Dict]:
    """
    Find large objects that could benefit from compression.

    Args:
        bucket_name: S3 bucket name
        min_size_mb: Minimum object size in MB
        limit: Maximum number of objects to return

    Returns:
        List of dicts with object info
    """
    s3 = boto3.client('s3')

    print(f"Scanning {bucket_name} for large objects (>{min_size_mb} MB)...")

    large_objects = []
    paginator = s3.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=bucket_name):
        if 'Contents' not in page:
            continue

        for obj in page['Contents']:
            size_mb = obj['Size'] / (1024**2)

            if size_mb >= min_size_mb:
                # Check if compressible (CSV, JSON, text, logs)
                key = obj['Key']
                ext = key.split('.')[-1].lower()
                compressible = ext in ['csv', 'json', 'txt', 'log', 'xml']

                large_objects.append({
                    'key': key,
                    'size_mb': size_mb,
                    'size_bytes': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'storage_class': obj.get('StorageClass', 'STANDARD'),
                    'compressible': compressible
                })

                if len(large_objects) >= limit:
                    break

        if len(large_objects) >= limit:
            break

    return large_objects


def calculate_compression_savings(large_objects: List[Dict]) -> Dict:
    """Calculate potential savings from compression."""
    total_size_gb = sum(obj['size_bytes'] for obj in large_objects) / (1024**3)

    # Estimate 60-70% compression for text formats
    compressible_objects = [obj for obj in large_objects if obj['compressible']]
    compressible_size_gb = sum(obj['size_bytes'] for obj in compressible_objects) / (1024**3)

    estimated_compressed_size_gb = compressible_size_gb * 0.35  # 65% reduction
    size_reduction_gb = compressible_size_gb - estimated_compressed_size_gb

    # Calculate cost savings (STANDARD pricing)
    monthly_savings = size_reduction_gb * STORAGE_PRICING['STANDARD']

    return {
        'total_objects': len(large_objects),
        'compressible_objects': len(compressible_objects),
        'total_size_gb': total_size_gb,
        'compressible_size_gb': compressible_size_gb,
        'estimated_compressed_size_gb': estimated_compressed_size_gb,
        'size_reduction_gb': size_reduction_gb,
        'reduction_percentage': (size_reduction_gb / compressible_size_gb * 100) if compressible_size_gb > 0 else 0,
        'monthly_savings': monthly_savings
    }


def find_duplicate_objects(bucket_name: str, prefix: str = '', limit: int = 10000) -> Dict:
    """
    Find potential duplicate objects by comparing file sizes and names.

    Args:
        bucket_name: S3 bucket name
        prefix: Object prefix to scan
        limit: Maximum objects to scan

    Returns:
        Dict with duplicate analysis
    """
    s3 = boto3.client('s3')

    print(f"Scanning {bucket_name} for duplicates (first {limit} objects)...")

    # Group objects by size
    objects_by_size = defaultdict(list)
    total_objects = 0

    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    for page in page_iterator:
        if 'Contents' not in page:
            continue

        for obj in page['Contents']:
            size = obj['Size']
            objects_by_size[size].append({
                'key': obj['Key'],
                'size': size,
                'last_modified': obj['LastModified']
            })

            total_objects += 1
            if total_objects >= limit:
                break

        if total_objects >= limit:
            break

    # Find potential duplicates (same size)
    potential_duplicates = []
    duplicate_size = 0

    for size, objects in objects_by_size.items():
        if len(objects) > 1:
            # Multiple objects with same size
            potential_duplicates.append({
                'size': size,
                'count': len(objects),
                'objects': objects[:10],  # Limit to 10 for display
                'total_duplicate_size': size * (len(objects) - 1)
            })
            duplicate_size += size * (len(objects) - 1)

    # Sort by duplicate size
    potential_duplicates.sort(key=lambda x: x['total_duplicate_size'], reverse=True)

    # Calculate savings if duplicates removed
    duplicate_size_gb = duplicate_size / (1024**3)
    monthly_savings = duplicate_size_gb * STORAGE_PRICING['STANDARD']

    return {
        'total_objects_scanned': total_objects,
        'potential_duplicate_groups': len(potential_duplicates),
        'duplicate_size_gb': duplicate_size_gb,
        'monthly_savings': monthly_savings,
        'top_duplicates': potential_duplicates[:20]
    }


def print_bucket_analysis(results: List[Dict]):
    """Print analysis summary for all buckets."""
    print()
    print("=" * 80)
    print("  S3 Storage Analysis")
    print("=" * 80)
    print()

    # Sort by cost
    results_sorted = sorted(results, key=lambda x: x['total_cost'], reverse=True)

    total_cost = sum(r['total_cost'] for r in results)
    total_size_gb = sum(
        sum(c['size_gb'] for c in r['costs'].values())
        for r in results
    )

    print(f"Total buckets: {len(results)}")
    print(f"Total storage: {total_size_gb:,.2f} GB")
    print(f"Total monthly cost: ${total_cost:,.2f}")
    print()

    # Top 10 most expensive buckets
    print("Top 10 Most Expensive Buckets:")
    print("-" * 80)

    for i, result in enumerate(results_sorted[:10], 1):
        bucket_name = result['bucket_name']
        cost = result['total_cost']
        size_gb = sum(c['size_gb'] for c in result['costs'].values())

        print(f"{i}. {bucket_name}")
        print(f"   Size: {size_gb:,.2f} GB")
        print(f"   Cost: ${cost:,.2f}/month")

        # Show storage class breakdown
        for storage_class, cost_data in result['costs'].items():
            if cost_data['size_gb'] > 0:
                print(f"     - {storage_class}: {cost_data['size_gb']:,.2f} GB (${cost_data['monthly_cost']:,.2f}/month)")

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and optimize S3 storage costs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all buckets
  %(prog)s --analyze

  # Analyze specific bucket
  %(prog)s --analyze --bucket my-bucket

  # Apply lifecycle policy (dry run)
  %(prog)s --apply-lifecycle --bucket my-bucket --dry-run

  # Apply lifecycle policy (execute)
  %(prog)s --apply-lifecycle --bucket my-bucket

  # Find large compressible objects
  %(prog)s --find-large --bucket my-bucket --min-size 100

  # Find duplicates
  %(prog)s --find-duplicates --bucket my-bucket
        """
    )

    parser.add_argument('--analyze', action='store_true', help='Analyze bucket costs')
    parser.add_argument('--apply-lifecycle', action='store_true', help='Apply lifecycle policy')
    parser.add_argument('--find-large', action='store_true', help='Find large objects for compression')
    parser.add_argument('--find-duplicates', action='store_true', help='Find potential duplicate objects')

    parser.add_argument('--bucket', help='Specific bucket name (default: all buckets)')
    parser.add_argument('--prefix', default='', help='Object prefix for filtering')
    parser.add_argument('--min-size', type=int, default=100, help='Minimum object size in MB (for --find-large)')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without applying')

    args = parser.parse_args()

    try:
        s3 = boto3.client('s3')

        # Get buckets
        if args.bucket:
            buckets = [args.bucket]
        else:
            response = s3.list_buckets()
            buckets = [b['Name'] for b in response['Buckets']]

        # Analyze
        if args.analyze:
            results = []
            for bucket_name in buckets:
                result = analyze_bucket(bucket_name)
                if result:
                    results.append(result)

            print_bucket_analysis(results)

        # Apply lifecycle
        if args.apply_lifecycle:
            if not args.bucket:
                print("Error: --bucket required for --apply-lifecycle")
                return 1

            recommendation = recommend_lifecycle_policy(args.bucket, args.prefix)
            print()
            print(f"Lifecycle Policy Recommendation: {recommendation['name']}")
            print(f"Estimated savings: {recommendation['savings_estimate']}")
            print()

            apply_lifecycle_policy(args.bucket, recommendation['policy'], dry_run=args.dry_run)

        # Find large objects
        if args.find_large:
            if not args.bucket:
                print("Error: --bucket required for --find-large")
                return 1

            large_objects = find_large_objects(args.bucket, args.min_size)

            if not large_objects:
                print(f"No objects larger than {args.min_size} MB found")
                return 0

            print(f"Found {len(large_objects)} large objects")
            print()

            # Calculate compression savings
            savings = calculate_compression_savings(large_objects)

            print(f"Compression Analysis:")
            print(f"  Total objects: {savings['total_objects']}")
            print(f"  Compressible objects: {savings['compressible_objects']}")
            print(f"  Compressible size: {savings['compressible_size_gb']:,.2f} GB")
            print(f"  Estimated compressed size: {savings['estimated_compressed_size_gb']:,.2f} GB")
            print(f"  Size reduction: {savings['size_reduction_gb']:,.2f} GB ({savings['reduction_percentage']:.1f}%)")
            print(f"  Monthly savings: ${savings['monthly_savings']:,.2f}")
            print()

            # Show top 10 largest compressible files
            compressible = [obj for obj in large_objects if obj['compressible']]
            compressible_sorted = sorted(compressible, key=lambda x: x['size_mb'], reverse=True)

            print("Top 10 Largest Compressible Objects:")
            for i, obj in enumerate(compressible_sorted[:10], 1):
                print(f"{i}. {obj['key']}")
                print(f"   Size: {obj['size_mb']:,.2f} MB")
                print()

        # Find duplicates
        if args.find_duplicates:
            if not args.bucket:
                print("Error: --bucket required for --find-duplicates")
                return 1

            duplicates = find_duplicate_objects(args.bucket, args.prefix)

            print()
            print(f"Duplicate Analysis:")
            print(f"  Objects scanned: {duplicates['total_objects_scanned']:,}")
            print(f"  Potential duplicate groups: {duplicates['potential_duplicate_groups']}")
            print(f"  Duplicate size: {duplicates['duplicate_size_gb']:,.2f} GB")
            print(f"  Monthly savings if removed: ${duplicates['monthly_savings']:,.2f}")
            print()

            if duplicates['top_duplicates']:
                print("Top 10 Duplicate Groups:")
                for i, dup_group in enumerate(duplicates['top_duplicates'][:10], 1):
                    dup_size_mb = dup_group['size'] / (1024**2)
                    dup_total_mb = dup_group['total_duplicate_size'] / (1024**2)
                    print(f"{i}. {dup_group['count']} objects, {dup_size_mb:.2f} MB each ({dup_total_mb:.2f} MB total)")
                    for obj in dup_group['objects'][:3]:
                        print(f"   - {obj['key']}")
                    if dup_group['count'] > 3:
                        print(f"   ... and {dup_group['count'] - 3} more")
                    print()

        return 0

    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Note: This script requires AWS credentials with S3 and CloudWatch read access")
        print("Run 'aws configure' to set up credentials")
        return 1


if __name__ == '__main__':
    sys.exit(main())
