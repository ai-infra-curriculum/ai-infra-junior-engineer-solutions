# Step-by-Step Implementation Guide: Cloud Cost Optimization

## Overview

Optimize cloud spending for ML workloads! Learn cost analysis, right-sizing, spot instances, reserved capacity, storage optimization, and FinOps best practices.

**Time**: 2-3 hours | **Difficulty**: Intermediate

---

## Learning Objectives

✅ Analyze cloud costs
✅ Right-size compute resources
✅ Use spot/preemptible instances
✅ Optimize storage costs
✅ Implement cost monitoring
✅ Set up budget alerts
✅ Apply FinOps principles

---

## Cost Analysis

### AWS Cost Explorer

```bash
# Get monthly costs
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity MONTHLY \
  --metrics UnblendedCost \
  --group-by Type=DIMENSION,Key=SERVICE

# Get costs by tag
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity DAILY \
  --metrics UnblendedCost \
  --group-by Type=TAG,Key=Project
```

### GCP Cost Reports

```bash
# Export billing to BigQuery
gcloud beta billing accounts get-iam-policy BILLING_ACCOUNT_ID

# Query costs
bq query --use_legacy_sql=false '
SELECT
  service.description,
  SUM(cost) as total_cost
FROM `project.dataset.gcp_billing_export`
WHERE _PARTITIONTIME >= "2024-01-01"
GROUP BY service.description
ORDER BY total_cost DESC
'
```

---

## Right-Sizing Compute

### AWS Compute Optimizer

```python
import boto3

optimizer = boto3.client('compute-optimizer')

# Get recommendations
recommendations = optimizer.get_ec2_instance_recommendations()

for rec in recommendations['instanceRecommendations']:
    current = rec['currentInstanceType']
    recommended = rec['recommendationOptions'][0]['instanceType']
    savings = rec['recommendationOptions'][0]['estimatedMonthlySavings']

    print(f"Instance: {rec['instanceArn']}")
    print(f"Current: {current}")
    print(f"Recommended: {recommended}")
    print(f"Est. Savings: ${savings['value']}/month")
```

### Automated Right-Sizing

```python
# Auto-resize based on usage
import boto3
from datetime import datetime, timedelta

cloudwatch = boto3.client('cloudwatch')
ec2 = boto3.client('ec2')

def get_cpu_utilization(instance_id, days=7):
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    response = cloudwatch.get_metric_statistics(
        Namespace='AWS/EC2',
        MetricName='CPUUtilization',
        Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
        StartTime=start_time,
        EndTime=end_time,
        Period=3600,
        Statistics=['Average']
    )

    avg_cpu = sum(p['Average'] for p in response['Datapoints']) / len(response['Datapoints'])
    return avg_cpu

def resize_if_needed(instance_id):
    avg_cpu = get_cpu_utilization(instance_id)

    if avg_cpu < 20:
        # Downsize
        print(f"Downsizing {instance_id} (CPU: {avg_cpu:.1f}%)")
        # Implement resize logic
    elif avg_cpu > 80:
        # Upsize
        print(f"Upsizing {instance_id} (CPU: {avg_cpu:.1f}%)")
```

---

## Spot Instances

### AWS Spot Instances

```bash
# Launch spot instance
aws ec2 run-instances \
  --instance-type p3.2xlarge \
  --instance-market-options '{
    "MarketType": "spot",
    "SpotOptions": {
      "MaxPrice": "1.00",
      "SpotInstanceType": "one-time",
      "InstanceInterruptionBehavior": "terminate"
    }
  }' \
  --image-id ami-0123456789abcdef0
```

### Spot Instance Handler

```python
# Handle spot interruptions
import requests
import time

METADATA_URL = "http://169.254.169.254/latest/meta-data/spot/instance-action"

while True:
    try:
        response = requests.get(METADATA_URL, timeout=1)
        if response.status_code == 200:
            # Interruption notice received
            print("Spot instance terminating in 2 minutes!")
            # Save checkpoint
            save_checkpoint()
            # Upload to S3
            upload_results()
            break
    except:
        pass
    time.sleep(5)
```

---

## Reserved Capacity

### Purchase Reserved Instances

```python
import boto3

ec2 = boto3.client('ec2')

# Purchase 1-year reserved instance
response = ec2.purchase_reserved_instances_offering(
    InstanceCount=1,
    ReservedInstancesOfferingId='offering-id',
    LimitPrice={
        'Amount': 1000.0,
        'CurrencyCode': 'USD'
    }
)
```

### Savings Plans

```bash
# AWS Savings Plans
aws savingsplans create-savings-plan \
  --savings-plan-type Compute \
  --commitment 100 \
  --upfront-payment-amount 0 \
  --savings-plan-offering-id sp-offering-id
```

---

## Storage Optimization

### S3 Lifecycle Policies

```json
{
  "Rules": [
    {
      "Id": "ArchiveOldData",
      "Status": "Enabled",
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 90,
          "StorageClass": "GLACIER"
        },
        {
          "Days": 365,
          "StorageClass": "DEEP_ARCHIVE"
        }
      ],
      "Expiration": {
        "Days": 730
      }
    },
    {
      "Id": "DeleteIncompleteUploads",
      "Status": "Enabled",
      "AbortIncompleteMultipartUpload": {
        "DaysAfterInitiation": 7
      }
    }
  ]
}
```

### Intelligent Tiering

```bash
# Enable S3 Intelligent-Tiering
aws s3api put-bucket-intelligent-tiering-configuration \
  --bucket my-ml-models \
  --id ModelArchive \
  --intelligent-tiering-configuration '{
    "Id": "ModelArchive",
    "Status": "Enabled",
    "Tierings": [
      {
        "Days": 90,
        "AccessTier": "ARCHIVE_ACCESS"
      },
      {
        "Days": 180,
        "AccessTier": "DEEP_ARCHIVE_ACCESS"
      }
    ]
  }'
```

---

## Cost Monitoring

### Budget Alerts

```python
import boto3

budgets = boto3.client('budgets')

# Create budget
budgets.create_budget(
    AccountId='123456789012',
    Budget={
        'BudgetName': 'ML-Training-Budget',
        'BudgetLimit': {
            'Amount': '1000',
            'Unit': 'USD'
        },
        'TimeUnit': 'MONTHLY',
        'BudgetType': 'COST'
    },
    NotificationsWithSubscribers=[
        {
            'Notification': {
                'NotificationType': 'ACTUAL',
                'ComparisonOperator': 'GREATER_THAN',
                'Threshold': 80,
                'ThresholdType': 'PERCENTAGE'
            },
            'Subscribers': [
                {
                    'SubscriptionType': 'EMAIL',
                    'Address': 'alerts@example.com'
                }
            ]
        }
    ]
)
```

### Cost Anomaly Detection

```python
# AWS Cost Anomaly Detection
import boto3

ce = boto3.client('ce')

# Create monitor
ce.create_anomaly_monitor(
    AnomalyMonitor={
        'MonitorName': 'ML-Training-Anomalies',
        'MonitorType': 'DIMENSIONAL',
        'MonitorDimension': 'SERVICE'
    }
)

# Create subscription
ce.create_anomaly_subscription(
    AnomalySubscription={
        'SubscriptionName': 'ML-Alerts',
        'MonitorArnList': ['monitor-arn'],
        'Subscribers': [
            {
                'Type': 'EMAIL',
                'Address': 'alerts@example.com'
            }
        ],
        'Threshold': 100,
        'Frequency': 'DAILY'
    }
)
```

---

## Auto-Scaling for Cost

### Kubernetes HPA with Cost Awareness

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

### Schedule-Based Scaling

```bash
# Scale down at night (cron)
0 20 * * * kubectl scale deployment ml-api --replicas=1
0 8 * * * kubectl scale deployment ml-api --replicas=5
```

---

## Best Practices

✅ Tag all resources for cost tracking
✅ Use spot instances for training
✅ Implement auto-shutdown for dev environments
✅ Right-size based on actual usage
✅ Use reserved capacity for predictable workloads
✅ Implement storage lifecycle policies
✅ Monitor costs daily
✅ Set budget alerts
✅ Review and optimize monthly
✅ Use cost allocation tags

---

**Cloud Cost Optimization mastered!** 💰
