# Exercise 05: Cost Optimization & Management

Learn advanced techniques for optimizing and managing costs across AWS, GCP, and Azure for ML workloads.

## Learning Objectives

- Analyze and understand cloud cost structures
- Implement cost optimization strategies
- Use spot/preemptible instances effectively
- Right-size compute and storage resources
- Set up cost monitoring and alerting
- Implement chargeback and cost allocation
- Forecast and budget cloud spending
- Optimize data transfer costs

## Cost Structure Overview

### Cloud Pricing Models

```
┌──────────────────────────────────────────────────────────────────┐
│                    Cloud Cost Components                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Compute                                                          │
│  ├─ On-demand instances      (Standard pricing)                  │
│  ├─ Reserved instances       (30-70% discount, 1-3 year commit)  │
│  ├─ Spot/Preemptible        (70-90% discount, interruptible)    │
│  └─ Savings plans           (Flexible commitment discounts)      │
│                                                                   │
│  Storage                                                          │
│  ├─ Hot/Standard storage    (Frequent access)                    │
│  ├─ Cool/Nearline storage   (Infrequent access, 30+ days)       │
│  ├─ Cold/Coldline storage   (Rare access, 90+ days)             │
│  └─ Archive                 (Long-term archival, 180+ days)      │
│                                                                   │
│  Data Transfer                                                    │
│  ├─ Ingress (incoming)      (Usually free)                       │
│  ├─ Egress (outgoing)       (Varies by destination)             │
│  ├─ Inter-region            (Higher cost)                        │
│  └─ Cross-cloud             (Highest cost)                       │
│                                                                   │
│  ML Services                                                      │
│  ├─ Training                (Per hour + storage)                 │
│  ├─ Inference               (Per request or per hour)            │
│  └─ Data processing         (Per GB processed)                   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Part 1: Compute Cost Optimization

### 1. Use Spot/Preemptible Instances

**AWS Spot Instances**:
```python
# spot_trainer.py
import boto3
import time

class SpotTrainer:
    def __init__(self):
        self.ec2 = boto3.client('ec2')

    def train_with_checkpointing(self, max_retries=3):
        """Train with automatic checkpoint/resume on spot interruption"""
        for attempt in range(max_retries):
            try:
                # Request spot instance
                response = self.ec2.request_spot_instances(
                    SpotPrice='1.00',
                    InstanceCount=1,
                    Type='one-time',
                    LaunchSpecification={
                        'ImageId': 'ami-xxx',
                        'InstanceType': 'p3.2xlarge',
                        'KeyName': 'my-key',
                        'UserData': self.get_training_script(attempt)
                    }
                )

                instance_id = self.wait_for_instance(response['SpotInstanceRequestId'])

                # Monitor for spot termination warning
                self.monitor_spot_termination(instance_id)

            except SpotInterruptedException:
                print(f"Spot interrupted, retrying... (attempt {attempt + 1}/{max_retries})")
                continue

    def get_training_script(self, resume_epoch=0):
        """Training script that supports checkpointing"""
        return f"""#!/bin/bash
        cd /home/ubuntu/training
        python train.py --resume-from-epoch {resume_epoch} --checkpoint-every 10
        aws s3 cp checkpoints/ s3://my-bucket/checkpoints/ --recursive
        """

    def monitor_spot_termination(self, instance_id):
        """Monitor for 2-minute termination warning"""
        while True:
            response = requests.get(
                'http://169.254.169.254/latest/meta-data/spot/termination-time',
                timeout=1
            )
            if response.status_code == 200:
                print("Spot termination warning! Saving checkpoint...")
                self.save_checkpoint()
                raise SpotInterruptedException()
            time.sleep(5)
```

**GCP Preemptible VMs**:
```bash
# Start preemptible training
gcloud compute instances create training-vm \
  --preemptible \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --metadata=startup-script='
    #!/bin/bash
    # Check if preempted
    if [ -f /tmp/preempted ]; then
      RESUME_FLAG="--resume"
    else
      RESUME_FLAG=""
    fi

    # Train with checkpointing
    python train.py $RESUME_FLAG --checkpoint-interval=600

    # On shutdown, save state
    trap "touch /tmp/preempted; gsutil cp -r checkpoints/ gs://my-bucket/" EXIT
  '
```

### 2. Right-Sizing Instances

```python
# right_sizer.py
import boto3
from datetime import datetime, timedelta

class InstanceRightSizer:
    def __init__(self):
        self.ec2 = boto3.client('ec2')
        self.cloudwatch = boto3.client('cloudwatch')

    def analyze_utilization(self, instance_id: str, days: int = 7):
        """Analyze instance utilization over time"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        # Get CPU utilization
        cpu_stats = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/EC2',
            MetricName='CPUUtilization',
            Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,  # 1 hour
            Statistics=['Average', 'Maximum']
        )

        # Get memory utilization (if CloudWatch agent installed)
        memory_stats = self.cloudwatch.get_metric_statistics(
            Namespace='CWAgent',
            MetricName='mem_used_percent',
            Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,
            Statistics=['Average', 'Maximum']
        )

        # Analyze
        avg_cpu = sum(d['Average'] for d in cpu_stats['Datapoints']) / len(cpu_stats['Datapoints'])
        max_cpu = max(d['Maximum'] for d in cpu_stats['Datapoints'])

        # Get current instance type
        instance = self.ec2.describe_instances(InstanceIds=[instance_id])
        current_type = instance['Reservations'][0]['Instances'][0]['InstanceType']

        # Recommend right-size
        recommendation = self.recommend_instance_type(avg_cpu, max_cpu, current_type)

        return {
            'instance_id': instance_id,
            'current_type': current_type,
            'avg_cpu': avg_cpu,
            'max_cpu': max_cpu,
            'recommendation': recommendation
        }

    def recommend_instance_type(self, avg_cpu: float, max_cpu: float, current_type: str):
        """Recommend instance type based on utilization"""
        if avg_cpu < 20 and max_cpu < 40:
            return {
                'action': 'downsize',
                'reason': 'Low utilization (avg: {:.1f}%, max: {:.1f}%)'.format(avg_cpu, max_cpu),
                'potential_savings': '50%'
            }
        elif avg_cpu > 70 or max_cpu > 90:
            return {
                'action': 'upsize',
                'reason': 'High utilization (avg: {:.1f}%, max: {:.1f}%)'.format(avg_cpu, max_cpu),
                'risk': 'Performance degradation'
            }
        else:
            return {
                'action': 'keep',
                'reason': 'Optimal utilization'
            }
```

### 3. Auto-Scaling Policies

```yaml
# kubernetes-hpa-vpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-inference
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  # Custom metrics
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
---
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: ml-inference-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-inference
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: api
      minAllowed:
        cpu: 1
        memory: 2Gi
      maxAllowed:
        cpu: 8
        memory: 16Gi
```

## Part 2: Storage Cost Optimization

### 1. Storage Lifecycle Policies

**AWS S3**:
```python
# s3_lifecycle.py
import boto3

def setup_intelligent_tiering(bucket_name):
    """Set up S3 Intelligent-Tiering"""
    s3 = boto3.client('s3')

    lifecycle_config = {
        'Rules': [
            {
                'Id': 'intelligent-tiering-all',
                'Status': 'Enabled',
                'Filter': {'Prefix': ''},
                'Transitions': [{
                    'Days': 0,
                    'StorageClass': 'INTELLIGENT_TIERING'
                }]
            },
            {
                'Id': 'archive-old-models',
                'Status': 'Enabled',
                'Filter': {'Prefix': 'models/'},
                'Transitions': [
                    {'Days': 30, 'StorageClass': 'STANDARD_IA'},
                    {'Days': 90, 'StorageClass': 'GLACIER'},
                    {'Days': 180, 'StorageClass': 'DEEP_ARCHIVE'}
                ],
                'NoncurrentVersionExpiration': {'NoncurrentDays': 90}
            },
            {
                'Id': 'delete-temp-logs',
                'Status': 'Enabled',
                'Filter': {'Prefix': 'logs/'},
                'Expiration': {'Days': 7}
            },
            {
                'Id': 'incomplete-multipart-cleanup',
                'Status': 'Enabled',
                'Filter': {'Prefix': ''},
                'AbortIncompleteMultipartUpload': {'DaysAfterInitiation': 7}
            }
        ]
    }

    s3.put_bucket_lifecycle_configuration(
        Bucket=bucket_name,
        LifecycleConfiguration=lifecycle_config
    )
```

### 2. Data Compression

```python
# data_compressor.py
import gzip
import lz4.frame
import zstd
from pathlib import Path

class DataCompressor:
    """Compress data before uploading to cloud storage"""

    @staticmethod
    def compress_dataset(input_path: str, output_path: str, method='zstd'):
        """Compress dataset with specified method"""
        input_size = Path(input_path).stat().st_size

        if method == 'gzip':
            with open(input_path, 'rb') as f_in:
                with gzip.open(output_path, 'wb', compresslevel=9) as f_out:
                    f_out.writelines(f_in)

        elif method == 'lz4':
            with open(input_path, 'rb') as f_in:
                with lz4.frame.open(output_path, 'wb') as f_out:
                    f_out.write(f_in.read())

        elif method == 'zstd':
            with open(input_path, 'rb') as f_in:
                compressed = zstd.compress(f_in.read(), level=19)
                with open(output_path, 'wb') as f_out:
                    f_out.write(compressed)

        output_size = Path(output_path).stat().st_size
        ratio = (1 - output_size / input_size) * 100

        print(f"Compressed {input_path}: {input_size:,} → {output_size:,} bytes")
        print(f"Compression ratio: {ratio:.1f}%")
        print(f"Storage savings: ${DataCompressor.calculate_savings(input_size, output_size):.2f}/month")

    @staticmethod
    def calculate_savings(original_size: int, compressed_size: int, storage_cost_per_gb=0.023):
        """Calculate monthly storage cost savings"""
        original_gb = original_size / (1024 ** 3)
        compressed_gb = compressed_size / (1024 ** 3)
        savings = (original_gb - compressed_gb) * storage_cost_per_gb
        return savings
```

## Part 3: Data Transfer Cost Optimization

### 1. Minimize Cross-Region Transfers

```python
# regional_routing.py
class RegionalRouter:
    """Route requests to nearest region to minimize data transfer"""

    REGIONS = {
        'us-east-1': {'lat': 38.13, 'lon': -78.45},
        'us-west-2': {'lat': 45.87, 'lon': -119.69},
        'eu-west-1': {'lat': 53.35, 'lon': -6.26},
        'ap-southeast-1': {'lat': 1.29, 'lon': 103.85}
    }

    @staticmethod
    def find_nearest_region(client_lat: float, client_lon: float) -> str:
        """Find nearest AWS region based on client location"""
        import math

        def distance(lat1, lon1, lat2, lon2):
            # Haversine formula
            R = 6371  # Earth radius in km
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = (math.sin(dlat / 2) ** 2 +
                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                 math.sin(dlon / 2) ** 2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            return R * c

        nearest = min(
            RegionalRouter.REGIONS.items(),
            key=lambda r: distance(client_lat, client_lon, r[1]['lat'], r[1]['lon'])
        )

        return nearest[0]

# Usage in API
@app.route('/predict')
def predict():
    client_ip = request.remote_addr
    client_location = geoip_lookup(client_ip)

    nearest_region = RegionalRouter.find_nearest_region(
        client_location['lat'],
        client_location['lon']
    )

    # Route to nearest endpoint
    return redirect(f"https://{nearest_region}.api.example.com/predict")
```

### 2. Use CDN for Inference

```python
# cdn_deployment.py
import boto3

def deploy_to_cloudfront(model_endpoint: str):
    """Deploy model behind CloudFront CDN"""
    cf = boto3.client('cloudfront')

    distribution_config = {
        'CallerReference': str(time.time()),
        'Comment': 'ML Inference CDN',
        'Enabled': True,
        'Origins': {
            'Quantity': 1,
            'Items': [{
                'Id': 'ml-inference-origin',
                'DomainName': model_endpoint,
                'CustomOriginConfig': {
                    'HTTPPort': 80,
                    'HTTPSPort': 443,
                    'OriginProtocolPolicy': 'https-only'
                }
            }]
        },
        'DefaultCacheBehavior': {
            'TargetOriginId': 'ml-inference-origin',
            'ViewerProtocolPolicy': 'redirect-to-https',
            'AllowedMethods': {
                'Quantity': 7,
                'Items': ['GET', 'HEAD', 'OPTIONS', 'PUT', 'POST', 'PATCH', 'DELETE'],
                'CachedMethods': {'Quantity': 2, 'Items': ['GET', 'HEAD']}
            },
            'Compress': True,
            'MinTTL': 0,
            'DefaultTTL': 300,  # Cache predictions for 5 minutes
            'MaxTTL': 3600
        },
        'PriceClass': 'PriceClass_100'  # Use only US, Canada, Europe
    }

    response = cf.create_distribution(DistributionConfig=distribution_config)
    print(f"CDN URL: {response['Distribution']['DomainName']}")
```

## Part 4: Cost Monitoring & Alerting

### Unified Cost Dashboard

```python
# cost_dashboard.py
import dash
from dash import dcc, html
import plotly.graph_objs as go
from aws_cost_monitor import AWSCostMonitor
from gcp_cost_monitor import GCPCostMonitor
from azure_cost_monitor import AzureCostMonitor

class CostDashboard:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.aws = AWSCostMonitor()
        self.gcp = GCPCostMonitor()
        self.azure = AzureCostMonitor()

    def create_layout(self):
        self.app.layout = html.Div([
            html.H1('Multi-Cloud Cost Dashboard'),

            # Total costs by cloud
            dcc.Graph(id='cost-by-cloud'),

            # Daily cost trend
            dcc.Graph(id='daily-cost-trend'),

            # Cost by service
            dcc.Graph(id='cost-by-service'),

            # Savings recommendations
            html.Div(id='recommendations'),

            # Auto-refresh
            dcc.Interval(id='interval', interval=300000)  # 5 minutes
        ])

    def update_graphs(self):
        # Get costs
        aws_cost = self.aws.get_monthly_cost()
        gcp_cost = self.gcp.get_monthly_cost()
        azure_cost = self.azure.get_monthly_cost()

        # Cost by cloud pie chart
        cost_by_cloud = go.Figure(data=[
            go.Pie(labels=['AWS', 'GCP', 'Azure'],
                   values=[aws_cost, gcp_cost, azure_cost])
        ])

        # Daily trend
        days = 30
        daily_costs = []
        for cloud, monitor in [('AWS', self.aws), ('GCP', self.gcp), ('Azure', self.azure)]:
            costs = monitor.get_daily_costs(days)
            daily_costs.append(
                go.Scatter(x=[c['date'] for c in costs],
                          y=[c['cost'] for c in costs],
                          name=cloud)
            )

        daily_trend = go.Figure(data=daily_costs)
        daily_trend.update_layout(title='Daily Cost Trend (Last 30 Days)')

        return cost_by_cloud, daily_trend
```

### Anomaly Detection

```python
# cost_anomaly_detector.py
import numpy as np
from sklearn.ensemble import IsolationForest

class CostAnomalyDetector:
    """Detect unusual cost spikes"""

    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)

    def train(self, historical_costs: list):
        """Train on historical cost data"""
        X = np.array(historical_costs).reshape(-1, 1)
        self.model.fit(X)

    def detect_anomalies(self, current_cost: float, historical_costs: list):
        """Detect if current cost is anomalous"""
        self.train(historical_costs)

        prediction = self.model.predict([[current_cost]])

        if prediction[0] == -1:
            # Anomaly detected
            avg_cost = np.mean(historical_costs)
            deviation = ((current_cost - avg_cost) / avg_cost) * 100

            return {
                'anomaly': True,
                'current_cost': current_cost,
                'average_cost': avg_cost,
                'deviation_pct': deviation,
                'alert': f"⚠️ Cost spike detected! {deviation:+.1f}% above average"
            }
        else:
            return {'anomaly': False}
```

## Part 5: Reserved Capacity & Savings Plans

### Capacity Planning

```python
# capacity_planner.py
class CapacityPlanner:
    """Analyze workload and recommend reserved capacity"""

    def analyze_usage_pattern(self, instance_type: str, days: int = 90):
        """Analyze instance usage over time"""
        usage_data = self.get_usage_data(instance_type, days)

        # Calculate percentiles
        p50 = np.percentile(usage_data, 50)
        p75 = np.percentile(usage_data, 75)
        p90 = np.percentile(usage_data, 90)

        # Recommend reserved instances
        recommendation = {
            'instance_type': instance_type,
            'current_on_demand': len([x for x in usage_data if x > 0]),
            'recommended_reserved': int(p75),  # Reserve at 75th percentile
            'use_spot_for_burst': int(np.max(usage_data) - p75),
            'estimated_savings': self.calculate_savings(int(p75))
        }

        return recommendation

    def calculate_savings(self, reserved_count: int):
        """Calculate savings from reserved instances"""
        on_demand_cost = reserved_count * 3.06 * 24 * 30  # p3.2xlarge
        reserved_cost = reserved_count * 1.40 * 24 * 30   # 1-year partial upfront

        monthly_savings = on_demand_cost - reserved_cost
        annual_savings = monthly_savings * 12

        return {
            'monthly_savings': monthly_savings,
            'annual_savings': annual_savings,
            'savings_percentage': (monthly_savings / on_demand_cost) * 100
        }
```

## Part 6: Chargeback & Cost Allocation

### Tag-Based Cost Allocation

```python
# cost_allocator.py
class CostAllocator:
    """Allocate costs to teams/projects based on tags"""

    def __init__(self, aws_client, gcp_client, azure_client):
        self.aws = aws_client
        self.gcp = gcp_client
        self.azure = azure_client

    def allocate_costs_by_team(self, start_date: str, end_date: str):
        """Break down costs by team"""
        # AWS
        aws_costs = self.aws.get_costs_by_tag(start_date, end_date, 'Team')

        # GCP
        gcp_costs = self.gcp.get_costs_by_label(start_date, end_date, 'team')

        # Azure
        azure_costs = self.azure.get_costs_by_tag(start_date, end_date, 'Team')

        # Combine
        all_teams = set(list(aws_costs.keys()) + list(gcp_costs.keys()) + list(azure_costs.keys()))

        team_costs = {}
        for team in all_teams:
            team_costs[team] = {
                'aws': aws_costs.get(team, 0),
                'gcp': gcp_costs.get(team, 0),
                'azure': azure_costs.get(team, 0),
                'total': (aws_costs.get(team, 0) +
                         gcp_costs.get(team, 0) +
                         azure_costs.get(team, 0))
            }

        return team_costs

    def generate_chargeback_report(self, team_costs: dict):
        """Generate chargeback report"""
        print("\n" + "="*70)
        print(f"{'Team':<20} {'AWS':<15} {'GCP':<15} {'Azure':<15} {'Total':<15}")
        print("="*70)

        for team, costs in sorted(team_costs.items(), key=lambda x: x[1]['total'], reverse=True):
            print(f"{team:<20} ${costs['aws']:<14.2f} ${costs['gcp']:<14.2f} ${costs['azure']:<14.2f} ${costs['total']:<14.2f}")

        print("="*70)
```

## Solutions

See `solutions/` directory for complete implementations:
- `spot_trainer.py` - Spot instance training with checkpointing
- `right_sizer.py` - Instance right-sizing analyzer
- `storage_optimizer.py` - Storage lifecycle management
- `cost_dashboard.py` - Interactive cost dashboard
- `capacity_planner.py` - Reserved capacity planner
- `cost_allocator.py` - Team cost allocation

## Exercises

1. Implement spot instance training with checkpointing
2. Analyze and right-size running instances
3. Set up storage lifecycle policies across clouds
4. Build a unified cost monitoring dashboard
5. Plan and purchase reserved instances
6. Implement tag-based cost allocation

## Summary

You've completed Module 008: Cloud Platforms! You can now:
- Deploy ML infrastructure on AWS, GCP, and Azure
- Optimize costs across multiple clouds
- Implement multi-cloud strategies
- Monitor and manage cloud spending
- Use spot/preemptible instances effectively

## Next Module

**Module 009: Advanced MLOps** - Production ML systems, A/B testing, feature stores

---

*Estimated completion time: 10-12 hours*
