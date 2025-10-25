# Capstone 02: Implementation Guide (Continued)

This document continues the implementation guide from README.md, covering load testing, monitoring, deployment, and A/B testing analysis.

## Phase 5: Load Testing & Performance Validation (6-8 hours)

### 5.1 Load Testing with k6

**Install k6**:
```bash
# macOS
brew install k6

# Linux
sudo gpg -k
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6
```

**Load Test Script**:
```javascript
// tests/load/fraud-detection-load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const latency = new Trend('latency');
const fraudDetections = new Counter('fraud_detections');

// Test configuration
export let options = {
  stages: [
    // Ramp up
    { duration: '2m', target: 1000 },  // Ramp to 1000 RPS
    { duration: '3m', target: 3000 },  // Ramp to 3000 RPS
    { duration: '5m', target: 5000 },  // Ramp to 5000 RPS (target load)

    // Sustain
    { duration: '10m', target: 5000 }, // Sustain 5000 RPS

    // Spike test
    { duration: '2m', target: 10000 }, // Spike to 10000 RPS
    { duration: '3m', target: 10000 }, // Sustain spike

    // Ramp down
    { duration: '2m', target: 1000 },  // Ramp down
    { duration: '1m', target: 0 },     // Cool down
  ],

  thresholds: {
    // Latency requirements
    'http_req_duration': ['p(95)<100', 'p(99)<200'],  // P95 < 100ms, P99 < 200ms

    // Error rate
    'errors': ['rate<0.01'],  // Error rate < 1%

    // Success rate
    'http_req_failed': ['rate<0.01'],  // < 1% failures
  },
};

// Generate realistic transaction data
function generateTransaction() {
  const users = 10000;  // Simulate 10k users
  const userId = `user_${Math.floor(Math.random() * users)}`;
  const transactionId = `txn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

  const merchantCategories = [
    'grocery', 'restaurant', 'retail', 'entertainment',
    'travel', 'utilities', 'healthcare', 'education'
  ];

  // Realistic amount distribution (log-normal)
  const amount = Math.exp(Math.random() * 5 + 2);  // $7 - $740

  return {
    transaction_id: transactionId,
    user_id: userId,
    amount: amount,
    merchant_category: merchantCategories[Math.floor(Math.random() * merchantCategories.length)],
    is_international: Math.random() < 0.1,  // 10% international
    timestamp: new Date().toISOString()
  };
}

export default function() {
  const url = 'http://fraud-detection.ml-serving.svc.cluster.local:8000/predict';

  const payload = JSON.stringify(generateTransaction());

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };

  const start = new Date();
  const response = http.post(url, payload, params);
  const duration = new Date() - start;

  // Record metrics
  latency.add(duration);

  const success = check(response, {
    'status is 200': (r) => r.status === 200,
    'latency < 200ms': () => duration < 200,
    'has fraud_probability': (r) => JSON.parse(r.body).fraud_probability !== undefined,
  });

  errorRate.add(!success);

  if (success && JSON.parse(response.body).is_fraud) {
    fraudDetections.add(1);
  }

  // Think time between requests (simulates real user behavior)
  sleep(Math.random() * 0.1);  // 0-100ms
}

// Teardown function
export function teardown(data) {
  console.log('Load test complete');
}
```

**Run Load Test**:
```bash
# Run load test
k6 run tests/load/fraud-detection-load-test.js

# Run with output to InfluxDB for visualization
k6 run --out influxdb=http://localhost:8086/k6 tests/load/fraud-detection-load-test.js

# Run with custom thresholds
k6 run \
  --vus 5000 \
  --duration 10m \
  --threshold 'http_req_duration{p(99)}<100' \
  tests/load/fraud-detection-load-test.js
```

### 5.2 Stress Testing

```javascript
// tests/load/stress-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    // Gradually increase to breaking point
    { duration: '2m', target: 5000 },
    { duration: '2m', target: 10000 },
    { duration: '2m', target: 15000 },
    { duration: '2m', target: 20000 },
    { duration: '2m', target: 25000 },

    // Find breaking point
    { duration: '5m', target: 30000 },

    // Recovery
    { duration: '5m', target: 5000 },
    { duration: '2m', target: 0 },
  ],

  thresholds: {
    'http_req_duration': ['p(95)<500'],  // Relaxed threshold for stress test
  },
};

export default function() {
  const url = 'http://fraud-detection.ml-serving.svc.cluster.local:8000/predict';

  const payload = JSON.stringify({
    transaction_id: `txn_${Date.now()}_${__VU}_${__ITER}`,
    user_id: `user_${__VU % 10000}`,
    amount: Math.random() * 1000,
    merchant_category: 'retail',
    is_international: false,
    timestamp: new Date().toISOString()
  });

  const response = http.post(url, payload, {
    headers: { 'Content-Type': 'application/json' }
  });

  check(response, {
    'status is 200': (r) => r.status === 200,
  });

  sleep(0.01);  // Minimal sleep for stress test
}
```

### 5.3 Soak Testing (Endurance Test)

```javascript
// tests/load/soak-test.js
import http from 'k6/http';
import { check } from 'k6';

export let options = {
  stages: [
    // Ramp up to normal load
    { duration: '5m', target: 3000 },

    // Sustain for extended period
    { duration: '4h', target: 3000 },  // 4 hour soak test

    // Ramp down
    { duration: '5m', target: 0 },
  ],

  thresholds: {
    'http_req_duration': ['p(99)<100'],
    'http_req_failed': ['rate<0.001'],  // Very low error rate for long duration
  },
};

export default function() {
  const url = 'http://fraud-detection.ml-serving.svc.cluster.local:8000/predict';

  const payload = JSON.stringify({
    transaction_id: `txn_${Date.now()}_${Math.random()}`,
    user_id: `user_${Math.floor(Math.random() * 100000)}`,
    amount: Math.random() * 500,
    merchant_category: 'grocery',
    is_international: false,
    timestamp: new Date().toISOString()
  });

  const response = http.post(url, payload, {
    headers: { 'Content-Type': 'application/json' }
  });

  check(response, {
    'status is 200': (r) => r.status === 200,
  });
}
```

### 5.4 Performance Analysis

```python
# tests/analysis/analyze_performance.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prometheus_api_client import PrometheusConnect
from datetime import datetime, timedelta

class PerformanceAnalyzer:
    """Analyze system performance during load tests"""

    def __init__(self, prometheus_url: str):
        self.prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)

    def get_latency_percentiles(self, start_time: datetime, end_time: datetime):
        """Get latency percentiles over time"""

        query = 'histogram_quantile(0.99, rate(prediction_latency_seconds_bucket[1m]))'

        result = self.prom.custom_query_range(
            query=query,
            start_time=start_time,
            end_time=end_time,
            step='15s'
        )

        # Convert to DataFrame
        data = []
        for series in result:
            for timestamp, value in series['values']:
                data.append({
                    'timestamp': datetime.fromtimestamp(timestamp),
                    'p99_latency_ms': float(value) * 1000
                })

        df = pd.DataFrame(data)
        return df

    def get_throughput(self, start_time: datetime, end_time: datetime):
        """Get request throughput over time"""

        query = 'rate(predictions_total[1m])'

        result = self.prom.custom_query_range(
            query=query,
            start_time=start_time,
            end_time=end_time,
            step='15s'
        )

        data = []
        for series in result:
            for timestamp, value in series['values']:
                data.append({
                    'timestamp': datetime.fromtimestamp(timestamp),
                    'requests_per_second': float(value)
                })

        df = pd.DataFrame(data)
        return df

    def get_error_rate(self, start_time: datetime, end_time: datetime):
        """Get error rate over time"""

        query = 'rate(model_errors_total[1m])'

        result = self.prom.custom_query_range(
            query=query,
            start_time=start_time,
            end_time=end_time,
            step='15s'
        )

        data = []
        for series in result:
            for timestamp, value in series['values']:
                data.append({
                    'timestamp': datetime.fromtimestamp(timestamp),
                    'errors_per_second': float(value)
                })

        df = pd.DataFrame(data)
        return df

    def generate_report(self, start_time: datetime, end_time: datetime):
        """Generate performance report"""

        # Get metrics
        latency_df = self.get_latency_percentiles(start_time, end_time)
        throughput_df = self.get_throughput(start_time, end_time)
        error_df = self.get_error_rate(start_time, end_time)

        # Create visualizations
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        # Latency over time
        axes[0].plot(latency_df['timestamp'], latency_df['p99_latency_ms'])
        axes[0].axhline(y=100, color='r', linestyle='--', label='P99 Target (100ms)')
        axes[0].set_ylabel('P99 Latency (ms)')
        axes[0].set_title('P99 Latency Over Time')
        axes[0].legend()
        axes[0].grid(True)

        # Throughput over time
        axes[1].plot(throughput_df['timestamp'], throughput_df['requests_per_second'])
        axes[1].axhline(y=5000, color='g', linestyle='--', label='Target Throughput (5000 RPS)')
        axes[1].set_ylabel('Requests per Second')
        axes[1].set_title('Throughput Over Time')
        axes[1].legend()
        axes[1].grid(True)

        # Error rate
        axes[2].plot(error_df['timestamp'], error_df['errors_per_second'], color='red')
        axes[2].set_ylabel('Errors per Second')
        axes[2].set_xlabel('Time')
        axes[2].set_title('Error Rate Over Time')
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig('performance_report.png', dpi=300, bbox_inches='tight')

        print("✓ Performance report generated: performance_report.png")

        # Summary statistics
        print("\n=== Performance Summary ===")
        print(f"P99 Latency: {latency_df['p99_latency_ms'].quantile(0.99):.2f}ms")
        print(f"Max Latency: {latency_df['p99_latency_ms'].max():.2f}ms")
        print(f"Avg Throughput: {throughput_df['requests_per_second'].mean():.0f} RPS")
        print(f"Max Throughput: {throughput_df['requests_per_second'].max():.0f} RPS")
        print(f"Total Errors: {error_df['errors_per_second'].sum():.0f}")

if __name__ == '__main__':
    analyzer = PerformanceAnalyzer(prometheus_url='http://prometheus.monitoring.svc.cluster.local:9090')

    # Analyze last load test (adjust times as needed)
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)

    analyzer.generate_report(start_time, end_time)
```

## Phase 6: Comprehensive Monitoring (6-8 hours)

### 6.1 Grafana Dashboards

**Dashboard JSON**:
```json
{
  "dashboard": {
    "title": "Fraud Detection System - Production",
    "tags": ["fraud-detection", "ml", "production"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Prediction Latency (P99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, rate(prediction_latency_seconds_bucket[5m]))"
          }
        ],
        "yaxes": [
          {
            "format": "ms",
            "label": "Latency"
          }
        ],
        "alert": {
          "conditions": [
            {
              "type": "query",
              "query": {
                "params": ["A", "5m", "now"]
              },
              "reducer": {
                "type": "avg"
              },
              "evaluator": {
                "type": "gt",
                "params": [100]
              }
            }
          ],
          "name": "High P99 Latency Alert",
          "message": "P99 latency exceeded 100ms threshold"
        }
      },
      {
        "title": "Throughput (Requests/sec)",
        "targets": [
          {
            "expr": "rate(predictions_total[1m])"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(model_errors_total[5m])"
          }
        ]
      },
      {
        "title": "Fraud Detection Rate",
        "targets": [
          {
            "expr": "rate(predictions_total{prediction=\"fraud\"}[5m]) / rate(predictions_total[5m])"
          }
        ],
        "yaxes": [
          {
            "format": "percentunit"
          }
        ]
      },
      {
        "title": "Model A/B Distribution",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum by (model_version) (predictions_total)"
          }
        ]
      },
      {
        "title": "Feature Retrieval Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, rate(feature_retrieval_latency_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Kafka Consumer Lag",
        "targets": [
          {
            "expr": "kafka_consumergroup_lag"
          }
        ]
      },
      {
        "title": "Redis Memory Usage",
        "targets": [
          {
            "expr": "redis_memory_used_bytes / redis_memory_max_bytes"
          }
        ]
      }
    ],
    "time": {
      "from": "now-6h",
      "to": "now"
    },
    "refresh": "10s"
  }
}
```

### 6.2 Alerting Rules

```yaml
# kubernetes/monitoring/prometheus-rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: fraud-detection-alerts
  namespace: monitoring
spec:
  groups:
  - name: fraud-detection
    interval: 30s
    rules:
    # Latency alerts
    - alert: HighP99Latency
      expr: histogram_quantile(0.99, rate(prediction_latency_seconds_bucket[5m])) > 0.1
      for: 2m
      labels:
        severity: warning
      annotations:
        summary: "High P99 prediction latency"
        description: "P99 latency is {{ $value | humanizeDuration }} (threshold: 100ms)"

    - alert: CriticalLatency
      expr: histogram_quantile(0.99, rate(prediction_latency_seconds_bucket[5m])) > 0.2
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "Critical prediction latency"
        description: "P99 latency is {{ $value | humanizeDuration }} (threshold: 200ms)"

    # Error rate alerts
    - alert: HighErrorRate
      expr: rate(model_errors_total[5m]) > 10
      for: 2m
      labels:
        severity: warning
      annotations:
        summary: "High model error rate"
        description: "{{ $value | humanize }} errors per second"

    # Throughput alerts
    - alert: LowThroughput
      expr: rate(predictions_total[5m]) < 100
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "Unusually low prediction throughput"
        description: "Only {{ $value | humanize }} predictions per second"

    # Kafka lag alerts
    - alert: KafkaConsumerLag
      expr: kafka_consumergroup_lag > 10000
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High Kafka consumer lag"
        description: "Consumer lag is {{ $value }} messages"

    # Redis alerts
    - alert: RedisMemoryHigh
      expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.9
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "Redis memory usage high"
        description: "Redis memory is {{ $value | humanizePercentage }} full"

    - alert: RedisDown
      expr: up{job="redis"} == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "Redis is down"
        description: "Redis instance {{ $labels.instance }} is down"

    # Model performance alerts
    - alert: AbnormalFraudRate
      expr: |
        rate(predictions_total{prediction="fraud"}[30m]) /
        rate(predictions_total[30m]) > 0.15
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "Abnormally high fraud detection rate"
        description: "Fraud rate is {{ $value | humanizePercentage }} (expected <10%)"

    - alert: LowFraudRate
      expr: |
        rate(predictions_total{prediction="fraud"}[30m]) /
        rate(predictions_total[30m]) < 0.01
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "Abnormally low fraud detection rate"
        description: "Fraud rate is {{ $value | humanizePercentage }} (expected >1%)"
```

## Phase 7: A/B Testing Analysis (4-6 hours)

### 7.1 A/B Test Results Collector

```python
# src/ab-testing/results_collector.py
from kafka import KafkaConsumer
import json
from collections import defaultdict
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy import stats

class ABTestCollector:
    """Collect A/B test results from Kafka"""

    def __init__(self, kafka_bootstrap_servers: str):
        self.consumer = KafkaConsumer(
            'predictions',
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',
            group_id='ab-test-collector'
        )

        self.results = {
            'v1': [],
            'v2': []
        }

    def collect(self, duration_minutes: int = 60):
        """Collect results for specified duration"""

        end_time = datetime.now() + timedelta(minutes=duration_minutes)

        print(f"Collecting A/B test results for {duration_minutes} minutes...")

        for message in self.consumer:
            result = message.value

            model_version = result['model_version']

            if model_version in self.results:
                self.results[model_version].append({
                    'transaction_id': result['transaction_id'],
                    'is_fraud': result['is_fraud'],
                    'fraud_probability': result['fraud_probability'],
                    'latency_ms': result['latency_ms'],
                    'timestamp': result['timestamp']
                })

            if datetime.now() >= end_time:
                break

            # Progress update
            if len(self.results['v1']) % 1000 == 0:
                print(f"Collected: v1={len(self.results['v1'])}, v2={len(self.results['v2'])}")

        print(f"\n✓ Collection complete")
        print(f"  Model v1: {len(self.results['v1'])} predictions")
        print(f"  Model v2: {len(self.results['v2'])} predictions")

    def analyze(self):
        """Analyze A/B test results"""

        df_v1 = pd.DataFrame(self.results['v1'])
        df_v2 = pd.DataFrame(self.results['v2'])

        print("\n=== A/B Test Analysis ===\n")

        # Latency comparison
        print("## Latency Comparison")
        print(f"Model v1: P50={df_v1['latency_ms'].quantile(0.50):.2f}ms, "
              f"P95={df_v1['latency_ms'].quantile(0.95):.2f}ms, "
              f"P99={df_v1['latency_ms'].quantile(0.99):.2f}ms")

        print(f"Model v2: P50={df_v2['latency_ms'].quantile(0.50):.2f}ms, "
              f"P95={df_v2['latency_ms'].quantile(0.95):.2f}ms, "
              f"P99={df_v2['latency_ms'].quantile(0.99):.2f}ms")

        # Statistical test for latency difference
        t_stat, p_value = stats.ttest_ind(df_v1['latency_ms'], df_v2['latency_ms'])
        print(f"T-test: t={t_stat:.4f}, p={p_value:.4f}")

        if p_value < 0.05:
            winner = 'v1' if df_v1['latency_ms'].mean() < df_v2['latency_ms'].mean() else 'v2'
            print(f"✓ Significant latency difference (winner: {winner})")
        else:
            print("No significant latency difference")

        # Fraud detection rate comparison
        print("\n## Fraud Detection Rate")
        fraud_rate_v1 = df_v1['is_fraud'].mean()
        fraud_rate_v2 = df_v2['is_fraud'].mean()

        print(f"Model v1: {fraud_rate_v1:.2%}")
        print(f"Model v2: {fraud_rate_v2:.2%}")

        # Chi-square test for fraud rate difference
        from scipy.stats import chi2_contingency

        contingency_table = pd.crosstab(
            [df_v1['is_fraud'].tolist() + df_v2['is_fraud'].tolist()],
            ['v1'] * len(df_v1) + ['v2'] * len(df_v2)
        )

        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-square test: χ²={chi2:.4f}, p={p_value:.4f}")

        if p_value < 0.05:
            print("✓ Significant fraud rate difference")
        else:
            print("No significant fraud rate difference")

        # Confidence intervals for fraud probability
        print("\n## Fraud Probability Distribution")
        print(f"Model v1: mean={df_v1['fraud_probability'].mean():.4f}, "
              f"std={df_v1['fraud_probability'].std():.4f}")
        print(f"Model v2: mean={df_v2['fraud_probability'].mean():.4f}, "
              f"std={df_v2['fraud_probability'].std():.4f}")

        # Recommendation
        print("\n## Recommendation")

        # Decision criteria
        latency_improvement = (df_v1['latency_ms'].mean() - df_v2['latency_ms'].mean()) / df_v1['latency_ms'].mean()
        fraud_rate_change = (fraud_rate_v2 - fraud_rate_v1) / fraud_rate_v1

        if latency_improvement > 0.05 and fraud_rate_change > -0.05:
            print("✓ Recommend promoting Model v2 to 100% traffic")
            print(f"  - Latency improvement: {latency_improvement:.1%}")
            print(f"  - Fraud rate change: {fraud_rate_change:+.1%}")
        elif latency_improvement < -0.1:
            print("⚠️  Model v2 has worse latency, keep Model v1")
        elif fraud_rate_change < -0.1:
            print("⚠️  Model v2 detects less fraud, keep Model v1")
        else:
            print("→ Continue A/B test for more data")

        # Save detailed report
        report = {
            'model_v1': {
                'count': len(df_v1),
                'latency_p50': float(df_v1['latency_ms'].quantile(0.50)),
                'latency_p95': float(df_v1['latency_ms'].quantile(0.95)),
                'latency_p99': float(df_v1['latency_ms'].quantile(0.99)),
                'fraud_rate': float(fraud_rate_v1)
            },
            'model_v2': {
                'count': len(df_v2),
                'latency_p50': float(df_v2['latency_ms'].quantile(0.50)),
                'latency_p95': float(df_v2['latency_ms'].quantile(0.95)),
                'latency_p99': float(df_v2['latency_ms'].quantile(0.99)),
                'fraud_rate': float(fraud_rate_v2)
            },
            'statistical_tests': {
                'latency_ttest_pvalue': float(p_value),
                'fraud_rate_chi2_pvalue': float(p_value)
            }
        }

        with open('ab_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        print("\n✓ Detailed report saved to ab_test_report.json")

if __name__ == '__main__':
    collector = ABTestCollector(
        kafka_bootstrap_servers='fraud-detection-kafka-kafka-bootstrap.streaming.svc.cluster.local:9092'
    )

    # Collect for 1 hour
    collector.collect(duration_minutes=60)

    # Analyze results
    collector.analyze()
```

## Phase 8: Production Deployment (4-6 hours)

### 8.1 Kubernetes Deployment

```yaml
# kubernetes/inference/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-detection
  namespace: ml-serving
spec:
  replicas: 10
  selector:
    matchLabels:
      app: fraud-detection
  template:
    metadata:
      labels:
        app: fraud-detection
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - fraud-detection
              topologyKey: kubernetes.io/hostname
      containers:
      - name: inference
        image: your-registry/fraud-detection:v1.0.0
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: REDIS_HOST
          value: redis.features.svc.cluster.local
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: fraud-detection-kafka-kafka-bootstrap.streaming.svc.cluster.local:9092
        - name: MODEL_VERSION_A
          value: "v1"
        - name: MODEL_VERSION_B
          value: "v2"
        - name: AB_TEST_SPLIT
          value: "90:10"
        resources:
          requests:
            cpu: 2
            memory: 4Gi
          limits:
            cpu: 4
            memory: 8Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        volumeMounts:
        - name: models
          mountPath: /app/models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: model-storage
---
apiVersion: v1
kind: Service
metadata:
  name: fraud-detection
  namespace: ml-serving
spec:
  selector:
    app: fraud-detection
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-detection-hpa
  namespace: ml-serving
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-detection
  minReplicas: 10
  maxReplicas: 100
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: prediction_latency_p99
      target:
        type: AverageValue
        averageValue: "100m"  # 100ms
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 5
        periodSeconds: 30
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60
```

### 8.2 Deployment Checklist

```markdown
# Fraud Detection System - Production Deployment Checklist

## Infrastructure

- [ ] Kafka cluster deployed and healthy (3+ brokers)
- [ ] Redis cluster deployed with replication
- [ ] Kubernetes cluster has sufficient capacity
- [ ] Storage provisioned for models and data
- [ ] Network policies configured
- [ ] DNS records configured

## Application

- [ ] Inference service deployed with 10+ replicas
- [ ] Feature processor deployed
- [ ] Models loaded and verified
- [ ] Health checks passing
- [ ] Auto-scaling configured

## Data Pipeline

- [ ] Kafka topics created with correct partitions
- [ ] Consumer groups configured
- [ ] Data retention policies set
- [ ] Monitoring for consumer lag enabled

## Performance

- [ ] Load testing completed successfully
- [ ] P99 latency < 100ms verified
- [ ] Sustained 5000 TPS achieved
- [ ] Auto-scaling tested under load
- [ ] Resource limits properly set

## Monitoring

- [ ] Prometheus scraping metrics
- [ ] Grafana dashboards configured
- [ ] Alerting rules deployed
- [ ] Alert routing configured (PagerDuty/Slack)
- [ ] Logs centralized (ELK/Loki)

## Security

- [ ] Network policies enforced
- [ ] TLS enabled for Kafka
- [ ] Secrets stored securely
- [ ] RBAC configured
- [ ] Vulnerability scanning completed

## Documentation

- [ ] Architecture diagrams updated
- [ ] Runbooks created
- [ ] Troubleshooting guide available
- [ ] On-call rotation established

## Disaster Recovery

- [ ] Backup procedures documented
- [ ] Recovery time objective (RTO) defined
- [ ] Failover tested
- [ ] Data retention policies set

## Go-Live

- [ ] Stakeholders notified
- [ ] Traffic gradually ramped up
- [ ] Metrics monitored in real-time
- [ ] Rollback plan ready
- [ ] Post-deployment review scheduled
```

## Summary

Congratulations! You've completed **Capstone 02: Real-Time Fraud Detection System**!

### What You've Built

A production-ready, high-performance fraud detection system featuring:
- ✅ Kafka streaming infrastructure (5000+ TPS)
- ✅ Redis feature store (<5ms latency)
- ✅ ONNX-optimized inference (<100ms P99)
- ✅ A/B testing framework
- ✅ Comprehensive load testing
- ✅ Production monitoring and alerting
- ✅ Auto-scaling infrastructure

### Skills Demonstrated

- Real-time streaming systems with Kafka
- Ultra-low latency optimization
- High-throughput system design
- Performance testing and benchmarking
- A/B testing methodology
- Production ML system operations
- Scalability and reliability engineering

### Performance Achieved

- **Latency**: P99 < 100ms ✓
- **Throughput**: 5,000+ TPS sustained ✓
- **Availability**: 99.99% target ✓
- **Scalability**: Auto-scale 10-100 pods ✓

### Estimated Time Spent

- Phase 1: Streaming Infrastructure (8-10 hours)
- Phase 2: Feature Pipeline (8-10 hours)
- Phase 3: Model Optimization (8-10 hours)
- Phase 4: Inference Service (10-12 hours)
- Phase 5: Load Testing (6-8 hours)
- Phase 6: Monitoring (6-8 hours)
- Phase 7: A/B Testing (4-6 hours)
- Phase 8: Deployment (4-6 hours)

**Total: 54-70 hours** (within the 35-45 hour target when following the guide)

### Portfolio Value

This project demonstrates:
- Real-time ML systems expertise
- Performance optimization skills
- Production-scale experience
- Data engineering capabilities
- DevOps and SRE knowledge

Perfect for interviews at fintech companies, payment processors, and high-scale ML infrastructure teams!

---

**Next**: Proceed to **Capstone 03: Multi-Cloud ML Infrastructure**
