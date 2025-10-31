# Capstone 02: Real-Time Fraud Detection System

Build a production-ready, real-time fraud detection system processing thousands of transactions per second with sub-100ms latency.

## Project Overview

**Duration**: 35-45 hours
**Difficulty**: Advanced
**Type**: Real-time streaming system

### Business Context

You're building a real-time fraud detection system for a payment processor that handles:
- 5,000+ transactions per second during peak hours
- Strict latency requirements (<100ms P99)
- High accuracy requirements (minimize false positives while catching fraud)
- Need for continuous model improvement through A/B testing
- 24/7 availability with zero tolerance for downtime

### Success Metrics

The system must achieve:
- **Latency**: P99 latency <100ms for predictions
- **Throughput**: Handle 5,000 TPS sustained
- **Availability**: 99.99% uptime (less than 1 hour downtime per year)
- **Accuracy**: Fraud detection rate >95%, false positive rate <2%
- **Scalability**: Auto-scale from 1,000 to 10,000 TPS
- **Recovery**: <5 minute recovery time from failures

## Learning Objectives

By completing this project, you will:
- Build real-time ML inference systems
- Implement streaming data pipelines with Kafka
- Optimize for ultra-low latency (<100ms)
- Design high-throughput systems (5,000+ TPS)
- Implement A/B testing for ML models
- Perform load testing and capacity planning
- Handle failure scenarios and disaster recovery
- Monitor production ML systems at scale

## Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│             Real-Time Fraud Detection Architecture                │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────┐         ┌────────────────┐                  │
│  │  Transaction   │   →     │  Kafka Topic   │                  │
│  │    Stream      │         │ (transactions) │                  │
│  └────────────────┘         └────────────────┘                  │
│                                      ↓                            │
│                        ┌──────────────────────────┐              │
│                        │  Feature Engineering     │              │
│                        │     (Kafka Streams)      │              │
│                        └──────────────────────────┘              │
│                                      ↓                            │
│                        ┌──────────────────────────┐              │
│                        │    Redis Feature Store   │              │
│                        │   (Online features)      │              │
│                        └──────────────────────────┘              │
│                                      ↓                            │
│                        ┌──────────────────────────┐              │
│                        │  ML Inference Service    │              │
│                        │    (FastAPI + ONNX)      │              │
│                        │  ┌────────┐  ┌────────┐ │              │
│                        │  │Model A │  │Model B │ │  (A/B Test)  │
│                        │  │ (90%)  │  │ (10%)  │ │              │
│                        │  └────────┘  └────────┘ │              │
│                        └──────────────────────────┘              │
│                                      ↓                            │
│                        ┌──────────────────────────┐              │
│                        │   Kafka Topic (results)  │              │
│                        └──────────────────────────┘              │
│                                      ↓                            │
│            ┌───────────────────────────────────────┐             │
│            │        Downstream Consumers            │             │
│            │  ┌────────┐  ┌────────┐  ┌────────┐  │             │
│            │  │Payment │  │Logging │  │Metrics │  │             │
│            │  │System  │  │        │  │        │  │             │
│            │  └────────┘  └────────┘  └────────┘  │             │
│            └───────────────────────────────────────┘             │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  Monitoring & Analytics                    │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │  │
│  │  │Prometheus│  │  Grafana │  │  Kafka   │  │  Model   │ │  │
│  │  │          │  │          │  │  Manager │  │ Monitor  │ │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Transaction Event
     ↓
Kafka Ingestion (<5ms)
     ↓
Feature Enrichment (<10ms)
     ↓
Redis Feature Lookup (<5ms)
     ↓
Model Inference (<20ms)
     ↓
Post-processing (<5ms)
     ↓
Result Published (<5ms)
──────────────────
Total: <50ms (P50)
       <100ms (P99)
```

## Prerequisites

### Knowledge Requirements
- Strong Python programming
- Understanding of streaming systems (Kafka)
- Experience with caching (Redis)
- Machine learning basics
- Performance optimization techniques
- Load testing and benchmarking

### Infrastructure Requirements
- Kubernetes cluster
- Kafka cluster (or managed service)
- Redis cluster
- Minimum resources:
  - 32GB RAM
  - 16 CPU cores
  - 200GB SSD storage

### Tools Required
- `kubectl`
- `kafka-console-producer/consumer`
- `redis-cli`
- `k6` or `locust` (load testing)

## Phase 1: Streaming Infrastructure Setup (8-10 hours)

### 1.1 Kafka Cluster Deployment

**Using Strimzi Operator on Kubernetes**:

```yaml
# kubernetes/kafka/kafka-cluster.yaml
apiVersion: kafka.strimzi.io/v1beta2
kind: Kafka
metadata:
  name: fraud-detection-kafka
  namespace: streaming
spec:
  kafka:
    version: 3.5.0
    replicas: 3
    listeners:
      - name: plain
        port: 9092
        type: internal
        tls: false
      - name: tls
        port: 9093
        type: internal
        tls: true
    config:
      offsets.topic.replication.factor: 3
      transaction.state.log.replication.factor: 3
      transaction.state.log.min.isr: 2
      default.replication.factor: 3
      min.insync.replicas: 2
      inter.broker.protocol.version: "3.5"
      # Performance tuning
      num.network.threads: 8
      num.io.threads: 16
      socket.send.buffer.bytes: 102400
      socket.receive.buffer.bytes: 102400
      socket.request.max.bytes: 104857600
      log.retention.hours: 168
      log.segment.bytes: 1073741824
      compression.type: lz4
    storage:
      type: persistent-claim
      size: 100Gi
      class: fast-ssd
    resources:
      requests:
        memory: 8Gi
        cpu: 2
      limits:
        memory: 16Gi
        cpu: 4
  zookeeper:
    replicas: 3
    storage:
      type: persistent-claim
      size: 10Gi
      class: fast-ssd
    resources:
      requests:
        memory: 2Gi
        cpu: 1
      limits:
        memory: 4Gi
        cpu: 2
  entityOperator:
    topicOperator: {}
    userOperator: {}
```

**Topic Definitions**:

```yaml
# kubernetes/kafka/topics.yaml
---
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaTopic
metadata:
  name: transactions
  namespace: streaming
  labels:
    strimzi.io/cluster: fraud-detection-kafka
spec:
  partitions: 12
  replicas: 3
  config:
    retention.ms: 604800000  # 7 days
    compression.type: lz4
    min.insync.replicas: 2
    max.message.bytes: 1048576
---
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaTopic
metadata:
  name: predictions
  namespace: streaming
  labels:
    strimzi.io/cluster: fraud-detection-kafka
spec:
  partitions: 12
  replicas: 3
  config:
    retention.ms: 2592000000  # 30 days
    compression.type: lz4
---
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaTopic
metadata:
  name: feature-updates
  namespace: streaming
  labels:
    strimzi.io/cluster: fraud-detection-kafka
spec:
  partitions: 12
  replicas: 3
  config:
    retention.ms: 3600000  # 1 hour (fast-moving features)
    compression.type: lz4
```

### 1.2 Redis Cluster for Features

**Redis Cluster Deployment**:

```yaml
# kubernetes/redis/redis-cluster.yaml
apiVersion: redis.redis.opstreelabs.in/v1beta1
kind: RedisCluster
metadata:
  name: fraud-features
  namespace: features
spec:
  clusterSize: 6
  kubernetesConfig:
    image: redis:7.0-alpine
    imagePullPolicy: IfNotPresent
    resources:
      requests:
        cpu: 1
        memory: 4Gi
      limits:
        cpu: 2
        memory: 8Gi
  storage:
    volumeClaimTemplate:
      spec:
        accessModes:
          - ReadWriteOnce
        resources:
          requests:
            storage: 50Gi
        storageClassName: fast-ssd
  redisExporter:
    enabled: true
    image: oliver006/redis_exporter:latest
  redisConfig:
    additionalRedisConfig: |
      maxmemory 7gb
      maxmemory-policy allkeys-lru
      save ""
      appendonly no
      tcp-backlog 511
      timeout 0
      tcp-keepalive 300
```

**Alternative: Redis Standalone with High Availability**:

```yaml
# kubernetes/redis/redis-ha.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
  namespace: features
spec:
  serviceName: redis
  replicas: 3
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7.0-alpine
        command:
          - redis-server
          - --appendonly
          - "no"
          - --save
          - ""
          - --maxmemory
          - "7gb"
          - --maxmemory-policy
          - allkeys-lru
        ports:
        - containerPort: 6379
          name: redis
        resources:
          requests:
            cpu: 2
            memory: 8Gi
          limits:
            cpu: 4
            memory: 16Gi
        volumeMounts:
        - name: data
          mountPath: /data
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 50Gi
```

## Phase 2: Feature Engineering Pipeline (8-10 hours)

### 2.1 Real-Time Feature Processing

```python
# src/feature-pipeline/stream_processor.py
from kafka import KafkaConsumer, KafkaProducer
import redis
import json
from typing import Dict
import time
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

class FeatureProcessor:
    """Real-time feature processor"""

    def __init__(
        self,
        kafka_bootstrap_servers: str,
        redis_host: str,
        redis_port: int = 6379
    ):
        self.consumer = KafkaConsumer(
            'transactions',
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True,
            max_poll_records=1000,
            fetch_min_bytes=1024,
            fetch_max_wait_ms=500
        )

        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            compression_type='lz4',
            linger_ms=10,
            batch_size=32768
        )

        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1
        )

        # In-memory state for aggregations
        self.user_windows = defaultdict(list)
        self.window_size = timedelta(minutes=30)

    def compute_user_features(self, user_id: str, transaction: Dict) -> Dict:
        """Compute real-time user features"""

        # Get historical features from Redis
        user_key = f"user:{user_id}"

        # Use pipelining for faster Redis operations
        pipe = self.redis_client.pipeline()
        pipe.hget(user_key, "total_transactions")
        pipe.hget(user_key, "total_amount")
        pipe.hget(user_key, "avg_amount")
        pipe.hget(user_key, "last_transaction_time")

        results = pipe.execute()

        total_transactions = int(results[0] or 0)
        total_amount = float(results[1] or 0)
        avg_amount = float(results[2] or 0)
        last_transaction_time = results[3]

        # Compute time since last transaction
        if last_transaction_time:
            last_time = datetime.fromisoformat(last_transaction_time)
            time_since_last = (datetime.now() - last_time).total_seconds()
        else:
            time_since_last = None

        # Compute windowed features
        current_time = datetime.now()
        window_start = current_time - self.window_size

        # Add current transaction to window
        self.user_windows[user_id].append({
            'amount': transaction['amount'],
            'timestamp': current_time
        })

        # Remove old transactions from window
        self.user_windows[user_id] = [
            t for t in self.user_windows[user_id]
            if t['timestamp'] > window_start
        ]

        # Compute window statistics
        window_transactions = self.user_windows[user_id]
        if window_transactions:
            window_amounts = [t['amount'] for t in window_transactions]
            window_count = len(window_amounts)
            window_sum = sum(window_amounts)
            window_avg = np.mean(window_amounts)
            window_std = np.std(window_amounts)
        else:
            window_count = 0
            window_sum = 0
            window_avg = 0
            window_std = 0

        features = {
            # Historical features
            'total_transactions': total_transactions,
            'total_amount': total_amount,
            'avg_amount': avg_amount,
            'time_since_last_transaction': time_since_last,

            # Window features (last 30 minutes)
            'window_transaction_count': window_count,
            'window_transaction_sum': window_sum,
            'window_transaction_avg': window_avg,
            'window_transaction_std': window_std,

            # Velocity features
            'transaction_velocity': window_count / 30.0,  # transactions per minute

            # Deviation from normal
            'amount_deviation': (transaction['amount'] - avg_amount) / (avg_amount + 1),
        }

        return features

    def update_user_features(self, user_id: str, transaction: Dict):
        """Update user features in Redis"""

        user_key = f"user:{user_id}"
        current_time = datetime.now().isoformat()

        # Get current values
        pipe = self.redis_client.pipeline()
        pipe.hget(user_key, "total_transactions")
        pipe.hget(user_key, "total_amount")

        results = pipe.execute()

        total_transactions = int(results[0] or 0)
        total_amount = float(results[1] or 0)

        # Update values
        new_total_transactions = total_transactions + 1
        new_total_amount = total_amount + transaction['amount']
        new_avg_amount = new_total_amount / new_total_transactions

        # Use pipelining for batch update
        pipe = self.redis_client.pipeline()
        pipe.hset(user_key, "total_transactions", new_total_transactions)
        pipe.hset(user_key, "total_amount", new_total_amount)
        pipe.hset(user_key, "avg_amount", new_avg_amount)
        pipe.hset(user_key, "last_transaction_time", current_time)
        pipe.expire(user_key, 86400 * 90)  # 90 day TTL

        pipe.execute()

    def process_transaction(self, transaction: Dict) -> Dict:
        """Process single transaction"""

        start_time = time.time()

        user_id = transaction['user_id']

        # Compute features
        user_features = self.compute_user_features(user_id, transaction)

        # Combine transaction data with features
        enriched_transaction = {
            **transaction,
            'features': user_features,
            'processing_time_ms': (time.time() - start_time) * 1000
        }

        # Update user state
        self.update_user_features(user_id, transaction)

        # Publish enriched transaction for inference
        self.producer.send('feature-updates', value=enriched_transaction)

        return enriched_transaction

    def run(self):
        """Run feature processor"""

        print("Starting feature processor...")

        for message in self.consumer:
            try:
                transaction = message.value

                enriched = self.process_transaction(transaction)

                # Log processing time
                if enriched['processing_time_ms'] > 10:
                    print(f"⚠️  Slow processing: {enriched['processing_time_ms']:.2f}ms")

            except Exception as e:
                print(f"Error processing transaction: {e}")
                continue

if __name__ == '__main__':
    processor = FeatureProcessor(
        kafka_bootstrap_servers='fraud-detection-kafka-kafka-bootstrap.streaming.svc.cluster.local:9092',
        redis_host='redis.features.svc.cluster.local'
    )

    processor.run()
```

### 2.2 Feature Store Interface

```python
# src/features/feature_store.py
import redis
from typing import Dict, List, Optional
import json
import time

class FeatureStore:
    """Redis-based feature store optimized for low latency"""

    def __init__(self, redis_host: str, redis_port: int = 6379):
        # Connection pool for better performance
        self.pool = redis.ConnectionPool(
            host=redis_host,
            port=redis_port,
            max_connections=100,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1
        )
        self.client = redis.Redis(connection_pool=self.pool)

    def get_user_features(self, user_id: str) -> Dict:
        """Get user features with minimal latency"""

        start = time.time()

        user_key = f"user:{user_id}"

        # Use pipelining to fetch all features in one round trip
        pipe = self.client.pipeline()
        pipe.hgetall(user_key)
        results = pipe.execute()

        features = results[0]

        # Convert string values to appropriate types
        typed_features = {}
        for key, value in features.items():
            try:
                # Try to convert to float
                typed_features[key] = float(value)
            except ValueError:
                # Keep as string if conversion fails
                typed_features[key] = value

        latency_ms = (time.time() - start) * 1000

        if latency_ms > 5:
            print(f"⚠️  Slow feature retrieval: {latency_ms:.2f}ms for user {user_id}")

        return typed_features

    def get_batch_user_features(self, user_ids: List[str]) -> Dict[str, Dict]:
        """Get features for multiple users efficiently"""

        start = time.time()

        # Use pipelining for batch retrieval
        pipe = self.client.pipeline()

        for user_id in user_ids:
            user_key = f"user:{user_id}"
            pipe.hgetall(user_key)

        results = pipe.execute()

        # Convert results
        batch_features = {}
        for user_id, features in zip(user_ids, results):
            typed_features = {}
            for key, value in features.items():
                try:
                    typed_features[key] = float(value)
                except ValueError:
                    typed_features[key] = value

            batch_features[user_id] = typed_features

        latency_ms = (time.time() - start) * 1000

        print(f"Batch feature retrieval: {len(user_ids)} users in {latency_ms:.2f}ms "
              f"({latency_ms/len(user_ids):.2f}ms per user)")

        return batch_features

    def set_user_features(self, user_id: str, features: Dict, ttl: int = 86400 * 90):
        """Set user features"""

        user_key = f"user:{user_id}"

        pipe = self.client.pipeline()
        pipe.hmset(user_key, features)
        pipe.expire(user_key, ttl)
        pipe.execute()

    def health_check(self) -> bool:
        """Check if Redis is healthy"""
        try:
            self.client.ping()
            return True
        except:
            return False
```

## Phase 3: ML Model Optimization (8-10 hours)

### 3.1 Model Training with Performance Focus

```python
# src/training/train_optimized.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import mlflow
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def train_fraud_model(data_path: str):
    """Train fraud detection model optimized for inference speed"""

    # Load data
    df = pd.read_parquet(data_path)

    # Feature engineering
    feature_cols = [
        'total_transactions',
        'total_amount',
        'avg_amount',
        'time_since_last_transaction',
        'window_transaction_count',
        'window_transaction_avg',
        'window_transaction_std',
        'transaction_velocity',
        'amount_deviation',
        'amount',
        'hour_of_day',
        'day_of_week',
        'is_international'
    ]

    X = df[feature_cols]
    y = df['is_fraud']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Fraud rate: {y.mean():.2%}")

    # MLflow tracking
    mlflow.set_experiment('fraud-detection-production')

    with mlflow.start_run():
        # Train model (optimized for speed vs accuracy trade-off)
        model = RandomForestClassifier(
            n_estimators=100,  # Fewer trees for faster inference
            max_depth=10,      # Shallower trees
            min_samples_leaf=10,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = (y_pred == y_test).mean()
        roc_auc = roc_auc_score(y_test, y_proba)

        print(f"\nModel Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(classification_report(y_test, y_pred))

        # Log metrics
        mlflow.log_params({
            "n_estimators": 100,
            "max_depth": 10,
            "n_features": len(feature_cols)
        })

        mlflow.log_metrics({
            "accuracy": accuracy,
            "roc_auc": roc_auc
        })

        # Convert to ONNX for optimized inference
        print("\nConverting to ONNX...")

        initial_type = [('float_input', FloatTensorType([None, len(feature_cols)]))]

        onnx_model = convert_sklearn(
            model,
            initial_types=initial_type,
            target_opset=12
        )

        # Save ONNX model
        onnx_path = "fraud_model.onnx"
        onnx.save_model(onnx_model, onnx_path)

        mlflow.log_artifact(onnx_path)

        # Benchmark inference speed
        print("\nBenchmarking inference speed...")

        # ONNX Runtime session
        sess = ort.InferenceSession(onnx_path)

        import time

        # Warm up
        for _ in range(100):
            sess.run(None, {'float_input': X_test[:1].values.astype(np.float32)})

        # Benchmark
        n_iterations = 1000
        start = time.time()

        for i in range(n_iterations):
            sess.run(None, {'float_input': X_test[i:i+1].values.astype(np.float32)})

        elapsed = time.time() - start
        avg_latency_ms = (elapsed / n_iterations) * 1000

        print(f"Average inference latency: {avg_latency_ms:.2f}ms")
        print(f"Throughput: {1000/avg_latency_ms:.0f} predictions/second")

        mlflow.log_metric("inference_latency_ms", avg_latency_ms)
        mlflow.log_metric("throughput_per_sec", 1000/avg_latency_ms)

        # Save feature names for inference service
        feature_metadata = {
            'feature_names': feature_cols,
            'model_type': 'RandomForest',
            'onnx_version': onnx.__version__
        }

        mlflow.log_dict(feature_metadata, "feature_metadata.json")

        print(f"\n✓ Model training complete. Run ID: {mlflow.active_run().info.run_id}")

if __name__ == '__main__':
    train_fraud_model('data/fraud_data.parquet')
```

## Phase 4: High-Performance Inference Service (10-12 hours)

### 4.1 FastAPI Inference Service

```python
# src/inference/service.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional
import onnxruntime as ort
import numpy as np
from prometheus_client import Counter, Histogram, Gauge
import time
import redis
from kafka import KafkaProducer
import json
import hashlib

app = FastAPI(title="Fraud Detection Service")

# ============================================================================
# Metrics
# ============================================================================

prediction_counter = Counter(
    'predictions_total',
    'Total predictions made',
    ['model_version', 'prediction']
)

prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency',
    ['model_version'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5]
)

feature_latency = Histogram(
    'feature_retrieval_latency_seconds',
    'Feature retrieval latency',
    buckets=[0.001, 0.002, 0.005, 0.01, 0.025, 0.05]
)

model_errors = Counter(
    'model_errors_total',
    'Model prediction errors',
    ['error_type']
)

active_model_gauge = Gauge(
    'active_model_version',
    'Current active model version'
)

# ============================================================================
# Models
# ============================================================================

class PredictionRequest(BaseModel):
    """Prediction request"""
    transaction_id: str
    user_id: str
    amount: float
    merchant_category: str
    is_international: bool
    timestamp: str

class PredictionResponse(BaseModel):
    """Prediction response"""
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    model_version: str
    latency_ms: float

# ============================================================================
# Model Management
# ============================================================================

class ModelManager:
    """Manage multiple model versions for A/B testing"""

    def __init__(self):
        self.models = {}
        self.feature_names = []
        self.model_weights = {}  # For A/B testing

    def load_model(self, model_path: str, version: str, weight: float = 1.0):
        """Load ONNX model"""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 2

        session = ort.InferenceSession(model_path, sess_options)

        self.models[version] = session
        self.model_weights[version] = weight

        print(f"✓ Loaded model version {version} (weight: {weight})")

    def get_model(self, transaction_id: str) -> tuple:
        """Select model for prediction (A/B testing)"""

        # Use transaction ID for consistent model selection
        hash_value = int(hashlib.md5(transaction_id.encode()).hexdigest(), 16)
        selector = (hash_value % 100) / 100.0

        cumulative_weight = 0
        for version, weight in self.model_weights.items():
            cumulative_weight += weight
            if selector < cumulative_weight:
                return version, self.models[version]

        # Fallback to first model
        version = list(self.models.keys())[0]
        return version, self.models[version]

# Initialize model manager
model_manager = ModelManager()

# Load models (A/B testing: 90% model A, 10% model B)
model_manager.load_model("models/model_v1.onnx", "v1", weight=0.9)
model_manager.load_model("models/model_v2.onnx", "v2", weight=0.1)

active_model_gauge.set(1)

# Feature store client
feature_store = redis.Redis(
    host='redis.features.svc.cluster.local',
    port=6379,
    decode_responses=True,
    socket_connect_timeout=0.5,
    socket_timeout=0.5
)

# Kafka producer for results
kafka_producer = KafkaProducer(
    bootstrap_servers='fraud-detection-kafka-kafka-bootstrap.streaming.svc.cluster.local:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    compression_type='lz4',
    linger_ms=5,
    batch_size=16384
)

# ============================================================================
# Inference
# ============================================================================

def get_features(user_id: str, transaction: PredictionRequest) -> np.ndarray:
    """Retrieve and construct feature vector"""

    start = time.time()

    # Get user features from Redis
    user_key = f"user:{user_id}"

    pipe = feature_store.pipeline()
    pipe.hget(user_key, "total_transactions")
    pipe.hget(user_key, "total_amount")
    pipe.hget(user_key, "avg_amount")
    pipe.hget(user_key, "window_transaction_count")
    pipe.hget(user_key, "window_transaction_avg")
    pipe.hget(user_key, "window_transaction_std")
    pipe.hget(user_key, "transaction_velocity")

    results = pipe.execute()

    # Convert to floats with defaults
    total_transactions = float(results[0] or 0)
    total_amount = float(results[1] or 0)
    avg_amount = float(results[2] or 0)
    window_count = float(results[3] or 0)
    window_avg = float(results[4] or 0)
    window_std = float(results[5] or 0)
    velocity = float(results[6] or 0)

    # Compute derived features
    amount_deviation = (transaction.amount - avg_amount) / (avg_amount + 1) if avg_amount > 0 else 0

    from datetime import datetime
    ts = datetime.fromisoformat(transaction.timestamp)
    hour_of_day = ts.hour
    day_of_week = ts.weekday()

    # Construct feature vector
    features = np.array([[
        total_transactions,
        total_amount,
        avg_amount,
        0,  # time_since_last_transaction (computed elsewhere)
        window_count,
        window_avg,
        window_std,
        velocity,
        amount_deviation,
        transaction.amount,
        hour_of_day,
        day_of_week,
        1 if transaction.is_international else 0
    ]], dtype=np.float32)

    feature_latency.observe(time.time() - start)

    return features

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """Make fraud prediction"""

    overall_start = time.time()

    try:
        # Get features
        features = get_features(request.user_id, request)

        # Select model for prediction
        model_version, model_session = model_manager.get_model(request.transaction_id)

        # Predict
        inference_start = time.time()

        outputs = model_session.run(None, {'float_input': features})
        probabilities = outputs[1][0]  # Probability outputs
        fraud_probability = float(probabilities[1])  # Probability of fraud class

        inference_time = time.time() - inference_start

        # Threshold for binary decision
        is_fraud = fraud_probability > 0.5

        total_latency = time.time() - overall_start
        latency_ms = total_latency * 1000

        # Metrics
        prediction_counter.labels(
            model_version=model_version,
            prediction='fraud' if is_fraud else 'legitimate'
        ).inc()

        prediction_latency.labels(model_version=model_version).observe(total_latency)

        # Async: publish result to Kafka
        background_tasks.add_task(
            publish_result,
            request,
            is_fraud,
            fraud_probability,
            model_version,
            latency_ms
        )

        response = PredictionResponse(
            transaction_id=request.transaction_id,
            is_fraud=is_fraud,
            fraud_probability=fraud_probability,
            model_version=model_version,
            latency_ms=latency_ms
        )

        # Warn on high latency
        if latency_ms > 50:
            print(f"⚠️  High latency: {latency_ms:.2f}ms for {request.transaction_id}")

        return response

    except Exception as e:
        model_errors.labels(error_type=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=str(e))

def publish_result(
    request: PredictionRequest,
    is_fraud: bool,
    probability: float,
    model_version: str,
    latency_ms: float
):
    """Publish prediction result to Kafka"""
    try:
        result = {
            'transaction_id': request.transaction_id,
            'user_id': request.user_id,
            'is_fraud': is_fraud,
            'fraud_probability': probability,
            'model_version': model_version,
            'latency_ms': latency_ms,
            'timestamp': time.time()
        }

        kafka_producer.send('predictions', value=result)
    except Exception as e:
        print(f"Error publishing result: {e}")

@app.get("/health")
async def health():
    """Health check"""
    try:
        feature_store.ping()
        return {"status": "healthy", "models": list(model_manager.models.keys())}
    except:
        raise HTTPException(status_code=503, detail="Feature store unavailable")

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from starlette.responses import Response

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,
        loop="uvloop",
        log_level="info"
    )
```

This capstone project is quite extensive. Let me continue with the remaining phases in a separate file.

Would you like me to:
1. Continue with Phase 5-8 (Load Testing, Monitoring, A/B Testing, Deployment)?
2. Or move on to Capstone 03: Multi-Cloud ML Infrastructure?