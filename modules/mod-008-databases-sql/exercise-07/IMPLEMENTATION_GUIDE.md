# Implementation Guide: NoSQL Databases for Machine Learning

## Overview

This step-by-step guide walks you through implementing a hybrid ML platform using PostgreSQL, MongoDB, and Redis.

**Estimated Time:** 3-4 hours
**Difficulty:** Intermediate

## Learning Path

1. **Setup** (30 min) - Get all databases running
2. **PostgreSQL** (45 min) - Complex queries and relationships
3. **MongoDB** (45 min) - Flexible document storage
4. **Redis** (45 min) - High-speed caching
5. **Hybrid Platform** (45 min) - Combine all three
6. **Benchmarks** (30 min) - Compare performance

---

## Phase 1: Environment Setup (30 minutes)

### Step 1.1: Start All Databases

```bash
# Clone/navigate to exercise directory
cd modules/mod-008-databases-sql/exercise-07

# Start all services
docker-compose up -d

# Verify all are healthy
docker-compose ps
```

**Expected Output:**
```
NAME            STATUS      PORTS
ml-postgres     Up (healthy)    0.0.0.0:5432->5432/tcp
ml-mongodb      Up (healthy)    0.0.0.0:27017->27017/tcp
ml-redis        Up (healthy)    0.0.0.0:6379->6379/tcp
```

### Step 1.2: Verify Database Connections

```bash
# PostgreSQL
docker-compose exec postgres psql -U mluser -d ml_platform -c "SELECT version();"

# MongoDB
docker-compose exec mongodb mongosh -u mluser -p mlpass123 --eval "db.version()"

# Redis
docker-compose exec redis redis-cli -a mlpass123 PING
```

### Step 1.3: Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installations
python -c "import sqlalchemy, pymongo, redis; print('✓ All libraries installed')"
```

---

## Phase 2: PostgreSQL - Structured Data (45 minutes)

### Step 2.1: Understand the Schema

**View the schema:**
```bash
docker-compose exec postgres psql -U mluser -d ml_platform -c "\\dt"
```

**Key tables:**
- `datasets` - ML training datasets
- `training_runs` - Model training metadata
- `model_artifacts` - Stored model files
- `deployments` - Production deployments

### Step 2.2: Run PostgreSQL Client

```bash
python src/postgres_client.py
```

**What this demonstrates:**
- ✅ Complex JOINs (training_runs + datasets)
- ✅ Aggregations (GROUP BY framework)
- ✅ Analytical queries (average accuracy)
- ✅ Connection pooling

### Step 2.3: Practice Complex Queries

**Query 1: Top models by accuracy**
```python
from src.postgres_client import get_top_models_by_accuracy

models = get_top_models_by_accuracy(top_n=5)
for model in models:
    print(f"{model['model_name']} - {model['accuracy']:.4f}")
```

**Query 2: Framework comparison**
```python
from src.postgres_client import get_framework_comparison

frameworks = get_framework_comparison()
for fw in frameworks:
    print(f"{fw['framework']}: {fw['avg_accuracy']:.4f} (n={fw['total_runs']})")
```

**Query 3: Dataset statistics**
```python
from src.postgres_client import get_dataset_training_stats

stats = get_dataset_training_stats("fraud-train-v1")
print(f"Total experiments: {stats['total_experiments']}")
print(f"Best accuracy: {stats['best_accuracy']:.4f}")
```

### Step 2.4: Key Takeaway - When to Use PostgreSQL

Use PostgreSQL when you need:
- **Complex queries** with JOINs across multiple tables
- **ACID transactions** for data consistency
- **Analytical queries** with aggregations
- **Referential integrity** with foreign keys

---

## Phase 3: MongoDB - Flexible Documents (45 minutes)

### Step 3.1: Understand Document Structure

MongoDB stores data as JSON-like documents. No fixed schema required!

**Example documents:**
- PyTorch model: Has `architecture.layers` array
- BERT model: Has `tokenizer` and `fine_tuning` config
- XGBoost model: Has `feature_engineering` section

All different structures, stored in the same collection!

### Step 3.2: Run MongoDB Client

```bash
python src/mongodb_client.py
```

**What this demonstrates:**
- ✅ Flexible schemas (different model types)
- ✅ Nested document queries (metrics.accuracy)
- ✅ Aggregation pipelines
- ✅ Schema evolution (add fields without migration)

### Step 3.3: Practice Document Operations

**Insert a model config:**
```python
from src.mongodb_client import insert_model_config

config = {
    "model_id": "my-custom-model",
    "model_name": "My Custom Model",
    "framework": "pytorch",
    "config": {
        "learning_rate": 0.001,
        "batch_size": 32,
        # Any structure you want!
    },
    "metrics": {
        "accuracy": 0.95
    }
}

insert_model_config(config)
```

**Query nested fields:**
```python
from src.mongodb_client import find_high_accuracy_models

# Query metrics.accuracy using dot notation
models = find_high_accuracy_models(min_accuracy=0.90)
for model in models:
    print(f"{model['model_name']}: {model['metrics']['accuracy']:.4f}")
```

**Update with new fields:**
```python
from src.mongodb_client import update_model_metrics

# Add new metrics without schema migration!
update_model_metrics("my-custom-model", {
    "inference_latency_ms": 12.3,
    "model_size_mb": 156.7,
    "gpu_memory_mb": 2048  # Completely new field
})
```

### Step 3.4: Key Takeaway - When to Use MongoDB

Use MongoDB when you need:
- **Flexible schemas** that change frequently
- **Nested/hierarchical data** (model configs, experiments)
- **Different document structures** in same collection
- **Horizontal scaling** across servers

---

## Phase 4: Redis - High-Speed Caching (45 minutes)

### Step 4.1: Understand Redis Data Structures

Redis provides multiple data structures:
- **Strings**: Simple key-value
- **Hashes**: Structured data (like a dictionary)
- **Sorted Sets**: Leaderboards, rankings
- **TTL**: Automatic expiration

### Step 4.2: Run Redis Client

```bash
python src/redis_client.py
```

**What this demonstrates:**
- ✅ Prediction caching with TTL
- ✅ Feature store with Redis Hashes
- ✅ Batch operations with Pipelines
- ✅ Leaderboards with Sorted Sets
- ✅ Rate limiting

### Step 4.3: Practice Redis Operations

**Cache a prediction:**
```python
from src.redis_client import cache_prediction, get_cached_prediction

# Cache for 5 minutes
cache_prediction(
    user_id="user_123",
    features=[0.5, 0.3, 0.8],
    prediction=0.92,
    ttl=300
)

# Retrieve from cache
result = get_cached_prediction("user_123")
print(f"Prediction: {result['prediction']}")  # 0.92
```

**Feature store:**
```python
from src.redis_client import store_user_features, get_user_features

# Store features with 1-hour TTL
store_user_features("user_001", {
    "age": 35.0,
    "account_balance": 5000.50,
    "num_transactions": 45.0
}, ttl=3600)

# Retrieve features (<1ms)
features = get_user_features("user_001")
print(features)  # {'age': 35.0, 'account_balance': 5000.50, ...}
```

**Leaderboard:**
```python
from src.redis_client import update_model_leaderboard, get_top_models

# Update leaderboard
update_model_leaderboard("fraud-detector-v1", 0.9845)
update_model_leaderboard("sentiment-bert", 0.9234)

# Get top 3 models
top_models = get_top_models(3)
for model in top_models:
    print(f"{model['rank']}. {model['model_name']}: {model['accuracy']:.4f}")
```

### Step 4.4: Key Takeaway - When to Use Redis

Use Redis when you need:
- **Sub-millisecond latency** (<1ms)
- **Caching** with automatic expiration (TTL)
- **High throughput** (100,000+ ops/sec)
- **Ephemeral data** (sessions, temporary results)

---

## Phase 5: Hybrid Platform (45 minutes)

### Step 5.1: Understand the Hybrid Architecture

The hybrid platform routes data to the optimal database:

```
User Request
     ↓
  ┌─────────────────┐
  │ Hybrid Platform │
  └─────────────────┘
     ↓         ↓        ↓
PostgreSQL  MongoDB   Redis
(Analytics) (Config)  (Cache)
```

### Step 5.2: Run Hybrid Platform

```bash
python src/hybrid_platform.py
```

**What this demonstrates:**
- ✅ Cross-database operations
- ✅ Multi-layer caching
- ✅ Unified model information

### Step 5.3: Practice Hybrid Operations

**Register a training run (PostgreSQL + MongoDB):**
```python
from src.hybrid_platform import MLPlatformHybrid

platform = MLPlatformHybrid()

ids = platform.register_training_run(
    dataset_name="fraud-train-v1",
    model_config={
        "model_name": "my-fraud-detector",
        "framework": "pytorch",
        "hyperparameters": {
            "learning_rate": 0.001,
            "batch_size": 64
        }
    }
)

print(f"PostgreSQL ID: {ids['postgres_id']}")
print(f"MongoDB ID: {ids['mongodb_id']}")
```

**Serve predictions with caching (Redis + compute):**
```python
# First call: Cache miss (50ms)
result1 = platform.serve_prediction_with_cache(
    user_id="user_999",
    features=[0.1, 0.2, 0.3]
)
print(f"Latency: {result1['latency_ms']:.2f}ms")  # ~50ms

# Second call: Cache hit (<1ms)
result2 = platform.serve_prediction_with_cache(
    user_id="user_999",
    features=[0.1, 0.2, 0.3]
)
print(f"Latency: {result2['latency_ms']:.2f}ms")  # ~0.5ms
```

**Get complete model info (all 3 databases):**
```python
info = platform.get_complete_model_info("my-fraud-detector")
print(f"PostgreSQL: {info['postgres']}")  # Training metadata
print(f"MongoDB: {info['mongodb']}")      # Model config
print(f"Redis: {info['redis']}")          # Prediction count
```

---

## Phase 6: Performance Benchmarks (30 minutes)

### Step 6.1: Run Benchmarks

```bash
python src/benchmark_comparison.py
```

**What this measures:**
- Simple read/write performance
- Complex query performance
- Batch operation performance

### Step 6.2: Analyze Results

**Expected results:**

| Operation | PostgreSQL | MongoDB | Redis | Winner |
|-----------|-----------|---------|-------|--------|
| Simple Read | 2-5ms | 1-3ms | 0.1-0.5ms | Redis (10-50x) |
| Simple Write | 3-8ms | 2-5ms | 0.2-1ms | Redis (5-15x) |
| Complex Query | 10-30ms | 20-50ms | N/A | PostgreSQL |
| Batch (100 items) | 20-50ms | 15-35ms | 5-10ms | Redis |

### Step 6.3: Key Insights

1. **Redis dominates simple operations** - 10-50x faster than SQL/NoSQL
2. **PostgreSQL best for complex queries** - JOINs and aggregations
3. **MongoDB middle ground** - Good flexibility + reasonable performance

---

## Phase 7: Testing and Validation (30 minutes)

### Step 7.1: Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Step 7.2: Verify Data Integrity

**PostgreSQL:**
```python
from src.postgres_client import get_session
from sqlalchemy import text

with get_session() as session:
    result = session.execute(text("SELECT COUNT(*) FROM training_runs"))
    count = result.fetchone()[0]
    print(f"Training runs: {count}")
```

**MongoDB:**
```python
from src.mongodb_client import db

count = db.model_configs.count_documents({})
print(f"Model configs: {count}")
```

**Redis:**
```python
from src.redis_client import redis_client

keys = redis_client.keys("*")
print(f"Redis keys: {len(keys)}")
```

---

## Common Issues and Solutions

### Issue 1: PostgreSQL Connection Timeout

**Symptoms:**
```
sqlalchemy.exc.OperationalError: could not connect to server
```

**Solution:**
```bash
# Check if container is running
docker-compose ps postgres

# Restart PostgreSQL
docker-compose restart postgres

# Check connection pool
python -c "from src.postgres_client import test_connection; test_connection()"
```

### Issue 2: MongoDB Authentication Failed

**Symptoms:**
```
pymongo.errors.OperationFailure: Authentication failed
```

**Solution:**
```bash
# Verify credentials
docker-compose exec mongodb mongosh -u mluser -p mlpass123

# Recreate container if needed
docker-compose down
docker-compose up -d mongodb
```

### Issue 3: Redis Out of Memory

**Symptoms:**
```
redis.exceptions.ResponseError: OOM command not allowed
```

**Solution:**
```bash
# Check memory usage
docker-compose exec redis redis-cli -a mlpass123 INFO memory

# Clear data
docker-compose exec redis redis-cli -a mlpass123 FLUSHALL

# Increase maxmemory in docker-compose.yml
```

---

## Best Practices Summary

### PostgreSQL

✅ **Do:**
- Use indexes on frequently queried columns
- Use connection pooling
- Use prepared statements for security
- Use views for complex queries

❌ **Don't:**
- Run heavy queries on production database
- Store large BLOBs in PostgreSQL
- Use VARCHAR(MAX) unnecessarily

### MongoDB

✅ **Do:**
- Create indexes on query fields
- Use projection to limit returned fields
- Use bulk operations for inserts
- Embed related data when appropriate

❌ **Don't:**
- Create too many indexes (slow writes)
- Store large files in documents
- Use MongoDB for complex JOINs

### Redis

✅ **Do:**
- Set appropriate TTLs
- Use pipelines for batch operations
- Use Redis Hashes for structured data
- Monitor memory usage

❌ **Don't:**
- Use Redis as primary database
- Store data without TTL
- Use Redis for complex queries

---

## Completion Checklist

- [ ] All three databases running and accessible
- [ ] PostgreSQL complex queries demonstrated
- [ ] MongoDB flexible schemas demonstrated
- [ ] Redis caching shows performance improvement
- [ ] Hybrid platform successfully combines all three
- [ ] Benchmarks show Redis >10x faster for reads
- [ ] All tests pass
- [ ] Understand when to use each database

---

## Next Steps

1. **Add Monitoring**: Prometheus + Grafana for all databases
2. **Implement CDC**: Stream PostgreSQL changes to MongoDB
3. **Add API Layer**: FastAPI endpoints for the platform
4. **Deploy to Cloud**: Managed database services
5. **Add Distributed Tracing**: OpenTelemetry integration

---

## Resources

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [MongoDB Manual](https://www.mongodb.com/docs/)
- [Redis Documentation](https://redis.io/documentation)
- [Database Selection Guide](https://www.prisma.io/dataguide/intro/comparing-database-types)

Congratulations! You've completed the NoSQL Databases for ML exercise.
