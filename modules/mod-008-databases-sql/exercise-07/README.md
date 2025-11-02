# NoSQL Databases for Machine Learning - Complete Solution

## Overview

This solution demonstrates a **hybrid ML platform** that leverages the strengths of three different database technologies:

- **PostgreSQL** - Structured data with complex queries and ACID transactions
- **MongoDB** - Flexible document storage for varying ML configurations
- **Redis** - High-speed caching and feature serving (<1ms latency)

**Key Learning:** No single database is optimal for all use cases. Modern ML platforms use polyglot persistence to match each data type with the most appropriate database.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ML Platform                               │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  PostgreSQL  │  │   MongoDB    │  │    Redis     │         │
│  │              │  │              │  │              │         │
│  │ • Datasets   │  │ • Model      │  │ • Predictions│         │
│  │ • Training   │  │   Configs    │  │ • Features   │         │
│  │   Runs       │  │ • Experiments│  │ • Sessions   │         │
│  │ • Deployments│  │ • Metadata   │  │ • Counters   │         │
│  │              │  │              │  │ • Leaderboard│         │
│  │ ACID + JOINs │  │ Flexible     │  │ <1ms Latency │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│         │                  │                  │                 │
│         └──────────────────┴──────────────────┘                 │
│                            │                                     │
│                   Hybrid Platform Layer                         │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Start All Databases

```bash
# Start PostgreSQL, MongoDB, and Redis
docker-compose up -d

# Verify all services are healthy
docker-compose ps

# Check logs
docker-compose logs -f
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Demonstrations

```bash
# Test PostgreSQL (complex queries, JOINs)
python src/postgres_client.py

# Test MongoDB (flexible schemas)
python src/mongodb_client.py

# Test Redis (caching, feature store)
python src/redis_client.py

# Test Hybrid Platform (all three together)
python src/hybrid_platform.py

# Run Performance Benchmarks
python src/benchmark_comparison.py
```

## Database Selection Matrix

### PostgreSQL: When to Use

✅ **Best For:**
- Training run metadata requiring JOINs with datasets
- Complex analytical queries (aggregations, window functions)
- Dataset lineage and provenance
- Deployment tracking with referential integrity
- Financial transactions requiring ACID guarantees

❌ **Avoid When:**
- Schema changes frequently
- Need <10ms latency for simple lookups
- Horizontal scaling across data centers required

**Example Use Cases:**
- "Which model has the highest accuracy on the fraud dataset?"
- "Compare average training time by framework"
- "Show me all deployments in production with accuracy > 0.95"

### MongoDB: When to Use

✅ **Best For:**
- Model configurations (PyTorch, TensorFlow, XGBoost all have different params)
- Experiment tracking with nested hyperparameters
- ML metadata that varies by framework
- Rapidly evolving schemas
- Hierarchical data structures

❌ **Avoid When:**
- Need strong ACID transactions across documents
- Complex JOINs are required frequently
- Data has clear relational structure

**Example Use Cases:**
- "Store a BERT model config with tokenizer, fine-tuning params, and deployment settings"
- "Find all models with accuracy >= 0.90"
- "Add new 'explainability' field to models without schema migration"

### Redis: When to Use

✅ **Best For:**
- Prediction result caching (5-minute TTL)
- Real-time feature serving (<10ms SLA)
- User session management
- Model accuracy leaderboards
- API rate limiting
- Prediction counters

❌ **Avoid When:**
- Need complex queries or aggregations
- Data must persist long-term
- Working with large datasets (Redis is in-memory)

**Example Use Cases:**
- "Cache the last prediction for user_123 for 5 minutes"
- "Get top 10 models by accuracy in real-time"
- "Serve user features with <5ms latency"
- "Limit users to 100 predictions per minute"

## Performance Characteristics

Based on benchmarks (`src/benchmark_comparison.py`):

| Operation | PostgreSQL | MongoDB | Redis | Winner |
|-----------|-----------|---------|-------|--------|
| Simple Read | ~2-5ms | ~1-3ms | ~0.1-0.5ms | **Redis** (5-50x faster) |
| Simple Write | ~3-8ms | ~2-5ms | ~0.2-1ms | **Redis** (3-15x faster) |
| Complex Query (JOIN + GROUP BY) | ~10-30ms | ~20-50ms | N/A | **PostgreSQL** |
| Aggregation | ~15-40ms | ~10-25ms | N/A | Tied (depends on query) |
| Batch Insert (100 rows) | ~20-50ms | ~15-35ms | ~5-10ms | **Redis Pipeline** |

**Key Takeaways:**
- Redis: 10-50x faster for simple key-value operations
- PostgreSQL: Best for complex analytical queries
- MongoDB: Good middle ground with flexible schemas

## File Structure

```
exercise-07/
├── docker-compose.yml           # Multi-database environment
├── requirements.txt             # Python dependencies
├── sql/
│   └── init.sql                 # PostgreSQL schema with sample data
├── src/
│   ├── postgres_client.py       # PostgreSQL operations
│   ├── mongodb_client.py        # MongoDB operations
│   ├── redis_client.py          # Redis operations
│   ├── hybrid_platform.py       # Unified ML platform
│   └── benchmark_comparison.py  # Performance benchmarks
├── tests/
│   ├── test_postgres.py
│   ├── test_mongodb.py
│   ├── test_redis.py
│   └── test_hybrid.py
├── docs/
│   └── DECISION_MATRIX.md       # Detailed selection criteria
├── README.md                     # This file
└── IMPLEMENTATION_GUIDE.md      # Step-by-step guide
```

## Key Features

### PostgreSQL Features Demonstrated

1. **Complex Queries with JOINs**
   ```python
   # Get top models by accuracy across all datasets
   get_top_models_by_accuracy(top_n=5)
   ```

2. **Aggregation and Analytics**
   ```python
   # Compare frameworks by average performance
   get_framework_comparison()
   ```

3. **Multi-table Relationships**
   ```python
   # Get deployment summary with model metrics
   get_deployment_summary()
   ```

4. **Views and Functions**
   - `model_performance` view: Consolidated model metrics
   - `get_top_models(n)` function: Top N models by accuracy

### MongoDB Features Demonstrated

1. **Flexible Schemas**
   ```python
   # PyTorch, TensorFlow, XGBoost - all different configs
   insert_model_config(pytorch_config)
   insert_model_config(transformer_config)
   insert_model_config(xgboost_config)
   ```

2. **Nested Document Queries**
   ```python
   # Query nested metrics.accuracy field
   find_high_accuracy_models(min_accuracy=0.90)
   ```

3. **Aggregation Pipeline**
   ```python
   # Group by framework and calculate averages
   aggregate_metrics_by_framework()
   ```

4. **Schema Evolution Without Migration**
   ```python
   # Add new fields on the fly
   update_model_metrics(model_id, {
       "inference_latency_ms": 15.3,
       "model_size_mb": 245.6
   })
   ```

### Redis Features Demonstrated

1. **Prediction Caching with TTL**
   ```python
   # Cache for 5 minutes
   cache_prediction(user_id, features, prediction, ttl=300)
   prediction = get_cached_prediction(user_id)
   ```

2. **Feature Store (Redis Hashes)**
   ```python
   # Store structured features
   store_user_features(user_id, {
       "age": 35.0,
       "account_balance": 5000.50,
       "num_transactions": 45.0
   })
   ```

3. **Batch Operations (Pipelines)**
   ```python
   # Retrieve features for 1000 users in <10ms
   batch_get_features(user_ids)
   ```

4. **Leaderboards (Sorted Sets)**
   ```python
   # Real-time model rankings
   update_model_leaderboard(model_name, accuracy)
   top_models = get_top_models(top_n=10)
   ```

5. **Rate Limiting**
   ```python
   # Limit to 100 requests per 60 seconds
   check_rate_limit(user_id, max_requests=100, window_seconds=60)
   ```

### Hybrid Platform Features

1. **Cross-Database Registration**
   ```python
   # Register training run in both PostgreSQL and MongoDB
   platform.register_training_run(dataset_name, model_config)
   ```

2. **Multi-Layer Caching**
   ```python
   # Check Redis → Compute → Cache
   platform.serve_prediction_with_cache(user_id, features)
   ```

3. **Unified Model Information**
   ```python
   # Fetch from all three databases
   info = platform.get_complete_model_info(model_id)
   ```

## Real-World Use Cases

### Use Case 1: Real-Time Fraud Detection API

**Requirements:**
- <10ms prediction latency
- Model metadata with varying schemas
- Analytics on model performance
- 10,000 predictions/second

**Solution:**
```python
# PostgreSQL: Training metadata and analytics
training_run_id = create_training_run(
    dataset_id=1,
    model_name="fraud-detector-v1",
    framework="pytorch"
)

# MongoDB: Flexible model configuration
insert_model_config({
    "model_id": "fraud-detector-v1",
    "architecture": {...},  # Complex nested config
    "hyperparameters": {...}
})

# Redis: Feature serving + prediction caching
store_user_features(user_id, features)  # <1ms
prediction = serve_prediction_with_cache(user_id, features)  # <5ms with cache
```

**Performance:**
- First prediction: 50ms (compute + cache)
- Cached predictions: 0.5ms (Redis lookup)
- 20,000+ predictions/second

### Use Case 2: Model Experiment Tracking

**Requirements:**
- Track 1000s of experiments
- Different frameworks have different parameters
- Query best experiments by metric
- Generate training reports

**Solution:**
```python
# MongoDB: Store flexible experiment configs
insert_experiment(
    model_id="recommender-v1",
    hyperparameters={
        "embedding_dim": 128,
        "num_layers": 3,
        # ... varies by model
    },
    metrics={"accuracy": 0.92, "f1": 0.89}
)

# PostgreSQL: Analytics and reporting
get_framework_comparison()  # Aggregate across all experiments
get_top_models_by_accuracy()  # Best performing models
```

### Use Case 3: Online Feature Store

**Requirements:**
- <5ms feature retrieval
- 1M+ users
- Features updated frequently
- 1-hour TTL

**Solution:**
```python
# Redis: High-speed feature storage
store_user_features(user_id, {
    "account_age_days": 365,
    "total_transactions": 1234,
    "avg_transaction_amount": 567.89,
    # ... 50+ features
}, ttl=3600)

# Batch retrieval for 1000 users in <10ms
features = batch_get_features(user_ids)
```

## Optimization Tips

### PostgreSQL

1. **Use Indexes**
   ```sql
   CREATE INDEX idx_training_runs_accuracy ON training_runs(accuracy DESC)
   WHERE status = 'completed';
   ```

2. **Use Views for Common Queries**
   ```sql
   CREATE VIEW model_performance AS
   SELECT tr.*, d.name as dataset_name
   FROM training_runs tr JOIN datasets d ON tr.dataset_id = d.id;
   ```

3. **Connection Pooling**
   - Pool size: 10
   - Max overflow: 20
   - Pre-ping enabled

### MongoDB

1. **Create Indexes**
   ```python
   db.model_configs.create_index("model_id", unique=True)
   db.model_configs.create_index([("metrics.accuracy", DESCENDING)])
   ```

2. **Use Projection**
   ```python
   # Only fetch needed fields
   db.model_configs.find({"framework": "pytorch"}, {"_id": 0, "metrics": 1})
   ```

3. **Bulk Operations**
   ```python
   db.model_configs.insert_many(configs)  # Faster than individual inserts
   ```

### Redis

1. **Use Pipelines for Batch Operations**
   ```python
   pipe = redis_client.pipeline()
   for user_id in user_ids:
       pipe.hgetall(f"features:user:{user_id}")
   results = pipe.execute()
   ```

2. **Set Appropriate TTLs**
   - Predictions: 5 minutes
   - Features: 1 hour
   - Sessions: 30 minutes

3. **Use Redis Hashes for Structured Data**
   ```python
   # More memory-efficient than separate keys
   redis_client.hset("features:user:123", mapping=features)
   ```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific database tests
pytest tests/test_postgres.py -v
pytest tests/test_mongodb.py -v
pytest tests/test_redis.py -v

# Run integration tests
pytest tests/test_hybrid.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Monitoring

### PostgreSQL

```bash
# Connection pool status
python -c "from src.postgres_client import get_pool_status; print(get_pool_status())"

# Database size
SELECT pg_size_pretty(pg_database_size('ml_platform'));
```

### MongoDB

```python
# Database stats
from src.mongodb_client import client
stats = client.command("dbStats")
print(f"Collections: {stats['collections']}")
print(f"Data Size: {stats['dataSize'] / 1048576:.2f} MB")
```

### Redis

```python
# Cache stats
from src.redis_client import get_cache_stats, get_memory_info
print(get_cache_stats())
print(get_memory_info())
```

## Troubleshooting

### Issue: PostgreSQL Connection Refused

```bash
# Check if container is running
docker-compose ps postgres

# Check logs
docker-compose logs postgres

# Restart PostgreSQL
docker-compose restart postgres
```

### Issue: MongoDB Authentication Failed

```bash
# Verify credentials in docker-compose.yml
docker-compose exec mongodb mongosh -u mluser -p mlpass123

# Check if database exists
db.adminCommand({listDatabases: 1})
```

### Issue: Redis Out of Memory

```bash
# Check memory usage
docker-compose exec redis redis-cli -a mlpass123 INFO memory

# Increase maxmemory in docker-compose.yml
command: redis-server --maxmemory 1gb
```

## Next Steps

1. **Add Distributed Tracing**: Integrate Jaeger/OpenTelemetry
2. **Implement Change Data Capture**: Stream PostgreSQL changes to MongoDB
3. **Add Redis Cluster**: High availability setup
4. **Implement Data Migration**: Tools for moving data between databases
5. **Add API Layer**: FastAPI endpoints for the hybrid platform

## Resources

- [PostgreSQL Performance Tips](https://www.postgresql.org/docs/current/performance-tips.html)
- [MongoDB Schema Design](https://www.mongodb.com/docs/manual/core/data-model-design/)
- [Redis Best Practices](https://redis.io/docs/manual/patterns/)
- [Polyglot Persistence](https://martinfowler.com/bliki/PolyglotPersistence.html)

## License

MIT License - See LICENSE file for details
