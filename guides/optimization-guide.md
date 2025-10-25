# Optimization Guide for Junior AI Infrastructure Engineers

A comprehensive guide to optimizing Docker images, Python code, databases, APIs, and cloud costs for AI/ML workloads.

---

## Table of Contents

1. [Introduction to Optimization](#introduction-to-optimization)
2. [Docker Image Optimization](#docker-image-optimization)
3. [Python Code Optimization](#python-code-optimization)
4. [Database Query Optimization](#database-query-optimization)
5. [API Performance Optimization](#api-performance-optimization)
6. [Cost Optimization](#cost-optimization)
7. [Monitoring for Optimization](#monitoring-for-optimization)

---

## Introduction to Optimization

### The Optimization Philosophy

**Important:** Premature optimization is the root of all evil. Follow this order:

1. **Make it work** - Get functionality correct first
2. **Make it right** - Clean, maintainable code
3. **Make it fast** - Optimize only when needed
4. **Make it cheap** - Reduce costs where possible

### Measure Before Optimizing

Always measure before and after optimization:

```bash
# Before optimization
time python train.py  # 5 minutes
docker images my-app  # 1.2 GB

# After optimization
time python train.py  # 2 minutes (60% faster!)
docker images my-app  # 350 MB (71% smaller!)
```

### The 80/20 Rule

- 80% of resources are used by 20% of the code
- Find that 20% and optimize it
- Don't waste time optimizing code that runs once

---

## Docker Image Optimization

### 1. Multi-Stage Builds

**Problem:** Including build tools in final image wastes space

**Before (1.2 GB):**
```dockerfile
FROM python:3.9

WORKDIR /app

# Install build dependencies (takes up space!)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

**After (350 MB - 71% smaller!):**
```dockerfile
# Stage 1: Build
FROM python:3.9 AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make

# Install Python packages to /install directory
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim

WORKDIR /app

# Copy only the installed packages, not build tools!
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

CMD ["python", "app.py"]
```

**Why this works:**
- Build tools (gcc, make, etc.) only in first stage
- Only compiled packages copied to final stage
- Final image much smaller

### 2. Choose Slim Base Images

**Image Size Comparison:**

| Base Image | Size | Use When |
|------------|------|----------|
| `python:3.9` | 915 MB | Need full development tools |
| `python:3.9-slim` | 122 MB | **Recommended for most apps** |
| `python:3.9-alpine` | 45 MB | Need absolute minimum (may have compatibility issues) |

```dockerfile
# ❌ Too large for production
FROM python:3.9

# ✅ Good for most use cases
FROM python:3.9-slim

# ⚠️ Smallest but may have issues (musl vs glibc)
FROM python:3.9-alpine
```

**Example: FastAPI + ML Model**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*  # Clean up apt cache!

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
```

### 3. Layer Caching Optimization

**How Docker Caching Works:**
- Each Dockerfile instruction creates a layer
- Layers are cached
- If instruction changes, all subsequent layers rebuild
- Order matters!

**❌ Bad Order (rebuilds dependencies every time):**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# This changes frequently!
COPY . .

# So this rebuilds every time, even if requirements.txt unchanged
RUN pip install -r requirements.txt

CMD ["python", "app.py"]
```

**✅ Good Order (caches dependencies):**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy only requirements first
COPY requirements.txt .

# Install dependencies (cached if requirements.txt unchanged)
RUN pip install --no-cache-dir -r requirements.txt

# Copy code last (changes frequently, but doesn't affect cached layers above)
COPY . .

CMD ["python", "app.py"]
```

**Build Time Comparison:**
```bash
# First build
$ docker build -t my-app .
# [+] Building 45.2s

# Change code, rebuild (with optimized order)
$ docker build -t my-app .
# [+] Building 2.1s (pip install cached!)

# Change code, rebuild (with bad order)
$ docker build -t my-app .
# [+] Building 43.8s (pip install runs again)
```

### 4. Use .dockerignore

**Problem:** COPY . . includes unnecessary files

**Create .dockerignore:**
```
# .dockerignore
# Version control
.git
.gitignore

# Python
__pycache__
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info
dist
build

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/

# Documentation
*.md
docs/

# Data files (large!)
data/
*.csv
*.parquet
*.pkl
*.h5
models/*.pth  # Don't include large model files

# OS
.DS_Store
Thumbs.db
```

**Impact:**
```bash
# Without .dockerignore
docker build -t my-app .
# Sending build context to Docker daemon  1.2GB

# With .dockerignore
docker build -t my-app .
# Sending build context to Docker daemon  15.3MB
```

### 5. Combine RUN Commands

**❌ Multiple RUN commands (creates more layers):**
```dockerfile
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y wget
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*
```

**✅ Combined RUN (fewer layers, smaller image):**
```dockerfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
```

**Size difference:** ~50MB saved!

### 6. Optimize Python Package Installation

**Techniques:**

```dockerfile
# 1. Use --no-cache-dir (don't keep pip cache in image)
RUN pip install --no-cache-dir -r requirements.txt

# 2. Install only production dependencies
# requirements.txt
tensorflow==2.13.0
fastapi==0.100.0
uvicorn==0.23.0

# requirements-dev.txt (don't install in production!)
pytest==7.4.0
black==23.7.0
mypy==1.4.0

# Dockerfile
RUN pip install --no-cache-dir -r requirements.txt
# Don't install requirements-dev.txt in production image!

# 3. Use pre-built wheels when available
RUN pip install --no-cache-dir --only-binary=:all: numpy pandas scikit-learn

# 4. Remove unnecessary dependencies
# Don't need jupyter in production!
# ❌ jupyter, matplotlib (only for development)
# ✅ Only packages needed at runtime
```

### 7. Complete Optimization Example

**Before: 1.5 GB image**
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

**After: 280 MB image (81% smaller!)**
```dockerfile
# Build stage
FROM python:3.9-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Runtime stage
FROM python:3.9-slim

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy only what's needed
COPY --from=builder /install /usr/local
COPY app.py .
COPY models/ models/

# Use non-root user
USER appuser

CMD ["python", "app.py"]
```

**.dockerignore**
```
.git
__pycache__
*.pyc
venv/
.pytest_cache/
data/
notebooks/
*.md
.DS_Store
```

### 8. Optimization Checklist

```
✅ Use slim base image (python:3.9-slim)
✅ Multi-stage build (separate build and runtime)
✅ Optimize layer caching (COPY requirements.txt before code)
✅ Use .dockerignore (exclude unnecessary files)
✅ Combine RUN commands (fewer layers)
✅ Use --no-cache-dir for pip
✅ Install only production dependencies
✅ Clean up apt cache (rm -rf /var/lib/apt/lists/*)
✅ Run as non-root user
✅ Use specific version tags (not :latest)
```

---

## Python Code Optimization

### 1. Profiling - Find What to Optimize

**Rule:** Always profile before optimizing!

#### Using cProfile
```python
# profile_example.py
import cProfile
import pstats

def slow_function():
    total = 0
    for i in range(1000000):
        total += i
    return total

def fast_function():
    return sum(range(1000000))

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    # Code to profile
    slow_function()
    fast_function()

    profiler.disable()

    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions
```

**Output shows:**
```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.052    0.052    0.052    0.052 profile_example.py:4(slow_function)
        1    0.008    0.008    0.008    0.008 profile_example.py:11(fast_function)
```

**Insight:** slow_function takes 6.5x longer!

#### Using line_profiler (line-by-line)
```bash
# Install
pip install line_profiler

# Add @profile decorator (don't import, it's added by kernprof)
# train.py
@profile
def train_model(X, y):
    # preprocessing
    X_scaled = scale(X)

    # training
    model = RandomForestClassifier()
    model.fit(X_scaled, y)

    return model

# Run profiler
kernprof -l -v train.py
```

**Output:**
```
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     5         1      1000.0   1000.0      0.1      X_scaled = scale(X)
     8         1    998000.0 998000.0     99.9      model.fit(X_scaled, y)
```

**Insight:** 99.9% of time is model training (can't optimize much), scaling is fast.

### 2. Common Python Optimizations

#### List Comprehensions vs Loops
```python
# ❌ Slow (0.08 seconds)
result = []
for i in range(1000000):
    result.append(i * 2)

# ✅ Fast (0.05 seconds - 37% faster!)
result = [i * 2 for i in range(1000000)]

# ✅ Even faster for large data (generator - constant memory)
result = (i * 2 for i in range(1000000))  # Only compute when needed
```

#### Use Built-in Functions
```python
# ❌ Slow
total = 0
for num in numbers:
    total += num

# ✅ Fast (10x faster!)
total = sum(numbers)

# ❌ Slow
maximum = numbers[0]
for num in numbers:
    if num > maximum:
        maximum = num

# ✅ Fast
maximum = max(numbers)
```

#### Avoid Repeated Computations
```python
# ❌ Bad - computes len() every iteration
for i in range(len(data)):
    if i < len(data) - 1:  # len() computed twice per iteration!
        process(data[i])

# ✅ Good - compute once
data_len = len(data)
for i in range(data_len):
    if i < data_len - 1:
        process(data[i])

# ✅ Better - use enumerate
for i, item in enumerate(data):
    if i < len(data) - 1:  # len() only computed once
        process(item)
```

#### Use Sets for Membership Testing
```python
# ❌ Slow - O(n) for each lookup
allowed_ids = [1, 2, 3, 4, 5, ...]  # List of 10,000 IDs

for item in items:  # 100,000 items
    if item.id in allowed_ids:  # Scans entire list each time!
        process(item)
# Total: 100,000 * 10,000 = 1 billion operations

# ✅ Fast - O(1) for each lookup
allowed_ids = {1, 2, 3, 4, 5, ...}  # Set of 10,000 IDs

for item in items:  # 100,000 items
    if item.id in allowed_ids:  # Instant lookup!
        process(item)
# Total: ~100,000 operations (10,000x faster!)
```

**Benchmark:**
```python
import time

# List
allowed_list = list(range(10000))
start = time.time()
for i in range(100000):
    _ = 5000 in allowed_list
print(f"List: {time.time() - start:.2f}s")  # 2.5 seconds

# Set
allowed_set = set(range(10000))
start = time.time()
for i in range(100000):
    _ = 5000 in allowed_set
print(f"Set: {time.time() - start:.2f}s")   # 0.005 seconds!
```

### 3. NumPy Optimization

```python
import numpy as np

# ❌ Slow - Python loops
data = list(range(1000000))
result = []
for x in data:
    result.append(x * 2 + 1)
# Time: 0.15 seconds

# ✅ Fast - NumPy vectorization
data = np.arange(1000000)
result = data * 2 + 1
# Time: 0.002 seconds (75x faster!)

# ❌ Slow - Python loop for calculations
total = 0
for x in data:
    total += x ** 2
# Time: 0.20 seconds

# ✅ Fast - NumPy
total = np.sum(data ** 2)
# Time: 0.003 seconds (67x faster!)
```

### 4. Pandas Optimization

#### Use Vectorized Operations
```python
import pandas as pd

df = pd.DataFrame({'value': range(1000000)})

# ❌ Slow - iterate rows (NEVER do this!)
result = []
for idx, row in df.iterrows():
    result.append(row['value'] * 2)
df['result'] = result
# Time: 45 seconds

# ✅ Fast - vectorized operation
df['result'] = df['value'] * 2
# Time: 0.015 seconds (3000x faster!)

# ❌ Slow - apply with lambda
df['result'] = df['value'].apply(lambda x: x * 2)
# Time: 0.20 seconds

# ✅ Fast - vectorized
df['result'] = df['value'] * 2
# Time: 0.015 seconds (13x faster!)
```

#### Optimize Data Types
```python
# ❌ Default dtypes waste memory
df = pd.read_csv('data.csv')
print(df.memory_usage(deep=True).sum())  # 800 MB

# ✅ Optimize dtypes
df = pd.read_csv('data.csv', dtype={
    'id': 'int32',           # Was int64
    'count': 'int16',        # Was int64 (but values < 32767)
    'category': 'category',  # Was object (string)
    'flag': 'bool',          # Was object
})
print(df.memory_usage(deep=True).sum())  # 150 MB (81% less!)
```

### 5. Caching Results

```python
from functools import lru_cache
import time

# ❌ Slow - recomputes every time
def expensive_computation(n):
    time.sleep(2)  # Simulates expensive operation
    return n ** 2

expensive_computation(10)  # 2 seconds
expensive_computation(10)  # 2 seconds again!
expensive_computation(10)  # 2 seconds again!

# ✅ Fast - cache results
@lru_cache(maxsize=128)
def expensive_computation(n):
    time.sleep(2)
    return n ** 2

expensive_computation(10)  # 2 seconds (computed)
expensive_computation(10)  # Instant! (cached)
expensive_computation(10)  # Instant! (cached)
```

**Real-world example: Model loading**
```python
from functools import lru_cache
import joblib

# ❌ Slow - loads model every prediction
def predict(features):
    model = joblib.load('model.pkl')  # Loads from disk every time!
    return model.predict([features])

# ✅ Fast - load once, cache
@lru_cache(maxsize=1)
def load_model():
    return joblib.load('model.pkl')

def predict(features):
    model = load_model()  # Loads once, then cached
    return model.predict([features])
```

### 6. Lazy Loading

```python
# ❌ Load everything upfront (slow startup)
class DataProcessor:
    def __init__(self):
        self.model = load_large_model()      # 5 seconds
        self.embeddings = load_embeddings()   # 3 seconds
        self.vocab = load_vocabulary()        # 2 seconds
        # Startup: 10 seconds

# ✅ Load on demand (fast startup)
class DataProcessor:
    def __init__(self):
        self._model = None
        self._embeddings = None
        self._vocab = None
        # Startup: instant

    @property
    def model(self):
        if self._model is None:
            self._model = load_large_model()
        return self._model

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = load_embeddings()
        return self._embeddings

# Only loads what's actually used
processor = DataProcessor()  # Instant
result = processor.model.predict(X)  # Loads model now (when needed)
```

---

## Database Query Optimization

### 1. Understanding Query Performance

#### Use EXPLAIN to Analyze Queries
```sql
-- Check query execution plan
EXPLAIN ANALYZE
SELECT u.name, COUNT(p.id)
FROM users u
JOIN posts p ON u.id = p.user_id
WHERE u.created_at > '2024-01-01'
GROUP BY u.name;

-- Output shows:
-- Seq Scan on users  (cost=0.00..100.00 rows=1000 width=32) (actual time=0.123..45.678 rows=1000 loops=1)
--   Filter: (created_at > '2024-01-01'::date)
-- Seq Scan on posts   (cost=0.00..500.00 rows=10000 width=16) (actual time=0.234..123.456 rows=10000 loops=1)
```

**Key things to look for:**
- **Seq Scan** = scanning entire table (slow if table is large)
- **Index Scan** = using index (fast)
- **actual time** = real execution time
- **rows** = number of rows processed

### 2. Add Indexes

**Problem: Slow query without index**
```sql
-- Slow query (scans all 1 million rows)
SELECT * FROM users WHERE email = 'user@example.com';
-- Time: 450ms

-- Add index
CREATE INDEX idx_users_email ON users(email);

-- Now fast! (uses index)
SELECT * FROM users WHERE email = 'user@example.com';
-- Time: 2ms (225x faster!)
```

**When to add indexes:**
```sql
-- ✅ Index on foreign keys
CREATE INDEX idx_posts_user_id ON posts(user_id);

-- ✅ Index on frequently queried columns
CREATE INDEX idx_users_created_at ON users(created_at);

-- ✅ Composite index for multi-column queries
CREATE INDEX idx_posts_user_status ON posts(user_id, status);
-- Optimizes: WHERE user_id = X AND status = Y

-- ❌ Don't over-index (slows down writes)
-- Don't index every column!
```

**Check existing indexes:**
```sql
-- PostgreSQL
SELECT tablename, indexname, indexdef
FROM pg_indexes
WHERE tablename = 'users';

-- MySQL
SHOW INDEXES FROM users;
```

### 3. Avoid N+1 Queries

**❌ N+1 Problem (100 queries for 100 users!)**
```python
# Query 1: Get all users
users = User.query.all()  # 1 query

# Queries 2-101: Get posts for each user
for user in users:  # 100 queries!
    posts = user.posts  # Lazy load - queries DB each time!
    print(f"{user.name}: {len(posts)} posts")

# Total: 101 queries for 100 users
```

**✅ Solution: Eager Loading (2 queries total)**
```python
# Using SQLAlchemy
from sqlalchemy.orm import joinedload

# Single query with JOIN
users = User.query.options(
    joinedload(User.posts)
).all()  # 1 query with JOIN

for user in users:
    posts = user.posts  # Already loaded, no query!
    print(f"{user.name}: {len(posts)} posts")

# Total: 1 query for 100 users (100x fewer queries!)
```

**Django ORM:**
```python
# ❌ N+1 problem
users = User.objects.all()
for user in users:
    posts = user.posts.all()  # Queries DB each time

# ✅ Prefetch
users = User.objects.prefetch_related('posts').all()
for user in users:
    posts = user.posts.all()  # Already loaded
```

### 4. Limit Data Retrieved

```python
# ❌ Load all columns (wastes bandwidth)
SELECT * FROM posts;  # Retrieves all 20 columns

# ✅ Load only needed columns
SELECT id, title, created_at FROM posts;  # Only 3 columns

# ❌ Load all rows (could be millions!)
SELECT * FROM logs;

# ✅ Use LIMIT
SELECT * FROM logs ORDER BY created_at DESC LIMIT 100;

# ✅ Use pagination
SELECT * FROM logs
ORDER BY created_at DESC
LIMIT 20 OFFSET 0;  # Page 1

SELECT * FROM logs
ORDER BY created_at DESC
LIMIT 20 OFFSET 20;  # Page 2
```

### 5. Use Database Connection Pooling

**❌ Without pooling (slow)**
```python
# Creates new connection for each request
def get_data():
    conn = psycopg2.connect(
        host="localhost",
        database="mydb"
    )  # Expensive! Takes 50-100ms
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    data = cursor.fetchall()
    conn.close()
    return data

# Each API request: 50-100ms just to connect
```

**✅ With pooling (fast)**
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# Create connection pool once at startup
engine = create_engine(
    'postgresql://localhost/mydb',
    poolclass=QueuePool,
    pool_size=10,        # Keep 10 connections open
    max_overflow=20,     # Allow up to 20 more if needed
    pool_pre_ping=True,  # Check connection is alive
)

def get_data():
    with engine.connect() as conn:  # Reuses existing connection
        result = conn.execute("SELECT * FROM users")
        return result.fetchall()

# Each API request: < 1ms to get connection (100x faster!)
```

### 6. Cache Query Results

```python
from functools import lru_cache
import time

# ❌ Query database every time
def get_user_stats():
    # Expensive query: joins, aggregations
    return db.execute("""
        SELECT u.id, u.name, COUNT(p.id) as post_count
        FROM users u
        LEFT JOIN posts p ON u.id = p.user_id
        GROUP BY u.id, u.name
    """).fetchall()
    # Takes 500ms

# ✅ Cache results for 5 minutes
@lru_cache(maxsize=1)
def get_user_stats():
    return db.execute("...").fetchall()

# Or use Redis for distributed caching
import redis
import json

redis_client = redis.Redis()

def get_user_stats():
    # Check cache first
    cached = redis_client.get('user_stats')
    if cached:
        return json.loads(cached)

    # Not in cache, query database
    data = db.execute("...").fetchall()

    # Cache for 5 minutes
    redis_client.setex('user_stats', 300, json.dumps(data))

    return data
```

---

## API Performance Optimization

### 1. Add Caching Headers

```python
from fastapi import FastAPI, Response

app = FastAPI()

@app.get("/api/data")
def get_data(response: Response):
    # Data doesn't change often
    data = {"results": [...]}

    # Cache for 1 hour (3600 seconds)
    response.headers["Cache-Control"] = "public, max-age=3600"

    return data

# Client can reuse cached response for 1 hour
# Reduces server load by ~90% for this endpoint
```

### 2. Implement Pagination

```python
from fastapi import FastAPI, Query

app = FastAPI()

# ❌ Bad - returns all data
@app.get("/api/posts")
def get_posts():
    posts = db.query("SELECT * FROM posts")  # Could be millions!
    return {"posts": posts}

# ✅ Good - paginated
@app.get("/api/posts")
def get_posts(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100)
):
    offset = (page - 1) * limit
    posts = db.query(
        "SELECT * FROM posts ORDER BY created_at DESC LIMIT ? OFFSET ?",
        limit, offset
    )
    total = db.query("SELECT COUNT(*) FROM posts")[0][0]

    return {
        "posts": posts,
        "page": page,
        "limit": limit,
        "total": total,
        "pages": (total + limit - 1) // limit
    }
```

### 3. Use Async for I/O-Bound Operations

```python
# ❌ Synchronous - blocks on I/O
from fastapi import FastAPI
import requests

app = FastAPI()

@app.get("/api/aggregate")
def aggregate_data():
    # These run sequentially (total: 3 seconds)
    data1 = requests.get('http://api1.com/data').json()  # 1 second
    data2 = requests.get('http://api2.com/data').json()  # 1 second
    data3 = requests.get('http://api3.com/data').json()  # 1 second

    return {"combined": [data1, data2, data3]}

# ✅ Async - runs in parallel
import asyncio
import httpx

@app.get("/api/aggregate")
async def aggregate_data():
    async with httpx.AsyncClient() as client:
        # These run in parallel! (total: 1 second)
        results = await asyncio.gather(
            client.get('http://api1.com/data'),
            client.get('http://api2.com/data'),
            client.get('http://api3.com/data'),
        )

    data = [r.json() for r in results]
    return {"combined": data}

# 3x faster for I/O-bound operations!
```

### 4. Compress Responses

```python
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI()

# Add gzip compression
app.add_middleware(GZIPMiddleware, minimum_size=1000)

@app.get("/api/data")
def get_data():
    # Large response (100KB uncompressed)
    return {"results": [...]}

# Response size: 100KB → 15KB (85% smaller!)
# Faster transfer, especially on slow connections
```

### 5. Add Response Caching with Redis

```python
from fastapi import FastAPI
import redis
import json
from functools import wraps

app = FastAPI()
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_response(expire_seconds=300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

            # Check cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)

            # Not cached, call function
            result = await func(*args, **kwargs)

            # Cache result
            redis_client.setex(cache_key, expire_seconds, json.dumps(result))

            return result
        return wrapper
    return decorator

@app.get("/api/expensive-computation")
@cache_response(expire_seconds=3600)  # Cache for 1 hour
async def expensive_computation(param: str):
    # Expensive operation
    result = perform_ml_inference(param)  # Takes 5 seconds
    return {"result": result}

# First request: 5 seconds
# Subsequent requests: < 10ms (500x faster!)
```

### 6. Rate Limiting

```python
from fastapi import FastAPI, Request, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Limit to 10 requests per minute
@app.get("/api/predict")
@limiter.limit("10/minute")
async def predict(request: Request, data: dict):
    result = model.predict(data)
    return {"prediction": result}

# Prevents abuse, protects resources
```

---

## Cost Optimization

### 1. Right-Sizing Resources

**Check resource usage:**
```bash
# Kubernetes
kubectl top pods
kubectl top nodes

# See if pods are using their limits
kubectl describe pod my-pod | grep -A 5 Limits

# If actual usage much lower than limits, reduce them
```

**Example:**
```yaml
# ❌ Over-provisioned
spec:
  containers:
  - name: api
    resources:
      requests:
        memory: "4Gi"   # Actually uses 500Mi
        cpu: "2000m"    # Actually uses 200m
      limits:
        memory: "8Gi"
        cpu: "4000m"
# Cost: $200/month

# ✅ Right-sized
spec:
  containers:
  - name: api
    resources:
      requests:
        memory: "1Gi"   # 20% buffer
        cpu: "250m"
      limits:
        memory: "2Gi"
        cpu: "500m"
# Cost: $50/month (75% savings!)
```

### 2. Use Spot/Preemptible Instances

**AWS EC2 Spot Instances:**
```bash
# Regular on-demand: $0.50/hour
# Spot instance: $0.15/hour (70% discount!)

# Great for:
# - ML training jobs (can handle interruption)
# - Batch processing
# - Non-critical workloads

# Kubernetes: Use node groups with spot instances
# For training pods, use node affinity to spot nodes
```

```yaml
# deployment.yaml for training job
apiVersion: batch/v1
kind: Job
metadata:
  name: train-model
spec:
  template:
    spec:
      nodeSelector:
        node-type: spot  # Use spot instances
      tolerations:
      - key: "spot"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
      containers:
      - name: trainer
        image: my-trainer:v1
```

### 3. Auto-scaling

**Horizontal Pod Autoscaler (HPA):**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 2    # Low traffic
  maxReplicas: 10   # High traffic
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70  # Scale up if CPU > 70%
```

**Benefits:**
- Scale down during low traffic (nights, weekends)
- Scale up during high traffic
- Average cost savings: 40-60%

### 4. Clean Up Unused Resources

```bash
# Find unused Docker images
docker images --filter "dangling=true"
docker image prune -a  # Remove unused images

# Find unused Kubernetes resources
kubectl get pods --all-namespaces | grep -E "Completed|Error"
kubectl delete pod <pod-name>  # Clean up completed pods

# Find idle load balancers (AWS)
aws elb describe-load-balancers --query 'LoadBalancerDescriptions[?Instances==`[]`]'
# Each idle LB costs ~$20/month

# Find unattached EBS volumes (AWS)
aws ec2 describe-volumes --filters "Name=status,Values=available"
# Each unattached volume wastes money
```

### 5. Use Reserved Instances for Stable Workloads

**AWS Reserved Instances:**
```
On-Demand: $0.50/hour = $4,380/year
1-Year Reserved: $0.30/hour = $2,628/year (40% off)
3-Year Reserved: $0.20/hour = $1,752/year (60% off)

For always-running production workloads: 40-60% savings
```

### 6. Optimize Data Transfer Costs

```bash
# Data transfer costs (AWS example):
# Within same AZ: Free
# Between AZs: $0.01/GB
# To internet: $0.09/GB

# ❌ Expensive: Downloading large model files from internet
# $0.09/GB × 5GB × 1000 requests = $450/month

# ✅ Cheap: Cache model in same AZ
# Store in S3, download once, cache locally
# $0.09/GB × 5GB × 1 request = $0.45/month
```

---

## Monitoring for Optimization

### 1. Set Up Metrics

```python
# Instrument your code with metrics
from prometheus_client import Counter, Histogram, Gauge
import time

# Count requests
request_count = Counter('api_requests_total', 'Total API requests', ['endpoint', 'status'])

# Measure latency
request_latency = Histogram('api_request_duration_seconds', 'Request duration', ['endpoint'])

# Track resource usage
model_memory = Gauge('model_memory_bytes', 'Model memory usage')

@app.get("/api/predict")
async def predict(data: dict):
    start_time = time.time()

    try:
        result = model.predict(data)
        request_count.labels(endpoint='/predict', status='success').inc()
        return {"result": result}
    except Exception as e:
        request_count.labels(endpoint='/predict', status='error').inc()
        raise
    finally:
        duration = time.time() - start_time
        request_latency.labels(endpoint='/predict').observe(duration)
```

### 2. Create Dashboards

**Key metrics to track:**
- **Response time**: p50, p95, p99
- **Error rate**: percentage of failed requests
- **Resource usage**: CPU, memory, disk
- **Request rate**: requests per second
- **Cost**: daily/monthly spend

### 3. Set Up Alerts

```yaml
# prometheus-alerts.yaml
groups:
- name: performance
  rules:
  # Alert if p95 latency > 1 second
  - alert: HighLatency
    expr: histogram_quantile(0.95, api_request_duration_seconds) > 1
    for: 5m
    annotations:
      summary: "API latency high ({{ $value }}s)"

  # Alert if error rate > 1%
  - alert: HighErrorRate
    expr: rate(api_requests_total{status="error"}[5m]) > 0.01
    for: 5m
    annotations:
      summary: "Error rate high ({{ $value }}%)"
```

### 4. Regular Review Process

**Monthly optimization review:**
```
1. Review dashboard
   - Which endpoints are slowest?
   - Where are resources over-provisioned?
   - What's costing the most?

2. Profile slow endpoints
   - Use cProfile, line_profiler
   - Check database query performance
   - Look for N+1 queries

3. Optimize top 3 issues
   - Focus on biggest impact
   - Measure before/after
   - Document improvements

4. Update resource limits
   - Based on actual usage
   - Right-size containers
   - Adjust HPA settings
```

---

## Optimization Checklist

### Docker Images
```
✅ Use slim base images (python:3.9-slim)
✅ Multi-stage builds
✅ Optimize layer caching order
✅ Use .dockerignore
✅ Combine RUN commands
✅ Target image size < 500MB for Python apps
```

### Python Code
```
✅ Profile before optimizing
✅ Use built-in functions
✅ Use list comprehensions
✅ Use sets for membership testing
✅ Use NumPy/Pandas vectorization
✅ Cache expensive computations
✅ Lazy load resources
```

### Database
```
✅ Add indexes on frequently queried columns
✅ Use EXPLAIN to analyze queries
✅ Avoid N+1 queries (use eager loading)
✅ Limit data retrieved (SELECT specific columns)
✅ Use connection pooling
✅ Cache query results
```

### API
```
✅ Add caching headers
✅ Implement pagination
✅ Use async for I/O operations
✅ Compress responses (gzip)
✅ Cache responses (Redis)
✅ Add rate limiting
```

### Cost
```
✅ Right-size resources
✅ Use spot instances for training
✅ Set up auto-scaling
✅ Clean up unused resources
✅ Use reserved instances for stable workloads
✅ Monitor and optimize monthly
```

---

## Additional Resources

- **Docker Best Practices**: https://docs.docker.com/develop/dev-best-practices/
- **Python Performance**: https://wiki.python.org/moin/PythonSpeed/PerformanceTips
- **PostgreSQL Performance**: https://wiki.postgresql.org/wiki/Performance_Optimization
- **API Design**: https://cloud.google.com/apis/design
- **AWS Cost Optimization**: https://aws.amazon.com/pricing/cost-optimization/

---

**Remember:** Optimize based on data, not assumptions. Always measure!
