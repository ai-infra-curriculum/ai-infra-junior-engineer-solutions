# Common Pitfalls for Junior AI Infrastructure Engineers

**Last Updated**: 2025-10-25
**Level**: Junior
**Purpose**: Learn from common mistakes and avoid them in your projects

---

## Table of Contents

1. [Introduction](#introduction)
2. [Docker Pitfalls](#docker-pitfalls)
3. [Kubernetes Pitfalls](#kubernetes-pitfalls)
4. [Python Pitfalls](#python-pitfalls)
5. [Database Pitfalls](#database-pitfalls)
6. [Security Pitfalls](#security-pitfalls)
7. [CI/CD Pitfalls](#cicd-pitfalls)
8. [Monitoring and Logging Pitfalls](#monitoring-and-logging-pitfalls)
9. [General Infrastructure Pitfalls](#general-infrastructure-pitfalls)

---

## Introduction

This guide documents the most common mistakes junior engineers make when building AI infrastructure. Each pitfall includes:
- **What**: Description of the mistake
- **Why it's bad**: The impact
- **How to spot it**: Red flags
- **How to fix it**: The solution
- **Real example**: Actual code showing the problem and fix

Learning from these mistakes will save you time, prevent production issues, and make you a better engineer.

---

## Docker Pitfalls

### Pitfall 1: Using `latest` Tag in Production

**What**:
```dockerfile
# ❌ BAD
FROM python:latest
FROM ubuntu:latest
```

**Why it's bad**:
- `latest` changes over time
- Builds become non-reproducible
- Surprise breaking changes
- Hard to debug version-specific issues

**How to spot it**:
```bash
# Check your Dockerfile
grep "FROM.*:latest" Dockerfile

# Check running containers
docker ps --format "{{.Image}}" | grep latest
```

**How to fix it**:
```dockerfile
# ✅ GOOD: Use specific versions
FROM python:3.9.18-slim
FROM ubuntu:22.04

# Even better: Pin to digest
FROM python:3.9.18-slim@sha256:abcd1234...
```

**Real example from Module 5**:
```dockerfile
# Before (from exercise-01)
FROM python:latest
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]

# After (from exercise-02)
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

---

### Pitfall 2: Poor Layer Caching

**What**:
```dockerfile
# ❌ BAD: Copy everything first
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

**Why it's bad**:
- Every code change invalidates all layers
- Reinstalls all dependencies every time
- Slow builds (10+ minutes for large projects)
- Wastes CI/CD time and resources

**How to spot it**:
```bash
# Build and time it
time docker build -t myapp .

# Make small code change
echo "# comment" >> app.py

# Build again - should be fast but isn't
time docker build -t myapp .
```

**How to fix it**:
```dockerfile
# ✅ GOOD: Copy dependencies first
FROM python:3.9-slim
WORKDIR /app

# Copy only requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code last (changes frequently)
COPY . .
CMD ["python", "app.py"]
```

**Impact**:
- Before: 10 minute builds
- After: 30 second builds (after first build)

---

### Pitfall 3: Running as Root

**What**:
```dockerfile
# ❌ BAD: Runs as root (UID 0)
FROM python:3.9-slim
COPY . /app
WORKDIR /app
CMD ["python", "app.py"]
```

**Why it's bad**:
- Security vulnerability
- If container is compromised, attacker has root
- Can modify host filesystem (with volumes)
- Violates principle of least privilege

**How to spot it**:
```bash
# Check running container
docker exec <container> id
# Output: uid=0(root) gid=0(root)  # BAD!

# Check in Dockerfile
grep -i "USER" Dockerfile
# No output = running as root
```

**How to fix it**:
```dockerfile
# ✅ GOOD: Create and use non-root user
FROM python:3.9-slim

# Create user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

CMD ["python", "app.py"]
```

**Verify**:
```bash
docker exec <container> id
# Output: uid=1000(appuser) gid=1000(appuser)  # GOOD!
```

---

### Pitfall 4: Huge Image Sizes

**What**:
```dockerfile
# ❌ BAD: 2+ GB image
FROM python:3.9
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    git \
    vim \
    curl \
    wget
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
```

**Why it's bad**:
- Slow downloads (minutes per deployment)
- Expensive storage costs
- Security vulnerabilities (more packages = more CVEs)
- Slow container startup

**How to spot it**:
```bash
# Check image size
docker images | grep myapp
# myapp    latest    2.1GB    # Too big!

# List large images
docker images --format "{{.Repository}}:{{.Tag}}\t{{.Size}}" | sort -k2 -h
```

**How to fix it**:
```dockerfile
# ✅ GOOD: Multi-stage build, slim base
# Build stage
FROM python:3.9 as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.9-slim
WORKDIR /app

# Copy only runtime dependencies
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY . .

CMD ["python", "app.py"]
```

**Results**:
- Before: 2.1 GB
- After: 180 MB
- Savings: 91%

**Additional optimizations**:
```dockerfile
# Use Alpine (if compatible)
FROM python:3.9-alpine  # ~50-100MB

# Clean up in same layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y gcc && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*
```

---

### Pitfall 5: No .dockerignore

**What**:
```
# No .dockerignore file
```

**Why it's bad**:
- Copies unnecessary files (data, models, .git)
- Huge build context (30+ GB)
- Slow builds
- Exposes sensitive files

**How to spot it**:
```bash
# Watch build output
docker build -t myapp .
# Sending build context to Docker daemon  15.5GB  # TOO BIG!

# Check what's being copied
docker build -t myapp . 2>&1 | grep "Sending build context"
```

**How to fix it**:
```dockerignore
# .dockerignore
# Git
.git
.gitignore
.github

# Python
__pycache__
*.pyc
*.pyo
*.pyd
.Python
*.so
pip-log.txt
pip-delete-this-directory.txt
.pytest_cache
.coverage
htmlcov/

# Data and models (large files)
data/
*.csv
*.pkl
*.h5
*.pth
models/*.pkl

# Development
.vscode
.idea
*.swp
*.swo
*~

# Documentation
*.md
docs/

# Environment
.env
.env.*
venv/
env/

# Logs
*.log

# OS
.DS_Store
Thumbs.db
```

**Results**:
- Before: 15.5 GB build context
- After: 50 MB build context

---

### Pitfall 6: Missing Health Checks

**What**:
```yaml
# docker-compose.yml - ❌ BAD
version: '3.8'
services:
  web:
    image: myapp
    ports:
      - "8000:8000"
```

**Why it's bad**:
- Container might be running but app is crashed
- Load balancers send traffic to unhealthy containers
- No automatic recovery
- Silent failures

**How to spot it**:
```bash
# Check container health
docker ps
# HEALTH shows "none" instead of "healthy"

# Check docker-compose.yml
grep -A 5 "healthcheck" docker-compose.yml
# No output = no health check
```

**How to fix it**:
```yaml
# ✅ GOOD
version: '3.8'
services:
  web:
    image: myapp
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 40s
```

**In Dockerfile**:
```dockerfile
# Add health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

**Verify**:
```bash
docker ps
# HEALTH shows "healthy" or "unhealthy"
```

---

## Kubernetes Pitfalls

### Pitfall 7: No Resource Limits

**What**:
```yaml
# ❌ BAD: No resource limits
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: ml-api:v1
```

**Why it's bad**:
- One pod can consume all node resources
- Other pods get evicted
- Cluster instability
- Expensive (pods request more than needed)

**How to spot it**:
```bash
# Check deployments
kubectl get deploy -o yaml | grep -A 10 resources
# Empty = no limits

# Check resource usage
kubectl top pods
# Shows unlimited consumption
```

**How to fix it**:
```yaml
# ✅ GOOD: Set requests and limits
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: ml-api:v1
        resources:
          requests:
            cpu: 500m      # Guaranteed
            memory: 512Mi
          limits:
            cpu: 1000m     # Max allowed
            memory: 1Gi
```

**Finding right values**:
```bash
# Monitor actual usage
kubectl top pod <pod-name>

# Set requests = typical usage
# Set limits = max acceptable spike (1.5-2x requests)
```

---

### Pitfall 8: No Liveness/Readiness Probes

**What**:
```yaml
# ❌ BAD: No health probes
spec:
  containers:
  - name: api
    image: ml-api:v1
    ports:
    - containerPort: 8000
```

**Why it's bad**:
- Kubernetes sends traffic to broken pods
- No automatic restart of crashed apps
- Slow rollouts (no readiness check)
- Poor user experience (errors)

**How to spot it**:
```bash
# Check deployment
kubectl describe deploy ml-api | grep -A 5 "Liveness\|Readiness"
# No output = no probes

# Broken pod keeps getting traffic
kubectl logs -f <pod-name>
# See errors but pod not restarting
```

**How to fix it**:
```yaml
# ✅ GOOD: Both probes configured
spec:
  containers:
  - name: api
    image: ml-api:v1
    ports:
    - containerPort: 8000

    # Restart if unhealthy
    livenessProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 60
      periodSeconds: 10
      timeoutSeconds: 3
      failureThreshold: 3

    # Remove from service if not ready
    readinessProbe:
      httpGet:
        path: /ready
        port: 8000
      initialDelaySeconds: 10
      periodSeconds: 5
      timeoutSeconds: 3
      failureThreshold: 3
```

**Implement in your app**:
```python
# app.py
from fastapi import FastAPI, status

app = FastAPI()

@app.get("/health")
async def health():
    """Liveness probe - is app running?"""
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    """Readiness probe - can handle traffic?"""
    # Check dependencies
    if not db_connection_ok():
        return Response(status_code=503)
    if not model_loaded():
        return Response(status_code=503)
    return {"status": "ready"}
```

---

### Pitfall 9: Hardcoded Configuration

**What**:
```yaml
# ❌ BAD: Values hardcoded in manifests
spec:
  containers:
  - name: api
    image: ml-api:v1
    env:
    - name: DB_HOST
      value: "postgres.production.svc.cluster.local"
    - name: DB_PASSWORD
      value: "password123"  # SECURITY RISK!
```

**Why it's bad**:
- Passwords in Git (security breach)
- Can't reuse manifests across environments
- Hard to update configuration
- Version control shows sensitive data

**How to spot it**:
```bash
# Check for hardcoded values
kubectl get deploy ml-api -o yaml | grep -A 5 "env:"

# Search Git history for secrets
git log -p | grep -i "password\|token\|key"
```

**How to fix it**:
```yaml
# ✅ GOOD: Use ConfigMaps and Secrets
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-api-config
data:
  db_host: postgres.production.svc.cluster.local
  db_port: "5432"
  log_level: INFO

---
apiVersion: v1
kind: Secret
metadata:
  name: ml-api-secrets
type: Opaque
stringData:
  db_password: ""  # Set via kubectl, not committed

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
spec:
  template:
    spec:
      containers:
      - name: api
        image: ml-api:v1
        env:
        - name: DB_HOST
          valueFrom:
            configMapKeyRef:
              name: ml-api-config
              key: db_host
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: ml-api-secrets
              key: db_password
```

**Create secrets securely**:
```bash
# Don't commit this!
kubectl create secret generic ml-api-secrets \
  --from-literal=db_password='actual_password' \
  -n production
```

---

### Pitfall 10: Single Replica in Production

**What**:
```yaml
# ❌ BAD: Only one replica
spec:
  replicas: 1
```

**Why it's bad**:
- No high availability
- Downtime during deployments
- Single point of failure
- No load distribution

**How to spot it**:
```bash
# Check replica count
kubectl get deploy
# NAME      READY   UP-TO-DATE   AVAILABLE
# ml-api    1/1     1            1          # Only 1!

# Check during rollout
kubectl rollout status deploy/ml-api
# Waiting for deployment to complete: 0 of 1 updated replicas are available
# Service is down during this time!
```

**How to fix it**:
```yaml
# ✅ GOOD: Multiple replicas
spec:
  replicas: 3  # Minimum for HA

  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # Add 1 extra during update
      maxUnavailable: 0  # Keep all available
```

**With PodDisruptionBudget**:
```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: ml-api-pdb
spec:
  minAvailable: 2  # At least 2 pods always
  selector:
    matchLabels:
      app: ml-api
```

---

### Pitfall 11: Forgetting to Set Namespace

**What**:
```bash
# ❌ BAD: No namespace specified
kubectl apply -f deployment.yaml
# Deploys to "default" namespace
```

**Why it's bad**:
- Mixes dev/staging/prod in default namespace
- Hard to manage and isolate
- Security risks (no network policies)
- Confusing kubectl output

**How to spot it**:
```bash
# Check where pods are
kubectl get pods
# Shows only "default" namespace

kubectl get pods --all-namespaces | grep ml-api
# Pods in random namespaces
```

**How to fix it**:
```yaml
# ✅ GOOD: Always specify namespace
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
  namespace: production  # Explicit namespace
spec:
  # ...
```

**Or use kubectl**:
```bash
# Apply to specific namespace
kubectl apply -f deployment.yaml -n production

# Set default namespace
kubectl config set-context --current --namespace=production

# Verify
kubectl config view --minify | grep namespace
```

---

## Python Pitfalls

### Pitfall 12: Mutable Default Arguments

**What**:
```python
# ❌ BAD
def process_items(items=[]):
    items.append("new")
    return items

# Call multiple times
print(process_items())  # ['new']
print(process_items())  # ['new', 'new']  # BUG!
print(process_items())  # ['new', 'new', 'new']  # Keeps growing!
```

**Why it's bad**:
- List is created once at function definition
- Shared across all function calls
- Leads to hard-to-debug bugs
- State leaks between calls

**How to spot it**:
```bash
# Search for mutable defaults
grep -r "def.*\[\]" *.py
grep -r "def.*{}" *.py
```

**How to fix it**:
```python
# ✅ GOOD: Use None and create new object
def process_items(items=None):
    if items is None:
        items = []
    items.append("new")
    return items

# Now works correctly
print(process_items())  # ['new']
print(process_items())  # ['new']  # CORRECT!

# Or use immutable default
from typing import List, Optional

def process_items(items: Optional[List[str]] = None) -> List[str]:
    if items is None:
        items = []
    items.append("new")
    return items
```

---

### Pitfall 13: Bare Except Clauses

**What**:
```python
# ❌ BAD: Catches everything
def load_model(path):
    try:
        model = pickle.load(open(path, 'rb'))
        return model
    except:  # Catches ALL exceptions!
        print("Error loading model")
        return None
```

**Why it's bad**:
- Hides bugs (catches `KeyboardInterrupt`, `SystemExit`)
- Can't debug what went wrong
- Silently fails
- Masks programming errors

**How to spot it**:
```bash
# Find bare except
grep -n "except:" *.py

# Linter will catch this
flake8 *.py
# E722 do not use bare 'except'
```

**How to fix it**:
```python
# ✅ GOOD: Catch specific exceptions
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_model(path: Path):
    """Load model from file."""
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model

    except FileNotFoundError:
        logger.error(f"Model file not found: {path}")
        raise

    except pickle.UnpicklingError as e:
        logger.error(f"Invalid model file: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}")
        raise
```

**Better yet**:
```python
# ✅ BEST: Let exceptions propagate, handle at top level
def load_model(path: Path):
    """Load model from file."""
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    with open(path, 'rb') as f:
        return pickle.load(f)

# Handle at application level
def main():
    try:
        model = load_model(Path('model.pkl'))
        app.state.model = model
    except Exception as e:
        logger.critical(f"Failed to start: {e}")
        sys.exit(1)
```

---

### Pitfall 14: Not Closing Resources

**What**:
```python
# ❌ BAD: File not closed if exception occurs
def process_file(filename):
    f = open(filename)
    data = f.read()
    result = process(data)  # Might raise exception
    f.close()  # Never reached if exception!
    return result
```

**Why it's bad**:
- File descriptors leak
- Database connections leak
- Eventually hit system limits
- Files may be locked

**How to spot it**:
```bash
# Check open files
lsof -p $(pgrep python) | wc -l
# Number keeps growing = leak

# In code
grep "\.open(" *.py
# Check if followed by close() or context manager
```

**How to fix it**:
```python
# ✅ GOOD: Use context managers
def process_file(filename):
    with open(filename) as f:
        data = f.read()
        result = process(data)
    return result
    # File automatically closed, even if exception

# For custom resources
class DatabaseConnection:
    def __enter__(self):
        self.conn = create_connection()
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

# Use it
with DatabaseConnection() as conn:
    conn.execute("SELECT * FROM users")
```

---

### Pitfall 15: Global State and Imports

**What**:
```python
# ❌ BAD: model.py
import pickle

# Global variable loaded on import
MODEL = pickle.load(open('model.pkl', 'rb'))

def predict(data):
    return MODEL.predict(data)
```

**Why it's bad**:
- Loaded on every import (even for tests)
- Can't control when/how it's loaded
- Hard to mock in tests
- Fails if file doesn't exist at import time

**How to spot it**:
```python
# Try importing
python -c "import model"
# Fails if model.pkl doesn't exist

# Test run slow
pytest tests/  # Takes 30 seconds just to import
```

**How to fix it**:
```python
# ✅ GOOD: Lazy loading
import pickle
from pathlib import Path
from typing import Optional

_MODEL: Optional[object] = None

def get_model():
    """Get or load the model (singleton pattern)."""
    global _MODEL
    if _MODEL is None:
        model_path = Path(os.getenv('MODEL_PATH', 'model.pkl'))
        with open(model_path, 'rb') as f:
            _MODEL = pickle.load(f)
    return _MODEL

def predict(data):
    model = get_model()
    return model.predict(data)
```

**For FastAPI**:
```python
from fastapi import FastAPI

app = FastAPI()

@app.on_event("startup")
async def load_model():
    """Load model on application startup."""
    model_path = Path(os.getenv('MODEL_PATH', 'model.pkl'))
    with open(model_path, 'rb') as f:
        app.state.model = pickle.load(f)

@app.post("/predict")
async def predict(data: dict):
    return {"prediction": app.state.model.predict(data)}
```

---

## Database Pitfalls

### Pitfall 16: No Connection Pooling

**What**:
```python
# ❌ BAD: New connection for each request
def get_user(user_id):
    conn = psycopg2.connect(
        host="localhost",
        database="mydb",
        user="postgres",
        password="password"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user
```

**Why it's bad**:
- Slow (connection setup is expensive)
- Database server gets overwhelmed
- Can exhaust database connections
- Poor performance under load

**How to spot it**:
```bash
# Check database connections
psql -c "SELECT count(*) FROM pg_stat_activity"
# Number keeps growing

# Slow API response
time curl http://localhost:8000/api/user/123
# Takes 500ms+ for simple query
```

**How to fix it**:
```python
# ✅ GOOD: Use connection pooling
from psycopg2 import pool
from contextlib import contextmanager

# Create pool once at startup
connection_pool = pool.SimpleConnectionPool(
    minconn=1,
    maxconn=20,
    host="localhost",
    database="mydb",
    user="postgres",
    password="password"
)

@contextmanager
def get_db_connection():
    """Get connection from pool."""
    conn = connection_pool.getconn()
    try:
        yield conn
    finally:
        connection_pool.putconn(conn)

def get_user(user_id):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        cursor.close()
    return user
```

**With SQLAlchemy**:
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Create engine with pool
engine = create_engine(
    'postgresql://user:pass@localhost/db',
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True  # Verify connections are alive
)

SessionLocal = sessionmaker(bind=engine)

def get_user(user_id):
    session = SessionLocal()
    try:
        user = session.query(User).filter(User.id == user_id).first()
        return user
    finally:
        session.close()
```

---

### Pitfall 17: N+1 Query Problem

**What**:
```python
# ❌ BAD: N+1 queries
def get_users_with_posts():
    users = session.query(User).all()  # 1 query
    result = []
    for user in users:  # N queries
        posts = session.query(Post).filter(Post.user_id == user.id).all()
        result.append({
            'user': user.name,
            'posts': [p.title for p in posts]
        })
    return result

# If 100 users: 1 + 100 = 101 queries!
```

**Why it's bad**:
- Extremely slow (hundreds of queries)
- Database overload
- Network latency multiplied
- Doesn't scale

**How to spot it**:
```python
# Enable SQLAlchemy query logging
import logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Run function and count queries
get_users_with_posts()
# See hundreds of SELECT statements
```

**How to fix it**:
```python
# ✅ GOOD: Eager loading with join
from sqlalchemy.orm import joinedload

def get_users_with_posts():
    users = session.query(User).options(
        joinedload(User.posts)
    ).all()  # Just 1 query with JOIN!

    result = []
    for user in users:
        result.append({
            'user': user.name,
            'posts': [p.title for p in user.posts]  # No query!
        })
    return result
```

**Raw SQL alternative**:
```python
# ✅ GOOD: Manual join
def get_users_with_posts():
    query = """
        SELECT u.id, u.name, p.id as post_id, p.title
        FROM users u
        LEFT JOIN posts p ON u.id = p.user_id
    """
    results = session.execute(query).fetchall()

    # Group by user
    users = {}
    for row in results:
        if row.id not in users:
            users[row.id] = {
                'user': row.name,
                'posts': []
            }
        if row.post_id:
            users[row.id]['posts'].append(row.title)

    return list(users.values())
```

---

### Pitfall 18: Missing Database Indexes

**What**:
```sql
-- ❌ BAD: No index on frequently queried column
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    model_version VARCHAR(50),
    prediction FLOAT,
    created_at TIMESTAMP
);

-- Query is slow
SELECT * FROM predictions WHERE user_id = 123;
```

**Why it's bad**:
- Full table scan on every query
- Query time grows linearly with table size
- 1M rows = seconds per query
- Database CPU at 100%

**How to spot it**:
```sql
-- Check query performance
EXPLAIN ANALYZE SELECT * FROM predictions WHERE user_id = 123;
-- Seq Scan on predictions  (cost=0.00..15000.00 rows=100 width=32)
-- ^ "Seq Scan" means no index!

-- Check slow queries
SELECT query, mean_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```

**How to fix it**:
```sql
-- ✅ GOOD: Create index on queried columns
CREATE INDEX idx_predictions_user_id ON predictions(user_id);
CREATE INDEX idx_predictions_created_at ON predictions(created_at);

-- Composite index for common query pattern
CREATE INDEX idx_predictions_user_created
ON predictions(user_id, created_at);

-- Now query uses index
EXPLAIN ANALYZE SELECT * FROM predictions WHERE user_id = 123;
-- Index Scan using idx_predictions_user_id  (cost=0.42..8.44 rows=100 width=32)
-- ^ "Index Scan" = fast!
```

**Results**:
- Before: 2000ms query time
- After: 5ms query time
- 400x faster!

---

## Security Pitfalls

### Pitfall 19: Hardcoded Secrets in Code

**What**:
```python
# ❌ BAD: Secrets in code (committed to Git!)
DATABASE_URL = "postgresql://admin:MySecretPass123@db.example.com/prod"
API_KEY = "sk-1234567890abcdef1234567890abcdef"
AWS_SECRET = "aB3dEf7hIjKlMnOpQrStUvWxYz1234567890ABCD"
```

**Why it's bad**:
- Exposed in Git history forever
- Anyone with repo access has secrets
- Secrets in logs, backups, CI artifacts
- Can't rotate without code changes
- **Major security breach**

**How to spot it**:
```bash
# Search for common patterns
grep -r "password\s*=\s*['\"]" .
grep -r "api_key\s*=\s*['\"]" .
grep -r "secret\s*=\s*['\"]" .

# Check Git history
git log -p | grep -i "password\|secret\|token"

# Use automated tools
pip install detect-secrets
detect-secrets scan
```

**How to fix it**:
```python
# ✅ GOOD: Use environment variables
import os

DATABASE_URL = os.environ["DATABASE_URL"]  # Required
API_KEY = os.getenv("API_KEY")  # Optional
AWS_SECRET = os.environ.get("AWS_SECRET")
```

**With .env files**:
```python
# .env (NOT committed to Git, in .gitignore)
DATABASE_URL=postgresql://admin:MySecretPass123@db.example.com/prod
API_KEY=sk-1234567890abcdef
AWS_SECRET=aB3dEf7hIjKlMnOpQrStUvWxYz

# .env.example (safe to commit, shows structure)
DATABASE_URL=postgresql://user:password@host/db
API_KEY=your_api_key_here
AWS_SECRET=your_aws_secret_here

# Load in Python
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
```

**If already committed**:
```bash
# Remove from Git history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch config.py" \
  --prune-empty --tag-name-filter cat -- --all

# Or use BFG Repo-Cleaner (easier)
bfg --delete-files config.py

# Force push (⚠️ coordinate with team)
git push origin --force --all

# Rotate all exposed secrets immediately!
```

---

### Pitfall 20: Running Privileged Containers

**What**:
```yaml
# ❌ BAD: Privileged container
spec:
  containers:
  - name: api
    image: ml-api:v1
    securityContext:
      privileged: true  # DANGEROUS!
```

**Why it's bad**:
- Container has root access to host
- Can access all host devices
- Can modify host kernel
- Container escape possible
- **Complete security breach if compromised**

**How to spot it**:
```bash
# Check deployments
kubectl get deploy -o yaml | grep -i privileged

# Check running pods
kubectl get pods -o yaml | grep -i privileged
```

**How to fix it**:
```yaml
# ✅ GOOD: Restrictive security context
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000

  containers:
  - name: api
    image: ml-api:v1
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      runAsNonRoot: true
      runAsUser: 1000
      capabilities:
        drop:
        - ALL
```

**Pod Security Standards**:
```yaml
# Apply pod security at namespace level
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

---

## CI/CD Pitfalls

### Pitfall 21: No Testing in CI Pipeline

**What**:
```yaml
# .github/workflows/deploy.yml
# ❌ BAD: Deploy without testing
name: Deploy
on: [push]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy
      run: kubectl apply -f k8s/
```

**Why it's bad**:
- Broken code reaches production
- No quality gates
- Bugs discovered by users
- Expensive rollbacks

**How to spot it**:
```bash
# Check CI config
grep -r "pytest\|test\|unittest" .github/workflows/
# No output = no tests!
```

**How to fix it**:
```yaml
# ✅ GOOD: Test before deploy
name: CI/CD
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Lint
      run: |
        flake8 src/
        black --check src/
        mypy src/

    - name: Test
      run: |
        pytest tests/ -v --cov=src --cov-report=xml

    - name: Security scan
      run: bandit -r src/

  deploy:
    needs: test  # Only deploy if tests pass
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to production
      run: kubectl apply -f k8s/
```

---

### Pitfall 22: Deploying to Production from Local Machine

**What**:
```bash
# ❌ BAD: Manual deployment from laptop
kubectl apply -f k8s/deployment.yaml --context=production
docker build -t myapp:latest .
docker push myapp:latest
```

**Why it's bad**:
- No audit trail (who deployed what?)
- No testing before deploy
- Can deploy wrong version
- "Works on my machine" syndrome
- No rollback mechanism

**How to spot it**:
```bash
# Check deployment history
kubectl rollout history deployment/ml-api
# No annotation of who/what/why

# Check Git vs deployed version
git log -1 --pretty=format:"%H"
kubectl get deploy ml-api -o yaml | grep image:
# Different = manual deploy!
```

**How to fix it**:
```yaml
# ✅ GOOD: Automated CI/CD pipeline
name: Deploy to Production
on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2

    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBECONFIG }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig

    - name: Deploy
      run: |
        kubectl set image deployment/ml-api \
          api=myregistry.io/ml-api:${{ github.sha }} \
          --record \
          -n production

    - name: Wait for rollout
      run: |
        kubectl rollout status deployment/ml-api -n production

    - name: Verify deployment
      run: |
        kubectl get pods -n production
        curl -f https://api.example.com/health
```

---

## Monitoring and Logging Pitfalls

### Pitfall 23: Using Print Instead of Logging

**What**:
```python
# ❌ BAD: Print statements
def process_request(request):
    print(f"Processing request: {request['id']}")
    result = do_work(request)
    print(f"Result: {result}")
    return result
```

**Why it's bad**:
- Can't filter by severity
- No timestamps
- Not structured (hard to parse)
- Can't route to different outputs
- Mixes with actual output

**How to spot it**:
```bash
# Search for print statements
grep -r "print(" src/

# Container logs are messy
kubectl logs <pod> | head
# No timestamps, severity, or structure
```

**How to fix it**:
```python
# ✅ GOOD: Use logging module
import logging

logger = logging.getLogger(__name__)

def process_request(request):
    logger.info(
        "Processing request",
        extra={'request_id': request['id']}
    )

    try:
        result = do_work(request)
        logger.info(
            "Request processed successfully",
            extra={
                'request_id': request['id'],
                'result': result
            }
        )
        return result

    except Exception as e:
        logger.error(
            "Request processing failed",
            extra={'request_id': request['id']},
            exc_info=True
        )
        raise
```

**Configure logging**:
```python
# logging_config.py
import logging
import sys

def setup_logging(log_level="INFO"):
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log')
        ]
    )
```

---

### Pitfall 24: No Metrics Collection

**What**:
```python
# ❌ BAD: No metrics
@app.post("/predict")
async def predict(data: dict):
    result = model.predict(data)
    return {"prediction": result}
```

**Why it's bad**:
- Can't track performance
- Don't know if service is healthy
- Can't set SLOs/SLAs
- Hard to debug issues
- No visibility into usage

**How to spot it**:
```bash
# Check for metrics endpoint
curl http://localhost:8000/metrics
# 404 = no metrics

# Check Prometheus targets
# No target for your service
```

**How to fix it**:
```python
# ✅ GOOD: Collect metrics
from prometheus_client import Counter, Histogram, make_asgi_app
from fastapi import FastAPI
import time

app = FastAPI()

# Define metrics
requests_total = Counter(
    'api_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'api_request_duration_seconds',
    'Request duration',
    ['method', 'endpoint']
)

predictions_total = Counter(
    'predictions_total',
    'Total predictions',
    ['model_version']
)

@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Track request metrics."""
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    requests_total.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)

    return response

@app.post("/predict")
async def predict(data: dict):
    result = model.predict(data)
    predictions_total.labels(model_version="v1.0.0").inc()
    return {"prediction": result}

# Metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

---

## General Infrastructure Pitfalls

### Pitfall 25: No Backup Strategy

**What**:
```yaml
# ❌ BAD: Database with no backups
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  # ... no backup configuration
```

**Why it's bad**:
- Data loss is permanent
- No recovery from mistakes
- No disaster recovery
- Can't restore to previous state

**How to spot it**:
```bash
# Check for backup CronJobs
kubectl get cronjobs
# No backup jobs = no backups!

# Check backup storage
aws s3 ls s3://backups/database/
# Empty or old = no backups
```

**How to fix it**:
```yaml
# ✅ GOOD: Automated database backups
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:14
            command:
            - /bin/sh
            - -c
            - |
              BACKUP_FILE="backup-$(date +%Y%m%d-%H%M%S).sql.gz"
              pg_dump -h postgres -U postgres mydb | gzip > /backups/$BACKUP_FILE
              aws s3 cp /backups/$BACKUP_FILE s3://my-backups/postgres/
              # Keep only last 30 days
              find /backups -mtime +30 -delete
            env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: postgres-secret
                  key: password
            volumeMounts:
            - name: backups
              mountPath: /backups
          volumes:
          - name: backups
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
```

**Test restores**:
```bash
# Regularly test that backups work!
# Download backup
aws s3 cp s3://my-backups/postgres/backup-20251025.sql.gz .

# Restore to test database
gunzip backup-20251025.sql.gz
psql -h test-db -U postgres testdb < backup-20251025.sql

# Verify data
psql -h test-db -U postgres testdb -c "SELECT COUNT(*) FROM users"
```

---

## Summary

### Top 10 Most Critical Pitfalls

1. **Hardcoded Secrets** - Leads to security breaches
2. **No Resource Limits** - Causes cluster instability
3. **Running as Root** - Security vulnerability
4. **No Health Checks** - Silent failures in production
5. **Single Replica** - No high availability
6. **Bare Except Clauses** - Hides bugs
7. **No Backups** - Permanent data loss
8. **Using `latest` Tag** - Non-reproducible builds
9. **N+1 Queries** - Performance disaster
10. **No Testing in CI** - Broken code in production

### Prevention Checklist

Before deploying to production:

- [ ] No secrets in code or Git
- [ ] Resource limits set on all containers
- [ ] Running as non-root user
- [ ] Health checks configured
- [ ] Multiple replicas (minimum 3)
- [ ] Specific error handling (no bare except)
- [ ] Backup strategy implemented and tested
- [ ] Specific version tags (no `latest`)
- [ ] Database queries optimized
- [ ] CI pipeline includes tests

### Learning from Mistakes

**When you encounter a bug**:

1. **Document it**: Write down what went wrong
2. **Understand it**: Why did this happen?
3. **Fix it**: Implement the solution
4. **Prevent it**: Add checks/tests to prevent recurrence
5. **Share it**: Help others learn from your mistake

**Remember**: Every expert was once a beginner who made these same mistakes. The difference is that they learned from them and now help others avoid them.

---

## Additional Resources

### Tools for Detection

- **detect-secrets**: Find secrets in code
- **bandit**: Security linting for Python
- **trivy**: Container vulnerability scanning
- **kube-score**: Kubernetes manifest validation
- **hadolint**: Dockerfile linting

### Learning Resources

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Production Best Practices](https://learnk8s.io/production-best-practices)
- [Python Anti-Patterns](https://docs.quantifiedcode.com/python-anti-patterns/)
- [Database Performance](https://use-the-index-luke.com/)

---

**Remember**: Making mistakes is part of learning. The key is to learn from them, prevent them in the future, and help others avoid them too!
