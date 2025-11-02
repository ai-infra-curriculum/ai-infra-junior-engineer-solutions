# Implementation Guide: Docker Compose for ML Applications

## Overview

This comprehensive guide covers Docker Compose fundamentals and advanced patterns for orchestrating multi-container ML applications. You'll learn how to build production-ready stacks with proper service dependencies, health checks, environment management, and scaling strategies.

**Target Audience**: AI Infrastructure Engineers
**Prerequisites**: Exercise 01 & 02, Docker fundamentals
**Estimated Time**: 3-4 hours
**Difficulty**: Intermediate

## Table of Contents

1. [Docker Compose Basics](#1-docker-compose-basics)
2. [Multi-Container ML Applications](#2-multi-container-ml-applications)
3. [Service Dependencies and Health Checks](#3-service-dependencies-and-health-checks)
4. [Networks and Volumes in Compose](#4-networks-and-volumes-in-compose)
5. [Environment Variable Management](#5-environment-variable-management)
6. [Scaling Services](#6-scaling-services)
7. [Production ML Stack with Compose](#7-production-ml-stack-with-compose)
8. [Advanced Patterns](#8-advanced-patterns)
9. [Troubleshooting](#9-troubleshooting)
10. [Best Practices](#10-best-practices)

---

## 1. Docker Compose Basics

### 1.1 Understanding Docker Compose

Docker Compose is a tool for defining and running multi-container Docker applications. Instead of managing containers individually with `docker run` commands, you define your entire application stack in a YAML file.

**Key Concepts**:

- **Services**: Containerized components of your application
- **Networks**: Communication channels between services
- **Volumes**: Persistent data storage
- **Environment**: Configuration and secrets management

**Compose File Structure**:

```yaml
version: '3.8'

services:
  # Define your application components
  service_name:
    image: image_name:tag
    # OR
    build: ./path/to/dockerfile

volumes:
  # Define named volumes for persistence
  volume_name:

networks:
  # Define custom networks
  network_name:
```

### 1.2 Basic Compose File Anatomy

```yaml
version: '3.8'  # Compose file format version

services:
  web:
    # Image to use (from Docker Hub or registry)
    image: nginx:alpine

    # OR build from Dockerfile
    build:
      context: .
      dockerfile: Dockerfile

    # Port mapping: HOST:CONTAINER
    ports:
      - "8080:80"

    # Environment variables
    environment:
      - ENV_VAR=value
      - DEBUG=true

    # Volume mounts
    volumes:
      - ./local/path:/container/path
      - named_volume:/data

    # Network configuration
    networks:
      - app_network

    # Restart policy
    restart: always

    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M

volumes:
  named_volume:

networks:
  app_network:
    driver: bridge
```

### 1.3 Essential Compose Commands

```bash
# Start all services (foreground)
docker compose up

# Start services in background (detached)
docker compose up -d

# Stop and remove containers, networks
docker compose down

# Stop and remove everything including volumes
docker compose down -v

# View running services
docker compose ps

# View logs from all services
docker compose logs

# Follow logs in real-time
docker compose logs -f

# View logs for specific service
docker compose logs -f web

# Restart specific service
docker compose restart web

# Execute command in running container
docker compose exec web bash

# Build/rebuild services
docker compose build

# Pull latest images
docker compose pull

# Validate and view the compose file
docker compose config

# Scale a service
docker compose up -d --scale web=3
```

### 1.4 Your First Compose Application

Let's create a simple web application with a database:

```bash
# Create project directory
mkdir -p ~/docker-compose-intro
cd ~/docker-compose-intro

# Create a simple Flask application
cat > app.py << 'EOF'
from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.route('/')
def hello():
    return jsonify({
        'message': 'Hello from Docker Compose!',
        'environment': os.environ.get('ENVIRONMENT', 'development')
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
EOF

# Create requirements file
cat > requirements.txt << 'EOF'
flask==3.0.0
EOF

# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]
EOF

# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - ENVIRONMENT=development
    volumes:
      - .:/app  # Mount current directory for hot reload

volumes: {}
networks: {}
EOF
```

**Run the application**:

```bash
# Start the application
docker compose up

# In another terminal, test it
curl http://localhost:5000
curl http://localhost:5000/health

# Stop with Ctrl+C, then clean up
docker compose down
```

**Key Takeaways**:
- Compose files use YAML syntax
- Services are defined under the `services:` key
- `build:` tells Compose to build from a Dockerfile
- `ports:` maps host ports to container ports
- `volumes:` can mount code for development

---

## 2. Multi-Container ML Applications

### 2.1 Why Multi-Container Architecture?

In ML applications, you typically need:
- **API Service**: Serves model predictions
- **Database**: Stores prediction logs, features, metadata
- **Cache**: Redis for caching predictions, features
- **Message Queue**: For async tasks (training, batch predictions)
- **Monitoring**: Track metrics, performance

**Benefits**:
- Separation of concerns
- Independent scaling
- Technology flexibility
- Easier testing and development

### 2.2 Simple ML Stack: API + Database

```bash
# Create ML project directory
mkdir -p ~/ml-compose-basic
cd ~/ml-compose-basic

# Create ML prediction API
cat > predict_api.py << 'EOF'
from flask import Flask, request, jsonify
import psycopg2
import os
from datetime import datetime
import numpy as np

app = Flask(__name__)

def get_db_connection():
    """Create database connection"""
    return psycopg2.connect(
        host=os.environ.get('DB_HOST', 'db'),
        database=os.environ.get('DB_NAME', 'mlapp'),
        user=os.environ.get('DB_USER', 'postgres'),
        password=os.environ.get('DB_PASSWORD', 'postgres')
    )

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        conn = get_db_connection()
        conn.close()
        return jsonify({'status': 'healthy', 'database': 'connected'}), 200
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Simple prediction endpoint"""
    try:
        data = request.json
        features = data.get('features', [])

        # Simple mock prediction (replace with real model)
        prediction = float(np.mean(features))
        confidence = float(np.random.random())

        # Log prediction to database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO predictions (input_features, prediction, confidence, created_at)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (str(features), prediction, confidence, datetime.utcnow())
        )
        prediction_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            'prediction_id': prediction_id,
            'prediction': prediction,
            'confidence': confidence
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def stats():
    """Get prediction statistics"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = cur.fetchone()[0]
        cur.close()
        conn.close()

        return jsonify({
            'total_predictions': total_predictions
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
EOF

# Create requirements
cat > requirements.txt << 'EOF'
flask==3.0.0
psycopg2-binary==2.9.9
numpy==1.26.0
EOF

# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict_api.py .

EXPOSE 8000

CMD ["python", "predict_api.py"]
EOF

# Create database initialization script
mkdir -p init-db
cat > init-db/01-schema.sql << 'EOF'
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    input_features TEXT NOT NULL,
    prediction REAL NOT NULL,
    confidence REAL NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_created_at ON predictions(created_at);
CREATE INDEX idx_confidence ON predictions(confidence);
EOF

# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DB_HOST=db
      - DB_NAME=mlapp
      - DB_USER=postgres
      - DB_PASSWORD=postgres
    depends_on:
      - db
    restart: on-failure

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=mlapp
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-db:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"  # Expose for debugging

volumes:
  postgres_data:
EOF
```

**Test the stack**:

```bash
# Start services
docker compose up -d

# Wait for services to start
sleep 5

# Check health
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, 4.0]}'

# Check stats
curl http://localhost:8000/stats

# View logs
docker compose logs -f api

# Cleanup
docker compose down -v
```

### 2.3 Adding Redis for Caching

Now let's add Redis to cache predictions:

```python
# Update predict_api.py to include caching
cat > predict_api.py << 'EOF'
from flask import Flask, request, jsonify
import psycopg2
import redis
import os
import json
import hashlib
from datetime import datetime
import numpy as np

app = Flask(__name__)

# Redis connection
cache = redis.Redis(
    host=os.environ.get('REDIS_HOST', 'redis'),
    port=6379,
    decode_responses=True
)

def get_db_connection():
    return psycopg2.connect(
        host=os.environ.get('DB_HOST', 'db'),
        database=os.environ.get('DB_NAME', 'mlapp'),
        user=os.environ.get('DB_USER', 'postgres'),
        password=os.environ.get('DB_PASSWORD', 'postgres')
    )

@app.route('/health')
def health():
    health_status = {
        'status': 'healthy',
        'database': 'unknown',
        'cache': 'unknown'
    }

    try:
        conn = get_db_connection()
        conn.close()
        health_status['database'] = 'connected'
    except Exception as e:
        health_status['database'] = f'error: {str(e)}'
        health_status['status'] = 'unhealthy'

    try:
        cache.ping()
        health_status['cache'] = 'connected'
    except Exception as e:
        health_status['cache'] = f'error: {str(e)}'
        health_status['status'] = 'unhealthy'

    return jsonify(health_status), 200 if health_status['status'] == 'healthy' else 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = data.get('features', [])

        # Create cache key from features
        cache_key = f"prediction:{hashlib.md5(str(features).encode()).hexdigest()}"

        # Check cache
        cached_result = cache.get(cache_key)
        if cached_result:
            cache.incr('cache_hits')
            result = json.loads(cached_result)
            result['cached'] = True
            return jsonify(result), 200

        # Cache miss - compute prediction
        cache.incr('cache_misses')
        prediction = float(np.mean(features))
        confidence = float(np.random.random())

        # Log to database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO predictions (input_features, prediction, confidence, created_at)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (str(features), prediction, confidence, datetime.utcnow())
        )
        prediction_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()

        result = {
            'prediction_id': prediction_id,
            'prediction': prediction,
            'confidence': confidence,
            'cached': False
        }

        # Cache the result for 1 hour
        cache.setex(cache_key, 3600, json.dumps(result))

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def stats():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = cur.fetchone()[0]
        cur.close()
        conn.close()

        cache_hits = int(cache.get('cache_hits') or 0)
        cache_misses = int(cache.get('cache_misses') or 0)
        total_requests = cache_hits + cache_misses
        hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0

        return jsonify({
            'total_predictions': total_predictions,
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'cache_hit_rate': f"{hit_rate:.2f}%"
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
EOF
```

Update requirements:

```bash
cat > requirements.txt << 'EOF'
flask==3.0.0
psycopg2-binary==2.9.9
redis==5.0.1
numpy==1.26.0
EOF
```

Update docker-compose.yml:

```yaml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DB_HOST=db
      - DB_NAME=mlapp
      - DB_USER=postgres
      - DB_PASSWORD=postgres
      - REDIS_HOST=redis
    depends_on:
      - db
      - redis
    restart: on-failure

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=mlapp
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-db:/docker-entrypoint-initdb.d

  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
EOF
```

**Test caching**:

```bash
# Start services
docker compose up -d

# Wait for startup
sleep 5

# First request (cache miss)
time curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0]}'

# Second request (cache hit - should be faster)
time curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0]}'

# Check stats
curl http://localhost:8000/stats

# Cleanup
docker compose down -v
```

---

## 3. Service Dependencies and Health Checks

### 3.1 Understanding depends_on

The `depends_on` option controls startup order but doesn't wait for services to be "ready":

```yaml
services:
  web:
    depends_on:
      - db  # Starts db before web, but doesn't wait for db to be ready
```

**Problem**: The web service may start before the database is ready to accept connections.

### 3.2 Health Checks

Health checks tell Docker when a service is truly ready:

```yaml
services:
  db:
    image: postgres:15
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s      # Check every 10 seconds
      timeout: 5s        # Timeout after 5 seconds
      retries: 5         # Retry 5 times before marking unhealthy
      start_period: 30s  # Grace period for startup
```

### 3.3 Waiting for Service Health

Combine `depends_on` with `condition`:

```yaml
services:
  web:
    depends_on:
      db:
        condition: service_healthy  # Wait for db to be healthy
      redis:
        condition: service_healthy
```

### 3.4 Complete Example with Health Checks

```yaml
cat > docker-compose-health.yml << 'EOF'
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DB_HOST=db
      - DB_NAME=mlapp
      - DB_USER=postgres
      - DB_PASSWORD=postgres
      - REDIS_HOST=redis
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: on-failure

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=mlapp
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-db:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
    restart: always

  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
    restart: always

volumes:
  postgres_data:
  redis_data:
EOF
```

**Test health-based startup**:

```bash
# Start services with health checks
docker compose -f docker-compose-health.yml up -d

# Watch services become healthy
watch -n 1 'docker compose -f docker-compose-health.yml ps'

# Check individual service health
docker compose -f docker-compose-health.yml ps
docker inspect $(docker compose -f docker-compose-health.yml ps -q api) | grep -A 10 Health
```

### 3.5 Custom Health Check Scripts

For more complex health checks, create a custom script:

```bash
# Create health check script
cat > healthcheck.sh << 'EOF'
#!/bin/bash
set -e

# Check if API is responding
if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "API health check failed"
    exit 1
fi

# Check if we can connect to database
if ! python -c "import psycopg2; psycopg2.connect('host=db user=postgres password=postgres')" 2>/dev/null; then
    echo "Database connection failed"
    exit 1
fi

# Check if Redis is accessible
if ! python -c "import redis; redis.Redis(host='redis').ping()" 2>/dev/null; then
    echo "Redis connection failed"
    exit 1
fi

echo "All health checks passed"
exit 0
EOF

chmod +x healthcheck.sh
```

Update Dockerfile:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict_api.py .
COPY healthcheck.sh .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD ["./healthcheck.sh"]

CMD ["python", "predict_api.py"]
```

---

## 4. Networks and Volumes in Compose

### 4.1 Default Networking

Docker Compose automatically creates a default network for your services:

```yaml
services:
  web:
    # Can access 'db' by hostname
  db:
    # Can access 'web' by hostname
```

**How it works**:
- Services can reach each other using service names as hostnames
- Each service gets its own IP address
- Built-in DNS resolution

### 4.2 Custom Networks

Create isolated networks for different parts of your application:

```yaml
version: '3.8'

services:
  # Public-facing service
  nginx:
    image: nginx:alpine
    networks:
      - frontend
    ports:
      - "80:80"

  # Application layer (both networks)
  api:
    build: .
    networks:
      - frontend
      - backend
    environment:
      - DB_HOST=db

  # Database (backend only)
  db:
    image: postgres:15
    networks:
      - backend
    environment:
      - POSTGRES_PASSWORD=postgres

  # Cache (backend only)
  redis:
    image: redis:7-alpine
    networks:
      - backend

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true  # No external access
```

**Network isolation benefits**:
- Security: Database not accessible from frontend
- Organization: Clear separation of concerns
- Traffic control: Different network policies

### 4.3 Static IP Addresses

Assign static IPs within your network:

```yaml
services:
  api:
    networks:
      app_net:
        ipv4_address: 172.28.0.10

  db:
    networks:
      app_net:
        ipv4_address: 172.28.0.11

networks:
  app_net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/16
          gateway: 172.28.0.1
```

### 4.4 Volume Types

**Named Volumes** (managed by Docker):

```yaml
services:
  db:
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:  # Docker manages this
```

**Bind Mounts** (mount host directories):

```yaml
services:
  web:
    volumes:
      - ./app:/app  # Mount local directory
      - ./config:/config:ro  # Read-only mount
```

**tmpfs Mounts** (in-memory):

```yaml
services:
  app:
    volumes:
      - type: tmpfs
        target: /tmp
        tmpfs:
          size: 1000000000  # 1GB
```

### 4.5 Volume Management Patterns

**Development setup** (code mounting):

```yaml
services:
  web:
    volumes:
      - ./app:/app  # Source code
      - /app/node_modules  # Preserve node_modules
      - ./logs:/var/log  # Logs accessible on host
```

**Production setup** (named volumes):

```yaml
services:
  db:
    volumes:
      - db_data:/var/lib/postgresql/data

  redis:
    volumes:
      - redis_data:/data

  app:
    volumes:
      - app_logs:/var/log/app

volumes:
  db_data:
    driver: local
  redis_data:
    driver: local
  app_logs:
    driver: local
```

### 4.6 Complete Network and Volume Example

```yaml
cat > docker-compose-network-volumes.yml << 'EOF'
version: '3.8'

services:
  # Nginx reverse proxy (frontend network only)
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf:ro
    networks:
      - frontend
    depends_on:
      - api

  # ML API (both networks)
  api:
    build: .
    networks:
      - frontend
      - backend
    environment:
      - DB_HOST=db
      - REDIS_HOST=cache
    volumes:
      - ./models:/app/models:ro  # Read-only model directory
      - api_logs:/var/log/api
    depends_on:
      db:
        condition: service_healthy
      cache:
        condition: service_healthy

  # PostgreSQL (backend network only)
  db:
    image: postgres:15
    networks:
      - backend
    environment:
      - POSTGRES_PASSWORD=postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups  # Backup directory
    healthcheck:
      test: ["CMD-SHELL", "pg_isready"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis cache (backend network only)
  cache:
    image: redis:7-alpine
    networks:
      - backend
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true  # No internet access

volumes:
  postgres_data:
  redis_data:
  api_logs:
EOF
```

**Test network isolation**:

```bash
# Start the stack
docker compose -f docker-compose-network-volumes.yml up -d

# Nginx can reach API
docker compose -f docker-compose-network-volumes.yml exec nginx ping -c 2 api

# Nginx CANNOT reach DB (different networks)
docker compose -f docker-compose-network-volumes.yml exec nginx ping -c 2 db
# This will fail!

# API can reach both
docker compose -f docker-compose-network-volumes.yml exec api ping -c 2 nginx
docker compose -f docker-compose-network-volumes.yml exec api ping -c 2 db

# Cleanup
docker compose -f docker-compose-network-volumes.yml down -v
```

---

## 5. Environment Variable Management

### 5.1 Using .env Files

Create a `.env` file for default values:

```bash
cat > .env << 'EOF'
# Database Configuration
POSTGRES_DB=mlapp
POSTGRES_USER=postgres
POSTGRES_PASSWORD=devpassword123

# Application Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Service Ports
API_PORT=8000
DB_PORT=5432

# Resource Limits
API_MEMORY=512M
DB_MEMORY=1G
EOF
```

Reference in docker-compose.yml:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "${API_PORT}:8000"
    environment:
      - DB_HOST=db
      - DB_NAME=${POSTGRES_DB}
      - DB_USER=${POSTGRES_USER}
      - DB_PASSWORD=${POSTGRES_PASSWORD}
      - ENVIRONMENT=${ENVIRONMENT}
      - DEBUG=${DEBUG}
    deploy:
      resources:
        limits:
          memory: ${API_MEMORY}

  db:
    image: postgres:15
    ports:
      - "${DB_PORT}:5432"
    environment:
      - POSTGRES_DB=${POSTGRES_DB}
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    deploy:
      resources:
        limits:
          memory: ${DB_MEMORY}
```

### 5.2 Multiple Environment Files

**Development** (.env.development):

```bash
cat > .env.development << 'EOF'
POSTGRES_DB=mlapp_dev
POSTGRES_USER=devuser
POSTGRES_PASSWORD=devpass123

ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

API_PORT=8000
DB_PORT=5432

API_MEMORY=512M
DB_MEMORY=512M
EOF
```

**Production** (.env.production):

```bash
cat > .env.production << 'EOF'
POSTGRES_DB=mlapp_prod
POSTGRES_USER=produser
POSTGRES_PASSWORD=super-secure-production-password-change-me

ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

API_PORT=8080
DB_PORT=5433

API_MEMORY=2G
DB_MEMORY=4G
EOF
```

**Use specific env file**:

```bash
# Development (uses .env by default)
docker compose up -d

# Production
docker compose --env-file .env.production up -d

# Check which config is being used
docker compose --env-file .env.production config
```

### 5.3 Environment-Specific Overrides

**Base configuration** (docker-compose.yml):

```yaml
version: '3.8'

services:
  api:
    build: .
    environment:
      - DB_HOST=db
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready"]
      interval: 10s

volumes:
  postgres_data:
```

**Development override** (docker-compose.override.yml - used automatically):

```yaml
version: '3.8'

services:
  api:
    ports:
      - "8000:8000"
    volumes:
      - .:/app  # Mount source code
    environment:
      - DEBUG=true
      - LOG_LEVEL=DEBUG
    command: python -m flask run --host=0.0.0.0 --reload

  db:
    ports:
      - "5432:5432"  # Expose DB for debugging
    environment:
      - POSTGRES_PASSWORD=devpassword
```

**Production override** (docker-compose.prod.yml):

```yaml
version: '3.8'

services:
  api:
    ports:
      - "8080:8000"
    environment:
      - DEBUG=false
      - LOG_LEVEL=INFO
    command: gunicorn --bind 0.0.0.0:8000 --workers 4 app:app
    restart: always
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G

  db:
    environment:
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
    secrets:
      - db_password
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G

secrets:
  db_password:
    file: ./secrets/db_password.txt
```

**Usage**:

```bash
# Development (automatically uses docker-compose.yml + docker-compose.override.yml)
docker compose up -d

# Production (explicitly specify production override)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Staging (can chain multiple files)
docker compose -f docker-compose.yml -f docker-compose.staging.yml up -d
```

### 5.4 Secrets Management

**Never commit secrets to version control!**

Create a secrets directory:

```bash
mkdir -p secrets
echo "super-secure-db-password" > secrets/db_password.txt
echo "api-secret-key-change-me" > secrets/api_secret.txt

# Add to .gitignore
echo "secrets/" >> .gitignore
echo ".env.production" >> .gitignore
```

Use secrets in compose:

```yaml
version: '3.8'

services:
  api:
    environment:
      - SECRET_KEY_FILE=/run/secrets/api_secret
    secrets:
      - api_secret

  db:
    environment:
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
    secrets:
      - db_password

secrets:
  api_secret:
    file: ./secrets/api_secret.txt
  db_password:
    file: ./secrets/db_password.txt
```

---

## 6. Scaling Services

### 6.1 Basic Scaling

Scale a service to multiple replicas:

```bash
# Scale to 3 replicas
docker compose up -d --scale api=3

# Check all instances
docker compose ps

# Scale back to 1
docker compose up -d --scale api=1
```

### 6.2 Declarative Scaling in Compose File

```yaml
version: '3.8'

services:
  api:
    build: .
    deploy:
      replicas: 3  # Run 3 instances
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
    environment:
      - DB_HOST=db
```

### 6.3 Load Balancing with Nginx

Create nginx configuration:

```bash
mkdir -p nginx
cat > nginx/nginx.conf << 'EOF'
upstream api_backend {
    # Load balancing algorithm
    least_conn;  # Route to server with least connections

    # Docker's DNS will resolve 'api' to all replicas
    server api:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;

    location / {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Connection settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /health {
        access_log off;  # Don't log health checks
        proxy_pass http://api_backend;
    }
}
EOF
```

Complete scaling setup:

```yaml
cat > docker-compose-scaling.yml << 'EOF'
version: '3.8'

services:
  # Load balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/conf.d/default.conf:ro
    depends_on:
      - api
    restart: always

  # Scaled API service
  api:
    build: .
    expose:
      - "8000"  # Expose to nginx, not to host
    environment:
      - DB_HOST=db
      - REDIS_HOST=cache
    depends_on:
      db:
        condition: service_healthy
      cache:
        condition: service_healthy
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3

  # Shared database (single instance)
  db:
    image: postgres:15
    environment:
      - POSTGRES_PASSWORD=postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready"]
      interval: 10s
    restart: always

  # Shared cache (single instance)
  cache:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
    restart: always

volumes:
  postgres_data:
  redis_data:
EOF
```

**Test load balancing**:

```bash
# Start with 3 API replicas
docker compose -f docker-compose-scaling.yml up -d

# Make multiple requests - see different containers respond
for i in {1..10}; do
    curl -s http://localhost/health | grep -i hostname
    sleep 0.5
done

# View logs from all API instances
docker compose -f docker-compose-scaling.yml logs api

# Scale up
docker compose -f docker-compose-scaling.yml up -d --scale api=5

# Scale down
docker compose -f docker-compose-scaling.yml up -d --scale api=2

# Cleanup
docker compose -f docker-compose-scaling.yml down -v
```

### 6.4 Session Affinity (Sticky Sessions)

If you need requests from the same client to hit the same backend:

```nginx
upstream api_backend {
    # Use client IP for session affinity
    ip_hash;

    server api:8000;
}
```

Or use cookie-based affinity (requires nginx-plus or custom build).

---

## 7. Production ML Stack with Compose

### 7.1 Complete Production ML Architecture

Let's build a production-ready ML stack with:
- ML API (PyTorch ResNet)
- PostgreSQL (prediction logging)
- Redis (result caching)
- Nginx (reverse proxy, load balancing)
- Prometheus (metrics)
- Grafana (monitoring dashboards)

```bash
# Create project directory
mkdir -p ~/ml-production-stack
cd ~/ml-production-stack
```

**ML Model Service** (serve.py):

```python
cat > serve.py << 'EOF'
from flask import Flask, request, jsonify
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import redis
import psycopg2
import json
import hashlib
import os
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
import time

app = Flask(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('prediction_requests_total', 'Total prediction requests')
PREDICTION_DURATION = Histogram('prediction_duration_seconds', 'Prediction duration')
CACHE_HITS = Counter('cache_hits_total', 'Total cache hits')
CACHE_MISSES = Counter('cache_misses_total', 'Total cache misses')

# Load model
print("Loading ResNet18 model...")
model = models.resnet18(pretrained=True)
model.eval()
print("Model loaded successfully!")

# Redis connection
cache = redis.Redis(
    host=os.environ.get('REDIS_HOST', 'redis'),
    port=6379,
    decode_responses=True
)

# Database connection pool
def get_db():
    return psycopg2.connect(
        host=os.environ.get('DB_HOST', 'db'),
        database=os.environ.get('DB_NAME', 'mlapp'),
        user=os.environ.get('DB_USER', 'postgres'),
        password=os.environ.get('DB_PASSWORD', 'postgres')
    )

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'resnet18',
        'timestamp': datetime.utcnow().isoformat()
    }), 200

@app.route('/ready')
def ready():
    """Readiness check - verify all dependencies"""
    health_status = {'status': 'ready', 'checks': {}}

    try:
        cache.ping()
        health_status['checks']['redis'] = 'ok'
    except Exception as e:
        health_status['checks']['redis'] = f'error: {str(e)}'
        health_status['status'] = 'not_ready'

    try:
        conn = get_db()
        conn.close()
        health_status['checks']['database'] = 'ok'
    except Exception as e:
        health_status['checks']['database'] = f'error: {str(e)}'
        health_status['status'] = 'not_ready'

    status_code = 200 if health_status['status'] == 'ready' else 503
    return jsonify(health_status), status_code

@app.route('/predict', methods=['POST'])
def predict():
    """Image classification prediction"""
    REQUEST_COUNT.inc()
    start_time = time.time()

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        img_bytes = file.read()
        img_hash = hashlib.md5(img_bytes).hexdigest()
        cache_key = f'prediction:{img_hash}'

        # Check cache
        cached = cache.get(cache_key)
        if cached:
            CACHE_HITS.inc()
            result = json.loads(cached)
            result['cached'] = True
            PREDICTION_DURATION.observe(time.time() - start_time)
            return jsonify(result), 200

        CACHE_MISSES.inc()

        # Load and preprocess image
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top5_prob, top5_idx = torch.topk(probabilities, 5)

        predictions = []
        for i in range(5):
            predictions.append({
                'class_id': int(top5_idx[i]),
                'confidence': float(top5_prob[i])
            })

        # Log to database
        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO predictions
                (image_hash, top_class, confidence, all_predictions, created_at)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    img_hash,
                    predictions[0]['class_id'],
                    predictions[0]['confidence'],
                    json.dumps(predictions),
                    datetime.utcnow()
                )
            )
            prediction_id = cur.fetchone()[0]
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Database error: {e}")
            prediction_id = None

        result = {
            'prediction_id': prediction_id,
            'image_hash': img_hash,
            'predictions': predictions,
            'cached': False
        }

        # Cache for 1 hour
        cache.setex(cache_key, 3600, json.dumps(result))

        PREDICTION_DURATION.observe(time.time() - start_time)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(REGISTRY), 200, {'Content-Type': 'text/plain; charset=utf-8'}

@app.route('/stats')
def stats():
    """Application statistics"""
    try:
        conn = get_db()
        cur = conn.cursor()

        # Total predictions
        cur.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = cur.fetchone()[0]

        # Recent predictions (last hour)
        cur.execute("""
            SELECT COUNT(*) FROM predictions
            WHERE created_at > NOW() - INTERVAL '1 hour'
        """)
        recent_predictions = cur.fetchone()[0]

        # Average confidence
        cur.execute('SELECT AVG(confidence) FROM predictions')
        avg_confidence = cur.fetchone()[0]

        cur.close()
        conn.close()

        return jsonify({
            'total_predictions': total_predictions,
            'predictions_last_hour': recent_predictions,
            'average_confidence': float(avg_confidence) if avg_confidence else 0
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
EOF
```

**Requirements**:

```bash
cat > requirements.txt << 'EOF'
flask==3.0.0
torch==2.1.0
torchvision==0.16.0
pillow==10.1.0
redis==5.0.1
psycopg2-binary==2.9.9
gunicorn==21.2.0
prometheus-client==0.19.0
EOF
```

**Dockerfile**:

```dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY serve.py .

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "--access-logfile", "-", "serve:app"]
EOF
```

**Database initialization**:

```bash
mkdir -p init-db
cat > init-db/01-schema.sql << 'EOF'
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    image_hash VARCHAR(32) NOT NULL,
    top_class INTEGER NOT NULL,
    confidence REAL NOT NULL,
    all_predictions JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_image_hash ON predictions(image_hash);
CREATE INDEX idx_created_at ON predictions(created_at DESC);
CREATE INDEX idx_confidence ON predictions(confidence DESC);

-- Create view for recent predictions
CREATE VIEW recent_predictions AS
SELECT
    id,
    image_hash,
    top_class,
    confidence,
    created_at
FROM predictions
WHERE created_at > NOW() - INTERVAL '24 hours'
ORDER BY created_at DESC;
EOF
```

**Nginx configuration**:

```bash
mkdir -p nginx
cat > nginx/nginx.conf << 'EOF'
upstream ml_api {
    least_conn;
    server api:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    client_max_body_size 10M;

    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://ml_api/health;
    }

    # Metrics endpoint (restrict in production!)
    location /metrics {
        proxy_pass http://ml_api/metrics;
    }

    # Prediction endpoint
    location /predict {
        proxy_pass http://ml_api/predict;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Timeout settings for ML inference
        proxy_connect_timeout 60s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
    }

    # Stats endpoint
    location /stats {
        proxy_pass http://ml_api/stats;
    }

    # Default location
    location / {
        return 404 '{"error": "Not found"}';
        add_header Content-Type application/json;
    }
}
EOF
```

**Prometheus configuration**:

```bash
cat > prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ml-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
EOF
```

**Complete Production Docker Compose**:

```yaml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  # Reverse Proxy / Load Balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/conf.d/default.conf:ro
    depends_on:
      - api
    restart: always
    networks:
      - frontend

  # ML API Service (scaled)
  api:
    build: .
    expose:
      - "8000"
    environment:
      - DB_HOST=db
      - DB_NAME=mlapp
      - DB_USER=postgres
      - DB_PASSWORD=${DB_PASSWORD:-postgres}
      - REDIS_HOST=redis
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: on-failure
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
    networks:
      - frontend
      - backend

  # PostgreSQL Database
  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=mlapp
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${DB_PASSWORD:-postgres}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-db:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: always
    networks:
      - backend

  # Redis Cache
  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru --appendonly yes
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
    restart: always
    networks:
      - backend

  # Prometheus Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
    restart: always
    networks:
      - backend
      - monitoring

  # Grafana Dashboards
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
      - GF_SERVER_ROOT_URL=http://localhost:3000
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    restart: always
    networks:
      - monitoring

  # PostgreSQL Exporter for Prometheus
  postgres-exporter:
    image: prometheuscommunity/postgres-exporter:latest
    environment:
      - DATA_SOURCE_NAME=postgresql://postgres:${DB_PASSWORD:-postgres}@db:5432/mlapp?sslmode=disable
    depends_on:
      db:
        condition: service_healthy
    restart: always
    networks:
      - backend
      - monitoring

  # Redis Exporter for Prometheus
  redis-exporter:
    image: oliver006/redis_exporter:latest
    environment:
      - REDIS_ADDR=redis:6379
    depends_on:
      redis:
        condition: service_healthy
    restart: always
    networks:
      - backend
      - monitoring

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true
  monitoring:
    driver: bridge
EOF
```

**Environment file**:

```bash
cat > .env << 'EOF'
# Database
DB_PASSWORD=secure-db-password-change-me

# Grafana
GRAFANA_PASSWORD=admin-change-me

# Application
ENVIRONMENT=production
EOF

cat > .env.example << 'EOF'
# Database
DB_PASSWORD=your-secure-password

# Grafana
GRAFANA_PASSWORD=your-grafana-password

# Application
ENVIRONMENT=production
EOF
```

**Production deployment script**:

```bash
cat > deploy.sh << 'EOF'
#!/bin/bash
set -e

echo "=== ML Production Stack Deployment ==="

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Copy .env.example to .env and update with your settings"
    exit 1
fi

# Pull latest images
echo "Pulling latest images..."
docker compose pull

# Build API service
echo "Building ML API..."
docker compose build api

# Start services
echo "Starting services..."
docker compose up -d

# Wait for services to be healthy
echo "Waiting for services to be healthy..."
sleep 30

# Check service health
echo "Checking service health..."
docker compose ps

# Test endpoints
echo "Testing endpoints..."
curl -f http://localhost/health || echo "Health check failed!"

echo "=== Deployment Complete ==="
echo "Services running:"
echo "  - ML API: http://localhost/"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana: http://localhost:3000"
echo ""
echo "View logs: docker compose logs -f"
echo "Scale API: docker compose up -d --scale api=4"
EOF

chmod +x deploy.sh
```

**Test the production stack**:

```bash
# Deploy
./deploy.sh

# Check all services
docker compose ps

# Test health
curl http://localhost/health

# Download a test image
curl -o test.jpg https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/300px-Cat03.jpg

# Make prediction
curl -X POST http://localhost/predict \
  -F "file=@test.jpg"

# Check stats
curl http://localhost/stats

# View metrics
curl http://localhost/metrics

# Access Grafana
echo "Grafana: http://localhost:3000 (admin/admin-change-me)"

# Access Prometheus
echo "Prometheus: http://localhost:9090"

# View logs
docker compose logs -f api

# Scale API
docker compose up -d --scale api=4

# Cleanup
docker compose down -v
```

### 7.2 GPU Support in Compose

For GPU-accelerated ML workloads:

```yaml
services:
  ml-gpu:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1  # Number of GPUs
              capabilities: [gpu]
```

Or for specific GPU:

```yaml
services:
  ml-gpu:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']  # Use GPU 0
              capabilities: [gpu]
```

---

## 8. Advanced Patterns

### 8.1 Using Compose Profiles

Profiles let you selectively start services:

```yaml
version: '3.8'

services:
  api:
    build: .
    # Always runs

  db:
    image: postgres:15
    # Always runs

  debug-tools:
    image: nicolaka/netshoot
    profiles: ["debug"]
    command: sleep infinity

  load-tester:
    image: williamyeh/wrk
    profiles: ["testing"]
    command: wrk -t 4 -c 100 -d 30s http://api:8000

  monitoring:
    image: prom/prometheus
    profiles: ["monitoring"]
```

**Usage**:

```bash
# Start only default services (api, db)
docker compose up -d

# Start with debug tools
docker compose --profile debug up -d

# Start with monitoring
docker compose --profile monitoring up -d

# Start with multiple profiles
docker compose --profile debug --profile monitoring up -d
```

### 8.2 Extension Fields (DRY principle)

Reduce duplication using YAML anchors:

```yaml
version: '3.8'

x-common-variables: &common-vars
  DB_HOST: db
  REDIS_HOST: redis
  ENVIRONMENT: production

x-resource-limits: &resource-limits
  resources:
    limits:
      cpus: '1.0'
      memory: 1G

services:
  api-v1:
    build: ./api-v1
    environment:
      <<: *common-vars
      API_VERSION: v1
    deploy:
      <<: *resource-limits

  api-v2:
    build: ./api-v2
    environment:
      <<: *common-vars
      API_VERSION: v2
    deploy:
      <<: *resource-limits
```

### 8.3 Init Containers Pattern

Run initialization tasks before main services:

```yaml
services:
  init-db:
    image: postgres:15
    volumes:
      - ./migrations:/migrations
    command: psql -h db -U postgres -f /migrations/schema.sql
    depends_on:
      db:
        condition: service_healthy
    restart: "no"  # Run once and exit

  api:
    build: .
    depends_on:
      init-db:
        condition: service_completed_successfully
```

### 8.4 Sidecar Pattern

Run helper containers alongside main service:

```yaml
services:
  app:
    build: .
    volumes:
      - shared-logs:/var/log

  log-shipper:
    image: fluent/fluentd
    volumes:
      - shared-logs:/var/log:ro
      - ./fluentd.conf:/fluentd/etc/fluent.conf
    depends_on:
      - app

volumes:
  shared-logs:
```

---

## 9. Troubleshooting

### 9.1 Common Issues and Solutions

**Issue: Service won't start**

```bash
# Check logs
docker compose logs service_name

# Check service status
docker compose ps

# Inspect service details
docker compose exec service_name env
docker compose exec service_name cat /proc/1/status
```

**Issue: Can't connect to database**

```bash
# Check if db is healthy
docker compose ps

# Test connectivity from app container
docker compose exec api ping db
docker compose exec api nc -zv db 5432

# Check database logs
docker compose logs db

# Verify environment variables
docker compose exec api env | grep DB_
```

**Issue: Volume permissions**

```bash
# Check volume permissions
docker compose exec service_name ls -la /path/to/volume

# Fix permissions (run as root)
docker compose exec -u root service_name chown -R appuser:appuser /path
```

**Issue: Out of memory**

```bash
# Check container stats
docker stats

# Check logs for OOM errors
docker compose logs | grep -i "out of memory"

# Increase memory limits
# In docker-compose.yml:
services:
  app:
    deploy:
      resources:
        limits:
          memory: 2G
```

### 9.2 Debugging Techniques

**Interactive debugging**:

```bash
# Run bash in running container
docker compose exec api bash

# Run new container with same image
docker compose run --rm api bash

# Override command
docker compose run --rm api python -c "import torch; print(torch.cuda.is_available())"
```

**Network debugging**:

```bash
# List networks
docker network ls

# Inspect network
docker network inspect project_network_name

# Test DNS resolution
docker compose exec api nslookup db
docker compose exec api ping -c 2 redis
```

**View resolved configuration**:

```bash
# See final configuration with variable substitution
docker compose config

# Validate without starting
docker compose config --quiet

# Show service names only
docker compose config --services
```

**Resource monitoring**:

```bash
# Real-time stats
docker stats

# Specific services
docker stats $(docker compose ps -q)

# Export to file
docker stats --no-stream > stats.txt
```

### 9.3 Debugging Checklist

```bash
# 1. Validate compose file
docker compose config

# 2. Check service status
docker compose ps

# 3. View logs
docker compose logs --tail=50

# 4. Check network connectivity
docker compose exec api ping db

# 5. Verify environment variables
docker compose exec api env

# 6. Check disk space
docker system df

# 7. Check resource usage
docker stats

# 8. Inspect volumes
docker volume ls
docker volume inspect volume_name

# 9. Test endpoints
curl http://localhost:8000/health

# 10. Check for port conflicts
netstat -tuln | grep LISTEN
```

---

## 10. Best Practices

### 10.1 Compose File Organization

**Use .env for configuration**:
```bash
# .env (not committed)
DB_PASSWORD=secret123
API_KEY=abc123

# .env.example (committed)
DB_PASSWORD=changeme
API_KEY=changeme
```

**Use override files for environments**:
```bash
# Development
docker-compose.yml + docker-compose.override.yml

# Production
docker-compose.yml + docker-compose.prod.yml

# Testing
docker-compose.yml + docker-compose.test.yml
```

**Structure for large projects**:
```
project/
├── docker-compose.yml           # Base configuration
├── docker-compose.override.yml  # Development overrides
├── docker-compose.prod.yml      # Production overrides
├── .env.example                 # Example environment
├── services/
│   ├── api/
│   │   ├── Dockerfile
│   │   └── app.py
│   ├── worker/
│   │   ├── Dockerfile
│   │   └── worker.py
│   └── nginx/
│       └── nginx.conf
├── init/
│   └── db/
│       └── schema.sql
└── scripts/
    ├── deploy.sh
    └── backup.sh
```

### 10.2 Security Best Practices

**1. Never hardcode secrets**:

```yaml
# BAD
environment:
  - DB_PASSWORD=secret123

# GOOD
environment:
  - DB_PASSWORD=${DB_PASSWORD}
secrets:
  - db_password
```

**2. Use secrets for sensitive data**:

```yaml
secrets:
  db_password:
    file: ./secrets/db_password.txt
  api_key:
    file: ./secrets/api_key.txt
```

**3. Run containers as non-root**:

```dockerfile
# In Dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

```yaml
# In docker-compose.yml
services:
  api:
    user: "1000:1000"
```

**4. Use read-only volumes where possible**:

```yaml
volumes:
  - ./config:/app/config:ro
  - ./nginx.conf:/etc/nginx/nginx.conf:ro
```

**5. Limit network exposure**:

```yaml
networks:
  frontend:  # Exposed to internet
  backend:
    internal: true  # No internet access
```

**6. Set resource limits**:

```yaml
deploy:
  resources:
    limits:
      cpus: '1.0'
      memory: 1G
```

### 10.3 Performance Best Practices

**1. Use build cache effectively**:

```dockerfile
# Copy dependencies first (cached layer)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code last (changes frequently)
COPY . .
```

**2. Optimize image size**:

```dockerfile
# Use slim/alpine base images
FROM python:3.11-slim

# Clean up in same layer
RUN apt-get update && \
    apt-get install -y package && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
```

**3. Use volumes for development**:

```yaml
# Development
services:
  web:
    volumes:
      - .:/app  # Hot reload

# Production
services:
  web:
    # No volume mount - code baked into image
```

**4. Healthchecks for reliability**:

```yaml
healthcheck:
  test: ["CMD-SHELL", "curl -f http://localhost/health || exit 1"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### 10.4 Production Deployment Checklist

```yaml
# Production-ready compose file
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.prod

    # ✓ Resource limits
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 1G

    # ✓ Health checks
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3

    # ✓ Restart policy
    restart: always

    # ✓ Logging configuration
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

    # ✓ Environment from secrets
    environment:
      - DB_PASSWORD_FILE=/run/secrets/db_password
    secrets:
      - db_password

    # ✓ Proper dependencies
    depends_on:
      db:
        condition: service_healthy

    # ✓ Network isolation
    networks:
      - backend

secrets:
  db_password:
    file: ./secrets/db_password.txt

networks:
  backend:
    driver: bridge
```

### 10.5 Monitoring and Logging

**Centralized logging**:

```yaml
services:
  api:
    logging:
      driver: "fluentd"
      options:
        fluentd-address: localhost:24224
        tag: api.logs

  fluentd:
    image: fluent/fluentd:latest
    ports:
      - "24224:24224"
    volumes:
      - ./fluentd.conf:/fluentd/etc/fluent.conf
```

**Application metrics**:

```yaml
services:
  api:
    # Expose metrics endpoint

  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
```

---

## Conclusion

You've now mastered Docker Compose for ML applications! This guide covered:

1. **Compose Basics**: YAML syntax, service definitions, essential commands
2. **Multi-Container ML Apps**: API + Database + Cache architectures
3. **Dependencies & Health**: Startup order, health checks, readiness probes
4. **Networks & Volumes**: Isolation, persistence, data management
5. **Environment Management**: .env files, overrides, secrets
6. **Scaling**: Replicas, load balancing, horizontal scaling
7. **Production ML Stack**: Complete production-ready ML serving infrastructure
8. **Advanced Patterns**: Profiles, DRY principle, init containers
9. **Troubleshooting**: Debugging techniques, common issues
10. **Best Practices**: Security, performance, deployment checklist

### Next Steps

1. **Practice**: Deploy the production ML stack
2. **Experiment**: Try different scaling strategies
3. **Customize**: Add your own ML models and services
4. **Monitor**: Set up Prometheus and Grafana dashboards
5. **Optimize**: Tune resource limits and caching strategies

### Additional Resources

- Docker Compose Documentation: https://docs.docker.com/compose/
- Docker Compose File Reference: https://docs.docker.com/compose/compose-file/
- Best Practices: https://docs.docker.com/develop/dev-best-practices/

---

**Version**: 1.0
**Last Updated**: November 2025
**Author**: AI Infrastructure Team
