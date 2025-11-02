# Implementation Guide: Docker Volume Management for ML Infrastructure

## Overview

This comprehensive implementation guide covers Docker volume management with a focus on ML workloads. You'll learn to manage persistent storage, implement backup strategies, optimize performance, and design production-ready storage architectures for machine learning applications.

**Target Audience**: Junior AI Infrastructure Engineers
**Prerequisites**: Docker fundamentals, basic Linux file systems
**Estimated Time**: 4-6 hours for complete implementation
**ML Focus Areas**: Dataset storage, model checkpoints, experiment tracking, distributed training

---

## Table of Contents

1. [Understanding Volume Types](#1-understanding-volume-types)
2. [Volume Lifecycle Management](#2-volume-lifecycle-management)
3. [Data Persistence Patterns](#3-data-persistence-patterns)
4. [Backup and Restore Strategies](#4-backup-and-restore-strategies)
5. [Performance Considerations](#5-performance-considerations)
6. [Shared Volumes for Distributed Systems](#6-shared-volumes-for-distributed-systems)
7. [Production ML Data Management](#7-production-ml-data-management)
8. [Advanced Topics](#8-advanced-topics)
9. [Troubleshooting Guide](#9-troubleshooting-guide)
10. [Best Practices Checklist](#10-best-practices-checklist)

---

## 1. Understanding Volume Types

### 1.1 Named Volumes - Production Data Storage

Named volumes are Docker-managed storage units that persist independently of containers. They're the recommended approach for production data.

**Architecture Overview**:
```
Host Filesystem: /var/lib/docker/volumes/
    └── my-volume/
        └── _data/          <- Actual storage location
            └── [your files]

Container Filesystem: /app/data
    └── [mounted to _data above]
```

**Implementation Example**:

```bash
# Create a named volume
docker volume create ml-models

# Inspect the volume to see metadata
docker volume inspect ml-models
```

**Expected Output**:
```json
[
    {
        "CreatedAt": "2025-11-02T10:00:00Z",
        "Driver": "local",
        "Labels": {},
        "Mountpoint": "/var/lib/docker/volumes/ml-models/_data",
        "Name": "ml-models",
        "Options": {},
        "Scope": "local"
    }
]
```

**Using Named Volumes**:

```bash
# Run container with named volume
docker run -d \
  --name model-trainer \
  -v ml-models:/models \
  pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime \
  python train.py

# Data persists after container removal
docker stop model-trainer
docker rm model-trainer

# Access same data in new container
docker run --rm \
  -v ml-models:/models \
  alpine ls -lh /models
```

**When to Use Named Volumes**:
- Production databases (PostgreSQL, MongoDB)
- ML model storage and versioning
- Application state that must survive restarts
- Shared data between multiple containers
- Data requiring backup/restore procedures

**Key Benefits**:
- Docker manages storage location
- Easy backup and migration
- Works across different host systems
- Automatic cleanup with `docker volume prune`
- Better isolation than bind mounts

---

### 1.2 Bind Mounts - Development and Host Integration

Bind mounts map a host directory directly into a container. They provide direct file system access but require careful path management.

**Architecture Overview**:
```
Host: /home/user/ml-project/
    └── src/
        └── train.py

Container: /workspace/
    └── src/            <- Direct mount to host path
        └── train.py    <- Same file, changes reflected immediately
```

**Implementation Example**:

```bash
# Setup project structure
mkdir -p ~/ml-project/{src,data,models,configs}

# Create sample training script
cat > ~/ml-project/src/train.py << 'EOF'
import json
import time
from pathlib import Path

def train():
    print("Starting training...")
    config_path = Path("/configs/model.json")

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"Loaded config: {config}")

    # Simulate training
    for epoch in range(5):
        print(f"Epoch {epoch + 1}/5")
        time.sleep(1)

    # Save model
    model_path = Path("/models/trained_model.pth")
    model_path.write_text("model_weights_here")
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()
EOF

# Create configuration file
cat > ~/ml-project/configs/model.json << 'EOF'
{
    "model": "resnet50",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 5
}
EOF

# Run with bind mounts
docker run --rm \
  -v ~/ml-project/src:/workspace/src:ro \
  -v ~/ml-project/configs:/configs:ro \
  -v ~/ml-project/models:/models \
  -w /workspace/src \
  python:3.11 \
  python train.py
```

**Permission Management**:

```bash
# Check current user ID
echo "User ID: $(id -u)"
echo "Group ID: $(id -g)"

# Run container as specific user to avoid permission issues
docker run --rm \
  -u $(id -u):$(id -g) \
  -v ~/ml-project/data:/data \
  alpine sh -c "echo 'test' > /data/test.txt"

# Verify file ownership on host
ls -l ~/ml-project/data/test.txt
```

**When to Use Bind Mounts**:
- Development with hot code reloading
- Host-managed configuration files
- Direct access to host-generated data
- Integration with existing file systems
- Debugging and log analysis

**Security Considerations**:
```bash
# Read-only mount for security
-v ~/configs:/configs:ro

# Never bind mount entire root!
# BAD: -v /:/host
# This exposes entire host filesystem
```

---

### 1.3 tmpfs Mounts - High-Performance Ephemeral Storage

tmpfs mounts use RAM instead of disk, providing ultra-fast I/O for temporary data that doesn't need persistence.

**Architecture Overview**:
```
Container Memory:
    └── /tmp (tmpfs)    <- Stored in RAM
        └── cache/      <- Lost on container stop
            └── temp.data

No persistence to disk!
```

**Implementation Example**:

```bash
# Create tmpfs mount with size limit
docker run -d \
  --name ml-inference \
  --tmpfs /cache:rw,size=1g,mode=1777 \
  --tmpfs /tmp:rw,size=500m \
  -v ml-models:/models:ro \
  pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime \
  python inference_server.py

# Verify tmpfs mounts
docker exec ml-inference df -h /cache /tmp

# Check mount details
docker exec ml-inference mount | grep tmpfs
```

**Performance Comparison Script**:

```bash
cat > test_storage_performance.sh << 'EOF'
#!/bin/bash

echo "=== Storage Performance Comparison ==="
echo

# Test 1: Regular volume
echo "1. Testing Named Volume (disk-based):"
docker volume create test-disk
time docker run --rm \
  -v test-disk:/data \
  alpine dd if=/dev/zero of=/data/test bs=1M count=100 oflag=direct 2>&1 | grep copied
docker volume rm test-disk

echo

# Test 2: tmpfs
echo "2. Testing tmpfs (RAM-based):"
time docker run --rm \
  --tmpfs /data:rw,size=200m \
  alpine dd if=/dev/zero of=/data/test bs=1M count=100 2>&1 | grep copied

echo

# Test 3: Random read/write patterns
echo "3. Testing Random I/O:"
echo "   Volume (disk):"
docker volume create test-random
docker run --rm \
  -v test-random:/data \
  alpine sh -c 'time dd if=/dev/urandom of=/data/random bs=1M count=50' 2>&1 | grep real
docker volume rm test-random

echo "   tmpfs (RAM):"
docker run --rm \
  --tmpfs /data:size=100m \
  alpine sh -c 'time dd if=/dev/urandom of=/data/random bs=1M count=50' 2>&1 | grep real
EOF

chmod +x test_storage_performance.sh
./test_storage_performance.sh
```

**ML Inference Server Example**:

```python
# inference_server.py
import os
from pathlib import Path
import torch
import hashlib

CACHE_DIR = Path("/cache")
MODEL_DIR = Path("/models")

def get_cached_result(input_hash):
    """Check tmpfs cache for previous inference result"""
    cache_file = CACHE_DIR / f"{input_hash}.pt"
    if cache_file.exists():
        print(f"Cache hit: {input_hash}")
        return torch.load(cache_file)
    return None

def cache_result(input_hash, result):
    """Store result in tmpfs cache"""
    cache_file = CACHE_DIR / f"{input_hash}.pt"
    torch.save(result, cache_file)
    print(f"Cached: {input_hash}")

def inference(input_data):
    # Generate hash for caching
    input_hash = hashlib.md5(str(input_data).encode()).hexdigest()

    # Check cache first
    cached = get_cached_result(input_hash)
    if cached is not None:
        return cached

    # Perform inference
    # ... model inference code ...
    result = {"prediction": "example"}

    # Cache result
    cache_result(input_hash, result)
    return result
```

**When to Use tmpfs**:
- Inference caching (temporary predictions)
- Intermediate training artifacts
- Temporary decompressed data
- Session storage
- Security-sensitive data (not written to disk)

**Size Planning**:
```bash
# Calculate available RAM
free -h

# Set tmpfs to reasonable percentage (e.g., 10% of RAM)
# For 32GB RAM: 3.2GB tmpfs
docker run --tmpfs /cache:size=3200m ...
```

---

### 1.4 Volume Type Comparison and Selection Guide

**Feature Matrix**:

| Feature | Named Volume | Bind Mount | tmpfs |
|---------|-------------|------------|-------|
| Persistence | Yes (survives restarts) | Yes (host-managed) | No (RAM only) |
| Performance | Good (disk) | Good (disk) | Excellent (RAM) |
| Portability | Excellent | Poor (path-dependent) | Excellent |
| Security | Good (isolated) | Risk (host exposure) | Excellent (ephemeral) |
| Backup | Easy | Manual | N/A |
| Multi-host | Plugin-dependent | No | N/A |
| Size Limit | Disk size | Disk size | RAM size |
| Use in Production | Yes | Limited | Yes (cache only) |

**Decision Tree**:

```
Need data persistence?
├── No → Use tmpfs
│   └── Examples: cache, temp files, sessions
│
└── Yes → Need host access?
    ├── Yes → Use bind mount
    │   └── Examples: development, host configs
    │
    └── No → Use named volume
        └── Examples: databases, models, app state
```

**ML Workload Recommendations**:

```yaml
# Recommended volume strategy for ML pipeline
version: '3.8'

services:
  data-prep:
    image: ml-pipeline/prep:latest
    volumes:
      # Raw data (read-only, maybe from host)
      - ./raw-data:/data/raw:ro

      # Processed output (named volume for sharing)
      - processed-data:/data/processed

      # Temp processing (tmpfs for speed)
      - type: tmpfs
        target: /tmp
        tmpfs:
          size: 2g

  training:
    image: ml-pipeline/train:latest
    volumes:
      # Input data (read-only)
      - processed-data:/data:ro

      # Models (persistent, backed up)
      - models:/models

      # Checkpoints (persistent, fast storage)
      - checkpoints:/checkpoints

      # TensorBoard logs (persistent)
      - tensorboard-logs:/logs

      # Training cache (tmpfs)
      - type: tmpfs
        target: /cache
        tmpfs:
          size: 4g

  inference:
    image: ml-pipeline/serve:latest
    volumes:
      # Models (read-only)
      - models:/models:ro

      # Inference cache (tmpfs for speed)
      - type: tmpfs
        target: /cache
        tmpfs:
          size: 1g

volumes:
  processed-data:
    driver: local
  models:
    driver: local
    labels:
      backup: "daily"
      retention: "90d"
  checkpoints:
    driver: local
    labels:
      backup: "hourly"
      retention: "7d"
  tensorboard-logs:
    driver: local
```

---

## 2. Volume Lifecycle Management

### 2.1 Volume Creation and Configuration

**Basic Creation**:

```bash
# Simple named volume
docker volume create my-data

# Volume with labels for organization
docker volume create \
  --label environment=production \
  --label application=ml-training \
  --label team=data-science \
  --label cost-center=ml-ops \
  ml-training-data

# Volume with custom driver options
docker volume create \
  --driver local \
  --opt type=tmpfs \
  --opt device=tmpfs \
  --opt o=size=2g,uid=1000,gid=1000 \
  fast-cache

# Volume pointing to specific directory
docker volume create \
  --driver local \
  --opt type=none \
  --opt device=/mnt/ssd/ml-data \
  --opt o=bind \
  ssd-ml-storage
```

**Label-Based Management**:

```bash
# Create multiple volumes with consistent labeling
for env in dev staging prod; do
  docker volume create \
    --label environment=${env} \
    --label application=ml-platform \
    db-data-${env}
done

# List volumes by environment
docker volume ls --filter "label=environment=production"

# List volumes by application
docker volume ls --filter "label=application=ml-platform"

# Format output for better readability
docker volume ls \
  --filter "label=environment=production" \
  --format "table {{.Name}}\t{{.Driver}}\t{{.Labels}}"
```

---

### 2.2 Volume Inspection and Monitoring

**Detailed Inspection**:

```bash
# Full volume information
docker volume inspect ml-models

# Extract specific fields
docker volume inspect ml-models --format '{{.Mountpoint}}'
docker volume inspect ml-models --format '{{.Driver}}'
docker volume inspect ml-models --format '{{json .Labels}}' | jq

# Check which containers are using a volume
docker ps -a --filter volume=ml-models \
  --format "table {{.Names}}\t{{.Status}}\t{{.Mounts}}"
```

**Volume Usage Monitoring Script**:

```bash
cat > monitor_volumes.sh << 'EOF'
#!/bin/bash

echo "=== Docker Volume Monitoring Report ==="
echo "Generated: $(date)"
echo

# Overall system usage
echo "1. SYSTEM OVERVIEW"
echo "=================="
docker system df -v | grep -A 5 "Local Volumes"
echo

# List all volumes with details
echo "2. VOLUME INVENTORY"
echo "==================="
docker volume ls --format "table {{.Name}}\t{{.Driver}}\t{{.Scope}}"
echo

# Volume sizes
echo "3. VOLUME SIZES"
echo "==============="
for vol in $(docker volume ls -q); do
    # Get size using a lightweight alpine container
    size=$(docker run --rm -v $vol:/data alpine du -sh /data 2>/dev/null | awk '{print $1}')

    # Get labels
    labels=$(docker volume inspect $vol --format '{{.Labels}}' 2>/dev/null)

    # Get mountpoint
    mountpoint=$(docker volume inspect $vol --format '{{.Mountpoint}}' 2>/dev/null)

    echo "Volume: $vol"
    echo "  Size: $size"
    echo "  Labels: $labels"
    echo "  Location: $mountpoint"
    echo
done

# Dangling volumes
echo "4. DANGLING VOLUMES (Not attached to containers)"
echo "=================================================="
docker volume ls -qf dangling=true
echo

# Volumes by labels
echo "5. PRODUCTION VOLUMES"
echo "====================="
docker volume ls --filter "label=environment=production"
echo

# Usage warnings
echo "6. WARNINGS"
echo "==========="
dangling_count=$(docker volume ls -qf dangling=true | wc -l)
if [ $dangling_count -gt 0 ]; then
    echo "WARNING: $dangling_count dangling volume(s) found"
    echo "Run 'docker volume prune' to clean up"
fi

# Check for volumes over 10GB
echo
echo "7. LARGE VOLUMES (>10GB)"
echo "========================"
for vol in $(docker volume ls -q); do
    size_mb=$(docker run --rm -v $vol:/data alpine du -sm /data 2>/dev/null | awk '{print $1}')
    if [ "$size_mb" -gt 10240 ]; then
        size_gb=$(echo "scale=2; $size_mb / 1024" | bc)
        echo "$vol: ${size_gb}GB"
    fi
done

echo
echo "=== End of Report ==="
EOF

chmod +x monitor_volumes.sh
./monitor_volumes.sh
```

**Automated Monitoring with cron**:

```bash
# Add to crontab for daily monitoring
# Run daily at 2 AM and save to log
# 0 2 * * * /path/to/monitor_volumes.sh >> /var/log/docker-volumes.log 2>&1

# Create alerting wrapper
cat > monitor_with_alerts.sh << 'EOF'
#!/bin/bash

OUTPUT=$(bash monitor_volumes.sh)
echo "$OUTPUT"

# Alert if dangling volumes exceed threshold
DANGLING=$(echo "$OUTPUT" | grep "WARNING:" | grep -oP '\d+')
if [ ! -z "$DANGLING" ] && [ "$DANGLING" -gt 5 ]; then
    # Send alert (example using mail)
    echo "$OUTPUT" | mail -s "Docker Volume Alert: $DANGLING dangling volumes" admin@example.com
fi

# Alert if any volume exceeds size threshold (50GB)
LARGE_VOLS=$(echo "$OUTPUT" | grep -A 100 "LARGE VOLUMES" | grep "GB" | wc -l)
if [ "$LARGE_VOLS" -gt 0 ]; then
    echo "Large volume alert sent"
    # Send notification
fi
EOF

chmod +x monitor_with_alerts.sh
```

---

### 2.3 Volume Cleanup and Maintenance

**Safe Cleanup Procedures**:

```bash
# List dangling volumes (not attached to any container)
docker volume ls -qf dangling=true

# Remove specific volume
docker volume rm my-old-volume

# Remove dangling volumes (safe - only unattached volumes)
docker volume prune

# Remove all unused volumes (CAUTION!)
docker volume prune -a

# Force remove volume (even if in use - DANGEROUS)
docker volume rm -f volume-name
```

**Selective Cleanup Script**:

```bash
cat > cleanup_volumes.sh << 'EOF'
#!/bin/bash

# Configuration
RETENTION_DAYS=30
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --retention)
            RETENTION_DAYS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== Docker Volume Cleanup ==="
echo "Retention period: $RETENTION_DAYS days"
echo "Dry run: $DRY_RUN"
echo

# Find old volumes
echo "Analyzing volumes..."
for vol in $(docker volume ls -q); do
    # Get creation time
    created=$(docker volume inspect $vol --format '{{.CreatedAt}}')
    created_ts=$(date -d "$created" +%s 2>/dev/null || echo "0")
    current_ts=$(date +%s)
    age_days=$(( ($current_ts - $created_ts) / 86400 ))

    # Check if volume is in use
    in_use=$(docker ps -a --filter volume=$vol --format '{{.Names}}' | wc -l)

    # Get labels
    environment=$(docker volume inspect $vol --format '{{index .Labels "environment"}}' 2>/dev/null)
    backup_flag=$(docker volume inspect $vol --format '{{index .Labels "backup"}}' 2>/dev/null)

    # Decision logic
    should_remove=false
    reason=""

    if [ $age_days -gt $RETENTION_DAYS ] && [ $in_use -eq 0 ]; then
        if [ "$environment" != "production" ] && [ "$backup_flag" != "true" ]; then
            should_remove=true
            reason="Age: ${age_days}d, Unused, Non-production"
        fi
    fi

    if [ "$should_remove" = true ]; then
        echo "Would remove: $vol ($reason)"
        if [ "$DRY_RUN" = false ]; then
            echo "  Removing..."
            docker volume rm $vol
        fi
    fi
done

echo
echo "Cleanup complete!"
EOF

chmod +x cleanup_volumes.sh

# Run in dry-run mode first
./cleanup_volumes.sh --dry-run --retention 30

# Actually remove volumes
./cleanup_volumes.sh --retention 30
```

**Production-Safe Cleanup**:

```bash
cat > production_cleanup.sh << 'EOF'
#!/bin/bash

echo "=== Production-Safe Volume Cleanup ==="

# Only remove volumes that meet ALL criteria:
# 1. Labeled as environment=dev or environment=test
# 2. Not attached to any container
# 3. Older than 7 days

for vol in $(docker volume ls -q); do
    env=$(docker volume inspect $vol --format '{{index .Labels "environment"}}' 2>/dev/null)

    if [ "$env" = "dev" ] || [ "$env" = "test" ]; then
        in_use=$(docker ps -a --filter volume=$vol -q | wc -l)

        if [ $in_use -eq 0 ]; then
            created=$(docker volume inspect $vol --format '{{.CreatedAt}}')
            created_ts=$(date -d "$created" +%s)
            age_days=$(( ($(date +%s) - $created_ts) / 86400 ))

            if [ $age_days -gt 7 ]; then
                echo "Removing: $vol (env=$env, age=${age_days}d)"
                docker volume rm $vol
            fi
        fi
    fi
done

# Separately handle dangling volumes from dev/test
echo
echo "Removing dangling volumes..."
docker volume prune -f --filter "label=environment=dev"
docker volume prune -f --filter "label=environment=test"

echo "Cleanup complete!"
EOF

chmod +x production_cleanup.sh
```

---

## 3. Data Persistence Patterns

### 3.1 Database Persistence

**PostgreSQL with Persistent Storage**:

```yaml
# docker-compose-postgres.yml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    container_name: ml-postgres
    environment:
      POSTGRES_USER: mluser
      POSTGRES_PASSWORD: ${DB_PASSWORD:-changeme}
      POSTGRES_DB: mldb
      PGDATA: /var/lib/postgresql/data/pgdata
    volumes:
      # Main database data
      - postgres-data:/var/lib/postgresql/data

      # Initialization scripts
      - ./init-scripts:/docker-entrypoint-initdb.d:ro

      # Custom PostgreSQL configuration
      - ./postgres.conf:/etc/postgresql/postgresql.conf:ro

      # Backup location
      - ./backups:/backups
    command: postgres -c config_file=/etc/postgresql/postgresql.conf
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mluser -d mldb"]
      interval: 10s
      timeout: 5s
      retries: 5
    labels:
      - "backup.enable=true"
      - "backup.schedule=daily"

volumes:
  postgres-data:
    driver: local
    labels:
      environment: production
      service: database
      backup: daily
```

**Database Initialization**:

```bash
mkdir -p init-scripts backups

cat > init-scripts/01-schema.sql << 'EOF'
-- ML Experiment Tracking Schema
CREATE TABLE IF NOT EXISTS experiments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'running',
    config JSONB,
    metrics JSONB
);

CREATE TABLE IF NOT EXISTS models (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER REFERENCES experiments(id),
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50),
    path TEXT,
    metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS datasets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50),
    size_bytes BIGINT,
    path TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_experiments_status ON experiments(status);
CREATE INDEX idx_models_experiment ON models(experiment_id);
CREATE INDEX idx_datasets_name ON datasets(name);
EOF

cat > init-scripts/02-seed-data.sql << 'EOF'
-- Sample data
INSERT INTO experiments (name, config, metrics, status) VALUES
('baseline-resnet50',
 '{"model": "resnet50", "lr": 0.001, "batch_size": 32}'::jsonb,
 '{"accuracy": 0.92, "loss": 0.15}'::jsonb,
 'completed'),
('improved-resnet50',
 '{"model": "resnet50", "lr": 0.0005, "batch_size": 64}'::jsonb,
 '{"accuracy": 0.94, "loss": 0.12}'::jsonb,
 'completed');
EOF

# Custom PostgreSQL configuration for ML workloads
cat > postgres.conf << 'EOF'
# Optimized for ML workload
max_connections = 100
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
work_mem = 4MB
max_wal_size = 2GB
min_wal_size = 1GB

# Logging
logging_collector = on
log_directory = 'pg_log'
log_filename = 'postgresql-%Y-%m-%d.log'
log_statement = 'mod'
log_duration = on
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
EOF
```

**Start and Verify**:

```bash
# Set environment variable
export DB_PASSWORD="secure_password_123"

# Start PostgreSQL
docker compose -f docker-compose-postgres.yml up -d

# Wait for initialization
sleep 10

# Verify schema
docker compose -f docker-compose-postgres.yml exec postgres \
  psql -U mluser -d mldb -c "\dt"

# Query sample data
docker compose -f docker-compose-postgres.yml exec postgres \
  psql -U mluser -d mldb -c "SELECT * FROM experiments;"

# Check volume
docker volume inspect mod-005-docker-containers_postgres-data
```

---

### 3.2 Application State Persistence

**Stateful ML Application**:

```yaml
# docker-compose-ml-app.yml
version: '3.8'

services:
  # Model training service
  trainer:
    image: ml-trainer:latest
    volumes:
      # Training data (read-only)
      - training-data:/data:ro

      # Model checkpoints (persistent)
      - model-checkpoints:/checkpoints

      # Training logs
      - training-logs:/logs

      # Application state
      - app-state:/app/state
    environment:
      CHECKPOINT_DIR: /checkpoints
      LOG_DIR: /logs
      STATE_DIR: /app/state
    deploy:
      restart_policy:
        condition: on-failure
        max_attempts: 3

  # TensorBoard for monitoring
  tensorboard:
    image: tensorflow/tensorflow:latest
    command: tensorboard --logdir=/logs --host=0.0.0.0
    volumes:
      - training-logs:/logs:ro
    ports:
      - "6006:6006"

  # Model registry
  registry:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: model_registry
      POSTGRES_PASSWORD: ${REGISTRY_PASSWORD}
    volumes:
      - registry-data:/var/lib/postgresql/data
    labels:
      backup: "hourly"

volumes:
  training-data:
    driver: local
    driver_opts:
      type: none
      device: /mnt/nfs/training-data
      o: bind

  model-checkpoints:
    driver: local
    labels:
      backup: "realtime"
      retention: "30d"

  training-logs:
    driver: local
    labels:
      retention: "90d"

  app-state:
    driver: local
    labels:
      backup: "daily"

  registry-data:
    driver: local
    labels:
      backup: "hourly"
      critical: "true"
```

**State Management Example**:

```python
# state_manager.py
import json
from pathlib import Path
from datetime import datetime
import fcntl
import os

class TrainingStateManager:
    """Manage training state with file locking for concurrent access"""

    def __init__(self, state_dir="/app/state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "training_state.json"
        self.lock_file = self.state_dir / "training_state.lock"

    def save_state(self, state_dict):
        """Save training state atomically with file locking"""
        # Add metadata
        state_dict['last_updated'] = datetime.now().isoformat()
        state_dict['pid'] = os.getpid()

        # Write to temporary file first
        temp_file = self.state_file.with_suffix('.tmp')

        with open(self.lock_file, 'w') as lock:
            # Acquire exclusive lock
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

            try:
                # Write to temp file
                with open(temp_file, 'w') as f:
                    json.dump(state_dict, f, indent=2)

                # Atomic rename
                temp_file.replace(self.state_file)

                print(f"State saved: {state_dict.get('epoch', 'N/A')}")

            finally:
                # Release lock
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def load_state(self):
        """Load training state with file locking"""
        if not self.state_file.exists():
            return None

        with open(self.lock_file, 'w') as lock:
            # Acquire shared lock
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)

            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                return state

            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def clear_state(self):
        """Clear training state"""
        if self.state_file.exists():
            self.state_file.unlink()

# Example usage
if __name__ == "__main__":
    manager = TrainingStateManager()

    # Simulate training
    for epoch in range(10):
        state = {
            'epoch': epoch,
            'train_loss': 0.5 - (epoch * 0.03),
            'val_loss': 0.6 - (epoch * 0.025),
            'best_accuracy': 0.85 + (epoch * 0.01)
        }
        manager.save_state(state)
        print(f"Epoch {epoch} state saved")

    # Load state
    loaded = manager.load_state()
    print(f"Loaded state: {json.dumps(loaded, indent=2)}")
```

---

### 3.3 Configuration Management

**Configuration Volume Pattern**:

```yaml
# docker-compose-config.yml
version: '3.8'

services:
  app:
    image: ml-app:latest
    volumes:
      # Mounted configurations (read-only)
      - ./configs/app-config.yaml:/config/app.yaml:ro
      - ./configs/model-config.json:/config/model.json:ro
      - ./configs/logging.conf:/config/logging.conf:ro

      # Secret configurations (from Docker secrets)
      - type: tmpfs
        target: /run/secrets
        tmpfs:
          mode: 0400
    environment:
      CONFIG_DIR: /config
    command: python app.py --config /config/app.yaml

configs:
  app-config:
    file: ./configs/app-config.yaml
  model-config:
    file: ./configs/model-config.json
```

**Configuration Files**:

```bash
mkdir -p configs

cat > configs/app-config.yaml << 'EOF'
application:
  name: ml-training-service
  version: 1.0.0
  environment: production

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  checkpoint_interval: 10

data:
  train_path: /data/train
  val_path: /data/val
  test_path: /data/test

model:
  architecture: resnet50
  pretrained: true
  num_classes: 1000

logging:
  level: INFO
  format: json
  output: /logs/training.log
EOF

cat > configs/model-config.json << 'EOF'
{
  "model": {
    "type": "classification",
    "architecture": "resnet50",
    "input_size": [224, 224, 3],
    "num_classes": 1000
  },
  "optimizer": {
    "type": "adam",
    "lr": 0.001,
    "betas": [0.9, 0.999],
    "weight_decay": 0.0001
  },
  "scheduler": {
    "type": "cosine",
    "T_max": 100,
    "eta_min": 0.00001
  },
  "augmentation": {
    "random_crop": true,
    "random_flip": true,
    "normalize": {
      "mean": [0.485, 0.456, 0.406],
      "std": [0.229, 0.224, 0.225]
    }
  }
}
EOF
```

---

## 4. Backup and Restore Strategies

### 4.1 Volume Backup Procedures

**Manual Backup**:

```bash
# Backup single volume to tar archive
backup_volume() {
    local volume_name=$1
    local backup_dir=${2:-./backups}
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="${backup_dir}/${volume_name}_${timestamp}.tar.gz"

    mkdir -p "$backup_dir"

    echo "Backing up volume: $volume_name"
    docker run --rm \
        -v ${volume_name}:/source:ro \
        -v ${backup_dir}:/backup \
        alpine \
        tar czf /backup/$(basename $backup_file) -C /source .

    echo "Backup created: $backup_file"
    ls -lh "$backup_file"
}

# Usage
backup_volume "ml-models" "/backups/volumes"
backup_volume "postgres-data" "/backups/databases"
```

**Automated Backup System**:

```bash
cat > backup_system.sh << 'EOF'
#!/bin/bash

# Configuration
BACKUP_ROOT="/backups"
RETENTION_DAYS=7
LOG_FILE="/var/log/docker-backup.log"
S3_BUCKET="s3://my-ml-backups"  # Optional cloud storage

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Backup single volume
backup_volume() {
    local volume=$1
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_dir="${BACKUP_ROOT}/${volume}"
    local backup_file="${volume}_${timestamp}.tar.gz"

    mkdir -p "$backup_dir"

    log "Starting backup of volume: $volume"

    # Create backup with progress
    docker run --rm \
        -v ${volume}:/source:ro \
        -v ${backup_dir}:/backup \
        alpine \
        sh -c "tar czf /backup/${backup_file} -C /source . 2>&1" \
        >> "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        local size=$(du -h "${backup_dir}/${backup_file}" | cut -f1)
        log "SUCCESS: Backup completed - ${backup_file} (${size})"

        # Optional: Upload to S3
        if command -v aws &> /dev/null && [ ! -z "$S3_BUCKET" ]; then
            log "Uploading to S3: $S3_BUCKET/${volume}/${backup_file}"
            aws s3 cp "${backup_dir}/${backup_file}" "${S3_BUCKET}/${volume}/" \
                >> "$LOG_FILE" 2>&1
        fi

        return 0
    else
        log "ERROR: Backup failed for $volume"
        return 1
    fi
}

# Clean old backups
cleanup_old_backups() {
    local volume_dir=$1
    log "Cleaning backups older than $RETENTION_DAYS days in $volume_dir"

    find "$volume_dir" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete

    local removed=$(find "$volume_dir" -name "*.tar.gz" -mtime +$RETENTION_DAYS | wc -l)
    log "Removed $removed old backup(s)"
}

# Main backup process
main() {
    log "===== Backup Process Started ====="

    # Get list of volumes to backup (labeled with backup=true)
    VOLUMES=$(docker volume ls --filter "label=backup=daily" -q)

    if [ -z "$VOLUMES" ]; then
        log "WARNING: No volumes found with backup=daily label"
        # Fallback: backup all volumes except system ones
        VOLUMES=$(docker volume ls -q | grep -v "^[0-9a-f]\{64\}$")
    fi

    local total=0
    local success=0
    local failed=0

    for vol in $VOLUMES; do
        ((total++))

        if backup_volume "$vol"; then
            ((success++))
            cleanup_old_backups "${BACKUP_ROOT}/${vol}"
        else
            ((failed++))
        fi
    done

    log "===== Backup Process Completed ====="
    log "Total: $total, Success: $success, Failed: $failed"

    # Send alert if failures
    if [ $failed -gt 0 ]; then
        log "WARNING: $failed backup(s) failed. Check logs."
        # Send notification (email, Slack, etc.)
        # mail -s "Docker Backup Failed" admin@example.com < "$LOG_FILE"
    fi
}

# Run main process
main
EOF

chmod +x backup_system.sh
```

**Schedule Backups with systemd**:

```bash
# Create systemd service
sudo tee /etc/systemd/system/docker-backup.service << 'EOF'
[Unit]
Description=Docker Volume Backup Service
Wants=docker.service
After=docker.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/backup_system.sh
User=root
StandardOutput=journal
StandardError=journal
EOF

# Create systemd timer
sudo tee /etc/systemd/system/docker-backup.timer << 'EOF'
[Unit]
Description=Docker Volume Backup Timer
Requires=docker-backup.service

[Timer]
# Run daily at 2 AM
OnCalendar=daily
OnCalendar=*-*-* 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Enable and start timer
sudo systemctl daemon-reload
sudo systemctl enable docker-backup.timer
sudo systemctl start docker-backup.timer

# Check status
sudo systemctl status docker-backup.timer
sudo systemctl list-timers docker-backup.timer
```

---

### 4.2 Restore Procedures

**Basic Restore**:

```bash
# Restore volume from backup
restore_volume() {
    local backup_file=$1
    local volume_name=$2

    if [ ! -f "$backup_file" ]; then
        echo "Error: Backup file not found: $backup_file"
        return 1
    fi

    echo "Restoring volume: $volume_name from $backup_file"

    # Create volume if it doesn't exist
    docker volume create "$volume_name"

    # Extract backup into volume
    docker run --rm \
        -v ${volume_name}:/target \
        -v $(dirname $backup_file):/backup \
        alpine \
        tar xzf /backup/$(basename $backup_file) -C /target

    echo "Restore completed successfully"
}

# Usage
restore_volume "/backups/volumes/ml-models_20251102_020000.tar.gz" "ml-models"
```

**Disaster Recovery Restore**:

```bash
cat > disaster_recovery.sh << 'EOF'
#!/bin/bash

# Disaster Recovery Script
# Restores all volumes from backup location

BACKUP_ROOT="/backups"
RESTORE_LOG="/var/log/docker-restore.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$RESTORE_LOG"
}

# Find latest backup for a volume
find_latest_backup() {
    local volume=$1
    local backup_dir="${BACKUP_ROOT}/${volume}"

    if [ ! -d "$backup_dir" ]; then
        return 1
    fi

    # Find most recent .tar.gz file
    latest=$(ls -t "${backup_dir}"/*.tar.gz 2>/dev/null | head -n1)
    echo "$latest"
}

# Restore single volume
restore_volume() {
    local volume=$1
    local backup_file=$2

    log "Restoring volume: $volume"

    # Stop all containers using this volume
    containers=$(docker ps -q --filter volume=$volume)
    if [ ! -z "$containers" ]; then
        log "Stopping containers using $volume: $containers"
        docker stop $containers
    fi

    # Remove existing volume
    docker volume rm -f "$volume" 2>/dev/null || true

    # Create new volume
    docker volume create "$volume"

    # Restore data
    log "Extracting backup: $(basename $backup_file)"
    docker run --rm \
        -v ${volume}:/target \
        -v $(dirname $backup_file):/backup \
        alpine \
        tar xzf /backup/$(basename $backup_file) -C /target

    if [ $? -eq 0 ]; then
        log "SUCCESS: Restored $volume"

        # Restart containers if needed
        if [ ! -z "$containers" ]; then
            log "Restarting containers: $containers"
            docker start $containers
        fi

        return 0
    else
        log "ERROR: Failed to restore $volume"
        return 1
    fi
}

# Main restore process
main() {
    log "===== Disaster Recovery Process Started ====="

    # Read volumes to restore from file or discover from backups
    if [ -f "volumes_to_restore.txt" ]; then
        VOLUMES=$(cat volumes_to_restore.txt)
    else
        # Discover from backup directory
        VOLUMES=$(ls -1 "$BACKUP_ROOT" 2>/dev/null)
    fi

    if [ -z "$VOLUMES" ]; then
        log "ERROR: No volumes to restore"
        exit 1
    fi

    log "Volumes to restore: $VOLUMES"

    local total=0
    local success=0
    local failed=0

    for vol in $VOLUMES; do
        ((total++))

        backup_file=$(find_latest_backup "$vol")

        if [ -z "$backup_file" ]; then
            log "WARNING: No backup found for $vol"
            ((failed++))
            continue
        fi

        log "Latest backup for $vol: $backup_file"

        if restore_volume "$vol" "$backup_file"; then
            ((success++))
        else
            ((failed++))
        fi
    done

    log "===== Disaster Recovery Completed ====="
    log "Total: $total, Success: $success, Failed: $failed"
}

# Confirm before proceeding
read -p "This will restore volumes from backup. Continue? (yes/no): " confirm
if [ "$confirm" = "yes" ]; then
    main
else
    echo "Restore cancelled"
fi
EOF

chmod +x disaster_recovery.sh
```

---

### 4.3 Database-Specific Backup

**PostgreSQL Backup Script**:

```bash
cat > backup_postgres.sh << 'EOF'
#!/bin/bash

CONTAINER_NAME="ml-postgres"
DB_NAME="mldb"
DB_USER="mluser"
BACKUP_DIR="/backups/postgres"
RETENTION_DAYS=30

mkdir -p "$BACKUP_DIR"

# Timestamp for backup file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/mldb_${TIMESTAMP}.sql"

echo "Starting PostgreSQL backup..."

# Create SQL dump
docker exec $CONTAINER_NAME pg_dump -U $DB_USER $DB_NAME > "${BACKUP_FILE}"

if [ $? -eq 0 ]; then
    # Compress backup
    gzip "${BACKUP_FILE}"
    COMPRESSED="${BACKUP_FILE}.gz"

    echo "Backup completed: $COMPRESSED"
    ls -lh "$COMPRESSED"

    # Verify backup integrity
    gunzip -t "$COMPRESSED"
    if [ $? -eq 0 ]; then
        echo "Backup integrity verified"
    else
        echo "WARNING: Backup may be corrupted"
    fi

    # Clean old backups
    find "$BACKUP_DIR" -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete
    echo "Old backups cleaned (retention: ${RETENTION_DAYS} days)"

else
    echo "ERROR: Backup failed"
    exit 1
fi
EOF

chmod +x backup_postgres.sh
```

**PostgreSQL Restore**:

```bash
cat > restore_postgres.sh << 'EOF'
#!/bin/bash

CONTAINER_NAME="ml-postgres"
DB_NAME="mldb"
DB_USER="mluser"
BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup-file.sql.gz>"
    exit 1
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo "Error: Backup file not found: $BACKUP_FILE"
    exit 1
fi

echo "Restoring PostgreSQL from: $BACKUP_FILE"

# Decompress if needed
if [[ $BACKUP_FILE == *.gz ]]; then
    echo "Decompressing backup..."
    gunzip -c "$BACKUP_FILE" | docker exec -i $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME
else
    cat "$BACKUP_FILE" | docker exec -i $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME
fi

if [ $? -eq 0 ]; then
    echo "Restore completed successfully"
else
    echo "ERROR: Restore failed"
    exit 1
fi
EOF

chmod +x restore_postgres.sh
```

---

## 5. Performance Considerations

### 5.1 Storage Performance Benchmarking

**Comprehensive Benchmark Script**:

```bash
cat > benchmark_storage.sh << 'EOF'
#!/bin/bash

echo "=== Docker Storage Performance Benchmark ==="
echo "Date: $(date)"
echo

# Test parameters
TEST_SIZE_MB=1000
BLOCK_SIZES=("1M" "4M" "16M")

# Cleanup function
cleanup() {
    docker volume rm -f bench-named 2>/dev/null
    rm -rf /tmp/bench-bind 2>/dev/null
}

trap cleanup EXIT

# Test 1: Named Volume
echo "1. NAMED VOLUME TEST"
echo "===================="
docker volume create bench-named

for bs in "${BLOCK_SIZES[@]}"; do
    echo "Block size: $bs"

    # Sequential write
    echo -n "  Write (sequential): "
    docker run --rm -v bench-named:/data alpine \
        dd if=/dev/zero of=/data/test bs=$bs count=$((TEST_SIZE_MB / ${bs%M})) \
        oflag=direct 2>&1 | grep "copied" | awk '{print $8" "$9}'

    # Sequential read
    echo -n "  Read (sequential): "
    docker run --rm -v bench-named:/data alpine \
        dd if=/data/test of=/dev/null bs=$bs \
        iflag=direct 2>&1 | grep "copied" | awk '{print $8" "$9}'

    echo
done

docker volume rm bench-named

# Test 2: Bind Mount
echo "2. BIND MOUNT TEST"
echo "=================="
mkdir -p /tmp/bench-bind

for bs in "${BLOCK_SIZES[@]}"; do
    echo "Block size: $bs"

    echo -n "  Write (sequential): "
    docker run --rm -v /tmp/bench-bind:/data alpine \
        dd if=/dev/zero of=/data/test bs=$bs count=$((TEST_SIZE_MB / ${bs%M})) \
        oflag=direct 2>&1 | grep "copied" | awk '{print $8" "$9}'

    echo -n "  Read (sequential): "
    docker run --rm -v /tmp/bench-bind:/data alpine \
        dd if=/data/test of=/dev/null bs=$bs \
        iflag=direct 2>&1 | grep "copied" | awk '{print $8" "$9}'

    echo
done

rm -rf /tmp/bench-bind

# Test 3: tmpfs
echo "3. TMPFS TEST (RAM)"
echo "==================="

for bs in "${BLOCK_SIZES[@]}"; do
    echo "Block size: $bs"

    echo -n "  Write (sequential): "
    docker run --rm --tmpfs /data:rw,size=2g alpine \
        dd if=/dev/zero of=/data/test bs=$bs count=$((TEST_SIZE_MB / ${bs%M})) \
        2>&1 | grep "copied" | awk '{print $8" "$9}'

    echo -n "  Read (sequential): "
    docker run --rm --tmpfs /data:rw,size=2g alpine \
        sh -c "dd if=/dev/zero of=/data/test bs=$bs count=$((TEST_SIZE_MB / ${bs%M})) 2>/dev/null && \
               dd if=/data/test of=/dev/null bs=$bs" \
        2>&1 | grep "copied" | tail -1 | awk '{print $8" "$9}'

    echo
done

# Test 4: Random I/O
echo "4. RANDOM I/O TEST"
echo "=================="

test_random_io() {
    local type=$1
    local mount_args=$2

    echo "Testing: $type"

    # Create test file
    docker run --rm $mount_args alpine \
        dd if=/dev/urandom of=/data/random bs=1M count=100 2>/dev/null

    # Random read test
    echo -n "  Random read (100MB): "
    docker run --rm $mount_args alpine \
        sh -c 'time dd if=/data/random of=/dev/null bs=4k' 2>&1 | \
        grep real | awk '{print $2}'
}

docker volume create bench-named
test_random_io "Named Volume" "-v bench-named:/data"
docker volume rm bench-named

mkdir -p /tmp/bench-bind
test_random_io "Bind Mount" "-v /tmp/bench-bind:/data"
rm -rf /tmp/bench-bind

test_random_io "tmpfs" "--tmpfs /data:size=500m"

echo
echo "=== Benchmark Complete ==="
EOF

chmod +x benchmark_storage.sh
./benchmark_storage.sh
```

---

### 5.2 Performance Optimization Techniques

**Read-Only Mounts for Shared Data**:

```yaml
# docker-compose-optimized.yml
version: '3.8'

services:
  # Model serving - read-only model access
  inference-1:
    image: ml-inference:latest
    volumes:
      - models:/models:ro  # Read-only for safety and performance
      - type: tmpfs
        target: /cache
        tmpfs:
          size: 1g
    deploy:
      replicas: 3

  # Only training can write models
  training:
    image: ml-training:latest
    volumes:
      - models:/models:rw  # Read-write access
      - datasets:/data:ro  # Read-only datasets

volumes:
  models:
    driver: local
  datasets:
    driver: local
    driver_opts:
      type: none
      device: /mnt/ssd/datasets
      o: bind,ro  # Mount as read-only at volume level
```

**Volume Driver Options for Performance**:

```bash
# Create volume on fast SSD
docker volume create \
  --driver local \
  --opt type=none \
  --opt device=/mnt/nvme/ml-data \
  --opt o=bind \
  fast-ml-storage

# Create tmpfs volume with specific options
docker volume create \
  --driver local \
  --opt type=tmpfs \
  --opt device=tmpfs \
  --opt o=size=4g,uid=1000,gid=1000,mode=1777 \
  ml-cache

# Verify configuration
docker volume inspect fast-ml-storage
docker volume inspect ml-cache
```

**ML Training with Optimized Storage**:

```yaml
# docker-compose-training-optimized.yml
version: '3.8'

services:
  training:
    image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
    volumes:
      # Datasets on fast SSD (read-only)
      - dataset-ssd:/data:ro

      # Model checkpoints on NVMe
      - checkpoints-nvme:/checkpoints

      # Training cache in tmpfs (RAM)
      - type: tmpfs
        target: /cache
        tmpfs:
          size: 8g  # 8GB RAM cache

      # Logs on regular disk
      - training-logs:/logs
    environment:
      DATA_PATH: /data
      CHECKPOINT_PATH: /checkpoints
      CACHE_PATH: /cache
      LOG_PATH: /logs
    shm_size: '16gb'  # Shared memory for DataLoader
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  dataset-ssd:
    driver: local
    driver_opts:
      type: none
      device: /mnt/ssd/datasets
      o: bind,ro

  checkpoints-nvme:
    driver: local
    driver_opts:
      type: none
      device: /mnt/nvme/checkpoints
      o: bind

  training-logs:
    driver: local
```

**Monitoring Storage Performance**:

```bash
cat > monitor_storage_performance.sh << 'EOF'
#!/bin/bash

echo "=== Storage Performance Monitor ==="
echo "Date: $(date)"
echo

# Monitor I/O statistics for Docker volumes
echo "1. Volume I/O Statistics"
echo "========================"

# Get volume mountpoints
for vol in $(docker volume ls -q); do
    mountpoint=$(docker volume inspect $vol --format '{{.Mountpoint}}')
    if [ ! -z "$mountpoint" ] && [ -d "$mountpoint" ]; then
        echo "Volume: $vol"
        echo "  Mountpoint: $mountpoint"

        # Get disk I/O stats
        device=$(df "$mountpoint" | tail -1 | awk '{print $1}')
        echo "  Device: $device"

        # Use iostat if available
        if command -v iostat &> /dev/null; then
            iostat -x "$device" 1 2 | tail -n 2
        fi
        echo
    fi
done

# Monitor container I/O
echo "2. Container I/O Statistics"
echo "==========================="
docker stats --no-stream --format \
    "table {{.Container}}\t{{.BlockIO}}\t{{.MemUsage}}\t{{.CPUPerc}}"

echo
echo "3. Overall Disk Usage"
echo "====================="
docker system df -v

EOF

chmod +x monitor_storage_performance.sh
```

---

## 6. Shared Volumes for Distributed Systems

### 6.1 NFS Shared Volumes

**NFS Server Setup** (on shared storage server):

```bash
# Install NFS server
sudo apt-get update
sudo apt-get install -y nfs-kernel-server

# Create shared directory
sudo mkdir -p /mnt/nfs/ml-shared
sudo chown nobody:nogroup /mnt/nfs/ml-shared
sudo chmod 777 /mnt/nfs/ml-shared

# Configure NFS exports
sudo tee -a /etc/exports << EOF
/mnt/nfs/ml-shared 192.168.1.0/24(rw,sync,no_subtree_check,no_root_squash)
EOF

# Apply configuration
sudo exportfs -a
sudo systemctl restart nfs-kernel-server

# Verify
sudo exportfs -v
```

**Docker Volume with NFS** (on Docker hosts):

```bash
# Install NFS client
sudo apt-get install -y nfs-common

# Create NFS-backed volume
docker volume create \
  --driver local \
  --opt type=nfs \
  --opt o=addr=192.168.1.100,rw,nfsvers=4 \
  --opt device=:/mnt/nfs/ml-shared \
  ml-shared-nfs

# Verify
docker volume inspect ml-shared-nfs

# Use in container
docker run -d \
  --name ml-worker-1 \
  -v ml-shared-nfs:/shared \
  ml-worker:latest

# On another host, create same volume
docker volume create \
  --driver local \
  --opt type=nfs \
  --opt o=addr=192.168.1.100,rw,nfsvers=4 \
  --opt device=:/mnt/nfs/ml-shared \
  ml-shared-nfs

docker run -d \
  --name ml-worker-2 \
  -v ml-shared-nfs:/shared \
  ml-worker:latest

# Both containers share the same data!
```

---

### 6.2 Distributed Training Data Sharing

**Multi-Node Training Setup**:

```yaml
# docker-compose-distributed-training.yml
version: '3.8'

services:
  # Master node
  trainer-master:
    image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
    hostname: master
    environment:
      MASTER_ADDR: trainer-master
      MASTER_PORT: 29500
      WORLD_SIZE: 3
      RANK: 0
    volumes:
      - nfs-datasets:/data:ro
      - nfs-models:/models
      - ./training-code:/workspace
    command: python /workspace/distributed_train.py
    networks:
      - training-network

  # Worker nodes
  trainer-worker-1:
    image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
    hostname: worker1
    environment:
      MASTER_ADDR: trainer-master
      MASTER_PORT: 29500
      WORLD_SIZE: 3
      RANK: 1
    volumes:
      - nfs-datasets:/data:ro
      - nfs-models:/models
      - ./training-code:/workspace
    command: python /workspace/distributed_train.py
    depends_on:
      - trainer-master
    networks:
      - training-network

  trainer-worker-2:
    image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
    hostname: worker2
    environment:
      MASTER_ADDR: trainer-master
      MASTER_PORT: 29500
      WORLD_SIZE: 3
      RANK: 2
    volumes:
      - nfs-datasets:/data:ro
      - nfs-models:/models
      - ./training-code:/workspace
    command: python /workspace/distributed_train.py
    depends_on:
      - trainer-master
    networks:
      - training-network

volumes:
  nfs-datasets:
    driver: local
    driver_opts:
      type: nfs
      o: addr=192.168.1.100,rw,nfsvers=4
      device: ":/mnt/nfs/datasets"

  nfs-models:
    driver: local
    driver_opts:
      type: nfs
      o: addr=192.168.1.100,rw,nfsvers=4
      device: ":/mnt/nfs/models"

networks:
  training-network:
    driver: bridge
```

**Distributed Training Code**:

```python
# distributed_train.py
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path

def setup_distributed():
    """Initialize distributed training"""
    # Get environment variables
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']

    # Initialize process group
    dist.init_process_group(
        backend='gloo',  # Use 'nccl' for GPUs
        init_method=f'tcp://{master_addr}:{master_port}',
        rank=rank,
        world_size=world_size
    )

    return rank, world_size

def load_checkpoint_from_shared_volume(model, checkpoint_path):
    """Load checkpoint from shared NFS volume"""
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Loaded checkpoint from epoch {epoch}, loss {loss:.4f}")
        return epoch
    return 0

def save_checkpoint_to_shared_volume(model, epoch, loss, checkpoint_path):
    """Save checkpoint to shared NFS volume - only rank 0"""
    if dist.get_rank() == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss
        }
        # Atomic write: write to temp file, then rename
        temp_path = f"{checkpoint_path}.tmp"
        torch.save(checkpoint, temp_path)
        os.rename(temp_path, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

def train():
    """Main training loop"""
    rank, world_size = setup_distributed()

    # Paths on shared NFS volumes
    data_path = Path("/data")
    model_path = Path("/models")
    checkpoint_path = model_path / "checkpoint.pth"

    print(f"[Rank {rank}/{world_size}] Starting training")

    # Create simple model
    model = nn.Linear(100, 10)
    ddp_model = DDP(model)

    # Load checkpoint if exists
    start_epoch = load_checkpoint_from_shared_volume(ddp_model.module, checkpoint_path)

    # Training loop
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    for epoch in range(start_epoch, 10):
        # Simulate training
        inputs = torch.randn(32, 100)
        labels = torch.randint(0, 10, (32,))

        outputs = ddp_model(inputs)
        loss = nn.functional.cross_entropy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

            # Save checkpoint every 2 epochs
            if epoch % 2 == 0:
                save_checkpoint_to_shared_volume(
                    ddp_model.module, epoch, loss.item(), checkpoint_path
                )

    # Cleanup
    dist.destroy_process_group()
    print(f"[Rank {rank}] Training complete")

if __name__ == "__main__":
    train()
```

---

## 7. Production ML Data Management

### 7.1 ML Pipeline Storage Architecture

**Complete ML Pipeline**:

```yaml
# docker-compose-ml-pipeline.yml
version: '3.8'

services:
  # Data ingestion
  data-collector:
    image: ml-pipeline/collector:latest
    volumes:
      - raw-data:/data/raw
      - ./configs:/configs:ro
    environment:
      OUTPUT_DIR: /data/raw
    labels:
      pipeline.stage: "ingestion"

  # Data preprocessing
  preprocessor:
    image: ml-pipeline/preprocessor:latest
    volumes:
      - raw-data:/data/raw:ro
      - processed-data:/data/processed
      - type: tmpfs
        target: /tmp
        tmpfs:
          size: 4g
    environment:
      INPUT_DIR: /data/raw
      OUTPUT_DIR: /data/processed
    depends_on:
      - data-collector
    labels:
      pipeline.stage: "preprocessing"

  # Feature engineering
  feature-engineer:
    image: ml-pipeline/features:latest
    volumes:
      - processed-data:/data/processed:ro
      - features:/data/features
      - type: tmpfs
        target: /cache
        tmpfs:
          size: 2g
    environment:
      INPUT_DIR: /data/processed
      OUTPUT_DIR: /data/features
    depends_on:
      - preprocessor
    labels:
      pipeline.stage: "features"

  # Model training
  trainer:
    image: ml-pipeline/training:latest
    volumes:
      - features:/data:ro
      - models:/models
      - checkpoints:/checkpoints
      - tensorboard-logs:/logs/tensorboard
      - training-logs:/logs/training
      - type: tmpfs
        target: /cache
        tmpfs:
          size: 8g
    environment:
      DATA_DIR: /data
      MODEL_DIR: /models
      CHECKPOINT_DIR: /checkpoints
      LOG_DIR: /logs
    shm_size: '16gb'
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    labels:
      pipeline.stage: "training"

  # Model evaluation
  evaluator:
    image: ml-pipeline/evaluation:latest
    volumes:
      - features:/data:ro
      - models:/models:ro
      - evaluation-results:/results
    environment:
      DATA_DIR: /data
      MODEL_DIR: /models
      OUTPUT_DIR: /results
    depends_on:
      - trainer
    labels:
      pipeline.stage: "evaluation"

  # Model registry
  registry:
    image: postgres:15-alpine
    volumes:
      - registry-db:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: model_registry
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: ${REGISTRY_PASSWORD}
    labels:
      pipeline.component: "registry"

  # MLflow tracking server
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    volumes:
      - mlflow-artifacts:/mlflow
      - models:/models:ro
    ports:
      - "5000:5000"
    environment:
      MLFLOW_BACKEND_STORE_URI: postgresql://mlflow:${REGISTRY_PASSWORD}@registry/model_registry
      MLFLOW_DEFAULT_ARTIFACT_ROOT: /mlflow/artifacts
    depends_on:
      - registry
    command: mlflow server --host 0.0.0.0 --port 5000
    labels:
      pipeline.component: "tracking"

  # TensorBoard
  tensorboard:
    image: tensorflow/tensorflow:latest
    volumes:
      - tensorboard-logs:/logs:ro
    ports:
      - "6006:6006"
    command: tensorboard --logdir=/logs --host=0.0.0.0
    labels:
      pipeline.component: "monitoring"

  # Model serving
  inference:
    image: ml-pipeline/inference:latest
    volumes:
      - models:/models:ro
      - type: tmpfs
        target: /cache
        tmpfs:
          size: 2g
    ports:
      - "8000:8000"
    environment:
      MODEL_DIR: /models
      CACHE_DIR: /cache
    depends_on:
      - evaluator
    labels:
      pipeline.stage: "serving"

volumes:
  # Data volumes
  raw-data:
    driver: local
    labels:
      tier: "bronze"
      retention: "30d"
      backup: "weekly"

  processed-data:
    driver: local
    labels:
      tier: "silver"
      retention: "60d"
      backup: "daily"

  features:
    driver: local
    driver_opts:
      type: none
      device: /mnt/ssd/features
      o: bind
    labels:
      tier: "gold"
      retention: "90d"
      backup: "daily"

  # Model volumes
  models:
    driver: local
    driver_opts:
      type: none
      device: /mnt/ssd/models
      o: bind
    labels:
      critical: "true"
      backup: "hourly"
      retention: "365d"

  checkpoints:
    driver: local
    labels:
      backup: "continuous"
      retention: "30d"

  # Logging volumes
  training-logs:
    driver: local
    labels:
      retention: "90d"

  tensorboard-logs:
    driver: local
    labels:
      retention: "180d"

  evaluation-results:
    driver: local
    labels:
      retention: "365d"
      backup: "daily"

  # Infrastructure volumes
  registry-db:
    driver: local
    labels:
      critical: "true"
      backup: "hourly"

  mlflow-artifacts:
    driver: local
    labels:
      backup: "daily"
      retention: "365d"
```

---

### 7.2 Dataset Version Management

**Dataset Versioning System**:

```python
# dataset_manager.py
import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

class DatasetVersionManager:
    """Manage dataset versions in Docker volumes"""

    def __init__(self, base_path="/data"):
        self.base_path = Path(base_path)
        self.versions_path = self.base_path / "versions"
        self.metadata_path = self.base_path / "metadata"

        self.versions_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)

    def compute_dataset_hash(self, dataset_path: Path) -> str:
        """Compute hash of dataset for versioning"""
        hasher = hashlib.sha256()

        # Hash all files in directory
        for file_path in sorted(dataset_path.rglob("*")):
            if file_path.is_file():
                hasher.update(file_path.name.encode())
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)

        return hasher.hexdigest()[:16]

    def create_version(self,
                       dataset_path: Path,
                       version_name: str,
                       metadata: Optional[Dict] = None) -> str:
        """Create new dataset version"""

        # Compute hash for deduplication
        dataset_hash = self.compute_dataset_hash(dataset_path)
        version_id = f"{version_name}_{dataset_hash}"
        version_path = self.versions_path / version_id

        # Check if version already exists
        if version_path.exists():
            print(f"Version {version_id} already exists (deduplicated)")
            return version_id

        # Copy dataset to version directory
        print(f"Creating version: {version_id}")
        shutil.copytree(dataset_path, version_path)

        # Save metadata
        meta = {
            "version_id": version_id,
            "version_name": version_name,
            "hash": dataset_hash,
            "created_at": datetime.now().isoformat(),
            "source_path": str(dataset_path),
            "size_bytes": sum(f.stat().st_size for f in version_path.rglob('*') if f.is_file()),
            **(metadata or {})
        }

        meta_file = self.metadata_path / f"{version_id}.json"
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"Version created: {version_id}")
        return version_id

    def get_version(self, version_id: str) -> Optional[Path]:
        """Get path to specific version"""
        version_path = self.versions_path / version_id
        if version_path.exists():
            return version_path
        return None

    def list_versions(self) -> Dict:
        """List all versions with metadata"""
        versions = {}
        for meta_file in self.metadata_path.glob("*.json"):
            with open(meta_file) as f:
                meta = json.load(f)
                versions[meta['version_id']] = meta
        return versions

    def link_version(self, version_id: str, link_name: str):
        """Create symbolic link to version (e.g., 'latest')"""
        version_path = self.get_version(version_id)
        if not version_path:
            raise ValueError(f"Version not found: {version_id}")

        link_path = self.base_path / link_name
        if link_path.exists():
            link_path.unlink()

        link_path.symlink_to(version_path)
        print(f"Linked {link_name} -> {version_id}")

# Usage example
if __name__ == "__main__":
    manager = DatasetVersionManager("/data/datasets")

    # Create versions
    v1 = manager.create_version(
        Path("/data/raw/imagenet"),
        version_name="imagenet-v1",
        metadata={"description": "Original ImageNet subset"}
    )

    v2 = manager.create_version(
        Path("/data/processed/imagenet"),
        version_name="imagenet-v2",
        metadata={"description": "Preprocessed with augmentation"}
    )

    # Link latest version
    manager.link_version(v2, "latest")

    # List all versions
    versions = manager.list_versions()
    print(json.dumps(versions, indent=2))
```

---

### 7.3 Model Checkpoint Management

**Checkpoint Manager**:

```python
# checkpoint_manager.py
import torch
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

class CheckpointManager:
    """Manage model checkpoints with automatic cleanup"""

    def __init__(self, checkpoint_dir="/checkpoints", max_checkpoints=10):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.metadata_file = self.checkpoint_dir / "checkpoints.json"

        # Load existing metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load checkpoint metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {"checkpoints": []}

    def _save_metadata(self):
        """Save checkpoint metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def save_checkpoint(self,
                       model,
                       optimizer,
                       epoch: int,
                       metrics: Dict,
                       is_best: bool = False):
        """Save checkpoint with automatic cleanup"""

        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        # Save checkpoint file
        checkpoint_name = f"checkpoint_epoch_{epoch:04d}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        torch.save(checkpoint, checkpoint_path)

        # Update metadata
        checkpoint_info = {
            'filename': checkpoint_name,
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': checkpoint['timestamp'],
            'is_best': is_best,
            'size_mb': checkpoint_path.stat().st_size / (1024 * 1024)
        }

        self.metadata['checkpoints'].append(checkpoint_info)

        # Handle best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint: epoch {epoch}")

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        # Save metadata
        self._save_metadata()

        print(f"Saved checkpoint: {checkpoint_name}")

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only max_checkpoints recent ones"""
        checkpoints = self.metadata['checkpoints']

        if len(checkpoints) <= self.max_checkpoints:
            return

        # Sort by epoch
        checkpoints.sort(key=lambda x: x['epoch'])

        # Keep best checkpoint and recent ones
        to_keep = set()
        for cp in checkpoints:
            if cp.get('is_best', False):
                to_keep.add(cp['filename'])

        # Keep most recent
        recent = checkpoints[-self.max_checkpoints:]
        for cp in recent:
            to_keep.add(cp['filename'])

        # Delete old checkpoints
        removed = []
        for cp in checkpoints:
            if cp['filename'] not in to_keep:
                checkpoint_file = self.checkpoint_dir / cp['filename']
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
                    removed.append(cp['filename'])
                    print(f"Removed old checkpoint: {cp['filename']}")

        # Update metadata
        self.metadata['checkpoints'] = [
            cp for cp in checkpoints
            if cp['filename'] not in removed
        ]

    def load_checkpoint(self, epoch: Optional[int] = None) -> Optional[Dict]:
        """Load checkpoint by epoch (None = latest)"""
        checkpoints = self.metadata['checkpoints']

        if not checkpoints:
            return None

        if epoch is None:
            # Load latest
            checkpoint = max(checkpoints, key=lambda x: x['epoch'])
        else:
            # Load specific epoch
            checkpoint = next(
                (cp for cp in checkpoints if cp['epoch'] == epoch),
                None
            )

        if checkpoint:
            checkpoint_path = self.checkpoint_dir / checkpoint['filename']
            return torch.load(checkpoint_path)

        return None

    def load_best(self) -> Optional[Dict]:
        """Load best checkpoint"""
        best_path = self.checkpoint_dir / "best_model.pth"
        if best_path.exists():
            return torch.load(best_path)
        return None

    def list_checkpoints(self) -> List[Dict]:
        """List all available checkpoints"""
        return self.metadata['checkpoints']

# Usage example
if __name__ == "__main__":
    import torch.nn as nn

    manager = CheckpointManager("/checkpoints", max_checkpoints=5)

    # Create dummy model
    model = nn.Linear(100, 10)
    optimizer = torch.optim.Adam(model.parameters())

    # Simulate training
    best_loss = float('inf')
    for epoch in range(20):
        loss = 1.0 - (epoch * 0.04)  # Simulated decreasing loss
        accuracy = 0.5 + (epoch * 0.02)  # Simulated increasing accuracy

        metrics = {
            'loss': loss,
            'accuracy': accuracy
        }

        is_best = loss < best_loss
        if is_best:
            best_loss = loss

        manager.save_checkpoint(model, optimizer, epoch, metrics, is_best)

    # List all checkpoints
    print("\nAvailable checkpoints:")
    for cp in manager.list_checkpoints():
        print(f"  Epoch {cp['epoch']}: loss={cp['metrics']['loss']:.4f}, "
              f"accuracy={cp['metrics']['accuracy']:.4f}, "
              f"best={cp.get('is_best', False)}")

    # Load best
    best = manager.load_best()
    print(f"\nBest checkpoint: epoch={best['epoch']}, "
          f"loss={best['metrics']['loss']:.4f}")
```

---

## 8. Advanced Topics

### 8.1 Volume Encryption

**Encrypted Volume Setup**:

```bash
# Install dm-crypt tools
sudo apt-get install -y cryptsetup

# Create encrypted volume script
cat > create_encrypted_volume.sh << 'EOF'
#!/bin/bash

VOLUME_NAME=$1
SIZE_GB=${2:-10}
MOUNT_PATH="/var/lib/docker-encrypted/${VOLUME_NAME}"

if [ -z "$VOLUME_NAME" ]; then
    echo "Usage: $0 <volume-name> [size-in-gb]"
    exit 1
fi

# Create directory for encrypted data
sudo mkdir -p "$MOUNT_PATH"

# Create encrypted file container
sudo dd if=/dev/zero of="${MOUNT_PATH}.img" bs=1G count=$SIZE_GB

# Setup encryption
sudo cryptsetup luksFormat "${MOUNT_PATH}.img"

# Open encrypted container
sudo cryptsetup open "${MOUNT_PATH}.img" "${VOLUME_NAME}_crypt"

# Create filesystem
sudo mkfs.ext4 /dev/mapper/"${VOLUME_NAME}_crypt"

# Mount
sudo mount /dev/mapper/"${VOLUME_NAME}_crypt" "$MOUNT_PATH"

# Set permissions
sudo chmod 777 "$MOUNT_PATH"

# Create Docker volume
docker volume create \
  --driver local \
  --opt type=none \
  --opt device="$MOUNT_PATH" \
  --opt o=bind \
  "${VOLUME_NAME}_encrypted"

echo "Encrypted volume created: ${VOLUME_NAME}_encrypted"
echo "Mounted at: $MOUNT_PATH"
EOF

chmod +x create_encrypted_volume.sh

# Create encrypted volume for sensitive model data
# sudo ./create_encrypted_volume.sh sensitive-models 5
```

---

### 8.2 Volume Monitoring and Alerting

**Comprehensive Monitoring System**:

```python
# volume_monitor.py
import docker
import time
import json
from datetime import datetime
from pathlib import Path
import subprocess

class VolumeMonitor:
    """Monitor Docker volumes and send alerts"""

    def __init__(self, alert_threshold_gb=50, check_interval=300):
        self.client = docker.from_env()
        self.alert_threshold_gb = alert_threshold_gb
        self.check_interval = check_interval
        self.metrics_file = Path("/var/log/docker-volumes-metrics.json")

    def get_volume_size(self, volume_name):
        """Get volume size in GB"""
        try:
            volume = self.client.volumes.get(volume_name)
            mountpoint = volume.attrs['Mountpoint']

            result = subprocess.run(
                ['du', '-sb', mountpoint],
                capture_output=True,
                text=True
            )

            size_bytes = int(result.stdout.split()[0])
            size_gb = size_bytes / (1024**3)
            return size_gb
        except Exception as e:
            print(f"Error getting size for {volume_name}: {e}")
            return 0

    def check_volumes(self):
        """Check all volumes and collect metrics"""
        volumes = self.client.volumes.list()
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'volumes': []
        }

        for volume in volumes:
            name = volume.name
            size_gb = self.get_volume_size(name)
            labels = volume.attrs.get('Labels', {})

            volume_metric = {
                'name': name,
                'size_gb': round(size_gb, 2),
                'labels': labels,
                'alert': size_gb > self.alert_threshold_gb
            }

            metrics['volumes'].append(volume_metric)

            # Alert if threshold exceeded
            if volume_metric['alert']:
                self.send_alert(name, size_gb)

        # Save metrics
        self.save_metrics(metrics)

        return metrics

    def save_metrics(self, metrics):
        """Save metrics to file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

    def send_alert(self, volume_name, size_gb):
        """Send alert for large volume"""
        message = (
            f"ALERT: Volume '{volume_name}' size ({size_gb:.2f}GB) "
            f"exceeds threshold ({self.alert_threshold_gb}GB)"
        )
        print(message)

        # Send to logging/monitoring system
        # Example: Slack, email, PagerDuty, etc.

    def run(self):
        """Main monitoring loop"""
        print(f"Starting volume monitor (threshold: {self.alert_threshold_gb}GB)")

        while True:
            try:
                metrics = self.check_volumes()
                print(f"Checked {len(metrics['volumes'])} volumes")

            except Exception as e:
                print(f"Error during monitoring: {e}")

            time.sleep(self.check_interval)

if __name__ == "__main__":
    monitor = VolumeMonitor(alert_threshold_gb=50, check_interval=300)
    monitor.run()
```

---

## 9. Troubleshooting Guide

### 9.1 Common Issues and Solutions

**Permission Denied Errors**:

```bash
# Problem: Permission denied when writing to volume
# Solution 1: Run container with specific user ID
docker run -u $(id -u):$(id -g) -v myvolume:/data alpine sh -c "touch /data/test"

# Solution 2: Fix volume permissions
docker run --rm -v myvolume:/data alpine chown -R 1000:1000 /data

# Solution 3: Use read-write permissions
docker run --rm -v myvolume:/data alpine chmod -R 777 /data
```

**Volume Not Found**:

```bash
# Check if volume exists
docker volume ls | grep myvolume

# Inspect volume
docker volume inspect myvolume

# Create if missing
docker volume create myvolume
```

**Volume Full / Out of Space**:

```bash
# Check disk usage
docker system df -v

# Find large volumes
for vol in $(docker volume ls -q); do
    size=$(docker run --rm -v $vol:/data alpine du -sh /data 2>/dev/null | awk '{print $1}')
    echo "$vol: $size"
done | sort -h -k2

# Clean up
docker volume prune -f
```

---

## 10. Best Practices Checklist

### Production Readiness Checklist:

- [ ] Use named volumes for all persistent data
- [ ] Implement automated backup system
- [ ] Test restore procedures regularly
- [ ] Label volumes with environment and purpose
- [ ] Set up volume monitoring and alerts
- [ ] Use read-only mounts where possible
- [ ] Implement volume cleanup policies
- [ ] Document volume dependencies
- [ ] Use tmpfs for temporary/sensitive data
- [ ] Configure appropriate retention policies
- [ ] Test disaster recovery procedures
- [ ] Monitor storage performance
- [ ] Implement access controls
- [ ] Use encryption for sensitive data
- [ ] Maintain volume documentation

---

## Summary

This implementation guide covered comprehensive Docker volume management for ML infrastructure, including:

1. **Volume Types**: Named volumes, bind mounts, and tmpfs
2. **Lifecycle Management**: Creation, inspection, and cleanup
3. **Data Persistence**: Databases, application state, and configuration
4. **Backup/Restore**: Automated systems and disaster recovery
5. **Performance**: Optimization techniques and benchmarking
6. **Distributed Systems**: NFS and shared storage
7. **ML Data Management**: Pipelines, versioning, and checkpoints
8. **Advanced Topics**: Encryption and monitoring
9. **Troubleshooting**: Common issues and solutions
10. **Best Practices**: Production-ready patterns

You now have the knowledge to design, implement, and maintain robust storage solutions for containerized ML workloads.

---

**Document Version**: 1.0
**Last Updated**: November 2025
**Target**: AI Infrastructure Junior Engineers
**License**: Educational Use
