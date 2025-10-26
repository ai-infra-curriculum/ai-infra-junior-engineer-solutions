# Step-by-Step Implementation Guide: Docker Volumes & Data Persistence

## Overview

Manage data persistence in Docker containers for ML datasets, model checkpoints, and databases. Learn named volumes, bind mounts, volume drivers, and data backup strategies.

**Time**: 2-3 hours | **Difficulty**: Intermediate

---

## Learning Objectives

✅ Understand volume types (named volumes, bind mounts, tmpfs)
✅ Persist ML models and training checkpoints
✅ Share data between containers
✅ Backup and restore volumes
✅ Optimize volume performance
✅ Manage volume permissions
✅ Implement volume drivers

---

## Volume Types

### 1. Named Volumes

```bash
# Create volume
docker volume create ml-data

# Use volume
docker run -v ml-data:/app/data ml-training

# With docker-compose
version: '3.8'
services:
  db:
    image: postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### 2. Bind Mounts

```bash
# Mount host directory to container
docker run -v $(pwd)/data:/app/data ml-training

# Read-only mount
docker run -v $(pwd)/models:/app/models:ro ml-api

# With docker-compose
services:
  training:
    volumes:
      - ./data:/workspace/data:ro
      - ./checkpoints:/workspace/checkpoints
```

### 3. tmpfs (Memory)

```bash
# Temporary in-memory storage
docker run --tmpfs /tmp ml-app

# With size limit
docker run --tmpfs /tmp:size=1g ml-app
```

---

## ML Use Cases

### Training with Persistent Checkpoints

```yaml
services:
  trainer:
    image: ml-training
    volumes:
      # Data (read-only)
      - ./data:/workspace/data:ro

      # Checkpoints (persistent)
      - checkpoints:/workspace/checkpoints

      # Logs (persistent)
      - logs:/workspace/logs

      # Cache (temporary, fast)
      - /tmp/cache

volumes:
  checkpoints:
  logs:
```

### Model Serving with Shared Models

```yaml
services:
  # Model trainer
  trainer:
    volumes:
      - models:/app/models

  # API servers share model volume
  api-1:
    volumes:
      - models:/app/models:ro

  api-2:
    volumes:
      - models:/app/models:ro

volumes:
  models:
```

---

## Volume Commands

```bash
# List volumes
docker volume ls

# Create volume
docker volume create my-volume

# Inspect volume
docker volume inspect my-volume

# Remove volume
docker volume rm my-volume

# Prune unused volumes
docker volume prune

# Remove all volumes
docker volume prune -a
```

---

## Backup & Restore

### Backup Volume

```bash
# Create backup container
docker run --rm \
  -v ml-data:/data \
  -v $(pwd):/backup \
  alpine \
  tar czf /backup/ml-data-backup.tar.gz -C /data .
```

### Restore Volume

```bash
# Restore from backup
docker run --rm \
  -v ml-data:/data \
  -v $(pwd):/backup \
  alpine \
  tar xzf /backup/ml-data-backup.tar.gz -C /data
```

---

## Performance Optimization

### Use Volume for Large Datasets

```yaml
# Fast volume access
services:
  training:
    volumes:
      - fast-data:/workspace/data

volumes:
  fast-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/fast-ssd/ml-data
```

---

## Best Practices

✅ Use named volumes for persistence
✅ Use bind mounts for development
✅ Mount datasets as read-only when possible
✅ Back up critical volumes regularly
✅ Use tmpfs for temporary cache
✅ Set appropriate permissions
✅ Clean up unused volumes

---

**Docker volumes mastered!** 💾
