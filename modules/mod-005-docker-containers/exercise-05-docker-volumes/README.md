# Exercise 05: Docker Volumes & Data Persistence

Master Docker volumes for ML data persistence, backups, and sharing data between containers.

## Complete Solution

**Production-ready volume management** in `solutions/`:
- **volume_manager.py** - Comprehensive volume management tool
- **20+ comprehensive tests**

## Quick Start

```bash
cd solutions/

# Create ML volumes
./volume_manager.py setup-ml

# List all volumes
./volume_manager.py list

# Create custom volume
./volume_manager.py create --name my-models

# Backup volume
./volume_manager.py backup --name ml-models --backup-path /backups/models.tar.gz

# Restore volume
./volume_manager.py restore --name ml-models --backup-path /backups/models.tar.gz

# Run tests
python test_volumes.py
```

## Learning Objectives

- Create and manage Docker volumes
- Implement data persistence for ML workloads
- Backup and restore volume data
- Share volumes between containers
- Optimize volume performance

## Key Concepts

### Named Volumes
```bash
# Create named volume
docker volume create ml-models

# Use in container
docker run -v ml-models:/models my-ml-app
```

### Bind Mounts
```bash
# Mount host directory
docker run -v $(pwd)/models:/models my-ml-app
```

### Volume Drivers
- **local**: Default driver
- **nfs**: Network File System
- **s3**: S3-backed volumes
- **gcs**: Google Cloud Storage

## Best Practices

1. Use named volumes for data persistence
2. Backup volumes regularly
3. Use readonly mounts when appropriate
4. Label volumes for organization
5. Clean up unused volumes

## Resources

- [Docker Volumes Documentation](https://docs.docker.com/storage/volumes/)
- [Volume Drivers](https://docs.docker.com/storage/volumes/#use-a-volume-driver)
