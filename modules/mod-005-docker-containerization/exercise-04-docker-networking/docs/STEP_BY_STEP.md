# Step-by-Step Implementation Guide: Docker Networking

## Overview

Master Docker networking for ML microservices. Learn bridge networks, custom networks, service discovery, port mapping, and inter-container communication for distributed ML systems.

**Time**: 2-3 hours | **Difficulty**: Intermediate

---

## Learning Objectives

✅ Understand Docker network drivers (bridge, host, overlay)
✅ Create custom networks for service isolation
✅ Implement service discovery
✅ Configure port mapping and exposure
✅ Enable inter-container communication
✅ Troubleshoot network connectivity
✅ Secure container networking

---

## Network Types

### 1. Bridge (Default)

```bash
# Create custom bridge network
docker network create ml-network

# Run containers on network
docker run -d --name api --network ml-network ml-api
docker run -d --name db --network ml-network postgres

# Containers can communicate via names
# api can reach db at: postgresql://db:5432
```

### 2. Host Network

```bash
# Use host network (no isolation)
docker run --network host ml-api

# Container uses host's IP directly
# Good for: high-performance networking
# Bad for: security isolation
```

### 3. Custom Networks

```yaml
# docker-compose.yml
version: '3.8'

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true  # No external access

services:
  api:
    networks:
      - frontend
      - backend

  db:
    networks:
      - backend  # Only internal access
```

---

## Service Discovery

### Automatic DNS Resolution

```bash
# Create network
docker network create app-network

# Start services
docker run -d --name database --network app-network postgres
docker run -d --name api --network app-network ml-api

# API can connect to database using hostname "database"
# Connection string: postgresql://database:5432/db
```

### docker-compose Example

```yaml
services:
  api:
    image: ml-api
    environment:
      - DB_HOST=postgres  # Service name as hostname
      - REDIS_HOST=redis
    networks:
      - app-network

  postgres:
    image: postgres:15
    networks:
      - app-network

  redis:
    image: redis:7
    networks:
      - app-network

networks:
  app-network:
    driver: bridge
```

---

## Port Mapping

```bash
# Expose port (container to host)
docker run -p 8000:8000 ml-api

# Bind to specific host IP
docker run -p 127.0.0.1:8000:8000 ml-api

# Random host port
docker run -P ml-api

# Multiple ports
docker run -p 8000:8000 -p 5432:5432 ml-app

# UDP port
docker run -p 8125:8125/udp statsd
```

---

## Network Commands

```bash
# List networks
docker network ls

# Create network
docker network create my-network

# Inspect network
docker network inspect my-network

# Connect container to network
docker network connect my-network container-name

# Disconnect container
docker network disconnect my-network container-name

# Remove network
docker network rm my-network

# Prune unused networks
docker network prune
```

---

## Troubleshooting

### Test Connectivity

```bash
# Ping from one container to another
docker exec api ping database

# Check DNS resolution
docker exec api nslookup database

# Test port connectivity
docker exec api nc -zv database 5432

# Curl test
docker exec api curl http://other-service:8000/health
```

### Inspect Network

```bash
# View network details
docker network inspect bridge

# See which containers are connected
docker network inspect my-network --format='{{range .Containers}}{{.Name}} {{end}}'
```

---

## Best Practices

✅ Use custom networks for service isolation
✅ Use service names for inter-container communication
✅ Minimize exposed ports
✅ Use internal networks for databases
✅ Implement proper DNS configuration
✅ Monitor network traffic
✅ Use network aliases when needed

---

**Docker networking complete!** 🌐
