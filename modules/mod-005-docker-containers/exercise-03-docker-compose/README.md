# Exercise 03: Docker Compose & Multi-Container Apps

Master Docker Compose for orchestrating multi-container ML applications including APIs, databases, caching, and monitoring.

## Complete Solution

This exercise includes **complete, production-ready solutions** with:

### 3 Complete ML Stacks:
1. **ML API Stack** - API + PostgreSQL + Redis + Monitoring
2. **Jupyter MLflow Stack** - Development environment with experiment tracking
3. **Model Serving Stack** - Load-balanced serving with replicas

### Tools & Scripts:
- **compose_manager.py** - Comprehensive stack management tool
- Multiple Dockerfiles for different services
- Environment configuration templates
- **20+ comprehensive tests**

## Quick Start

### Using the Compose Manager

```bash
cd solutions/

# List available stacks
./compose_manager.py list

# Start ML API stack
./compose_manager.py up --stack ml-api --build

# Check service health
./compose_manager.py health --stack ml-api

# View logs
./compose_manager.py logs --stack ml-api --follow

# Stop stack
./compose_manager.py down --stack ml-api
```

### Using Docker Compose Directly

```bash
# ML API stack
docker-compose -f docker-compose-ml-api.yml up -d

# Jupyter MLflow stack
docker-compose -f docker-compose-jupyter-mlflow.yml up -d

# Model serving stack
docker-compose -f docker-compose-model-serving.yml up -d
```

### Run Tests

```bash
python test_compose.py
```

## Available Stacks

### 1. ML API Stack (`docker-compose-ml-api.yml`)

Complete ML API with monitoring:
- **ml-api**: Flask ML API service
- **postgres**: PostgreSQL database
- **redis**: Redis cache
- **prometheus**: Metrics collection
- **grafana**: Visualization dashboards

Access:
- API: http://localhost:8000
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

### 2. Jupyter MLflow Stack (`docker-compose-jupyter-mlflow.yml`)

ML development and experiment tracking:
- **jupyter**: JupyterLab environment
- **mlflow**: MLflow tracking server
- **postgres**: Backend store
- **minio**: S3-compatible artifact storage

Access:
- Jupyter: http://localhost:8888
- MLflow: http://localhost:5000
- MinIO: http://localhost:9001

### 3. Model Serving Stack (`docker-compose-model-serving.yml`)

Production model serving:
- **nginx**: Load balancer
- **model-server-1/2/3**: 3 replicas
- **redis**: Response caching
- **prometheus**: Metrics
- **grafana**: Dashboards
- **locust**: Load testing

Access:
- API (via LB): http://localhost:80
- Locust: http://localhost:8089
- Grafana: http://localhost:3000

## Key Features

### Health Checks
All critical services have health checks:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### Service Dependencies
Proper startup ordering with conditions:
```yaml
depends_on:
  postgres:
    condition: service_healthy
  redis:
    condition: service_healthy
```

### Resource Limits
Production-ready resource constraints:
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 2G
    reservations:
      cpus: '1'
      memory: 1G
```

### Volume Management
Named volumes for persistence:
```yaml
volumes:
  postgres-data:
    driver: local
  redis-data:
    driver: local
```

### Network Isolation
Custom networks with subnets:
```yaml
networks:
  ml-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## Usage Examples

### Example 1: Start ML API Stack

```bash
# Start with build
./compose_manager.py up --stack ml-api --build

# Check health
./compose_manager.py health --stack ml-api

# View specific service logs
./compose_manager.py logs --stack ml-api --service ml-api --follow

# Execute command in container
./compose_manager.py exec --stack ml-api --service postgres --command psql -U mluser -d mldb

# Restart service
./compose_manager.py restart --stack ml-api

# Stop stack (keep volumes)
./compose_manager.py down --stack ml-api

# Stop stack (remove volumes)
./compose_manager.py down --stack ml-api --volumes
```

### Example 2: Jupyter MLflow Development

```bash
# Start stack
./compose_manager.py up --stack jupyter-mlflow

# Access Jupyter at http://localhost:8888
# Access MLflow at http://localhost:5000

# View logs
./compose_manager.py logs --stack jupyter-mlflow --follow

# Execute in Jupyter container
./compose_manager.py exec --stack jupyter-mlflow --service jupyter --command bash
```

### Example 3: Model Serving with Load Balancing

```bash
# Start serving stack
./compose_manager.py up --stack model-serving

# Check all replicas are healthy
./compose_manager.py health --stack model-serving

# Scale model servers
docker-compose -f docker-compose-model-serving.yml up -d --scale model-server=5

# Run load test (via Locust at http://localhost:8089)
# Monitor in Grafana at http://localhost:3000
```

### Example 4: Managing Environment Variables

```bash
# Copy example env file
cp .env.example .env

# Edit environment variables
vim .env

# Start with env file
docker-compose -f docker-compose-ml-api.yml --env-file .env up -d
```

## Best Practices Implemented

1. **Health Checks**: All services have proper health checks
2. **Service Dependencies**: Correct startup ordering with conditions
3. **Resource Limits**: CPU and memory constraints
4. **Volume Management**: Named volumes for data persistence
5. **Network Isolation**: Custom networks for security
6. **Restart Policies**: Automatic restart on failure
7. **Environment Configuration**: Externalized configuration
8. **Monitoring**: Prometheus + Grafana integration
9. **Non-root Users**: All services run as non-root
10. **Security**: No secrets in files, use environment variables

## Troubleshooting

### Services Won't Start

```bash
# Validate compose file
./compose_manager.py validate --stack ml-api

# Check service status
./compose_manager.py ps --stack ml-api

# View logs
./compose_manager.py logs --stack ml-api --tail 100
```

### Database Connection Issues

```bash
# Check database health
docker-compose -f docker-compose-ml-api.yml exec postgres pg_isready

# Connect to database
docker-compose -f docker-compose-ml-api.yml exec postgres psql -U mluser -d mldb

# View database logs
./compose_manager.py logs --stack ml-api --service postgres
```

### Network Issues

```bash
# Inspect network
docker network inspect <network_name>

# Check service connectivity
docker-compose -f docker-compose-ml-api.yml exec ml-api ping postgres

# Restart networking
./compose_manager.py down --stack ml-api
./compose_manager.py up --stack ml-api
```

## Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Compose File Reference](https://docs.docker.com/compose/compose-file/)
