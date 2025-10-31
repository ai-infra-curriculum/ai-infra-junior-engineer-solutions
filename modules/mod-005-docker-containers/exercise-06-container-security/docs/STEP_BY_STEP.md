# Step-by-Step Implementation Guide: Container Security

## Overview

Secure Docker containers for production ML deployments. Learn vulnerability scanning, secret management, user permissions, network security, and compliance best practices.

**Time**: 2-3 hours | **Difficulty**: Intermediate to Advanced

---

## Learning Objectives

✅ Scan images for vulnerabilities
✅ Implement non-root users
✅ Manage secrets securely
✅ Configure resource limits
✅ Apply security best practices
✅ Use security scanning tools
✅ Implement least privilege principles

---

## Security Fundamentals

### 1. Non-Root Users

```dockerfile
# ❌ Bad: Running as root
FROM python:3.10-slim
COPY app.py /app/
CMD ["python", "/app/app.py"]

# ✅ Good: Non-root user
FROM python:3.10-slim

RUN useradd -m -u 1000 appuser && \
    mkdir /app && \
    chown appuser:appuser /app

USER appuser
WORKDIR /app

COPY --chown=appuser:appuser app.py .
CMD ["python", "app.py"]
```

### 2. Read-Only Filesystem

```yaml
services:
  api:
    image: ml-api
    read_only: true
    tmpfs:
      - /tmp
      - /var/run
```

### 3. Drop Capabilities

```yaml
services:
  api:
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE  # Only if needed
```

---

## Secret Management

### Using Docker Secrets

```yaml
version: '3.8'

services:
  api:
    image: ml-api
    secrets:
      - db_password
      - api_key

secrets:
  db_password:
    file: ./secrets/db_password.txt
  api_key:
    external: true
```

### Environment Variables (Less Secure)

```yaml
# .env file (gitignored)
API_KEY=secret-key-here
DB_PASSWORD=password-here

# docker-compose.yml
services:
  api:
    env_file: .env
```

---

## Vulnerability Scanning

### Docker Scan

```bash
# Scan image
docker scan ml-api:latest

# Scan and show details
docker scan --severity high ml-api:latest

# Scan with specific scanner
docker scan --file Dockerfile ml-api:latest
```

### Trivy Scanner

```bash
# Install Trivy
brew install aquasecurity/trivy/trivy

# Scan image
trivy image ml-api:latest

# Scan for HIGH and CRITICAL only
trivy image --severity HIGH,CRITICAL ml-api:latest

# Generate report
trivy image --format json --output report.json ml-api:latest
```

---

## Resource Limits

```yaml
services:
  trainer:
    image: ml-training
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G
          pids: 100
        reservations:
          memory: 8G
```

---

## Network Security

```yaml
# Isolate backend services
networks:
  frontend:
  backend:
    internal: true  # No internet access

services:
  api:
    networks:
      - frontend
      - backend

  database:
    networks:
      - backend  # Internal only
```

---

## Security Checklist

✅ Use official base images
✅ Keep images updated
✅ Scan for vulnerabilities
✅ Run as non-root user
✅ Use read-only filesystems
✅ Limit container capabilities
✅ Set resource limits
✅ Manage secrets properly
✅ Use security profiles (AppArmor/SELinux)
✅ Enable content trust
✅ Minimize attack surface

---

## Best Practices

```dockerfile
# 1. Minimal base image
FROM python:3.10-alpine  # Smaller attack surface

# 2. Multi-stage build
FROM python:3.10 AS builder
# Build dependencies
FROM python:3.10-slim
COPY --from=builder /app /app

# 3. No secrets in image
# Use runtime env vars or Docker secrets

# 4. Health checks
HEALTHCHECK CMD curl -f http://localhost/health || exit 1

# 5. Metadata
LABEL maintainer="team@example.com"
LABEL version="1.0"
```

---

**Container security implemented!** 🔒
