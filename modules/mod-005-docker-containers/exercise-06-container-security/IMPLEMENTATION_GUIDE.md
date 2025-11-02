# Implementation Guide: Container Security for ML Infrastructure

## Overview

This comprehensive implementation guide covers container security best practices with a focus on ML workloads. You'll learn to implement vulnerability scanning, secure container configurations, secrets management, runtime security, and production-ready security architectures for machine learning applications.

**Target Audience**: Junior AI Infrastructure Engineers
**Prerequisites**: Docker fundamentals, basic security concepts, Linux permissions
**Estimated Time**: 5-7 hours for complete implementation
**ML Focus Areas**: Model protection, API security, data privacy, secure model serving

---

## Table of Contents

1. [Container Security Fundamentals](#1-container-security-fundamentals)
2. [Image Vulnerability Scanning](#2-image-vulnerability-scanning)
3. [Running Containers as Non-Root](#3-running-containers-as-non-root)
4. [Secrets Management](#4-secrets-management)
5. [Security Scanning Automation](#5-security-scanning-automation)
6. [AppArmor and SELinux Profiles](#6-apparmor-and-selinux-profiles)
7. [Production ML Security](#7-production-ml-security)
8. [Advanced Security Topics](#8-advanced-security-topics)
9. [Troubleshooting Guide](#9-troubleshooting-guide)
10. [Security Checklist](#10-security-checklist)

---

## 1. Container Security Fundamentals

### 1.1 Understanding the Container Security Model

Container security operates at multiple layers of the stack:

**Security Layers**:
```
┌─────────────────────────────────────┐
│   Application Security              │  <- Code vulnerabilities
├─────────────────────────────────────┤
│   Container Runtime Security        │  <- Capability restrictions
├─────────────────────────────────────┤
│   Image Security                    │  <- Vulnerability scanning
├─────────────────────────────────────┤
│   Host Operating System             │  <- Kernel security
├─────────────────────────────────────┤
│   Network Security                  │  <- Firewall, segmentation
└─────────────────────────────────────┘
```

### 1.2 Principle of Least Privilege

Start by creating a basic security setup directory:

```bash
# Create working directory
mkdir -p ~/docker-security-exercises
cd ~/docker-security-exercises

# Create directory structure
mkdir -p {images,configs,secrets,scripts,policies}
```

**Insecure Dockerfile (DON'T USE)**:
```dockerfile
# BAD EXAMPLE - Security issues highlighted
FROM python:3.11

# Running as root (ISSUE #1)
WORKDIR /app

# Installing unnecessary packages (ISSUE #2)
RUN apt-get update && apt-get install -y \
    vim nano curl wget git ssh

# Copying with root ownership (ISSUE #3)
COPY . .

# Installing packages globally (ISSUE #4)
RUN pip install -r requirements.txt

# Exposing privileged port (ISSUE #5)
EXPOSE 80

# Running as root user (ISSUE #6)
CMD ["python", "app.py"]
```

**Secure Dockerfile (BEST PRACTICE)**:
```dockerfile
# GOOD EXAMPLE - Security best practices
FROM python:3.11-slim AS builder

# Install only build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# Minimal runtime image
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser -u 1000 -m -s /sbin/nologin appuser

# Install runtime dependencies only
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache /wheels/* && rm -rf /wheels

# Set working directory
WORKDIR /app

# Copy application with correct ownership
COPY --chown=appuser:appuser app.py .

# Switch to non-root user
USER appuser

# Use unprivileged port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# Run application
CMD ["python", "app.py"]
```

### 1.3 Container Hardening Basics

Create a secure configuration file:

```bash
cat > configs/secure-container.yml << 'EOF'
# Docker Compose security configuration
version: '3.8'

services:
  ml-api:
    build:
      context: .
      dockerfile: Dockerfile.secure

    # Security options
    security_opt:
      - no-new-privileges:true  # Prevent privilege escalation
      - apparmor=docker-default  # Apply AppArmor profile

    # Drop all capabilities, add only needed ones
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE  # Only if binding to port < 1024

    # Read-only root filesystem
    read_only: true

    # Temporary filesystems for writable directories
    tmpfs:
      - /tmp
      - /var/run:size=100M
      - /var/log:size=100M

    # Writable volume for application data
    volumes:
      - app-data:/app/data

    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
          pids: 100  # Limit number of processes

    # Logging
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
        labels: "service,environment"

volumes:
  app-data:
    driver: local
EOF
```

**Why Each Setting Matters**:
- `no-new-privileges`: Prevents processes from gaining more privileges than parent
- `cap_drop: ALL`: Removes all Linux capabilities (root-like permissions)
- `read_only: true`: Prevents modifications to container filesystem
- `pids: 100`: Prevents fork bombs and runaway processes
- Resource limits: Prevents resource exhaustion attacks

---

## 2. Image Vulnerability Scanning

### 2.1 Installing and Using Trivy

Trivy is a comprehensive vulnerability scanner for containers.

**Installation**:
```bash
# Install Trivy
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

# Verify installation
trivy --version
```

**Basic Image Scanning**:
```bash
# Scan a Docker image
trivy image python:3.11

# Scan with severity filter
trivy image --severity HIGH,CRITICAL python:3.11

# Output to JSON for CI/CD integration
trivy image --format json --output results.json python:3.11

# Scan specific image you built
docker build -t myapp:latest .
trivy image myapp:latest
```

**Expected Output**:
```
myapp:latest (alpine 3.18.4)
============================
Total: 5 (UNKNOWN: 0, LOW: 2, MEDIUM: 1, HIGH: 2, CRITICAL: 0)

┌─────────────┬────────────────┬──────────┬───────────────────┬───────────────┬──────────────────────────────────────┐
│   Library   │ Vulnerability  │ Severity │ Installed Version │ Fixed Version │                Title                 │
├─────────────┼────────────────┼──────────┼───────────────────┼───────────────┼──────────────────────────────────────┤
│ openssl     │ CVE-2023-1234  │   HIGH   │ 3.1.0-r1          │ 3.1.0-r2      │ OpenSSL security vulnerability       │
└─────────────┴────────────────┴──────────┴───────────────────┴───────────────┴──────────────────────────────────────┘
```

### 2.2 Advanced Trivy Scanning

Create a comprehensive scanning script:

```bash
cat > scripts/scan-image.sh << 'EOF'
#!/bin/bash

# Comprehensive image security scan
set -e

IMAGE_NAME="${1:-myapp:latest}"
SCAN_DIR="./scan-results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$SCAN_DIR"

echo "🔍 Starting comprehensive security scan for: $IMAGE_NAME"
echo "================================================"

# 1. Vulnerability scan
echo ""
echo "1️⃣  Scanning for vulnerabilities..."
trivy image \
  --severity HIGH,CRITICAL \
  --format json \
  --output "$SCAN_DIR/vulnerabilities_${TIMESTAMP}.json" \
  "$IMAGE_NAME"

trivy image \
  --severity HIGH,CRITICAL \
  --format table \
  "$IMAGE_NAME" | tee "$SCAN_DIR/vulnerabilities_${TIMESTAMP}.txt"

# 2. Configuration scan
echo ""
echo "2️⃣  Scanning for misconfigurations..."
trivy config \
  --format json \
  --output "$SCAN_DIR/misconfig_${TIMESTAMP}.json" \
  .

# 3. Secret scan
echo ""
echo "3️⃣  Scanning for exposed secrets..."
trivy image \
  --scanners secret \
  --format json \
  --output "$SCAN_DIR/secrets_${TIMESTAMP}.json" \
  "$IMAGE_NAME"

# 4. License scan
echo ""
echo "4️⃣  Scanning for license issues..."
trivy image \
  --scanners license \
  --severity HIGH,CRITICAL \
  "$IMAGE_NAME" | tee "$SCAN_DIR/licenses_${TIMESTAMP}.txt"

# 5. Generate summary
echo ""
echo "📊 Generating summary report..."
cat > "$SCAN_DIR/summary_${TIMESTAMP}.md" << SUMMARY
# Security Scan Summary

**Image**: $IMAGE_NAME
**Scan Date**: $(date)
**Scan ID**: $TIMESTAMP

## Results

- Vulnerabilities: See \`vulnerabilities_${TIMESTAMP}.json\`
- Misconfigurations: See \`misconfig_${TIMESTAMP}.json\`
- Secrets: See \`secrets_${TIMESTAMP}.json\`
- Licenses: See \`licenses_${TIMESTAMP}.txt\`

## Critical Findings

SUMMARY

# Extract critical vulnerabilities
jq -r '.Results[].Vulnerabilities[] | select(.Severity=="CRITICAL") | "- [\(.VulnerabilityID)] \(.Title)"' \
  "$SCAN_DIR/vulnerabilities_${TIMESTAMP}.json" >> "$SCAN_DIR/summary_${TIMESTAMP}.md" 2>/dev/null || true

echo ""
echo "✅ Scan complete! Results saved to: $SCAN_DIR"
echo "📄 Summary report: $SCAN_DIR/summary_${TIMESTAMP}.md"
EOF

chmod +x scripts/scan-image.sh
```

**Usage**:
```bash
# Scan your image
./scripts/scan-image.sh myapp:latest

# Scan base image before using
./scripts/scan-image.sh python:3.11-slim

# Compare scan results
./scripts/scan-image.sh python:3.11-slim
./scripts/scan-image.sh python:3.11-alpine
```

### 2.3 Docker Scout Integration

Docker Scout provides native vulnerability scanning:

```bash
# Enable Docker Scout
docker scout quickview

# Analyze image
docker scout cves myapp:latest

# Compare with base image
docker scout compare --to python:3.11-slim myapp:latest

# Get recommendations
docker scout recommendations myapp:latest

# Check policy compliance
docker scout policy myapp:latest
```

### 2.4 Clair Scanner Setup

Clair is another popular vulnerability scanner:

```bash
# Start Clair service
cat > configs/docker-compose-clair.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: password
      POSTGRES_DB: clair
    volumes:
      - clair-db:/var/lib/postgresql/data

  clair:
    image: quay.io/coreos/clair:latest
    ports:
      - "6060:6060"
      - "6061:6061"
    depends_on:
      - postgres
    volumes:
      - ./clair-config.yaml:/config/config.yaml
    command: ["-config", "/config/config.yaml"]

volumes:
  clair-db:
EOF

# Start Clair
docker compose -f configs/docker-compose-clair.yml up -d

# Scan image with Clair
docker run --rm \
  --network host \
  quay.io/coreos/clair-scanner \
  --clair=http://localhost:6060 \
  --ip=localhost \
  myapp:latest
```

---

## 3. Running Containers as Non-Root

### 3.1 Creating Non-Root User in Dockerfile

**Complete Example**:

```dockerfile
# images/Dockerfile.nonroot
FROM python:3.11-slim

# Create user and group with specific UID/GID
RUN groupadd -r mluser -g 1000 && \
    useradd -r -u 1000 -g mluser -m -s /bin/bash mluser

# Create application directory
RUN mkdir -p /app /app/data /app/models /app/logs && \
    chown -R mluser:mluser /app

# Install dependencies as root
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Switch to working directory
WORKDIR /app

# Copy application files with correct ownership
COPY --chown=mluser:mluser app.py .
COPY --chown=mluser:mluser models/ ./models/

# Switch to non-root user (CRITICAL STEP)
USER mluser

# Expose non-privileged port
EXPOSE 8080

# Run application
CMD ["python", "app.py"]
```

**Create Test Application**:

```python
# app.py
import os
import sys
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'user': os.getenv('USER', 'unknown'),
        'uid': os.getuid(),
        'gid': os.getgid()
    })

@app.route('/security-info')
def security_info():
    return jsonify({
        'user': os.getenv('USER', 'unknown'),
        'uid': os.getuid(),
        'gid': os.getgid(),
        'home': os.path.expanduser('~'),
        'writable_dirs': [
            '/tmp' if os.access('/tmp', os.W_OK) else None,
            '/app' if os.access('/app', os.W_OK) else None,
            '/app/data' if os.access('/app/data', os.W_OK) else None,
        ]
    })

if __name__ == '__main__':
    print(f"Running as UID: {os.getuid()}, GID: {os.getgid()}")
    app.run(host='0.0.0.0', port=8080)
```

```txt
# requirements.txt
flask==3.0.0
gunicorn==21.2.0
```

**Build and Test**:

```bash
# Build image
docker build -f images/Dockerfile.nonroot -t secure-ml-app:latest .

# Run container
docker run -d --name test-security -p 8080:8080 secure-ml-app:latest

# Check user context
docker exec test-security id
# Output: uid=1000(mluser) gid=1000(mluser) groups=1000(mluser)

# Test security endpoint
curl http://localhost:8080/security-info
# Should show UID 1000, not 0 (root)

# Verify cannot write to root filesystem
docker exec test-security touch /etc/test-file
# Should fail with permission denied

# Cleanup
docker stop test-security
docker rm test-security
```

### 3.2 User Namespaces

User namespaces remap container UIDs to different host UIDs:

```bash
# Enable user namespace remapping in daemon.json
sudo cat > /etc/docker/daemon.json << 'EOF'
{
  "userns-remap": "default"
}
EOF

# Restart Docker daemon
sudo systemctl restart docker

# Now root in container (UID 0) maps to unprivileged UID on host
docker run --rm alpine id
# Container sees: uid=0(root)
# But host sees: uid=100000(dockremap)
```

### 3.3 Handling File Permissions

When volumes are involved, permission management becomes critical:

```bash
# Create volume with proper permissions
cat > configs/docker-compose-permissions.yml << 'EOF'
version: '3.8'

services:
  ml-app:
    build:
      context: .
      dockerfile: images/Dockerfile.nonroot
    user: "1000:1000"  # Explicit user specification
    volumes:
      - ./data:/app/data:rw
      - ./models:/app/models:ro  # Read-only models
      - ./logs:/app/logs:rw
    tmpfs:
      - /tmp:uid=1000,gid=1000,mode=1777
EOF
```

**Permission Setup Script**:

```bash
cat > scripts/setup-permissions.sh << 'EOF'
#!/bin/bash

# Create directories with proper ownership
mkdir -p data models logs

# Set ownership to match container user
sudo chown -R 1000:1000 data logs
sudo chown -R 1000:1000 models

# Set appropriate permissions
chmod 755 data models logs
chmod 644 models/*

echo "✅ Permissions configured for UID/GID 1000"
ls -la data models logs
EOF

chmod +x scripts/setup-permissions.sh
./scripts/setup-permissions.sh
```

---

## 4. Secrets Management

### 4.1 Docker Secrets (Swarm Mode)

Docker secrets provide native secret management in Swarm mode:

```bash
# Initialize Swarm (required for secrets)
docker swarm init

# Create secrets from files
echo "super_secret_password_123" | docker secret create db_password -
echo "my-api-key-xyz789" | docker secret create api_key -
echo "sk-openai-key-123abc" | docker secret create openai_api_key -

# List secrets
docker secret ls

# Inspect secret (shows metadata, not content)
docker secret inspect db_password
```

**Using Secrets in Services**:

```bash
cat > configs/docker-compose-secrets.yml << 'EOF'
version: '3.8'

services:
  ml-api:
    image: secure-ml-app:latest
    secrets:
      - db_password
      - api_key
      - openai_api_key
    environment:
      # Point to secret files
      - DB_PASSWORD_FILE=/run/secrets/db_password
      - API_KEY_FILE=/run/secrets/api_key
      - OPENAI_API_KEY_FILE=/run/secrets/openai_api_key
    deploy:
      replicas: 2

  postgres:
    image: postgres:15
    secrets:
      - db_password
    environment:
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password

secrets:
  db_password:
    external: true
  api_key:
    external: true
  openai_api_key:
    external: true
EOF

# Deploy stack with secrets
docker stack deploy -c configs/docker-compose-secrets.yml ml-stack
```

**Reading Secrets in Application**:

```python
# secure_app.py
import os

def read_secret(secret_name):
    """Read secret from Docker secret file"""
    secret_file = os.getenv(f'{secret_name.upper()}_FILE')
    if secret_file and os.path.exists(secret_file):
        with open(secret_file, 'r') as f:
            return f.read().strip()
    # Fallback to environment variable (for dev)
    return os.getenv(secret_name.upper())

# Usage
db_password = read_secret('db_password')
api_key = read_secret('api_key')
openai_api_key = read_secret('openai_api_key')

# Never log secrets!
print(f"DB Password: {'*' * len(db_password)}")
```

### 4.2 Environment-Specific Secrets

Manage secrets for different environments:

```bash
# Create directory structure
mkdir -p secrets/{dev,staging,prod}

# Development secrets
cat > secrets/dev/db_password.txt << 'EOF'
dev_password_123
EOF

cat > secrets/dev/api_key.txt << 'EOF'
dev_api_key_xyz
EOF

# Production secrets (example - use proper secret management)
cat > secrets/prod/db_password.txt << 'EOF'
REPLACE_WITH_ACTUAL_PROD_PASSWORD
EOF

cat > secrets/prod/api_key.txt << 'EOF'
REPLACE_WITH_ACTUAL_PROD_API_KEY
EOF

# Secure the secrets directory
chmod 700 secrets
chmod 600 secrets/*/*.txt

# Add to .gitignore
echo "secrets/" >> .gitignore
```

**Environment-Aware Compose File**:

```bash
cat > configs/docker-compose-env-secrets.yml << 'EOF'
version: '3.8'

services:
  ml-api:
    build: .
    secrets:
      - source: db_password
        target: /run/secrets/db_password
        mode: 0400
      - source: api_key
        target: /run/secrets/api_key
        mode: 0400
    environment:
      - ENVIRONMENT=${ENVIRONMENT:-dev}

secrets:
  db_password:
    file: ./secrets/${ENVIRONMENT:-dev}/db_password.txt
  api_key:
    file: ./secrets/${ENVIRONMENT:-dev}/api_key.txt
EOF

# Run with different environments
ENVIRONMENT=dev docker compose -f configs/docker-compose-env-secrets.yml up -d
ENVIRONMENT=prod docker compose -f configs/docker-compose-env-secrets.yml up -d
```

### 4.3 HashiCorp Vault Integration

For production systems, integrate with HashiCorp Vault:

**Vault Setup**:

```bash
# Start Vault server (dev mode - not for production!)
docker run -d \
  --name vault \
  --cap-add=IPC_LOCK \
  -p 8200:8200 \
  -e 'VAULT_DEV_ROOT_TOKEN_ID=dev-token' \
  vault:latest

# Wait for Vault to start
sleep 5

# Set environment variables
export VAULT_ADDR='http://localhost:8200'
export VAULT_TOKEN='dev-token'

# Store secrets in Vault
docker exec vault vault kv put secret/ml-app \
  db_password="vault_secret_password" \
  api_key="vault_api_key_xyz" \
  openai_key="sk-vault-key-123"

# Read secret
docker exec vault vault kv get secret/ml-app
```

**Python Application with Vault**:

```python
# vault_app.py
import os
import hvac

class VaultSecretManager:
    def __init__(self):
        self.vault_url = os.getenv('VAULT_ADDR', 'http://vault:8200')
        self.vault_token = os.getenv('VAULT_TOKEN')

        if not self.vault_token:
            # Try to read token from file
            token_file = '/run/secrets/vault_token'
            if os.path.exists(token_file):
                with open(token_file, 'r') as f:
                    self.vault_token = f.read().strip()

        self.client = hvac.Client(
            url=self.vault_url,
            token=self.vault_token
        )

    def get_secret(self, path, key):
        """Retrieve secret from Vault"""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=path
            )
            return response['data']['data'].get(key)
        except Exception as e:
            print(f"Error reading secret from Vault: {e}")
            return None

    def get_all_secrets(self, path):
        """Retrieve all secrets from a path"""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=path
            )
            return response['data']['data']
        except Exception as e:
            print(f"Error reading secrets from Vault: {e}")
            return {}

# Usage
vault = VaultSecretManager()
secrets = vault.get_all_secrets('ml-app')

db_password = secrets.get('db_password')
api_key = secrets.get('api_key')
openai_key = secrets.get('openai_key')
```

**Docker Compose with Vault**:

```bash
cat > configs/docker-compose-vault.yml << 'EOF'
version: '3.8'

services:
  vault:
    image: vault:latest
    ports:
      - "8200:8200"
    environment:
      - VAULT_DEV_ROOT_TOKEN_ID=dev-token
      - VAULT_DEV_LISTEN_ADDRESS=0.0.0.0:8200
    cap_add:
      - IPC_LOCK

  ml-api:
    build: .
    depends_on:
      - vault
    environment:
      - VAULT_ADDR=http://vault:8200
      - VAULT_TOKEN=dev-token
    secrets:
      - vault_token
    command: python vault_app.py

secrets:
  vault_token:
    file: ./secrets/vault_token.txt
EOF
```

### 4.4 Secret Rotation Strategy

Implement automatic secret rotation:

```bash
cat > scripts/rotate-secrets.sh << 'EOF'
#!/bin/bash

# Secret rotation script
set -e

SECRET_NAME="${1}"
NEW_SECRET="${2}"
STACK_NAME="${3:-ml-stack}"

if [ -z "$SECRET_NAME" ] || [ -z "$NEW_SECRET" ]; then
    echo "Usage: $0 <secret_name> <new_secret> [stack_name]"
    exit 1
fi

echo "🔄 Rotating secret: $SECRET_NAME"

# Create new secret with version suffix
NEW_SECRET_NAME="${SECRET_NAME}_v$(date +%s)"
echo "$NEW_SECRET" | docker secret create "$NEW_SECRET_NAME" -

# Update service to use new secret
docker service update \
  --secret-rm "$SECRET_NAME" \
  --secret-add "source=$NEW_SECRET_NAME,target=/run/secrets/$SECRET_NAME" \
  "${STACK_NAME}_ml-api"

# Wait for rollout
echo "⏳ Waiting for service update..."
sleep 10

# Remove old secret
docker secret rm "$SECRET_NAME"

# Create new secret with original name
echo "$NEW_SECRET" | docker secret create "$SECRET_NAME" -

# Update service again to use standard name
docker service update \
  --secret-rm "$NEW_SECRET_NAME" \
  --secret-add "source=$SECRET_NAME,target=/run/secrets/$SECRET_NAME" \
  "${STACK_NAME}_ml-api"

# Remove temporary secret
docker secret rm "$NEW_SECRET_NAME"

echo "✅ Secret rotation complete"
EOF

chmod +x scripts/rotate-secrets.sh
```

---

## 5. Security Scanning Automation

### 5.1 CI/CD Integration with Trivy

**GitHub Actions Workflow**:

```yaml
# .github/workflows/security-scan.yml
name: Container Security Scan

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly scan

jobs:
  security-scan:
    runs-on: ubuntu-latest

    permissions:
      contents: read
      security-events: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Build Docker image
        run: docker build -t ${{ github.repository }}:${{ github.sha }} .

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: '${{ github.repository }}:${{ github.sha }}'
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'

      - name: Upload Trivy results to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'

      - name: Run Trivy configuration scan
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'config'
          format: 'table'
          exit-code: '1'
          severity: 'CRITICAL,HIGH'

      - name: Check for secrets
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: '${{ github.repository }}:${{ github.sha }}'
          scanners: 'secret'
          format: 'table'
          exit-code: '1'

      - name: Generate security report
        if: always()
        run: |
          trivy image \
            --format json \
            --output security-report.json \
            ${{ github.repository }}:${{ github.sha }}

      - name: Upload security report
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: security-report
          path: security-report.json
```

### 5.2 GitLab CI Security Pipeline

```yaml
# .gitlab-ci.yml
stages:
  - build
  - security
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  IMAGE: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker build -t $IMAGE .
    - docker push $IMAGE

trivy-scan:
  stage: security
  image: aquasec/trivy:latest
  script:
    - trivy image --exit-code 0 --severity HIGH,CRITICAL $IMAGE
    - trivy image --format json --output gl-container-scanning-report.json $IMAGE
  artifacts:
    reports:
      container_scanning: gl-container-scanning-report.json
    paths:
      - gl-container-scanning-report.json
    expire_in: 1 week

secret-detection:
  stage: security
  image: aquasec/trivy:latest
  script:
    - trivy fs --scanners secret --exit-code 1 .
  allow_failure: false

sast-scan:
  stage: security
  image: python:3.11
  script:
    - pip install bandit safety
    - bandit -r . -f json -o bandit-report.json || true
    - safety check --json > safety-report.json || true
  artifacts:
    paths:
      - bandit-report.json
      - safety-report.json
    expire_in: 1 week

deploy:
  stage: deploy
  script:
    - echo "Deploy only if security scans pass"
  only:
    - main
  when: on_success
```

### 5.3 Pre-Commit Security Hooks

```bash
# Install pre-commit
pip install pre-commit

# Create pre-commit configuration
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: detect-private-key
      - id: check-yaml
      - id: check-json

  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']

  - repo: local
    hooks:
      - id: trivy-config
        name: Trivy configuration scan
        entry: trivy config .
        language: system
        pass_filenames: false

      - id: dockerfile-lint
        name: Dockerfile linting
        entry: docker run --rm -i hadolint/hadolint
        language: system
        files: Dockerfile.*
EOF

# Install hooks
pre-commit install

# Initialize secrets baseline
detect-secrets scan > .secrets.baseline

# Run manually
pre-commit run --all-files
```

### 5.4 Automated Security Reporting

```bash
cat > scripts/security-report.sh << 'EOF'
#!/bin/bash

# Generate comprehensive security report
set -e

REPORT_DIR="./security-reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
IMAGE_NAME="${1:-myapp:latest}"

mkdir -p "$REPORT_DIR"

echo "📊 Generating Security Report for: $IMAGE_NAME"
echo "=============================================="

# HTML Report header
cat > "$REPORT_DIR/report_${TIMESTAMP}.html" << HTML
<!DOCTYPE html>
<html>
<head>
    <title>Security Report - $IMAGE_NAME</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .critical { color: #d9534f; }
        .high { color: #f0ad4e; }
        .medium { color: #5bc0de; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
    </style>
</head>
<body>
    <h1>Container Security Report</h1>
    <p><strong>Image:</strong> $IMAGE_NAME</p>
    <p><strong>Scan Date:</strong> $(date)</p>
    <h2>Vulnerability Summary</h2>
HTML

# Run scans and generate report
trivy image --format json --output /tmp/trivy.json "$IMAGE_NAME"

# Extract summary
CRITICAL=$(jq '[.Results[].Vulnerabilities[]? | select(.Severity=="CRITICAL")] | length' /tmp/trivy.json)
HIGH=$(jq '[.Results[].Vulnerabilities[]? | select(.Severity=="HIGH")] | length' /tmp/trivy.json)
MEDIUM=$(jq '[.Results[].Vulnerabilities[]? | select(.Severity=="MEDIUM")] | length' /tmp/trivy.json)
LOW=$(jq '[.Results[].Vulnerabilities[]? | select(.Severity=="LOW")] | length' /tmp/trivy.json)

cat >> "$REPORT_DIR/report_${TIMESTAMP}.html" << HTML
    <table>
        <tr>
            <th>Severity</th>
            <th>Count</th>
        </tr>
        <tr class="critical">
            <td>CRITICAL</td>
            <td>$CRITICAL</td>
        </tr>
        <tr class="high">
            <td>HIGH</td>
            <td>$HIGH</td>
        </tr>
        <tr class="medium">
            <td>MEDIUM</td>
            <td>$MEDIUM</td>
        </tr>
        <tr>
            <td>LOW</td>
            <td>$LOW</td>
        </tr>
    </table>

    <h2>Critical Vulnerabilities</h2>
    <table>
        <tr>
            <th>CVE</th>
            <th>Package</th>
            <th>Severity</th>
            <th>Installed</th>
            <th>Fixed</th>
        </tr>
HTML

# Add critical vulnerabilities to HTML
jq -r '.Results[].Vulnerabilities[]? | select(.Severity=="CRITICAL" or .Severity=="HIGH") |
    "<tr><td>\(.VulnerabilityID)</td><td>\(.PkgName)</td><td>\(.Severity)</td><td>\(.InstalledVersion)</td><td>\(.FixedVersion // "N/A")</td></tr>"' \
    /tmp/trivy.json >> "$REPORT_DIR/report_${TIMESTAMP}.html"

cat >> "$REPORT_DIR/report_${TIMESTAMP}.html" << HTML
    </table>
</body>
</html>
HTML

echo "✅ Report generated: $REPORT_DIR/report_${TIMESTAMP}.html"

# Generate JSON summary
jq '{
    image: "'$IMAGE_NAME'",
    scan_date: "'$(date -Iseconds)'",
    summary: {
        critical: '$CRITICAL',
        high: '$HIGH',
        medium: '$MEDIUM',
        low: '$LOW',
        total: ('$CRITICAL' + '$HIGH' + '$MEDIUM' + '$LOW')
    }
}' /tmp/trivy.json > "$REPORT_DIR/summary_${TIMESTAMP}.json"

echo "✅ Summary generated: $REPORT_DIR/summary_${TIMESTAMP}.json"

# Fail build if critical vulnerabilities found
if [ "$CRITICAL" -gt 0 ]; then
    echo "❌ FAIL: $CRITICAL critical vulnerabilities found!"
    exit 1
fi

echo "✅ PASS: No critical vulnerabilities found"
EOF

chmod +x scripts/security-report.sh
```

---

## 6. AppArmor and SELinux Profiles

### 6.1 AppArmor Profile for ML Containers

AppArmor provides mandatory access control:

```bash
# Check if AppArmor is enabled
sudo aa-status

# Create custom AppArmor profile
sudo cat > /etc/apparmor.d/docker-ml-secure << 'EOF'
#include <tunables/global>

profile docker-ml-secure flags=(attach_disconnected,mediate_deleted) {
  #include <abstractions/base>

  # Network access
  network inet tcp,
  network inet udp,
  network inet icmp,

  # Deny dangerous capabilities
  deny capability sys_admin,
  deny capability sys_module,
  deny capability sys_rawio,
  deny capability sys_ptrace,
  deny capability sys_boot,
  deny capability mac_admin,
  deny capability mac_override,

  # Allow necessary capabilities
  capability chown,
  capability dac_override,
  capability setgid,
  capability setuid,
  capability net_bind_service,

  # File system permissions
  /app/** rw,
  /app/data/** rw,
  /app/models/** r,
  /tmp/** rw,
  /var/log/** w,

  # Read-only access to system files
  /etc/passwd r,
  /etc/group r,
  /etc/hosts r,
  /etc/resolv.conf r,

  # Python and libraries
  /usr/local/lib/python*/** r,
  /usr/lib/python*/** r,

  # Deny access to sensitive files
  deny /etc/shadow r,
  deny /etc/sudoers r,
  deny /root/** rw,
  deny /home/** rw,
  deny /var/run/docker.sock rw,

  # Deny kernel module loading
  deny @{PROC}/sys/kernel/modules/** w,

  # Allow proc access for monitoring
  @{PROC}/cpuinfo r,
  @{PROC}/meminfo r,
  @{PROC}/stat r,
  @{PROC}/uptime r,
  @{PROC}/loadavg r,
}
EOF

# Load the profile
sudo apparmor_parser -r /etc/apparmor.d/docker-ml-secure

# Verify profile is loaded
sudo aa-status | grep docker-ml-secure
```

**Using AppArmor Profile**:

```bash
# Run container with custom AppArmor profile
docker run -d \
  --name ml-secure \
  --security-opt apparmor=docker-ml-secure \
  secure-ml-app:latest

# In Docker Compose
cat > configs/docker-compose-apparmor.yml << 'EOF'
version: '3.8'

services:
  ml-api:
    image: secure-ml-app:latest
    security_opt:
      - apparmor=docker-ml-secure
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
EOF
```

### 6.2 SELinux Configuration

For Red Hat-based systems:

```bash
# Check SELinux status
sestatus

# Set SELinux to enforcing mode
sudo setenforce 1

# Create custom SELinux policy for containers
sudo cat > docker-ml.te << 'EOF'
module docker-ml 1.0;

require {
    type container_t;
    type container_file_t;
    class file { read write create unlink };
    class dir { read write add_name remove_name };
}

# Allow container to manage its files
allow container_t container_file_t:file { read write create unlink };
allow container_t container_file_t:dir { read write add_name remove_name };
EOF

# Compile and load policy
checkmodule -M -m -o docker-ml.mod docker-ml.te
semodule_package -o docker-ml.pp -m docker-ml.mod
sudo semodule -i docker-ml.pp

# Label volumes appropriately
sudo chcon -R -t container_file_t /path/to/data
```

**Docker with SELinux**:

```bash
# Run with SELinux label
docker run -d \
  --name ml-selinux \
  --security-opt label=type:container_t \
  -v /data:/app/data:z \  # :z relabels for container access
  secure-ml-app:latest

# In Docker Compose
cat > configs/docker-compose-selinux.yml << 'EOF'
version: '3.8'

services:
  ml-api:
    image: secure-ml-app:latest
    security_opt:
      - label=type:container_t
    volumes:
      - /data:/app/data:z
      - /models:/app/models:ro,z
EOF
```

### 6.3 Seccomp Profiles

Seccomp filters system calls:

```bash
# Create custom seccomp profile
cat > policies/seccomp-ml.json << 'EOF'
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": [
    "SCMP_ARCH_X86_64",
    "SCMP_ARCH_X86",
    "SCMP_ARCH_X32"
  ],
  "syscalls": [
    {
      "names": [
        "accept",
        "accept4",
        "access",
        "arch_prctl",
        "bind",
        "brk",
        "chmod",
        "chown",
        "clone",
        "close",
        "connect",
        "dup",
        "dup2",
        "epoll_create",
        "epoll_ctl",
        "epoll_wait",
        "execve",
        "exit",
        "exit_group",
        "fchmod",
        "fchown",
        "fcntl",
        "fstat",
        "futex",
        "getcwd",
        "getdents",
        "getegid",
        "geteuid",
        "getgid",
        "getpid",
        "getppid",
        "getrlimit",
        "getsockname",
        "getsockopt",
        "gettid",
        "getuid",
        "listen",
        "lseek",
        "madvise",
        "mkdir",
        "mmap",
        "mprotect",
        "munmap",
        "open",
        "openat",
        "poll",
        "read",
        "readlink",
        "recvfrom",
        "recvmsg",
        "rt_sigaction",
        "rt_sigprocmask",
        "rt_sigreturn",
        "select",
        "sendmsg",
        "sendto",
        "set_robust_list",
        "set_tid_address",
        "setgid",
        "setuid",
        "socket",
        "socketpair",
        "stat",
        "uname",
        "unlink",
        "wait4",
        "write"
      ],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
EOF

# Use seccomp profile
docker run -d \
  --security-opt seccomp=policies/seccomp-ml.json \
  secure-ml-app:latest
```

---

## 7. Production ML Security

### 7.1 Secure ML Model Serving

Complete example for secure ML API:

```python
# secure_ml_api.py
import os
import logging
from flask import Flask, request, jsonify
from functools import wraps
import hashlib
import hmac
import time

app = Flask(__name__)

# Configure logging (no secrets!)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load API key from secret
def load_api_key():
    key_file = os.getenv('API_KEY_FILE', '/run/secrets/api_key')
    if os.path.exists(key_file):
        with open(key_file, 'r') as f:
            return f.read().strip()
    return os.getenv('API_KEY', 'dev-key-change-me')

API_KEY = load_api_key()

# Authentication decorator
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('X-API-Key')

        if not auth_header:
            logger.warning(f"Missing API key from {request.remote_addr}")
            return jsonify({'error': 'Missing API key'}), 401

        if not hmac.compare_digest(auth_header, API_KEY):
            logger.warning(f"Invalid API key from {request.remote_addr}")
            return jsonify({'error': 'Invalid API key'}), 403

        return f(*args, **kwargs)
    return decorated_function

# Rate limiting (simple implementation)
request_counts = {}

def rate_limit(max_per_minute=60):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            current_minute = int(time.time() / 60)
            key = f"{client_ip}:{current_minute}"

            request_counts[key] = request_counts.get(key, 0) + 1

            if request_counts[key] > max_per_minute:
                logger.warning(f"Rate limit exceeded for {client_ip}")
                return jsonify({'error': 'Rate limit exceeded'}), 429

            # Cleanup old entries
            for k in list(request_counts.keys()):
                if not k.endswith(str(current_minute)):
                    del request_counts[k]

            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/health')
def health():
    """Public health check endpoint"""
    return jsonify({'status': 'healthy', 'version': '1.0.0'})

@app.route('/predict', methods=['POST'])
@require_api_key
@rate_limit(max_per_minute=100)
def predict():
    """Secured prediction endpoint"""
    try:
        data = request.get_json()

        # Input validation
        if not data or 'input' not in data:
            return jsonify({'error': 'Invalid input'}), 400

        # Log request (sanitized)
        logger.info(f"Prediction request from {request.remote_addr}")

        # TODO: Add your model inference here
        result = {
            'prediction': 'dummy_result',
            'confidence': 0.95
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/model-info')
@require_api_key
def model_info():
    """Return model metadata (no sensitive info)"""
    return jsonify({
        'model_name': 'secure-ml-model',
        'version': '1.0.0',
        'input_shape': [28, 28],
        'output_classes': 10
    })

if __name__ == '__main__':
    # Never run with debug=True in production!
    app.run(
        host='0.0.0.0',
        port=8080,
        debug=False
    )
```

**Secure Dockerfile for ML API**:

```dockerfile
# images/Dockerfile.ml-secure
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels \
    -r requirements.txt

# Runtime stage
FROM python:3.11-slim

# Security: Create non-root user
RUN groupadd -r mluser -g 1000 && \
    useradd -r -u 1000 -g mluser -m -s /sbin/nologin mluser

# Install runtime dependencies
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache /wheels/* && rm -rf /wheels

# Create necessary directories
RUN mkdir -p /app/models /app/logs && \
    chown -R mluser:mluser /app

WORKDIR /app

# Copy application
COPY --chown=mluser:mluser secure_ml_api.py .
COPY --chown=mluser:mluser models/ ./models/

# Switch to non-root user
USER mluser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Expose non-privileged port
EXPOSE 8080

# Run with Gunicorn (production WSGI server)
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8080", \
     "--workers", "4", \
     "--timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "secure_ml_api:app"]
```

### 7.2 Data Privacy and Encryption

```python
# data_privacy.py
import os
from cryptography.fernet import Fernet
import json

class SecureDataHandler:
    def __init__(self):
        # Load encryption key from secret
        key_file = os.getenv('ENCRYPTION_KEY_FILE', '/run/secrets/encryption_key')
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                self.key = f.read()
        else:
            # Generate new key (development only!)
            self.key = Fernet.generate_key()

        self.cipher = Fernet(self.key)

    def encrypt_data(self, data):
        """Encrypt sensitive data"""
        if isinstance(data, dict):
            data = json.dumps(data)
        return self.cipher.encrypt(data.encode())

    def decrypt_data(self, encrypted_data):
        """Decrypt sensitive data"""
        decrypted = self.cipher.decrypt(encrypted_data).decode()
        try:
            return json.loads(decrypted)
        except:
            return decrypted

    def anonymize_user_data(self, user_data):
        """Anonymize user data for ML training"""
        import hashlib

        anonymized = user_data.copy()

        # Hash PII fields
        if 'user_id' in anonymized:
            anonymized['user_id'] = hashlib.sha256(
                anonymized['user_id'].encode()
            ).hexdigest()[:16]

        # Remove sensitive fields
        for field in ['email', 'phone', 'ssn', 'address']:
            anonymized.pop(field, None)

        return anonymized

# Usage
handler = SecureDataHandler()

# Encrypt sensitive model inputs
sensitive_data = {'user_id': '12345', 'medical_history': 'sensitive'}
encrypted = handler.encrypt_data(sensitive_data)

# Store only encrypted data
# ...

# Decrypt when needed
decrypted = handler.decrypt_data(encrypted)
```

### 7.3 Secure Model Storage

```bash
# Create encrypted model storage
cat > scripts/setup-encrypted-storage.sh << 'EOF'
#!/bin/bash

set -e

MOUNT_POINT="/app/secure-models"
ENCRYPTED_FILE="/var/encrypted-models.img"
SIZE_MB=1000

echo "🔒 Setting up encrypted model storage..."

# Create encrypted file
sudo dd if=/dev/zero of=$ENCRYPTED_FILE bs=1M count=$SIZE_MB

# Setup encryption
sudo cryptsetup luksFormat $ENCRYPTED_FILE

# Open encrypted volume
sudo cryptsetup luksOpen $ENCRYPTED_FILE secure-models

# Create filesystem
sudo mkfs.ext4 /dev/mapper/secure-models

# Create mount point
sudo mkdir -p $MOUNT_POINT

# Mount
sudo mount /dev/mapper/secure-models $MOUNT_POINT

# Set permissions
sudo chown 1000:1000 $MOUNT_POINT

echo "✅ Encrypted storage ready at: $MOUNT_POINT"
EOF

chmod +x scripts/setup-encrypted-storage.sh
```

### 7.4 Production ML Security Checklist

```bash
cat > docs/ml-security-checklist.md << 'EOF'
# Production ML Security Checklist

## Image Security
- [ ] Base image from trusted registry
- [ ] Base image regularly updated
- [ ] Multi-stage build to minimize attack surface
- [ ] No secrets in image layers
- [ ] Vulnerability scan passed (0 CRITICAL, 0 HIGH)
- [ ] Image signed and verified

## Runtime Security
- [ ] Running as non-root user (UID 1000)
- [ ] Read-only root filesystem
- [ ] Capabilities dropped (cap_drop: ALL)
- [ ] Only necessary capabilities added
- [ ] Resource limits configured (CPU, memory, PIDs)
- [ ] AppArmor/SELinux profile applied
- [ ] Seccomp profile applied

## Network Security
- [ ] Network policies configured
- [ ] Only necessary ports exposed
- [ ] TLS/SSL encryption enabled
- [ ] API authentication required
- [ ] Rate limiting implemented
- [ ] CORS properly configured

## Data Security
- [ ] Secrets managed via Docker secrets or Vault
- [ ] Sensitive data encrypted at rest
- [ ] Sensitive data encrypted in transit
- [ ] PII data anonymized for ML training
- [ ] Model files access-controlled
- [ ] Logs don't contain secrets

## Access Control
- [ ] API key authentication
- [ ] Role-based access control (RBAC)
- [ ] Audit logging enabled
- [ ] Failed authentication attempts logged
- [ ] Session management implemented

## Monitoring & Compliance
- [ ] Security scanning automated in CI/CD
- [ ] Runtime security monitoring
- [ ] Anomaly detection configured
- [ ] Compliance requirements met (GDPR, HIPAA, etc.)
- [ ] Incident response plan documented
- [ ] Security updates automated

## Model Protection
- [ ] Model versioning implemented
- [ ] Model access logged
- [ ] Model tampering detection
- [ ] Intellectual property protected
- [ ] Model extraction attacks mitigated

## Testing
- [ ] Penetration testing completed
- [ ] Dependency vulnerabilities checked
- [ ] Security regression tests in CI/CD
- [ ] Disaster recovery plan tested
EOF
```

---

## 8. Advanced Security Topics

### 8.1 Runtime Security Monitoring with Falco

```bash
# Install Falco
curl -s https://falco.org/repo/falcosecurity-3672BA8F.asc | sudo apt-key add -
echo "deb https://download.falco.org/packages/deb stable main" | \
    sudo tee -a /etc/apt/sources.list.d/falcosecurity.list
sudo apt-get update
sudo apt-get install -y falco

# Custom Falco rules for ML containers
sudo cat > /etc/falco/rules.d/ml-containers.yaml << 'EOF'
- rule: Unauthorized Process in ML Container
  desc: Detect unexpected processes in ML containers
  condition: >
    container and
    container.image.repository = "secure-ml-app" and
    not proc.name in (python, gunicorn, sh, bash)
  output: >
    Unexpected process in ML container
    (user=%user.name command=%proc.cmdline container=%container.name image=%container.image.repository)
  priority: WARNING

- rule: Sensitive File Access in ML Container
  desc: Detect access to sensitive files
  condition: >
    open_read and
    container and
    container.image.repository = "secure-ml-app" and
    fd.name in (/etc/shadow, /etc/sudoers, /root/.ssh/id_rsa)
  output: >
    Sensitive file accessed in ML container
    (user=%user.name file=%fd.name container=%container.name)
  priority: CRITICAL

- rule: Network Connection from ML Container
  desc: Detect outbound network connections
  condition: >
    outbound and
    container and
    container.image.repository = "secure-ml-app" and
    not fd.sip in (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
  output: >
    Outbound connection from ML container
    (user=%user.name destination=%fd.rip:%fd.rport container=%container.name)
  priority: WARNING
EOF

# Start Falco
sudo systemctl start falco
sudo systemctl enable falco

# View alerts
sudo journalctl -fu falco
```

### 8.2 Image Signing and Verification

```bash
# Install Cosign (Sigstore)
wget https://github.com/sigstore/cosign/releases/latest/download/cosign-linux-amd64
sudo mv cosign-linux-amd64 /usr/local/bin/cosign
sudo chmod +x /usr/local/bin/cosign

# Generate key pair
cosign generate-key-pair

# Sign image
cosign sign --key cosign.key myapp:latest

# Verify image
cosign verify --key cosign.pub myapp:latest

# Admission controller to verify signatures
cat > configs/image-verification-policy.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: image-verification-policy
data:
  policy.yaml: |
    apiVersion: v1
    kind: Policy
    spec:
      validation:
        - pattern:
            spec:
              containers:
              - image: "myregistry.io/*:*"
          verify:
            cosign:
              key: |
                -----BEGIN PUBLIC KEY-----
                [Your public key here]
                -----END PUBLIC KEY-----
EOF
```

### 8.3 Supply Chain Security

```bash
# Generate SBOM (Software Bill of Materials)
docker sbom myapp:latest

# Using Syft for SBOM generation
curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin

syft myapp:latest -o spdx-json > sbom.spdx.json

# Scan SBOM for vulnerabilities
grype sbom:./sbom.spdx.json

# Provenance generation with BuildKit
docker buildx build \
  --provenance=true \
  --sbom=true \
  --tag myapp:latest \
  --output type=image,push=true .
```

### 8.4 Container Network Policies

```bash
# Create network segmentation
cat > configs/docker-compose-network-security.yml << 'EOF'
version: '3.8'

services:
  ml-api:
    image: secure-ml-app:latest
    networks:
      - frontend
      - backend

  database:
    image: postgres:15
    networks:
      - backend
    # Database only accessible from backend network

  nginx:
    image: nginx:alpine
    networks:
      - frontend
    ports:
      - "443:443"
    # Nginx only on frontend, no direct DB access

networks:
  frontend:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/24
  backend:
    driver: bridge
    internal: true  # No external access
    ipam:
      config:
        - subnet: 172.21.0.0/24
EOF
```

---

## 9. Troubleshooting Guide

### 9.1 Common Security Issues

**Issue 1: Permission Denied Errors**

```bash
# Symptom
docker run myapp:latest
# Error: Permission denied accessing /app/data

# Diagnosis
docker run --rm myapp:latest id
# Check UID/GID

ls -la ./data
# Check host directory permissions

# Solution
sudo chown -R 1000:1000 ./data
# or
docker run --user 0 myapp:latest chown -R mluser:mluser /app/data
```

**Issue 2: Secrets Not Loading**

```bash
# Symptom
Container starts but can't read secrets

# Diagnosis
docker exec container-name ls -la /run/secrets/
docker exec container-name cat /run/secrets/db_password

# Solution
# Check secret exists
docker secret ls

# Verify service has access
docker service inspect service-name | grep Secrets

# Recreate secret
docker secret rm db_password
echo "new_password" | docker secret create db_password -
```

**Issue 3: Vulnerability Scan Failures**

```bash
# Symptom
trivy image myapp:latest
# Shows CRITICAL vulnerabilities

# Diagnosis
trivy image --format json myapp:latest | \
  jq '.Results[].Vulnerabilities[] | select(.Severity=="CRITICAL")'

# Solution
# Update base image
docker pull python:3.11-slim

# Rebuild
docker build --no-cache -t myapp:latest .

# If specific package issue
# Add to Dockerfile:
RUN pip install --upgrade vulnerable-package==fixed-version
```

### 9.2 Security Testing

```bash
cat > scripts/security-test.sh << 'EOF'
#!/bin/bash

echo "🔍 Running Security Tests..."

# Test 1: User is non-root
echo "Test 1: Checking user context..."
USER_ID=$(docker run --rm myapp:latest id -u)
if [ "$USER_ID" -eq 0 ]; then
    echo "❌ FAIL: Running as root"
    exit 1
else
    echo "✅ PASS: Running as non-root (UID: $USER_ID)"
fi

# Test 2: No vulnerabilities
echo "Test 2: Scanning for vulnerabilities..."
CRITICAL=$(trivy image --severity CRITICAL --format json myapp:latest | \
    jq '[.Results[].Vulnerabilities[]?] | length')
if [ "$CRITICAL" -gt 0 ]; then
    echo "❌ FAIL: Found $CRITICAL critical vulnerabilities"
    exit 1
else
    echo "✅ PASS: No critical vulnerabilities"
fi

# Test 3: No secrets in image
echo "Test 3: Checking for secrets..."
SECRETS=$(trivy image --scanners secret --format json myapp:latest | \
    jq '[.Results[].Secrets[]?] | length')
if [ "$SECRETS" -gt 0 ]; then
    echo "❌ FAIL: Found $SECRETS secrets in image"
    exit 1
else
    echo "✅ PASS: No secrets found in image"
fi

# Test 4: Read-only filesystem
echo "Test 4: Testing filesystem..."
docker run --rm --read-only --tmpfs /tmp myapp:latest touch /test 2>/dev/null
if [ $? -eq 0 ]; then
    echo "❌ FAIL: Can write to root filesystem"
    exit 1
else
    echo "✅ PASS: Root filesystem is read-only"
fi

echo "✅ All security tests passed!"
EOF

chmod +x scripts/security-test.sh
./scripts/security-test.sh
```

---

## 10. Security Checklist

### 10.1 Pre-Deployment Security Checklist

```markdown
# Container Security Checklist

## Build Time

### Image Security
- [ ] Using official or verified base image
- [ ] Base image version pinned (not using :latest)
- [ ] Multi-stage build implemented
- [ ] Minimal base image (slim/alpine)
- [ ] Unnecessary tools removed
- [ ] Package manager cache cleared

### Dockerfile Best Practices
- [ ] Non-root user created and used
- [ ] UID/GID explicitly set (1000:1000)
- [ ] COPY with --chown flag
- [ ] No secrets in ENV or ARG
- [ ] Health check configured
- [ ] .dockerignore file present

### Vulnerability Management
- [ ] Image scanned with Trivy/Grype
- [ ] 0 CRITICAL vulnerabilities
- [ ] 0 HIGH vulnerabilities (or documented exceptions)
- [ ] Dependencies up to date
- [ ] SBOM generated

## Runtime

### Container Configuration
- [ ] Running as non-root (USER mluser)
- [ ] Read-only root filesystem
- [ ] Capabilities dropped (cap_drop: ALL)
- [ ] Only necessary capabilities added
- [ ] No privileged mode
- [ ] no-new-privileges security option

### Resource Management
- [ ] CPU limits set
- [ ] Memory limits set
- [ ] PID limits set
- [ ] Disk I/O limits (if needed)
- [ ] Restart policy configured

### Network Security
- [ ] Only required ports exposed
- [ ] Using non-privileged ports (>1024)
- [ ] Network policies defined
- [ ] Internal networks for backend
- [ ] TLS/SSL enabled

### Secrets Management
- [ ] Using Docker secrets or Vault
- [ ] No secrets in environment variables
- [ ] No secrets in code
- [ ] Secret rotation policy defined
- [ ] Secrets have minimum scope

### Access Control
- [ ] AppArmor/SELinux profile applied
- [ ] Seccomp profile configured
- [ ] File permissions verified
- [ ] Volume mounts minimal
- [ ] No host path volumes (except necessary)

## Monitoring & Maintenance

### Logging
- [ ] Centralized logging configured
- [ ] Logs don't contain secrets
- [ ] Access logs enabled
- [ ] Error logs enabled
- [ ] Log retention policy set

### Monitoring
- [ ] Runtime security monitoring (Falco)
- [ ] Vulnerability scanning automated
- [ ] Performance monitoring
- [ ] Anomaly detection
- [ ] Alerts configured

### Compliance
- [ ] Security policy documented
- [ ] Compliance requirements met
- [ ] Audit trail enabled
- [ ] Incident response plan
- [ ] Regular security reviews scheduled

## ML-Specific

### Model Security
- [ ] Model files access-controlled
- [ ] Model versioning implemented
- [ ] Model tampering detection
- [ ] Intellectual property protected
- [ ] Model extraction mitigated

### Data Privacy
- [ ] PII data encrypted
- [ ] Data anonymization for training
- [ ] GDPR compliance (if applicable)
- [ ] Data retention policies
- [ ] Secure data deletion

### API Security
- [ ] Authentication required
- [ ] Rate limiting implemented
- [ ] Input validation
- [ ] Output sanitization
- [ ] API versioning

## Sign-off

- [ ] Security team approval
- [ ] Penetration testing completed
- [ ] Documentation updated
- [ ] Runbook created
- [ ] Team trained

**Deployment Date**: ___________
**Approved By**: ___________
**Next Review**: ___________
```

### 10.2 Quick Security Audit Script

```bash
cat > scripts/quick-audit.sh << 'EOF'
#!/bin/bash

# Quick security audit for Docker containers
IMAGE="${1:-myapp:latest}"

echo "🔒 Security Audit for: $IMAGE"
echo "================================"

# Check 1: Scan vulnerabilities
echo -e "\n1. Vulnerability Scan"
trivy image --severity HIGH,CRITICAL --format table "$IMAGE" | head -20

# Check 2: Check user
echo -e "\n2. User Context"
docker run --rm "$IMAGE" id

# Check 3: Check capabilities
echo -e "\n3. Image Configuration"
docker inspect "$IMAGE" | jq '.[0].Config.User'

# Check 4: Check for secrets
echo -e "\n4. Secret Scan"
trivy image --scanners secret "$IMAGE"

# Check 5: List files with SUID bit
echo -e "\n5. SUID Files"
docker run --rm "$IMAGE" find / -perm -4000 2>/dev/null || echo "Cannot scan"

# Generate score
SCORE=100
if docker run --rm "$IMAGE" id -u | grep -q "^0$"; then
    SCORE=$((SCORE - 30))
    echo -e "\n⚠️  Running as root (-30)"
fi

CRITICAL=$(trivy image --severity CRITICAL --format json "$IMAGE" 2>/dev/null | jq '[.Results[].Vulnerabilities[]?] | length')
SCORE=$((SCORE - CRITICAL * 10))
echo -e "\n📊 Security Score: $SCORE/100"

if [ $SCORE -ge 80 ]; then
    echo "✅ GOOD: Image meets security standards"
elif [ $SCORE -ge 60 ]; then
    echo "⚠️  WARNING: Image needs security improvements"
else
    echo "❌ CRITICAL: Image has serious security issues"
fi
EOF

chmod +x scripts/quick-audit.sh
./scripts/quick-audit.sh myapp:latest
```

---

## Summary

### What You've Learned

✅ **Container Security Fundamentals**: Understood multi-layer security model, least privilege principle, and container hardening

✅ **Vulnerability Scanning**: Mastered Trivy, Docker Scout, and Clair for comprehensive image scanning

✅ **Non-Root Containers**: Implemented proper user management, file permissions, and user namespaces

✅ **Secrets Management**: Configured Docker secrets, environment-specific secrets, Vault integration, and rotation

✅ **Automated Security**: Integrated security scanning in CI/CD, automated reporting, and pre-commit hooks

✅ **Advanced Security**: Applied AppArmor/SELinux profiles, seccomp filters, and runtime monitoring

✅ **Production ML Security**: Secured ML APIs, protected models, encrypted data, implemented access control

✅ **Security Testing**: Created comprehensive checklists, automated audits, and troubleshooting guides

### Key Takeaways

1. **Defense in Depth**: Security must be implemented at every layer
2. **Shift Left**: Integrate security scanning early in development
3. **Least Privilege**: Grant only necessary permissions
4. **Secrets Never in Code**: Always use proper secrets management
5. **Continuous Monitoring**: Security is ongoing, not one-time
6. **ML-Specific Concerns**: Model and data protection are critical

### Production Deployment

When deploying to production, ensure you:
- Run containers as non-root
- Apply resource limits
- Use read-only filesystems
- Implement comprehensive monitoring
- Automate vulnerability scanning
- Encrypt sensitive data
- Maintain audit trails
- Have incident response plans

### Next Steps

1. **Practice**: Implement security for your ML projects
2. **Automation**: Integrate into CI/CD pipelines
3. **Learn More**: Study Kubernetes security (next module)
4. **Stay Updated**: Follow security advisories and CVEs
5. **Share Knowledge**: Document and teach security practices

---

## Additional Resources

### Tools & Documentation
- [Trivy Documentation](https://aquasecurity.github.io/trivy/)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [OWASP Container Security](https://owasp.org/www-project-docker-top-10/)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [Falco Rules](https://falco.org/docs/rules/)

### Further Reading
- Container Security by Liz Rice
- Docker Deep Dive by Nigel Poulton
- Kubernetes Security by Liz Rice and Michael Hausenblas

### Practice Environments
- [TryHackMe Docker Security](https://tryhackme.com/)
- [OWASP WebGoat](https://owasp.org/www-project-webgoat/)
- [Vulnerable Docker Images](https://github.com/warvariuc/docker-vulnerabilities)

---

**Guide Version**: 1.0
**Last Updated**: November 2025
**Estimated Completion Time**: 5-7 hours
**Difficulty**: Advanced
**ML Focus**: Model Security, Data Privacy, Secure Serving
