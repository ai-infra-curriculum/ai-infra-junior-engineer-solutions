# Exercise 06: Container Security

Master Docker container security including scanning, secrets management, and security best practices for ML applications.

## Complete Solution

**Production-ready security tools** in `solutions/`:
- **security_scanner.py** - Comprehensive security scanner and auditor
- **secrets_manager.py** - Secrets management tool
- **20+ comprehensive tests**

## Quick Start

```bash
cd solutions/

# Scan Dockerfile
./security_scanner.py --dockerfile ../Dockerfile

# Scan Docker image
./security_scanner.py --image ml-tensorflow:latest

# Check running container
./security_scanner.py --container my-container

# Generate security report
./security_scanner.py --dockerfile Dockerfile --image ml-api:latest --report security-report.json

# Generate random secret
./secrets_manager.py generate --length 32

# Create secret file
./secrets_manager.py create-file --name db-password --value "secure-password"

# Validate env file
./secrets_manager.py validate --env-file .env

# Run tests
python test_security.py
```

## Learning Objectives

- Scan containers for vulnerabilities
- Implement Dockerfile security best practices
- Manage secrets securely
- Configure runtime security
- Audit container configurations

## Security Best Practices

### 1. Use Non-Root Users
```dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

### 2. Pin Base Image Versions
```dockerfile
# Good
FROM python:3.11-slim

# Bad
FROM python:latest
```

### 3. Add Health Checks
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

### 4. Minimize Attack Surface
```dockerfile
# Use slim/alpine images
FROM python:3.11-slim

# Clean up in same layer
RUN apt-get update && \
    apt-get install -y package && \
    rm -rf /var/lib/apt/lists/*
```

### 5. Never Store Secrets in Images
```bash
# Bad
ENV DB_PASSWORD=mysecret

# Good - use environment variables or secrets
docker run -e DB_PASSWORD=$DB_PASSWORD my-app
```

### 6. Use Read-Only Filesystems
```bash
docker run --read-only --tmpfs /tmp my-app
```

### 7. Limit Capabilities
```bash
docker run --cap-drop=ALL --cap-add=NET_BIND_SERVICE my-app
```

### 8. Scan Images Regularly
```bash
# With Trivy
trivy image --severity HIGH,CRITICAL my-app:latest

# With Grype
grype my-app:latest
```

## Security Checklist

- [ ] Run as non-root user
- [ ] Pin base image versions
- [ ] Include HEALTHCHECK
- [ ] No secrets in ENV variables
- [ ] Clean up package manager cache
- [ ] Use .dockerignore
- [ ] Scan for vulnerabilities
- [ ] Limit container capabilities
- [ ] Use read-only filesystem when possible
- [ ] Keep images up to date
- [ ] Implement proper logging
- [ ] Use official base images
- [ ] Minimize installed packages
- [ ] Set resource limits

## Common Vulnerabilities

### Critical
- Running as root
- Privileged mode
- Mounting /var/run/docker.sock
- Using latest tag
- Hardcoded secrets

### High
- Missing security updates
- Unnecessary capabilities
- World-writable volumes
- Exposed sensitive ports

### Medium
- Missing health checks
- No resource limits
- Verbose error messages
- Outdated base images

## Tools Integration

### Trivy
```bash
# Scan image
trivy image my-app:latest

# Generate report
trivy image --format json --output report.json my-app:latest
```

### Grype
```bash
# Scan image
grype my-app:latest

# Only high/critical
grype --fail-on high my-app:latest
```

### Docker Bench Security
```bash
docker run --rm -it \
  -v /var/run/docker.sock:/var/run/docker.sock \
  docker/docker-bench-security
```

## Resources

- [Docker Security Documentation](https://docs.docker.com/engine/security/)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [OWASP Docker Security](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)
- [Trivy](https://github.com/aquasecurity/trivy)
- [Grype](https://github.com/anchore/grype)
