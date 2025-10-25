# Troubleshooting Guide: [Module/Exercise Name]

**Last Updated**: [Date]
**Applies To**: [Module Number] - [Module Name]
**Difficulty**: [Beginner/Intermediate/Advanced]

---

## 📋 Overview

This guide helps you diagnose and resolve common issues encountered in [module/exercise name]. Issues are organized by category and include symptoms, causes, and solutions.

---

## 🔍 Quick Diagnosis

Before diving into specific issues, try these general debugging steps:

1. **Check Logs**
   ```bash
   # Application logs
   tail -f logs/application.log

   # Docker logs
   docker logs <container-name>

   # Kubernetes logs
   kubectl logs <pod-name>
   ```

2. **Verify Environment**
   ```bash
   # Check Python version
   python --version

   # Check installed packages
   pip list | grep [package-name]

   # Check environment variables
   printenv | grep [VAR_NAME]
   ```

3. **Test Connectivity**
   ```bash
   # Test network connectivity
   ping [host]
   curl [url]

   # Test database connection
   psql -h [host] -U [user] -d [database]
   ```

---

## 🐛 Common Issues

### Installation & Setup Issues

#### Issue 1: Dependencies Fail to Install

**Symptoms**:
```
ERROR: Could not find a version that satisfies the requirement [package]
```

**Possible Causes**:
- Python version mismatch
- pip outdated
- Package not available for your platform

**Solutions**:

1. **Check Python version**
   ```bash
   python --version  # Should be 3.11+ for this curriculum
   ```

2. **Upgrade pip**
   ```bash
   python -m pip install --upgrade pip
   ```

3. **Use specific package versions**
   ```bash
   pip install [package]==[version]
   ```

4. **Try with --user flag**
   ```bash
   pip install --user -r requirements.txt
   ```

**Prevention**:
- Always use virtual environments
- Keep requirements.txt with exact versions

---

#### Issue 2: Permission Denied Errors

**Symptoms**:
```
PermissionError: [Errno 13] Permission denied: '/usr/local/...'
```

**Possible Causes**:
- Installing to system Python
- Insufficient file permissions
- Running as wrong user

**Solutions**:

1. **Use virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Fix file permissions**
   ```bash
   chmod +x [script-name]
   chmod 644 [config-file]
   ```

3. **Run with appropriate permissions**
   ```bash
   # Only if absolutely necessary
   sudo [command]
   ```

**Prevention**:
- Never use system Python for projects
- Always create virtual environments
- Check file permissions before running

---

### Runtime Errors

#### Issue 3: ImportError / ModuleNotFoundError

**Symptoms**:
```python
ModuleNotFoundError: No module named '[module]'
```

**Possible Causes**:
- Package not installed
- Wrong Python environment active
- Incorrect PYTHONPATH

**Diagnostic Steps**:
```bash
# Check if package is installed
pip show [package-name]

# Check Python path
python -c "import sys; print(sys.path)"

# Verify virtual environment
which python
```

**Solutions**:

1. **Install missing package**
   ```bash
   pip install [package-name]
   ```

2. **Activate correct environment**
   ```bash
   source venv/bin/activate
   ```

3. **Add to PYTHONPATH**
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

---

#### Issue 4: Configuration Errors

**Symptoms**:
```
KeyError: 'CONFIG_VAR'
FileNotFoundError: config.yml not found
```

**Possible Causes**:
- Missing .env file
- Incorrect configuration file path
- Environment variables not set

**Solutions**:

1. **Create .env file**
   ```bash
   cp .env.example .env
   # Edit .env with your values
   ```

2. **Set environment variables**
   ```bash
   export API_KEY="your-key-here"
   export DATABASE_URL="postgresql://..."
   ```

3. **Verify configuration loading**
   ```python
   import os
   print(os.getenv('CONFIG_VAR', 'not found'))
   ```

**Prevention**:
- Always provide .env.example
- Document all required environment variables
- Use sensible defaults where possible

---

### Docker Issues

#### Issue 5: Docker Build Fails

**Symptoms**:
```
ERROR [stage 1/5] failed to solve: ...
```

**Possible Causes**:
- Syntax errors in Dockerfile
- Base image not available
- Network issues
- Insufficient disk space

**Diagnostic Steps**:
```bash
# Check Docker is running
docker ps

# Check disk space
df -h

# Try pulling base image manually
docker pull python:3.11-slim
```

**Solutions**:

1. **Clean Docker cache**
   ```bash
   docker system prune -a
   ```

2. **Build with no cache**
   ```bash
   docker build --no-cache -t [image-name] .
   ```

3. **Check Dockerfile syntax**
   ```dockerfile
   # Ensure proper ordering
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["python", "app.py"]
   ```

---

#### Issue 6: Container Exits Immediately

**Symptoms**:
```bash
docker ps -a
# Shows container with status "Exited (1) 2 seconds ago"
```

**Diagnostic Steps**:
```bash
# Check container logs
docker logs [container-name]

# Run container interactively
docker run -it [image-name] /bin/bash

# Check entrypoint
docker inspect [image-name] | grep -A 5 Entrypoint
```

**Common Causes & Solutions**:

1. **Application crashes on startup**
   - Check logs for error messages
   - Verify all dependencies installed
   - Test application locally first

2. **Missing CMD or ENTRYPOINT**
   ```dockerfile
   # Add to Dockerfile
   CMD ["python", "app.py"]
   ```

3. **Environment variables missing**
   ```bash
   docker run -e API_KEY=value [image-name]
   # Or use --env-file
   docker run --env-file .env [image-name]
   ```

---

### Kubernetes Issues

#### Issue 7: Pods Stuck in Pending State

**Symptoms**:
```bash
kubectl get pods
# Shows pod in "Pending" state
```

**Diagnostic Steps**:
```bash
# Describe pod for details
kubectl describe pod [pod-name]

# Check events
kubectl get events --sort-by='.lastTimestamp'

# Check node resources
kubectl top nodes
```

**Common Causes**:

1. **Insufficient Resources**
   ```yaml
   # Reduce resource requests
   resources:
     requests:
       memory: "256Mi"  # Was 2Gi
       cpu: "250m"      # Was 2
   ```

2. **ImagePullBackOff**
   ```bash
   # Check image exists
   docker pull [image-name]

   # Use imagePullSecrets if private
   ```

3. **PersistentVolumeClaim not bound**
   ```bash
   kubectl get pvc
   # Check PVC status
   ```

---

#### Issue 8: Service Not Accessible

**Symptoms**:
- Cannot access service via URL
- Connection refused errors
- Timeouts

**Diagnostic Steps**:
```bash
# Check service
kubectl get svc [service-name]

# Check endpoints
kubectl get endpoints [service-name]

# Port forward for testing
kubectl port-forward svc/[service-name] 8080:80

# Test from within cluster
kubectl run test --image=curlimages/curl -it --rm -- curl [service-name]
```

**Solutions**:

1. **Verify selector labels match**
   ```yaml
   # In Service
   selector:
     app: myapp  # Must match pod labels
   ```

2. **Check port configuration**
   ```yaml
   ports:
     - port: 80          # Service port
       targetPort: 8080  # Container port
   ```

3. **Verify Ingress configuration**
   ```bash
   kubectl get ingress
   kubectl describe ingress [ingress-name]
   ```

---

### ML-Specific Issues

#### Issue 9: Model Loading Fails

**Symptoms**:
```python
FileNotFoundError: model.pkl not found
RuntimeError: CUDA out of memory
```

**Solutions**:

1. **Check model file path**
   ```python
   import os
   model_path = os.getenv('MODEL_PATH', '/models/model.pkl')
   assert os.path.exists(model_path), f"Model not found at {model_path}"
   ```

2. **Handle CUDA memory**
   ```python
   import torch
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # Clear CUDA cache
   torch.cuda.empty_cache()

   # Use smaller batch size
   batch_size = 16  # Reduce if OOM
   ```

3. **Mount model volume correctly**
   ```yaml
   # Kubernetes
   volumeMounts:
     - name: models
       mountPath: /models
   ```

---

#### Issue 10: Slow Inference Performance

**Symptoms**:
- API timeouts
- High latency
- Poor throughput

**Diagnostic Steps**:
```python
import time

start = time.time()
result = model.predict(data)
duration = time.time() - start
print(f"Inference took {duration:.3f}s")
```

**Solutions**:

1. **Use batch inference**
   ```python
   # Instead of one at a time
   results = model.predict_batch(data_list)
   ```

2. **Optimize model**
   ```python
   # Quantization
   import torch
   quantized_model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```

3. **Use caching**
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=1000)
   def cached_predict(input_hash):
       return model.predict(input_data)
   ```

---

## 🔧 Debugging Tools

### Essential Commands

```bash
# Process inspection
ps aux | grep python
top -p [PID]
htop

# Network debugging
netstat -tulpn
lsof -i :[port]
tcpdump -i any port [port]

# Disk usage
du -sh *
df -h

# Memory usage
free -h
cat /proc/meminfo
```

### Python Debugging

```python
# Add breakpoints
import pdb; pdb.set_trace()

# Or use ipdb for better experience
import ipdb; ipdb.set_trace()

# Print stack trace
import traceback
traceback.print_exc()

# Profile performance
import cProfile
cProfile.run('function()')
```

### Docker Debugging

```bash
# Execute commands in running container
docker exec -it [container] /bin/bash

# Copy files from container
docker cp [container]:/path/to/file ./local/path

# Inspect container
docker inspect [container]

# View container processes
docker top [container]
```

### Kubernetes Debugging

```bash
# Get into pod shell
kubectl exec -it [pod] -- /bin/bash

# Copy files from pod
kubectl cp [pod]:/path/to/file ./local/path

# Check resource usage
kubectl top pods
kubectl top nodes

# View cluster info
kubectl cluster-info dump
```

---

## 📚 Additional Resources

### Documentation
- [Official Python docs](https://docs.python.org/)
- [Docker documentation](https://docs.docker.com/)
- [Kubernetes docs](https://kubernetes.io/docs/)

### Community
- Stack Overflow
- Reddit: r/learnpython, r/docker, r/kubernetes
- Discord/Slack communities

### Tools
- [Docker Debug Guide](https://docs.docker.com/config/containers/logging/)
- [K8s Troubleshooting](https://kubernetes.io/docs/tasks/debug/)

---

## 💬 Getting Help

If you're still stuck after trying these solutions:

1. **Search existing issues** on GitHub
2. **Create a new issue** with:
   - Error message (full stack trace)
   - Steps to reproduce
   - Your environment (OS, Python version, etc.)
   - What you've already tried
3. **Ask in discussions** for general questions

---

## ✅ Prevention Checklist

Before running into issues, always:

- [ ] Use virtual environments
- [ ] Read error messages carefully
- [ ] Check logs first
- [ ] Verify configuration
- [ ] Test locally before deploying
- [ ] Keep dependencies updated
- [ ] Follow best practices
- [ ] Document your setup

---

**Last Updated**: [Date]
**Maintainers**: AI Infrastructure Curriculum Team
**Report Issues**: [GitHub Issues Link]
