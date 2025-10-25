# Debugging Guide for Junior AI Infrastructure Engineers

A comprehensive, beginner-friendly guide to debugging Docker containers, Kubernetes pods, Python applications, and network issues in AI/ML infrastructure.

---

## Table of Contents

1. [Introduction to Debugging](#introduction-to-debugging)
2. [Docker Debugging](#docker-debugging)
3. [Kubernetes Debugging](#kubernetes-debugging)
4. [Python Debugging](#python-debugging)
5. [Network Debugging](#network-debugging)
6. [Common Error Patterns](#common-error-patterns)
7. [Debugging Tools Reference](#debugging-tools-reference)
8. [Step-by-Step Troubleshooting Workflows](#step-by-step-troubleshooting-workflows)

---

## Introduction to Debugging

### The Debugging Mindset

Debugging is a systematic process, not guesswork. Follow these principles:

1. **Understand the Expected Behavior**: What should happen?
2. **Identify the Actual Behavior**: What is actually happening?
3. **Form a Hypothesis**: Why might this be occurring?
4. **Test Your Hypothesis**: Gather evidence to confirm or reject
5. **Fix and Verify**: Implement fix and ensure it works
6. **Document**: Record the issue and solution for future reference

### The Scientific Method for Debugging

```
1. Observe → What's the symptom?
2. Question → What could cause this?
3. Hypothesize → I think it's X because...
4. Experiment → Let me test by doing Y
5. Analyze → The result tells me...
6. Conclude → The root cause is Z
```

### Reading Error Messages

**Most errors tell you exactly what's wrong!** Learn to read them:

```
Error: ModuleNotFoundError: No module named 'tensorflow'
       ^                                    ^
    What went wrong               Specific detail
```

**Key parts of an error message:**
- **Error Type**: What kind of error (ImportError, ConnectionError, etc.)
- **Error Message**: Human-readable description
- **Stack Trace**: The path the code took to reach the error
- **Line Numbers**: Where in the code the error occurred

---

## Docker Debugging

### 1. Container Won't Start

#### Symptoms
```bash
$ docker ps -a
CONTAINER ID   IMAGE          STATUS
abc123         my-app:latest  Exited (1) 2 seconds ago
```

#### Debug Steps

**Step 1: Check Container Logs**
```bash
# View logs for stopped container
docker logs abc123

# View logs with timestamps
docker logs --timestamps abc123

# Follow logs in real-time
docker logs --follow abc123

# Last 100 lines
docker logs --tail 100 abc123
```

**Common Issues in Logs:**

```bash
# Issue 1: Module not installed
ModuleNotFoundError: No module named 'fastapi'
# Fix: Add to requirements.txt and rebuild image

# Issue 2: Port already in use
Error: bind: address already in use
# Fix: Change port or stop conflicting service

# Issue 3: File not found
FileNotFoundError: [Errno 2] No such file or directory: 'model.pkl'
# Fix: Ensure file is copied in Dockerfile or mounted as volume

# Issue 4: Permission denied
PermissionError: [Errno 13] Permission denied: '/app/data'
# Fix: Adjust file permissions or run as correct user
```

**Step 2: Inspect Container Configuration**
```bash
# View detailed container info
docker inspect abc123

# Check specific fields
docker inspect --format='{{.State.ExitCode}}' abc123
docker inspect --format='{{.Config.Env}}' abc123
docker inspect --format='{{.Mounts}}' abc123
```

**Useful Inspection Queries:**
```bash
# Exit code (0 = success, non-zero = error)
docker inspect --format='{{.State.ExitCode}}' abc123

# Environment variables
docker inspect --format='{{json .Config.Env}}' abc123 | jq

# Volume mounts
docker inspect --format='{{json .Mounts}}' abc123 | jq

# Network settings
docker inspect --format='{{json .NetworkSettings}}' abc123 | jq

# Command that was run
docker inspect --format='{{.Config.Cmd}}' abc123
```

**Step 3: Try Running Interactively**
```bash
# Override entrypoint to get a shell
docker run -it --entrypoint /bin/bash my-app:latest

# Now you're inside the container - test manually
$ python app.py
$ ls -la /app/
$ env | grep -i api
```

**Step 4: Check Image Build Process**
```bash
# Rebuild with no cache to see each step
docker build --no-cache -t my-app:debug .

# Build with progress output
docker build --progress=plain -t my-app:debug .
```

### 2. Container Running but Not Responding

#### Symptoms
- Container shows as "Up" but application not accessible
- Connection refused or timeout errors

#### Debug Steps

**Step 1: Verify Container is Actually Running**
```bash
# Check status
docker ps | grep my-app

# Check resource usage
docker stats my-app

# If CPU/Memory at 0%, application might have crashed
```

**Step 2: Exec Into Running Container**
```bash
# Get a shell in running container
docker exec -it my-app /bin/bash

# Once inside, check:
# 1. Is the process running?
ps aux | grep python

# 2. Is it listening on the expected port?
netstat -tlnp | grep 8000
# Or if netstat not available:
ss -tlnp | grep 8000

# 3. Can I curl it locally?
curl http://localhost:8000/health
```

**Step 3: Check Port Mappings**
```bash
# View port mappings
docker port my-app

# Expected output:
# 8000/tcp -> 0.0.0.0:8000

# Test from host
curl http://localhost:8000/health

# Test with specific IP
curl http://127.0.0.1:8000/health
```

**Common Port Issues:**

```bash
# Issue: Container port not exposed
# Fix: Add EXPOSE 8000 to Dockerfile
# Or: docker run -p 8000:8000 my-app

# Issue: Wrong port mapping
# Wrong: docker run -p 8080:8000 my-app (if app listens on 8080 inside)
# Right: docker run -p 8000:8080 my-app

# Issue: Application binding to localhost instead of 0.0.0.0
# Wrong: app.run(host='127.0.0.1', port=8000)  # Only accessible inside container
# Right: app.run(host='0.0.0.0', port=8000)    # Accessible from host
```

**Step 4: Check Application Logs**
```bash
# Application-level logs (inside container)
docker exec my-app cat /app/logs/app.log

# Or tail them in real-time
docker exec my-app tail -f /app/logs/app.log
```

### 3. Container Crashes or Restarts Repeatedly

#### Symptoms
```bash
$ docker ps
CONTAINER ID   STATUS
abc123         Restarting (1) 3 seconds ago
```

#### Debug Steps

**Step 1: Check Restart Policy**
```bash
# View restart policy
docker inspect --format='{{.HostConfig.RestartPolicy}}' abc123

# Remove restart policy to debug
docker update --restart=no abc123

# Now container will stay stopped when it crashes
```

**Step 2: Identify Crash Pattern**
```bash
# View all logs including previous crashes
docker logs abc123 2>&1 | grep -i error

# Check system logs
journalctl -u docker.service | grep abc123
```

**Step 3: Common Crash Causes**

```python
# Cause 1: Uncaught Exception
# app.py
def predict(data):
    model = load_model('model.pkl')  # File doesn't exist!
    return model.predict(data)

# Debug: Add error handling
try:
    model = load_model('model.pkl')
except FileNotFoundError:
    logger.error("Model file not found!")
    sys.exit(1)  # Exit cleanly with error code
```

```python
# Cause 2: Out of Memory
# Check with docker stats
$ docker stats my-app
CONTAINER   MEM USAGE / LIMIT
my-app      1.8GiB / 2GiB      # Hitting limit!

# Fix: Increase memory limit
docker run -m 4g my-app
# Or in docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 4G
```

```python
# Cause 3: Application Exits Immediately
# Wrong Dockerfile CMD:
CMD ["python", "script.py"]  # Script runs and exits

# Fix: Keep application running
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0"]
```

### 4. Docker Build Failures

#### Symptoms
```
Step 5/10 : RUN pip install -r requirements.txt
ERROR: Failed building wheel for package-name
```

#### Debug Steps

**Step 1: Build with Detailed Output**
```bash
# Build with full output
docker build --progress=plain --no-cache -t my-app:debug . 2>&1 | tee build.log

# Review build.log for errors
```

**Step 2: Test Problematic Steps Locally**
```bash
# If "RUN pip install" fails, test locally:
docker run -it python:3.9 /bin/bash
# Inside container:
$ pip install -r requirements.txt  # Same error?
$ pip install package-name==1.2.3  # Try specific version
```

**Step 3: Common Build Errors**

```dockerfile
# Error 1: COPY fails - file not found
COPY model.pkl /app/
# Fix: Ensure file exists relative to build context
# Check: ls -la ./model.pkl

# Error 2: Permission denied during build
RUN mkdir /app/data
# Fix: Ensure build runs with correct permissions
RUN mkdir -p /app/data && chmod 755 /app/data

# Error 3: Network issues during pip install
RUN pip install tensorflow
# Fix: Use --no-cache-dir and specify index
RUN pip install --no-cache-dir --index-url https://pypi.org/simple tensorflow

# Error 4: Platform mismatch (building on Mac M1 for Linux)
# Fix: Specify platform
docker build --platform linux/amd64 -t my-app .
```

**Step 4: Debug Multi-Stage Builds**
```dockerfile
# Original (failing):
FROM python:3.9 AS builder
RUN pip install -r requirements.txt

FROM python:3.9-slim
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Debug by building just first stage:
docker build --target builder -t my-app:builder .
docker run -it my-app:builder /bin/bash
# Test inside container
```

### 5. Docker Compose Issues

#### Common Issues

**Issue 1: Services Can't Communicate**
```yaml
# docker-compose.yml
services:
  api:
    build: ./api
    environment:
      - DATABASE_URL=postgresql://db:5432/mydb  # Using service name ✓

  db:
    image: postgres:13
```

```bash
# Debug connectivity
docker-compose exec api /bin/bash

# Inside api container:
$ ping db  # Should resolve
$ nslookup db  # Should show IP
$ curl http://db:5432  # Test connection (will fail but proves routing works)
```

**Issue 2: Services Start in Wrong Order**
```yaml
# Problem: API starts before DB is ready
services:
  api:
    depends_on:
      - db  # Only waits for container to start, not DB to be ready!

  db:
    image: postgres:13

# Solution 1: Add healthcheck
services:
  api:
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:13
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

# Solution 2: Add retry logic in application
# app.py
import time
from sqlalchemy import create_engine

def connect_with_retry(url, max_retries=5):
    for i in range(max_retries):
        try:
            engine = create_engine(url)
            engine.connect()
            return engine
        except Exception as e:
            if i == max_retries - 1:
                raise
            time.sleep(5)
```

**Issue 3: Volume Permission Issues**
```yaml
services:
  api:
    volumes:
      - ./data:/app/data
    user: "1000:1000"  # Run as specific user

# Debug: Check permissions
$ ls -la ./data
drwxr-xr-x  2 root root  # Owned by root!

# Fix: Change ownership
$ sudo chown -R 1000:1000 ./data
```

### 6. Docker Networking Issues

#### Debug Network Connectivity

```bash
# List networks
docker network ls

# Inspect network
docker network inspect bridge

# See which containers are on a network
docker network inspect my-network | jq '.[0].Containers'

# Create custom network
docker network create my-network

# Connect running container to network
docker network connect my-network my-container

# Run container on specific network
docker run --network my-network my-app
```

#### Test Network Connectivity
```bash
# From container A, test connection to container B
docker exec container-a ping container-b

# Check DNS resolution
docker exec container-a nslookup container-b

# Test HTTP endpoint
docker exec container-a curl http://container-b:8000/health

# Check open ports
docker exec container-a nc -zv container-b 8000
```

### 7. Docker Resource Issues

#### Check Resource Usage
```bash
# Real-time stats
docker stats

# Specific container
docker stats my-app

# Format output
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

#### Common Resource Problems

```bash
# Problem 1: Out of Disk Space
$ docker build -t my-app .
ERROR: failed to solve: write /var/lib/docker/...: no space left on device

# Check disk usage
docker system df

# Clean up
docker system prune -a  # Remove unused images
docker volume prune     # Remove unused volumes

# Problem 2: Memory Limit Hit
$ docker logs my-app
Killed  # OOM killer terminated process

# Check events
docker events --filter container=my-app

# Set memory limit
docker run -m 4g --memory-swap 4g my-app

# Problem 3: Too many open files
ulimit: too many open files

# Fix: Increase limits
docker run --ulimit nofile=65536:65536 my-app
```

---

## Kubernetes Debugging

### 1. Pod Won't Start

#### Symptoms
```bash
$ kubectl get pods
NAME         READY   STATUS             RESTARTS
my-app-pod   0/1     CrashLoopBackOff   5
```

#### Debug Steps

**Step 1: Describe the Pod**
```bash
# Most important command for debugging!
kubectl describe pod my-app-pod

# Look for key sections:
# 1. Events (bottom of output) - shows what happened
# 2. Containers state - why it failed
# 3. Conditions - pod readiness
```

**Example Output:**
```
Events:
  Type     Reason     Age                From               Message
  ----     ------     ----               ----               -------
  Normal   Scheduled  2m                 default-scheduler  Successfully assigned default/my-app-pod to node-1
  Normal   Pulled     1m (x4 over 2m)    kubelet           Container image "my-app:latest" already present
  Warning  BackOff    30s (x6 over 2m)   kubelet           Back-off restarting failed container
```

**Step 2: Check Pod Logs**
```bash
# Current logs
kubectl logs my-app-pod

# Previous container logs (if restarted)
kubectl logs my-app-pod --previous

# Logs from specific container (if multiple)
kubectl logs my-app-pod -c container-name

# Follow logs in real-time
kubectl logs -f my-app-pod

# Last 50 lines
kubectl logs my-app-pod --tail=50

# Logs with timestamps
kubectl logs my-app-pod --timestamps
```

**Step 3: Common Pod Failure Reasons**

```yaml
# Reason 1: ImagePullBackOff
Status: ImagePullBackOff

# Debug:
kubectl describe pod my-app-pod | grep -A 5 Events
# Output: Failed to pull image "my-app:latest": rpc error: code = Unknown

# Causes:
# - Image doesn't exist
# - Wrong image name/tag
# - Private registry without credentials

# Fix:
# 1. Verify image exists:
docker pull my-app:latest

# 2. Use correct image name:
spec:
  containers:
  - name: my-app
    image: myregistry.io/my-app:v1.0.0  # Full path

# 3. Add image pull secret:
kubectl create secret docker-registry regcred \
  --docker-server=myregistry.io \
  --docker-username=user \
  --docker-password=pass

spec:
  containers:
  - name: my-app
    image: myregistry.io/my-app:v1.0.0
  imagePullSecrets:
  - name: regcred
```

```yaml
# Reason 2: CrashLoopBackOff
Status: CrashLoopBackOff

# Debug:
kubectl logs my-app-pod --previous  # See why it crashed

# Common causes:
# - Application error on startup
# - Missing configuration
# - Failed health checks

# Fix: Check logs for specific error and address root cause
```

```yaml
# Reason 3: Pending
Status: Pending

# Debug:
kubectl describe pod my-app-pod

# Common causes:
Events:
  Warning  FailedScheduling  Insufficient cpu  # Not enough resources

# Fix:
# 1. Reduce resource requests:
spec:
  containers:
  - name: my-app
    resources:
      requests:
        cpu: 100m      # Was 4 cores
        memory: 256Mi  # Was 16Gi

# 2. Add more nodes to cluster:
kubectl get nodes
kubectl describe node node-1  # Check available resources
```

### 2. Pod Running but Not Accessible

#### Debug Service Connectivity

**Step 1: Verify Pod is Running**
```bash
# Check pod status
kubectl get pod my-app-pod -o wide

# Expected:
NAME         READY   STATUS    RESTARTS   IP
my-app-pod   1/1     Running   0          10.244.0.5

# If READY shows 0/1, check logs
kubectl logs my-app-pod
```

**Step 2: Test Pod Directly**
```bash
# Port-forward to pod (bypass service)
kubectl port-forward pod/my-app-pod 8080:8000

# In another terminal:
curl http://localhost:8080/health

# If this works, pod is fine - issue is with Service
# If this fails, issue is with pod
```

**Step 3: Check Service Configuration**
```bash
# Describe service
kubectl describe service my-app-service

# Key things to check:
# 1. Endpoints - should show pod IPs
Endpoints: 10.244.0.5:8000

# 2. Selector - must match pod labels
Selector: app=my-app

# 3. Port mappings
Port: 80/TCP
TargetPort: 8000/TCP

# Verify pod labels match service selector
kubectl get pod my-app-pod --show-labels
```

**Step 4: Test Service Connectivity**
```bash
# Get service IP
kubectl get service my-app-service

# Test from another pod
kubectl run debug --image=curlimages/curl -it --rm -- sh
# Inside pod:
$ curl http://my-app-service/health
$ curl http://my-app-service.default.svc.cluster.local/health  # FQDN

# Check DNS resolution
$ nslookup my-app-service
```

**Common Service Issues:**

```yaml
# Issue 1: Selector doesn't match pod labels
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: my-app-service
spec:
  selector:
    app: my-app      # Must match pod labels!
  ports:
  - port: 80
    targetPort: 8000

# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  template:
    metadata:
      labels:
        app: my-app  # Must match service selector!

# Debug:
kubectl get endpoints my-app-service
# If no endpoints, labels don't match
```

```yaml
# Issue 2: Wrong targetPort
# App listens on 8080 inside container
spec:
  containers:
  - name: my-app
    ports:
    - containerPort: 8080

# But service targets 8000 (wrong!)
spec:
  ports:
  - port: 80
    targetPort: 8000  # Should be 8080!
```

### 3. Debugging with kubectl exec

#### Get Shell Access to Pod
```bash
# Exec into running pod
kubectl exec -it my-app-pod -- /bin/bash

# If bash not available, try sh
kubectl exec -it my-app-pod -- /bin/sh

# Run single command
kubectl exec my-app-pod -- ps aux
kubectl exec my-app-pod -- env
kubectl exec my-app-pod -- ls -la /app
```

#### Common Debugging Commands Inside Pod
```bash
# Once inside pod:

# 1. Check if application is running
ps aux | grep python

# 2. Check listening ports
netstat -tlnp
# or
ss -tlnp

# 3. Test local connectivity
curl http://localhost:8000/health

# 4. Check environment variables
env | grep -i api
env | sort

# 5. Verify files are present
ls -la /app/
cat /app/config.yaml

# 6. Check disk space
df -h

# 7. Check DNS resolution
nslookup database-service
cat /etc/resolv.conf

# 8. Test network connectivity to other services
ping database-service
curl http://database-service:5432
```

### 4. Resource Issues in Kubernetes

#### Check Resource Usage
```bash
# Pod resource usage
kubectl top pod my-app-pod

# All pods
kubectl top pods

# Node resource usage
kubectl top nodes

# If 'kubectl top' doesn't work, metrics-server not installed:
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

#### Debug OOMKilled Pods
```bash
# Check if pod was OOMKilled
kubectl describe pod my-app-pod | grep -i oom

# Output:
Last State: Terminated
  Reason: OOMKilled
  Exit Code: 137

# Fix: Increase memory limits
spec:
  containers:
  - name: my-app
    resources:
      requests:
        memory: "256Mi"
      limits:
        memory: "1Gi"  # Increase this
```

#### Debug CPU Throttling
```bash
# Check if pod is CPU throttled
kubectl describe pod my-app-pod

# Look for:
State: Running
CPU: 1000m (throttled)

# Fix: Increase CPU limits or remove them
spec:
  containers:
  - name: my-app
    resources:
      limits:
        cpu: "2000m"  # Increase or remove
```

### 5. ConfigMap and Secret Issues

#### Debug ConfigMap
```bash
# View ConfigMap
kubectl get configmap my-config -o yaml

# Check if mounted correctly
kubectl describe pod my-app-pod | grep -A 10 Mounts

# Exec into pod and verify
kubectl exec my-app-pod -- cat /etc/config/app.conf

# If file is empty or wrong, check mounting:
spec:
  containers:
  - name: my-app
    volumeMounts:
    - name: config-volume
      mountPath: /etc/config
  volumes:
  - name: config-volume
    configMap:
      name: my-config  # Must exist
```

#### Debug Secrets
```bash
# View secret (base64 encoded)
kubectl get secret my-secret -o yaml

# Decode secret value
kubectl get secret my-secret -o jsonpath='{.data.password}' | base64 --decode

# Check if pod can access secret
kubectl exec my-app-pod -- env | grep PASSWORD

# Common issue: Secret not mounted as env var
spec:
  containers:
  - name: my-app
    env:
    - name: DATABASE_PASSWORD
      valueFrom:
        secretKeyRef:
          name: my-secret
          key: password  # Key must exist in secret
```

### 6. Debugging Probes

#### Liveness Probe Failures
```bash
# Check probe configuration
kubectl describe pod my-app-pod | grep -A 10 Liveness

# Example output:
Liveness: http-get http://:8000/health delay=0s timeout=1s period=10s
Warning: Unhealthy: Liveness probe failed: HTTP probe failed with statuscode: 500

# Debug:
# 1. Test endpoint manually
kubectl port-forward my-app-pod 8080:8000
curl http://localhost:8080/health

# 2. Check logs for errors during health check
kubectl logs my-app-pod | grep health

# 3. Adjust probe timing
spec:
  containers:
  - name: my-app
    livenessProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 30  # Give app time to start
      periodSeconds: 10
      timeoutSeconds: 5
      failureThreshold: 3      # Allow some failures
```

#### Readiness Probe Failures
```bash
# Pod shows 0/1 READY
kubectl get pods
NAME         READY   STATUS
my-app-pod   0/1     Running

# Check why
kubectl describe pod my-app-pod | grep -A 10 Readiness

# Common causes:
# 1. Endpoint not ready (waiting for DB connection)
# 2. Endpoint returns error
# 3. Timeout too short

# Fix: Ensure readiness endpoint properly indicates when app is ready
# health.py
@app.get("/ready")
async def readiness_check():
    # Check dependencies
    if not database_connected():
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "reason": "database unavailable"}
        )
    return {"status": "ready"}
```

### 7. Debugging Events

#### View Cluster Events
```bash
# All events in namespace
kubectl get events --sort-by='.lastTimestamp'

# Events for specific pod
kubectl get events --field-selector involvedObject.name=my-app-pod

# Watch events in real-time
kubectl get events --watch

# Events in all namespaces
kubectl get events --all-namespaces --sort-by='.lastTimestamp'
```

#### Common Event Messages
```
# Scheduling issues
Warning  FailedScheduling  0/3 nodes available: insufficient memory

# Image issues
Warning  Failed  Error: ImagePullBackOff

# Container issues
Warning  BackOff  Back-off restarting failed container

# Probe issues
Warning  Unhealthy  Liveness probe failed: Get http://10.244.0.5:8000/health: dial tcp 10.244.0.5:8000: connect: connection refused
```

---

## Python Debugging

### 1. Using Print Statements (The Beginner's Friend)

#### Strategic Print Debugging
```python
# app.py
def predict(data):
    print(f"[DEBUG] Received data: {data}")  # What came in?

    processed = preprocess(data)
    print(f"[DEBUG] After preprocessing: {processed}")  # Transformation correct?

    prediction = model.predict(processed)
    print(f"[DEBUG] Raw prediction: {prediction}")  # Model output

    result = postprocess(prediction)
    print(f"[DEBUG] Final result: {result}")  # What we're returning

    return result

# Better: Use logging instead
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def predict(data):
    logger.debug(f"Received data: {data}")
    processed = preprocess(data)
    logger.debug(f"After preprocessing: {processed}")
    return model.predict(processed)
```

### 2. Python Debugger (pdb)

#### Basic pdb Usage
```python
# Add breakpoint in code
import pdb

def train_model(data):
    X_train, y_train = split_data(data)

    # Drop into debugger here
    pdb.set_trace()  # Python < 3.7
    # OR
    breakpoint()     # Python >= 3.7

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model
```

#### pdb Commands
```python
# Run the code above, when it hits breakpoint:

# Commands:
(Pdb) h          # Help
(Pdb) l          # List source code around current line
(Pdb) ll         # List entire function
(Pdb) p X_train  # Print variable
(Pdb) pp X_train # Pretty-print variable
(Pdb) type(X_train)  # Check type
(Pdb) len(X_train)   # Check length

# Navigation:
(Pdb) n          # Next line (step over)
(Pdb) s          # Step into function
(Pdb) c          # Continue execution
(Pdb) r          # Return from current function
(Pdb) q          # Quit debugger

# Conditions:
(Pdb) b 45       # Set breakpoint at line 45
(Pdb) b train_model  # Set breakpoint at function
(Pdb) b app.py:45    # Set breakpoint in specific file
(Pdb) condition 1 X_train.shape[0] > 1000  # Conditional breakpoint
(Pdb) ignore 1 10    # Ignore breakpoint 1 for 10 hits

# Execute code:
(Pdb) X_train.shape[0]  # Execute any Python
(Pdb) !import sys       # Execute command
```

#### Example Debugging Session
```python
# bug.py
def calculate_average(numbers):
    total = sum(numbers)
    count = len(numbers)
    average = total / count  # Bug: division by zero if empty list
    return average

# Debug session:
$ python -m pdb bug.py
(Pdb) b calculate_average
(Pdb) c
(Pdb) numbers = []
(Pdb) calculate_average(numbers)
# ...
ZeroDivisionError: division by zero
(Pdb) p numbers   # Empty list!
[]
(Pdb) p count
0
# Found the bug: need to handle empty list
```

### 3. Logging Best Practices

#### Setting Up Proper Logging
```python
# logging_config.py
import logging
import sys

def setup_logging(level=logging.INFO):
    """Configure application logging"""

    # Format: timestamp - logger name - level - message
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))

    # File handler
    file_handler = logging.FileHandler('app.log')
    file_handler.setFormatter(logging.Formatter(log_format))

    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=[console_handler, file_handler]
    )

# app.py
import logging
from logging_config import setup_logging

setup_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def predict(data):
    logger.info(f"Starting prediction for {len(data)} samples")

    try:
        result = model.predict(data)
        logger.info("Prediction successful")
        return result
    except ValueError as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)  # Include stack trace
        raise
```

#### Log Levels - When to Use Each
```python
# DEBUG - Detailed diagnostic information
logger.debug(f"Data shape: {X.shape}, dtype: {X.dtype}")
logger.debug(f"Model parameters: {model.get_params()}")

# INFO - General informational messages
logger.info("Starting model training")
logger.info(f"Processed {count} records")
logger.info("Model training completed successfully")

# WARNING - Something unexpected but not an error
logger.warning(f"Training data only has {len(X)} samples, recommend > 1000")
logger.warning("API rate limit approaching (80% used)")

# ERROR - Error occurred but application continues
logger.error(f"Failed to load model from {path}: {e}")
logger.error("Database query failed, using cached results")

# CRITICAL - Serious error, application may not continue
logger.critical("Database connection lost, shutting down")
logger.critical("Out of memory, cannot proceed")
```

#### Structured Logging for Production
```python
# Use structured logging for easier parsing
import json
import logging

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_data)

# Usage
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)

# Output: {"timestamp": "2025-01-15 10:30:00", "level": "INFO", "logger": "app", "message": "Prediction completed"}
```

### 4. Common Python Errors and Fixes

#### TypeError
```python
# Error: TypeError: can only concatenate str (not "int") to str
result = "Score: " + 95  # Wrong!

# Fix:
result = "Score: " + str(95)  # Convert to string
result = f"Score: {95}"        # Use f-string
```

#### AttributeError
```python
# Error: AttributeError: 'NoneType' object has no attribute 'predict'
model = None
prediction = model.predict(X)  # model is None!

# Fix: Check for None
if model is None:
    raise ValueError("Model not loaded")
prediction = model.predict(X)
```

#### KeyError
```python
# Error: KeyError: 'missing_key'
config = {'host': 'localhost'}
port = config['port']  # Key doesn't exist!

# Fix: Use .get() with default
port = config.get('port', 8000)  # Returns 8000 if 'port' not in config

# Or check first
if 'port' in config:
    port = config['port']
else:
    port = 8000
```

#### IndexError
```python
# Error: IndexError: list index out of range
data = [1, 2, 3]
value = data[5]  # Only 3 elements!

# Fix: Check length
if len(data) > 5:
    value = data[5]
else:
    value = None

# Or use try/except
try:
    value = data[5]
except IndexError:
    value = None
```

#### ImportError / ModuleNotFoundError
```python
# Error: ModuleNotFoundError: No module named 'tensorflow'

# Fix 1: Install the module
# $ pip install tensorflow

# Fix 2: Check you're in the right environment
# $ which python
# $ pip list | grep tensorflow

# Fix 3: Add to requirements.txt
# tensorflow==2.13.0

# Fix 4: Check spelling
import tensorflow as tf  # Not 'tensorFlow' or 'tensor_flow'
```

---

## Network Debugging

### 1. Basic Network Tools

#### curl - Test HTTP Endpoints
```bash
# Basic GET request
curl http://localhost:8000/health

# Verbose output (see full request/response)
curl -v http://localhost:8000/health

# Include headers in output
curl -i http://localhost:8000/health

# POST request with JSON data
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1, 2, 3, 4]}'

# Save response to file
curl http://localhost:8000/model -o model.pkl

# Follow redirects
curl -L http://localhost:8000/redirect

# Custom headers
curl -H "Authorization: Bearer token123" \
     -H "Custom-Header: value" \
     http://localhost:8000/api

# Test with timeout
curl --max-time 5 http://slow-api.com

# Show only response code
curl -o /dev/null -s -w "%{http_code}\n" http://localhost:8000/health
```

#### ping - Test Basic Connectivity
```bash
# Ping a host
ping google.com

# Ping specific number of times
ping -c 4 google.com

# Ping with timeout
ping -W 2 google.com

# Ping container by name (from another container)
docker exec container-a ping container-b

# Ping Kubernetes service
kubectl exec my-app-pod -- ping database-service
```

#### nslookup/dig - DNS Debugging
```bash
# Look up domain
nslookup example.com

# Look up with specific DNS server
nslookup example.com 8.8.8.8

# Reverse lookup
nslookup 1.2.3.4

# In Kubernetes - check service DNS
kubectl exec my-app-pod -- nslookup database-service
# Should return: Name: database-service.default.svc.cluster.local

# Use dig for more details
dig example.com
dig @8.8.8.8 example.com  # Use specific DNS server
```

#### netstat/ss - Check Open Ports
```bash
# Show listening TCP ports
netstat -tlnp

# Show all TCP connections
netstat -tan

# Show listening UDP ports
netstat -ulnp

# Modern alternative: ss
ss -tlnp          # TCP listening ports
ss -tan           # All TCP connections
ss -o state established '( sport = :8000 )'  # Connections on port 8000
```

#### telnet/nc - Test Port Connectivity
```bash
# Test if port is open
telnet localhost 8000
# If connects: port is open
# If "Connection refused": nothing listening
# If hangs: firewall blocking

# Alternative: nc (netcat)
nc -zv localhost 8000
# Output: Connection to localhost 8000 port [tcp/*] succeeded!

# Test range of ports
nc -zv localhost 8000-8010

# Simple port listener (for testing)
nc -l 8000  # Listen on port 8000
```

### 2. Debugging API Calls

#### Check API is Accessible
```bash
# 1. Check locally (from server)
curl http://localhost:8000/health
# Should return: {"status": "healthy"}

# 2. Check from Docker host
curl http://localhost:8000/health

# 3. Check from another container
docker exec other-container curl http://my-app:8000/health

# 4. Check from Kubernetes pod
kubectl exec debug-pod -- curl http://my-app-service/health
```

#### Debug Slow API Responses
```bash
# Measure request time
time curl http://localhost:8000/predict -d '{"data": [1,2,3]}'

# Detailed timing
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/predict

# curl-format.txt:
    time_namelookup: %{time_namelookup}s
       time_connect: %{time_connect}s
    time_appconnect: %{time_appconnect}s
   time_pretransfer: %{time_pretransfer}s
      time_redirect: %{time_redirect}s
 time_starttransfer: %{time_starttransfer}s
                    ----------
         time_total: %{time_total}s
```

#### Debug API Authentication Issues
```bash
# Test without auth (should fail)
curl http://localhost:8000/api/protected
# Response: 401 Unauthorized

# Test with Bearer token
curl -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJh..." \
     http://localhost:8000/api/protected

# Test with Basic Auth
curl -u username:password http://localhost:8000/api/protected

# Test with API key
curl -H "X-API-Key: abc123" http://localhost:8000/api/protected

# Debug: Check what headers are sent
curl -v -H "Authorization: Bearer token" http://localhost:8000/api/protected 2>&1 | grep ">"
```

### 3. Debugging Network Issues in Docker

#### Container Can't Reach Host
```bash
# From container, try to reach host
docker exec my-container curl http://host.docker.internal:8000

# On Linux, use host's IP
docker exec my-container curl http://172.17.0.1:8000

# Check container's network settings
docker inspect my-container | jq '.[0].NetworkSettings'
```

#### Containers Can't Communicate
```bash
# List networks
docker network ls

# Check which network containers are on
docker inspect container-a | jq '.[0].NetworkSettings.Networks'
docker inspect container-b | jq '.[0].NetworkSettings.Networks'

# If on different networks, connect them to same network
docker network create my-network
docker network connect my-network container-a
docker network connect my-network container-b

# Now test connectivity
docker exec container-a ping container-b
docker exec container-a curl http://container-b:8000
```

### 4. Debugging Network Issues in Kubernetes

#### Pod Can't Reach External Service
```bash
# Test from pod
kubectl exec my-app-pod -- curl https://api.external-service.com

# If fails, check DNS
kubectl exec my-app-pod -- nslookup api.external-service.com

# If DNS fails, check CoreDNS
kubectl get pods -n kube-system | grep coredns
kubectl logs -n kube-system coredns-xxxxx

# Check DNS configuration
kubectl exec my-app-pod -- cat /etc/resolv.conf
```

#### Pods Can't Communicate
```bash
# From pod-a, try to reach pod-b service
kubectl exec pod-a -- curl http://service-b

# If fails, check service endpoints
kubectl get endpoints service-b
# Should show IP addresses of pods

# Check network policies
kubectl get networkpolicies
kubectl describe networkpolicy my-policy

# Test with debug pod
kubectl run debug --image=nicolaka/netshoot -it --rm -- bash
# Inside pod:
$ curl http://service-b
$ nslookup service-b
$ traceroute service-b
```

---

## Common Error Patterns

### 1. "Connection Refused"

**What it means:** Nothing is listening on that port

**Debug steps:**
```bash
# 1. Check if application is running
ps aux | grep python

# 2. Check if it's listening on correct port
netstat -tlnp | grep 8000
ss -tlnp | grep 8000

# 3. Check if binding to correct interface
# Wrong: app.run(host='127.0.0.1')  # Only localhost
# Right: app.run(host='0.0.0.0')    # All interfaces

# 4. Check firewall
sudo iptables -L
sudo ufw status
```

### 2. "Cannot Connect to Docker Daemon"

```bash
# Error: Cannot connect to the Docker daemon at unix:///var/run/docker.sock

# Check if Docker is running
sudo systemctl status docker

# Start Docker
sudo systemctl start docker

# Check permissions
sudo usermod -aG docker $USER
# Then log out and back in

# Check socket permissions
ls -la /var/run/docker.sock
```

### 3. "Port Already in Use"

```bash
# Error: bind: address already in use

# Find what's using the port
sudo lsof -i :8000
sudo netstat -tlnp | grep 8000
sudo ss -tlnp | grep 8000

# Kill the process
kill <PID>

# Or use different port
docker run -p 8080:8000 my-app
```

### 4. "No Space Left on Device"

```bash
# Check disk usage
df -h

# Check Docker disk usage
docker system df

# Clean up
docker system prune -a
docker volume prune
docker image prune -a

# Check for large logs
find /var/lib/docker/containers -name "*-json.log" -exec ls -lh {} \;
```

### 5. "ImagePullBackOff"

```bash
# Kubernetes can't pull image

# Check image name
kubectl describe pod my-pod | grep Image

# Try pulling manually
docker pull <image-name>

# If private registry, check secret
kubectl get secret regcred -o yaml

# Create secret
kubectl create secret docker-registry regcred \
  --docker-server=<registry> \
  --docker-username=<user> \
  --docker-password=<pass>
```

---

## Debugging Tools Reference

### Quick Reference Table

| Tool | Purpose | Example |
|------|---------|---------|
| `docker logs` | View container logs | `docker logs -f my-container` |
| `docker exec` | Run command in container | `docker exec -it my-container bash` |
| `docker inspect` | View container details | `docker inspect my-container` |
| `kubectl logs` | View pod logs | `kubectl logs my-pod -f` |
| `kubectl describe` | View pod/resource details | `kubectl describe pod my-pod` |
| `kubectl exec` | Run command in pod | `kubectl exec -it my-pod -- bash` |
| `curl` | Test HTTP endpoints | `curl -v http://localhost:8000` |
| `ping` | Test connectivity | `ping google.com` |
| `netstat`/`ss` | View network connections | `ss -tlnp` |
| `nslookup` | DNS lookup | `nslookup example.com` |
| `pdb` | Python debugger | `breakpoint()` |

---

## Step-by-Step Troubleshooting Workflows

### Workflow 1: Container Won't Start

```
1. Check container status
   └─> docker ps -a

2. Read logs
   └─> docker logs <container-id>

3. If logs show error:
   ├─> Module not found → Add to requirements.txt, rebuild
   ├─> File not found → Check COPY commands in Dockerfile
   ├─> Permission denied → Check file ownership, user in Dockerfile
   └─> Port in use → Change port or stop conflicting service

4. If no clear error, inspect config
   └─> docker inspect <container-id>

5. Try running interactively
   └─> docker run -it --entrypoint /bin/bash <image>

6. Test commands manually inside container
```

### Workflow 2: Application Not Responding

```
1. Verify container/pod is running
   └─> docker ps / kubectl get pods

2. Check application logs
   └─> docker logs / kubectl logs

3. Test connectivity locally
   ├─> docker exec <container> curl localhost:8000
   └─> kubectl exec <pod> -- curl localhost:8000

4. If local test works, check port mapping
   ├─> Docker: docker port <container>
   └─> K8s: kubectl describe service

5. Test from outside
   ├─> curl http://localhost:8000
   └─> kubectl port-forward pod/<pod> 8080:8000

6. If still fails, check network/firewall
```

### Workflow 3: Kubernetes Pod Failing

```
1. Check pod status
   └─> kubectl get pod <pod-name>

2. Describe pod (most important!)
   └─> kubectl describe pod <pod-name>

3. Check Events section for:
   ├─> ImagePullBackOff → Check image name/registry auth
   ├─> CrashLoopBackOff → Check logs for app error
   ├─> Pending → Check resource availability
   └─> OOMKilled → Increase memory limits

4. Check logs
   ├─> kubectl logs <pod-name>
   └─> kubectl logs <pod-name> --previous (if restarted)

5. Check service if pod is running
   ├─> kubectl describe service <service-name>
   └─> kubectl get endpoints <service-name>

6. Exec into pod and debug
   └─> kubectl exec -it <pod-name> -- /bin/bash
```

### Workflow 4: Slow API Performance

```
1. Measure response time
   └─> time curl http://localhost:8000/predict

2. Check resource usage
   ├─> docker stats
   └─> kubectl top pod

3. Check application logs for errors
   └─> Look for exceptions, warnings

4. Profile the code
   ├─> Add timing logs
   └─> Use Python profiler (cProfile)

5. Check database queries
   ├─> Enable query logging
   └─> Look for N+1 queries

6. Check network latency
   └─> Measure time to dependent services
```

### Workflow 5: Debugging Production Issues

```
1. Gather information
   ├─> What changed recently? (deploy, config change)
   ├─> When did it start?
   ├─> Is it affecting all users or subset?
   └─> Can you reproduce locally?

2. Check monitoring/metrics
   ├─> Error rate increased?
   ├─> Response time increased?
   ├─> CPU/Memory spike?
   └─> Database connection issues?

3. Check logs
   ├─> Application logs
   ├─> Container/pod logs
   ├─> Infrastructure logs
   └─> Search for errors around incident time

4. Check recent changes
   ├─> Recent deploys
   ├─> Configuration changes
   ├─> Infrastructure changes
   └─> Dependency updates

5. Mitigate first, debug later
   ├─> Rollback if recent deploy
   ├─> Scale up if resource issue
   ├─> Restart if temporary glitch
   └─> Then investigate root cause
```

---

## Tips for Effective Debugging

### 1. Be Systematic
- Don't randomly try things
- Form hypothesis, test it, adjust
- Document what you've tried

### 2. Read Error Messages Carefully
- Error message usually tells you what's wrong
- Pay attention to line numbers
- Look up unfamiliar errors

### 3. Start Simple
- Is it running?
- Can I reach it locally?
- Are the basics configured correctly?

### 4. Use the Right Tool
- Logs for "what happened"
- Describe for "current state"
- Exec for "interactive exploration"

### 5. Reproduce Reliably
- Can you make the error happen consistently?
- What are the exact steps?
- Does it happen in different environments?

### 6. Divide and Conquer
- Isolate components
- Test each piece separately
- Find where the failure occurs

### 7. Keep Notes
- What you tried
- What the results were
- What you learned
- Document solution for next time

---

## Additional Resources

- **Docker Debugging**: https://docs.docker.com/config/containers/logging/
- **Kubernetes Debugging**: https://kubernetes.io/docs/tasks/debug/
- **Python Debugging**: https://docs.python.org/3/library/pdb.html
- **Network Troubleshooting**: https://www.redhat.com/sysadmin/troubleshooting-network-issues

---

**Remember:** Debugging is a skill that improves with practice. Every bug you fix makes you a better engineer!
