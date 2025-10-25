# ConfigMaps & Secrets: Step-by-Step Guide

This guide walks you through the ConfigMaps and Secrets exercise from basic concepts to advanced patterns.

## Table of Contents

- [Part 1: Understanding the Basics](#part-1-understanding-the-basics)
- [Part 2: Creating ConfigMaps](#part-2-creating-configmaps)
- [Part 3: Creating Secrets](#part-3-creating-secrets)
- [Part 4: Using ConfigMaps in Pods](#part-4-using-configmaps-in-pods)
- [Part 5: Using Secrets in Pods](#part-5-using-secrets-in-pods)
- [Part 6: Security Best Practices](#part-6-security-best-practices)
- [Part 7: Configuration Patterns](#part-7-configuration-patterns)
- [Part 8: Secret Rotation](#part-8-secret-rotation)
- [Part 9: Troubleshooting](#part-9-troubleshooting)
- [Part 10: Production Deployment](#part-10-production-deployment)

## Part 1: Understanding the Basics

### What Are ConfigMaps?

ConfigMaps store non-sensitive configuration data as key-value pairs. They decouple configuration from container images, making applications portable.

**Use ConfigMaps for:**
- Application settings (log levels, database hosts, ports)
- Feature flags
- Configuration files (nginx.conf, application.properties)
- Command-line arguments
- Environment variables

**Do NOT use ConfigMaps for:**
- Passwords
- API keys
- Tokens
- Certificates
- Any sensitive data

### What Are Secrets?

Secrets store sensitive data in base64-encoded form. While not encrypted by default, they can be encrypted at rest with proper cluster configuration.

**Use Secrets for:**
- Database passwords
- API keys
- OAuth tokens
- TLS certificates
- SSH keys
- Docker registry credentials

**Important:** Secrets are base64-encoded, NOT encrypted. Anyone with read access can decode them:

```bash
echo "c3VwZXJzZWNyZXQ=" | base64 -d
# Output: supersecret
```

### Key Differences

| Feature | ConfigMap | Secret |
|---------|-----------|---------|
| Purpose | Non-sensitive config | Sensitive data |
| Encoding | Plain text | Base64 |
| Encryption | No | Optional (at rest) |
| RBAC | Standard | Should be stricter |
| Best Practice | Public configuration | Private credentials |

### Step 1.1: Set Up Your Environment

```bash
# Check cluster connection
kubectl cluster-info

# Create namespace
kubectl apply -f manifests/01-namespace.yaml

# Verify namespace
kubectl get namespace config-demo
```

Expected output:
```
NAME          STATUS   AGE
config-demo   Active   5s
```

## Part 2: Creating ConfigMaps

### Step 2.1: Create ConfigMap from Literal Values

```bash
# Create ConfigMap imperatively
kubectl create configmap app-config \
  --from-literal=APP_NAME="My Application" \
  --from-literal=APP_VERSION="1.0.0" \
  --from-literal=LOG_LEVEL="info" \
  --from-literal=DATABASE_HOST="postgres.database.svc.cluster.local" \
  --from-literal=DATABASE_PORT="5432" \
  -n config-demo

# View the ConfigMap
kubectl get configmap app-config -n config-demo -o yaml
```

Expected output:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: config-demo
data:
  APP_NAME: "My Application"
  APP_VERSION: "1.0.0"
  DATABASE_HOST: "postgres.database.svc.cluster.local"
  DATABASE_PORT: "5432"
  LOG_LEVEL: "info"
```

### Step 2.2: Create ConfigMap from Files

```bash
# Create a sample configuration file
cat > /tmp/app.properties << 'EOF'
server.port=8080
server.host=0.0.0.0
database.url=jdbc:postgresql://postgres:5432/mydb
database.pool.size=20
logging.level=INFO
EOF

# Create ConfigMap from file
kubectl create configmap app-properties \
  --from-file=/tmp/app.properties \
  -n config-demo

# View the ConfigMap
kubectl describe configmap app-properties -n config-demo
```

The file content becomes a single key in the ConfigMap:
```
Data
====
app.properties:
----
server.port=8080
server.host=0.0.0.0
...
```

### Step 2.3: Create ConfigMap from Directory

```bash
# Create directory with multiple config files
mkdir -p /tmp/configs
cat > /tmp/configs/database.conf << 'EOF'
host=postgres
port=5432
database=mydb
EOF

cat > /tmp/configs/redis.conf << 'EOF'
host=redis
port=6379
EOF

# Create ConfigMap from directory
kubectl create configmap app-configs \
  --from-file=/tmp/configs/ \
  -n config-demo

# View the ConfigMap
kubectl get configmap app-configs -n config-demo -o yaml
```

Each file becomes a separate key:
```yaml
data:
  database.conf: |
    host=postgres
    port=5432
  redis.conf: |
    host=redis
    port=6379
```

### Step 2.4: Create ConfigMap Declaratively

```bash
# Deploy the example ConfigMaps
kubectl apply -f manifests/02-configmap-examples.yaml

# List all ConfigMaps
kubectl get configmaps -n config-demo
```

Expected output:
```
NAME                   DATA   AGE
app-config-literals    7      10s
app-config-files       5      10s
app-config-binary      1      10s
app-config-dev         6      10s
app-config-prod        6      10s
app-scripts            3      10s
```

### Step 2.5: Explore ConfigMap Contents

```bash
# View specific ConfigMap
kubectl get configmap app-config-literals -n config-demo -o yaml

# Get specific key value
kubectl get configmap app-config-literals -n config-demo \
  -o jsonpath='{.data.APP_NAME}'

# View multi-line content (like nginx.conf)
kubectl get configmap app-config-files -n config-demo \
  -o jsonpath='{.data.nginx\.conf}' | head -20
```

## Part 3: Creating Secrets

### Step 3.1: Create Secret from Literal Values

```bash
# Create Secret imperatively
kubectl create secret generic app-secrets \
  --from-literal=database-username=appuser \
  --from-literal=database-password=supersecret \
  --from-literal=api-key=api-key-1234567890 \
  -n config-demo

# View the Secret (data is base64 encoded)
kubectl get secret app-secrets -n config-demo -o yaml
```

Expected output:
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
  namespace: config-demo
type: Opaque
data:
  api-key: YXBpLWtleS0xMjM0NTY3ODkw
  database-password: c3VwZXJzZWNyZXQ=
  database-username: YXBwdXNlcg==
```

**Important:** The data is base64 encoded:
```bash
echo "c3VwZXJzZWNyZXQ=" | base64 -d
# Output: supersecret
```

### Step 3.2: Create TLS Secret

```bash
# Generate self-signed certificate (for testing)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /tmp/tls.key \
  -out /tmp/tls.crt \
  -subj "/CN=test.example.com/O=test"

# Create TLS secret
kubectl create secret tls tls-secret \
  --cert=/tmp/tls.crt \
  --key=/tmp/tls.key \
  -n config-demo

# View the secret
kubectl describe secret tls-secret -n config-demo
```

### Step 3.3: Create Docker Registry Secret

```bash
# Create Docker registry secret
kubectl create secret docker-registry my-registry \
  --docker-server=https://index.docker.io/v1/ \
  --docker-username=myuser \
  --docker-password=mypassword \
  --docker-email=myemail@example.com \
  -n config-demo

# View the secret
kubectl get secret my-registry -n config-demo -o yaml
```

### Step 3.4: Create Secrets Declaratively

```bash
# Deploy the example Secrets
kubectl apply -f manifests/03-secret-examples.yaml

# List all Secrets
kubectl get secrets -n config-demo
```

Expected output:
```
NAME                      TYPE                             DATA   AGE
app-secrets               Opaque                           6      15s
docker-registry-secret    kubernetes.io/dockerconfigjson   1      15s
tls-secret                kubernetes.io/tls                2      15s
ssh-auth-secret           kubernetes.io/ssh-auth           1      15s
basic-auth-secret         kubernetes.io/basic-auth         2      15s
...
```

### Step 3.5: Using stringData (Convenience)

Instead of base64 encoding manually, use `stringData`:

```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: simple-secret
  namespace: config-demo
type: Opaque
stringData:
  username: "admin"
  password: "my-password"
EOF

# Kubernetes automatically base64 encodes stringData
kubectl get secret simple-secret -n config-demo -o yaml
```

The output shows `data` (base64 encoded), not `stringData`:
```yaml
data:
  password: bXktcGFzc3dvcmQ=
  username: YWRtaW4=
```

## Part 4: Using ConfigMaps in Pods

### Step 4.1: ConfigMap as Environment Variables (Individual Keys)

```bash
# Deploy pod using ConfigMap for env vars
kubectl apply -f manifests/04-using-configmaps.yaml

# Check pod status
kubectl get pod pod-configmap-env -n config-demo

# View logs to see environment variables
kubectl logs pod-configmap-env -n config-demo
```

Expected output:
```
=== Environment Variables from ConfigMap ===
APP_NAME: My Application
APP_VERSION: 1.0.0
LOG_LEVEL: info
DATABASE_HOST: postgres.database.svc.cluster.local
DATABASE_PORT: 5432
```

**How it works:**
```yaml
env:
- name: APP_NAME
  valueFrom:
    configMapKeyRef:
      name: app-config-literals
      key: APP_NAME
```

### Step 4.2: ConfigMap as Environment Variables (All Keys)

```bash
# Check pod that imports all keys
kubectl logs pod-configmap-envfrom -n config-demo
```

Expected output shows all ConfigMap keys as environment variables:
```
=== All Environment Variables ===
APP_NAME=My Application
APP_VERSION=1.0.0
DATABASE_HOST=postgres.database.svc.cluster.local
DATABASE_PORT=5432
FEATURE_FLAG_NEW_UI=true
LOG_LEVEL=info
MAX_CONNECTIONS=100
```

**How it works:**
```yaml
envFrom:
- configMapRef:
    name: app-config-literals
```

### Step 4.3: ConfigMap as Volume Mount

```bash
# Check pod with ConfigMap mounted as volume
kubectl logs pod-configmap-volume -n config-demo
```

Expected output:
```
=== ConfigMap Files ===
total 20
drwxrwxrwx 3 root root 4096 ...
-rw-r--r-- 1 root root  789 ... application.properties
-rw-r--r-- 1 root root 2341 ... nginx.conf
-rw-r--r-- 1 root root  456 ... config.json
...
```

**Explore the mounted files:**
```bash
# List files
kubectl exec pod-configmap-volume -n config-demo -- ls -la /etc/config/

# View file content
kubectl exec pod-configmap-volume -n config-demo -- cat /etc/config/nginx.conf

# Check file permissions
kubectl exec pod-configmap-volume -n config-demo -- stat /etc/config/nginx.conf
```

**How it works:**
```yaml
volumeMounts:
- name: config-volume
  mountPath: /etc/config
  readOnly: true

volumes:
- name: config-volume
  configMap:
    name: app-config-files
```

### Step 4.4: Selective Key Mounting

```bash
# Check pod with only specific keys mounted
kubectl exec pod-configmap-selective -n config-demo -- ls -la /etc/config/
```

Only the selected file is present:
```
-rw-r--r-- 1 root root 2341 ... nginx.conf
```

**How it works:**
```yaml
volumes:
- name: config-volume
  configMap:
    name: app-config-files
    items:
    - key: nginx.conf
      path: nginx.conf
```

### Step 4.5: ConfigMap with Custom Permissions

```bash
# Check file permissions
kubectl exec pod-configmap-permissions -n config-demo -- ls -la /etc/config/

# Try executing script
kubectl logs pod-configmap-permissions -n config-demo
```

Files have custom permissions (0755):
```
-rwxr-xr-x 1 root root 456 ... init.sh
```

**How it works:**
```yaml
volumes:
- name: scripts-volume
  configMap:
    name: app-scripts
    defaultMode: 0750
    items:
    - key: init.sh
      path: init.sh
      mode: 0755  # More permissive for this file
```

## Part 5: Using Secrets in Pods

### Step 5.1: Secret as Environment Variables

```bash
# Deploy pods using Secrets
kubectl apply -f manifests/05-using-secrets.yaml

# Check pod logs
kubectl logs pod-secret-env -n config-demo
```

Expected output:
```
=== Environment Variables from Secret ===
Database Username: appuser
Database Password: supersecret
API Key: api-key-1234567890
JWT Secret: jwt-secret-key-for-signing-tokens
```

**Important:** Secrets in env vars can leak in logs, error messages, and child processes. Prefer volume mounts for production.

### Step 5.2: Secret as Volume Mount (Recommended)

```bash
# Check pod logs
kubectl logs pod-secret-volume -n config-demo
```

Expected output:
```
=== Secret Files ===
total 24
drwxrwxrwt 3 root root  200 ...
-rw-r--r-- 1 root root   30 ... api-key
-rw-r--r-- 1 root root   23 ... api-secret
-rw-r--r-- 1 root root   13 ... database-password
-rw-r--r-- 1 root root    7 ... database-username
...

=== Reading Secrets from Files ===
Database Username: appuser
Database Password: supersecret
API Key: api-key-1234567890
```

**Explore mounted secrets:**
```bash
# List secret files
kubectl exec pod-secret-volume -n config-demo -- ls -la /etc/secrets/

# Read secret (base64 decoded automatically when mounted)
kubectl exec pod-secret-volume -n config-demo -- cat /etc/secrets/database-username

# Check permissions
kubectl exec pod-secret-volume -n config-demo -- stat /etc/secrets/database-password
```

### Step 5.3: Secret with Restrictive Permissions

```bash
# Check pod with 0400 permissions
kubectl exec pod-secret-permissions -n config-demo -- ls -la /etc/secrets/
```

Files are read-only for owner:
```
-r-------- 1 root root 23 ... database-password
-r-------- 1 root root  7 ... database-username
```

**Why this is important:**
- Prevents other containers from reading secrets
- Limits exposure if container is compromised
- Follows principle of least privilege

### Step 5.4: Using imagePullSecrets

```bash
# View pod definition
kubectl get pod pod-private-image -n config-demo -o yaml | grep -A 3 imagePullSecrets
```

Output:
```yaml
imagePullSecrets:
- name: docker-registry-secret
```

**Use case:** Pull images from private Docker registries.

### Step 5.5: TLS Secret with Nginx

```bash
# Check if Nginx pod is running with TLS
kubectl get pod pod-tls-secret -n config-demo

# View how TLS certs are mounted
kubectl describe pod pod-tls-secret -n config-demo | grep -A 10 "Mounts:"
```

TLS certificates are mounted at `/etc/nginx/ssl/`:
```
Mounts:
  /etc/nginx/ssl from tls-certs (ro)
```

## Part 6: Security Best Practices

### Step 6.1: Verify Secret Encoding

**Demonstration: Secrets are NOT encrypted by default**

```bash
# Get secret value (base64 encoded)
encoded=$(kubectl get secret app-secrets -n config-demo \
  -o jsonpath='{.data.database-password}')

echo "Encoded: $encoded"

# Decode it
decoded=$(echo "$encoded" | base64 -d)
echo "Decoded: $decoded"
```

**Critical lesson:** Anyone with read access to secrets can decode them!

### Step 6.2: Check RBAC Permissions

```bash
# Check if you can read secrets
kubectl auth can-i get secrets -n config-demo
kubectl auth can-i list secrets -n config-demo

# Check for a specific service account
kubectl auth can-i get secrets -n config-demo \
  --as=system:serviceaccount:config-demo:default
```

### Step 6.3: Audit Secret Access

```bash
# View events related to secrets
kubectl get events -n config-demo \
  --field-selector involvedObject.kind=Secret

# Check which pods are using a secret
kubectl get pods -n config-demo -o json | \
  jq -r '.items[] | select(
    (.spec.volumes[]?.secret.secretName == "app-secrets") or
    (.spec.containers[].envFrom[]?.secretRef.name == "app-secrets")
  ) | .metadata.name'
```

### Step 6.4: Security Context

```bash
# Check secure-app deployment
kubectl get deployment secure-app -n config-demo -o yaml | \
  grep -A 15 "securityContext:"
```

Best practices implemented:
```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  capabilities:
    drop:
    - ALL
```

### Step 6.5: Review Security Documentation

```bash
# Open security best practices
cat docs/SECURITY_BEST_PRACTICES.md | less

# Key sections:
# - Encryption at Rest
# - RBAC Configuration
# - Secret Rotation
# - External Secret Management
# - Compliance Considerations
```

## Part 7: Configuration Patterns

### Step 7.1: 12-Factor App Pattern

```bash
# Deploy 12-factor app
kubectl apply -f examples/config-patterns.yaml

# View configuration
kubectl logs deployment/twelve-factor-app -n config-demo
```

All configuration through environment variables:
```
APP_NAME=My Application
DATABASE_HOST=postgres.database.svc.cluster.local
DATABASE_PASSWORD=secure-db-password
...
```

### Step 7.2: Layered Configuration

```bash
# View layered config app
kubectl logs deployment/layered-config-app-dev -n config-demo
```

Shows base config + dev overrides:
```
APP_NAME: demo-app (from base)
LOG_LEVEL: debug (from dev override)
DATABASE_HOST: postgres-dev.svc.cluster.local (from dev override)
```

**How it works:**
```yaml
envFrom:
- configMapRef:
    name: base-config  # Applied first
- configMapRef:
    name: dev-overrides  # Overrides base
```

### Step 7.3: Sidecar Configuration Injection

```bash
# View logs from init container
kubectl logs sidecar-config-injection -n config-demo -c config-renderer

# View logs from app container
kubectl logs sidecar-config-injection -n config-demo -c app
```

Init container renders config template with secrets, then app container uses the rendered config.

### Step 7.4: Hot-Reload Pattern

```bash
# Watch hot-reload demo
kubectl logs deployment/hot-reload-app -n config-demo -c app -f
```

In another terminal:
```bash
# Edit ConfigMap
kubectl edit configmap app-config-literals -n config-demo
# Change LOG_LEVEL value

# Watch logs - config hash will change after ~60 seconds
# (kubelet sync period)
```

**Important:**
- Volume-mounted configs update automatically (~60s)
- Environment variables do NOT update (pod restart required)

### Step 7.5: Multi-Tenant Pattern

```bash
# View tenant A logs
kubectl logs deployment/app-tenant-a -n config-demo

# View tenant B logs
kubectl logs deployment/app-tenant-b -n config-demo
```

Each tenant has separate configuration and secrets.

### Step 7.6: Blue-Green Pattern

```bash
# Check blue deployment
kubectl get deployment app-blue -n config-demo
kubectl logs deployment/app-blue -n config-demo

# Check green deployment (scaled to 0)
kubectl get deployment app-green -n config-demo

# Simulate cutover
kubectl scale deployment app-green -n config-demo --replicas=3
kubectl scale deployment app-blue -n config-demo --replicas=0

# Switch service
kubectl patch service app-service -n config-demo \
  -p '{"spec":{"selector":{"deployment":"green"}}}'
```

## Part 8: Secret Rotation

### Step 8.1: Run Secret Rotation Demo

```bash
# Launch interactive rotation tool
./scripts/rotate-secrets.sh
```

**Menu options:**
1. Zero-Downtime Rotation (Dual Secrets)
2. Rolling Restart Rotation
3. Immutable Secret Rotation
4. External Secrets Operator (Simulation)
5. Blue-Green Deployment Rotation
6. Show Best Practices
7. Verify Current Secrets
8. Backup All Secrets

### Step 8.2: Zero-Downtime Rotation

```bash
# From the menu, select option 1

# This demonstrates:
# 1. Create new secret with rotated credentials
# 2. Update app to accept both old and new credentials
# 3. Wait for all connections to migrate
# 4. Remove old secret
```

**Best for:** Database passwords, API keys where dual authentication is possible.

### Step 8.3: Rolling Restart Rotation

```bash
# From the menu, select option 2

# This demonstrates:
# 1. Backup current secret
# 2. Update secret with new values
# 3. Trigger rolling restart of deployments
```

**Best for:** Simple rotation where brief downtime per pod is acceptable.

### Step 8.4: Immutable Secret Rotation

```bash
# From the menu, select option 3

# This demonstrates:
# 1. Create new immutable secret with version suffix
# 2. Update deployments to use new secret
# 3. Clean up old versioned secrets
```

**Best for:** Production environments requiring audit trails and rollback capability.

### Step 8.5: External Secrets Operator

```bash
# From the menu, select option 4

# Shows pattern with:
# - HashiCorp Vault
# - AWS Secrets Manager
# - Azure Key Vault
# - External Secrets Operator
```

**Best for:** Production systems requiring automated rotation and centralized secret management.

## Part 9: Troubleshooting

### Step 9.1: Pod Won't Start - ConfigMap Missing

```bash
# Create a pod that references non-existent ConfigMap
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: test-missing-config
  namespace: config-demo
spec:
  containers:
  - name: app
    image: busybox
    command: ["sleep", "3600"]
    envFrom:
    - configMapRef:
        name: nonexistent-configmap
EOF

# Check pod status
kubectl get pod test-missing-config -n config-demo

# Describe to see error
kubectl describe pod test-missing-config -n config-demo
```

Error message:
```
Events:
  Warning  FailedMount  configmap "nonexistent-configmap" not found
```

**Solution:**
```bash
# Create the missing ConfigMap
kubectl create configmap nonexistent-configmap \
  --from-literal=KEY=value \
  -n config-demo

# Pod will start automatically
```

### Step 9.2: Environment Variables Not Set

```bash
# Check environment variables in pod
kubectl exec pod-configmap-env -n config-demo -- env | grep APP_NAME

# If empty, verify ConfigMap exists
kubectl get configmap app-config-literals -n config-demo

# Verify key exists in ConfigMap
kubectl get configmap app-config-literals -n config-demo \
  -o jsonpath='{.data.APP_NAME}'

# Check pod definition
kubectl get pod pod-configmap-env -n config-demo -o yaml | grep -A 10 "env:"
```

### Step 9.3: ConfigMap Changes Not Reflected

```bash
# Edit ConfigMap
kubectl edit configmap app-config-literals -n config-demo
# Change LOG_LEVEL to "debug"

# For environment variables: Pod restart required
kubectl delete pod pod-configmap-env -n config-demo
# Pod will be recreated if part of deployment

# For volume mounts: Wait ~60 seconds
kubectl exec pod-configmap-volume -n config-demo -- \
  cat /etc/config/application.properties
# Changes will appear after kubelet sync
```

### Step 9.4: Permission Denied on Secret Files

```bash
# Try to read secret as different user
kubectl exec pod-secret-permissions -n config-demo -- su nobody -c \
  "cat /etc/secrets/database-password"
```

Error: Permission denied

**Solution:** This is correct behavior! Secrets should have restrictive permissions (0400).

### Step 9.5: Secret Visible in Logs

```bash
# Check for secrets in logs (common mistake)
kubectl logs deployment/secure-app -n config-demo | grep -i password
```

**If you see secrets:** The application is logging them (don't do this!).

**Best practices:**
- Never log secret values
- Use `[REDACTED]` or `***` in logs
- Sanitize error messages
- Review code for secret leakage

## Part 10: Production Deployment

### Step 10.1: Production Checklist

Before deploying to production, verify:

```bash
# 1. Namespace isolation
kubectl get namespace production || kubectl create namespace production

# 2. ConfigMaps and Secrets are environment-specific
kubectl get configmaps -n production
kubectl get secrets -n production

# 3. RBAC is configured
kubectl get rolebindings -n production

# 4. Secrets have restrictive RBAC
kubectl auth can-i get secrets -n production \
  --as=system:serviceaccount:production:default

# 5. Encryption at rest is enabled (check with cluster admin)
# 6. Audit logging is enabled (check with cluster admin)
```

### Step 10.2: Deploy Production Application

```bash
# Create production ConfigMap
kubectl create configmap app-config-prod \
  --from-literal=ENV="production" \
  --from-literal=LOG_LEVEL="warn" \
  --from-literal=DATABASE_HOST="postgres-prod.svc.cluster.local" \
  -n production

# Create production Secrets (use stringData for convenience)
kubectl apply -f - <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets-prod
  namespace: production
type: Opaque
immutable: true  # Cannot be modified
stringData:
  database-password: "$(openssl rand -base64 32)"
  api-key: "$(openssl rand -base64 32)"
  jwt-secret: "$(openssl rand -base64 32)"
EOF

# Deploy application
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: production-app
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: production-app
  template:
    metadata:
      labels:
        app: production-app
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: app
        image: busybox:1.36
        command: ["/bin/sh", "-c"]
        args:
          - |
            echo "Production app starting..."
            echo "Environment: \$ENV"
            echo "Log Level: \$LOG_LEVEL"
            DB_PASS=\$(cat /etc/secrets/database-password)
            echo "Database password length: \${#DB_PASS}"

            while true; do
              echo "[\$(date)] Production app running"
              sleep 60
            done
        envFrom:
        - configMapRef:
            name: app-config-prod
        volumeMounts:
        - name: secrets
          mountPath: /etc/secrets
          readOnly: true
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
        resources:
          requests:
            memory: "64Mi"
            cpu: "100m"
          limits:
            memory: "128Mi"
            cpu: "200m"
      volumes:
      - name: secrets
        secret:
          secretName: app-secrets-prod
          defaultMode: 0400
EOF

# Verify deployment
kubectl get deployment production-app -n production
kubectl get pods -n production -l app=production-app
```

### Step 10.3: Verify Production Security

```bash
# 1. Check security context
kubectl get deployment production-app -n production -o yaml | \
  grep -A 10 "securityContext:"

# 2. Verify secret permissions
kubectl exec -n production \
  $(kubectl get pod -n production -l app=production-app -o jsonpath='{.items[0].metadata.name}') \
  -- stat /etc/secrets/database-password

# 3. Verify running as non-root
kubectl exec -n production \
  $(kubectl get pod -n production -l app=production-app -o jsonpath='{.items[0].metadata.name}') \
  -- id

# 4. Check logs (should not contain secrets)
kubectl logs -n production \
  $(kubectl get pod -n production -l app=production-app -o jsonpath='{.items[0].metadata.name}')
```

### Step 10.4: Set Up Monitoring

```bash
# Monitor ConfigMap changes
kubectl get events -n production --watch | grep configmap

# Monitor Secret access
kubectl get events -n production --watch | grep secret

# Set up alerts (example with kubectl)
kubectl create configmap monitoring-config -n production \
  --from-literal=alert-on-secret-change="true" \
  --from-literal=alert-on-pod-failure="true"
```

### Step 10.5: Document Configuration

```bash
# Create documentation ConfigMap
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-documentation
  namespace: production
  labels:
    type: documentation
data:
  README.md: |
    # Production Application Configuration

    ## Required ConfigMaps
    - app-config-prod: Application configuration

    ## Required Secrets
    - app-secrets-prod: Database and API credentials

    ## Environment Variables
    - ENV: Environment name (production)
    - LOG_LEVEL: Logging level (warn)
    - DATABASE_HOST: Database hostname

    ## Secret Files (mounted at /etc/secrets)
    - database-password: PostgreSQL password
    - api-key: External API key
    - jwt-secret: JWT signing secret

    ## Rotation Schedule
    - Secrets rotated every 90 days
    - Next rotation: $(date -d "+90 days" +%Y-%m-%d)

    ## Contact
    - Team: Platform Engineering
    - Slack: #platform-support
EOF

# View documentation
kubectl get configmap app-documentation -n production \
  -o jsonpath='{.data.README\.md}'
```

### Step 10.6: Implement Rotation Schedule

```bash
# Create CronJob for automated rotation (example)
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: rotate-secrets
  namespace: production
spec:
  schedule: "0 2 1 */3 *"  # 2 AM on 1st day of quarter
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: secret-rotator  # Needs RBAC
          containers:
          - name: rotator
            image: bitnami/kubectl:latest
            command: ["/bin/sh", "-c"]
            args:
              - |
                echo "Starting secret rotation..."
                # Rotation logic here
                echo "Rotation complete"
          restartPolicy: OnFailure
EOF
```

## Conclusion

You've completed the ConfigMaps and Secrets exercise! You've learned:

- ✅ Creating and managing ConfigMaps and Secrets
- ✅ Different methods to inject configuration into pods
- ✅ Security best practices for secret management
- ✅ Configuration patterns for production applications
- ✅ Secret rotation strategies
- ✅ Troubleshooting common issues
- ✅ Production deployment checklist

### Next Steps

1. **Review Security Documentation**
   ```bash
   cat docs/SECURITY_BEST_PRACTICES.md
   ```

2. **Run Comprehensive Tests**
   ```bash
   ./scripts/test-configs.sh
   ```

3. **Practice Secret Rotation**
   ```bash
   ./scripts/rotate-secrets.sh
   ```

4. **Explore External Secret Management**
   - Try External Secrets Operator
   - Set up HashiCorp Vault (local)
   - Explore Sealed Secrets

5. **Move to Exercise 06: Ingress & Load Balancing**

### Key Takeaways

**ConfigMaps:**
- Use for non-sensitive configuration
- Can be updated without rebuilding images
- Volume mounts update automatically (~60s)
- Environment variables require pod restart

**Secrets:**
- Base64 encoded, NOT encrypted by default
- Enable encryption at rest in production
- Prefer volume mounts over environment variables
- Use external secret managers for production
- Implement regular rotation
- Strict RBAC and audit logging

**Production:**
- Separate namespaces per environment
- Immutable secrets for production
- Security context (non-root, read-only filesystem)
- Restrictive file permissions (0400)
- Comprehensive monitoring and auditing
- Documented rotation procedures

---

**Remember:** Configuration management is critical for cloud-native applications. Master these patterns to build secure, maintainable systems.
