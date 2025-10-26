# Step-by-Step Implementation Guide: ConfigMaps & Secrets

## Overview

Manage configuration and sensitive data securely in Kubernetes! Learn ConfigMaps for application config, Secrets for credentials, external secret management, and production security best practices for ML applications.

**Time**: 1-2 hours | **Difficulty**: Beginner to Intermediate

---

## Learning Objectives

✅ Create and use ConfigMaps for configuration
✅ Manage Secrets securely
✅ Inject config as environment variables
✅ Mount config as files
✅ Update configuration without redeployment
✅ Use external secret managers (Sealed Secrets, External Secrets Operator)
✅ Implement secret rotation
✅ Follow security best practices

---

## Phase 1: ConfigMaps

### Create ConfigMap from Literals

```bash
# From literals
kubectl create configmap app-config \
  --from-literal=ENV=production \
  --from-literal=LOG_LEVEL=INFO \
  --from-literal=MAX_WORKERS=4

# View ConfigMap
kubectl get configmap app-config -o yaml
```

### Create ConfigMap from File

```bash
# Create config file
cat > app.properties <<EOF
database.host=postgres.default.svc.cluster.local
database.port=5432
database.name=mldb
cache.enabled=true
cache.ttl=3600
EOF

# Create ConfigMap from file
kubectl create configmap app-config \
  --from-file=app.properties

# Create from multiple files
kubectl create configmap app-config \
  --from-file=config/ \
  --from-literal=EXTRA_VAR=value
```

### Create ConfigMap from YAML

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-api-config
  labels:
    app: ml-api
data:
  # Simple key-value pairs
  ENV: "production"
  LOG_LEVEL: "INFO"
  MODEL_VERSION: "v2.1.0"

  # Multi-line values
  app.yaml: |
    server:
      host: 0.0.0.0
      port: 8080
      workers: 4

    model:
      path: /models/model.pkl
      batch_size: 32
      timeout: 30

    inference:
      max_batch_delay_ms: 100
      max_queue_size: 1000

  # Config file
  logging.conf: |
    [loggers]
    keys=root,app

    [handlers]
    keys=console,file

    [formatters]
    keys=detailed

    [logger_root]
    level=INFO
    handlers=console
```

```bash
# Apply ConfigMap
kubectl apply -f configmap.yaml

# View data
kubectl get configmap ml-api-config -o jsonpath='{.data}'
```

---

## Phase 2: Using ConfigMaps

### As Environment Variables

```yaml
# pod-env.yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-api
spec:
  containers:
  - name: api
    image: ml-api:latest
    env:
    # Single environment variable
    - name: ENVIRONMENT
      valueFrom:
        configMapKeyRef:
          name: ml-api-config
          key: ENV

    # All keys as environment variables
    envFrom:
    - configMapRef:
        name: ml-api-config

    # With prefix
    - prefix: CONFIG_
      configMapRef:
        name: ml-api-config
```

### As Volume Mounts

```yaml
# pod-volume.yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-api
spec:
  containers:
  - name: api
    image: ml-api:latest
    volumeMounts:
    # Mount all keys as files
    - name: config
      mountPath: /etc/config
      readOnly: true

    # Mount specific keys
    - name: app-config
      mountPath: /app/config/app.yaml
      subPath: app.yaml
      readOnly: true

  volumes:
  - name: config
    configMap:
      name: ml-api-config

  - name: app-config
    configMap:
      name: ml-api-config
      items:
      - key: app.yaml
        path: app.yaml
```

```bash
# Verify mounted files
kubectl exec ml-api -- ls -la /etc/config
kubectl exec ml-api -- cat /etc/config/app.yaml
```

---

## Phase 3: Secrets

### Create Secret

```bash
# From literals
kubectl create secret generic db-credentials \
  --from-literal=username=mluser \
  --from-literal=password=SecureP@ssw0rd

# From files
echo -n 'mluser' > username.txt
echo -n 'SecureP@ssw0rd' > password.txt

kubectl create secret generic db-credentials \
  --from-file=username=username.txt \
  --from-file=password=password.txt

# TLS secret
kubectl create secret tls api-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key

# Docker registry secret
kubectl create secret docker-registry regcred \
  --docker-server=registry.example.com \
  --docker-username=user \
  --docker-password=password \
  --docker-email=user@example.com
```

### Create Secret from YAML

```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: api-secrets
type: Opaque
data:
  # Base64 encoded values
  db-password: U2VjdXJlUEBzc3cwcmQ=
  api-key: eW91ci1hcGkta2V5LWhlcmU=
  jwt-secret: c3VwZXItc2VjcmV0LWp3dC1rZXk=
stringData:
  # Plain text (auto-encoded)
  db-username: mluser
  redis-url: redis://redis:6379/0
```

```bash
# Encode secrets
echo -n 'SecureP@ssw0rd' | base64

# Apply secret
kubectl apply -f secret.yaml

# View secret (data hidden)
kubectl get secret api-secrets -o yaml
```

---

## Phase 4: Using Secrets

### As Environment Variables

```yaml
# deployment-with-secrets.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: api
        image: ml-api:latest
        env:
        # Single secret
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: db-password

        # Multiple secrets
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: api-key

        # All secrets as env vars
        envFrom:
        - secretRef:
            name: api-secrets
```

### As Volume Mounts

```yaml
spec:
  containers:
  - name: api
    image: ml-api:latest
    volumeMounts:
    - name: secrets
      mountPath: /etc/secrets
      readOnly: true

  volumes:
  - name: secrets
    secret:
      secretName: api-secrets
      defaultMode: 0400  # Read-only for owner
```

```bash
# Verify
kubectl exec <pod> -- ls -la /etc/secrets
kubectl exec <pod> -- cat /etc/secrets/db-password
```

### Image Pull Secrets

```yaml
spec:
  imagePullSecrets:
  - name: regcred
  containers:
  - name: app
    image: registry.example.com/ml-api:latest
```

---

## Phase 5: Update Configuration

### ConfigMap Updates

```bash
# Edit ConfigMap
kubectl edit configmap ml-api-config

# Or patch
kubectl patch configmap ml-api-config -p '
data:
  LOG_LEVEL: "DEBUG"'

# Or replace from file
kubectl create configmap ml-api-config \
  --from-file=app.yaml \
  --dry-run=client -o yaml | kubectl replace -f -
```

**Important**: Pods don't automatically reload ConfigMaps!

### Trigger Pod Restart

```bash
# Method 1: Rollout restart
kubectl rollout restart deployment ml-api

# Method 2: Add annotation to trigger update
kubectl patch deployment ml-api -p \
  '{"spec":{"template":{"metadata":{"annotations":{"reloaded-at":"'$(date +%s)'"}}}}}'

# Method 3: Use checksum annotation (in template)
```

### Automatic Reload with Checksum

```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    metadata:
      annotations:
        checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
```

When ConfigMap changes, checksum changes → pods restart

---

## Phase 6: External Secrets Management

### Sealed Secrets

```bash
# Install Sealed Secrets controller
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/controller.yaml

# Install kubeseal CLI
brew install kubeseal

# Create sealed secret
echo -n 'SecureP@ssw0rd' | \
  kubectl create secret generic db-password \
  --dry-run=client \
  --from-file=password=/dev/stdin -o yaml | \
  kubeseal -o yaml > sealed-secret.yaml

# Apply (safe to commit to git)
kubectl apply -f sealed-secret.yaml

# Controller decrypts to regular Secret
kubectl get secret db-password
```

### External Secrets Operator

```yaml
# Install ESO
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets

# Configure AWS Secrets Manager backend
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-west-2
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets-sa
---
# External Secret
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: ml-api-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets
    kind: SecretStore
  target:
    name: api-secrets
    creationPolicy: Owner
  data:
  - secretKey: db-password
    remoteRef:
      key: production/ml-api/db-password
  - secretKey: api-key
    remoteRef:
      key: production/ml-api/api-key
```

---

## Phase 7: Production Best Practices

### Secret Security Checklist

```yaml
# ✅ Good practices
apiVersion: v1
kind: Secret
metadata:
  name: secure-secret
type: Opaque
stringData:
  password: "{{ .Values.externalSecret }}"  # From Helm values
---
# ❌ Bad practices - DO NOT DO THIS
apiVersion: v1
kind: Secret
data:
  password: cGFzc3dvcmQxMjM=  # Never hardcode in git!
```

### RBAC for Secrets

```yaml
# Restrict secret access
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: secret-reader
rules:
- apiGroups: [""]
  resources: ["secrets"]
  resourceNames: ["api-secrets"]  # Specific secrets only
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-api-secrets
subjects:
- kind: ServiceAccount
  name: ml-api
roleRef:
  kind: Role
  name: secret-reader
  apiGroup: rbac.authorization.k8s.io
```

### Encryption at Rest

```bash
# Enable encryption at rest (cluster-level)
# /etc/kubernetes/enc/enc.yaml
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
  - resources:
      - secrets
    providers:
      - aescbc:
          keys:
            - name: key1
              secret: <BASE64_ENCODED_SECRET>
      - identity: {}

# Add to kube-apiserver:
--encryption-provider-config=/etc/kubernetes/enc/enc.yaml
```

### Secret Rotation

```bash
#!/bin/bash
# rotate-secret.sh

SECRET_NAME="api-secrets"
NEW_PASSWORD=$(openssl rand -base64 32)

# Update secret
kubectl patch secret $SECRET_NAME -p "{
  \"data\": {
    \"db-password\": \"$(echo -n $NEW_PASSWORD | base64)\"
  }
}"

# Restart pods
kubectl rollout restart deployment ml-api

# Update database password
kubectl exec postgres-0 -- psql -U postgres -c \
  "ALTER USER mluser PASSWORD '$NEW_PASSWORD'"

echo "Secret rotated successfully"
```

---

## Phase 8: ML-Specific Patterns

### Model Registry Credentials

```yaml
# model-registry-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: mlflow-credentials
stringData:
  MLFLOW_TRACKING_URI: https://mlflow.example.com
  MLFLOW_TRACKING_USERNAME: mlops
  MLFLOW_TRACKING_PASSWORD: secure-password
  AWS_ACCESS_KEY_ID: AKIAIOSFODNN7EXAMPLE
  AWS_SECRET_ACCESS_KEY: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

### Multi-Environment Configuration

```yaml
# configmap-dev.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-api-config
  namespace: development
data:
  ENV: "development"
  LOG_LEVEL: "DEBUG"
  MODEL_PATH: "s3://dev-models/latest"
---
# configmap-prod.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-api-config
  namespace: production
data:
  ENV: "production"
  LOG_LEVEL: "WARNING"
  MODEL_PATH: "s3://prod-models/v2.1.0"
```

### API Keys Management

```yaml
# api-keys-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: api-keys
stringData:
  # External APIs
  openai-api-key: sk-...
  huggingface-token: hf_...
  google-api-key: AIza...

  # Internal services
  inference-api-key: internal-key-xyz
  monitoring-api-key: mon-key-abc
```

---

## Troubleshooting

```bash
# Secret not found
kubectl describe pod <pod> | grep -A 10 Events
# Error: secret "api-secrets" not found

# ConfigMap not updated
kubectl get configmap ml-api-config -o yaml
kubectl exec <pod> -- env | grep LOG_LEVEL
# Restart deployment if needed

# Permission denied
kubectl get secret api-secrets
# Error from server (Forbidden): secrets "api-secrets" is forbidden

# Debug secret values (BE CAREFUL!)
kubectl get secret api-secrets -o jsonpath='{.data.db-password}' | base64 -d
```

---

## Verification Checklist

✅ ConfigMaps created for non-sensitive config
✅ Secrets created for credentials/keys
✅ Secrets NOT committed to git
✅ Using external secret management in production
✅ RBAC configured to restrict secret access
✅ Encryption at rest enabled
✅ Secret rotation implemented
✅ Pods restart when config changes
✅ Monitoring config/secret usage
✅ Documented secret rotation procedures

---

## Best Practices

✅ Never commit secrets to git
✅ Use external secret managers (AWS Secrets Manager, Vault)
✅ Implement secret rotation
✅ Use RBAC to restrict access
✅ Enable encryption at rest
✅ Use Sealed Secrets or External Secrets Operator
✅ Mount secrets as files, not env vars (more secure)
✅ Set appropriate file permissions (0400)
✅ Audit secret access
✅ Use separate secrets per environment

---

**ConfigMaps and Secrets mastered!** 🔐

**Next Exercise**: Ingress and Load Balancing
