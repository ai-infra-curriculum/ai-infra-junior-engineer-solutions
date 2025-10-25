# Security Best Practices for ConfigMaps and Secrets

This document provides comprehensive security guidance for managing configuration and secrets in Kubernetes.

## Table of Contents

- [Understanding Kubernetes Secrets Security](#understanding-kubernetes-secrets-security)
- [Encryption at Rest](#encryption-at-rest)
- [Access Control (RBAC)](#access-control-rbac)
- [Secret Management Patterns](#secret-management-patterns)
- [External Secret Management](#external-secret-management)
- [ConfigMap Security](#configmap-security)
- [Auditing and Monitoring](#auditing-and-monitoring)
- [Secret Rotation](#secret-rotation)
- [Container Security](#container-security)
- [CI/CD Security](#cicd-security)
- [Compliance Considerations](#compliance-considerations)
- [Security Checklist](#security-checklist)

## Understanding Kubernetes Secrets Security

### What Secrets Are NOT

**Critical Understanding:**
- Secrets in Kubernetes are **base64 encoded**, NOT encrypted
- Base64 encoding is reversible: `echo "encoded" | base64 -d`
- Anyone with read access to secrets can decode them
- Secrets in etcd are stored in plaintext by default

```bash
# This shows how easily secrets can be decoded
kubectl get secret app-secrets -n config-demo \
  -o jsonpath='{.data.database-password}' | base64 -d
```

### Default Security Model

Kubernetes secrets provide:
- ✅ Separation from code
- ✅ Access control via RBAC
- ✅ Controlled distribution to pods
- ✅ Avoiding hardcoded values

Kubernetes secrets DO NOT provide (by default):
- ❌ Encryption at rest in etcd
- ❌ Encryption in transit to pods (unless cluster configured)
- ❌ Automatic rotation
- ❌ Audit logging of access
- ❌ Secrets versioning
- ❌ Secret leakage prevention

## Encryption at Rest

### Enable etcd Encryption

**Why:** Protects secrets if etcd data is compromised or backed up.

#### 1. Create Encryption Configuration

```yaml
# /etc/kubernetes/enc/enc.yaml
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
  - resources:
      - secrets
      - configmaps  # Optional: encrypt ConfigMaps too
    providers:
      # Use AES-CBC with PKCS#7 padding
      - aescbc:
          keys:
            - name: key1
              secret: <base64-encoded-32-byte-random-key>
      # Fallback to unencrypted for old data
      - identity: {}
```

Generate encryption key:
```bash
head -c 32 /dev/urandom | base64
```

#### 2. Configure API Server

Add to kube-apiserver flags:
```bash
--encryption-provider-config=/etc/kubernetes/enc/enc.yaml
```

#### 3. Encrypt Existing Secrets

```bash
# Force re-encryption of all secrets
kubectl get secrets --all-namespaces -o json | \
  kubectl replace -f -
```

### Verify Encryption

```bash
# Check if secrets are encrypted in etcd
ETCDCTL_API=3 etcdctl \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key \
  get /registry/secrets/default/test-secret | hexdump -C

# Encrypted secrets start with: k8s:enc:aescbc:v1:
```

### Key Rotation Strategy

```bash
# 1. Add new key to encryption config (key2)
# 2. Restart API server
# 3. Re-encrypt all secrets
kubectl get secrets --all-namespaces -o json | kubectl replace -f -

# 4. Remove old key (key1) from config
# 5. Restart API server again
```

## Access Control (RBAC)

### Principle of Least Privilege

**Default:** Don't grant broad secret access

```yaml
# ❌ BAD: Overly permissive
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: bad-role
rules:
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["*"]  # Too broad!
```

```yaml
# ✅ GOOD: Specific access
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: app-secrets-reader
  namespace: production
rules:
- apiGroups: [""]
  resources: ["secrets"]
  resourceNames:
    - app-secrets  # Specific secret only
  verbs: ["get"]  # Read-only
```

### Separate Secrets by Sensitivity

```yaml
# High-sensitivity secrets (database, API keys)
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: high-security-secrets
  namespace: production
rules:
- apiGroups: [""]
  resources: ["secrets"]
  resourceNames:
    - database-credentials
    - api-keys
  verbs: ["get"]

---
# Low-sensitivity config (feature flags, non-secrets)
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: config-reader
  namespace: production
rules:
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["get", "list"]
```

### Service Account Isolation

```yaml
# Dedicated service account per application
apiVersion: v1
kind: ServiceAccount
metadata:
  name: payment-service
  namespace: production

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: payment-service-secrets
  namespace: production
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: payment-secrets-reader
subjects:
- kind: ServiceAccount
  name: payment-service
  namespace: production

---
# Deployment uses dedicated service account
apiVersion: apps/v1
kind: Deployment
metadata:
  name: payment-service
spec:
  template:
    spec:
      serviceAccountName: payment-service  # Specific SA
      automountServiceAccountToken: false  # Disable if not using K8s API
```

### Deny Access to Cluster Admins (Optional)

For highly sensitive environments:

```yaml
# Separate namespace for highly sensitive secrets
apiVersion: v1
kind: Namespace
metadata:
  name: vault-secrets

---
# Deny even cluster-admin from viewing secrets
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: secret-admin
  namespace: vault-secrets
rules:
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["*"]

---
# Only specific service accounts can access
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: external-secrets-operator
  namespace: vault-secrets
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: secret-admin
subjects:
- kind: ServiceAccount
  name: external-secrets
  namespace: vault-secrets
```

## Secret Management Patterns

### 1. Volume Mounts vs Environment Variables

**Prefer volume mounts over environment variables:**

```yaml
# ✅ BETTER: Volume mount
spec:
  containers:
  - name: app
    volumeMounts:
    - name: secrets
      mountPath: /etc/secrets
      readOnly: true  # Always read-only!
  volumes:
  - name: secrets
    secret:
      secretName: app-secrets
      defaultMode: 0400  # Restrictive permissions
```

**Why volume mounts are better:**
- Permissions can be controlled (e.g., 0400)
- Easier to rotate (volumes update, env vars don't)
- Less likely to leak in logs/errors
- Not visible in `kubectl describe pod`
- Not inherited by child processes

**When to use environment variables:**
- Application requires env vars (can't change code)
- Temporary/development environments
- Non-sensitive configuration

### 2. Immutable Secrets

```yaml
# Prevents accidental modification
apiVersion: v1
kind: Secret
metadata:
  name: prod-secrets
type: Opaque
immutable: true  # Kubernetes 1.21+
data:
  key: dmFsdWU=
```

**Benefits:**
- Prevents accidental overwrites
- Improves cluster performance (no watches needed)
- Forces versioned secret pattern
- Clear audit trail

### 3. Projected Volumes

Combine multiple secrets and configs:

```yaml
volumes:
- name: combined
  projected:
    sources:
    - secret:
        name: database-creds
        items:
        - key: username
          path: db/username
        - key: password
          path: db/password
    - secret:
        name: api-keys
        items:
        - key: key
          path: api/key
    - configMap:
        name: app-config
        items:
        - key: config.yaml
          path: config/config.yaml
    defaultMode: 0400
```

### 4. Secret Scoping by Namespace

```bash
# Production secrets
kubectl create namespace production
kubectl create secret generic db-prod \
  --from-literal=password=<strong-password> \
  -n production

# Development secrets
kubectl create namespace development
kubectl create secret generic db-dev \
  --from-literal=password=dev-password \
  -n development
```

## External Secret Management

### Why External Secret Managers?

Built-in Kubernetes secrets limitations:
- No built-in rotation
- Limited audit capabilities
- No fine-grained access policies
- No secret versioning
- Manual encryption key management

### Solutions Comparison

| Solution | Pros | Cons | Use Case |
|----------|------|------|----------|
| **HashiCorp Vault** | Industry standard, feature-rich, dynamic secrets | Complex setup, requires management | Enterprise, multi-cloud |
| **AWS Secrets Manager** | Managed, auto-rotation, AWS integration | AWS-only, cost | AWS-native applications |
| **Azure Key Vault** | Managed, HSM support, Azure integration | Azure-only | Azure-native applications |
| **Google Secret Manager** | Managed, GCP integration, simple | GCP-only | GCP-native applications |
| **Sealed Secrets** | GitOps-friendly, simple, free | No dynamic secrets, basic features | GitOps workflows |
| **External Secrets Operator** | Multi-backend, unified API | Requires backend setup | Multi-cloud, migration |

### External Secrets Operator Pattern

```yaml
# Install: https://external-secrets.io/

# 1. Configure secret store backend
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
  namespace: production
spec:
  provider:
    vault:
      server: "https://vault.example.com"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "production-role"
          serviceAccountRef:
            name: external-secrets

---
# 2. Define external secret
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: database-credentials
  namespace: production
spec:
  refreshInterval: 15m  # Check every 15 minutes
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: database-secret  # K8s secret to create
    creationPolicy: Owner
    template:
      engineVersion: v2
      data:
        # Template secrets from multiple sources
        dsn: "postgresql://{{ .username }}:{{ .password }}@postgres:5432/mydb"
  data:
  - secretKey: username
    remoteRef:
      key: database/production
      property: username
  - secretKey: password
    remoteRef:
      key: database/production
      property: password

---
# 3. Application uses the created secret normally
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  template:
    spec:
      containers:
      - name: app
        envFrom:
        - secretRef:
            name: database-secret  # Created by External Secrets Operator
```

### HashiCorp Vault with Kubernetes

```yaml
# Vault Agent Injector Pattern
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  template:
    metadata:
      annotations:
        vault.hashicorp.com/agent-inject: "true"
        vault.hashicorp.com/role: "myapp"
        vault.hashicorp.com/agent-inject-secret-database: "database/creds/readonly"
        vault.hashicorp.com/agent-inject-template-database: |
          {{- with secret "database/creds/readonly" -}}
          export DB_USERNAME="{{ .Data.username }}"
          export DB_PASSWORD="{{ .Data.password }}"
          {{- end }}
    spec:
      serviceAccountName: myapp
      containers:
      - name: app
        image: myapp:latest
        command: ["/bin/sh", "-c"]
        args:
        - source /vault/secrets/database && ./app
```

**Benefits:**
- Dynamic secrets (automatically rotated)
- Vault agent handles renewal
- No secrets in Kubernetes at all (option)
- Comprehensive audit logs

## ConfigMap Security

### ConfigMaps Are Not Secrets

**Important:** ConfigMaps are NOT encrypted and should never contain sensitive data.

```yaml
# ❌ NEVER DO THIS
apiVersion: v1
kind: ConfigMap
metadata:
  name: bad-config
data:
  database-password: "supersecret"  # Visible to anyone!
  api-key: "sk-1234567890"  # Not encrypted!
```

### RBAC for ConfigMaps

```yaml
# Restrict ConfigMap access
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: config-reader
rules:
- apiGroups: [""]
  resources: ["configmaps"]
  resourceNames:
    - app-config  # Specific ConfigMap
  verbs: ["get"]
```

### ConfigMap Best Practices

1. **Use ConfigMaps for:**
   - Application configuration (non-sensitive)
   - Feature flags
   - Environment-specific settings
   - Configuration files (nginx.conf, etc.)

2. **Never put in ConfigMaps:**
   - Passwords
   - API keys
   - Tokens
   - Certificates/private keys
   - PII (personally identifiable information)

3. **Audit ConfigMaps:**
   ```bash
   # Search for potential secrets in ConfigMaps
   kubectl get configmaps --all-namespaces -o yaml | \
     grep -E "(password|secret|token|key)" -i
   ```

## Auditing and Monitoring

### Enable Audit Logging

```yaml
# /etc/kubernetes/audit-policy.yaml
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
# Log secret access at Metadata level
- level: Metadata
  resources:
  - group: ""
    resources: ["secrets"]
  verbs: ["get", "list", "watch"]

# Log secret modifications at RequestResponse level
- level: RequestResponse
  resources:
  - group: ""
    resources: ["secrets"]
  verbs: ["create", "update", "patch", "delete"]

# Log ConfigMap modifications
- level: Metadata
  resources:
  - group: ""
    resources: ["configmaps"]
  verbs: ["create", "update", "patch", "delete"]
```

Configure API server:
```bash
--audit-policy-file=/etc/kubernetes/audit-policy.yaml
--audit-log-path=/var/log/kubernetes/audit.log
--audit-log-maxage=30
--audit-log-maxbackup=10
--audit-log-maxsize=100
```

### Monitor Secret Access

```bash
# Watch secret access in real-time
kubectl get events --watch --all-namespaces | grep -i secret

# Check recent secret-related events
kubectl get events --all-namespaces \
  --field-selector involvedObject.kind=Secret \
  --sort-by='.lastTimestamp'
```

### Detect Suspicious Activity

Common red flags:
- Secret accessed by unexpected service account
- Secret accessed from unusual namespace
- Bulk secret reads (potential data exfiltration)
- Secret created without proper labels/annotations
- Secret accessed outside business hours

### Alert on Secret Changes

```yaml
# Example: Prometheus AlertRule
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: secret-alerts
spec:
  groups:
  - name: secrets
    interval: 1m
    rules:
    - alert: SecretModified
      expr: |
        kube_secret_metadata_resource_version
        != on(namespace,secret) kube_secret_metadata_resource_version offset 5m
      for: 1m
      annotations:
        summary: "Secret {{ $labels.namespace }}/{{ $labels.secret }} was modified"
```

## Secret Rotation

### Rotation Frequency

| Secret Type | Rotation Frequency | Priority |
|-------------|-------------------|----------|
| Database passwords | 30-90 days | High |
| API keys (third-party) | 90-180 days | High |
| JWT signing keys | 180-365 days | Medium |
| TLS certificates | Before expiry | High |
| Service account tokens | 365 days | Medium |
| Encryption keys | 1-2 years | Critical |

### Rotation Patterns

See `scripts/rotate-secrets.sh` for detailed examples.

**Key principles:**
1. **Zero-downtime:** Use dual secrets during transition
2. **Automated:** Use External Secrets Operator or Vault
3. **Tested:** Rotate in dev/staging first
4. **Monitored:** Alert on rotation failures
5. **Audited:** Log all rotation events

### Automated Rotation with Kubernetes CronJobs

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: rotate-database-password
spec:
  schedule: "0 2 1 * *"  # 2 AM on 1st of every month
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: secret-rotator
          containers:
          - name: rotator
            image: custom/secret-rotator:latest
            env:
            - name: SECRET_NAME
              value: database-credentials
            - name: NAMESPACE
              value: production
          restartPolicy: OnFailure
```

## Container Security

### Security Context

```yaml
spec:
  securityContext:
    # Pod-level security
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault

  containers:
  - name: app
    securityContext:
      # Container-level security
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      runAsNonRoot: true
      runAsUser: 1000
      capabilities:
        drop:
        - ALL

    volumeMounts:
    - name: secrets
      mountPath: /etc/secrets
      readOnly: true  # Critical: always read-only for secrets

    # Writable scratch space if needed
    - name: tmp
      mountPath: /tmp

  volumes:
  - name: secrets
    secret:
      secretName: app-secrets
      defaultMode: 0400  # Only owner can read
  - name: tmp
    emptyDir: {}
```

### Network Policies

```yaml
# Restrict secret access to specific pods only
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: restrict-api-access
  namespace: production
spec:
  podSelector:
    matchLabels:
      role: backend
  policyTypes:
  - Egress
  egress:
  # Allow only necessary connections
  - to:
    - podSelector:
        matchLabels:
          app: database
    ports:
    - protocol: TCP
      port: 5432
```

## CI/CD Security

### Never Commit Secrets to Git

```bash
# .gitignore
*.env
*-secrets.yaml
secrets/
.secrets
```

### Pre-commit Hooks

```bash
# .git/hooks/pre-commit
#!/bin/bash

# Check for potential secrets
if git diff --cached --name-only | xargs grep -E '(password|secret|key|token).*=.*["\047]' ; then
    echo "Error: Potential secret detected in commit"
    echo "Please use Kubernetes Secrets or external secret management"
    exit 1
fi
```

### Use git-secrets

```bash
# Install git-secrets
brew install git-secrets  # macOS
# or
sudo apt-get install git-secrets  # Ubuntu

# Set up git-secrets
git secrets --install
git secrets --register-aws
git secrets --add 'password\s*=\s*["\047][^"\047]+'
git secrets --add 'api[_-]key\s*=\s*["\047][^"\047]+'
```

### Sealed Secrets for GitOps

```yaml
# Encrypted secret (safe to commit)
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: my-secret
  namespace: production
spec:
  encryptedData:
    password: AgBy3i4OJSWK+PiTySYZZA9rO43cGDEq...

# Controller decrypts and creates real secret
```

### CI/CD Pipeline Security

```yaml
# GitHub Actions example
name: Deploy
on: push

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2

    # ✅ GOOD: Use encrypted secrets
    - name: Deploy to Kubernetes
      env:
        KUBE_CONFIG: ${{ secrets.KUBE_CONFIG }}
        DB_PASSWORD: ${{ secrets.DB_PASSWORD }}
      run: |
        # Never echo secrets!
        echo "$KUBE_CONFIG" | base64 -d > /tmp/kubeconfig
        export KUBECONFIG=/tmp/kubeconfig

        # Create secret from CI secret
        kubectl create secret generic app-secrets \
          --from-literal=db-password="$DB_PASSWORD" \
          --dry-run=client -o yaml | kubectl apply -f -

    # Clean up
    - name: Cleanup
      if: always()
      run: rm -f /tmp/kubeconfig
```

## Compliance Considerations

### PCI-DSS

For payment card data:
- Encrypt secrets at rest and in transit
- Implement key rotation (quarterly minimum)
- Audit all secret access
- Restrict access (need-to-know basis)
- Use HSM for key storage (if applicable)

### HIPAA

For healthcare data:
- Enable encryption at rest
- Implement comprehensive audit logging
- Use access controls (RBAC)
- Regular access reviews
- Incident response plan for secret exposure

### GDPR

For EU personal data:
- Minimize secret scope (purpose limitation)
- Implement data deletion procedures
- Document secret retention policies
- Audit secret access
- Right to erasure (delete secrets on request)

### SOC 2

For service organizations:
- Document secret management procedures
- Implement change management for secrets
- Regular security assessments
- Audit logging and monitoring
- Incident response procedures

## Security Checklist

### Cluster Configuration
- [ ] Enable encryption at rest for etcd
- [ ] Configure audit logging for secret access
- [ ] Enable RBAC and restrict secret access
- [ ] Use network policies to limit pod communication
- [ ] Enable Pod Security Standards/Admission

### Secret Management
- [ ] Never store sensitive data in ConfigMaps
- [ ] Use volume mounts instead of environment variables
- [ ] Set restrictive file permissions (0400) on secret volumes
- [ ] Use immutable secrets for production
- [ ] Implement secret rotation schedule
- [ ] Use external secret management (Vault, etc.) for production
- [ ] Namespace isolation for different environments

### Application Security
- [ ] Run containers as non-root users
- [ ] Use read-only root filesystems
- [ ] Drop all capabilities
- [ ] Disable privilege escalation
- [ ] Use dedicated service accounts per application
- [ ] Disable automounting service account tokens if not needed

### CI/CD Security
- [ ] Never commit secrets to version control
- [ ] Use pre-commit hooks to detect secrets
- [ ] Use Sealed Secrets or similar for GitOps
- [ ] Encrypt secrets in CI/CD systems
- [ ] Implement secret scanning in pipelines
- [ ] Use short-lived credentials when possible

### Monitoring & Auditing
- [ ] Enable audit logging
- [ ] Monitor secret access patterns
- [ ] Alert on unexpected secret modifications
- [ ] Regular access reviews
- [ ] Incident response plan for secret exposure
- [ ] Log aggregation and analysis

### Development Practices
- [ ] Use separate secrets for each environment
- [ ] Document secret rotation procedures
- [ ] Regular security training for developers
- [ ] Code reviews include secret handling checks
- [ ] Automated security scanning
- [ ] Principle of least privilege

### Compliance
- [ ] Document secret retention policies
- [ ] Implement data deletion procedures
- [ ] Regular compliance audits
- [ ] Third-party security assessments
- [ ] Incident response and disaster recovery plans

## Resources

### Official Documentation
- [Kubernetes Secrets](https://kubernetes.io/docs/concepts/configuration/secret/)
- [Encrypting Secret Data at Rest](https://kubernetes.io/docs/tasks/administer-cluster/encrypt-data/)
- [RBAC Authorization](https://kubernetes.io/docs/reference/access-authn-authz/rbac/)
- [Audit Logging](https://kubernetes.io/docs/tasks/debug/debug-cluster/audit/)

### Tools
- [External Secrets Operator](https://external-secrets.io/)
- [Sealed Secrets](https://github.com/bitnami-labs/sealed-secrets)
- [HashiCorp Vault](https://www.vaultproject.io/)
- [git-secrets](https://github.com/awslabs/git-secrets)
- [kube-bench](https://github.com/aquasecurity/kube-bench) - CIS benchmark testing

### Further Reading
- [CIS Kubernetes Benchmark](https://www.cisecurity.org/benchmark/kubernetes)
- [OWASP Kubernetes Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Kubernetes_Security_Cheat_Sheet.html)
- [NSA/CISA Kubernetes Hardening Guide](https://media.defense.gov/2022/Aug/29/2003066362/-1/-1/0/CTR_KUBERNETES_HARDENING_GUIDANCE_1.2_20220829.PDF)

---

**Remember:** Security is a continuous process, not a one-time setup. Regular reviews, updates, and training are essential for maintaining a secure Kubernetes environment.
