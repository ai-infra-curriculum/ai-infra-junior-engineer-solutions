# Exercise 05: ConfigMaps & Secrets

Managing configuration and sensitive data securely in Kubernetes applications.

## Overview

This exercise demonstrates how to manage application configuration and secrets in Kubernetes. You'll learn:

- ConfigMaps for non-sensitive configuration data
- Secrets for sensitive information (passwords, keys, tokens)
- Different ways to consume configuration (environment variables, volume mounts)
- Security best practices for secret management
- Secret rotation strategies
- Common configuration patterns

## Prerequisites

- Completed Exercise 04 (StatefulSets & Storage) or equivalent Kubernetes knowledge
- kubectl configured and connected to a Kubernetes cluster
- Basic understanding of base64 encoding
- Familiarity with environment variables and file systems

## Learning Objectives

By the end of this exercise, you will be able to:

1. Create and manage ConfigMaps and Secrets
2. Use different methods to inject configuration into pods
3. Implement security best practices for secrets
4. Understand the difference between ConfigMaps and Secrets
5. Rotate secrets with zero downtime
6. Use external secret management systems
7. Apply configuration patterns for production applications

## Directory Structure

```
exercise-05-configmaps-secrets/
├── manifests/
│   ├── 01-namespace.yaml                  # Namespace definition
│   ├── 02-configmap-examples.yaml         # ConfigMap examples (5 types)
│   ├── 03-secret-examples.yaml            # Secret examples (10 types)
│   ├── 04-using-configmaps.yaml           # Pods using ConfigMaps (11 examples)
│   └── 05-using-secrets.yaml              # Pods using Secrets (13 examples)
├── examples/
│   └── config-patterns.yaml               # Advanced configuration patterns
├── scripts/
│   ├── deploy-all.sh                      # Automated deployment
│   ├── test-configs.sh                    # Test suite (15 tests)
│   ├── rotate-secrets.sh                  # Secret rotation demonstrations
│   └── cleanup.sh                         # Cleanup script
├── docs/
│   └── SECURITY_BEST_PRACTICES.md         # Comprehensive security guide
├── README.md                               # This file
└── STEP_BY_STEP.md                        # Detailed walkthrough
```

## Quick Start

### 1. Deploy Everything

```bash
# Deploy all manifests and examples
./scripts/deploy-all.sh
```

### 2. Verify Deployment

```bash
# Check namespace
kubectl get all -n config-demo

# Check ConfigMaps
kubectl get configmaps -n config-demo

# Check Secrets
kubectl get secrets -n config-demo
```

### 3. Run Tests

```bash
# Run comprehensive test suite
./scripts/test-configs.sh
```

### 4. Explore Examples

```bash
# View ConfigMap usage
kubectl logs -n config-demo pod-configmap-env
kubectl logs -n config-demo pod-configmap-volume

# View Secret usage
kubectl logs -n config-demo pod-secret-env
kubectl exec -n config-demo pod-secret-volume -- cat /etc/secrets/database-username
```

### 5. Cleanup

```bash
# Interactive cleanup
./scripts/cleanup.sh

# Or force cleanup
./scripts/cleanup.sh -f
```

## Key Concepts

### ConfigMaps vs Secrets

| Feature | ConfigMap | Secret |
|---------|-----------|---------|
| **Purpose** | Non-sensitive configuration | Sensitive data (passwords, keys) |
| **Storage** | Plain text | Base64 encoded (not encrypted!) |
| **Size Limit** | 1 MB | 1 MB |
| **Encryption** | Not encrypted | Encrypted at rest (if configured) |
| **Use Cases** | App settings, feature flags, config files | Passwords, API keys, certificates |
| **RBAC** | Standard RBAC | Stricter RBAC recommended |
| **Audit** | Standard audit logging | Enhanced audit logging recommended |

### ConfigMap Types

1. **Literal Values** - Simple key-value pairs
   ```yaml
   data:
     APP_NAME: "My Application"
     LOG_LEVEL: "info"
   ```

2. **File Content** - Configuration files (nginx.conf, application.properties)
   ```yaml
   data:
     nginx.conf: |
       server {
         listen 80;
       }
   ```

3. **Binary Data** - Non-UTF8 data
   ```yaml
   binaryData:
     logo.png: iVBORw0KGgoAAAANSU...
   ```

4. **Environment-Specific** - Different configs per environment
   ```yaml
   data:
     ENV: "production"
     LOG_LEVEL: "warn"
   ```

5. **Scripts** - Shell scripts for initialization
   ```yaml
   data:
     init.sh: |
       #!/bin/bash
       echo "Initializing..."
   ```

### Secret Types

1. **Opaque** (generic) - Most common type
   ```yaml
   type: Opaque
   data:
     password: c3VwZXJzZWNyZXQ=  # base64 encoded
   ```

2. **kubernetes.io/dockerconfigjson** - Docker registry credentials
   ```yaml
   type: kubernetes.io/dockerconfigjson
   data:
     .dockerconfigjson: eyJhdXRocyI6...
   ```

3. **kubernetes.io/tls** - TLS certificates and keys
   ```yaml
   type: kubernetes.io/tls
   data:
     tls.crt: LS0tLS1CRUdJTi...
     tls.key: LS0tLS1CRUdJTi...
   ```

4. **kubernetes.io/ssh-auth** - SSH private keys
5. **kubernetes.io/basic-auth** - Basic HTTP authentication
6. **kubernetes.io/service-account-token** - Service account tokens

### Ways to Use ConfigMaps and Secrets

#### 1. Environment Variables (Individual Keys)

```yaml
env:
- name: APP_NAME
  valueFrom:
    configMapKeyRef:
      name: app-config
      key: APP_NAME
- name: DB_PASSWORD
  valueFrom:
    secretKeyRef:
      name: db-secrets
      key: password
```

#### 2. Environment Variables (All Keys)

```yaml
envFrom:
- configMapRef:
    name: app-config
- secretRef:
    name: db-secrets
```

#### 3. Volume Mounts

```yaml
volumeMounts:
- name: config
  mountPath: /etc/config
  readOnly: true

volumes:
- name: config
  configMap:
    name: app-config
- name: secrets
  secret:
    secretName: db-secrets
    defaultMode: 0400  # Restrictive permissions
```

#### 4. Selective Key Mounting

```yaml
volumes:
- name: config
  configMap:
    name: app-config
    items:
    - key: nginx.conf
      path: nginx.conf
```

#### 5. Projected Volumes

```yaml
volumes:
- name: combined
  projected:
    sources:
    - configMap:
        name: app-config
    - secret:
        name: app-secrets
```

## Examples Included

### ConfigMap Examples (11 total)

1. **pod-configmap-env** - Individual keys as environment variables
2. **pod-configmap-envfrom** - All keys imported as environment variables
3. **pod-configmap-envfrom-prefix** - All keys with prefix
4. **pod-configmap-volume** - Mounted as files
5. **pod-configmap-selective** - Only specific keys mounted
6. **pod-configmap-permissions** - Custom file permissions
7. **pod-configmap-subpath** - Mount without overwriting directory
8. **pod-multiple-configmaps** - Multiple ConfigMaps in one pod
9. **pod-configmap-optional** - Optional ConfigMap (pod starts if missing)
10. **demo-app** - Deployment using ConfigMaps
11. **pod-configmap-hot-reload** - Demonstrates config updates

### Secret Examples (13 total)

1. **pod-secret-env** - Individual keys as environment variables
2. **pod-secret-envfrom** - All keys imported as environment variables
3. **pod-secret-volume** - Mounted as files
4. **pod-secret-selective** - Only specific keys mounted
5. **pod-secret-permissions** - Restrictive file permissions (0400)
6. **pod-private-image** - Using imagePullSecrets
7. **pod-tls-secret** - TLS certificate mounting
8. **pod-ssh-secret** - SSH key mounting
9. **pod-basic-auth** - Basic authentication credentials
10. **pod-multiple-secrets** - Multiple Secrets in one pod
11. **pod-secret-optional** - Optional Secret (pod starts if missing)
12. **secure-app** - Production deployment using both ConfigMaps and Secrets
13. **pod-projected-volume** - Combined ConfigMaps and Secrets

### Configuration Patterns (7 total)

1. **12-Factor App** - Environment variable configuration
2. **Layered Configuration** - Base + environment overrides
3. **Sidecar Injection** - Init container renders configuration
4. **External Configuration** - Vault/external secret management
5. **Hot-Reload** - Watcher detects config changes
6. **Multi-Tenant** - Different configs per tenant
7. **Blue-Green** - Configuration for blue-green deployments

## Security Best Practices

### Critical Points

1. **Secrets are base64 encoded, NOT encrypted by default**
   ```bash
   # Anyone can decode them
   kubectl get secret app-secrets -n config-demo \
     -o jsonpath='{.data.password}' | base64 -d
   ```

2. **Enable encryption at rest**
   ```yaml
   # API server configuration
   --encryption-provider-config=/etc/kubernetes/enc/enc.yaml
   ```

3. **Use RBAC to restrict access**
   ```yaml
   # Principle of least privilege
   rules:
   - apiGroups: [""]
     resources: ["secrets"]
     resourceNames: ["app-secrets"]  # Specific secret only
     verbs: ["get"]  # Read-only
   ```

4. **Prefer volume mounts over environment variables**
   - Volumes update automatically (env vars don't)
   - Permissions can be controlled
   - Less likely to leak in logs
   - Not visible in `kubectl describe pod`

5. **Use external secret management in production**
   - HashiCorp Vault
   - AWS Secrets Manager
   - Azure Key Vault
   - Google Secret Manager
   - Sealed Secrets
   - External Secrets Operator

6. **Never commit secrets to version control**
   ```bash
   # Use pre-commit hooks
   git secrets --install
   git secrets --register-aws
   ```

7. **Implement secret rotation**
   ```bash
   # Run rotation demonstrations
   ./scripts/rotate-secrets.sh
   ```

8. **Audit secret access**
   ```yaml
   # Enable audit logging
   --audit-policy-file=/etc/kubernetes/audit-policy.yaml
   ```

### Security Checklist

- [ ] Enable encryption at rest for etcd
- [ ] Configure RBAC with least privilege
- [ ] Use volume mounts instead of env vars for secrets
- [ ] Set restrictive permissions (0400) on secret volumes
- [ ] Implement secret rotation schedule
- [ ] Use external secret management for production
- [ ] Enable audit logging for secret access
- [ ] Never commit secrets to version control
- [ ] Run containers as non-root users
- [ ] Use read-only root filesystems
- [ ] Implement network policies

See `docs/SECURITY_BEST_PRACTICES.md` for comprehensive guidance.

## Common Commands

### Creating ConfigMaps and Secrets

```bash
# ConfigMap from literal values
kubectl create configmap app-config \
  --from-literal=APP_NAME="My App" \
  --from-literal=LOG_LEVEL="info" \
  -n config-demo

# ConfigMap from file
kubectl create configmap nginx-config \
  --from-file=nginx.conf \
  -n config-demo

# ConfigMap from directory
kubectl create configmap app-configs \
  --from-file=./configs/ \
  -n config-demo

# Secret from literal values
kubectl create secret generic db-secrets \
  --from-literal=username=admin \
  --from-literal=password=supersecret \
  -n config-demo

# Secret from files
kubectl create secret generic tls-secrets \
  --from-file=tls.crt \
  --from-file=tls.key \
  -n config-demo

# TLS secret
kubectl create secret tls tls-secret \
  --cert=tls.crt \
  --key=tls.key \
  -n config-demo

# Docker registry secret
kubectl create secret docker-registry regcred \
  --docker-server=https://index.docker.io/v1/ \
  --docker-username=myuser \
  --docker-password=mypassword \
  --docker-email=myemail@example.com \
  -n config-demo
```

### Viewing ConfigMaps and Secrets

```bash
# List ConfigMaps
kubectl get configmaps -n config-demo

# Describe ConfigMap
kubectl describe configmap app-config -n config-demo

# View ConfigMap YAML
kubectl get configmap app-config -n config-demo -o yaml

# Get specific key from ConfigMap
kubectl get configmap app-config -n config-demo \
  -o jsonpath='{.data.APP_NAME}'

# List Secrets
kubectl get secrets -n config-demo

# Describe Secret (data is hidden)
kubectl describe secret app-secrets -n config-demo

# View Secret YAML
kubectl get secret app-secrets -n config-demo -o yaml

# Decode secret value
kubectl get secret app-secrets -n config-demo \
  -o jsonpath='{.data.password}' | base64 -d
```

### Updating ConfigMaps and Secrets

```bash
# Edit ConfigMap
kubectl edit configmap app-config -n config-demo

# Patch ConfigMap
kubectl patch configmap app-config -n config-demo \
  --type='json' -p='[{"op": "replace", "path": "/data/LOG_LEVEL", "value": "debug"}]'

# Replace ConfigMap from file
kubectl create configmap app-config \
  --from-file=config.yaml \
  --dry-run=client -o yaml | kubectl replace -f -

# Update Secret
kubectl create secret generic app-secrets \
  --from-literal=password=newsecret \
  --dry-run=client -o yaml | kubectl apply -f -

# Trigger rollout after config change
kubectl rollout restart deployment app -n config-demo
```

### Debugging Configuration

```bash
# Check environment variables in pod
kubectl exec -n config-demo pod-name -- env | sort

# Check mounted files
kubectl exec -n config-demo pod-name -- ls -la /etc/config/
kubectl exec -n config-demo pod-name -- cat /etc/config/app.properties

# Check file permissions
kubectl exec -n config-demo pod-name -- stat /etc/secrets/password

# View pod with config/secrets
kubectl get pod pod-name -n config-demo -o yaml

# Check events
kubectl get events -n config-demo --sort-by='.lastTimestamp'

# Check which pods use a ConfigMap or Secret
kubectl get pods -n config-demo -o json | \
  jq '.items[] | select(
    (.spec.volumes[]?.configMap.name == "app-config") or
    (.spec.containers[].envFrom[]?.configMapRef.name == "app-config")
  ) | .metadata.name'
```

## Testing

### Run All Tests

```bash
./scripts/test-configs.sh
```

The test suite includes 15 tests:
1. ConfigMap existence checks
2. Secret existence checks
3. ConfigMap data integrity
4. Secret data integrity
5. ConfigMap as environment variables
6. Secret as environment variables
7. ConfigMap as volume mounts
8. Secret as volume mounts
9. Secret file permissions
10. Deployment using ConfigMaps
11. Deployment using Secrets
12. envFrom ConfigMap
13. envFrom Secret
14. Selective key mounting
15. ConfigMap update detection

### Manual Testing

```bash
# Test ConfigMap as env vars
kubectl logs -n config-demo pod-configmap-env

# Test Secret as env vars
kubectl logs -n config-demo pod-secret-env

# Test ConfigMap as volume
kubectl exec -n config-demo pod-configmap-volume -- cat /etc/config/nginx.conf

# Test Secret as volume
kubectl exec -n config-demo pod-secret-volume -- cat /etc/secrets/database-username

# Test hot-reload
kubectl logs -n config-demo pod-configmap-hot-reload -f
# In another terminal:
kubectl edit configmap app-config-files -n config-demo
# Watch logs to see config hash change (~60s delay)
```

## Secret Rotation

```bash
# Run interactive rotation demonstrations
./scripts/rotate-secrets.sh
```

The script demonstrates:
1. Zero-downtime rotation (dual secrets)
2. Rolling restart rotation
3. Immutable secret rotation
4. External Secrets Operator pattern
5. Blue-green deployment rotation

## Troubleshooting

### Pod Won't Start

```bash
# Check events
kubectl describe pod pod-name -n config-demo

# Common issues:
# - ConfigMap or Secret doesn't exist
# - Wrong namespace
# - RBAC permissions issue
# - Volume mount conflicts
```

### ConfigMap/Secret Not Found

```bash
# Verify it exists
kubectl get configmap app-config -n config-demo
kubectl get secret app-secrets -n config-demo

# Check namespace
kubectl get configmaps --all-namespaces | grep app-config

# Recreate if needed
kubectl apply -f manifests/02-configmap-examples.yaml
```

### Environment Variables Not Set

```bash
# Check if configMapKeyRef or secretKeyRef is correct
kubectl get pod pod-name -n config-demo -o yaml | grep -A 5 "envFrom\|env:"

# Verify the key exists in ConfigMap/Secret
kubectl get configmap app-config -n config-demo -o yaml
kubectl get secret app-secrets -n config-demo -o yaml | grep "^data:"
```

### Volume Mount Issues

```bash
# Check volume mounts
kubectl describe pod pod-name -n config-demo | grep -A 10 "Mounts:"

# Check if files are present
kubectl exec -n config-demo pod-name -- ls -la /etc/config/

# Check permissions
kubectl exec -n config-demo pod-name -- stat /etc/config/file

# Check volume definition
kubectl get pod pod-name -n config-demo -o yaml | grep -A 10 "volumes:"
```

### ConfigMap Changes Not Reflected

**Important:** Environment variables do NOT update when ConfigMap/Secret changes.

```bash
# For volume mounts: Wait for kubelet sync (~60 seconds)
kubectl exec -n config-demo pod-name -- cat /etc/config/file

# For env vars: Must restart pod
kubectl delete pod pod-name -n config-demo
# or
kubectl rollout restart deployment deployment-name -n config-demo
```

### Permission Denied

```bash
# Check file permissions on mounted secrets
kubectl exec -n config-demo pod-name -- ls -la /etc/secrets/

# Check container security context
kubectl get pod pod-name -n config-demo -o yaml | grep -A 5 "securityContext:"

# Check if running as non-root
kubectl exec -n config-demo pod-name -- id
```

## Best Practices

1. **Separate Concerns**
   - Use ConfigMaps for non-sensitive configuration
   - Use Secrets for sensitive data (passwords, keys, tokens)

2. **Namespace Isolation**
   - Use separate namespaces for different environments
   - Create environment-specific ConfigMaps and Secrets

3. **Immutability**
   - Use immutable ConfigMaps/Secrets in production
   - Version your configuration (app-config-v1, app-config-v2)

4. **Security**
   - Always use `readOnly: true` for secret volume mounts
   - Set restrictive permissions (0400) on secret files
   - Run containers as non-root users
   - Use RBAC to restrict access

5. **Configuration Management**
   - Use labels to organize ConfigMaps and Secrets
   - Document what each ConfigMap/Secret contains
   - Implement configuration validation

6. **Updates**
   - Prefer volume mounts over env vars (volumes update automatically)
   - Trigger rolling restarts after config changes
   - Test configuration changes in non-production first

7. **Monitoring**
   - Monitor for unexpected configuration changes
   - Alert on failed pods due to missing configuration
   - Audit secret access

8. **Documentation**
   - Document required configuration keys
   - Provide example configurations
   - Document secret rotation procedures

## Additional Resources

- **Documentation:** `docs/SECURITY_BEST_PRACTICES.md` - Comprehensive security guide
- **Step-by-Step Guide:** `STEP_BY_STEP.md` - Detailed walkthrough
- **Scripts:** `scripts/` directory for automation and testing

### Official Kubernetes Documentation

- [ConfigMaps](https://kubernetes.io/docs/concepts/configuration/configmap/)
- [Secrets](https://kubernetes.io/docs/concepts/configuration/secret/)
- [Encrypting Secret Data at Rest](https://kubernetes.io/docs/tasks/administer-cluster/encrypt-data/)
- [Secrets Good Practices](https://kubernetes.io/docs/concepts/security/secrets-good-practices/)

### External Secret Management

- [External Secrets Operator](https://external-secrets.io/)
- [HashiCorp Vault](https://www.vaultproject.io/)
- [Sealed Secrets](https://github.com/bitnami-labs/sealed-secrets)
- [AWS Secrets Manager](https://aws.amazon.com/secrets-manager/)
- [Azure Key Vault](https://azure.microsoft.com/en-us/products/key-vault/)
- [Google Secret Manager](https://cloud.google.com/secret-manager)

## Next Steps

After completing this exercise, you should:

1. Review the security best practices document
2. Practice creating ConfigMaps and Secrets imperatively
3. Implement a secret rotation strategy
4. Explore external secret management solutions
5. Move on to Exercise 06: Ingress & Load Balancing

## License

This exercise is part of the AI Infrastructure Junior Engineer curriculum.

## Support

For questions or issues:
1. Check `STEP_BY_STEP.md` for detailed guidance
2. Review `docs/SECURITY_BEST_PRACTICES.md` for security questions
3. Check official Kubernetes documentation
4. Review logs and events for debugging

---

**Remember:** ConfigMaps and Secrets are fundamental to Kubernetes application configuration. Master these concepts to build secure, maintainable applications.
