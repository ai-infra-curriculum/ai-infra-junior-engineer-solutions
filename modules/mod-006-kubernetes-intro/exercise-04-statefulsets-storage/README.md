# Exercise 04: StatefulSets and Persistent Storage

## Overview

This exercise teaches you how to deploy and manage stateful applications in Kubernetes using StatefulSets and persistent storage. You'll learn the differences between StatefulSets and Deployments, understand various volume types, and work with real-world stateful applications like PostgreSQL and Redis.

## Learning Objectives

By completing this exercise, you will:

1. Understand the difference between StatefulSets and Deployments
2. Deploy stateful applications with persistent storage
3. Work with PersistentVolumes (PV) and PersistentVolumeClaims (PVC)
4. Configure and use different volume types
5. Understand StatefulSet pod management and scaling
6. Test data persistence across pod restarts
7. Use headless services for stable network identities
8. Manage storage classes and dynamic provisioning

## Prerequisites

- Completed Exercise 01 (First Deployment)
- Completed Exercise 02 (Helm Chart)
- Completed Exercise 03 (Debugging)
- Kubernetes cluster with storage support
- kubectl configured
- Understanding of:
  - Kubernetes Pods and Services
  - Basic database concepts
  - Linux file systems

## Key Concepts

### StatefulSets vs Deployments

| Feature | StatefulSet | Deployment |
|---------|------------|------------|
| **Pod Identity** | Stable, unique (pod-0, pod-1) | Random (hash-based) |
| **Network Identity** | Stable DNS hostname | Dynamic IP only |
| **Storage** | Persistent per pod | Shared or ephemeral |
| **Ordering** | Sequential (0→1→2) | Parallel |
| **Scaling** | Ordered, graceful | Parallel |
| **Use Cases** | Databases, queues, distributed systems | Stateless apps, APIs, web servers |

### Volume Types

1. **emptyDir**: Temporary storage, pod lifetime only
2. **hostPath**: Mount from host node (dev/testing only)
3. **PersistentVolume (PV)**: Cluster-level storage resource
4. **PersistentVolumeClaim (PVC)**: Request for storage
5. **ConfigMap**: Configuration as volumes
6. **Secret**: Sensitive data as volumes
7. **Projected**: Combine multiple sources

## Directory Structure

```
exercise-04-statefulsets-storage/
├── README.md                                   # This file
├── STEP_BY_STEP.md                            # Detailed walkthrough
├── manifests/                                  # Kubernetes manifests
│   ├── 01-namespace.yaml
│   ├── 02-storageclass.yaml
│   ├── 03-postgresql-statefulset.yaml
│   ├── 04-volume-types.yaml
│   └── 05-redis-statefulset.yaml
├── examples/
│   └── statefulset-vs-deployment.yaml         # Comparison example
├── scripts/
│   ├── deploy-all.sh                          # Deploy everything
│   ├── test-persistence.sh                    # Test data persistence
│   └── cleanup.sh                             # Clean up resources
└── docs/
    └── VOLUME_TYPES.md                        # Volume types reference
```

## Quick Start

### 1. Deploy All Examples

```bash
cd scripts
./deploy-all.sh
```

This deploys:
- Namespace: `statefulset-demo`
- PostgreSQL StatefulSet (3 replicas)
- Redis StatefulSet (3 replicas)
- Volume type examples
- StatefulSet vs Deployment comparison

### 2. Check Deployment

```bash
# Check all resources
kubectl get all -n statefulset-demo

# Check StatefulSets
kubectl get statefulsets -n statefulset-demo

# Check persistent storage
kubectl get pvc -n statefulset-demo
```

### 3. Test Data Persistence

```bash
./test-persistence.sh
```

### 4. Explore Examples

```bash
# Connect to PostgreSQL
kubectl exec -it postgres-0 -n statefulset-demo -- psql -U myuser -d mydatabase

# Connect to Redis
kubectl exec -it redis-0 -n statefulset-demo -- redis-cli

# Check stable DNS names
kubectl run -it dns-test --rm --image=busybox -n statefulset-demo -- sh
nslookup postgres-0.postgres-headless.statefulset-demo.svc.cluster.local
nslookup redis-0.redis-headless.statefulset-demo.svc.cluster.local
```

### 5. Clean Up

```bash
./cleanup.sh
```

---

## Detailed Examples

### Example 1: PostgreSQL StatefulSet

**File**: `manifests/03-postgresql-statefulset.yaml`

This example demonstrates a production-ready PostgreSQL deployment with:
- 3 replicas for high availability
- Persistent storage per pod (1Gi each)
- Headless service for stable network identity
- ConfigMap-based configuration
- Secret-based password management
- Health probes (liveness and readiness)
- Init containers for permissions
- Resource limits

**Key Features**:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: postgres-headless  # Headless service
  replicas: 3
  podManagementPolicy: OrderedReady  # Pods start in order
  volumeClaimTemplates:  # PVC per pod
  - metadata:
      name: postgres-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 1Gi
```

**Accessing PostgreSQL**:

```bash
# Access specific instance
kubectl exec -it postgres-0 -n statefulset-demo -- psql -U myuser -d mydatabase

# Via stable DNS name
kubectl exec -it postgres-0 -n statefulset-demo -- \
  psql -h postgres-0.postgres-headless.statefulset-demo.svc.cluster.local -U myuser -d mydatabase

# Create test data
kubectl exec -it postgres-0 -n statefulset-demo -- psql -U myuser -d mydatabase -c \
  "CREATE TABLE test (id SERIAL PRIMARY KEY, data TEXT);"

kubectl exec -it postgres-0 -n statefulset-demo -- psql -U myuser -d mydatabase -c \
  "INSERT INTO test (data) VALUES ('persistent data');"

# Verify data
kubectl exec -it postgres-0 -n statefulset-demo -- psql -U myuser -d mydatabase -c \
  "SELECT * FROM test;"
```

---

### Example 2: Redis StatefulSet

**File**: `manifests/05-redis-statefulset.yaml`

Redis deployment with persistence:
- 3 replicas
- AOF (Append Only File) persistence enabled
- ConfigMap-based Redis configuration
- 500Mi storage per pod
- Ordered scaling

**Testing Redis**:

```bash
# Connect to Redis instance
kubectl exec -it redis-0 -n statefulset-demo -- redis-cli

# Set some data
redis> SET user:1 "Alice"
redis> SET user:2 "Bob"
redis> KEYS *

# Save to disk
redis> SAVE

# Check persistence info
redis> INFO persistence
```

---

### Example 3: Volume Types

**File**: `manifests/04-volume-types.yaml`

This file contains 7 different volume type examples:

#### 1. emptyDir (Temporary Storage)

```yaml
volumes:
- name: cache-volume
  emptyDir:
    sizeLimit: 100Mi
```

**Use cases**: Cache, temporary files, shared data between containers

#### 2. hostPath (Host Node Storage)

```yaml
volumes:
- name: host-data
  hostPath:
    path: /tmp/nginx-data
    type: DirectoryOrCreate
```

**Warning**: Only for development! Not portable across nodes.

#### 3. PersistentVolume/PersistentVolumeClaim

```yaml
# PersistentVolume (admin-created)
apiVersion: v1
kind: PersistentVolume
metadata:
  name: manual-pv-001
spec:
  capacity:
    storage: 5Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  hostPath:
    path: /mnt/data/pv-001

---
# PersistentVolumeClaim (user-created)
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: manual-pvc-001
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
```

#### 4. ConfigMap as Volume

```yaml
volumes:
- name: config-volume
  configMap:
    name: app-config-vol
```

**Use cases**: Configuration files, application settings

#### 5. Secret as Volume

```yaml
volumes:
- name: secret-volume
  secret:
    secretName: app-secrets-vol
    defaultMode: 0400  # Read-only
```

**Use cases**: Certificates, keys, passwords

#### 6. Projected Volume

```yaml
volumes:
- name: projected-volume
  projected:
    sources:
    - configMap:
        name: app-config-vol
    - secret:
        name: app-secrets-vol
```

**Use cases**: Combine multiple sources in one directory

---

### Example 4: StatefulSet vs Deployment

**File**: `examples/statefulset-vs-deployment.yaml`

Side-by-side comparison showing:

**Deployment Characteristics**:
- Random pod names: `nginx-deployment-7d8f9c5b4-x7k2m`
- No stable DNS
- Shared or no storage
- Parallel scaling

**StatefulSet Characteristics**:
- Predictable names: `nginx-statefulset-0`, `nginx-statefulset-1`
- Stable DNS: `nginx-statefulset-0.nginx-statefulset-svc`
- Per-pod storage
- Ordered scaling

**Test the Difference**:

```bash
# Deploy both
kubectl apply -f examples/statefulset-vs-deployment.yaml

# Check pod names
kubectl get pods -n statefulset-demo -l type=deployment-example
kubectl get pods -n statefulset-demo -l type=statefulset-example

# Test DNS resolution
kubectl run -it dns-test --rm --image=busybox -n statefulset-demo -- sh
nslookup nginx-statefulset-0.nginx-statefulset-svc  # Works!
nslookup nginx-deployment-xxx-yyy  # Fails

# Check storage
kubectl get pvc -n statefulset-demo
# StatefulSet has PVCs, Deployment doesn't

# Test persistence
kubectl exec nginx-statefulset-0 -n statefulset-demo -- sh -c 'echo "test" > /usr/share/nginx/html/data.txt'
kubectl delete pod nginx-statefulset-0 -n statefulset-demo
# Wait for recreation
kubectl exec nginx-statefulset-0 -n statefulset-demo -- cat /usr/share/nginx/html/data.txt
# Data persists!
```

---

## StatefulSet Management

### Scaling

```bash
# Scale up (creates pods 3, 4 in order)
kubectl scale statefulset redis --replicas=5 -n statefulset-demo

# Watch ordered creation
kubectl get pods -n statefulset-demo -l app=redis -w

# Scale down (deletes pods 4, 3 in reverse order)
kubectl scale statefulset redis --replicas=3 -n statefulset-demo
```

### Updating

```bash
# Rolling update (updates in reverse order: 2, 1, 0)
kubectl set image statefulset/redis redis=redis:7.2-alpine -n statefulset-demo

# Watch rollout
kubectl rollout status statefulset/redis -n statefulset-demo

# Rollback if needed
kubectl rollout undo statefulset/redis -n statefulset-demo
```

### Partitioned Updates (Canary)

```bash
# Update only pods >= 2 (canary deployment)
kubectl patch statefulset redis -n statefulset-demo -p '
{
  "spec": {
    "updateStrategy": {
      "rollingUpdate": {
        "partition": 2
      }
    }
  }
}'

# Only redis-2 will be updated
kubectl set image statefulset/redis redis=redis:7.2-alpine -n statefulset-demo

# Verify canary
kubectl get pods -n statefulset-demo -l app=redis -o wide

# Roll out to all pods
kubectl patch statefulset redis -n statefulset-demo -p '
{
  "spec": {
    "updateStrategy": {
      "rollingUpdate": {
        "partition": 0
      }
    }
  }
}'
```

### Pod Management Policies

#### OrderedReady (Default)

```yaml
spec:
  podManagementPolicy: OrderedReady
```

- Pods start in order: 0 → 1 → 2
- Each pod must be Ready before next starts
- Good for databases requiring initialization order

#### Parallel

```yaml
spec:
  podManagementPolicy: Parallel
```

- All pods start simultaneously
- Faster startup
- Good for independent instances

---

## Data Persistence

### Testing Persistence

Use the provided script:

```bash
./scripts/test-persistence.sh
```

**Or manually**:

```bash
# 1. Write data to PostgreSQL
kubectl exec -it postgres-0 -n statefulset-demo -- psql -U myuser -d mydatabase << EOF
CREATE TABLE test (id SERIAL, message TEXT);
INSERT INTO test (message) VALUES ('This data should persist');
SELECT * FROM test;
EOF

# 2. Delete the pod
kubectl delete pod postgres-0 -n statefulset-demo

# 3. Wait for recreation
kubectl wait --for=condition=ready pod/postgres-0 -n statefulset-demo --timeout=120s

# 4. Verify data persists
kubectl exec -it postgres-0 -n statefulset-demo -- \
  psql -U myuser -d mydatabase -c "SELECT * FROM test;"
```

### PVC Lifecycle

**PVCs survive pod deletion**:
```bash
# Delete StatefulSet (keeps PVCs)
kubectl delete statefulset postgres -n statefulset-demo

# PVCs still exist
kubectl get pvc -n statefulset-demo

# Recreate StatefulSet
kubectl apply -f manifests/03-postgresql-statefulset.yaml

# Data is intact!
```

**PVC Retention Policies** (Kubernetes 1.23+):

```yaml
persistentVolumeClaimRetentionPolicy:
  whenDeleted: Retain  # Keep PVCs when StatefulSet deleted
  whenScaled: Delete   # Delete PVCs when scaling down
```

---

## Storage Classes

### Default Storage Class

```bash
# Check default StorageClass
kubectl get storageclass

# Check which is default
kubectl get storageclass -o jsonpath='{.items[?(@.metadata.annotations.storageclass\.kubernetes\.io/is-default-class=="true")].metadata.name}'
```

### Dynamic Provisioning

When you create a PVC without specifying a StorageClass, the default is used:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  # storageClassName omitted = uses default
```

### Cloud Provider Storage Classes

**AWS EBS**:
```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iopsPerGB: "10"
```

**GCP Persistent Disk**:
```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/gce-pd
parameters:
  type: pd-ssd
```

**Azure Disk**:
```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/azure-disk
parameters:
  storageaccounttype: Premium_LRS
```

---

## Troubleshooting

### Pods Stuck in Pending

**Check**: PVC not bound to PV

```bash
# Check PVC status
kubectl get pvc -n statefulset-demo

# If "Pending":
kubectl describe pvc <pvc-name> -n statefulset-demo

# Common causes:
# - No PV matches PVC requirements
# - StorageClass doesn't exist
# - No storage available
```

**Solution**:
```bash
# Check available PVs
kubectl get pv

# Check StorageClass exists
kubectl get storageclass

# Check events
kubectl get events -n statefulset-demo --sort-by='.lastTimestamp'
```

### Data Not Persisting

**Check**: Volume mounted correctly

```bash
# Check volume mounts
kubectl describe pod <pod-name> -n statefulset-demo | grep -A 10 "Mounts:"

# Check PVC is bound
kubectl get pvc -n statefulset-demo

# Verify data location
kubectl exec <pod-name> -n statefulset-demo -- df -h
kubectl exec <pod-name> -n statefulset-demo -- ls -la /path/to/data
```

### PVC Deletion Stuck

**Cause**: PVC has finalizers or still in use

```bash
# Check PVC details
kubectl describe pvc <pvc-name> -n statefulset-demo

# Check if pods are using it
kubectl get pods -n statefulset-demo -o yaml | grep -A 5 persistentVolumeClaim

# Force delete (careful!)
kubectl patch pvc <pvc-name> -n statefulset-demo -p '{"metadata":{"finalizers":null}}'
```

### StatefulSet Update Stuck

**Check**: Pod not becoming Ready

```bash
# Check rollout status
kubectl rollout status statefulset/<name> -n statefulset-demo

# Check pod events
kubectl describe pod <pod-name> -n statefulset-demo

# Check logs
kubectl logs <pod-name> -n statefulset-demo
kubectl logs <pod-name> -n statefulset-demo --previous

# Force update (deletes pods)
kubectl delete pod <pod-name> -n statefulset-demo
```

---

## Best Practices

### 1. Always Use PersistentVolumes for Stateful Data

❌ **Bad** (data lost on pod restart):
```yaml
volumes:
- name: data
  emptyDir: {}
```

✅ **Good** (data persists):
```yaml
volumeClaimTemplates:
- metadata:
    name: data
  spec:
    accessModes: [ "ReadWriteOnce" ]
    resources:
      requests:
        storage: 1Gi
```

### 2. Use Headless Services

```yaml
apiVersion: v1
kind: Service
metadata:
  name: postgres-headless
spec:
  clusterIP: None  # Headless
  selector:
    app: postgres
```

### 3. Set Resource Limits

```yaml
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

### 4. Configure Health Probes

```yaml
livenessProbe:
  exec:
    command:
    - pg_isready
    - -U
    - myuser
  initialDelaySeconds: 30
  periodSeconds: 10
```

### 5. Use Init Containers for Setup

```yaml
initContainers:
- name: init-permissions
  image: busybox
  command:
  - sh
  - -c
  - chown -R 999:999 /data
```

### 6. Set PVC Retention Policy

```yaml
persistentVolumeClaimRetentionPolicy:
  whenDeleted: Retain
  whenScaled: Delete
```

### 7. Use StorageClass for Dynamic Provisioning

```yaml
volumeClaimTemplates:
- spec:
    storageClassName: fast-ssd
```

---

## Commands Reference

### StatefulSet Commands

```bash
# Create
kubectl apply -f statefulset.yaml

# Get
kubectl get statefulsets -n <namespace>
kubectl get sts -n <namespace>  # Short form

# Describe
kubectl describe statefulset <name> -n <namespace>

# Scale
kubectl scale statefulset <name> --replicas=5 -n <namespace>

# Update
kubectl set image statefulset/<name> <container>=<image> -n <namespace>

# Rollout status
kubectl rollout status statefulset/<name> -n <namespace>

# Rollback
kubectl rollout undo statefulset/<name> -n <namespace>

# Delete (keeps PVCs)
kubectl delete statefulset <name> -n <namespace>

# Delete (deletes PVCs)
kubectl delete statefulset <name> -n <namespace> --cascade=orphan
kubectl delete pvc -l app=<name> -n <namespace>
```

### PVC Commands

```bash
# List PVCs
kubectl get pvc -n <namespace>

# Describe PVC
kubectl describe pvc <name> -n <namespace>

# Check binding
kubectl get pvc <name> -n <namespace> -o yaml

# Delete PVC
kubectl delete pvc <name> -n <namespace>

# Check PV
kubectl get pv
```

### Storage Commands

```bash
# List StorageClasses
kubectl get storageclass
kubectl get sc

# Describe StorageClass
kubectl describe storageclass <name>

# Set default StorageClass
kubectl patch storageclass <name> -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
```

---

## Next Steps

After completing this exercise, you should:

1. Understand when to use StatefulSets vs Deployments
2. Be able to deploy databases on Kubernetes
3. Know how to manage persistent storage
4. Understand different volume types and their use cases
5. Be comfortable with PV/PVC concepts
6. Know how to test data persistence

**Continue to**: Exercise 05 - ConfigMaps and Secrets Management

---

## Additional Resources

- [Kubernetes StatefulSets Documentation](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/)
- [Persistent Volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)
- [Storage Classes](https://kubernetes.io/docs/concepts/storage/storage-classes/)
- [Volume Types](https://kubernetes.io/docs/concepts/storage/volumes/)

---

**Exercise Type**: Hands-on Implementation
**Difficulty**: Intermediate
**Estimated Time**: 3-4 hours
**Prerequisites**: Exercises 01-03
