# Implementation Guide: StatefulSets and Persistent Storage

## Table of Contents

1. [Introduction](#introduction)
2. [StatefulSets vs Deployments](#statefulsets-vs-deployments)
3. [PersistentVolumes and PersistentVolumeClaims](#persistentvolumes-and-persistentvolumeclaims)
4. [StorageClasses and Dynamic Provisioning](#storageclasses-and-dynamic-provisioning)
5. [StatefulSet Deployment with Ordered Startup](#statefulset-deployment-with-ordered-startup)
6. [Headless Services for Stable Network IDs](#headless-services-for-stable-network-ids)
7. [Scaling StatefulSets](#scaling-statefulsets)
8. [Production ML Stateful Workloads](#production-ml-stateful-workloads)
9. [Advanced Topics](#advanced-topics)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)

---

## Introduction

StatefulSets are Kubernetes resources designed for managing stateful applications that require stable network identities, persistent storage, and ordered deployment. This guide provides comprehensive coverage of StatefulSets and persistent storage concepts with practical examples focused on ML infrastructure workloads.

### When to Use StatefulSets

**Use StatefulSets for:**
- Databases (PostgreSQL, MySQL, MongoDB)
- Message queues (Kafka, RabbitMQ, Redis)
- Distributed systems requiring stable identities (Elasticsearch, ZooKeeper, etcd)
- ML workloads with persistent state (MLflow, feature stores, model registries)
- Distributed training with checkpoint synchronization

**Use Deployments for:**
- Stateless web applications
- API servers
- Microservices without state
- Worker pools processing from external queues
- ML model serving (when models are loaded from external storage)

---

## StatefulSets vs Deployments

### Key Differences

| Feature | StatefulSet | Deployment |
|---------|-------------|------------|
| **Pod Naming** | Predictable: `postgres-0`, `postgres-1`, `postgres-2` | Random hash: `nginx-7d8f9c5b4-x7k2m` |
| **DNS Hostname** | Stable: `pod-0.service.namespace.svc.cluster.local` | No stable DNS (only Service IP) |
| **Pod Identity** | Persistent across restarts | Changes on every restart |
| **Storage** | Per-pod PersistentVolumeClaims | Shared or ephemeral |
| **Startup Order** | Sequential (0→1→2) or Parallel | Always parallel |
| **Update Strategy** | Reverse order (2→1→0) | Rolling update any order |
| **Scaling** | Ordered addition/removal | Parallel |
| **Network Identity** | Stable, persistent | Dynamic |
| **Use Case** | Databases, clustered apps | Stateless services |

### Practical Example: Pod Naming and DNS

**Deployment Behavior:**
```bash
# Deploy nginx with Deployment
kubectl create deployment nginx --image=nginx --replicas=3

# Pod names are random
kubectl get pods
# NAME                     READY   STATUS
# nginx-7d8f9c5b4-9k2x7   1/1     Running
# nginx-7d8f9c5b4-m4p8r   1/1     Running
# nginx-7d8f9c5b4-z1t5q   1/1     Running

# Delete a pod - new pod gets different name
kubectl delete pod nginx-7d8f9c5b4-9k2x7
# New pod: nginx-7d8f9c5b4-3n6w8  # Different hash
```

**StatefulSet Behavior:**
```bash
# Deploy nginx with StatefulSet
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: web
spec:
  serviceName: "nginx"
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx
EOF

# Pod names are predictable
kubectl get pods
# NAME    READY   STATUS
# web-0   1/1     Running
# web-1   1/1     Running
# web-2   1/1     Running

# Delete a pod - same name is recreated
kubectl delete pod web-0
# New pod: web-0  # Same name!
```

### StatefulSet Pod Identity

Each StatefulSet pod has a persistent identity composed of:

1. **Ordinal Index**: Integer starting from 0 (web-0, web-1, web-2)
2. **Stable Hostname**: `$(podname).$(service).$(namespace).svc.cluster.local`
3. **Stable Storage**: PersistentVolumeClaim bound to the pod's ordinal

**Example: PostgreSQL Cluster**

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: "postgres-headless"
  replicas: 3
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        # Each pod knows its identity
        command:
        - sh
        - -c
        - |
          echo "I am pod: $POD_NAME"
          echo "My hostname: $(hostname)"
          echo "My DNS: $POD_NAME.postgres-headless.default.svc.cluster.local"
          exec docker-entrypoint.sh postgres
```

Result:
```
postgres-0: I am pod: postgres-0
            My hostname: postgres-0
            My DNS: postgres-0.postgres-headless.default.svc.cluster.local

postgres-1: I am pod: postgres-1
            My hostname: postgres-1
            My DNS: postgres-1.postgres-headless.default.svc.cluster.local
```

---

## PersistentVolumes and PersistentVolumeClaims

### Understanding the PV/PVC Model

Kubernetes separates storage provisioning (admin responsibility) from storage consumption (developer responsibility):

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Storage   │─────>│ Persistent  │─────>│ Persistent  │
│  Provider   │      │   Volume    │      │   Volume    │
│  (NFS/EBS)  │      │    (PV)     │      │   Claim     │
└─────────────┘      └─────────────┘      │   (PVC)     │
     Admin                Admin            └─────────────┘
   Provision                                  Developer
                                             Requests
```

### PersistentVolume (PV)

A PersistentVolume is a cluster-level resource representing physical storage.

**Example: Manual PV Creation**

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: ml-models-pv
  labels:
    type: local
    use-case: ml-storage
spec:
  # Storage capacity
  capacity:
    storage: 10Gi

  # Access modes
  # - ReadWriteOnce (RWO): Single node read-write
  # - ReadOnlyMany (ROX): Multiple nodes read-only
  # - ReadWriteMany (RWX): Multiple nodes read-write
  accessModes:
    - ReadWriteOnce

  # What happens when PVC is deleted
  # - Retain: Keep the PV and data (manual cleanup)
  # - Delete: Delete the PV and data
  # - Recycle: Deprecated - basic scrub (rm -rf)
  persistentVolumeReclaimPolicy: Retain

  # Storage class (for dynamic provisioning)
  storageClassName: standard

  # Actual storage backend
  hostPath:
    path: /mnt/data/ml-models
    type: DirectoryOrCreate
```

**Cloud Provider Examples:**

```yaml
# AWS EBS Volume
apiVersion: v1
kind: PersistentVolume
metadata:
  name: aws-ebs-pv
spec:
  capacity:
    storage: 20Gi
  accessModes:
    - ReadWriteOnce
  awsElasticBlockStore:
    volumeID: vol-0a1b2c3d4e5f6g7h8
    fsType: ext4

---
# GCP Persistent Disk
apiVersion: v1
kind: PersistentVolume
metadata:
  name: gcp-pd-pv
spec:
  capacity:
    storage: 50Gi
  accessModes:
    - ReadWriteOnce
  gcePersistentDisk:
    pdName: ml-training-disk
    fsType: ext4

---
# Azure Disk
apiVersion: v1
kind: PersistentVolume
metadata:
  name: azure-disk-pv
spec:
  capacity:
    storage: 100Gi
  accessModes:
    - ReadWriteOnce
  azureDisk:
    diskName: ml-data-disk
    diskURI: /subscriptions/.../ml-data-disk
    kind: Managed
    fsType: ext4

---
# NFS (for ReadWriteMany)
apiVersion: v1
kind: PersistentVolume
metadata:
  name: nfs-pv
spec:
  capacity:
    storage: 200Gi
  accessModes:
    - ReadWriteMany  # Multiple pods can share
  nfs:
    server: nfs-server.example.com
    path: /shared/ml-datasets
```

### PersistentVolumeClaim (PVC)

A PVC is a request for storage by a user/pod.

**Example: Basic PVC**

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ml-models-pvc
  namespace: ml-pipeline
spec:
  # Must match or be subset of PV
  accessModes:
    - ReadWriteOnce

  # Storage class - binds to PV with same class
  storageClassName: standard

  # Storage request
  resources:
    requests:
      storage: 5Gi  # Can be less than PV capacity

  # Optional: Selector to bind to specific PV
  selector:
    matchLabels:
      use-case: ml-storage
```

### PV/PVC Binding Process

```
1. PVC Created → Kubernetes searches for matching PV
   ├─ Same StorageClass
   ├─ Sufficient capacity
   ├─ Compatible access modes
   └─ Matching labels (if selector defined)

2. PV Found → Binding occurs
   ├─ PVC status: Bound
   ├─ PV status: Bound
   └─ 1:1 relationship established

3. PVC Not Satisfied → Remains Pending
   ├─ Wait for manual PV creation, OR
   └─ Wait for dynamic provisioning
```

**Checking Binding Status:**

```bash
# Check PVC status
kubectl get pvc ml-models-pvc
# NAME             STATUS   VOLUME         CAPACITY   ACCESS MODES
# ml-models-pvc    Bound    ml-models-pv   10Gi       RWO

# Check which PV is bound
kubectl get pvc ml-models-pvc -o yaml | grep volumeName
#   volumeName: ml-models-pv

# Check PV status
kubectl get pv ml-models-pv
# NAME           CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS   CLAIM
# ml-models-pv   10Gi       RWO            Retain           Bound    ml-pipeline/ml-models-pvc
```

### Using PVC in Pods

**Example: Pod with PVC**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-training-pod
spec:
  containers:
  - name: trainer
    image: python:3.9
    command: ["python", "train.py"]
    volumeMounts:
    - name: model-storage
      mountPath: /models  # Where to mount in container

  volumes:
  - name: model-storage
    persistentVolumeClaim:
      claimName: ml-models-pvc  # Reference to PVC
```

**Example: StatefulSet with VolumeClaimTemplates**

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: ml-workers
spec:
  serviceName: ml-workers
  replicas: 3
  selector:
    matchLabels:
      app: ml-worker
  template:
    metadata:
      labels:
        app: ml-worker
    spec:
      containers:
      - name: worker
        image: ml-worker:latest
        volumeMounts:
        - name: workspace
          mountPath: /workspace

  # Creates PVC for each pod automatically
  volumeClaimTemplates:
  - metadata:
      name: workspace
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 50Gi

# Results in PVCs:
# - workspace-ml-workers-0 (for pod ml-workers-0)
# - workspace-ml-workers-1 (for pod ml-workers-1)
# - workspace-ml-workers-2 (for pod ml-workers-2)
```

---

## StorageClasses and Dynamic Provisioning

### What is a StorageClass?

A StorageClass provides a way to describe different "classes" of storage with different performance characteristics, backup policies, or cost profiles.

**Key Benefits:**
1. **Dynamic Provisioning**: Automatically create PVs when PVCs are created
2. **Abstraction**: Developers request storage without knowing underlying infrastructure
3. **Policy Enforcement**: Admins control storage types, performance, and costs

### StorageClass Anatomy

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
  annotations:
    # Mark as default StorageClass (optional)
    storageclass.kubernetes.io/is-default-class: "true"

# Storage provisioner (cloud provider or CSI driver)
provisioner: kubernetes.io/aws-ebs

# Provisioner-specific parameters
parameters:
  type: gp3              # EBS volume type
  iopsPerGB: "10"        # IOPS per GB
  encrypted: "true"      # Enable encryption
  kmsKeyId: "arn:aws:kms:..." # KMS key for encryption

# What happens to PV when PVC is deleted
# - Delete: Remove PV and underlying storage
# - Retain: Keep PV and data for manual cleanup
reclaimPolicy: Delete

# Allow volume expansion after creation
allowVolumeExpansion: true

# When to create and bind PV
# - Immediate: Create PV as soon as PVC is created
# - WaitForFirstConsumer: Wait until pod using PVC is scheduled
volumeBindingMode: WaitForFirstConsumer

# Mount options for volumes
mountOptions:
  - debug
```

### Cloud Provider StorageClasses

**AWS EBS Examples:**

```yaml
# General Purpose SSD (gp3) - Balanced price/performance
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gp3-standard
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
reclaimPolicy: Delete
volumeBindingMode: WaitForFirstConsumer

---
# Provisioned IOPS SSD (io2) - High performance
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: io2-high-perf
provisioner: ebs.csi.aws.com
parameters:
  type: io2
  iops: "10000"
  encrypted: "true"
reclaimPolicy: Retain  # Keep data for safety
volumeBindingMode: WaitForFirstConsumer

---
# Throughput Optimized HDD (st1) - Large sequential workloads
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: st1-throughput
provisioner: ebs.csi.aws.com
parameters:
  type: st1
reclaimPolicy: Delete
volumeBindingMode: Immediate
```

**GCP Persistent Disk Examples:**

```yaml
# Standard Persistent Disk (pd-standard) - Cost-effective
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: pd-standard
provisioner: pd.csi.storage.gke.io
parameters:
  type: pd-standard
  replication-type: regional-pd  # Multi-zone replication
reclaimPolicy: Delete
volumeBindingMode: WaitForFirstConsumer

---
# SSD Persistent Disk (pd-ssd) - High performance
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: pd-ssd
provisioner: pd.csi.storage.gke.io
parameters:
  type: pd-ssd
reclaimPolicy: Delete
volumeBindingMode: WaitForFirstConsumer

---
# Extreme Persistent Disk - Ultra-high IOPS
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: pd-extreme
provisioner: pd.csi.storage.gke.io
parameters:
  type: pd-extreme
  provisioned-iops-on-create: "10000"
reclaimPolicy: Retain
volumeBindingMode: WaitForFirstConsumer
```

**Azure Disk Examples:**

```yaml
# Standard HDD
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: azure-standard
provisioner: disk.csi.azure.com
parameters:
  storageaccounttype: Standard_LRS  # Locally redundant storage
  kind: Managed
reclaimPolicy: Delete

---
# Premium SSD
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: azure-premium
provisioner: disk.csi.azure.com
parameters:
  storageaccounttype: Premium_LRS
  kind: Managed
reclaimPolicy: Delete
volumeBindingMode: WaitForFirstConsumer

---
# Ultra Disk (ultra-low latency)
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: azure-ultra
provisioner: disk.csi.azure.com
parameters:
  storageaccounttype: UltraSSD_LRS
  cachingMode: None
reclaimPolicy: Delete
volumeBindingMode: WaitForFirstConsumer
```

### ML-Focused StorageClass Examples

```yaml
# Training data (large datasets, sequential reads)
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: ml-training-data
  labels:
    purpose: ml-training
provisioner: ebs.csi.aws.com
parameters:
  type: st1  # Throughput optimized
  throughput: "500"
reclaimPolicy: Retain  # Never delete training data
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer

---
# Model checkpoints (frequent writes, low latency)
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: ml-checkpoints
  labels:
    purpose: ml-checkpoints
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  iops: "5000"
  throughput: "250"
reclaimPolicy: Delete  # Checkpoints can be recreated
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer

---
# Feature store (high IOPS, low latency)
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: ml-feature-store
  labels:
    purpose: feature-store
provisioner: ebs.csi.aws.com
parameters:
  type: io2
  iops: "20000"
  encrypted: "true"
reclaimPolicy: Retain  # Critical data
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer

---
# Shared datasets (NFS for multi-node access)
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: ml-shared-datasets
provisioner: nfs.csi.k8s.io
parameters:
  server: nfs-server.ml-infra.svc.cluster.local
  share: /datasets
reclaimPolicy: Retain
volumeBindingMode: Immediate
```

### Dynamic Provisioning in Action

```yaml
# 1. Create PVC (no PV exists yet)
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: dynamic-pvc
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: fast-ssd  # Reference to StorageClass
  resources:
    requests:
      storage: 100Gi

# What happens:
# 1. Kubernetes sees PVC with storageClassName: fast-ssd
# 2. Looks up fast-ssd StorageClass
# 3. Calls provisioner (e.g., ebs.csi.aws.com)
# 4. Provisioner creates 100Gi EBS volume
# 5. Creates PV representing that volume
# 6. Binds PVC to newly created PV
# 7. PVC becomes "Bound" and ready to use
```

**Watching Dynamic Provisioning:**

```bash
# Create PVC
kubectl apply -f dynamic-pvc.yaml

# Watch events
kubectl get events --watch
# LAST SEEN   TYPE     REASON                      MESSAGE
# 0s          Normal   Provisioning                Provisioning volume...
# 2s          Normal   ProvisioningSucceeded       Successfully provisioned volume pvc-abc123
# 2s          Normal   Bound                       Successfully bound to pvc-abc123

# Check PVC
kubectl get pvc dynamic-pvc
# NAME          STATUS   VOLUME       CAPACITY   STORAGECLASS
# dynamic-pvc   Bound    pvc-abc123   100Gi      fast-ssd

# Check auto-created PV
kubectl get pv pvc-abc123
# NAME         CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS   STORAGECLASS
# pvc-abc123   100Gi      RWO            Delete           Bound    fast-ssd
```

### Managing StorageClasses

```bash
# List StorageClasses
kubectl get storageclass
kubectl get sc  # Short form

# Check default StorageClass
kubectl get sc -o jsonpath='{.items[?(@.metadata.annotations.storageclass\.kubernetes\.io/is-default-class=="true")].metadata.name}'

# Set default StorageClass
kubectl patch storageclass fast-ssd -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'

# Remove default annotation
kubectl patch storageclass old-default -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"false"}}}'

# Describe StorageClass
kubectl describe sc fast-ssd
```

---

## StatefulSet Deployment with Ordered Startup

### Pod Management Policies

StatefulSets support two pod management policies that control deployment and scaling behavior:

#### 1. OrderedReady (Default)

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  podManagementPolicy: OrderedReady
  replicas: 3
  # ...
```

**Behavior:**
- Pods are created sequentially: 0 → 1 → 2
- Each pod must be Running and Ready before next pod starts
- Updates occur in reverse order: 2 → 1 → 0
- Scale-down removes highest ordinal first

**Use Cases:**
- Databases requiring primary/replica setup
- Distributed systems with leader election
- Applications where startup order matters

#### 2. Parallel

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-cache
spec:
  podManagementPolicy: Parallel
  replicas: 5
  # ...
```

**Behavior:**
- All pods created simultaneously
- No waiting for Ready status
- Updates can happen to all pods at once
- Faster deployment and scaling

**Use Cases:**
- Independent cache nodes
- Sharded workloads
- ML training workers (no coordination needed)

### Practical Example: PostgreSQL with Ordered Startup

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: ml-db
spec:
  serviceName: postgres-headless
  replicas: 3
  podManagementPolicy: OrderedReady  # Ordered deployment

  selector:
    matchLabels:
      app: postgres

  template:
    metadata:
      labels:
        app: postgres
    spec:
      # Init container runs before main container
      initContainers:
      - name: init-postgres
        image: postgres:15-alpine
        command:
        - sh
        - -c
        - |
          set -ex
          # Get pod ordinal from hostname
          [[ $(hostname) =~ -([0-9]+)$ ]] || exit 1
          ordinal=${BASH_REMATCH[1]}

          echo "Pod ordinal: $ordinal"

          # postgres-0 is primary, others are replicas
          if [[ $ordinal -eq 0 ]]; then
            echo "I am the primary (postgres-0)"
            echo "primary" > /var/lib/postgresql/data/role
          else
            echo "I am a replica (postgres-$ordinal)"
            echo "replica" > /var/lib/postgresql/data/role

            # Wait for primary to be ready
            echo "Waiting for postgres-0 to be ready..."
            until pg_isready -h postgres-0.postgres-headless -U postgres; do
              sleep 2
            done
            echo "Primary is ready, proceeding..."
          fi
        volumeMounts:
        - name: data
          mountPath: /var/lib/postgresql/data

      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name

        # Determine replication role
        command:
        - sh
        - -c
        - |
          role=$(cat /var/lib/postgresql/data/role)
          echo "Starting as: $role"

          if [[ "$role" == "primary" ]]; then
            echo "Configuring as primary..."
            # Primary configuration
            export POSTGRES_INITDB_ARGS="--encoding=UTF8"
          else
            echo "Configuring as replica..."
            # Replica configuration - set up streaming replication
            export PGDATA=/var/lib/postgresql/data/pgdata
            rm -rf $PGDATA
            pg_basebackup -h postgres-0.postgres-headless -D $PGDATA -U postgres -Fp -Xs -R
          fi

          exec docker-entrypoint.sh postgres

        ports:
        - containerPort: 5432
          name: postgres

        readinessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - postgres
          initialDelaySeconds: 10
          periodSeconds: 5

        volumeMounts:
        - name: data
          mountPath: /var/lib/postgresql/data

  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 10Gi
```

**Deployment Process:**

```
Time 0s:  postgres-0 created
          └─ Init container runs (primary setup)
          └─ Main container starts
          └─ Readiness probe checks...

Time 30s: postgres-0 becomes Ready
          postgres-1 created
          └─ Init container runs (replica setup, waits for postgres-0)
          └─ Main container starts (streaming replication from postgres-0)
          └─ Readiness probe checks...

Time 60s: postgres-1 becomes Ready
          postgres-2 created
          └─ Same process as postgres-1

Time 90s: All pods Running and Ready
```

### Monitoring Ordered Startup

```bash
# Watch pod creation in real-time
kubectl get pods -n ml-db -w

# Check pod events
kubectl describe pod postgres-0 -n ml-db
kubectl describe pod postgres-1 -n ml-db

# View init container logs
kubectl logs postgres-1 -n ml-db -c init-postgres

# Check readiness
kubectl get pods -n ml-db -o wide
```

### Update Strategy

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
spec:
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      partition: 0  # Update all pods (default)
      # partition: 2  # Only update pods >= 2 (canary)
```

**Update Behavior:**

```bash
# Trigger update
kubectl set image statefulset/redis redis=redis:7.2-alpine -n ml-db

# With OrderedReady, updates happen in reverse:
# 1. Delete and recreate redis-2
# 2. Wait for redis-2 to be Ready
# 3. Delete and recreate redis-1
# 4. Wait for redis-1 to be Ready
# 5. Delete and recreate redis-0
# 6. Wait for redis-0 to be Ready

# Watch update progress
kubectl rollout status statefulset/redis -n ml-db
```

---

## Headless Services for Stable Network IDs

### What is a Headless Service?

A headless Service is a Service with `clusterIP: None`. Instead of load-balancing to pods, it returns DNS records for each pod.

**Regular Service:**
```
nslookup postgres.ml-db.svc.cluster.local
→ Returns: 10.96.1.5 (Service ClusterIP)
→ Traffic load-balanced to random pod
```

**Headless Service:**
```
nslookup postgres-headless.ml-db.svc.cluster.local
→ Returns: 10.244.1.10, 10.244.2.11, 10.244.3.12 (All pod IPs)

nslookup postgres-0.postgres-headless.ml-db.svc.cluster.local
→ Returns: 10.244.1.10 (Specific pod IP)
```

### Creating Headless Services

```yaml
apiVersion: v1
kind: Service
metadata:
  name: postgres-headless
  namespace: ml-db
spec:
  clusterIP: None  # This makes it headless
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
    name: postgres
  # No type specified (defaults to ClusterIP, but None means headless)
```

### Stable Network Identity

Each StatefulSet pod gets a DNS record:

```
<pod-name>.<service-name>.<namespace>.svc.cluster.local
```

**Example:**
```
postgres-0.postgres-headless.ml-db.svc.cluster.local → 10.244.1.10
postgres-1.postgres-headless.ml-db.svc.cluster.local → 10.244.2.11
postgres-2.postgres-headless.ml-db.svc.cluster.local → 10.244.3.12
```

**Key Benefit:** DNS name stays same even if pod is deleted and recreated with new IP!

### Practical Example: MLflow Tracking Server

```yaml
---
# Headless Service for StatefulSet
apiVersion: v1
kind: Service
metadata:
  name: mlflow-headless
  namespace: ml-tracking
  labels:
    app: mlflow
spec:
  clusterIP: None
  selector:
    app: mlflow
  ports:
  - port: 5000
    targetPort: 5000
    name: http
  - port: 5432
    targetPort: 5432
    name: postgres

---
# Regular Service for client access (load-balanced)
apiVersion: v1
kind: Service
metadata:
  name: mlflow
  namespace: ml-tracking
  labels:
    app: mlflow
spec:
  type: ClusterIP
  selector:
    app: mlflow
  ports:
  - port: 5000
    targetPort: 5000
    name: http

---
# StatefulSet
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mlflow
  namespace: ml-tracking
spec:
  serviceName: mlflow-headless
  replicas: 3
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      containers:
      - name: mlflow
        image: mlflow-server:latest
        env:
        - name: BACKEND_STORE_URI
          value: "postgresql://mlflow:password@localhost:5432/mlflow"
        - name: DEFAULT_ARTIFACT_ROOT
          value: "s3://ml-artifacts/$(POD_NAME)"
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        - name: POD_FQDN
          value: "$(POD_NAME).mlflow-headless.ml-tracking.svc.cluster.local"
        ports:
        - containerPort: 5000
          name: http

      # Sidecar PostgreSQL for backend store
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: mlflow
        - name: POSTGRES_USER
          value: mlflow
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mlflow-db-secret
              key: password
        ports:
        - containerPort: 5432
          name: postgres
        volumeMounts:
        - name: db-data
          mountPath: /var/lib/postgresql/data

  volumeClaimTemplates:
  - metadata:
      name: db-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 20Gi
```

### Accessing Pods via Stable DNS

```bash
# Access specific pod
kubectl exec -it mlflow-0 -n ml-tracking -- curl http://localhost:5000/health

# Access via DNS from another pod
kubectl run -it --rm debug --image=curlimages/curl -n ml-tracking -- sh
/ $ curl http://mlflow-0.mlflow-headless:5000/health
/ $ curl http://mlflow-1.mlflow-headless:5000/health
/ $ curl http://mlflow-2.mlflow-headless:5000/health

# Access load-balanced (via regular Service)
/ $ curl http://mlflow:5000/health  # Round-robin to any pod

# DNS lookup
/ $ nslookup mlflow-headless
# Returns all pod IPs:
# mlflow-0.mlflow-headless.ml-tracking.svc.cluster.local → 10.244.1.10
# mlflow-1.mlflow-headless.ml-tracking.svc.cluster.local → 10.244.2.11
# mlflow-2.mlflow-headless.ml-tracking.svc.cluster.local → 10.244.3.12

/ $ nslookup mlflow
# Returns Service ClusterIP:
# mlflow.ml-tracking.svc.cluster.local → 10.96.5.20
```

### When to Use Headless vs Regular Services

**Use Headless Service when:**
- Need to address specific pod instances
- Implementing peer discovery in distributed systems
- Database replication (connect to specific primary/replica)
- Stateful applications needing direct pod-to-pod communication

**Use Regular Service when:**
- Need load balancing across pods
- Clients don't care which pod handles request
- Stateless applications
- External access via LoadBalancer/NodePort

**Use Both:**
```yaml
# Common pattern: Both headless and regular Service
# - Headless for StatefulSet (inter-pod communication)
# - Regular for clients (load-balanced access)

apiVersion: v1
kind: Service
metadata:
  name: app-headless
spec:
  clusterIP: None
  selector:
    app: myapp
  ports:
  - port: 8080

---
apiVersion: v1
kind: Service
metadata:
  name: app
spec:
  type: LoadBalancer
  selector:
    app: myapp
  ports:
  - port: 80
    targetPort: 8080
```

---

## Scaling StatefulSets

### Manual Scaling

```bash
# Scale up (adds pods 3, 4)
kubectl scale statefulset mlflow --replicas=5 -n ml-tracking

# Scale down (removes pods 4, 3)
kubectl scale statefulset mlflow --replicas=3 -n ml-tracking

# Watch scaling process
kubectl get pods -n ml-tracking -w

# Or using kubectl patch
kubectl patch statefulset mlflow -n ml-tracking -p '{"spec":{"replicas":5}}'
```

### Scaling Behavior

**Scale Up (OrderedReady):**
```
Current: 3 replicas (mlflow-0, mlflow-1, mlflow-2)
Target:  5 replicas

Process:
1. Create mlflow-3
2. Wait for mlflow-3 to be Ready
3. Create mlflow-4
4. Wait for mlflow-4 to be Ready
5. Done
```

**Scale Down (OrderedReady):**
```
Current: 5 replicas (mlflow-0, mlflow-1, mlflow-2, mlflow-3, mlflow-4)
Target:  3 replicas

Process:
1. Delete mlflow-4 (highest ordinal first)
2. Wait for mlflow-4 to terminate
3. Delete mlflow-3
4. Wait for mlflow-3 to terminate
5. Done

Note: PVCs workspace-mlflow-3 and workspace-mlflow-4 remain!
```

### PVC Management During Scaling

**Default Behavior:**
- PVCs are NOT deleted when scaling down
- PVCs are reattached when scaling back up

```bash
# Start with 5 replicas
kubectl get pvc -n ml-tracking
# NAME                 STATUS   VOLUME
# db-data-mlflow-0     Bound    pvc-abc123
# db-data-mlflow-1     Bound    pvc-abc456
# db-data-mlflow-2     Bound    pvc-abc789
# db-data-mlflow-3     Bound    pvc-def123
# db-data-mlflow-4     Bound    pvc-def456

# Scale down to 3
kubectl scale statefulset mlflow --replicas=3 -n ml-tracking

# PVCs 3 and 4 still exist!
kubectl get pvc -n ml-tracking
# NAME                 STATUS   VOLUME
# db-data-mlflow-0     Bound    pvc-abc123
# db-data-mlflow-1     Bound    pvc-abc456
# db-data-mlflow-2     Bound    pvc-abc789
# db-data-mlflow-3     Bound    pvc-def123  # Still here
# db-data-mlflow-4     Bound    pvc-def456  # Still here

# Scale back up to 5
kubectl scale statefulset mlflow --replicas=5 -n ml-tracking

# Pods reattach to existing PVCs - data is preserved!
```

**Manual PVC Cleanup:**

```bash
# Delete specific PVC
kubectl delete pvc db-data-mlflow-4 -n ml-tracking

# Delete all PVCs for a StatefulSet
kubectl delete pvc -l app=mlflow -n ml-tracking

# Force delete if stuck
kubectl patch pvc db-data-mlflow-4 -n ml-tracking -p '{"metadata":{"finalizers":null}}'
```

### PVC Retention Policy (Kubernetes 1.23+)

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mlflow
spec:
  persistentVolumeClaimRetentionPolicy:
    # What happens to PVCs when StatefulSet is deleted
    whenDeleted: Retain  # Options: Retain, Delete

    # What happens to PVCs when scaling down
    whenScaled: Delete   # Options: Retain, Delete

  # Rest of spec...
```

**Policy Combinations:**

| whenDeleted | whenScaled | Behavior |
|-------------|------------|----------|
| Retain | Retain | Keep PVCs always (default) |
| Retain | Delete | Keep on StatefulSet delete, clean up on scale-down |
| Delete | Retain | Clean on StatefulSet delete, keep on scale-down |
| Delete | Delete | Always clean up PVCs |

### Horizontal Pod Autoscaler (HPA) with StatefulSets

StatefulSets can use HPA for automatic scaling:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mlflow-hpa
  namespace: ml-tracking
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: StatefulSet
    name: mlflow

  minReplicas: 3
  maxReplicas: 10

  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70

  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80

  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min before scale-down
      policies:
      - type: Pods
        value: 1
        periodSeconds: 60  # Remove max 1 pod per minute

    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60  # Add max 50% per minute
```

**Important HPA Considerations for StatefulSets:**
1. Ensure PVC retention policy is set appropriately
2. Scale-down can be slow due to ordered termination
3. Consider using Parallel pod management for faster scaling
4. Monitor PVC usage to avoid orphaned storage

---

## Production ML Stateful Workloads

### Use Case 1: Distributed Training with Checkpoints

**Scenario:** Multi-node PyTorch distributed training with checkpoint synchronization.

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: pytorch-training
  namespace: ml-training
spec:
  serviceName: pytorch-headless
  replicas: 4  # 4 training nodes
  podManagementPolicy: Parallel  # Start all nodes together

  selector:
    matchLabels:
      app: pytorch-training

  template:
    metadata:
      labels:
        app: pytorch-training
    spec:
      containers:
      - name: trainer
        image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
        command:
        - torchrun
        - --nnodes=4
        - --nproc_per_node=1
        - --rdzv_id=training-job-001
        - --rdzv_backend=c10d
        - --rdzv_endpoint=pytorch-training-0.pytorch-headless:29400
        - train.py

        env:
        - name: WORLD_SIZE
          value: "4"
        - name: RANK
          valueFrom:
            fieldRef:
              fieldPath: metadata.labels['apps.kubernetes.io/pod-index']
        - name: MASTER_ADDR
          value: pytorch-training-0.pytorch-headless
        - name: MASTER_PORT
          value: "29400"
        - name: CHECKPOINT_DIR
          value: /checkpoints

        ports:
        - containerPort: 29400
          name: distributed

        resources:
          requests:
            nvidia.com/gpu: 1
            memory: 32Gi
            cpu: 8
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi

        volumeMounts:
        # Per-node checkpoint storage
        - name: checkpoints
          mountPath: /checkpoints

        # Shared dataset (ReadWriteMany NFS)
        - name: training-data
          mountPath: /data
          readOnly: true

      volumes:
      - name: training-data
        persistentVolumeClaim:
          claimName: imagenet-dataset  # Shared PVC

  volumeClaimTemplates:
  - metadata:
      name: checkpoints
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi  # Checkpoint storage per node

---
# Shared training data (NFS for ReadWriteMany)
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: imagenet-dataset
  namespace: ml-training
spec:
  accessModes:
    - ReadWriteMany  # All pods can read
  storageClassName: nfs-storage
  resources:
    requests:
      storage: 500Gi

---
# Headless service for pod-to-pod communication
apiVersion: v1
kind: Service
metadata:
  name: pytorch-headless
  namespace: ml-training
spec:
  clusterIP: None
  selector:
    app: pytorch-training
  ports:
  - port: 29400
    name: distributed
```

**Benefits:**
- Each pod has dedicated checkpoint storage (survives pod restart)
- Shared dataset accessible by all pods (NFS)
- Stable network identities for distributed communication
- Checkpoints preserved across training job restarts

### Use Case 2: MLflow Tracking Server Cluster

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mlflow
  namespace: ml-tracking
spec:
  serviceName: mlflow-headless
  replicas: 3
  podManagementPolicy: OrderedReady

  selector:
    matchLabels:
      app: mlflow

  template:
    metadata:
      labels:
        app: mlflow
    spec:
      initContainers:
      # Initialize PostgreSQL backend
      - name: init-db
        image: postgres:15-alpine
        command:
        - sh
        - -c
        - |
          set -ex
          [[ $(hostname) =~ -([0-9]+)$ ]] || exit 1
          ordinal=${BASH_REMATCH[1]}

          # Only mlflow-0 initializes the schema
          if [[ $ordinal -eq 0 ]]; then
            until pg_isready -h mlflow-postgres -U mlflow; do
              echo "Waiting for PostgreSQL..."
              sleep 2
            done

            # Run migrations
            psql -h mlflow-postgres -U mlflow -d mlflow -c "
              CREATE TABLE IF NOT EXISTS experiments (
                experiment_id VARCHAR(32) PRIMARY KEY,
                name VARCHAR(256) NOT NULL UNIQUE,
                artifact_location VARCHAR(256),
                lifecycle_stage VARCHAR(32)
              );
            "
          fi
        env:
        - name: PGPASSWORD
          valueFrom:
            secretKeyRef:
              name: mlflow-db-secret
              key: password

      containers:
      - name: mlflow
        image: ghcr.io/mlflow/mlflow:v2.8.0
        command:
        - mlflow
        - server
        - --host=0.0.0.0
        - --port=5000
        - --backend-store-uri=$(BACKEND_STORE_URI)
        - --default-artifact-root=$(ARTIFACT_ROOT)

        env:
        - name: BACKEND_STORE_URI
          value: "postgresql://mlflow:$(DB_PASSWORD)@mlflow-postgres:5432/mlflow"
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mlflow-db-secret
              key: password
        - name: ARTIFACT_ROOT
          value: "s3://ml-artifacts/mlflow"
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: access-key
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: secret-key
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name

        ports:
        - containerPort: 5000
          name: http

        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10

        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 10
          periodSeconds: 5

        resources:
          requests:
            memory: 2Gi
            cpu: 1
          limits:
            memory: 4Gi
            cpu: 2

        volumeMounts:
        # Local cache for artifacts
        - name: cache
          mountPath: /tmp/mlflow-cache

  volumeClaimTemplates:
  - metadata:
      name: cache
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 50Gi

---
# PostgreSQL for backend store
apiVersion: v1
kind: Service
metadata:
  name: mlflow-postgres
  namespace: ml-tracking
spec:
  type: ClusterIP
  selector:
    app: mlflow-postgres
  ports:
  - port: 5432

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mlflow-postgres
  namespace: ml-tracking
spec:
  serviceName: mlflow-postgres
  replicas: 1
  selector:
    matchLabels:
      app: mlflow-postgres
  template:
    metadata:
      labels:
        app: mlflow-postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: mlflow
        - name: POSTGRES_USER
          value: mlflow
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mlflow-db-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: db-data
          mountPath: /var/lib/postgresql/data

  volumeClaimTemplates:
  - metadata:
      name: db-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 20Gi
```

### Use Case 3: Feature Store (Feast)

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: feast-online
  namespace: ml-features
spec:
  serviceName: feast-online-headless
  replicas: 3
  podManagementPolicy: Parallel

  selector:
    matchLabels:
      app: feast-online
      component: online-store

  template:
    metadata:
      labels:
        app: feast-online
        component: online-store
    spec:
      containers:
      # Redis for online feature store
      - name: redis
        image: redis:7.2-alpine
        command:
        - redis-server
        - /etc/redis/redis.conf

        ports:
        - containerPort: 6379
          name: redis

        livenessProbe:
          exec:
            command:
            - redis-cli
            - ping
          initialDelaySeconds: 30
          periodSeconds: 10

        readinessProbe:
          exec:
            command:
            - redis-cli
            - ping
          initialDelaySeconds: 5
          periodSeconds: 3

        resources:
          requests:
            memory: 4Gi
            cpu: 2
          limits:
            memory: 8Gi
            cpu: 4

        volumeMounts:
        - name: redis-data
          mountPath: /data
        - name: redis-config
          mountPath: /etc/redis

      volumes:
      - name: redis-config
        configMap:
          name: redis-config

  volumeClaimTemplates:
  - metadata:
      name: redis-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: ml-feature-store  # High IOPS storage
      resources:
        requests:
          storage: 100Gi

---
# Redis configuration for persistence
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-config
  namespace: ml-features
data:
  redis.conf: |
    # Persistence
    save 900 1      # Save after 900s if 1 key changed
    save 300 10     # Save after 300s if 10 keys changed
    save 60 10000   # Save after 60s if 10000 keys changed

    # AOF (Append Only File) for durability
    appendonly yes
    appendfilename "appendonly.aof"
    appendfsync everysec

    # Memory management
    maxmemory 6gb
    maxmemory-policy allkeys-lru

    # Network
    bind 0.0.0.0
    protected-mode no

    # Logging
    loglevel notice
    logfile "/data/redis.log"
```

---

## Advanced Topics

### Partitioned Updates (Canary Deployments)

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mlflow
spec:
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      partition: 2  # Only update pods with ordinal >= 2
```

**Usage:**

```bash
# Set partition to 2 (canary: only mlflow-2 updates)
kubectl patch statefulset mlflow -n ml-tracking -p '
{
  "spec": {
    "updateStrategy": {
      "rollingUpdate": {
        "partition": 2
      }
    }
  }
}'

# Update image
kubectl set image statefulset/mlflow mlflow=mlflow:v2.9.0 -n ml-tracking

# Only mlflow-2 updates (canary)
kubectl get pods -n ml-tracking -o wide

# Test canary
curl http://mlflow-2.mlflow-headless:5000/version

# If good, roll out to pod 1
kubectl patch statefulset mlflow -n ml-tracking -p '{"spec":{"updateStrategy":{"rollingUpdate":{"partition":1}}}}'

# Finally, roll out to all (partition 0)
kubectl patch statefulset mlflow -n ml-tracking -p '{"spec":{"updateStrategy":{"rollingUpdate":{"partition":0}}}}'
```

### Volume Expansion

```yaml
# StorageClass must support expansion
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: expandable-storage
provisioner: ebs.csi.aws.com
allowVolumeExpansion: true  # Enable expansion

---
# Expand PVC
kubectl patch pvc db-data-mlflow-0 -n ml-tracking -p '{"spec":{"resources":{"requests":{"storage":"20Gi"}}}}'

# Check expansion status
kubectl get pvc db-data-mlflow-0 -n ml-tracking -o yaml | grep -A 5 status

# May need to delete pod to trigger filesystem resize
kubectl delete pod mlflow-0 -n ml-tracking
```

### StatefulSet with Multiple Volume Types

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: ml-pipeline
spec:
  template:
    spec:
      containers:
      - name: pipeline
        volumeMounts:
        # Fast SSD for checkpoints
        - name: checkpoints
          mountPath: /checkpoints

        # HDD for logs (cheaper)
        - name: logs
          mountPath: /logs

        # Shared NFS for datasets
        - name: datasets
          mountPath: /datasets
          readOnly: true

        # Temp storage (emptyDir)
        - name: temp
          mountPath: /tmp

      volumes:
      # Shared dataset (NFS)
      - name: datasets
        persistentVolumeClaim:
          claimName: shared-datasets

      # Temp storage
      - name: temp
        emptyDir:
          sizeLimit: 10Gi

  volumeClaimTemplates:
  # Per-pod SSD for checkpoints
  - metadata:
      name: checkpoints
    spec:
      storageClassName: fast-ssd
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 100Gi

  # Per-pod HDD for logs
  - metadata:
      name: logs
    spec:
      storageClassName: standard-hdd
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 50Gi
```

---

## Troubleshooting

### Problem: Pods Stuck in Pending (PVC Not Bound)

**Symptoms:**
```bash
kubectl get pods -n ml-training
# NAME                READY   STATUS    RESTARTS   AGE
# pytorch-training-0  0/1     Pending   0          5m
```

**Diagnosis:**

```bash
# Check pod events
kubectl describe pod pytorch-training-0 -n ml-training
# Events:
#   Warning  FailedScheduling  pod has unbound immediate PersistentVolumeClaims

# Check PVC status
kubectl get pvc -n ml-training
# NAME                           STATUS    VOLUME   CAPACITY
# checkpoints-pytorch-training-0 Pending

# Describe PVC
kubectl describe pvc checkpoints-pytorch-training-0 -n ml-training
# Events:
#   Normal   WaitForFirstConsumer   waiting for first consumer to be created before binding
```

**Solutions:**

1. **No StorageClass:** Create or specify existing StorageClass
```bash
kubectl get storageclass
kubectl patch pvc checkpoints-pytorch-training-0 -n ml-training -p '{"spec":{"storageClassName":"fast-ssd"}}'
```

2. **No PV Available:** Create PV or enable dynamic provisioning
```bash
# Check available PVs
kubectl get pv

# Enable dynamic provisioning
kubectl get storageclass fast-ssd -o yaml
# Ensure provisioner is configured
```

3. **Insufficient Storage:** Reduce PVC request or add storage
```bash
# Check node storage
kubectl get nodes -o custom-columns=NAME:.metadata.name,CAPACITY:.status.capacity.storage

# Reduce request (requires recreating PVC)
kubectl delete pvc checkpoints-pytorch-training-0 -n ml-training
# Edit StatefulSet to reduce storage request
kubectl edit statefulset pytorch-training -n ml-training
```

### Problem: Data Not Persisting Across Pod Restarts

**Diagnosis:**

```bash
# Write test data
kubectl exec mlflow-0 -n ml-tracking -- sh -c 'echo "test" > /checkpoints/test.txt'

# Delete pod
kubectl delete pod mlflow-0 -n ml-tracking

# Wait for recreation
kubectl wait --for=condition=ready pod/mlflow-0 -n ml-tracking --timeout=120s

# Check if data persists
kubectl exec mlflow-0 -n ml-tracking -- cat /checkpoints/test.txt
# If empty or error: data not persisting
```

**Solutions:**

1. **Check Volume Mount:**
```bash
kubectl describe pod mlflow-0 -n ml-tracking | grep -A 10 "Mounts:"
# Ensure volumeMount path matches where data is written
```

2. **Check PVC Binding:**
```bash
kubectl get pvc -n ml-tracking
# Ensure STATUS is "Bound"

# Check PV
kubectl get pv | grep ml-tracking
```

3. **Verify Correct StorageClass:**
```bash
kubectl get pvc checkpoints-mlflow-0 -n ml-tracking -o jsonpath='{.spec.storageClassName}'
# Ensure it's not "emptyDir" or ephemeral storage
```

### Problem: StatefulSet Update Stuck

**Symptoms:**
```bash
kubectl rollout status statefulset/mlflow -n ml-tracking
# Waiting for partition rollout to finish: 1 out of 3 new pods have been updated...
```

**Diagnosis:**

```bash
# Check pod status
kubectl get pods -n ml-tracking -l app=mlflow

# Describe stuck pod
kubectl describe pod mlflow-2 -n ml-tracking

# Check logs
kubectl logs mlflow-2 -n ml-tracking
kubectl logs mlflow-2 -n ml-tracking --previous  # Previous container instance
```

**Solutions:**

1. **Failed Readiness Probe:**
```bash
# Check probe status
kubectl get pods mlflow-2 -n ml-tracking -o jsonpath='{.status.conditions[?(@.type=="Ready")]}'

# Temporarily disable or adjust probe
kubectl edit statefulset mlflow -n ml-tracking
# Increase initialDelaySeconds or modify probe
```

2. **Force Pod Restart:**
```bash
# Delete stuck pod
kubectl delete pod mlflow-2 -n ml-tracking --grace-period=0 --force
```

3. **Rollback Update:**
```bash
kubectl rollout undo statefulset/mlflow -n ml-tracking
```

### Problem: Cannot Delete PVC (Stuck in Terminating)

**Diagnosis:**

```bash
kubectl get pvc -n ml-training
# NAME                           STATUS        VOLUME
# checkpoints-pytorch-training-3 Terminating   pvc-abc123

# Check finalizers
kubectl get pvc checkpoints-pytorch-training-3 -n ml-training -o yaml | grep finalizers -A 5
```

**Solutions:**

1. **Check if Pod Still Using PVC:**
```bash
# Find pods using PVC
kubectl get pods -n ml-training -o yaml | grep -B 5 checkpoints-pytorch-training-3

# Delete the pod first
kubectl delete pod pytorch-training-3 -n ml-training
```

2. **Remove Finalizers:**
```bash
kubectl patch pvc checkpoints-pytorch-training-3 -n ml-training -p '{"metadata":{"finalizers":null}}'
```

---

## Best Practices

### 1. Always Define Resource Limits

```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1"
  limits:
    memory: "4Gi"
    cpu: "2"
```

### 2. Use Init Containers for Setup

```yaml
initContainers:
- name: fix-permissions
  image: busybox
  command: ['sh', '-c', 'chown -R 999:999 /data && chmod 700 /data']
  volumeMounts:
  - name: data
    mountPath: /data
```

### 3. Configure Proper Health Probes

```yaml
livenessProbe:
  exec:
    command: ['/health-check.sh']
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 5
```

### 4. Use PVC Retention Policies

```yaml
persistentVolumeClaimRetentionPolicy:
  whenDeleted: Retain  # Keep data on StatefulSet deletion
  whenScaled: Delete   # Clean up on scale-down
```

### 5. Implement Graceful Shutdown

```yaml
lifecycle:
  preStop:
    exec:
      command:
      - /bin/sh
      - -c
      - |
        # Graceful shutdown script
        echo "Shutting down gracefully..."
        # Save state, close connections, etc.
        kill -TERM 1
        sleep 30
```

### 6. Use Appropriate Pod Management Policy

```yaml
# Use OrderedReady for databases
podManagementPolicy: OrderedReady

# Use Parallel for independent workers
podManagementPolicy: Parallel
```

### 7. Label Everything

```yaml
metadata:
  labels:
    app: mlflow
    component: tracking-server
    environment: production
    version: v2.8.0
```

### 8. Monitor Storage Usage

```bash
# Add sidecar for storage monitoring
containers:
- name: storage-monitor
  image: prom/node-exporter
  volumeMounts:
  - name: data
    mountPath: /data
    readOnly: true
```

---

## Summary

This guide covered:

1. **StatefulSets vs Deployments**: Understanding when to use each
2. **PersistentVolumes & PVCs**: Storage provisioning and consumption
3. **StorageClasses**: Dynamic provisioning and storage policies
4. **Ordered Startup**: Pod management policies and deployment strategies
5. **Headless Services**: Stable network identities for StatefulSet pods
6. **Scaling**: Manual and automatic scaling with PVC management
7. **ML Workloads**: Production patterns for distributed training, MLflow, and feature stores
8. **Advanced Topics**: Canary deployments, volume expansion, troubleshooting

**Key Takeaways:**
- Use StatefulSets for applications requiring stable identity and persistent storage
- Leverage StorageClasses for flexible, policy-driven storage provisioning
- Headless Services provide stable DNS names for inter-pod communication
- Properly configure PVC retention policies to avoid data loss
- Implement health probes and graceful shutdown for production reliability
- Choose appropriate pod management policies based on workload requirements

**Next Steps:**
- Practice deploying StatefulSets in your cluster
- Experiment with different StorageClasses and volume types
- Implement the ML workload examples
- Set up monitoring and alerting for StatefulSet health
- Explore advanced topics like volume snapshots and cloning
