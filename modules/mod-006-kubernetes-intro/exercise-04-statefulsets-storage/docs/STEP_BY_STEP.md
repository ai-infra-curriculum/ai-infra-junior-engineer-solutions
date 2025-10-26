# Step-by-Step Implementation Guide: StatefulSets & Storage

## Overview

Deploy stateful ML applications with persistent data! Learn StatefulSets for ordered deployments, PersistentVolumes for model storage, StatefulSet patterns for distributed ML training, and storage best practices.

**Time**: 2-3 hours | **Difficulty**: Intermediate to Advanced

---

## Learning Objectives

✅ Understand StatefulSets vs Deployments
✅ Create and manage PersistentVolumes (PV) and PersistentVolumeClaims (PVC)
✅ Use storage classes for dynamic provisioning
✅ Deploy stateful applications (databases, ML training clusters)
✅ Implement volume snapshots and backups
✅ Configure headless services for StatefulSets
✅ Handle StatefulSet scaling and updates
✅ Store ML models and training data persistently

---

## StatefulSets vs Deployments

| Feature | Deployment | StatefulSet |
|---------|-----------|-------------|
| Pod naming | Random | Ordered (pod-0, pod-1, ...) |
| Pod identity | Non-sticky | Sticky (same name after restart) |
| Scaling | Parallel | Sequential |
| Storage | Shared or ephemeral | Dedicated PVCs per pod |
| Use case | Stateless apps | Databases, distributed ML |

---

## Phase 1: PersistentVolumes Basics

### PersistentVolume (PV)

```yaml
# pv.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: ml-models-pv
  labels:
    type: local
spec:
  storageClassName: manual
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: "/mnt/data/ml-models"
  persistentVolumeReclaimPolicy: Retain
  volumeMode: Filesystem
```

**Access Modes**:
- `ReadWriteOnce` (RWO): Single node read-write
- `ReadOnlyMany` (ROX): Multiple nodes read-only
- `ReadWriteMany` (RWX): Multiple nodes read-write

**Reclaim Policies**:
- `Retain`: Manual cleanup required
- `Delete`: Auto-delete when claim deleted
- `Recycle`: Scrub and reuse (deprecated)

### PersistentVolumeClaim (PVC)

```yaml
# pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ml-models-claim
spec:
  storageClassName: manual
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
```

### Using PVC in Pod

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: model-server
spec:
  containers:
  - name: server
    image: ml-server:latest
    volumeMounts:
    - name: model-storage
      mountPath: /models
  volumes:
  - name: model-storage
    persistentVolumeClaim:
      claimName: ml-models-claim
```

### Deploy and Verify

```bash
# Create PV and PVC
kubectl apply -f pv.yaml
kubectl apply -f pvc.yaml

# Check status
kubectl get pv
kubectl get pvc

# PVC should be Bound to PV
kubectl describe pvc ml-models-claim
```

---

## Phase 2: Dynamic Provisioning with StorageClasses

### StorageClass

```yaml
# storageclass.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs  # AWS EBS
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
  encrypted: "true"
reclaimPolicy: Delete
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
```

**Common Provisioners**:
- AWS: `kubernetes.io/aws-ebs`
- GCP: `kubernetes.io/gce-pd`
- Azure: `kubernetes.io/azure-disk`
- Local: `kubernetes.io/no-provisioner`

### PVC with StorageClass

```yaml
# dynamic-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-data
spec:
  storageClassName: fast-ssd
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
```

```bash
# Apply PVC (PV auto-created)
kubectl apply -f dynamic-pvc.yaml

# Watch PV creation
kubectl get pv -w
```

---

## Phase 3: StatefulSet for ML Training Cluster

### Headless Service

```yaml
# headless-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-training
  labels:
    app: ml-training
spec:
  clusterIP: None  # Headless service
  selector:
    app: ml-training
  ports:
  - port: 2222
    name: dist-training
```

**Why Headless?**
- Each pod gets unique DNS: `pod-0.ml-training.default.svc.cluster.local`
- Required for StatefulSet pod discovery
- Enables peer-to-peer communication

### StatefulSet for Distributed Training

```yaml
# statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: ml-training
spec:
  serviceName: ml-training
  replicas: 3
  selector:
    matchLabels:
      app: ml-training
  template:
    metadata:
      labels:
        app: ml-training
    spec:
      containers:
      - name: trainer
        image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
        command:
        - python
        - /app/distributed_train.py
        env:
        - name: MASTER_ADDR
          value: "ml-training-0.ml-training.default.svc.cluster.local"
        - name: MASTER_PORT
          value: "2222"
        - name: WORLD_SIZE
          value: "3"
        - name: RANK
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        ports:
        - containerPort: 2222
          name: dist-training
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
        volumeMounts:
        - name: training-data
          mountPath: /data
        - name: model-checkpoints
          mountPath: /checkpoints
  volumeClaimTemplates:
  - metadata:
      name: training-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi
  - metadata:
      name: model-checkpoints
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 50Gi
```

### Deploy StatefulSet

```bash
# Create service first
kubectl apply -f headless-service.yaml

# Create StatefulSet
kubectl apply -f statefulset.yaml

# Watch pods come up sequentially
kubectl get pods -l app=ml-training -w

# Pods created in order: ml-training-0, ml-training-1, ml-training-2

# Check PVCs (auto-created from volumeClaimTemplates)
kubectl get pvc
```

---

## Phase 4: StatefulSet Operations

### Scaling

```bash
# Scale up (new pods added sequentially)
kubectl scale statefulset ml-training --replicas=5

# Scale down (pods removed in reverse order)
kubectl scale statefulset ml-training --replicas=2

# Watch scaling
kubectl get pods -l app=ml-training -w
```

### Updating

```yaml
# Update strategy in StatefulSet
spec:
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      partition: 2  # Only update pods >= partition
```

```bash
# Update image
kubectl set image statefulset/ml-training \
  trainer=pytorch/pytorch:2.1.0-cuda11.7-cudnn8-runtime

# Watch rollout (updates in reverse: pod-2, pod-1, pod-0)
kubectl rollout status statefulset/ml-training

# Rollback
kubectl rollout undo statefulset/ml-training
```

### Deleting

```bash
# Delete StatefulSet (keeps PVCs)
kubectl delete statefulset ml-training

# Delete StatefulSet and cascade delete pods
kubectl delete statefulset ml-training --cascade=orphan

# Delete PVCs manually
kubectl delete pvc training-data-ml-training-0
kubectl delete pvc training-data-ml-training-1
```

---

## Phase 5: PostgreSQL StatefulSet Example

### Complete PostgreSQL Deployment

```yaml
# postgres-statefulset.yaml
apiVersion: v1
kind: Service
metadata:
  name: postgres
spec:
  clusterIP: None
  selector:
    app: postgres
  ports:
  - port: 5432
    name: postgres
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: postgres
  replicas: 1
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
        ports:
        - containerPort: 5432
          name: postgres
        env:
        - name: POSTGRES_DB
          value: mldb
        - name: POSTGRES_USER
          value: mluser
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
  volumeClaimTemplates:
  - metadata:
      name: postgres-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 10Gi
```

```bash
# Create secret
kubectl create secret generic postgres-secret \
  --from-literal=password=mypassword

# Deploy PostgreSQL
kubectl apply -f postgres-statefulset.yaml

# Connect to database
kubectl exec -it postgres-0 -- psql -U mluser -d mldb
```

---

## Phase 6: Volume Snapshots

### VolumeSnapshot

```yaml
# snapshot.yaml
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: training-data-snapshot
spec:
  volumeSnapshotClassName: csi-snapclass
  source:
    persistentVolumeClaimName: training-data-ml-training-0
```

```bash
# Create snapshot
kubectl apply -f snapshot.yaml

# List snapshots
kubectl get volumesnapshot

# Restore from snapshot
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: restored-data
spec:
  dataSource:
    name: training-data-snapshot
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
```

---

## Phase 7: ML Model Storage Patterns

### Shared Model Store (ReadOnlyMany)

```yaml
# model-pv.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: shared-models
spec:
  capacity:
    storage: 50Gi
  accessModes:
    - ReadOnlyMany
  nfs:
    server: nfs-server.default.svc.cluster.local
    path: "/models"
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: shared-models-claim
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 50Gi
```

### Inference Deployment with Shared Models

```yaml
# inference-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-api
spec:
  replicas: 5
  selector:
    matchLabels:
      app: inference
  template:
    metadata:
      labels:
        app: inference
    spec:
      containers:
      - name: api
        image: ml-inference:latest
        volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: shared-models-claim
```

### Training with Checkpointing

```yaml
# training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: model-training
spec:
  template:
    spec:
      containers:
      - name: trainer
        image: ml-trainer:latest
        command:
        - python
        - train.py
        - --checkpoint-dir=/checkpoints
        volumeMounts:
        - name: checkpoints
          mountPath: /checkpoints
      volumes:
      - name: checkpoints
        persistentVolumeClaim:
          claimName: training-checkpoints
      restartPolicy: OnFailure
```

---

## Phase 8: Storage Best Practices

### Backup Strategy

```bash
#!/bin/bash
# backup-pvc.sh

PVC_NAME=$1
SNAPSHOT_NAME="backup-$(date +%Y%m%d-%H%M%S)"

cat <<EOF | kubectl apply -f -
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: $SNAPSHOT_NAME
spec:
  volumeSnapshotClassName: csi-snapclass
  source:
    persistentVolumeClaimName: $PVC_NAME
EOF

echo "Snapshot created: $SNAPSHOT_NAME"
```

### Monitor Storage Usage

```bash
# Check PVC usage
kubectl get pvc

# Describe PVC for details
kubectl describe pvc <pvc-name>

# Check disk usage in pod
kubectl exec <pod-name> -- df -h

# Check StorageClass
kubectl get storageclass
```

### Cleanup Orphaned PVCs

```bash
# Find PVCs not used by any pod
kubectl get pvc --all-namespaces -o json | \
  jq -r '.items[] | select(.status.phase=="Bound") |
  select(.metadata.annotations."volume.kubernetes.io/selected-node"==null) |
  "\(.metadata.namespace)/\(.metadata.name)"'
```

---

## Common Patterns

### InitContainer for Data Download

```yaml
spec:
  initContainers:
  - name: download-data
    image: amazon/aws-cli
    command:
    - aws
    - s3
    - sync
    - s3://ml-datasets/imagenet
    - /data
    volumeMounts:
    - name: training-data
      mountPath: /data
  containers:
  - name: trainer
    image: ml-trainer:latest
    volumeMounts:
    - name: training-data
      mountPath: /data
```

### Lifecycle Hooks for Cleanup

```yaml
spec:
  containers:
  - name: trainer
    lifecycle:
      preStop:
        exec:
          command:
          - sh
          - -c
          - |
            # Save checkpoint before termination
            python save_checkpoint.py /checkpoints/final.pth
            # Upload to S3
            aws s3 cp /checkpoints/final.pth s3://ml-checkpoints/
```

---

## Troubleshooting

```bash
# PVC stuck in Pending
kubectl describe pvc <pvc-name>
# Check: StorageClass exists, provisioner working, sufficient capacity

# PVC stuck in Lost
kubectl get pv
# PV deleted before PVC - manual intervention needed

# Expand PVC (if allowVolumeExpansion: true)
kubectl patch pvc <pvc-name> -p '{"spec":{"resources":{"requests":{"storage":"200Gi"}}}}'
kubectl get pvc <pvc-name> -w

# Force delete stuck PVC
kubectl patch pvc <pvc-name> -p '{"metadata":{"finalizers":null}}'
```

---

## Verification Checklist

✅ StatefulSet pods created in order
✅ Each pod has unique PVC
✅ Pods maintain identity after restart
✅ Headless service provides stable network IDs
✅ Storage persists across pod deletions
✅ Backups/snapshots configured
✅ Resource limits set appropriately
✅ Monitoring storage usage
✅ Tested scaling up/down
✅ Tested rollback procedures

---

**StatefulSets and Storage mastered!** 💾

**Next Exercise**: ConfigMaps and Secrets
