# Step-by-Step Guide: StatefulSets and Persistent Storage

This guide walks you through every step of deploying and managing stateful applications in Kubernetes, from basic concepts to production-ready configurations.

## Table of Contents

1. [Understanding StatefulSets](#step-1-understanding-statefulsets)
2. [Deploying PostgreSQL StatefulSet](#step-2-deploying-postgresql-statefulset)
3. [Testing Data Persistence](#step-3-testing-data-persistence)
4. [Deploying Redis StatefulSet](#step-4-deploying-redis-statefulset)
5. [Exploring Volume Types](#step-5-exploring-volume-types)
6. [StatefulSet vs Deployment](#step-6-statefulset-vs-deployment)
7. [Scaling and Updating](#step-7-scaling-and-updating)
8. [Troubleshooting](#step-8-troubleshooting)

---

## Step 1: Understanding StatefulSets

### What is a StatefulSet?

A StatefulSet is a Kubernetes workload controller designed for stateful applications that require:
- **Stable, unique network identifiers**
- **Stable, persistent storage**
- **Ordered, graceful deployment and scaling**
- **Ordered, automated rolling updates**

### Key Differences from Deployments

| Aspect | Deployment | StatefulSet |
|--------|------------|-------------|
| Pod names | Random (nginx-7d8f9c5b4-x7k2m) | Ordered (nginx-0, nginx-1, nginx-2) |
| DNS names | No stable DNS | Stable DNS per pod |
| Storage | Shared or none | Dedicated per pod |
| Creation order | Parallel | Sequential |
| Update order | Random | Reverse sequential |
| Use cases | Stateless apps | Databases, queues, distributed systems |

### When to Use StatefulSets

✅ **Use StatefulSets for**:
- Databases (PostgreSQL, MySQL, MongoDB)
- Message queues (Kafka, RabbitMQ)
- Distributed systems (etcd, ZooKeeper, Consul)
- Applications requiring stable network identity
- Applications with per-instance storage

❌ **Don't use StatefulSets for**:
- Stateless web applications
- REST APIs without state
- Microservices
- Worker processes without state

---

## Step 2: Deploying PostgreSQL StatefulSet

### 2.1 Create Namespace

```bash
kubectl create namespace statefulset-demo
```

Or apply the namespace manifest:

```bash
kubectl apply -f manifests/01-namespace.yaml
```

### 2.2 Understand the Components

The PostgreSQL StatefulSet includes:

1. **ConfigMap** - PostgreSQL configuration
2. **Secret** - Database password
3. **Headless Service** - Stable network identity
4. **Regular Service** - Load-balanced access
5. **StatefulSet** - The application deployment

Let's examine each:

#### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: postgres-config
data:
  POSTGRES_DB: mydatabase
  POSTGRES_USER: myuser
  postgresql.conf: |
    max_connections = 100
    shared_buffers = 128MB
```

This stores non-sensitive configuration.

#### Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: postgres-secret
type: Opaque
data:
  POSTGRES_PASSWORD: bXlzZWNyZXRwYXNzd29yZA==  # Base64 encoded
```

Stores sensitive data (password).

#### Headless Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: postgres-headless
spec:
  clusterIP: None  # This makes it headless!
  selector:
    app: postgresql
```

**Why headless?** Each pod gets a stable DNS name:
- `postgres-0.postgres-headless.statefulset-demo.svc.cluster.local`
- `postgres-1.postgres-headless.statefulset-demo.svc.cluster.local`
- `postgres-2.postgres-headless.statefulset-demo.svc.cluster.local`

### 2.3 Deploy PostgreSQL

```bash
kubectl apply -f manifests/03-postgresql-statefulset.yaml
```

### 2.4 Watch Pods Start in Order

```bash
kubectl get pods -n statefulset-demo -l app=postgresql -w
```

**What you'll see**:
1. `postgres-0` starts first
2. Waits until `postgres-0` is Ready
3. Then `postgres-1` starts
4. Waits until `postgres-1` is Ready
5. Then `postgres-2` starts

Press `Ctrl+C` to stop watching.

### 2.5 Verify Deployment

```bash
# Check StatefulSet
kubectl get statefulset postgres -n statefulset-demo

# Check pods
kubectl get pods -n statefulset-demo -l app=postgresql

# Check services
kubectl get svc -n statefulset-demo

# Check PVCs (one per pod!)
kubectl get pvc -n statefulset-demo
```

**Expected output**:
```
NAME                   STATUS   VOLUME                                     CAPACITY
postgres-data-postgres-0   Bound    pvc-xxxxx   1Gi
postgres-data-postgres-1   Bound    pvc-yyyyy   1Gi
postgres-data-postgres-2   Bound    pvc-zzzzz   1Gi
```

### 2.6 Connect to PostgreSQL

```bash
# Connect to postgres-0
kubectl exec -it postgres-0 -n statefulset-demo -- psql -U myuser -d mydatabase

# Inside psql:
mydatabase=# \dt  # List tables (empty initially)
mydatabase=# \l   # List databases
mydatabase=# \q   # Quit
```

### 2.7 Create Test Data

```bash
kubectl exec -it postgres-0 -n statefulset-demo -- psql -U myuser -d mydatabase << 'EOF'
-- Create a test table
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    department VARCHAR(50),
    hired_date DATE DEFAULT CURRENT_DATE
);

-- Insert some data
INSERT INTO employees (name, department) VALUES
    ('Alice Johnson', 'Engineering'),
    ('Bob Smith', 'Marketing'),
    ('Carol White', 'Engineering'),
    ('David Brown', 'Sales');

-- Query the data
SELECT * FROM employees;
EOF
```

### 2.8 Test Stable DNS Names

```bash
# Run a temporary pod to test DNS
kubectl run -it dns-test --rm --image=busybox -n statefulset-demo -- sh

# Inside the pod:
nslookup postgres-0.postgres-headless
nslookup postgres-1.postgres-headless
nslookup postgres-2.postgres-headless

# Try to connect (if psql was available)
# psql -h postgres-0.postgres-headless.statefulset-demo.svc.cluster.local -U myuser
```

---

## Step 3: Testing Data Persistence

### 3.1 Write Data to a Pod

```bash
kubectl exec -it postgres-0 -n statefulset-demo -- psql -U myuser -d mydatabase << 'EOF'
CREATE TABLE persistence_test (
    id SERIAL PRIMARY KEY,
    message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO persistence_test (message) VALUES
    ('This data must survive pod deletion'),
    ('Testing persistence in StatefulSet'),
    ('Pod: postgres-0');

SELECT * FROM persistence_test;
EOF
```

### 3.2 Note the Data

```bash
# Save the output to verify later
kubectl exec -it postgres-0 -n statefulset-demo -- \
  psql -U myuser -d mydatabase -c "SELECT * FROM persistence_test;" \
  > /tmp/before-deletion.txt

cat /tmp/before-deletion.txt
```

### 3.3 Delete the Pod

```bash
# Delete postgres-0
kubectl delete pod postgres-0 -n statefulset-demo

# Immediately watch it recreate
kubectl get pods -n statefulset-demo -l app=postgresql -w
```

**What happens**:
1. Pod `postgres-0` is terminated
2. Kubernetes automatically creates a new `postgres-0`
3. New pod has same name and network identity
4. **Same PVC is attached** (postgres-data-postgres-0)

### 3.4 Verify Data Persisted

```bash
# Wait for pod to be ready
kubectl wait --for=condition=ready pod/postgres-0 -n statefulset-demo --timeout=120s

# Check data is still there
kubectl exec -it postgres-0 -n statefulset-demo -- \
  psql -U myuser -d mydatabase -c "SELECT * FROM persistence_test;"

# Compare with original
diff /tmp/before-deletion.txt <(kubectl exec -it postgres-0 -n statefulset-demo -- \
  psql -U myuser -d mydatabase -c "SELECT * FROM persistence_test;")
```

**Result**: Data is exactly the same! ✅

### 3.5 Understand Why It Persisted

```bash
# Check PVC for postgres-0
kubectl get pvc postgres-data-postgres-0 -n statefulset-demo

# Describe the PVC to see which PV it's bound to
kubectl describe pvc postgres-data-postgres-0 -n statefulset-demo

# Check the PV
PV_NAME=$(kubectl get pvc postgres-data-postgres-0 -n statefulset-demo -o jsonpath='{.spec.volumeName}')
kubectl describe pv $PV_NAME
```

**Key insight**: The PVC (and its underlying PV) remains even when the pod is deleted. When a new pod with the same name is created, it attaches to the same PVC.

---

## Step 4: Deploying Redis StatefulSet

### 4.1 Deploy Redis

```bash
kubectl apply -f manifests/05-redis-statefulset.yaml
```

### 4.2 Watch Ordered Creation

```bash
kubectl get pods -n statefulset-demo -l app=redis -w
```

Again, you'll see sequential creation: redis-0 → redis-1 → redis-2

### 4.3 Connect to Redis

```bash
kubectl exec -it redis-0 -n statefulset-demo -- redis-cli

# Inside redis-cli:
127.0.0.1:6379> PING
PONG

127.0.0.1:6379> SET mykey "Hello from redis-0"
OK

127.0.0.1:6379> GET mykey
"Hello from redis-0"

127.0.0.1:6379> SAVE
OK

127.0.0.1:6379> exit
```

### 4.4 Test Redis Persistence

```bash
# Write data
kubectl exec -it redis-0 -n statefulset-demo -- redis-cli SET test:persistence "This data must persist"
kubectl exec -it redis-0 -n statefulset-demo -- redis-cli SAVE

# Delete pod
kubectl delete pod redis-0 -n statefulset-demo

# Wait for recreation
kubectl wait --for=condition=ready pod/redis-0 -n statefulset-demo --timeout=120s

# Verify data persisted
kubectl exec -it redis-0 -n statefulset-demo -- redis-cli GET test:persistence
# Output: "This data must persist"
```

### 4.5 Access Different Redis Instances

```bash
# Write different data to each instance
kubectl exec -it redis-0 -n statefulset-demo -- redis-cli SET pod:id "redis-0"
kubectl exec -it redis-1 -n statefulset-demo -- redis-cli SET pod:id "redis-1"
kubectl exec -it redis-2 -n statefulset-demo -- redis-cli SET pod:id "redis-2"

# Read back from each
kubectl exec -it redis-0 -n statefulset-demo -- redis-cli GET pod:id  # "redis-0"
kubectl exec -it redis-1 -n statefulset-demo -- redis-cli GET pod:id  # "redis-1"
kubectl exec -it redis-2 -n statefulset-demo -- redis-cli GET pod:id  # "redis-2"
```

Each pod has its own independent storage!

---

## Step 5: Exploring Volume Types

### 5.1 Deploy Volume Examples

```bash
kubectl apply -f manifests/04-volume-types.yaml
```

### 5.2 Examine emptyDir Volume

```bash
# Check the pod
kubectl get pod emptydir-example -n statefulset-demo

# View logs from writer container
kubectl logs emptydir-example -n statefulset-demo -c writer

# View logs from reader container
kubectl logs emptydir-example -n statefulset-demo -c reader

# Exec into the pod
kubectl exec -it emptydir-example -n statefulset-demo -c writer -- sh
ls -la /data/
cat /data/timestamp.txt
exit
```

**Delete and see data is lost**:
```bash
kubectl delete pod emptydir-example -n statefulset-demo
kubectl apply -f manifests/04-volume-types.yaml
# Data is gone!
```

### 5.3 Examine ConfigMap as Volume

```bash
# Check pod
kubectl get pod configmap-volume-example -n statefulset-demo

# View mounted config files
kubectl exec configmap-volume-example -n statefulset-demo -- ls -la /config/
kubectl exec configmap-volume-example -n statefulset-demo -- cat /config/app.properties
kubectl exec configmap-volume-example -n statefulset-demo -- cat /config/nginx.conf
```

### 5.4 Examine Secret as Volume

```bash
# View mounted secrets (automatically decoded!)
kubectl exec secret-volume-example -n statefulset-demo -- ls -la /secrets/
kubectl exec secret-volume-example -n statefulset-demo -- cat /secrets/database-password
kubectl exec secret-volume-example -n statefulset-demo -- cat /secrets/api-key
```

### 5.5 Examine PVC Example

```bash
# Check PV and PVC
kubectl get pv manual-pv-001
kubectl get pvc manual-pvc-001 -n statefulset-demo

# Check pod using the PVC
kubectl get pod pvc-example -n statefulset-demo

# Write data to PVC
kubectl exec pvc-example -n statefulset-demo -- sh -c 'echo "Persistent data in PVC" > /usr/share/nginx/html/index.html'

# Read data
kubectl exec pvc-example -n statefulset-demo -- cat /usr/share/nginx/html/index.html

# Delete pod
kubectl delete pod pvc-example -n statefulset-demo

# Recreate pod
kubectl apply -f manifests/04-volume-types.yaml

# Data persists!
kubectl exec pvc-example -n statefulset-demo -- cat /usr/share/nginx/html/index.html
```

---

## Step 6: StatefulSet vs Deployment

### 6.1 Deploy Both

```bash
kubectl apply -f examples/statefulset-vs-deployment.yaml
```

### 6.2 Compare Pod Names

```bash
# Deployment pods (random names)
kubectl get pods -n statefulset-demo -l type=deployment-example

# Output:
# nginx-deployment-7d8f9c5b4-x7k2m
# nginx-deployment-7d8f9c5b4-p9q3n
# nginx-deployment-7d8f9c5b4-r5t8w

# StatefulSet pods (ordered names)
kubectl get pods -n statefulset-demo -l type=statefulset-example

# Output:
# nginx-statefulset-0
# nginx-statefulset-1
# nginx-statefulset-2
```

### 6.3 Test DNS Resolution

```bash
kubectl run -it dns-test --rm --image=busybox -n statefulset-demo -- sh

# Try Deployment pod (fails - random name changes)
nslookup nginx-deployment-7d8f9c5b4-x7k2m.nginx-deployment-svc
# Error: can't resolve

# Try StatefulSet pod (works!)
nslookup nginx-statefulset-0.nginx-statefulset-svc
# Returns IP address

nslookup nginx-statefulset-1.nginx-statefulset-svc
# Returns different IP address
```

### 6.4 Compare Storage

```bash
# Check PVCs
kubectl get pvc -n statefulset-demo

# Deployment: No PVCs
# StatefulSet: One PVC per pod (www-nginx-statefulset-0, www-nginx-statefulset-1, www-nginx-statefulset-2)
```

### 6.5 Test Independent Storage

```bash
# Write different data to each StatefulSet pod
kubectl exec nginx-statefulset-0 -n statefulset-demo -- sh -c 'echo "Data from pod 0" > /usr/share/nginx/html/pod-data.txt'
kubectl exec nginx-statefulset-1 -n statefulset-demo -- sh -c 'echo "Data from pod 1" > /usr/share/nginx/html/pod-data.txt'
kubectl exec nginx-statefulset-2 -n statefulset-demo -- sh -c 'echo "Data from pod 2" > /usr/share/nginx/html/pod-data.txt'

# Read back from each
kubectl exec nginx-statefulset-0 -n statefulset-demo -- cat /usr/share/nginx/html/pod-data.txt
kubectl exec nginx-statefulset-1 -n statefulset-demo -- cat /usr/share/nginx/html/pod-data.txt
kubectl exec nginx-statefulset-2 -n statefulset-demo -- cat /usr/share/nginx/html/pod-data.txt

# Each has its own data!
```

---

## Step 7: Scaling and Updating

### 7.1 Scale Up

```bash
# Current replicas
kubectl get statefulset redis -n statefulset-demo

# Scale to 5 replicas
kubectl scale statefulset redis --replicas=5 -n statefulset-demo

# Watch ordered creation
kubectl get pods -n statefulset-demo -l app=redis -w

# You'll see:
# redis-3 created (waits until Ready)
# redis-4 created (waits until Ready)
```

### 7.2 Verify New Pods Get PVCs

```bash
# Check PVCs
kubectl get pvc -n statefulset-demo -l app=redis

# Should see:
# redis-data-redis-0
# redis-data-redis-1
# redis-data-redis-2
# redis-data-redis-3 (new!)
# redis-data-redis-4 (new!)
```

### 7.3 Scale Down

```bash
# Scale to 2 replicas
kubectl scale statefulset redis --replicas=2 -n statefulset-demo

# Watch ordered deletion (reverse order!)
kubectl get pods -n statefulset-demo -l app=redis -w

# You'll see:
# redis-4 terminated
# redis-3 terminated
# redis-2 terminated
# Only redis-0 and redis-1 remain
```

### 7.4 Check PVCs After Scale Down

```bash
kubectl get pvc -n statefulset-demo -l app=redis

# PVCs for redis-2, redis-3, redis-4 STILL EXIST!
# They're retained in case you scale back up
```

### 7.5 Scale Back Up (PVCs Reused!)

```bash
# Scale back to 5
kubectl scale statefulset redis --replicas=5 -n statefulset-demo

# Watch pods come back
kubectl get pods -n statefulset-demo -l app=redis -w

# Check if PVCs were reused
kubectl get pvc -n statefulset-demo -l app=redis

# Same PVCs are reattached!
# Data from before scale-down persists
```

### 7.6 Rolling Update

```bash
# Update Redis image
kubectl set image statefulset/redis redis=redis:7.2-alpine -n statefulset-demo

# Watch rollout (happens in REVERSE order)
kubectl rollout status statefulset/redis -n statefulset-demo

# Monitor pods being updated
kubectl get pods -n statefulset-demo -l app=redis -w

# Order: redis-2 updated, then redis-1, then redis-0
```

### 7.7 Partitioned Update (Canary)

```bash
# Set partition to 2 (only update pods >= 2)
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

# Update image (only affects redis-2 and higher)
kubectl set image statefulset/redis redis=redis:7.0-alpine -n statefulset-demo

# Check pod images
kubectl get pods -n statefulset-demo -l app=redis -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[0].image}{"\n"}{end}'

# redis-0 and redis-1: old image
# redis-2+: new image

# Roll out to all (set partition to 0)
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

### 7.8 Rollback

```bash
# View rollout history
kubectl rollout history statefulset/redis -n statefulset-demo

# Rollback to previous version
kubectl rollout undo statefulset/redis -n statefulset-demo

# Rollback to specific revision
kubectl rollout undo statefulset/redis --to-revision=1 -n statefulset-demo
```

---

## Step 8: Troubleshooting

### 8.1 Pod Stuck in Pending

**Symptom**:
```bash
kubectl get pods -n statefulset-demo
# postgres-0   0/1     Pending   0          2m
```

**Diagnose**:
```bash
kubectl describe pod postgres-0 -n statefulset-demo

# Look for events:
# Warning  FailedScheduling  ... 0/1 nodes are available: 1 pod has unbound immediate PersistentVolumeClaims
```

**Solution**:
```bash
# Check PVC status
kubectl get pvc -n statefulset-demo

# If PVC is Pending:
kubectl describe pvc postgres-data-postgres-0 -n statefulset-demo

# Common issues:
# 1. No PV available
# 2. StorageClass doesn't exist
# 3. No storage provisioner

# Fix: Create a PV or check StorageClass
kubectl get storageclass
kubectl get pv
```

### 8.2 Data Not Persisting

**Diagnose**:
```bash
# Check if PVC is mounted
kubectl describe pod postgres-0 -n statefulset-demo | grep -A 10 "Volumes:"

# Check PVC binding
kubectl get pvc postgres-data-postgres-0 -n statefulset-demo

# Check actual mount inside pod
kubectl exec postgres-0 -n statefulset-demo -- df -h
kubectl exec postgres-0 -n statefulset-demo -- mount | grep postgres
```

**Solution**:
```bash
# Verify data directory
kubectl exec postgres-0 -n statefulset-demo -- ls -la /var/lib/postgresql/data/

# Check permissions
kubectl exec postgres-0 -n statefulset-demo -- ls -la /var/lib/postgresql/data/pgdata/

# If permissions wrong, delete pod to run init container again
kubectl delete pod postgres-0 -n statefulset-demo
```

### 8.3 Pod Won't Update

**Symptom**:
```bash
kubectl rollout status statefulset/redis -n statefulset-demo
# Waiting for statefulset rolling update to complete 1 out of 3 new pods have been updated...
# (stuck)
```

**Diagnose**:
```bash
# Check pod status
kubectl get pods -n statefulset-demo -l app=redis

# Describe the problematic pod
kubectl describe pod redis-2 -n statefulset-demo

# Check logs
kubectl logs redis-2 -n statefulset-demo
kubectl logs redis-2 -n statefulset-demo --previous
```

**Solution**:
```bash
# Force delete stuck pod
kubectl delete pod redis-2 -n statefulset-demo --force --grace-period=0

# Or rollback
kubectl rollout undo statefulset/redis -n statefulset-demo
```

### 8.4 Can't Delete PVC

**Symptom**:
```bash
kubectl delete pvc postgres-data-postgres-0 -n statefulset-demo
# Hangs...
```

**Diagnose**:
```bash
# Check if PVC has finalizers
kubectl get pvc postgres-data-postgres-0 -n statefulset-demo -o yaml | grep -A 5 finalizers

# Check if pod is using it
kubectl get pods -n statefulset-demo -o yaml | grep -A 10 persistentVolumeClaim
```

**Solution**:
```bash
# Delete pod first
kubectl delete pod postgres-0 -n statefulset-demo

# Then delete PVC
kubectl delete pvc postgres-data-postgres-0 -n statefulset-demo

# If still stuck, remove finalizer (careful!)
kubectl patch pvc postgres-data-postgres-0 -n statefulset-demo -p '{"metadata":{"finalizers":null}}'
```

---

## Summary

Congratulations! You've learned:

✅ What StatefulSets are and when to use them
✅ How to deploy databases (PostgreSQL, Redis)
✅ How data persists across pod restarts
✅ Different volume types and their uses
✅ The differences between StatefulSets and Deployments
✅ How to scale StatefulSets
✅ How to perform rolling updates and rollbacks
✅ How to troubleshoot common issues

### Next Steps

1. Practice deploying your own stateful application
2. Experiment with different StorageClasses
3. Try setting up replication between PostgreSQL instances
4. Explore Kubernetes Operators for managed databases
5. Continue to Exercise 05: ConfigMaps and Secrets

### Key Takeaways

1. **Use StatefulSets for stateful applications** - databases, queues, distributed systems
2. **PVCs survive pod deletion** - data persists
3. **Headless services provide stable DNS** - predictable networking
4. **Scaling is ordered** - sequential creation and deletion
5. **Updates are safer** - reverse order, partitioned updates possible
6. **Test persistence** - always verify data survives pod restarts
