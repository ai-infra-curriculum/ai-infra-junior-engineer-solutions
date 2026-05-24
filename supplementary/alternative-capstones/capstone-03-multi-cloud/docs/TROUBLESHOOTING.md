# Capstone 03 (Multi-Cloud) — Troubleshooting

Common issues junior engineers hit when working through the multi-cloud capstone. Each entry: symptom → diagnosis → fix.

---

## Cloud account / credentials

### 1. "InvalidClientTokenId" or "AccessDenied" on AWS commands

**Symptom**

```text
An error occurred (InvalidClientTokenId) when calling the GetCallerIdentity operation: The security token included in the request is invalid.
```

**Diagnosis**
- Expired session credentials, wrong profile, or copy-pasted long-term key with extra whitespace.

**Fix**
- `aws sts get-caller-identity` — confirms whether credentials work at all.
- `aws configure list` — shows which profile is active and where credentials come from.
- If using SSO: `aws sso login --profile <profile>`.
- If using static keys: re-export `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` carefully (no trailing newlines).

---

### 2. GCP "permission denied" but you're an Owner

**Symptom**
- Owner role in IAM but commands return `Permission ... denied`.

**Diagnosis**
- Application Default Credentials (ADC) not set up for the gcloud user, or different account is active.

**Fix**
- `gcloud auth list` — confirms active account.
- `gcloud auth application-default login` — refreshes ADC.
- Confirm the active project: `gcloud config get-value project`.

---

### 3. Azure CLI says "subscription not found"

**Symptom**
- `az ...` commands report the subscription doesn't exist.

**Diagnosis**
- Logged in but no default subscription selected, or wrong tenant.

**Fix**
- `az account list -o table` — lists available subscriptions.
- `az account set --subscription <id>` — set the active one.
- If you don't see expected subscriptions: `az login --tenant <tenant-id>`.

---

## Terraform across clouds

### 4. "Backend configuration changed" forces re-init

**Symptom**
- After switching state backends or workspaces:
```text
Error: Backend configuration changed
```

**Fix**
- `terraform init -reconfigure` (you accept that local state will be discarded).
- Or `terraform init -migrate-state` if you want to copy state to the new backend.

---

### 5. Provider version mismatch

**Symptom**
- Provider plugin fails to load or reports unknown attribute.

**Fix**
- Pin provider versions in `versions.tf`. Don't rely on `~>` for production-style work; pin to exact versions.
- Run `terraform init -upgrade` only when intentionally upgrading.

---

### 6. State file locking

**Symptom**
- `terraform apply` hangs or errors with "state locked".

**Diagnosis**
- A previous `apply` was killed (Ctrl-C) and left the lock in DynamoDB / GCS / Azure backend.

**Fix**
- `terraform force-unlock <LOCK_ID>` (the lock ID is in the error message).
- Only use force-unlock when you're sure no other process is running.

---

## Networking + DNS

### 7. Health-check shows the service "unhealthy" across clouds

**Symptom**
- AWS Route 53 / GCP global LB / Azure Traffic Manager marks an endpoint unhealthy even though the app appears to work.

**Diagnosis**
- The health-check probes from public IPs; security groups / firewalls / NSGs may be blocking them.
- Health-check path returns 200 but with a body that fails the configured string-match.
- TLS certificate doesn't include the health-check hostname.

**Fix**
- Allow the health-check source IP ranges in each cloud's firewall.
- Use a simple `GET /healthz` returning `200 OK` with `text/plain` body `ok`.
- Use a wildcard cert or include all hostnames in SANs.

---

### 8. DNS propagation delays

**Symptom**
- DNS change shows in the console but `dig`/`nslookup` returns the old value.

**Fix**
- Wait. TTLs are real. If the TTL was 3600s, expect up to an hour.
- Test from multiple resolvers: `dig @8.8.8.8 example.com`, `dig @1.1.1.1 example.com`.
- For testing only: `dig +trace example.com` reveals the actual resolution path.

---

## Cost / billing surprises

### 9. NAT Gateway bill is huge

**Symptom**
- AWS NAT Gateway shows hundreds of $/day.

**Diagnosis**
- Cross-AZ traffic through NAT (data-transfer charges), or one workload pulling large amounts of data through NAT.

**Fix**
- Use VPC endpoints (Gateway endpoints are free for S3/DynamoDB) instead of routing through NAT.
- Identify the chatty workload via VPC Flow Logs + Athena.
- For deployments that don't need internet egress, put them in private subnets without NAT.

---

### 10. Cross-cloud egress costs

**Symptom**
- Egress costs on AWS or GCP are 5-10x what you expected.

**Diagnosis**
- A workload in AWS is calling a service in GCP (or vice versa), and you're paying egress on both sides.

**Fix**
- Use dedicated interconnects (Direct Connect, Cloud Interconnect, ExpressRoute) for sustained cross-cloud traffic.
- Or co-locate services within a single cloud and replicate data using batch jobs at low-rate times.

---

## Kubernetes / cluster issues

### 11. Cluster nodes are NotReady

**Symptom**
- `kubectl get nodes` shows `NotReady` for one or more nodes.

**Diagnosis**
- `kubectl describe node <node>` — look at conditions and events.
- Common causes: CNI plugin not running, kubelet failing to start, full disk.

**Fix**
- Check CNI: `kubectl get pods -n kube-system -l k8s-app=<cni-name>`.
- Check kubelet logs on the node (cloud-specific — usually via SSM/SSH or the cloud's serial console).
- Check disk: `kubectl describe node` shows disk pressure conditions.

---

### 12. Pods in Pending forever

**Symptom**
- `kubectl get pods` shows pods stuck Pending.

**Diagnosis**
- `kubectl describe pod <pod>` — bottom of output explains why.
- Common causes: insufficient resources on any node, PVC can't be provisioned, image pull errors, scheduling constraints (taints/tolerations).

**Fix**
- Scale up node pool if cluster is full.
- Verify storage class exists and has working dynamic provisioning.
- Add tolerations or remove taints if pods are blocked from scheduling.

---

## Cross-cutting

### 13. CI fails for one cloud but passes for others

**Symptom**
- Identical pipeline succeeds on AWS, fails on GCP (or vice versa).

**Diagnosis**
- Cloud-specific behavior differences: instance types, IAM service names, default storage classes, regional availability.

**Fix**
- Run the failing step locally with the same env (use the cloud's CLI in your local shell).
- Don't assume CLI flags or behavior are consistent across clouds; check each cloud's docs.

---

### 14. Secret leaks in logs or repo

**Symptom**
- A secret you set as an env var shows up in `terraform plan` output or in CI logs.

**Fix**
- Mark Terraform variables `sensitive = true`.
- Use GitHub Actions / GitLab CI / Cloud Build secret masking.
- Rotate the leaked secret immediately. Don't delete it from logs and call it fixed — assume it's compromised.

---

## When all else fails

1. Re-read the IMPLEMENTATION_GUIDE.md from the top. Most issues come from skipping a setup step.
2. Compare your config to the `examples/` directory.
3. Open the issue on GitHub with the exact error message and the surrounding context.
4. For cloud-specific bugs, check each cloud's status page first; outages happen.

See also: [docs/IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for the build walkthrough, and the parent capstone [README.md](../README.md) for the full context.
