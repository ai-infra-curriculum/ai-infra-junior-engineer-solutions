# SOLUTION — Kubernetes Model Serving

> Read this *after* you have deployed the API yourself. This file
> explains *why* the Kubernetes manifests are shaped the way they
> are, what we deliberately rejected, and what production failures
> each piece prevents. `SOLUTION_GUIDE.md` is the how-to-run-it
> walkthrough; this is the design rationale.

## What this project is really teaching

Project 01 produced a working API. This project produces an API
that **stays working** when:

- A node dies in the middle of a request.
- Traffic 10x's without warning.
- A bad deploy goes out and needs to roll back in under five minutes.
- A noisy neighbor pod tries to starve yours of CPU.
- The model takes 45 seconds to load and the readiness probe is set
  at 10 seconds.

Kubernetes is not magic — it is a set of building blocks that solve
those specific failures *if you configure them right*. Every choice
below traces back to one of them.

## Architectural decisions and *why*

### Decision 1: Liveness and readiness probes point at different endpoints

The Deployment's `livenessProbe` hits `/health`; the `readinessProbe`
hits `/info` (or `/ready`). Two probes, two semantics:

- **Liveness failure → pod killed.** This must only fire when the
  process is genuinely stuck. A slow model load is *not* a liveness
  failure; it's a not-ready-yet state.
- **Readiness failure → pod removed from the Service endpoint list,
  but kept running.** Traffic stops; the pod has time to recover.

Beginners almost always conflate these and end up with kill-loops
during deploys. The split is the single most consequential decision
in the file.

### Decision 2: `maxSurge: 1` and `maxUnavailable: 0` on the rolling update

Default Kubernetes rolling update tolerates losing 25% of replicas
mid-deploy. That's appropriate for a fleet of 100 pods, but for our
3-replica deployment it means a deploy can drop us to 2 pods even
on a healthy day. `maxUnavailable: 0` keeps us at full capacity
throughout the deploy at the cost of one extra ephemeral pod.

The trade-off is explicit: slightly higher peak cost during deploys
in exchange for no observable capacity dip. For a customer-facing
service this is almost always correct.

### Decision 3: Resource `requests` < `limits`, with realistic numbers

`requests: cpu=500m memory=1Gi`, `limits: cpu=1000m memory=2Gi`.

- **Requests** drive the scheduler — they're the floor the kubelet
  guarantees. Set them low and pods overflow each other's CPU.
  Set them too high and the cluster looks full when it isn't.
- **Limits** drive the kernel cgroup — they're the ceiling. Hit
  the memory limit and you get OOMKilled; hit the CPU limit and
  you get throttled.

The 1:2 ratio (requests:limits) gives headroom for traffic spikes
without letting any single pod monopolize a node. The numbers come
from actually profiling the API with `kubectl top pod` under load
— not from guessing.

**Anti-pattern to avoid**: setting requests == limits ("Guaranteed
QoS class") for a workload that has predictable steady-state usage
but bursty peaks. You'll either pay for headroom you don't use, or
OOM during legitimate spikes.

### Decision 4: HPA on CPU, not on request rate

The HorizontalPodAutoscaler scales on `cpu_utilization > 70%`.
Several alternatives exist:

- Custom metric (requests/sec) via the Prometheus adapter.
- Memory utilization.
- Pod-level inference queue depth.

CPU works because the inference path is CPU-bound. Request rate is
more semantically aligned with the workload but adds operational
complexity (Prometheus adapter, custom metric definitions). At
Project 02's complexity tier, CPU is the right starting point.

The senior-engineer track's `mod-104/ex-04-k8s-cluster-autoscaler`
exercise covers the custom-metric scaling path.

### Decision 5: SecurityContext with non-root user + read-only root filesystem

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  readOnlyRootFilesystem: true
  capabilities:
    drop: ["ALL"]
```

- `runAsNonRoot: true` is enforced by admission; the container
  *cannot* start as root even if the Dockerfile drops the USER.
- `readOnlyRootFilesystem: true` blocks an attacker who lands in
  the container from writing payloads to disk.
- Dropping all capabilities is the explicit-allow posture: add
  back only what's needed (here, nothing).

Beginners skip this section because everything works without it.
Production clusters refuse to admit pods without it. Setting it up
once now means it stays correct forever.

### Decision 6: Pod anti-affinity across nodes

The Deployment includes `podAntiAffinity` requiring replicas on
different nodes. Without it, the scheduler is free to pack all
three pods onto one node — and one node failure becomes a full
outage.

For a 3-replica deployment in a 3-AZ cluster, this turns the
"survive an AZ loss" property from an aspiration into a
configuration guarantee.

### Decision 7: Service type ClusterIP + Ingress, not LoadBalancer

Every Service can be exposed as `type: LoadBalancer`, which makes
the cloud provision a dedicated LB per Service. That's $20/month
per service, and beginners often end up with 15 of them by Project
04.

Instead we use one Ingress that fronts every Service via host- or
path-based routing. One LB, one TLS cert, N services. The cost
delta and the operational simplicity both compound.

### Decision 8: ConfigMap for non-secrets + Secret for secrets

API configuration (model name, top-K default, log level) lives in
a ConfigMap. The Hugging Face token (if needed) lives in a Secret.
The Deployment mounts both as environment variables.

**Anti-pattern to avoid**: stuffing secrets into the ConfigMap
because it's easier. ConfigMaps are world-readable inside the
namespace; Secrets are gated by RBAC. The difference matters when
a compromised application pod tries to enumerate its environment.

## Trade-offs we deliberately accepted

### Helm chart in addition to raw YAML

The `helm/` directory ships a Helm chart that templates the same
manifests. Some teams loathe Helm and prefer Kustomize. We ship the
Helm version because:

- It's the more common production tool by deployment count.
- The values.yaml shape teaches **what should be configurable**
  (replicas, image tag, env vars) vs hardcoded.
- Kustomize is mechanically simpler but doesn't teach the same
  parameterization lesson.

The senior-engineer track explores GitOps-with-Argo for the same
deployments.

### Prometheus pull, not OTLP push

The monitoring stack uses Prometheus scraping the API's `/metrics`
endpoint. OTLP push is the more modern standard, but Prometheus is
the dominant on-cluster monitoring system today and teaches the
ServiceMonitor + PodMonitor pattern.

### NGINX Ingress, not a service mesh

We use NGINX as the ingress controller, not Istio/Linkerd. Service
mesh is the right answer for a 50-microservice fleet; for a
three-pod single-service deployment it is wildly out of scope.
The `engineer/mod-104/ex-05-service-mesh-observability` exercise
covers the mesh path.

### Load test in Locust, not k6 or Vegeta

`loadtest/` uses Locust because the Python syntax matches the rest
of the curriculum. k6 (Go-flavored JavaScript) is faster per
worker; Vegeta is leaner; both are valid choices.

## What the tests cover

The `tests/` directory has both unit tests (testing the API code
unchanged from Project 01) *and* integration tests that hit a
running cluster:

1. **`kubectl rollout status`** runs to ensure the deploy
   converges.
2. **HPA scale-up** is verified by generating synthetic load and
   asserting the replica count grows.
3. **Pod anti-affinity** is verified by checking that all pods land
   on different nodes after a rolling update.

These tests fail if any of the deliberate decisions above are
removed.

## Common mistakes graders see

1. **`livenessProbe` and `readinessProbe` pointing at the same
   endpoint**: kill-loop on slow model loads.
2. **`initialDelaySeconds: 0` on probes**: the probe fires before
   the model has loaded, the pod gets killed, the Deployment never
   reaches Ready.
3. **No `terminationGracePeriodSeconds`**: in-flight requests get
   dropped when a pod is removed; clients see 5xx errors during
   deploys.
4. **HPA configured without resource requests**: the autoscaler
   has nothing to scale against (CPU utilization is computed
   against requests).
5. **`replicas: 1` because "the HPA will scale it up"**: the HPA
   does not run during a deploy; the first 30-60 seconds after
   deploy are single-replica capacity until the HPA catches up.
6. **Exposing each Service as a LoadBalancer**: works, expensive,
   doesn't scale to more than a handful of services.

## When to go beyond this implementation

The implementation is intentionally minimal. Production paths
beyond it:

- Service mesh for mTLS + canary deploys + circuit breaking.
- GitOps with Argo CD / Flux instead of `kubectl apply`.
- Pod Disruption Budgets to protect against voluntary disruptions
  (node drains, cluster upgrades) in addition to involuntary ones.
- Network policies to constrain pod-to-pod traffic.
- OPA / Kyverno for cluster-wide policy enforcement.

Each is its own exercise later. Master this one first.

## Related curriculum touchpoints

- `junior/project-01` — the API itself, which this project
  deploys.
- `engineer/mod-104/ex-04-k8s-cluster-autoscaler` — the
  custom-metric scaling path.
- `engineer/mod-104/ex-05-service-mesh-observability` — Istio +
  golden signals + canary deploys.
- `engineer/mod-104/ex-06-k8s-operator-framework` — the CRD-driven
  ModelDeployment operator that supersedes raw Deployment YAML.
