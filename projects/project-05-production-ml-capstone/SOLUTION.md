# SOLUTION — Production-Ready ML System (Capstone)

> Read this *after* you have stood up the full system yourself.
> This is the design-rationale doc for the capstone: how the four
> prior projects compose, what the integration tax actually is,
> and which failure modes the full system addresses that the
> individual projects can't. `SOLUTION_GUIDE.md` is the how-to-
> run-it walkthrough.

## What this project is really teaching

The four prior projects each teach one concern in isolation:

- **Project 01**: model serving.
- **Project 02**: orchestration + scaling.
- **Project 03**: training + experiment tracking.
- **Project 04**: monitoring + alerting.

What the capstone teaches is what **most of the actual job is** —
the **glue and the discipline** of integrating those concerns:

- Code changes flow through CI to staging to production
  automatically and safely.
- Secrets are managed, not committed.
- Failed deploys roll back without a human deciding to.
- Infrastructure changes go through the same review process as
  application changes.
- The whole thing keeps running when one piece breaks.

A junior engineer who can compose these pieces is operating at the
boundary of the next tier. Every decision below is shaped by that.

## Architectural decisions and *why*

### Decision 1: Infrastructure as code (Terraform) for the
**entire** cluster, not just the apps

The `terraform/` directory provisions:

- The cluster itself.
- Node pools (CPU and GPU).
- IAM roles + service accounts.
- The container registry.
- The object storage buckets (datasets, model artifacts,
  Velero backups).
- DNS records + TLS certificates.

The `kubernetes/` directory then applies Kubernetes objects *into*
the cluster Terraform created. Why split it that way:

- Cluster-level changes need separate review + apply velocity from
  app-level changes.
- An "oops" rolling out a Kubernetes manifest doesn't risk
  destroying the cluster.
- The Terraform state file is the only thing that knows what
  cloud resources exist; `kubectl` can't replace that.

**Anti-pattern to avoid**: doing it all through `kubectl` and
clicking around in the cloud console. Works once, breaks at the
first "rebuild the staging cluster from scratch" requirement.

### Decision 2: GitOps via `cicd/` + Argo CD (or Flux), not
direct `kubectl apply`

The deploy path:

1. Engineer pushes commit to git.
2. CI builds image, runs tests, pushes to registry.
3. CI updates the Argo CD application manifest (image tag bump).
4. Argo CD detects the manifest change and applies to the cluster.

No human runs `kubectl apply`. The cluster state is **always** the
last applied git commit, and "what's running in prod?" is
answerable by `git log` instead of cluster archaeology.

The trade-off: every change is now a code review, which slows
emergency hotfixes. The mitigation is well-defined break-glass
procedures, not skipping the GitOps gate.

### Decision 3: Two clusters (staging + prod), one codebase

Same Helm chart, two `values-{env}.yaml` files. Same Argo CD
config, two `Application` manifests pointing at different cluster
contexts. Staging and prod are configurationally distinguishable
in **exactly one place per concern** (replicas, resource sizes,
the cost-alarm threshold).

**Anti-pattern to avoid**: a separate codebase per environment.
Drift is inevitable; the first time prod has a feature staging
doesn't, the value of staging collapses.

### Decision 4: Multi-AZ everything

The Deployment has pod anti-affinity on `topology.kubernetes.io/zone`.
The node pools span three AZs. RDS / managed cache run multi-AZ.
The object storage bucket is replicated across AZs.

The capstone is graded against an AZ-loss drill: shut down all
nodes in one AZ; the API stays up. Hitting that bar requires every
single layer above to be multi-AZ-aware. One missing piece breaks
the chain.

### Decision 5: TLS everywhere, including pod-to-pod

cert-manager issues certs from Let's Encrypt for the public
ingress. Internal mTLS is handled either via Istio sidecars (if
the service mesh is deployed) or via cert-manager-issued
certificates mounted as Secrets when the mesh isn't installed.

`networkpolicies/` denies all pod-to-pod traffic by default, then
explicitly allows the flows the system actually uses. This blocks
lateral movement if any single container is compromised.

### Decision 6: Secrets in External Secrets Operator (or
sealed-secrets), never in plain YAML

The git repo contains `ExternalSecret` resources that reference
secrets in AWS Secrets Manager / GCP Secret Manager / HashiCorp
Vault. The actual secret material never enters git.

**Anti-pattern to avoid**: committing `Secret` manifests with
base64-encoded values. Base64 is not encryption; the secret is
public the moment the repo is.

### Decision 7: SLO-based alerting, not threshold alerting

The capstone replaces Project 04's static thresholds with SLO-
based alerts:

- **SLO**: 99.9% of requests succeed in under 500ms over a
  30-day window.
- **Error budget**: 1 - 0.999 = 0.1% of requests per month.
- **Alerts** fire on burn-rate: page if we'd exhaust a month's
  budget in less than 1 hour at the current rate.

The Multi-window multi-burn-rate pattern (Google SRE workbook)
gives:

- Fast page on catastrophic failure (<1 hour to budget
  exhaustion).
- Slow page on slow degradation (>3 days to budget exhaustion).
- No alert noise from transient blips.

### Decision 8: Velero for cluster backup + restore

The `velero/` directory configures hourly snapshots of the
cluster's persistent volumes + Kubernetes object state to object
storage. The recovery runbook covers:

- Restore a single PVC after accidental deletion.
- Restore an entire namespace after configuration drift.
- Rebuild the cluster from a clean state and restore everything.

The capstone's disaster-recovery drill consists of deleting the
namespace and restoring it from a Velero backup within 30 minutes.

## Trade-offs we deliberately accepted

### Argo CD over Flux, Helm over Kustomize

Argo CD has a UI; Flux is CLI-first. Both work. Argo's UI helps
junior engineers visualize the sync state during the learning
curve; production teams often standardize on Flux for the GitOps-
purist reasons.

Helm vs Kustomize is similar — Helm's templating is more
expressive, Kustomize's patch model is cleaner. We chose Helm for
ecosystem reach.

### Single cloud per environment

Staging on AWS, prod on AWS (different accounts). Multi-cloud
disaster recovery is the senior-architect-track problem. For a
junior capstone, single-cloud isolation between accounts is the
right safety boundary.

### NGINX Ingress, not a full service mesh

Service mesh (Istio/Linkerd) is the right answer for >10
microservices. For 3-5 services, the operational cost outweighs
the benefit. The capstone uses NGINX ingress + cert-manager + a
small set of NetworkPolicies.

The engineer-track `mod-104/ex-05-service-mesh-observability`
exercise picks up the mesh story.

## What the integration tests cover

`tests/` includes integration tests the prior projects can't run:

1. **End-to-end deploy** — push a code change, assert it lands
   in staging via GitOps within 5 minutes.
2. **Promotion** — assert a staging release can be promoted to
   prod by a single PR merge.
3. **Rollback drill** — inject a failing change, assert the
   automatic rollback fires within the SLO budget.
4. **AZ-loss drill** — cordon and drain one AZ's nodes, assert
   the API stays available.
5. **Backup-restore drill** — delete the production namespace,
   restore from Velero, assert the API resumes within 30 minutes.

A capstone that doesn't run these drills is a capstone that
hasn't proven the integration story works.

## Common mistakes graders see

1. **Single AZ everything** — passes individual tests, fails the
   capstone drill.
2. **`kubectl apply` from a laptop** — no audit trail, no
   rollback path, no GitOps benefit.
3. **Plain-text secrets in YAML** — base64 ≠ encrypted.
4. **No PodDisruptionBudget** — voluntary disruptions (node
   drain) take down the API.
5. **Default Helm values in prod** — `replicas: 1`, no probes
   configured, no resource requests.
6. **No CI gate** — humans push images, drift is inevitable.
7. **Static alert thresholds copied from Project 04** — SLO
   thinking is the capstone's distinguishing skill.
8. **Backups configured but never tested** — restore-untested
   backups are non-backups.

## When to go beyond this implementation

After the capstone, the next discipline tiers:

- Multi-cloud (active-active or warm-standby).
- Progressive deployment (Argo Rollouts / Flagger with canary +
  baking + auto-promotion).
- Cost optimization with spot instances + cluster-autoscaler
  bin-packing.
- Compliance automation (continuous-compliance scanning via
  Kyverno / OPA Gatekeeper / Falco).
- Internal developer platform layer that abstracts this entire
  stack behind a `mlctl deploy` CLI.

Each is its own future exercise. Master integration before
abstraction.

## Related curriculum touchpoints

- `junior/project-01,02,03,04` — the components this capstone
  composes.
- `engineer/mod-106/ex-06-ci-cd-ml-pipelines` — the next-tier
  CI/CD pipeline with formal gates.
- `engineer/mod-109/ex-01-terraform-ml-infrastructure` — the
  Terraform module discipline at full scale.
- `mlops/project-1-ml-pipeline` — the next-tier full ML pipeline.
- `architect/project-301-enterprise-mlops` — what the same shape
  looks like at enterprise scale.
