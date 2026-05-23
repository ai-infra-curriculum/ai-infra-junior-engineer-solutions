# Solutions Index

**Last Updated:** 2026-05-22
**Status:** Aligned with [learning CURRICULUM.md](https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-learning/blob/main/CURRICULUM.md)

Inventory of every solution in this repository, grouped by module. Click into each link for the per-exercise README + solutions/ + tests/.

---

## Modules

### Module 000: Orientation (pre-curriculum)

Onboarding material that helps before starting the canonical curriculum. Not part of `CURRICULUM.md`.

| # | Exercise | Topics |
|---|---|---|
| 01 | [AI infrastructure overview](modules/mod-000-orientation/exercise-01-ai-infra-overview/) | What an ML infra team does; the landscape |
| 02 | [Development environment setup](modules/mod-000-orientation/exercise-02-dev-environment/) | VS Code, Python, Docker, Git tooling |

### Module 001: Python Fundamentals for Infrastructure

| # | Exercise | Topics |
|---|---|---|
| 01 | [Python basics](modules/mod-001-python-fundamentals/exercise-01-python-basics/) | Functions, decorators, context managers, error handling |
| 02 | [OOP](modules/mod-001-python-fundamentals/exercise-02-oop/) | Classes, inheritance, dataclasses |
| 03 | [File I/O and errors](modules/mod-001-python-fundamentals/exercise-03-file-io-errors/) | pathlib, JSON/YAML, exception design |
| 04 | [Testing with pytest](modules/mod-001-python-fundamentals/exercise-04-testing-pytest/) | Fixtures, parametrize, mocking |
| 05 | [Data processing](modules/mod-001-python-fundamentals/exercise-05-data-processing/) | itertools, generators, large-file streaming |

### Module 002: Linux Essentials

| # | Exercise | Topics |
|---|---|---|
| 01 | [Bash scripting](modules/mod-002-linux-essentials/exercise-01-bash-scripting/) | set -euo pipefail, traps, functions |
| 02 | [Filesystem and processes](modules/mod-002-linux-essentials/exercise-02-filesystem-processes/) | find, grep, ps, lsof |
| 03 | [SSH and networking](modules/mod-002-linux-essentials/exercise-03-ssh-networking/) | Key auth, port forwarding, ss/netstat |
| 04 | [System administration](modules/mod-002-linux-essentials/exercise-04-system-administration/) | systemd, journalctl, cron |

### Module 003: Git & Version Control

| # | Exercise | Topics |
|---|---|---|
| 01 | [Git fundamentals](modules/mod-003-git-version-control/exercise-01-git-fundamentals/) | Branching, merging, rebase, conflict resolution, GitHub workflows |

### Module 004: ML Basics

| # | Exercise | Topics |
|---|---|---|
| 01 | [ML fundamentals](modules/mod-004-ml-basics/exercise-01-ml-fundamentals/) | Train/val/test, common metrics |
| 02 | [Training pipeline](modules/mod-004-ml-basics/exercise-02-model-training-pipeline/) | End-to-end training script |
| 03 | [Model deployment](modules/mod-004-ml-basics/exercise-03-model-deployment/) | ONNX, TorchScript, SavedModel |
| 04 | [LLM basics](modules/mod-004-ml-basics/exercise-04-llm-basics/) | Hugging Face Transformers |
| 05 | [GPU fundamentals](modules/mod-004-ml-basics/exercise-05-gpu-fundamentals/) | CUDA, mixed precision |

### Module 005: Docker & Containers

| # | Exercise | Topics |
|---|---|---|
| 01 | [Docker fundamentals](modules/mod-005-docker-containers/exercise-01-docker-fundamentals/) | Images, containers, registries |
| 02 | [Building ML images](modules/mod-005-docker-containers/exercise-02-building-ml-images/) | Multi-stage builds, layer caching |
| 03 | [Docker Compose](modules/mod-005-docker-containers/exercise-03-docker-compose/) | Multi-container apps |
| 04 | [Docker networking](modules/mod-005-docker-containers/exercise-04-docker-networking/) | Bridge, host, overlay |
| 05 | [Docker volumes](modules/mod-005-docker-containers/exercise-05-docker-volumes/) | Bind mounts, named volumes |
| 06 | [Container security](modules/mod-005-docker-containers/exercise-06-container-security/) | Non-root, read-only FS, scanning |
| 07 | [Production deployment](modules/mod-005-docker-containers/exercise-07-production-deployment/) | Gunicorn, health checks, signals |

### Module 006: Kubernetes Introduction

| # | Exercise | Topics |
|---|---|---|
| 01 | [First deployment](modules/mod-006-kubernetes-intro/exercise-01-first-deployment/) | Pods, Deployments, Services |
| 02 | [Helm chart](modules/mod-006-kubernetes-intro/exercise-02-helm-chart/) | Templating, values, releases |
| 03 | [Debugging](modules/mod-006-kubernetes-intro/exercise-03-debugging/) | kubectl describe, logs, events |
| 04 | [StatefulSets & storage](modules/mod-006-kubernetes-intro/exercise-04-statefulsets-storage/) | PVCs, storage classes |
| 05 | [ConfigMaps & Secrets](modules/mod-006-kubernetes-intro/exercise-05-configmaps-secrets/) | Configuration management |
| 06 | [Ingress & load balancing](modules/mod-006-kubernetes-intro/exercise-06-ingress-loadbalancing/) | NGINX Ingress |
| 07 | [ML workloads](modules/mod-006-kubernetes-intro/exercise-07-ml-workloads/) | HPA, affinity, GPU |

### Module 007: APIs & Web Services

| # | Exercise | Topics |
|---|---|---|
| 01 | [FastAPI fundamentals](modules/mod-007-apis-web-services/exercise-01-fastapi-fundamentals/) | Routing, Pydantic, auto-docs |
| 02 | [Model serving](modules/mod-007-apis-web-services/exercise-02-model-serving/) | FastAPI ML API |
| 03 | [Production API](modules/mod-007-apis-web-services/exercise-03-production-api/) | Auth, rate limiting, caching |
| 04 | [Performance optimization](modules/mod-007-apis-web-services/exercise-04-performance-optimization/) | Async, batching, profiling |
| 05 | [Flask framework](modules/mod-007-apis-web-services/exercise-05-flask-framework/) NEW | Flask + Smorest + Marshmallow |

### Module 008: Databases & SQL — NEW

| # | Exercise | Topics |
|---|---|---|
| 01 | [SQL basics & CRUD](modules/mod-008-databases-sql/exercise-01-sql-basics-crud/) | DDL, DML, transactions, upserts |
| 02 | [Database design — ML registry](modules/mod-008-databases-sql/exercise-02-database-design-ml-registry/) | 3NF, partial unique indexes, lineage tables |
| 03 | [Advanced SQL joins](modules/mod-008-databases-sql/exercise-03-advanced-sql-joins/) | CTEs, window functions, LATERAL |
| 04 | [SQLAlchemy ORM integration](modules/mod-008-databases-sql/exercise-04-sqlalchemy-orm-integration/) | Declarative 2.0, repositories, Alembic |
| 05 | [Indexing & optimization](modules/mod-008-databases-sql/exercise-05-optimization-indexing/) | EXPLAIN, B-tree/GIN/BRIN, partial/expression indexes |

### Module 009: Monitoring & Logging Basics

| # | Exercise | Topics |
|---|---|---|
| 01 | [Observability foundations](modules/mod-009-monitoring-basics/exercise-01-observability-foundations/) | Metrics, logs, traces, SLIs |
| 02 | [Prometheus stack](modules/mod-009-monitoring-basics/exercise-02-prometheus-stack/) | Scrape configs, PromQL |
| 03 | [Grafana dashboards](modules/mod-009-monitoring-basics/exercise-03-grafana-dashboards/) | Panels, variables, alerts |
| 04 | [Logging pipeline](modules/mod-009-monitoring-basics/exercise-04-logging-pipeline/) | ELK or Loki |
| 05 | [Alerting & incident response](modules/mod-009-monitoring-basics/exercise-05-alerting-incident-response/) | Alertmanager, runbooks |
| 06 | [Airflow workflow monitoring](modules/mod-009-monitoring-basics/exercise-06-airflow-workflow-monitoring/) | DAG observability |

### Module 010: Cloud Platforms

| # | Exercise | Topics |
|---|---|---|
| 01 | [AWS fundamentals](modules/mod-010-cloud-platforms/exercise-01-aws-fundamentals/) | IAM, EC2, S3, VPC |
| 02 | [GCP ML infrastructure](modules/mod-010-cloud-platforms/exercise-02-gcp-ml-infrastructure/) | GKE, Cloud Storage, Vertex AI |
| 03 | [Azure ML services](modules/mod-010-cloud-platforms/exercise-03-azure-ml-services/) | AKS, Blob, ML Studio |
| 04 | [Multi-cloud deployment](modules/mod-010-cloud-platforms/exercise-04-multi-cloud-deployment/) | Portability, federation |
| 05 | [Cost optimization](modules/mod-010-cloud-platforms/exercise-05-cost-optimization/) | Reserved vs spot, rightsizing |
| 07 | [Terraform basics](modules/mod-010-cloud-platforms/exercise-07-terraform-basics/) | IaC fundamentals |

---

## Projects

| # | Project | Solution status |
|---|---|---|
| 01 | [Simple Model API](projects/project-01-simple-model-api/) | Full implementation (Flask + Docker + tests) |
| 02 | [Kubernetes Model Serving](projects/project-02-kubernetes-serving/) | Full: manifests, Helm chart, Grafana dashboard, Locust loadtest, ServiceMonitor |
| 03 | [ML Pipeline with Tracking](projects/project-03-ml-pipeline-tracking/) | Full: Airflow DAGs (incl. retraining), DVC, Great Expectations, MLflow, src/, tests/ |
| 04 | [Monitoring & Alerting](projects/project-04-monitoring-alerting/) | Full: Prometheus, Alertmanager, ELK, Grafana dashboards, 5 runbooks, instrumentation src/ |
| 05 | [Production ML Capstone](projects/project-05-production-ml-capstone/) | Full: Terraform (VPC/EKS/RDS/IAM), Kustomize overlays (dev/staging/prod), Velero, security, CI/CD |

---

## Supplementary

Material outside the canonical curriculum:

- [`supplementary/cicd-basics/`](supplementary/cicd-basics/) — CI/CD focused module from earlier curriculum drafts.
- [`supplementary/alternative-capstones/`](supplementary/alternative-capstones/) — Three alternative capstone projects (ML platform, fraud detection, multi-cloud) for learners who want a different angle than `projects/project-05`.

---

## Honesty Note

This index was previously inaccurate (claimed completeness in places that were stubs). The 2026-05-22 restructure brought reality into alignment with the inventory. If you find another discrepancy, please open an issue.
