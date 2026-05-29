# Curriculum

Navigation map for the modules and projects in this repository. Each link points to the on-disk solution.

For per-exercise notes and solution status, see [SOLUTIONS_INDEX.md](SOLUTIONS_INDEX.md).

---

## Modules

### Module 001: Python Fundamentals for Infrastructure

| # | Exercise | Topics |
|---|---|---|
| 01 | [Python basics](modules/mod-001-python-fundamentals/exercise-01-python-basics/) | Functions, decorators, context managers, error handling |
| 02 | [OOP](modules/mod-001-python-fundamentals/exercise-02-oop/) | Classes, inheritance, dataclasses |
| 03 | [File I/O and errors](modules/mod-001-python-fundamentals/exercise-03-file-io-errors/) | pathlib, JSON/YAML, exception design |
| 04 | [Testing with pytest](modules/mod-001-python-fundamentals/exercise-04-testing-pytest/) | Fixtures, parametrize, mocking |
| 05 | [Data processing](modules/mod-001-python-fundamentals/exercise-05-data-processing/) | itertools, generators, large-file streaming |
| 06 | [Async programming for ML](modules/mod-001-python-fundamentals/exercise-06-async/) | asyncio for concurrent I/O |
| 07 | [Testing Python ML code](modules/mod-001-python-fundamentals/exercise-07-testing/) | Unit, integration, property-based |

### Module 002: Linux Essentials

| # | Exercise | Topics |
|---|---|---|
| 01 | [Bash scripting](modules/mod-002-linux-essentials/exercise-01-bash-scripting/) | set -euo pipefail, traps, functions |
| 02 | [Filesystem and processes](modules/mod-002-linux-essentials/exercise-02-filesystem-processes/) | find, grep, ps, lsof |
| 03 | [SSH and networking](modules/mod-002-linux-essentials/exercise-03-ssh-networking/) | Key auth, port forwarding, ss/netstat |
| 04 | [System administration](modules/mod-002-linux-essentials/exercise-04-system-administration/) | systemd, journalctl, cron |
| 05 | [Linux package management](modules/mod-002-linux-essentials/exercise-05-package-mgmt/) | apt, dnf, pip, conda |
| 06 | [Linux log management](modules/mod-002-linux-essentials/exercise-06-logs/) | Reading, rotating, shipping logs |
| 07 | [Linux troubleshooting](modules/mod-002-linux-essentials/exercise-07-troubleshooting/) | Disk full, OOM, network issues |
| 08 | [Linux system automation](modules/mod-002-linux-essentials/exercise-08-system-automation/) | cron, systemd timers, scripted ops |

### Module 003: Git & Version Control

| # | Exercise | Topics |
|---|---|---|
| 01 | [Git fundamentals](modules/mod-003-git-version-control/exercise-01-git-fundamentals/) | Branching, merging, rebase, conflict resolution, GitHub workflows |
| 02 | [Working with commits and history](modules/mod-003-git-version-control/exercise-02-commits-history/) | log, diff, reflog, history navigation |
| 03 | [Branching strategies](modules/mod-003-git-version-control/exercise-03-branching/) | Trunk-based, GitFlow, release branches |
| 04 | [Merging and resolving conflicts](modules/mod-003-git-version-control/exercise-04-merging-conflicts/) | Merge vs rebase, conflict resolution |
| 05 | [Git collaboration on ML teams](modules/mod-003-git-version-control/exercise-05-collaboration/) | PRs, code review, team workflows |
| 06 | [Git workflows for ML projects](modules/mod-003-git-version-control/exercise-06-ml-workflows/) | Notebooks, model artifacts, experiments |
| 07 | [Advanced Git](modules/mod-003-git-version-control/exercise-07-advanced/) | cherry-pick, bisect, hooks, submodules |
| 08 | [Git LFS for ML projects](modules/mod-003-git-version-control/exercise-08-git-lfs-ml-projects/) | Large files, datasets, model binaries |

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
| 08 | [Production ML deployment with Docker](modules/mod-005-docker-containers/exercise-08-production-ml-deployment/) | Production-grade ML serving images |

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
| 05 | [Flask framework](modules/mod-007-apis-web-services/exercise-05-flask-framework/) | Flask + Smorest + Marshmallow |
| 06 | [Production ML API](modules/mod-007-apis-web-services/exercise-06-production-ml-api/) | Production-grade FastAPI serving |

### Module 008: Databases & SQL

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
| 06 | [ML infrastructure on AWS](modules/mod-010-cloud-platforms/exercise-06-ml-infrastructure-aws/) | End-to-end AWS ML stack |
| 07 | [Terraform basics](modules/mod-010-cloud-platforms/exercise-07-terraform-basics/) | IaC fundamentals |

---

## Projects

| # | Project | Description |
|---|---|---|
| 01 | [Simple Model API](projects/project-01-simple-model-api/) | REST API serving image classification predictions |
| 02 | [Kubernetes Model Serving](projects/project-02-kubernetes-serving/) | Production K8s deployment with autoscaling and monitoring |
| 03 | [ML Pipeline with Experiment Tracking](projects/project-03-ml-pipeline-tracking/) | Airflow + MLflow + DVC end-to-end pipeline |
| 04 | [Monitoring & Alerting System](projects/project-04-monitoring-alerting/) | Prometheus + Grafana + ELK + Alertmanager stack |
| 05 | [Production-Ready ML System (Capstone)](projects/project-05-production-ml-capstone/) | Integration of projects 1-4 with CI/CD, security, HA |
