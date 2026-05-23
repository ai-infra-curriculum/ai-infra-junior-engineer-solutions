# AI Infrastructure Junior Engineer — Solutions Repository

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Modules: 10](https://img.shields.io/badge/Modules-10-blue.svg)]()
[![Aligned with curriculum](https://img.shields.io/badge/aligned%20with-CURRICULUM.md-green.svg)](https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-learning/blob/main/CURRICULUM.md)

Reference solutions for every exercise and project in the [Junior AI Infrastructure Engineer learning path](https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-learning). This repository is structured to **mirror the learning repo's curriculum 1:1** — same module numbers, same topic ordering — so you can compare your work side-by-side.

> **Learning philosophy:** Attempt each exercise in the learning repo first. Only consult these solutions after a genuine attempt. The point is to compare *your approach* to a reference, not to copy code.

---

## What's New — May 2026

The repository was restructured to align module numbering with the canonical curriculum (`CURRICULUM.md`). Highlights:

- ✅ **Modules renamed to match learning curriculum**: `mod-001-python-fundamentals`, `mod-002-linux-essentials`, `mod-003-git-version-control`, `mod-007-apis-web-services`, `mod-009-monitoring-basics`, `mod-010-cloud-platforms` — same names as in learning.
- ✅ **New `mod-008-databases-sql`** with 5 exercise solutions (CRUD, ML registry design, advanced SQL, SQLAlchemy ORM, indexing & optimization).
- ✅ **New Flask exercise** (`mod-007/exercise-05-flask-framework`) to complement the existing FastAPI exercise.
- ✅ **Project 02-05 solutions** are now real code — previously they were markdown stubs only. Includes Helm chart, Grafana dashboards, Locust load tests, DVC pipeline, Great Expectations suite, Alertmanager + runbooks, Terraform modules, Velero config.
- ✅ **Orphan and duplicate modules cleaned up**: `mod-010-cloud-platforms` orphan merged; `mod-007-cicd-basics` moved to `supplementary/` (no learning curriculum slot); alternative capstones moved to `supplementary/alternative-capstones/`.
- ✅ **`mod-000-orientation`** holds pre-curriculum onboarding (AI infra overview, dev environment setup) — supplements but does not duplicate any learning module.

---

## Repository Structure

```
ai-infra-junior-engineer-solutions/
├── README.md                       (this file)
├── SOLUTIONS_INDEX.md              Full inventory + completion status
├── LEARNING_GUIDE.md               How to use the solutions effectively
│
├── modules/                        Aligned 1:1 with learning curriculum
│   ├── mod-000-orientation/        Pre-curriculum onboarding (not in learning's CURRICULUM.md)
│   ├── mod-001-python-fundamentals/
│   ├── mod-002-linux-essentials/
│   ├── mod-003-git-version-control/
│   ├── mod-004-ml-basics/
│   ├── mod-005-docker-containers/
│   ├── mod-006-kubernetes-intro/
│   ├── mod-007-apis-web-services/  Includes FastAPI + Flask
│   ├── mod-008-databases-sql/      NEW: 5 SQL/NoSQL/ORM exercises
│   ├── mod-009-monitoring-basics/
│   └── mod-010-cloud-platforms/
│
├── projects/                       Reference implementations
│   ├── project-01-simple-model-api/
│   ├── project-02-kubernetes-serving/   + Helm chart, Grafana, loadtest
│   ├── project-03-ml-pipeline-tracking/ + DVC, GE, retraining DAG
│   ├── project-04-monitoring-alerting/  + Alertmanager, runbooks, dashboards
│   └── project-05-production-ml-capstone/ + Terraform, Velero, kustomize
│
├── supplementary/                  Material outside the canonical curriculum
│   ├── cicd-basics/                CI/CD module from prior repo layout
│   └── alternative-capstones/      ML platform, fraud detection, multi-cloud variants
│
├── guides/                         Cross-cutting practical guidance
│   ├── debugging-guide.md
│   ├── optimization-guide.md
│   ├── production-readiness-checklist.md
│   └── common-pitfalls.md
│
├── resources/                      Pointers to further reading and tools
│   ├── additional-reading.md
│   ├── useful-tools.md
│   └── community-resources.md
│
└── TEMPLATES/                      Reusable templates for new solutions
```

---

## Exercise Solutions

| Module | Topic | Exercises | Status |
|---|---|---|---|
| 000 | Orientation (pre-curriculum) | 2 | ✅ |
| 001 | Python Fundamentals | 5 | ✅ |
| 002 | Linux Essentials | 4 | ✅ |
| 003 | Git & Version Control | 1 | ✅ |
| 004 | ML Basics (PyTorch/TensorFlow/LLM/GPU) | 5 | ✅ |
| 005 | Docker & Containers | 7 | ✅ |
| 006 | Kubernetes Intro | 7 | ✅ |
| 007 | APIs & Web Services (FastAPI + Flask) | 5 | ✅ |
| 008 | Databases & SQL | 5 | ✅ NEW |
| 009 | Monitoring & Logging | 6 | ✅ |
| 010 | Cloud Platforms (AWS/GCP/Azure + Terraform) | 6 | ✅ |

Total: **53 exercise solutions** + **5 project solutions**.

See [SOLUTIONS_INDEX.md](SOLUTIONS_INDEX.md) for per-exercise details, technology mappings, and links.

---

## Project Solutions

| Project | Description | Key files |
|---|---|---|
| 01 | Simple Model API | Flask + Docker + tests |
| 02 | Kubernetes Model Serving | K8s manifests + **Helm chart** + Grafana dashboard + Locust loadtest + ServiceMonitor |
| 03 | ML Pipeline with Tracking | Airflow DAGs + **DVC pipeline** + **Great Expectations suite** + MLflow + retraining DAG |
| 04 | Monitoring & Alerting | Prometheus + **Alertmanager** + ELK + Grafana dashboards + **5 runbooks** |
| 05 | Production ML Capstone | **Terraform modules** (VPC/EKS/RDS/IAM) + Kustomize overlays + **Velero** backup + security |

---

## Quick Start

Each module has its own README explaining how to run the exercises. Common prerequisites:

```bash
# Tools
brew install python@3.11 docker kubectl helm terraform aws-cli

# Per-exercise setup is usually:
cd modules/mod-001-python-fundamentals/exercise-01-python-basics
pip install -r requirements.txt        # if present
pytest tests/                          # to validate
```

For database exercises:

```bash
docker run -d --name junior-db \
    -e POSTGRES_PASSWORD=devpass \
    -e POSTGRES_DB=ml \
    -p 5432:5432 postgres:15
export DATABASE_URL=postgresql://postgres:devpass@localhost:5432/ml
```

For project solutions, see each project's `README.md` and `SOLUTION_GUIDE.md`.

---

## How to Use This Repository

### For Learners

1. **Attempt first.** Open the [learning repo exercise](https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-learning) and complete it without looking here.
2. **Compare second.** Look at the matching solution here. Note *where your approach differs* — not just whether you "got it right."
3. **Read the why.** Solutions include inline comments and `SOLUTION_GUIDE.md` files that explain the trade-offs. The decisions are the lesson, not the code.
4. **Run the tests.** Every solution that ships with tests should pass `pytest tests/`. If yours doesn't, that's a starting point.

### For Instructors

- Use as reference implementations for grading.
- Adapt the `TEMPLATES/` for new exercises in your variant.
- The `SOLUTION_GUIDE.md` per project is a teach-able walkthrough.

### For Reviewers

- Open issues against specific exercise solutions if you find bugs or improvements.
- PRs welcome — see `CONTRIBUTING.md`.

---

## Module Alignment with Learning Curriculum

This repository's module numbering and names match the canonical [`CURRICULUM.md`](https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-learning/blob/main/CURRICULUM.md) exactly. The previous structure (with `mod-001-foundations`, `mod-007-cicd-basics`, etc.) has been migrated as follows:

| Old solutions path | New path |
|---|---|
| `modules/mod-001-foundations/exercise-01-ai-infra-overview` | `modules/mod-000-orientation/exercise-01-ai-infra-overview` |
| `modules/mod-001-foundations/exercise-02-dev-environment`   | `modules/mod-000-orientation/exercise-02-dev-environment` |
| `modules/mod-001-foundations/exercise-03-version-control`   | `modules/mod-003-git-version-control/exercise-01-git-fundamentals` |
| `modules/mod-002-python-programming/*`                       | `modules/mod-001-python-fundamentals/*` |
| `modules/mod-003-linux-command-line/*`                       | `modules/mod-002-linux-essentials/*` |
| `modules/mod-005-docker-containerization/*`                  | `modules/mod-005-docker-containers/*` |
| `modules/mod-008-cloud-platforms/*` + orphan `mod-010-cloud-platforms/*` | `modules/mod-010-cloud-platforms/*` |
| `modules/mod-009-monitoring-logging/*`                       | `modules/mod-009-monitoring-basics/*` |
| `modules/mod-011-ml-serving-apis/*`                          | `modules/mod-007-apis-web-services/*` |
| `modules/mod-007-cicd-basics/*` (no curriculum slot)         | `supplementary/cicd-basics/*` |
| `modules/mod-010-capstone-projects/*` (alt variants)         | `supplementary/alternative-capstones/*` |

---

## Related Repositories

- **Learning curriculum:** [ai-infra-junior-engineer-learning](https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-learning)
- **Next level:** [ai-infra-engineer-learning](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning)
- **All tracks:** [ai-infra-curriculum org](https://github.com/ai-infra-curriculum)

---

## Contributing

Open an issue or PR. See `CONTRIBUTING.md`.

---

**Last Updated:** May 2026
**Version:** 2.0.0 (full structural realignment)
**License:** MIT
