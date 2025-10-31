# AI Infrastructure Junior Engineer - Solutions Repository

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Modules: 10](https://img.shields.io/badge/Modules-10-blue.svg)]()
[![Exercises: 79](https://img.shields.io/badge/Exercises-79-green.svg)]()

Complete, production-ready solutions for the AI Infrastructure Junior Engineer Learning Path. This repository contains fully implemented code, comprehensive documentation, and step-by-step guides for all exercises.

**📌 Note**: Module naming has been updated to align with the learning repository. See [EXERCISE_SOLUTIONS_MAP.md](EXERCISE_SOLUTIONS_MAP.md) for complete exercise-to-solution mapping.

---

## 📚 Overview

This repository provides **reference solutions** for all exercises in the [ai-infra-junior-engineer-learning](../learning/ai-infra-junior-engineer-learning) curriculum. Each solution includes:

- ✅ **Complete, working code** ready to run
- ✅ **Step-by-step implementation guides**
- ✅ **Architecture documentation** with diagrams
- ✅ **Comprehensive test suites**
- ✅ **Docker & Kubernetes configurations**
- ✅ **Deployment scripts** and automation
- ✅ **Troubleshooting guides**
- ✅ **Production best practices**

## ✨ What's New

**🎓 Capstone Project Solutions Added!**
- 🚀 **Project 01: Simple Model API** - Flask + Docker + PyTorch serving
- ☸️ **Project 02: Kubernetes Model Serving** - K8s + HPA + Ingress
- 🔄 **Project 03: ML Pipeline with Tracking** - Airflow + MLflow + DVC
- 📊 **Project 04: Monitoring & Alerting** - Prometheus + Grafana + ELK
- 🏗️ **Project 05: Production ML System** - Complete CI/CD + Security + HA

**Recently Added Exercise Solutions:**
- 🤖 **LLM Basics Exercise** (Module 004) - Complete solution for running your first language model with Hugging Face Transformers
- ⚡ **GPU Fundamentals Exercise** (Module 004) - Full implementation of GPU-accelerated ML inference with PyTorch
- 🏗️ **Terraform/IaC Exercise** (Module 010) - Production-ready Infrastructure as Code with hands-on AWS deployment
- 🔄 **Airflow Workflow Exercise** (Module 009) - Complete ML pipeline orchestration with monitoring and alerting

**New Documentation:**
- 📋 **[Technology Versions Guide](https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-learning/blob/main/VERSIONS.md)** - Version specifications for all tools
- 🗺️ **[Curriculum Cross-Reference](https://github.com/ai-infra-curriculum/.github/blob/main/CURRICULUM_CROSS_REFERENCE.md)** - Mapping to Engineer track
- 📈 **[Career Progression Guide](https://github.com/ai-infra-curriculum/.github/blob/main/CAREER_PROGRESSION.md)** - Complete career ladder

---

## 🎯 Learning Philosophy

**Important**: These solutions are meant to be used **AFTER** attempting the exercises yourself. The learning path is designed to:

1. **Try First**: Attempt each exercise independently using the learning repository
2. **Compare**: Review the solution to see different approaches
3. **Understand**: Read the step-by-step guide to understand design decisions
4. **Improve**: Identify gaps and refine your own implementation

**Don't just copy code** - understand the WHY behind each decision.

---

## 📁 Repository Structure

```
ai-infra-junior-engineer-solutions/
├── README.md (this file)
├── EXERCISE_SOLUTIONS_MAP.md (complete exercise-to-solution mapping)
├── LEARNING_GUIDE.md (how to use this repository effectively)
├── modules/
│   ├── mod-001-python-fundamentals/
│   ├── mod-002-linux-essentials/
│   ├── mod-003-git-version-control/
│   ├── mod-004-ml-basics/ ✨ NEW
│   │   ├── exercise-04-llm-basics/
│   │   └── exercise-05-gpu-fundamentals/
│   ├── mod-005-docker-containers/
│   │   ├── exercise-01-docker-basics/
│   │   │   ├── README.md
│   │   │   ├── STEP_BY_STEP.md
│   │   │   ├── src/ (complete code)
│   │   │   ├── tests/
│   │   │   ├── docker/
│   │   │   └── scripts/
│   │   ├── exercise-02-multi-stage-builds/
│   │   ├── exercise-03-docker-compose/
│   │   └── ...
│   ├── mod-006-kubernetes-intro/
│   ├── mod-007-apis-web-services/
│   ├── mod-008-databases-sql/
│   ├── mod-009-monitoring-basics/
│   │   └── exercise-06-airflow-workflow-monitoring/ ✨ NEW
│   └── mod-010-cloud-platforms/
│       └── exercise-07-terraform-basics/ ✨ NEW
├── _deprecated/ (archived modules)
│   ├── mod-010-capstone-projects/ (moved to projects/)
│   └── mod-011-ml-serving-apis/ (integrated into curriculum)
├── projects/ 🎓 NEW
│   ├── project-01-simple-model-api/
│   ├── project-02-kubernetes-serving/
│   ├── project-03-ml-pipeline-tracking/
│   ├── project-04-monitoring-alerting/
│   └── project-05-production-ml-capstone/
├── .github/
│   └── workflows/
│       ├── ci-cd.yml
│       └── docker-build.yml
├── guides/
│   ├── debugging-guide.md
│   ├── optimization-guide.md
│   ├── production-readiness-checklist.md
│   └── common-pitfalls.md
└── resources/
    ├── additional-reading.md
    ├── useful-tools.md
    └── community-resources.md
```

---

## 🚀 Quick Start

### Prerequisites

Ensure you have the following installed:
- **Docker** (20.10+): `docker --version`
- **Docker Compose** (2.0+): `docker compose version`
- **Kubernetes** (kubectl 1.25+): `kubectl version --client`
- **Python** (3.11+): `python --version`
- **Node.js** (18+): `node --version`
- **AWS CLI** (2.x): `aws --version`
- **Terraform** (1.5+): `terraform --version`

### Clone Repository

```bash
git clone https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-solutions.git
cd ai-infra-junior-engineer-solutions
```

### Run a Solution

Each exercise has a `scripts/` directory with automated setup:

```bash
# Navigate to an exercise
cd modules/mod-005-docker-containers/exercise-01-docker-basics

# Run setup script
./scripts/setup.sh

# Run the application
./scripts/run.sh

# Run tests
./scripts/test.sh
```

---

## 📖 Modules & Solutions

### Module 004: ML Basics (2 exercises) ✨ NEW

| Exercise | Description | Complexity | Concepts |
|----------|-------------|------------|----------|
| **01** | LLM Basics | ⭐⭐ Medium | Hugging Face Transformers, model loading, inference |
| **02** | GPU Fundamentals | ⭐⭐⭐ Hard | CUDA, PyTorch GPU acceleration, performance optimization |

**Total Lines of Code**: ~2,800
**Estimated Completion**: 8-12 hours

---

### Module 005: Docker Containers (7 exercises)

| Exercise | Description | Complexity | Concepts |
|----------|-------------|------------|----------|
| **01** | Docker Basics | ⭐ Easy | Dockerfile, images, containers |
| **02** | Multi-Stage Builds | ⭐⭐ Medium | Build optimization, layer caching |
| **03** | Docker Compose | ⭐⭐ Medium | Multi-container apps, networking |
| **04** | ML Model Serving | ⭐⭐⭐ Hard | Flask API, model loading, optimization |
| **05** | Container Optimization | ⭐⭐⭐ Hard | Image size reduction, security scanning |
| **06** | Docker Networking | ⭐⭐ Medium | Bridge, overlay, host networking |
| **07** | Persistent Data | ⭐⭐ Medium | Volumes, bind mounts, data management |

**Total Lines of Code**: ~3,500
**Estimated Completion**: 12-16 hours

---

### Module 006: Kubernetes Introduction (7 exercises)

| Exercise | Description | Complexity | Concepts |
|----------|-------------|------------|----------|
| **01** | Kubernetes Basics | ⭐⭐ Medium | Pods, deployments, services |
| **02** | ConfigMaps & Secrets | ⭐⭐ Medium | Configuration management |
| **03** | Persistent Volumes | ⭐⭐ Medium | StatefulSets, PVCs, storage classes |
| **04** | Ingress & Load Balancing | ⭐⭐⭐ Hard | NGINX Ingress, path-based routing |
| **05** | Autoscaling | ⭐⭐⭐ Hard | HPA, VPA, cluster autoscaler |
| **06** | Helm Charts | ⭐⭐⭐ Hard | Package management, templating |
| **07** | Production ML Deployment | ⭐⭐⭐⭐ Expert | Complete stack with monitoring |

**Total Lines of Code**: ~4,200
**Estimated Completion**: 14-18 hours

---

### Module 007: APIs & Web Services (5 exercises)

| Exercise | Description | Complexity | Concepts |
|----------|-------------|------------|----------|
| **01** | REST API with Flask | ⭐⭐ Medium | REST principles, Flask routing |
| **02** | FastAPI ML Service | ⭐⭐ Medium | Async API, Pydantic validation |
| **03** | gRPC Service | ⭐⭐⭐ Hard | Protocol buffers, streaming |
| **04** | GraphQL API | ⭐⭐⭐ Hard | Schema design, resolvers |
| **05** | Production API | ⭐⭐⭐⭐ Expert | Auth, rate limiting, caching, docs |

**Total Lines of Code**: ~5,800
**Estimated Completion**: 16-20 hours

---

### Module 008: Databases & SQL (5 exercises)

| Exercise | Description | Complexity | Concepts |
|----------|-------------|------------|----------|
| **01** | SQL Fundamentals | ⭐ Easy | CRUD, joins, aggregations |
| **02** | PostgreSQL for ML | ⭐⭐ Medium | Schema design, indexes, performance |
| **03** | Database Migrations | ⭐⭐ Medium | Alembic, version control |
| **04** | NoSQL with MongoDB | ⭐⭐ Medium | Document storage, aggregation pipeline |
| **05** | Production Database | ⭐⭐⭐⭐ Expert | HA, replication, backup, monitoring |

**Total Lines of Code**: ~4,500
**Estimated Completion**: 14-18 hours

---

### Module 009: Monitoring Basics (6 exercises)

| Exercise | Description | Complexity | Concepts |
|----------|-------------|------------|----------|
| **01** | Observability Foundations | ⭐⭐ Medium | Metrics, logs, traces |
| **02** | Prometheus Stack | ⭐⭐⭐ Hard | Prometheus, exporters, PromQL |
| **03** | Grafana Dashboards | ⭐⭐ Medium | Visualization, alerts |
| **04** | Logging with Loki | ⭐⭐⭐ Hard | Log aggregation, querying |
| **05** | Alerting & Incidents | ⭐⭐⭐⭐ Expert | Alertmanager, runbooks, postmortems |
| **06** | Airflow Workflow ✨ NEW | ⭐⭐⭐ Hard | Pipeline orchestration, DAGs, monitoring |

**Total Lines of Code**: ~14,500
**Estimated Completion**: 22-26 hours

---

### Module 010: Cloud Platforms (6 exercises)

| Exercise | Description | Complexity | Concepts |
|----------|-------------|------------|----------|
| **01** | AWS Account & IAM | ⭐ Easy | IAM, MFA, tagging, budgets |
| **02** | Compute & Storage | ⭐⭐ Medium | EC2, S3, EBS, Spot instances |
| **03** | Networking & Security | ⭐⭐⭐ Hard | VPC, Security Groups, Terraform |
| **04** | Containerized Deployment | ⭐⭐⭐⭐ Expert | ECS, EKS, ECR, auto-scaling |
| **05** | SageMaker & Optimization | ⭐⭐⭐⭐ Expert | ML platform, cost optimization |
| **06** | Terraform IaC ✨ NEW | ⭐⭐⭐ Hard | Infrastructure as Code, modules, state management |

**Total Lines of Code**: ~9,400
**Estimated Completion**: 26-30 hours

---

## 🎓 Capstone Projects (5 projects)

| Project | Description | Complexity | Technologies |
|---------|-------------|------------|--------------|
| **01** | Simple Model API | ⭐⭐⭐ Hard | Flask, PyTorch, Docker, ResNet-50 |
| **02** | Kubernetes Model Serving | ⭐⭐⭐⭐ Expert | Kubernetes, HPA, Ingress, NGINX |
| **03** | ML Pipeline with Tracking | ⭐⭐⭐⭐ Expert | Airflow, MLflow, DVC, Great Expectations |
| **04** | Monitoring & Alerting | ⭐⭐⭐⭐ Expert | Prometheus, Grafana, ELK, Alertmanager |
| **05** | Production ML System | ⭐⭐⭐⭐⭐ Master | CI/CD, Security, HA, Canary, SLOs |

**Total Documentation**: ~3,500 lines across comprehensive SOLUTION_GUIDE.md files
**Estimated Completion**: 40-60 hours total
**Portfolio Ready**: Yes - production-grade implementations

Each capstone project includes:
- Complete source code with tests
- Production configurations (Docker, K8s, CI/CD)
- Comprehensive SOLUTION_GUIDE.md (500-900 lines)
- Architecture diagrams and design decisions
- Deployment automation and troubleshooting

---

## 🛠️ Technology Stack

### Languages
- **Python** 3.11+ (ML, APIs, automation)
- **SQL** (PostgreSQL, MySQL)
- **YAML** (Kubernetes, Docker Compose)
- **HCL** (Terraform)
- **Bash** (scripting, automation)

### Frameworks & Libraries
- **Flask** / **FastAPI** (APIs)
- **PyTorch** / **TensorFlow** (ML models)
- **SQLAlchemy** (ORM)
- **Pytest** (testing)

### Infrastructure
- **Docker** & **Docker Compose**
- **Kubernetes** (Minikube, Kind, EKS)
- **Helm** (package management)
- **Terraform** (IaC)

### Cloud & Services
- **AWS** (EC2, S3, ECS, EKS, SageMaker)
- **Prometheus** & **Grafana** (monitoring)
- **Loki** (logging)

---

## 📝 Using This Repository Effectively

### For Self-Learners

1. **Attempt First**: Try the exercise from the learning repository
2. **Get Stuck?**: Review the STEP_BY_STEP.md for guidance
3. **Compare Solutions**: See how your approach differs
4. **Run Tests**: Ensure your solution passes all test cases
5. **Deploy**: Use the deployment scripts to test in real environments

### For Instructors

1. **Reference Implementation**: Use as canonical examples
2. **Grading**: Compare student submissions against solutions
3. **Discussion**: Point out design decisions and trade-offs
4. **Extensions**: Challenge students to improve upon solutions

### For Hiring Managers

1. **Skill Assessment**: Use exercises as take-home assignments
2. **Code Review**: Evaluate candidate solutions against references
3. **Interview Prep**: Discuss architecture decisions in interviews

---

## 🧪 Testing

All solutions include comprehensive test suites:

```bash
# Run all tests for a module
cd modules/mod-007-apis-web-services
./scripts/test-all.sh

# Run tests for specific exercise
cd exercise-02-fastapi-ml-service
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

**Test Coverage Goals**: 80%+ for all production code

---

## 🚢 Deployment

Each exercise includes deployment configurations:

### Local Development
```bash
# Docker Compose
docker compose up -d

# Kubernetes (Minikube)
kubectl apply -f k8s/
```

### Cloud Deployment (AWS)
```bash
# Terraform
cd terraform/
terraform init
terraform apply

# ECS/EKS
./scripts/deploy-aws.sh
```

---

## 🐛 Troubleshooting

### Common Issues

**Docker build fails**:
```bash
# Clear cache and rebuild
docker builder prune
docker build --no-cache -t myapp .
```

**Kubernetes pods not starting**:
```bash
# Check events
kubectl describe pod <pod-name>
kubectl logs <pod-name>

# Check resources
kubectl top nodes
kubectl top pods
```

**AWS credentials issues**:
```bash
# Verify credentials
aws sts get-caller-identity

# Reconfigure
aws configure
```

See [guides/debugging-guide.md](guides/debugging-guide.md) for comprehensive troubleshooting.

---

## 📚 Additional Resources

### Official Documentation
- [Docker Docs](https://docs.docker.com/)
- [Kubernetes Docs](https://kubernetes.io/docs/)
- [AWS Documentation](https://docs.aws.amazon.com/)
- [Terraform Docs](https://www.terraform.io/docs)

### Learning Resources
- [Docker Mastery Course](https://www.udemy.com/course/docker-mastery/)
- [Kubernetes in Action](https://www.manning.com/books/kubernetes-in-action)
- [AWS Certified Solutions Architect](https://aws.amazon.com/certification/certified-solutions-architect-associate/)

### Community
- [Kubernetes Slack](https://slack.k8s.io/)
- [r/kubernetes](https://reddit.com/r/kubernetes)
- [Stack Overflow - Docker](https://stackoverflow.com/questions/tagged/docker)

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improve-exercise-05`)
3. Make your changes
4. Add tests
5. Submit a pull request

### Contribution Ideas
- Improve documentation
- Add more test cases
- Optimize Docker images
- Add new deployment targets (GCP, Azure)
- Create video walkthroughs

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Learning Repository**: [ai-infra-junior-engineer-learning](../learning/ai-infra-junior-engineer-learning)
- **Contributors**: See [CONTRIBUTORS.md](CONTRIBUTORS.md)
- **Inspired By**: Industry best practices from Google, Netflix, Uber ML teams

---

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-solutions/issues)
- **Email**: ai-infra-curriculum@joshua-ferguson.com
- **Organization**: [AI Infrastructure Curriculum](https://github.com/ai-infra-curriculum)

---

## 🎯 Success Metrics

After completing all exercises with these solutions, you should be able to:

✅ Build and optimize Docker containers for ML workloads
✅ Deploy scalable applications on Kubernetes
✅ Design and implement production-grade REST/gRPC APIs
✅ Manage databases with proper schema design and migrations
✅ Implement comprehensive monitoring and alerting systems
✅ Deploy ML infrastructure on AWS with cost optimization
✅ Write Infrastructure as Code with Terraform
✅ Debug production issues effectively
✅ Pass Junior AI Infrastructure Engineer interviews

---

**Happy Learning! 🚀**

*Last Updated: October 30, 2025*
*Version: 1.1.0 - Repository Structure Alignment*
