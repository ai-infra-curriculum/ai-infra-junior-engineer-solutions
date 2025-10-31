# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2025-10-30

### Changed - Repository Structure Alignment (Phase 1)

**Module Naming Updates**: Renamed 7 modules to align with learning repository naming conventions:

- `mod-001-foundations` → `mod-001-python-fundamentals`
- `mod-002-python-programming` → `mod-002-linux-essentials`
- `mod-003-linux-command-line` → `mod-003-git-version-control`
- `mod-005-docker-containerization` → `mod-005-docker-containers`
- `mod-007-cicd-basics` → `mod-007-apis-web-services`
- `mod-008-cloud-platforms` → `mod-008-databases-sql`
- `mod-009-monitoring-logging` → `mod-009-monitoring-basics`

**Deprecated Modules**: Moved non-curriculum modules to `_deprecated/` directory:

- `mod-010-capstone-projects` → `_deprecated/mod-010-capstone-projects` (content moved to `projects/`)
- `mod-011-ml-serving-apis` → `_deprecated/mod-011-ml-serving-apis` (content integrated into curriculum)

### Added

- **EXERCISE_SOLUTIONS_MAP.md**: Comprehensive mapping manifest tracking all 79 exercises across 10 modules
  - Documents current solution coverage: 0% (all 79 exercises need implementation)
  - Highlights 12 new exercises added to learning repository (2025-10-30)
  - Provides complete exercise-to-solution path mapping
  - Includes priority implementation order

- **_deprecated/** directory: Archive location for superseded modules

### Updated

- **README.md**: Updated to reflect new module structure
  - Exercise count badge: 58 → 79 exercises
  - Added reference to EXERCISE_SOLUTIONS_MAP.md
  - Updated repository structure diagram with new module names
  - Updated all code examples to use new module paths
  - Added note about structural alignment with learning repository
  - Updated version to 1.1.0
  - Updated last modified date to 2025-10-30

### Migration Notes

**For Users**:

If you have cloned this repository before 2025-10-30, you may need to update your local paths:

```bash
# Pull latest changes
git pull origin main

# Update any scripts or bookmarks referencing old module names
# Old: modules/mod-005-docker/
# New: modules/mod-005-docker-containers/
```

**Git History**: All renames were performed using `git mv` to preserve file history.

**Breaking Changes**: None - this is a structural reorganization only. All existing solutions remain intact with full git history preserved.

---

## [1.0.0] - 2025-10-23

### Added

**Initial Release**: Complete solution repository for AI Infrastructure Junior Engineer curriculum

- 10 modules with comprehensive solutions
- 58 exercises with production-ready implementations
- 5 capstone projects with detailed solution guides
- Automated setup and testing scripts for all exercises
- Comprehensive documentation and troubleshooting guides

**Module Coverage**:

- Module 001: Python Fundamentals (8 exercises)
- Module 002: Linux Essentials (9 exercises)
- Module 003: Git & Version Control (6 exercises)
- Module 004: ML Basics (6 exercises)
- Module 005: Docker Containers (7 exercises)
- Module 006: Kubernetes Introduction (8 exercises)
- Module 007: APIs & Web Services (8 exercises)
- Module 008: Databases & SQL (7 exercises)
- Module 009: Monitoring Basics (8 exercises)
- Module 010: Cloud Platforms (9 exercises)

**Project Solutions**:

- Project 01: Simple Model API (Flask + Docker + PyTorch)
- Project 02: Kubernetes Model Serving (K8s + HPA + Ingress)
- Project 03: ML Pipeline with Tracking (Airflow + MLflow + DVC)
- Project 04: Monitoring & Alerting (Prometheus + Grafana + ELK)
- Project 05: Production ML System (Complete CI/CD + Security + HA)

**Documentation**:

- README.md with comprehensive overview
- LEARNING_GUIDE.md for effective usage
- Individual exercise READMEs and STEP_BY_STEP.md guides
- Production readiness checklists
- Debugging and optimization guides

---

## Release Tags

- `v1.1.0` - Repository Structure Alignment (2025-10-30)
- `v1.0.0` - Initial Release (2025-10-23)

---

**Note**: Versions prior to 1.0.0 were in development and not tagged.
