# Junior Engineer Solutions - Directory Migration Map

**Generated**: 2025-11-01
**Purpose**: Document directory realignment to match learning repository structure

---

## Migration Overview

**Problem**: Solutions repository directory structure doesn't match the current learning repository module organization after curriculum restructuring.

**Impact**:
- Solutions impossible to find
- EXERCISE_SOLUTIONS_MAP.md reports incorrect coverage (15% vs actual 42%)
- Contributor confusion

**Solution**: Migrate misaligned directories using `git mv` to preserve history

---

## Current State Analysis

### Module 001: Python Fundamentals ✅ CORRECT
**Status**: No migration needed
- exercise-01 through exercise-08: All correctly placed

### Module 002: Linux Essentials ❌ MISALIGNED
**Current**: Contains PYTHON exercises (wrong content!)
- exercise-01-python-basics
- exercise-02-oop
- exercise-03-file-io-errors
- exercise-04-testing-pytest
- exercise-05-data-processing
- exercises 06-09: Correctly placed Linux content

**Issues**:
1. Exercises 01-05 are Python content (belong in mod-001)
2. Only exercises 06-09 are actual Linux content
3. Missing 7 Linux exercises that should be here

**Learning Repo Has**:
- exercise-01: Linux basics
- exercise-02: Shell scripting
- exercise-03: File permissions
- exercise-04: Process management
- exercise-05: Networking tools
- exercise-06: SSH and remote access
- exercise-07: System administration
- exercise-08: Package management
- exercise-09: Shell automation

**Migration Needed**:
- Import Linux exercises from mod-003 (bash exercises)
- Keep existing exercises 06-09
- Remove Python exercises (they're duplicates of mod-001)

### Module 003: Git Version Control ❌ MISALIGNED
**Current**: Contains LINUX/BASH exercises (wrong content!)
- exercise-01-bash-scripting
- exercise-02-filesystem-processes
- exercise-03-ssh-networking
- exercise-04-system-administration
- exercises 05-08: Correctly placed (presumably Git content)

**Issues**:
1. Exercises 01-04 are Linux/bash content (should be in mod-002)
2. Only exercises 05-08 might be Git content
3. Missing implementation guides for all 8 Git exercises

**Learning Repo Has**:
- exercise-01: Git basics
- exercise-02: Branching and merging
- exercise-03: Collaboration workflows
- exercise-04: Pull requests and code review
- exercise-05: Git for ML projects
- exercise-06: Git LFS for models
- exercise-07: Advanced Git techniques
- exercise-08: Team workflows

**Migration Needed**:
- Move exercises 01-04 (bash/Linux) → mod-002
- Verify exercises 05-08 are actually Git content
- Create implementation guides for all Git exercises

### Module 004: ML Basics ✅ MOSTLY CORRECT
**Status**: Minor issues only
- exercise-01 through exercise-05: Correct
- exercise-06: Missing implementation guide

### Module 005: Docker Containers ✅ MOSTLY CORRECT
**Status**: Minor issues only
- exercise-01 through exercise-07: Correct
- exercise-08: Missing (production ML deployment)

### Module 006: Kubernetes Intro ✅ CORRECT
**Status**: Complete coverage
- exercise-01 through exercise-08: All present with guides

### Module 007: APIs & Web Services ⚠️ MOSTLY CORRECT
**Current**: Contains CI/CD content (partially correct)
- Exercises 01-06: CI/CD focused (correct for Junior level)
- Exercise 07-08: Missing proper implementation guides

### Module 008: Databases & SQL ❌ COMPLETELY MISALIGNED
**Current**: Contains CLOUD PLATFORM exercises (wrong content!)
- exercise-01-aws-fundamentals
- exercise-02-gcp-ml-infrastructure
- exercise-03-azure-ml-services
- exercise-04-multi-cloud-deployment
- exercise-05-cost-optimization
- exercises 06-07: Empty or minimal

**Issues**:
1. ALL current content is cloud platforms (should be in mod-010)
2. ZERO database/SQL content exists
3. Missing all 7 database exercises

**Learning Repo Has**:
- exercise-01: SQL fundamentals
- exercise-02: Database design
- exercise-03: Query optimization
- exercise-04: Transactions and ACID
- exercise-05: Indexing strategies
- exercise-06: NoSQL comparison
- exercise-07: Database operations

**Migration Needed**:
- Move ALL cloud exercises (01-05) → mod-010
- Create NEW database/SQL solutions for all 7 exercises

### Module 009: Monitoring Basics ✅ MOSTLY CORRECT
**Status**: Good coverage, minor gaps
- exercise-01 through exercise-06: Correct with guides
- exercises 07-08: Missing SLO/runbook guides

### Module 010: Cloud Platforms ❌ INCOMPLETE
**Current**: Only FinOps content
- exercise-07-terraform-basics
- exercise-08: (FinOps)
- exercise-09: (FinOps)

**Issues**:
1. Missing exercises 01-06 (AWS, GCP, Azure basics)
2. Need to import cloud exercises from mod-008
3. Exercise 07 (Terraform) correctly placed

**Learning Repo Has**:
- exercise-01: AWS fundamentals
- exercise-02: GCP ML infrastructure
- exercise-03: Azure ML services
- exercise-04: Multi-cloud deployment
- exercise-05: Cloud cost optimization
- exercise-06: Cloud security
- exercise-07: Terraform basics
- exercise-08: Multi-cloud with Terraform
- exercise-09: FinOps practices

**Migration Needed**:
- Import exercises 01-05 from mod-008
- Keep existing exercises 07-09
- Create exercise-06 (cloud security)
- Create exercise-08 (multi-cloud Terraform)

---

## Detailed Migration Plan

### Phase 1: Move Linux Exercises from mod-003 to mod-002

**Source**: `modules/mod-003-git-version-control/`
**Destination**: `modules/mod-002-linux-essentials/`

```bash
# Move bash/Linux exercises to correct location
git mv modules/mod-003-git-version-control/exercise-01-bash-scripting \
       modules/mod-002-linux-essentials/exercise-10-bash-scripting

git mv modules/mod-003-git-version-control/exercise-02-filesystem-processes \
       modules/mod-002-linux-essentials/exercise-11-filesystem-processes

git mv modules/mod-003-git-version-control/exercise-03-ssh-networking \
       modules/mod-002-linux-essentials/exercise-12-ssh-networking

git mv modules/mod-003-git-version-control/exercise-04-system-administration \
       modules/mod-002-linux-essentials/exercise-13-system-administration
```

**Note**: Numbering as 10-13 to preserve existing 01-09, or renumber all consistently

### Phase 2: Move Cloud Exercises from mod-008 to mod-010

**Source**: `modules/mod-008-databases-sql/`
**Destination**: `modules/mod-010-cloud-platforms/`

```bash
# Move cloud platform exercises to correct location
git mv modules/mod-008-databases-sql/exercise-01-aws-fundamentals \
       modules/mod-010-cloud-platforms/exercise-01-aws-fundamentals

git mv modules/mod-008-databases-sql/exercise-02-gcp-ml-infrastructure \
       modules/mod-010-cloud-platforms/exercise-02-gcp-ml-infrastructure

git mv modules/mod-008-databases-sql/exercise-03-azure-ml-services \
       modules/mod-010-cloud-platforms/exercise-03-azure-ml-services

git mv modules/mod-008-databases-sql/exercise-04-multi-cloud-deployment \
       modules/mod-010-cloud-platforms/exercise-04-multi-cloud-deployment

git mv modules/mod-008-databases-sql/exercise-05-cost-optimization \
       modules/mod-010-cloud-platforms/exercise-05-cost-optimization
```

### Phase 3: Clean Up mod-002 Python Duplicates

**Action**: Remove or consolidate duplicate Python exercises in mod-002

**Decision Point**:
- Option A: Delete duplicates (if identical to mod-001)
- Option B: Move to mod-001 if they're different/better
- Option C: Keep as alternative implementations with note

```bash
# After verification, remove duplicates
git rm -r modules/mod-002-linux-essentials/exercise-01-python-basics
git rm -r modules/mod-002-linux-essentials/exercise-02-oop
git rm -r modules/mod-002-linux-essentials/exercise-03-file-io-errors
git rm -r modules/mod-002-linux-essentials/exercise-04-testing-pytest
git rm -r modules/mod-002-linux-essentials/exercise-05-data-processing
```

### Phase 4: Renumber for Consistency

After migrations, renumber exercises to match learning repo:

**mod-002-linux-essentials**:
- Current 06-09 become 01-04 (or keep if they match learning repo)
- Imported 10-13 become 05-08
- Final: exercise-01 through exercise-09

**mod-003-git-version-control**:
- Current 05-08 become 01-04 (or verify they match learning repo)
- Need to create 05-08 if missing
- Final: exercise-01 through exercise-08

**mod-010-cloud-platforms**:
- Imported 01-05 stay as is
- Need to create 06 (cloud security)
- Keep 07-09 as is
- Final: exercise-01 through exercise-09

---

## New Content Creation Needed

After migrations, still need to CREATE solutions for:

### Module 002: Linux Essentials
If current exercises 06-09 don't align with learning repo exercises 01-09:
- Potentially 4-7 new Linux solutions

### Module 003: Git Version Control
- 4-8 new Git implementation guides (depending on what exists in 05-08)

### Module 008: Databases & SQL
- **ALL 7 exercises** need solutions (complete new content)
  - SQL fundamentals
  - Database design
  - Query optimization
  - Transactions and ACID
  - Indexing strategies
  - NoSQL comparison
  - Database operations

### Module 010: Cloud Platforms
- exercise-06: Cloud security (NEW)
- exercise-08: Multi-cloud with Terraform (NEW if missing)

---

## Verification Checklist

After each migration:

### Directory Structure
- [ ] Exercise directory exists in correct module
- [ ] No duplicate directories remain
- [ ] Numbering is consistent (01-XX)
- [ ] Git history preserved (use `git log --follow`)

### File Contents
- [ ] README.md exists with correct module context
- [ ] IMPLEMENTATION_GUIDE.md or STEP_BY_STEP.md present
- [ ] Code files present and functional
- [ ] Test files exist
- [ ] No references to old module names

### Documentation Updates
- [ ] Module README updated with new exercise list
- [ ] EXERCISE_SOLUTIONS_MAP.md regenerated
- [ ] Cross-references updated
- [ ] CHANGELOG.md entry added

---

## Risk Mitigation

### Backup Strategy
```bash
# Before any migrations, create backup branch
git checkout -b pre-migration-backup
git checkout main
```

### Testing Strategy
After each migration:
1. Verify git history with `git log --follow <path>`
2. Check for broken links with link checker
3. Run any existing tests
4. Manually verify exercise content matches module

### Rollback Plan
If migration causes issues:
```bash
# Revert individual migration
git revert <commit-hash>

# Or restore from backup
git checkout pre-migration-backup -- modules/mod-XXX/
```

---

## Timeline

**Phase 1: Linux Migration** - 2 hours
- Move 4 bash exercises from mod-003 to mod-002
- Verify alignment with learning repo
- Update documentation

**Phase 2: Cloud Migration** - 2 hours
- Move 5 cloud exercises from mod-008 to mod-010
- Verify alignment with learning repo
- Update documentation

**Phase 3: Cleanup** - 2 hours
- Remove Python duplicates from mod-002
- Clean up empty directories
- Verify no broken references

**Phase 4: Renumbering** - 2 hours
- Renumber exercises for consistency
- Update all references
- Regenerate exercise map

**Total Estimated Time**: 8 hours

---

## Success Criteria

- [ ] All exercises in correct module matching learning repo structure
- [ ] No misaligned content (Python in Linux, Cloud in Databases, etc.)
- [ ] Consistent exercise numbering across all modules
- [ ] Git history preserved for all moved files
- [ ] No broken links or references
- [ ] EXERCISE_SOLUTIONS_MAP.md shows accurate coverage
- [ ] All module READMEs updated

---

## Next Steps

1. ✅ Complete this migration map
2. 🔄 Create backup branch
3. Execute Phase 1: Linux migrations
4. Execute Phase 2: Cloud migrations
5. Execute Phase 3: Cleanup
6. Execute Phase 4: Renumbering
7. Regenerate EXERCISE_SOLUTIONS_MAP.md
8. Create git commit with detailed message
9. Begin implementation guide creation for newly aligned modules

---

## Document History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-11-01 | Initial migration map created | AI Infrastructure Curriculum Team |

---

**Related Documents**:
- REMEDIATION_PLAN.md (Parent plan)
- EXERCISE_SOLUTIONS_MAP.md (To be regenerated after migration)
- Solutions Audit: `/home/s0v3r1gn/ai-infra-project/reports/ai-infra-junior-engineer-solutions.md`
