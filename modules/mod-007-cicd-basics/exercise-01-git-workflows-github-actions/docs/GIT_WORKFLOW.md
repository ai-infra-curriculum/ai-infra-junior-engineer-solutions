# Git Workflow Guide

## Overview

This guide describes the Git workflows and branching strategies used in this project. Following these practices ensures consistent, manageable, and traceable development.

## Table of Contents

- [Branch Strategy](#branch-strategy)
- [Commit Message Format](#commit-message-format)
- [Feature Development Workflow](#feature-development-workflow)
- [Hotfix Workflow](#hotfix-workflow)
- [Release Workflow](#release-workflow)
- [Best Practices](#best-practices)
- [Common Scenarios](#common-scenarios)

## Branch Strategy

### Branch Types

#### `main`
- **Purpose**: Production-ready code
- **Protected**: ✅ Yes
- **Direct commits**: ❌ Not allowed
- **Requires**: PR approval, passing CI checks
- **Deployment**: Automatically deploys to production (if configured)

#### `develop`
- **Purpose**: Integration branch for features
- **Protected**: ✅ Yes
- **Direct commits**: ❌ Not allowed (use PRs)
- **Requires**: PR approval, passing CI checks
- **Deployment**: Automatically deploys to staging (if configured)

#### Feature Branches (`feature/*`)
- **Purpose**: Develop new features
- **Naming**: `feature/short-description`
- **Branch from**: `develop`
- **Merge to**: `develop`
- **Example**: `feature/add-model-training`

#### Bugfix Branches (`bugfix/*`)
- **Purpose**: Fix non-critical bugs
- **Naming**: `bugfix/issue-number-description`
- **Branch from**: `develop`
- **Merge to**: `develop`
- **Example**: `bugfix/123-fix-inference-error`

#### Hotfix Branches (`hotfix/*`)
- **Purpose**: Fix critical production bugs
- **Naming**: `hotfix/description`
- **Branch from**: `main`
- **Merge to**: `main` and `develop`
- **Example**: `hotfix/critical-security-fix`

#### Release Branches (`release/*`)
- **Purpose**: Prepare for a new release
- **Naming**: `release/v1.2.3`
- **Branch from**: `develop`
- **Merge to**: `main` and `develop`
- **Example**: `release/v1.2.0`

#### Documentation Branches (`docs/*`)
- **Purpose**: Documentation updates only
- **Naming**: `docs/description`
- **Branch from**: `develop` or `main`
- **Merge to**: Same as source
- **Example**: `docs/update-readme`

## Commit Message Format

### Conventional Commits

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Purpose | Example |
|------|---------|---------|
| `feat` | New feature | `feat(training): Add distributed training support` |
| `fix` | Bug fix | `fix(api): Handle edge case in preprocessing` |
| `docs` | Documentation only | `docs: Update deployment guide` |
| `style` | Code style/formatting | `style: Apply black formatting` |
| `refactor` | Code refactoring | `refactor(model): Simplify prediction logic` |
| `test` | Add/update tests | `test: Add integration tests for API` |
| `chore` | Maintenance tasks | `chore: Update dependencies` |
| `ci` | CI/CD changes | `ci: Add code coverage checks` |
| `perf` | Performance improvements | `perf: Optimize inference speed` |
| `build` | Build system changes | `build: Configure Docker multi-stage build` |
| `revert` | Revert previous commit | `revert: Revert "feat: Add feature X"` |

### Scope (Optional)

The scope specifies the area of the codebase:
- `api` - API endpoints
- `model` - ML model code
- `training` - Training scripts
- `inference` - Inference code
- `data` - Data processing
- `deploy` - Deployment code

### Examples

**Good commit messages:**
```bash
feat(training): Add distributed training with Ray
fix(api): Handle timeout in model loading
docs: Add troubleshooting section to README
refactor(model): Extract feature engineering into separate module
test(api): Add integration tests for /predict endpoint
chore(deps): Bump tensorflow from 2.12.0 to 2.13.0
ci: Add automated security scanning
```

**Bad commit messages:**
```bash
Update code          # Too vague
Fix bug              # Not descriptive
WIP                  # Work in progress commits should be squashed
asdfasdf             # Meaningless
Fixed the thing      # Not specific
```

### Breaking Changes

Indicate breaking changes with `!` after the type/scope:

```bash
feat(api)!: Change response format to JSON-API spec

BREAKING CHANGE: The API response format has changed from a flat
structure to JSON-API specification. Clients will need to update their
parsing logic.

Migration guide: See docs/migrations/v2.0.0.md
```

## Feature Development Workflow

### 1. Create Feature Branch

```bash
# Ensure you're on develop and it's up to date
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/add-model-versioning
```

### 2. Develop the Feature

```bash
# Make changes
vim src/model_versioning.py

# Check status
git status

# Stage changes
git add src/model_versioning.py

# Commit with conventional commit format
git commit -m "feat(model): Add model versioning system"
```

### 3. Keep Branch Updated

```bash
# Regularly sync with develop
git fetch origin
git rebase origin/develop

# Or use merge if rebasing is complex
git merge origin/develop
```

### 4. Push Changes

```bash
# Push feature branch
git push origin feature/add-model-versioning

# If you rebased, you may need to force push
git push origin feature/add-model-versioning --force-with-lease
```

### 5. Create Pull Request

1. Go to GitHub repository
2. Click "New Pull Request"
3. Select base: `develop`, compare: `feature/add-model-versioning`
4. Fill in PR template:
   - Clear title following conventional commits
   - Description of changes
   - Testing instructions
   - Related issues
5. Assign reviewers
6. Wait for CI checks to pass
7. Address review comments

### 6. Merge Pull Request

After approval and passing checks:

1. **Squash and Merge** (recommended for feature branches)
   - Combines all commits into one
   - Keeps main/develop history clean
   - Use when you have many small commits

2. **Rebase and Merge**
   - Replays commits on top of base branch
   - Maintains individual commit history
   - Use when commits are well-organized

3. **Merge Commit**
   - Creates a merge commit
   - Preserves full branch history
   - Use for significant features or releases

### 7. Clean Up

```bash
# Switch to develop
git checkout develop
git pull origin develop

# Delete local feature branch
git branch -d feature/add-model-versioning

# Delete remote feature branch (if not auto-deleted)
git push origin --delete feature/add-model-versioning
```

## Hotfix Workflow

For critical bugs in production:

### 1. Create Hotfix Branch

```bash
# Branch from main
git checkout main
git pull origin main
git checkout -b hotfix/fix-critical-auth-bug
```

### 2. Fix the Issue

```bash
# Make the fix
vim src/auth.py

# Test thoroughly
pytest tests/

# Commit
git commit -m "fix(auth): Resolve authentication bypass vulnerability

CVE-2024-XXXXX: Fixed critical authentication bypass by validating
tokens before checking permissions.

Fixes #456"
```

### 3. Create Pull Request

- Target: `main`
- Mark as high priority
- Request fast-track review
- Ensure all tests pass

### 4. Merge to Main

```bash
# After PR approval
git checkout main
git pull origin main

# Tag the hotfix
git tag -a v1.2.1 -m "Hotfix: Authentication bug"
git push origin v1.2.1
```

### 5. Backport to Develop

```bash
# Merge hotfix to develop
git checkout develop
git pull origin develop
git merge hotfix/fix-critical-auth-bug
git push origin develop

# Delete hotfix branch
git branch -d hotfix/fix-critical-auth-bug
```

## Release Workflow

### 1. Create Release Branch

```bash
# Branch from develop
git checkout develop
git pull origin develop
git checkout -b release/v1.3.0
```

### 2. Prepare Release

```bash
# Update version numbers
vim setup.py pyproject.toml

# Update changelog
vim CHANGELOG.md

# Commit changes
git commit -m "chore(release): Prepare v1.3.0 release"
```

### 3. Test Release

```bash
# Run comprehensive tests
pytest tests/ --cov

# Build artifacts
python -m build

# Test in staging environment
```

### 4. Finalize Release

```bash
# Push release branch
git push origin release/v1.3.0

# Create PR to main
# After approval and merge:

git checkout main
git pull origin main

# Tag release
git tag -a v1.3.0 -m "Release v1.3.0"
git push origin v1.3.0

# Merge back to develop
git checkout develop
git merge main
git push origin develop

# Delete release branch
git branch -d release/v1.3.0
```

## Best Practices

### Do's ✅

1. **Write Clear Commit Messages**
   - Follow conventional commits format
   - Be descriptive but concise
   - Reference issues when applicable

2. **Commit Often**
   - Make small, atomic commits
   - Each commit should be a logical unit
   - Easier to review and revert if needed

3. **Pull Before Push**
   - Always pull latest changes before pushing
   - Resolve conflicts locally
   - Test after merging

4. **Review Your Changes**
   - Use `git diff` before committing
   - Ensure no debug code or secrets
   - Check for unintended changes

5. **Use Branches**
   - Never commit directly to main/develop
   - Create feature branches for all work
   - Keep branches short-lived

6. **Test Before Committing**
   - Run tests locally
   - Ensure code builds
   - Verify changes work as expected

### Don'ts ❌

1. **Don't Commit Secrets**
   - No API keys, passwords, or tokens
   - Use environment variables
   - Review files before committing

2. **Don't Commit Binary Files**
   - No large files (>10MB)
   - Use Git LFS for necessary binaries
   - Keep repository size manageable

3. **Don't Force Push to Shared Branches**
   - Never force push to main/develop
   - Only force push to your feature branches
   - Use `--force-with-lease` when needed

4. **Don't Mix Concerns**
   - One feature/fix per branch
   - Don't combine unrelated changes
   - Makes review and rollback easier

5. **Don't Leave Broken Code**
   - Ensure code compiles/runs
   - Fix test failures before pushing
   - Don't break CI for others

6. **Don't Ignore Reviews**
   - Address all review comments
   - Ask questions if unclear
   - Don't merge without approval

## Common Scenarios

### Syncing Fork with Upstream

```bash
# Add upstream remote
git remote add upstream https://github.com/original/repo.git

# Fetch upstream changes
git fetch upstream

# Merge upstream changes
git checkout main
git merge upstream/main
git push origin main
```

### Undoing Changes

```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Undo changes to specific file
git checkout -- file.py

# Revert a commit (create new commit)
git revert <commit-hash>
```

### Resolving Merge Conflicts

```bash
# When conflict occurs
git status  # See conflicted files

# Edit files to resolve conflicts
vim conflicted_file.py

# Mark as resolved
git add conflicted_file.py

# Continue merge
git merge --continue
# or
git rebase --continue
```

### Cherry-Picking Commits

```bash
# Apply specific commit to current branch
git cherry-pick <commit-hash>

# Cherry-pick without committing
git cherry-pick -n <commit-hash>

# Cherry-pick a range
git cherry-pick <start-hash>..<end-hash>
```

### Stashing Changes

```bash
# Stash current changes
git stash

# List stashes
git stash list

# Apply most recent stash
git stash pop

# Apply specific stash
git stash apply stash@{1}

# Delete stash
git stash drop stash@{0}
```

### Viewing History

```bash
# View commit history
git log --oneline --graph --all

# View changes in commit
git show <commit-hash>

# View file history
git log --follow file.py

# Search commit messages
git log --grep="bug fix"

# View changes by author
git log --author="John Doe"
```

## Additional Resources

- [Git Documentation](https://git-scm.com/doc)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [Git Best Practices](https://www.atlassian.com/git/tutorials/comparing-workflows)
- [Pro Git Book](https://git-scm.com/book/en/v2)
