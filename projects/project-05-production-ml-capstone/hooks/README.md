# Git Hooks for Production ML System

This directory contains Git hooks that automate quality checks and enforce best practices. Based on Module 003 Exercise 07 advanced Git techniques.

## Available Hooks

### 1. pre-commit
Runs before each commit to catch issues early.

**Checks:**
- ✅ Python syntax validation
- ✅ Debug statement detection (print, pdb, breakpoint)
- ✅ Secret detection (API keys, passwords, tokens)
- ✅ File size limits (warns on files > 5MB)
- ✅ YAML syntax validation
- ✅ Kubernetes manifest validation
- ✅ Shell script validation (shellcheck)

### 2. pre-push
Runs before pushing to remote repository.

**Checks:**
- ✅ Branch name validation
- ✅ Protected branch warnings
- ✅ Commit message format validation
- ✅ Test execution
- ✅ Security scans on commit history

### 3. commit-msg
Validates commit messages follow conventional format.

**Checks:**
- ✅ Conventional commit format: `type(scope): description`
- ✅ First line length (max 72 characters)
- ✅ Description length (min 10 characters)
- ✅ Proper body formatting
- ✅ Breaking change detection

## Installation

### Quick Install (All Hooks)

```bash
# From repository root
chmod +x hooks/*
cp hooks/* .git/hooks/
```

### Individual Hook Installation

```bash
# Install specific hook
cp hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### Verify Installation

```bash
# Check installed hooks
ls -la .git/hooks/

# Test pre-commit hook
git add .
git commit -m "test: verify hooks working"
```

## Usage

### Pre-Commit Hook

Automatically runs when you execute `git commit`:

```bash
# Make changes
echo "print('debug')" >> src/api/main.py

# Try to commit (hook will warn about debug statement)
git add src/api/main.py
git commit -m "feat(api): add debug"
```

**Output:**
```
🔍 Running pre-commit checks...

[1/8] Checking Python syntax...
✓ All Python files have valid syntax

[2/8] Checking for debug statements...
⚠ Debug statement found in src/api/main.py
```

### Pre-Push Hook

Automatically runs when you execute `git push`:

```bash
# Push to remote (hook validates before push)
git push origin feature/my-feature
```

**Output:**
```
🚀 Running pre-push checks...

[1/5] Validating branch name...
✓ Branch name is valid: feature/my-feature

[2/5] Checking protected branches...
✓ Protected branch check complete

[3/5] Validating commit messages...
✓ All commit messages follow conventional format
```

### Commit-Msg Hook

Automatically validates commit message format:

```bash
# Try invalid commit message
git commit -m "Added feature"
```

**Output:**
```
📝 Validating commit message...

[1/4] Checking conventional commit format...
✗ Commit message doesn't follow conventional format

Required format:
  type(scope): description

Valid types:
  • feat:     New feature
  • fix:      Bug fix
  ...

Examples:
  ✓ feat(api): add health check endpoint
  ✓ fix(model): correct preprocessing bug
```

**Correct format:**
```bash
git commit -m "feat(api): add health check endpoint"
```

## Bypassing Hooks

**Only use in emergencies!**

```bash
# Skip pre-commit hook
git commit --no-verify -m "feat: emergency fix"

# Skip pre-push hook
git push --no-verify origin main
```

## Conventional Commit Format

### Format
```
type(scope): description

[optional body]

[optional footer]
```

### Valid Types

| Type | Description | Example |
|------|-------------|---------|
| `feat` | New feature | `feat(api): add user authentication` |
| `fix` | Bug fix | `fix(model): correct preprocessing` |
| `docs` | Documentation only | `docs: update API guide` |
| `style` | Code style changes | `style: format with black` |
| `refactor` | Code refactoring | `refactor(api): simplify error handling` |
| `test` | Add/update tests | `test(api): add integration tests` |
| `chore` | Maintenance | `chore: update dependencies` |
| `perf` | Performance | `perf(model): optimize inference` |
| `ci` | CI/CD changes | `ci: add Docker build workflow` |
| `build` | Build system | `build: update Dockerfile` |
| `revert` | Revert commit | `revert: revert feature X` |
| `model` | ML model updates | `model(bert): release v2.1.0` |

### Examples

**Good commit messages:**
```
feat(api): add health check endpoint

Add /health endpoint that returns server status and model version.
Helps with Kubernetes liveness probes.

Closes #123

---

fix(model): correct image preprocessing bug

The normalization was using wrong mean values.
Updated to ImageNet standard values.

---

model(bert): release v2.1.0 with attention mechanism

Performance improvements:
- Accuracy: 96.2% (+0.7% vs v2.0.0)
- Latency: 38ms P95 (-4ms vs v2.0.0)

Git tag: model-v2.1.0

---

docs: update deployment guide with Helm charts

Added section on Helm chart deployment.
Includes examples for staging and production.
```

**Bad commit messages:**
```
✗ Added feature          (missing type)
✗ feat:                  (missing description)
✗ feat: Added Feature    (description starts with capital)
✗ Updated code           (vague, missing type)
✗ WIP                    (not descriptive)
✗ Fix                    (missing scope and description)
```

## Branch Naming Convention

Hooks validate branch names follow this pattern:

### Format
```
<type>/<description>
```

### Valid Patterns

**Main branches:**
- `main`
- `master`
- `develop`
- `staging`
- `production`

**Feature branches:**
```
feature/<name>     - New features
fix/<name>         - Bug fixes
hotfix/<name>      - Critical production fixes
refactor/<name>    - Code refactoring
docs/<name>        - Documentation updates
test/<name>        - Test additions
chore/<name>       - Maintenance tasks
```

### Examples

**Good branch names:**
```
✓ feature/add-model-monitoring
✓ fix/memory-leak-inference
✓ hotfix/security-vulnerability
✓ refactor/api-error-handling
✓ docs/deployment-guide
✓ test/integration-tests
```

**Bad branch names:**
```
✗ my-feature               (missing type)
✗ Feature/AddModel         (use lowercase)
✗ feature/Add_Model        (use hyphens, not underscores)
✗ add-monitoring           (missing type prefix)
```

## Troubleshooting

### Hook Not Running

```bash
# Check if hook is executable
ls -la .git/hooks/pre-commit

# Make executable
chmod +x .git/hooks/pre-commit

# Verify hook exists
cat .git/hooks/pre-commit
```

### Hook Failing Unexpectedly

```bash
# Run hook manually to see detailed output
.git/hooks/pre-commit

# Check Python availability
which python3
python3 --version

# Check YAML validation dependencies
python3 -c "import yaml; print('YAML OK')"
```

### False Positives

**Secret detection:**
```python
# If you have non-secret text matching patterns, use:

# Acceptable: This is not a real password
password = "placeholder"  # nosec

# Or use environment variables
password = os.getenv("PASSWORD")
```

**Large files:**
```bash
# Track with Git LFS instead
git lfs track "*.pth"
git lfs track "*.h5"
```

## Hook Configuration

### Disable Specific Checks

Edit the hook file and comment out sections:

```bash
# Edit .git/hooks/pre-commit
vim .git/hooks/pre-commit

# Comment out a check section
# if [ -n "$PYTHON_FILES" ]; then
#     echo "Checking Python syntax..."
# fi
```

### Adjust Thresholds

```bash
# In pre-commit hook, modify:

# Change max file size (default 5MB)
MAX_FILE_SIZE=$((10 * 1024 * 1024))  # 10MB

# In commit-msg hook, modify:

# Change max first line length (default 72)
MAX_LENGTH=80
```

## CI/CD Integration

These hooks are also enforced in CI/CD pipelines:

- **GitHub Actions**: `.github/workflows/ci.yml`
- **Pre-commit framework**: `.pre-commit-config.yaml`

See [.github/workflows/README.md](../.github/workflows/README.md) for CI/CD setup.

## Best Practices

1. **Install hooks immediately** when cloning repository
2. **Don't bypass hooks** unless absolutely necessary
3. **Fix issues locally** before pushing
4. **Update hooks** when contributing improvements
5. **Document exceptions** when bypassing hooks

## Related Documentation

- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [.github/PULL_REQUEST_TEMPLATE.md](../.github/PULL_REQUEST_TEMPLATE.md) - PR template
- [MODELS.md](../MODELS.md) - Model versioning
- Module 003 Exercise 07 - Advanced Git techniques

---

**Last Updated**: 2024-01-31
**Maintained By**: ML Platform Team
