# Contributing to Production ML System

Thank you for contributing to our production ML system! This guide covers Git workflows, coding standards, and collaboration practices based on Module 003 best practices.

## Table of Contents

- [Quick Start](#quick-start)
- [Git Workflow](#git-workflow)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Branch Naming Convention](#branch-naming-convention)
- [Pull Request Process](#pull-request-process)
- [Model Versioning](#model-versioning)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Git Hooks](#git-hooks)

---

## Quick Start

```bash
# 1. Fork and clone repository
git clone https://github.com/YOUR-USERNAME/production-ml-system.git
cd production-ml-system

# 2. Install Git LFS (for model files)
git lfs install
git lfs pull

# 3. Install Git hooks
chmod +x hooks/*
cp hooks/* .git/hooks/

# 4. Create a feature branch
git checkout -b feature/your-feature-name

# 5. Make changes and commit
git add .
git commit -m "feat(api): add your feature"

# 6. Push and create PR
git push origin feature/your-feature-name
gh pr create
```

---

## Git Workflow

We follow the **Feature Branch Workflow** with **Fork and Pull Request** model.

### Workflow Diagram

```
main (production)
 ├── develop (staging)
 │    ├── feature/add-monitoring
 │    ├── feature/improve-latency
 │    └── fix/memory-leak
 └── hotfix/security-patch
```

### Step-by-Step Workflow

#### 1. Fork the Repository

```bash
# Fork via GitHub UI, then clone your fork
git clone https://github.com/YOUR-USERNAME/production-ml-system.git
cd production-ml-system

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL-OWNER/production-ml-system.git
git remote -v
```

#### 2. Create Feature Branch

```bash
# Update your local main
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

#### 3. Make Changes

```bash
# Make your changes
vim src/api/endpoints.py

# Stage changes
git add src/api/endpoints.py

# Commit with conventional format
git commit -m "feat(api): add health check endpoint"
```

#### 4. Keep Branch Updated

```bash
# Fetch upstream changes
git fetch upstream

# Rebase on upstream/main
git rebase upstream/main

# If conflicts, resolve them
git add <conflicted-files>
git rebase --continue
```

#### 5. Push to Your Fork

```bash
# Push to your fork
git push origin feature/your-feature-name

# If you rebased, force push (only to your fork!)
git push --force-with-lease origin feature/your-feature-name
```

#### 6. Create Pull Request

```bash
# Using GitHub CLI
gh pr create --title "feat(api): add health check endpoint" --body "..."

# Or use GitHub web interface
```

---

## Commit Message Guidelines

We follow **Conventional Commits** specification.

### Format

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

### Valid Types

| Type | Description | Example |
|------|-------------|---------|
| `feat` | New feature | `feat(api): add user authentication` |
| `fix` | Bug fix | `fix(model): correct preprocessing` |
| `docs` | Documentation | `docs: update API guide` |
| `style` | Code style | `style: format with black` |
| `refactor` | Code refactoring | `refactor(api): simplify error handling` |
| `test` | Add/update tests | `test(api): add integration tests` |
| `chore` | Maintenance | `chore: update dependencies` |
| `perf` | Performance | `perf(model): optimize inference` |
| `ci` | CI/CD changes | `ci: add Docker build workflow` |
| `build` | Build system | `build: update Dockerfile` |
| `revert` | Revert commit | `revert: revert feature X` |
| `model` | ML model updates | `model(bert): release v2.1.0` |

### Scope Examples

Common scopes in our project:
- `api` - API endpoints and routes
- `model` - ML model code
- `k8s` - Kubernetes manifests
- `docker` - Docker configurations
- `monitoring` - Monitoring and observability
- `ci` - CI/CD pipelines
- `db` - Database related

### Rules

1. **Subject line**:
   - Max 72 characters
   - Lowercase after colon
   - No period at end
   - Imperative mood ("add" not "added")

2. **Body** (optional):
   - Blank line after subject
   - Wrap at 72 characters
   - Explain *what* and *why*, not *how*

3. **Footer** (optional):
   - Reference issues: `Closes #123`
   - Breaking changes: `BREAKING CHANGE: description`

### Examples

**Good commits:**

```bash
# Simple feature
git commit -m "feat(api): add health check endpoint"

# With body
git commit -m "feat(model): add attention mechanism

Implement multi-head attention layers to improve accuracy.
Based on the Transformer architecture from 'Attention Is All You Need'.

Closes #234"

# Breaking change
git commit -m "feat(api)!: change authentication to OAuth2

BREAKING CHANGE: API now requires OAuth2 tokens instead of API keys.
Migration guide: docs/migration-oauth2.md

Closes #456"

# Model release
git commit -m "model(bert): release v2.1.0 with attention

Performance improvements:
- Accuracy: 96.2% (+0.7% vs v2.0.0)
- Latency: 38ms P95 (-4ms vs v2.0.0)

Git tag: model-v2.1.0"
```

**Bad commits:**

```bash
✗ "Added feature"              # Missing type
✗ "feat:"                      # Missing description
✗ "feat: Added Feature"        # Capital after colon
✗ "Updated code"               # Vague, missing type
✗ "WIP"                        # Not descriptive
```

---

## Branch Naming Convention

### Format

```
<type>/<description>
```

### Valid Patterns

**Main branches:**
- `main` - Production
- `master` - Production (legacy)
- `develop` - Staging
- `staging` - Staging environment
- `production` - Production environment

**Feature branches:**

| Type | Description | Example |
|------|-------------|---------|
| `feature/` | New features | `feature/add-model-monitoring` |
| `fix/` | Bug fixes | `fix/memory-leak-inference` |
| `hotfix/` | Critical fixes | `hotfix/security-vulnerability` |
| `refactor/` | Code refactoring | `refactor/api-error-handling` |
| `docs/` | Documentation | `docs/deployment-guide` |
| `test/` | Test additions | `test/integration-tests` |
| `chore/` | Maintenance | `chore/update-dependencies` |

### Rules

1. Use lowercase
2. Use hyphens, not underscores
3. Be descriptive but concise
4. Match the commit type

### Examples

**Good branch names:**
```
✓ feature/add-model-monitoring
✓ fix/memory-leak-inference
✓ hotfix/security-vulnerability
✓ refactor/api-error-handling
✓ docs/deployment-guide
```

**Bad branch names:**
```
✗ my-feature                # Missing type
✗ Feature/AddModel          # Use lowercase
✗ feature/Add_Model         # Use hyphens
✗ add-monitoring            # Missing type prefix
```

---

## Pull Request Process

### Before Creating PR

1. **Run tests locally:**
   ```bash
   pytest tests/ -v
   ```

2. **Check code quality:**
   ```bash
   black src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

3. **Update documentation** if needed

4. **Rebase on latest main:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

### Creating PR

1. **Use PR template** (auto-populated)

2. **Write clear title:**
   ```
   feat(api): add health check endpoint
   ```

3. **Provide context in description:**
   - What changes were made
   - Why the changes were needed
   - How to test the changes
   - Screenshots/demos if applicable
   - Links to related issues

4. **Add labels:**
   - `feature`, `bug`, `documentation`, etc.
   - `priority: high/medium/low`
   - `needs review`

5. **Request reviewers**

### PR Checklist

Before marking PR as ready:

- [ ] Code follows style guide
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] Branch rebased on latest main
- [ ] No merge conflicts
- [ ] CI checks passing
- [ ] Reviewed by at least 2 team members

### Code Review Guidelines

**For Authors:**
- Respond to all comments
- Don't take feedback personally
- Explain complex changes
- Update PR based on feedback

**For Reviewers:**
- Be respectful and constructive
- Focus on code, not the person
- Explain *why*, not just *what*
- Approve when ready

### Merging PR

**Merge strategies:**

1. **Squash and Merge** (default):
   - Use for feature branches
   - Creates single commit on main
   - Keeps history clean

2. **Rebase and Merge**:
   - Use for well-organized commits
   - Preserves commit history
   - Linear history

3. **Merge Commit**:
   - Use for releases
   - Preserves branch history

**After merge:**
```bash
# Delete remote branch
git push origin --delete feature/your-feature

# Delete local branch
git branch -d feature/your-feature

# Update your local main
git checkout main
git pull upstream main
```

---

## Model Versioning

We use **Semantic Versioning** for models: `MAJOR.MINOR.PATCH`

### Version Increments

**MAJOR** (X.0.0) - Breaking changes:
- Different model architecture
- Incompatible API changes
- Different input/output schema

**MINOR** (X.Y.0) - Backward-compatible improvements:
- Accuracy improvements
- Performance optimizations
- New features (compatible)

**PATCH** (X.Y.Z) - Bug fixes:
- Preprocessing bug fixes
- Small optimizations
- Documentation updates

### Releasing Models

#### 1. Train Model

```bash
# Run experiment
python scripts/train.py --config experiments/exp-2024-01-20.yaml

# Validate results
python scripts/validate_model.py --model models/new_model.pth
```

#### 2. Create Model Metadata

```yaml
# models/production/model-v2.1.0.yaml
model:
  name: bert-classifier
  version: 2.1.0
  architecture: BERT
  framework: pytorch
  framework_version: 2.1.0

performance:
  metrics:
    accuracy: 0.962
    f1_score: 0.958
  inference:
    latency_p95_ms: 38
```

#### 3. Commit Model Files

```bash
# Add model file (tracked by Git LFS)
git add models/production/model-v2.1.0.onnx
git add models/production/model-v2.1.0.yaml

# Commit
git commit -m "model(bert): release v2.1.0 with attention

Performance improvements:
- Accuracy: 96.2% (+0.7% vs v2.0.0)
- Latency: 38ms P95 (-4ms vs v2.0.0)

Training:
- Experiment: exp-2024-01-20-resnet50-attention
- Dataset: ImageNet-1K v2024.1
- 100 epochs, 8x A100 GPUs"
```

#### 4. Create Git Tag

```bash
# Create annotated tag
git tag -a model-v2.1.0 -m "Model Release v2.1.0

ResNet-50 with attention mechanism
- Accuracy: 96.2%
- F1 Score: 95.8%
- Latency P95: 38ms

Training:
- Experiment: exp-2024-01-20-resnet50-attention
- Dataset: ImageNet-1K v2024.1
- 100 epochs, 8x A100 GPUs

Improvements:
- +0.7% accuracy vs v2.0.0
- -4ms latency vs v2.0.0
- Added attention mechanism
- Improved data augmentation"

# Push tag
git push origin model-v2.1.0
```

#### 5. Update Registry

Update `MODELS.md` with new version details.

#### 6. Create GitHub Release

The `model-release.yml` workflow will automatically create a GitHub release when you push a model tag.

---

## Code Standards

### Python

**Style:**
- PEP 8 compliant
- Use `black` for formatting
- Use `isort` for imports
- Line length: 100 characters

**Type hints:**
```python
def predict(data: List[float], model: torch.nn.Module) -> np.ndarray:
    """Make prediction with model."""
    return model(data)
```

**Documentation:**
```python
def preprocess_image(image: Image, size: Tuple[int, int]) -> torch.Tensor:
    """Preprocess image for model input.

    Args:
        image: Input PIL image
        size: Target size (width, height)

    Returns:
        Preprocessed tensor ready for model

    Raises:
        ValueError: If image dimensions are invalid
    """
    pass
```

### Kubernetes Manifests

```yaml
# Use consistent naming
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-api
  labels:
    app: model-api
    version: v2.1.0
```

### Docker

```dockerfile
# Use multi-stage builds
FROM python:3.11-slim as builder
# Build stage

FROM python:3.11-slim
# Runtime stage
```

---

## Testing Requirements

### Test Types

1. **Unit Tests** (`tests/unit/`):
   - Test individual functions
   - Mock external dependencies
   - Fast execution

2. **Integration Tests** (`tests/integration/`):
   - Test component interactions
   - Use real services (DB, Redis)
   - Slower execution

3. **Model Tests** (`tests/model_validation/`):
   - Validate model accuracy
   - Test inference performance
   - Check model outputs

### Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test
pytest tests/unit/test_api.py::test_health_check -v
```

### Writing Tests

```python
def test_preprocess_image():
    """Test image preprocessing."""
    # Arrange
    image = Image.new('RGB', (256, 256))

    # Act
    result = preprocess_image(image, (224, 224))

    # Assert
    assert result.shape == (3, 224, 224)
    assert result.dtype == torch.float32
```

---

## Git Hooks

We use Git hooks to enforce quality standards automatically.

### Installation

```bash
# Copy hooks
chmod +x hooks/*
cp hooks/* .git/hooks/

# Verify installation
ls -la .git/hooks/
```

### Available Hooks

1. **pre-commit**:
   - Python syntax check
   - Debug statement detection
   - Secret detection
   - File size limits
   - YAML validation

2. **commit-msg**:
   - Conventional commit format
   - Message length limits
   - Description validation

3. **pre-push**:
   - Branch name validation
   - Test execution
   - Commit message validation

### Bypassing Hooks (Emergency Only)

```bash
# Skip pre-commit
git commit --no-verify -m "feat: emergency fix"

# Skip pre-push
git push --no-verify origin main
```

**⚠️ Only bypass hooks in emergencies!**

---

## Additional Resources

### Documentation

- [MODELS.md](MODELS.md) - Model registry and versioning
- [README.md](README.md) - Project overview
- [hooks/README.md](hooks/README.md) - Git hooks documentation
- [.github/workflows/README.md](.github/workflows/README.md) - CI/CD pipelines

### Module 003 References

Based on Module 003 best practices:
- Exercise 01: Git fundamentals
- Exercise 02: Commit best practices
- Exercise 03: Branching strategies
- Exercise 04: Merge and conflict resolution
- Exercise 05: Collaboration workflows
- Exercise 06: ML workflows with DVC
- Exercise 07: Advanced Git techniques
- Exercise 08: Git LFS for models

### External Resources

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)
- [Git LFS Documentation](https://git-lfs.github.com/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)

---

## Getting Help

- **Slack**: #ml-platform
- **Email**: ml-platform@company.com
- **GitHub Issues**: [Create an issue](../../issues/new)
- **Weekly Office Hours**: Thursdays 2-3pm

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

**Last Updated**: 2024-01-31
**Maintained By**: ML Platform Team
**Review Schedule**: Monthly

Thank you for contributing! 🎉
