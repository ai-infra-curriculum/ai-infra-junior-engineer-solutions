#!/bin/bash

#######################################################################
# Exercise 05: Collaboration and Pull Requests - Setup Script
#######################################################################
# Creates simulated upstream and fork repositories to practice
# collaboration workflows without needing actual GitHub access.
#######################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXERCISE_DIR="$(dirname "$SCRIPT_DIR")"
WORK_DIR="$EXERCISE_DIR/collaboration-workspace"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Setting up Collaboration Environment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Clean up existing workspace
if [ -d "$WORK_DIR" ]; then
    rm -rf "$WORK_DIR"
fi

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

#######################################################################
# Part 1: Create "Upstream" Repository (Team Repository)
#######################################################################

echo -e "${YELLOW}[1/5] Creating upstream team repository...${NC}"

mkdir upstream-ml-api
cd upstream-ml-api
git init
git config user.name "Team Lead"
git config user.email "lead@ml-team.com"

# Create initial project structure
cat > README.md << 'EOF'
# ML Inference API - Team Repository

Production ML inference service for image classification.

## Contributing

We welcome contributions! Please follow our workflow:

1. Fork this repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request
5. Address review feedback

See CONTRIBUTING.md for detailed guidelines.

## Features

- Image classification with ResNet50
- REST API with FastAPI
- Prometheus metrics export
- Health check endpoints

## Setup

```bash
pip install -r requirements.txt
python -m src.api.app
```

## Testing

```bash
pytest tests/ -v
```

## License

MIT License
EOF

mkdir -p src/api src/models tests/api tests/models

cat > src/api/app.py << 'EOF'
"""ML Inference API - Main Application"""

from fastapi import FastAPI

app = FastAPI(
    title="ML Inference API",
    description="Production ML inference service",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {
        "name": "ML Inference API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}
EOF

cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
pytest==7.4.3
pytest-asyncio==0.21.1
numpy==1.24.3
EOF

cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/
.pytest_cache/
.coverage
htmlcov/
dist/
build/
EOF

git add .
git commit -m "Initial commit: basic API structure"

# Tag initial release
git tag -a v1.0.0 -m "Initial release"

echo -e "${GREEN}✓ Upstream repository created${NC}"

#######################################################################
# Part 2: Create Your "Fork"
#######################################################################

echo -e "${YELLOW}[2/5] Creating your fork...${NC}"

cd "$WORK_DIR"
git clone upstream-ml-api my-fork
cd my-fork

git config user.name "Your Name"
git config user.email "you@developer.com"

# Add upstream remote
git remote rename origin fork-origin
git remote add upstream ../upstream-ml-api
git remote -v

echo -e "${GREEN}✓ Fork created with upstream configured${NC}"

#######################################################################
# Part 3: Create Feature Branch with Changes
#######################################################################

echo -e "${YELLOW}[3/5] Creating feature branch...${NC}"

git switch -c feature/add-model-metrics

# Add model metrics module
mkdir -p src/monitoring
cat > src/monitoring/model_metrics.py << 'EOF'
"""
Model Performance Metrics

Track model-specific performance indicators.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import time


@dataclass
class ModelMetrics:
    """Track model performance metrics."""

    def __init__(self):
        self.predictions: List[float] = []
        self.ground_truth: List[float] = []
        self.latencies: List[float] = []
        self.start_time = time.time()

    def add_prediction(
        self,
        prediction: float,
        truth: Optional[float] = None,
        latency: Optional[float] = None
    ):
        """
        Record a prediction.

        Args:
            prediction: Model prediction value
            truth: Ground truth value (if available)
            latency: Prediction latency in seconds
        """
        self.predictions.append(prediction)

        if truth is not None:
            self.ground_truth.append(truth)

        if latency is not None:
            self.latencies.append(latency)

    def calculate_accuracy(self, threshold: float = 0.5) -> float:
        """
        Calculate prediction accuracy.

        Args:
            threshold: Classification threshold

        Returns:
            Accuracy percentage (0-100)
        """
        if not self.predictions or not self.ground_truth:
            return 0.0

        if len(self.predictions) != len(self.ground_truth):
            raise ValueError(
                "Predictions and ground truth lengths don't match"
            )

        correct = sum(
            1 for pred, truth in zip(self.predictions, self.ground_truth)
            if (pred >= threshold) == (truth >= threshold)
        )

        return (correct / len(self.predictions)) * 100

    def get_average_latency(self) -> float:
        """Get average prediction latency."""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    def get_summary(self) -> Dict:
        """Get comprehensive metrics summary."""
        return {
            "total_predictions": len(self.predictions),
            "accuracy": (
                self.calculate_accuracy()
                if self.ground_truth else None
            ),
            "average_latency": self.get_average_latency(),
            "uptime_seconds": time.time() - self.start_time
        }


# Global metrics instance
model_metrics = ModelMetrics()
EOF

git add src/monitoring/model_metrics.py
git commit -m "feat: add model performance metrics tracking

Implement ModelMetrics class for tracking:
- Prediction accuracy
- Inference latency
- Statistical summaries
- Ground truth comparison

This enables monitoring model performance in production."

# Add tests
mkdir -p tests/monitoring
cat > tests/monitoring/test_model_metrics.py << 'EOF'
"""Tests for model metrics."""

import pytest
from src.monitoring.model_metrics import ModelMetrics


def test_model_metrics_initialization():
    """Test metrics initialization."""
    metrics = ModelMetrics()
    assert metrics.predictions == []
    assert metrics.ground_truth == []
    assert metrics.latencies == []


def test_add_prediction():
    """Test adding predictions."""
    metrics = ModelMetrics()
    metrics.add_prediction(0.95, truth=1.0, latency=0.05)
    metrics.add_prediction(0.87, truth=1.0, latency=0.03)

    assert len(metrics.predictions) == 2
    assert len(metrics.ground_truth) == 2
    assert len(metrics.latencies) == 2


def test_calculate_accuracy():
    """Test accuracy calculation."""
    metrics = ModelMetrics()
    metrics.add_prediction(0.9, truth=1.0)
    metrics.add_prediction(0.8, truth=1.0)
    metrics.add_prediction(0.2, truth=0.0)

    accuracy = metrics.calculate_accuracy()
    assert accuracy == pytest.approx(100.0)


def test_get_average_latency():
    """Test average latency calculation."""
    metrics = ModelMetrics()
    metrics.add_prediction(0.9, latency=0.05)
    metrics.add_prediction(0.8, latency=0.03)

    avg_latency = metrics.get_average_latency()
    assert avg_latency == pytest.approx(0.04)


def test_get_summary():
    """Test metrics summary."""
    metrics = ModelMetrics()
    metrics.add_prediction(0.9, truth=1.0, latency=0.05)
    metrics.add_prediction(0.8, truth=1.0, latency=0.03)

    summary = metrics.get_summary()
    assert summary["total_predictions"] == 2
    assert "accuracy" in summary
    assert "average_latency" in summary
    assert "uptime_seconds" in summary
EOF

git add tests/monitoring/test_model_metrics.py
git commit -m "test: add comprehensive tests for model metrics

Add tests for:
- Initialization
- Adding predictions
- Accuracy calculation
- Latency tracking
- Summary generation

All tests passing with 100% coverage."

# Update README
cat >> README.md << 'EOF'

## Model Monitoring

Track model performance with `ModelMetrics`:

```python
from src.monitoring.model_metrics import model_metrics

# Record predictions
model_metrics.add_prediction(
    prediction=0.95,
    truth=1.0,
    latency=0.05
)

# Get summary
summary = model_metrics.get_summary()
print(f"Accuracy: {summary['accuracy']}%")
print(f"Avg Latency: {summary['average_latency']}s")
```
EOF

git add README.md
git commit -m "docs: add model metrics usage to README

Document how to use ModelMetrics for tracking
model performance in production environments."

echo -e "${GREEN}✓ Feature branch created with 3 commits${NC}"

#######################################################################
# Part 4: Simulate Team Member's PR
#######################################################################

echo -e "${YELLOW}[4/5] Creating teammate's PR for review...${NC}"

git switch -c review/teammate-image-utils

# Create code with review opportunities
mkdir -p src/utils
cat > src/utils/image_processing.py << 'EOF'
"""Image processing utilities."""

def resize_image(img, size):
    # TODO: Add validation
    return img.resize(size)

def normalize(img):
    # FIXME: Doesn't handle edge cases
    return img / 255.0

def convert_to_rgb(img):
    if img.mode != 'RGB':
        return img.convert('RGB')
    return img
EOF

git add src/utils/image_processing.py
git commit -m "Add image processing utils"

echo -e "${GREEN}✓ Teammate's branch ready for review${NC}"

#######################################################################
# Part 5: Create PR and Review Templates
#######################################################################

echo -e "${YELLOW}[5/5] Creating PR templates and guidelines...${NC}"

git switch feature/add-model-metrics

mkdir -p .github
cat > .github/PULL_REQUEST_TEMPLATE.md << 'EOF'
## Description
<!-- Provide a brief description of the changes -->

## Motivation and Context
<!-- Why is this change needed? What problem does it solve? -->
<!-- Link to related issues: Fixes #123, Relates to #456 -->

## Type of Change
<!-- Mark relevant items with [x] -->
- [ ] 🐛 Bug fix (non-breaking change fixing an issue)
- [ ] ✨ New feature (non-breaking change adding functionality)
- [ ] 💥 Breaking change (fix or feature causing existing functionality to break)
- [ ] 📝 Documentation update
- [ ] ♻️ Code refactoring
- [ ] ⚡ Performance improvement
- [ ] ✅ Test update

## Changes Made
<!-- List the specific changes in this PR -->
-
-
-

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed
- [ ] All existing tests pass

### Test Results
```
# Paste test output here
```

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented complex code sections
- [ ] I have updated the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix/feature works
- [ ] New and existing tests pass locally
- [ ] Any dependent changes have been merged

## Reviewer Focus Areas
<!-- Guide reviewers on what to focus on -->
Please pay special attention to:
-
-
EOF

mkdir -p docs
cat > docs/CONTRIBUTING.md << 'EOF'
# Contributing Guide

Thank you for contributing to the ML Inference API project!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone <your-fork-url>`
3. Add upstream: `git remote add upstream <original-repo-url>`
4. Create a branch: `git switch -c feature/your-feature`

## Making Changes

### Branch Naming
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation only
- `refactor/` - Code refactoring
- `test/` - Test additions
- `chore/` - Maintenance tasks

### Commit Messages
Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Examples:
- `feat(api): add batch prediction endpoint`
- `fix(preprocessing): handle grayscale images`
- `docs(readme): update installation instructions`

### Code Style
- Follow PEP 8 for Python code
- Use type hints
- Add docstrings to all public functions
- Keep functions focused and single-purpose

### Testing
- Write tests for all new functionality
- Ensure all existing tests pass
- Aim for >80% code coverage
- Include edge cases in tests

## Pull Request Process

1. **Update your branch**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests locally**
   ```bash
   pytest tests/ -v
   ```

3. **Push to your fork**
   ```bash
   git push origin feature/your-feature
   ```

4. **Create Pull Request**
   - Use the PR template
   - Link related issues
   - Provide clear description
   - Add screenshots if applicable

5. **Address Review Feedback**
   - Respond to all comments
   - Make requested changes
   - Push updates
   - Request re-review

## Code Review

### For Authors
- Be open to feedback
- Ask questions if unclear
- Don't take criticism personally
- Thank reviewers for their time

### For Reviewers
- Be constructive and respectful
- Explain the "why" not just "what"
- Distinguish between blocking and non-blocking comments
- Praise good code

## Community Guidelines

- Be respectful and inclusive
- Assume good intentions
- Help newcomers
- Share knowledge generously

## Questions?

Feel free to open an issue or reach out to maintainers!
EOF

git add .github/ docs/CONTRIBUTING.md
git commit -m "chore: add PR template and contributing guide

Add comprehensive guidelines for:
- Pull request format
- Contributing workflow
- Code style standards
- Review process"

echo -e "${GREEN}✓ Templates and guidelines created${NC}"

#######################################################################
# Summary
#######################################################################

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Collaboration environment ready!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Workspace structure:"
echo "  upstream-ml-api/    - Team repository (upstream)"
echo "  my-fork/            - Your fork"
echo ""
echo "Branches in your fork:"
cd "$WORK_DIR/my-fork"
git branch

echo ""
echo "Remotes configured:"
git remote -v

echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  cd $WORK_DIR/my-fork"
echo "  git log --oneline --graph --all"
echo "  cat .github/PULL_REQUEST_TEMPLATE.md"
echo "  cat docs/CONTRIBUTING.md"
