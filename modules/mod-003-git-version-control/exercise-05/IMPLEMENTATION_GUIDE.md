# Exercise 05: Collaboration and Pull Requests - Implementation Guide

## Overview

Master professional Git collaboration workflows including forking, pull requests, code reviews, and team development practices. Learn the complete GitHub workflow used by ML engineering teams.

**Estimated Time**: 90-120 minutes
**Difficulty**: Intermediate
**Prerequisites**: Exercise 04 - Merging and Conflicts

## What You'll Learn

- ✅ Fork and clone repositories
- ✅ Work with multiple remotes
- ✅ Create professional pull requests
- ✅ Conduct effective code reviews
- ✅ Address review feedback
- ✅ Sync with upstream repositories
- ✅ Handle concurrent development
- ✅ PR templates and best practices

---

## Part 1: Understanding Collaborative Git

### Step 1.1: Collaboration Models

**Two Main Models:**

**1. Fork & Pull Model (Open Source):**
```
Upstream Repo (Organization)
    ↓
Your Fork (Your Account)
    ↓
Local Clone (Your Machine)
    ↓
Pull Request → Upstream Repo
```

**2. Shared Repository Model (Team):**
```
Team Repo
    ↓
Local Clone (Your Machine)
    ↓
Feature Branch → Pull Request → Main Branch
```

### Step 1.2: Set Up GitHub CLI

```bash
# Install GitHub CLI
# macOS
brew install gh

# Linux (Debian/Ubuntu)
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | \
  sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | \
  sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh

# Verify installation
gh --version

# Authenticate with GitHub
gh auth login

# Follow prompts:
# 1. Select: GitHub.com
# 2. Select: HTTPS
# 3. Select: Login with a web browser
# 4. Copy one-time code
# 5. Press Enter to open browser
# 6. Paste code and authorize

# Verify authentication
gh auth status
# Output:
# ✓ Logged in to github.com as YourUsername
```

---

## Part 2: Fork and Clone Workflow

### Step 2.1: Simulate Upstream Repository

For this exercise, we'll create a local "upstream" repository.

```bash
# Create upstream repository (simulates organization repo)
cd /tmp
mkdir ml-inference-api-upstream
cd ml-inference-api-upstream
git init --initial-branch=main

# Create initial project structure
cat > README.md << 'EOF'
# ML Inference API

Production-ready ML inference service for image classification.

## Features

- FastAPI-based REST API
- PyTorch model serving
- Prometheus metrics
- Docker deployment

## Contributing

We welcome contributions! Please:
1. Fork this repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

See CONTRIBUTING.md for details.

## Setup

```bash
pip install -r requirements.txt
python src/main.py
```

## Testing

```bash
pytest tests/
```
EOF

mkdir -p src/api src/models configs tests
cat > src/api/__init__.py << 'EOF'
"""API module."""
EOF

cat > src/models/__init__.py << 'EOF'
"""Models module."""
EOF

cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn==0.24.0
torch==2.1.0
pydantic==2.5.0
prometheus-client==0.19.0
pytest==7.4.3
EOF

cat > .gitignore << 'EOF'
__pycache__/
*.pyc
.pytest_cache/
.env
models/*.pth
venv/
.venv/
EOF

git add .
git commit -m "Initial commit: ML Inference API

Setup initial project structure:
- FastAPI application skeleton
- Model serving infrastructure
- Testing framework
- Docker configuration
- Documentation"

# Tag initial release
git tag -a v1.0.0 -m "Release v1.0.0: Initial production release"

echo "Upstream repository created at /tmp/ml-inference-api-upstream"
```

### Step 2.2: Fork the Repository

```bash
# Clone the "upstream" repo to simulate forking
cd ~
mkdir -p workspace
cd workspace
git clone /tmp/ml-inference-api-upstream ml-inference-api
cd ml-inference-api

# Configure remote for fork model
git remote rename origin upstream

# Simulate your fork (in real scenario, this would be your GitHub fork)
git remote add origin /tmp/ml-inference-api-upstream

# View remotes
git remote -v
# Output:
# origin    /tmp/ml-inference-api-upstream (fetch)
# origin    /tmp/ml-inference-api-upstream (push)
# upstream  /tmp/ml-inference-api-upstream (fetch)
# upstream  /tmp/ml-inference-api-upstream (push)

# In real GitHub workflow:
# gh repo fork organization/ml-inference-api --clone
# This automatically:
# - Forks repo to your account
# - Clones to local machine
# - Sets up upstream remote
```

### Step 2.3: Sync with Upstream

```bash
# Fetch latest changes from upstream
git fetch upstream

# View remote branches
git branch -r
# Output:
# upstream/main

# Update your local main
git switch main
git merge upstream/main --ff-only

# In real workflow, push to your fork:
# git push origin main
```

---

## Part 3: Creating a Pull Request

### Step 3.1: Create Feature Branch

```bash
# Always branch from latest main
git switch main
git pull upstream main  # Ensure you're up-to-date

# Create descriptive feature branch
git switch -c feature/add-batch-prediction

# Implement the feature
cat > src/api/batch.py << 'EOF'
"""
Batch Prediction Endpoint

Process multiple images in a single request for improved throughput.
"""

from typing import List, Dict
from fastapi import APIRouter, HTTPException, UploadFile, File
import structlog

from src.models.inference import ImageClassifier

logger = structlog.get_logger()

router = APIRouter(prefix="/batch", tags=["batch"])

classifier = ImageClassifier()


@router.post("/predict")
async def batch_predict(files: List[UploadFile] = File(...)) -> Dict:
    """
    Predict classes for multiple images.

    Args:
        files: List of image files (max 32)

    Returns:
        Dictionary with predictions for each file

    Raises:
        HTTPException: If batch size exceeded or processing fails
    """
    # Validate batch size
    if len(files) > 32:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(files)} exceeds maximum of 32"
        )

    logger.info("batch_prediction_started", batch_size=len(files))

    results = []
    errors = []

    for idx, file in enumerate(files):
        try:
            # Validate file type
            if not file.content_type.startswith("image/"):
                errors.append({
                    "file_index": idx,
                    "filename": file.filename,
                    "error": "Invalid file type"
                })
                continue

            # Read and process
            image_bytes = await file.read()
            prediction = await classifier.predict(image_bytes)

            results.append({
                "file_index": idx,
                "filename": file.filename,
                "prediction": prediction
            })

        except Exception as e:
            logger.error("batch_prediction_failed", idx=idx, error=str(e))
            errors.append({
                "file_index": idx,
                "filename": file.filename,
                "error": str(e)
            })

    logger.info(
        "batch_prediction_completed",
        total=len(files),
        successful=len(results),
        failed=len(errors)
    )

    return {
        "total_files": len(files),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors
    }
EOF

git add src/api/batch.py
git commit -m "feat(api): add batch prediction endpoint

Implement batch processing for multiple images:
- Process up to 32 images per request
- Individual error handling per image
- Comprehensive logging
- Input validation

Improves throughput for bulk inference workloads.

Closes #156"

# Add tests
cat > tests/api/test_batch.py << 'EOF'
"""Tests for batch prediction endpoint."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
import io
from PIL import Image

from src.main import app

client = TestClient(app)


@pytest.fixture
def sample_image():
    """Create sample image file."""
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


def test_batch_predict_success(sample_image):
    """Test successful batch prediction."""
    files = [
        ("files", ("image1.jpg", sample_image, "image/jpeg")),
        ("files", ("image2.jpg", sample_image, "image/jpeg"))
    ]

    with patch('src.api.batch.classifier.predict', new_callable=AsyncMock) as mock_predict:
        mock_predict.return_value = {"class": "cat", "confidence": 0.95}

        response = client.post("/batch/predict", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["total_files"] == 2
        assert data["successful"] == 2
        assert len(data["results"]) == 2


def test_batch_predict_exceeds_max_size():
    """Test batch size validation."""
    # Create 33 files (exceeds max of 32)
    files = [
        ("files", (f"image{i}.jpg", io.BytesIO(b"fake"), "image/jpeg"))
        for i in range(33)
    ]

    response = client.post("/batch/predict", files=files)

    assert response.status_code == 400
    assert "exceeds maximum" in response.json()["detail"]


def test_batch_predict_invalid_file_type(sample_image):
    """Test handling of invalid file types."""
    files = [
        ("files", ("image.jpg", sample_image, "image/jpeg")),
        ("files", ("document.txt", io.BytesIO(b"text"), "text/plain"))
    ]

    with patch('src.api.batch.classifier.predict', new_callable=AsyncMock) as mock_predict:
        mock_predict.return_value = {"class": "cat", "confidence": 0.95}

        response = client.post("/batch/predict", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["successful"] == 1
        assert data["failed"] == 1
        assert data["errors"][0]["error"] == "Invalid file type"
EOF

git add tests/api/test_batch.py
git commit -m "test(api): add comprehensive tests for batch endpoint

Test coverage:
- Successful batch predictions
- Batch size validation
- Invalid file type handling
- Error recovery

Achieves 100% test coverage for batch module."

# Update documentation
cat >> README.md << 'EOF'

## Batch Prediction

Process multiple images in a single request:

```python
import requests

files = [
    ('files', open('image1.jpg', 'rb')),
    ('files', open('image2.jpg', 'rb'))
]

response = requests.post('http://localhost:8000/batch/predict', files=files)
results = response.json()
```

**Limits:**
- Maximum 32 images per batch
- Supported formats: JPEG, PNG
- Individual errors don't fail entire batch
EOF

git add README.md
git commit -m "docs: add batch prediction usage to README

Document batch endpoint usage:
- Example code
- Limits and constraints
- Error handling behavior"

# View your commits
git log --oneline upstream/main..HEAD
# Output:
# abc1234 (HEAD -> feature/add-batch-prediction) docs: add batch prediction usage to README
# def5678 test(api): add comprehensive tests for batch endpoint
# ghi9012 feat(api): add batch prediction endpoint
```

### Step 3.2: Push and Create Pull Request

```bash
# In real workflow, push to your fork
# git push origin feature/add-batch-prediction

# Create PR using GitHub CLI
# gh pr create \
#   --title "Add batch prediction endpoint" \
#   --body-file .github/pr-description.md \
#   --base main \
#   --head feature/add-batch-prediction

# Or create PR on GitHub web interface:
# 1. Push branch to your fork
# 2. Visit fork on GitHub
# 3. Click "Compare & pull request"
# 4. Fill in details
# 5. Click "Create pull request"

# Create comprehensive PR description
cat > pr-description.md << 'EOF'
# Add Batch Prediction Endpoint

## Summary
Implements batch prediction endpoint for processing multiple images in a single request, improving throughput for bulk inference workloads.

## Motivation
Users running large-scale inference need to process hundreds or thousands of images. Making individual API calls is inefficient due to:
- Network overhead
- Connection establishment costs
- Sequential processing limitations

Batch processing addresses these issues by:
- Reducing HTTP overhead
- Enabling parallel processing
- Improving overall throughput by 3-5x

## Type of Change
- [x] ✨ New feature (non-breaking change adding functionality)
- [ ] 🐛 Bug fix
- [ ] 💥 Breaking change
- [ ] 📝 Documentation update

## Changes Made
- **New batch prediction endpoint** (`POST /batch/predict`)
  - Accepts up to 32 images per request
  - Validates batch size
  - Individual error handling per image
  - Comprehensive logging

- **Test suite** with 100% coverage
  - Successful batch prediction tests
  - Batch size validation tests
  - Invalid file type handling tests
  - Error recovery tests

- **Documentation updates**
  - Usage examples in README
  - API constraints documented
  - Error handling behavior explained

## Testing

### Unit Tests
```bash
$ pytest tests/api/test_batch.py -v

tests/api/test_batch.py::test_batch_predict_success PASSED
tests/api/test_batch.py::test_batch_predict_exceeds_max_size PASSED
tests/api/test_batch.py::test_batch_predict_invalid_file_type PASSED

========================= 3 passed in 0.45s =========================
```

### Manual Testing
Tested with:
- 1 image: ✅ 150ms response time
- 10 images: ✅ 850ms response time
- 32 images: ✅ 2.5s response time
- 33 images: ✅ Rejected with 400 error

### Integration Testing
- [x] Tested with production PyTorch model
- [x] Tested with various image formats (JPEG, PNG)
- [x] Tested error scenarios
- [x] Tested concurrent requests

## Performance Impact
**Throughput Improvement:**
- Single requests: ~6.7 req/sec
- Batch requests (32): ~40 images/sec (6x improvement)

**Latency:**
- Per-image latency increases slightly (batching overhead)
- Overall throughput significantly improved
- Acceptable trade-off for bulk workloads

**Memory:**
- Temporary increase during batch processing
- Memory released after batch completion
- No memory leaks detected

## API Changes
**New Endpoint:**
```
POST /batch/predict
Content-Type: multipart/form-data

files: List[UploadFile]  # Max 32 files

Response: {
  "total_files": int,
  "successful": int,
  "failed": int,
  "results": [
    {
      "file_index": int,
      "filename": str,
      "prediction": {...}
    }
  ],
  "errors": [
    {
      "file_index": int,
      "filename": str,
      "error": str
    }
  ]
}
```

## Security Considerations
- ✅ File type validation (images only)
- ✅ Batch size limits (prevents DoS)
- ✅ No temporary files stored on disk
- ✅ Memory cleaned up after processing
- ✅ Error messages don't leak sensitive info

## Checklist
- [x] Code follows project style guidelines
- [x] Self-review completed
- [x] Complex logic commented
- [x] Documentation updated
- [x] No new warnings generated
- [x] Tests added with 100% coverage
- [x] All existing tests pass
- [x] Performance tested

## Related Issues
Closes #156
Related to #148 (bulk inference epic)

## Reviewer Focus Areas
Please pay special attention to:
1. **Error handling** - Each image error is isolated
2. **Batch size validation** - Ensure limit is appropriate
3. **Memory usage** - No leaks during batch processing
4. **Test coverage** - All edge cases covered

## Deployment Notes
**Configuration:** No changes needed
**Database migrations:** None
**Dependencies:** No new dependencies
**Breaking changes:** None (backward compatible)

## Next Steps
After this PR:
- [ ] Add batch endpoint to API documentation
- [ ] Update client SDK with batch support
- [ ] Add batch metrics to monitoring dashboard
- [ ] Consider implementing async batch queue (future PR)
EOF

echo "PR description ready! Would be submitted with:"
echo "gh pr create --body-file pr-description.md"
```

---

## Part 4: Code Review Process

### Step 4.1: Receiving a Code Review

Simulate receiving review comments:

```bash
# Create review feedback document
cat > review-feedback.md << 'EOF'
# Code Review: Add Batch Prediction Endpoint

## Overall
Great work on the batch endpoint! The implementation is solid. I have a few suggestions below.

## Approval Status
**🔄 Request Changes** - Please address the items marked as "Required"

---

## Comments

### src/api/batch.py

**Line 45: Batch size limit**
❓ **Question**: Why is the batch size limit hardcoded to 32?

Consider making this configurable:
```python
from src.config import settings

MAX_BATCH_SIZE = settings.batch_max_size or 32

if len(files) > MAX_BATCH_SIZE:
    raise HTTPException(...)
```

**Severity**: Nice to have

---

**Line 58-65: Error handling**
⚠️ **Important**: The loop continues on errors, but doesn't track which files succeeded.

Current behavior could be confusing if some files fail. Consider adding a "status" field to each result:

```python
results.append({
    "file_index": idx,
    "filename": file.filename,
    "status": "success",
    "prediction": prediction
})
```

**Severity**: Required

---

**Line 70-78: Exception handling**
🚨 **Blocker**: Catching all exceptions with broad `except Exception` is risky.

Should catch specific exceptions and handle appropriately:
```python
except ValidationError as e:
    errors.append({...})
except ModelError as e:
    errors.append({...})
except Exception as e:
    logger.error("unexpected_error", exc_info=True)
    errors.append({...})
```

**Severity**: Required

---

### tests/api/test_batch.py

**Line 35: Test coverage**
💅 **Nitpick**: Missing test for partial failures (some succeed, some fail).

Please add test:
```python
def test_batch_predict_partial_failure():
    """Test handling of mixed success/failure."""
    ...
```

**Severity**: Nice to have

---

**Line 55: Mock usage**
👍 **Nice**: Great use of AsyncMock for async function testing!

---

## Required Changes
- [ ] Fix broad exception handling (BLOCKER)
- [ ] Add status field to results (IMPORTANT)

## Suggested Changes
- [ ] Make batch size configurable
- [ ] Add partial failure test

## Questions
- How does this perform with very large images (>10MB)?
- Have you tested with concurrent batch requests?

---

**Review by**: @tech-lead
**Date**: 2025-11-01
EOF

echo "Review feedback received!"
```

### Step 4.2: Address Review Feedback

```bash
# Address the required changes
git switch feature/add-batch-prediction

# 1. Fix exception handling (BLOCKER)
cat > src/api/batch.py << 'EOF'
"""
Batch Prediction Endpoint

Process multiple images in a single request for improved throughput.
"""

from typing import List, Dict
from fastapi import APIRouter, HTTPException, UploadFile, File
import structlog

from src.models.inference import ImageClassifier
from src.models.exceptions import ModelError, ValidationError
from src.config import settings

logger = structlog.get_logger()

router = APIRouter(prefix="/batch", tags=["batch"])

classifier = ImageClassifier()

# Make batch size configurable
MAX_BATCH_SIZE = getattr(settings, 'batch_max_size', 32)


@router.post("/predict")
async def batch_predict(files: List[UploadFile] = File(...)) -> Dict:
    """
    Predict classes for multiple images.

    Args:
        files: List of image files (max configurable)

    Returns:
        Dictionary with predictions for each file

    Raises:
        HTTPException: If batch size exceeded or processing fails
    """
    # Validate batch size
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(files)} exceeds maximum of {MAX_BATCH_SIZE}"
        )

    logger.info("batch_prediction_started", batch_size=len(files))

    results = []
    errors = []

    for idx, file in enumerate(files):
        try:
            # Validate file type
            if not file.content_type.startswith("image/"):
                errors.append({
                    "file_index": idx,
                    "filename": file.filename,
                    "status": "error",
                    "error": "Invalid file type"
                })
                continue

            # Read and process
            image_bytes = await file.read()
            prediction = await classifier.predict(image_bytes)

            results.append({
                "file_index": idx,
                "filename": file.filename,
                "status": "success",
                "prediction": prediction
            })

        except ValidationError as e:
            logger.warning("batch_validation_error", idx=idx, error=str(e))
            errors.append({
                "file_index": idx,
                "filename": file.filename,
                "status": "error",
                "error": f"Validation error: {str(e)}"
            })

        except ModelError as e:
            logger.error("batch_model_error", idx=idx, error=str(e))
            errors.append({
                "file_index": idx,
                "filename": file.filename,
                "status": "error",
                "error": f"Model error: {str(e)}"
            })

        except Exception as e:
            logger.error("batch_unexpected_error", idx=idx, error=str(e), exc_info=True)
            errors.append({
                "file_index": idx,
                "filename": file.filename,
                "status": "error",
                "error": "Internal server error"
            })

    logger.info(
        "batch_prediction_completed",
        total=len(files),
        successful=len(results),
        failed=len(errors)
    )

    return {
        "total_files": len(files),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors
    }
EOF

git add src/api/batch.py
git commit -m "refactor(batch): improve exception handling and configuration

Address code review feedback:
- Replace broad exception catching with specific exceptions
- Add status field to all results
- Make batch size configurable via settings
- Improve error categorization
- Add exc_info for unexpected errors

Addresses review by @tech-lead"

# 2. Add partial failure test
cat >> tests/api/test_batch.py << 'EOF'


def test_batch_predict_partial_failure(sample_image):
    """Test handling of mixed success and failure."""
    files = [
        ("files", ("image1.jpg", sample_image, "image/jpeg")),
        ("files", ("document.txt", io.BytesIO(b"text"), "text/plain")),
        ("files", ("image2.jpg", sample_image, "image/jpeg"))
    ]

    with patch('src.api.batch.classifier.predict', new_callable=AsyncMock) as mock_predict:
        mock_predict.return_value = {"class": "cat", "confidence": 0.95}

        response = client.post("/batch/predict", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["total_files"] == 3
        assert data["successful"] == 2
        assert data["failed"] == 1

        # Verify success results have status
        for result in data["results"]:
            assert result["status"] == "success"

        # Verify error results have status
        for error in data["errors"]:
            assert error["status"] == "error"
EOF

git add tests/api/test_batch.py
git commit -m "test(batch): add partial failure test case

Add test for mixed success/failure scenarios:
- Validates status field on all results
- Tests error isolation
- Ensures successful items still process

Addresses review feedback."

# Create response to reviewer
cat > review-response.md << 'EOF'
# Response to Code Review

Thanks @tech-lead for the thorough review! I've addressed all the feedback.

## Changes Made

### ✅ Required Changes

**1. Fixed broad exception handling (BLOCKER)**
- Replaced `except Exception` with specific exception types
- Added `ValidationError` handling for input validation failures
- Added `ModelError` handling for inference failures
- Added `exc_info=True` for unexpected errors to aid debugging
- Improved error messages for users

**2. Added status field to results (IMPORTANT)**
- All results now have `"status": "success"`
- All errors now have `"status": "error"`
- Makes it easy to filter results programmatically

### ✅ Suggested Changes

**3. Made batch size configurable**
- Moved hardcoded `32` to `settings.batch_max_size`
- Falls back to 32 if not configured
- Easy to adjust for different deployment environments

**4. Added partial failure test**
- New test: `test_batch_predict_partial_failure`
- Validates mixed success/error scenarios
- Ensures status field on all responses
- Test coverage now 100%

## Test Results

```bash
$ pytest tests/api/test_batch.py -v

tests/api/test_batch.py::test_batch_predict_success PASSED
tests/api/test_batch.py::test_batch_predict_exceeds_max_size PASSED
tests/api/test_batch.py::test_batch_predict_invalid_file_type PASSED
tests/api/test_batch.py::test_batch_predict_partial_failure PASSED

========================= 4 passed in 0.52s =========================
```

## Answers to Questions

**Q: How does this perform with very large images (>10MB)?**
A: Tested with 15MB images - works fine but slower (~500ms per image). We may want to add a file size limit in future PR. Added issue #167 to track.

**Q: Have you tested with concurrent batch requests?**
A: Yes! Tested with 10 concurrent batch requests of 32 images each. No issues. FastAPI's async handling works well here.

## Summary of Changes

- Commit 1: `refactor(batch): improve exception handling and configuration`
- Commit 2: `test(batch): add partial failure test case`

All tests passing ✅
Ready for re-review!
EOF

# View updated commits
git log --oneline upstream/main..HEAD
```

---

## Part 5: Best Practices

### Step 5.1: Create PR Template

```bash
# Create comprehensive PR template for the project
mkdir -p .github
cat > .github/PULL_REQUEST_TEMPLATE.md << 'EOF'
## Description
<!-- Brief description of changes -->

## Motivation
<!-- Why is this change needed? What problem does it solve? -->

## Type of Change
<!-- Mark with [x] -->
- [ ] 🐛 Bug fix (non-breaking)
- [ ] ✨ New feature (non-breaking)
- [ ] 💥 Breaking change
- [ ] 📝 Documentation update
- [ ] ♻️ Code refactor
- [ ] ⚡ Performance improvement
- [ ] ✅ Test update

## Changes Made
<!-- List specific changes -->
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
# Paste test output
```

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Complex code commented
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tests prove fix/feature works
- [ ] Dependent changes merged

## Performance Impact
<!-- If applicable -->
- Latency:
- Memory:
- Throughput:

## Screenshots
<!-- If applicable -->

## Related Issues
<!-- Closes #123, Relates to #456 -->

## Reviewer Focus
<!-- Guide reviewers -->
-
-

## Additional Context
<!-- Any other information -->
EOF

git add .github/PULL_REQUEST_TEMPLATE.md
git commit -m "chore: add comprehensive PR template

Standardize pull request format:
- Clear structure for descriptions
- Checklist for author validation
- Performance impact section
- Testing requirements
- Reviewer guidance"
```

### Step 5.2: Create Code Review Guidelines

```bash
mkdir -p docs
cat > docs/CODE_REVIEW_GUIDE.md << 'EOF'
# Code Review Guidelines

## For PR Authors

### Before Creating PR
1. ✅ Self-review your code
2. ✅ Run all tests locally
3. ✅ Update documentation
4. ✅ Write clear PR description
5. ✅ Link related issues
6. ✅ Ensure CI passes

### PR Title Format
```
<type>(<scope>): <subject>

Examples:
feat(api): add batch prediction endpoint
fix(cache): handle null values correctly
docs(readme): update installation instructions
```

### PR Description
- **What** changed
- **Why** it changed
- **How** to test it
- **Any** breaking changes

## For Reviewers

### Review Checklist
- [ ] **Correctness**: Does the code work as intended?
- [ ] **Tests**: Adequate coverage and quality?
- [ ] **Documentation**: Clear and updated?
- [ ] **Performance**: Any concerns?
- [ ] **Security**: Vulnerabilities addressed?
- [ ] **Style**: Follows project conventions?

### Comment Severity Levels

**🚨 BLOCKER** - Must fix before merging
```
🚨 BLOCKER: This causes a memory leak in production.
```

**⚠️ IMPORTANT** - Should fix before merging
```
⚠️ IMPORTANT: Missing input validation here.
```

**💅 NITPICK** - Optional improvement
```
💅 NITPICK: Variable name could be clearer.
```

**👍 PRAISE** - Positive feedback
```
👍 NICE: Excellent error handling!
```

**❓ QUESTION** - Clarification needed
```
❓ QUESTION: Why use approach X over Y?
```

### Comment Tone

**✅ Good Examples:**
- "Consider adding error handling here for null inputs."
- "This function could benefit from extracting the validation logic."
- "Have you thought about using a set for O(1) lookup?"

**❌ Bad Examples:**
- "This is wrong."
- "Why didn't you add error handling?"
- "This is inefficient."

### Providing Feedback

**BE:**
- Specific (cite line numbers)
- Constructive (suggest solutions)
- Respectful (assume good intent)
- Clear (explain reasoning)

**DON'T:**
- Nitpick style (use linter instead)
- Rewrite code in comments
- Demand specific solutions
- Be condescending

### Approval Guidelines

**✅ Approve**: No blocking issues, minor suggestions only

**🔄 Request Changes**: Blocking issues present that must be fixed

**💬 Comment**: Questions or suggestions without blocking

## ML-Specific Review Points

### Model Changes
- [ ] Model versioning updated
- [ ] Backward compatibility maintained
- [ ] Migration path documented
- [ ] Performance benchmarked

### Data Processing
- [ ] Input validation comprehensive
- [ ] Data privacy maintained
- [ ] Edge cases handled
- [ ] Reproducibility ensured

### Inference Code
- [ ] Batch processing optimized
- [ ] Memory usage reasonable
- [ ] Latency acceptable
- [ ] Error handling robust

### Metrics
- [ ] Appropriate metrics tracked
- [ ] Logging comprehensive
- [ ] Monitoring alerts updated
- [ ] Dashboards updated

## Examples

### Good Review Comment
```
⚠️ IMPORTANT: Missing validation in batch_predict()

The batch size check should happen before reading files to prevent
memory issues with large batches.

Suggestion:
```python
if len(files) > MAX_BATCH_SIZE:
    raise HTTPException(400, detail="Batch too large")
```

This prevents loading all files into memory before validation.
```

### Good Author Response
```
Thanks for catching this! You're absolutely right - moved the
validation before file reading in commit abc123.

Also added a test case for this scenario.
```

## Response Time Expectations

- **First response**: Within 1 business day
- **Follow-up**: Within 4 hours during work hours
- **Approval/Changes**: Within 1 business day
- **Urgent PRs**: Flag with [URGENT] in title

## Conflict Resolution

If you disagree with feedback:
1. Ask clarifying questions
2. Explain your reasoning
3. Suggest alternatives
4. Escalate to tech lead if needed

Remember: Code review is about improving code, not winning arguments.
EOF

git add docs/CODE_REVIEW_GUIDE.md
git commit -m "docs: add comprehensive code review guidelines

Document review process:
- Author checklist
- Reviewer guidelines
- Comment severity levels
- ML-specific review points
- Conflict resolution

Ensures consistent, high-quality reviews."
```

---

## Part 6: Handling Concurrent Development

### Step 6.1: Rebase on Updated Main

```bash
# Scenario: Main branch updated while you were working

# Simulate update to main
git switch main
echo "# Update" >> README.md
git add README.md
git commit -m "docs: update README"

# Return to feature branch
git switch feature/add-batch-prediction

# Fetch latest from upstream
git fetch upstream

# Rebase your commits on updated main
git rebase upstream/main

# If conflicts occur:
# 1. Fix conflicts in files
# 2. git add <resolved-files>
# 3. git rebase --continue

# If rebase gets too complex:
# git rebase --abort
# Consider merge instead:
# git merge upstream/main

# After rebase, force push (history changed)
# git push --force-with-lease origin feature/add-batch-prediction
```

### Step 6.2: Split Large PRs

```bash
# Strategy: Break large feature into smaller PRs

# PR 1: Foundation (infrastructure)
git switch main
git switch -c feature/batch-foundation

cat > src/models/exceptions.py << 'EOF'
"""Custom exceptions for model operations."""

class ModelError(Exception):
    """Base exception for model errors."""
    pass

class ValidationError(ModelError):
    """Input validation error."""
    pass

class InferenceError(ModelError):
    """Inference execution error."""
    pass
EOF

git add src/models/exceptions.py
git commit -m "feat(models): add custom exception types"

# Submit PR 1
# gh pr create --title "Add model exception types"

# PR 2: Batch endpoint (depends on PR 1)
# Wait for PR 1 to be merged
# Then create PR 2 with batch endpoint

# This approach:
# - Smaller, easier to review
# - Faster review cycles
# - Earlier feedback
# - Lower risk
```

---

## Verification Checklist

After completing collaboration exercises:

- [ ] Forked and cloned repository successfully
- [ ] Set up upstream remote correctly
- [ ] Created feature branch with descriptive name
- [ ] Made commits with clear messages
- [ ] Created comprehensive PR description
- [ ] Addressed code review feedback
- [ ] Added review response comments
- [ ] Created PR template for project
- [ ] Created code review guidelines
- [ ] Understand rebase workflow
- [ ] Can handle concurrent development
- [ ] Know when to split PRs

---

## Common Issues and Solutions

### Issue 1: "PR has conflicts"

```bash
# Update your branch
git fetch upstream
git rebase upstream/main

# Or merge if rebase is complex
git merge upstream/main

# Resolve conflicts
git add <resolved-files>
git rebase --continue  # if rebasing
# OR
git commit  # if merging

# Push with force-with-lease (safer than --force)
git push --force-with-lease origin feature/branch
```

### Issue 2: "Accidentally committed to main"

```bash
# Create feature branch from current state
git branch feature/my-changes

# Reset main to match upstream
git switch main
git reset --hard upstream/main

# Push corrected main (carefully!)
# git push --force-with-lease origin main
```

### Issue 3: "Need to update PR with new commits"

```bash
# Make additional commits on feature branch
git add <files>
git commit -m "additional changes"

# Push to update PR automatically
# git push origin feature/branch

# PR updates automatically!
```

### Issue 4: "Reviewer requested changes, but I disagree"

```markdown
# Leave polite, reasoned response:

Thanks for the feedback! I understand your concern about X.

I chose this approach because:
1. Reason 1
2. Reason 2

However, I'm open to alternatives. Would Y approach address
your concerns? Or do you have another suggestion?

Happy to discuss on Slack/Zoom if easier.
```

---

## Best Practices

### DO:
- ✅ Keep PRs focused and small
- ✅ Write descriptive titles and descriptions
- ✅ Respond to reviews promptly
- ✅ Test before requesting review
- ✅ Update documentation
- ✅ Link related issues
- ✅ Be respectful in discussions

### DON'T:
- ❌ Create massive PRs
- ❌ Push broken code
- ❌ Ignore review feedback
- ❌ Force push to main
- ❌ Merge your own PRs without approval
- ❌ Take review comments personally
- ❌ Ghost reviewers

---

## Summary

You've mastered collaborative Git workflows:

- ✅ Fork and clone repositories
- ✅ Work with multiple remotes
- ✅ Create professional pull requests
- ✅ Conduct effective code reviews
- ✅ Address review feedback constructively
- ✅ Handle concurrent development
- ✅ Manage upstream synchronization
- ✅ Best practices for team collaboration

**Key Takeaways:**
- Clear communication is essential
- Small, focused PRs are easier to review
- Code review improves code quality
- Be respectful and constructive
- Documentation matters

**Time to Complete:** ~120 minutes

**Next Exercise:** Exercise 06 - ML-Specific Git Workflows
