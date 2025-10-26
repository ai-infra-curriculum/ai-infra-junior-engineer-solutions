# Step-by-Step Implementation Guide: Git Workflows & GitHub Actions

## Overview

Master CI/CD with GitHub Actions! Learn workflow automation, testing pipelines, deployment strategies, and production CI/CD patterns for ML infrastructure projects.

**Time**: 2-3 hours | **Difficulty**: Beginner to Intermediate

---

## Learning Objectives

✅ Understand GitHub Actions architecture and syntax
✅ Create automated CI/CD workflows
✅ Implement testing and linting pipelines
✅ Configure multi-job workflows
✅ Use workflow triggers (push, PR, schedule, manual)
✅ Manage secrets and environment variables
✅ Implement deployment automation
✅ Monitor workflow performance

---

## Phase 1: Basic GitHub Actions

### First Workflow

```.github/workflows/01-basic-ci.yml
name: Basic CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  hello-world:
    name: Hello World Job
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Say hello
        run: echo "Hello from GitHub Actions!"

      - name: Show environment
        run: |
          echo "Event: ${{ github.event_name }}"
          echo "Branch: ${{ github.ref }}"
          echo "Commit: ${{ github.sha }}"
```

### Workflow Components

- **name**: Workflow display name
- **on**: Triggers (push, pull_request, schedule, workflow_dispatch)
- **jobs**: One or more jobs to run
- **runs-on**: Runner OS (ubuntu-latest, windows-latest, macos-latest)
- **steps**: Sequential tasks within a job

---

## Phase 2: Python Linting & Formatting

### Lint Workflow

```.github/workflows/02-python-lint.yml
name: Python Lint

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}

      - name: Install dependencies
        run: |
          pip install flake8 black isort mypy pylint

      - name: Run Black
        run: black --check .

      - name: Run isort
        run: isort --check-only .

      - name: Run Flake8
        run: flake8 --max-line-length=100 .

      - name: Run MyPy
        run: mypy --ignore-missing-imports .

      - name: Run Pylint
        run: pylint --max-line-length=100 **/*.py
```

---

## Phase 3: Automated Testing

### Test Workflow

```.github/workflows/03-python-test.yml
name: Python Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-xdist

      - name: Run tests
        run: |
          pytest -v --cov=src --cov-report=xml --cov-report=term

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          flags: unittests
          name: codecov-${{ matrix.python-version }}
```

---

## Phase 4: Code Quality & Security

### Quality Checks

```.github/workflows/04-code-quality.yml
name: Code Quality

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # For SonarCloud

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install bandit safety

      - name: Security check with Bandit
        run: bandit -r src/

      - name: Dependency security check
        run: safety check

      - name: SonarCloud Scan
        uses: SonarSource/sonarcloud-github-action@master
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
```

---

## Phase 5: Pull Request Checks

### PR Workflow

```.github/workflows/05-pr-checks.yml
name: PR Checks

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  validate:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Validate PR title
        uses: amannn/action-semantic-pull-request@v5
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Check for breaking changes
        run: |
          if git diff origin/main...HEAD | grep -q "BREAKING CHANGE"; then
            echo "⚠️ Breaking change detected!"
          fi

      - name: Label PR
        uses: actions/labeler@v4
        with:
          repo-token: ${{ secrets.GITHUB_TOKEN }}

  size-check:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Check PR size
        run: |
          LINES=$(git diff --stat origin/main...HEAD | tail -1 | awk '{print $4}')
          if [ "$LINES" -gt 500 ]; then
            echo "⚠️ Large PR: $LINES lines changed"
            exit 1
          fi
```

---

## Phase 6: Release Automation

### Release Workflow

```.github/workflows/06-release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Build package
        run: |
          pip install build
          python -m build

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}

      - name: Create GitHub Release
        uses: ncipollo/release-action@v1
        with:
          artifacts: "dist/*"
          generateReleaseNotes: true
```

---

## Phase 7: Scheduled Workflows

### Nightly Builds

```.github/workflows/07-schedule.yml
name: Nightly Build

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily
  workflow_dispatch:  # Manual trigger

jobs:
  nightly:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Run full test suite
        run: pytest --full

      - name: Notify on failure
        if: failure()
        uses: slackapi/slack-github-action@v1
        with:
          webhook-url: ${{ secrets.SLACK_WEBHOOK }}
          payload: |
            {
              "text": "Nightly build failed!"
            }
```

---

## Phase 8: Secrets Management

### Using Secrets

```yaml
steps:
  - name: Deploy
    env:
      AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
      AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    run: |
      aws s3 sync dist/ s3://my-bucket
```

### Add Secrets

1. Repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, etc.

---

## Best Practices

✅ Use caching to speed up workflows
✅ Run jobs in parallel when possible
✅ Use matrix strategy for multi-version testing
✅ Store secrets securely
✅ Implement branch protection rules
✅ Use concurrency controls for deployments
✅ Monitor workflow execution time
✅ Keep workflows DRY with composite actions
✅ Use environments for deployment stages
✅ Implement proper error handling

---

**GitHub Actions mastered!** 🚀

**Next Exercise**: Automated Testing
