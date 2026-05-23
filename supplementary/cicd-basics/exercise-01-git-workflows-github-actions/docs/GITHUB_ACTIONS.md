# GitHub Actions Reference

## Quick Start

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest
```

## Workflow Syntax

### Triggers

```yaml
on:
  push:
    branches: [ main, develop ]
  pull_request:
  schedule:
    - cron: '0 0 * * *'
  workflow_dispatch:
```

### Jobs

```yaml
jobs:
  job-name:
    runs-on: ubuntu-latest
    needs: [previous-job]
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']
    steps:
      - name: Step name
        run: echo "Hello"
```

### Common Actions

- `actions/checkout@v4` - Clone repository
- `actions/setup-python@v4` - Install Python
- `actions/cache@v3` - Cache dependencies
- `actions/upload-artifact@v3` - Save artifacts
- `actions/download-artifact@v3` - Retrieve artifacts

### Secrets

```yaml
- name: Use secret
  env:
    API_KEY: ${{ secrets.API_KEY }}
  run: echo "Using API key"
```

### Conditional Execution

```yaml
- name: Deploy
  if: github.ref == 'refs/heads/main'
  run: ./deploy.sh
```

## Best Practices

1. Use specific action versions (`@v4` not `@latest`)
2. Cache dependencies to speed up workflows
3. Use matrix builds for testing multiple versions
4. Fail fast when appropriate
5. Limit workflow permissions
6. Use secrets for sensitive data
7. Add timeout limits
8. Test workflows in feature branches

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Marketplace](https://github.com/marketplace?type=actions)
- [Workflow Syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
