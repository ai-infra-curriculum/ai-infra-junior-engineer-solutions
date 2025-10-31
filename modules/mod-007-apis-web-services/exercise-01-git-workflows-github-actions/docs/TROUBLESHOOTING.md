# Troubleshooting Guide

## Git Issues

### Problem: Merge Conflicts

**Symptoms**: Git reports conflicts during merge/rebase

**Solutions**:
```bash
# View conflicted files
git status

# Option 1: Manual resolution
vim conflicted_file.py
git add conflicted_file.py
git commit

# Option 2: Use merge tool
git mergetool

# Option 3: Abort merge
git merge --abort
```

### Problem: Accidentally Committed to Wrong Branch

**Solution**:
```bash
# Move commit to correct branch
git log  # Note the commit hash
git reset --hard HEAD~1  # Undo commit on current branch
git checkout correct-branch
git cherry-pick <commit-hash>
```

### Problem: Need to Change Last Commit Message

**Solution**:
```bash
# If not pushed yet
git commit --amend -m "New message"

# If already pushed (creates new commit)
git revert HEAD
git commit -m "Correct message"
```

## GitHub Actions Issues

### Problem: Workflow Not Triggering

**Possible Causes**:
1. Workflow file syntax error
2. Wrong trigger configuration
3. Branch name doesn't match filter
4. Workflow file not in `.github/workflows/`

**Solutions**:
```bash
# Check workflow syntax
yamllint .github/workflows/ci.yml

# View workflow runs
gh run list

# Check workflow logs
gh run view <run-id>
```

### Problem: Workflow Fails with Permission Error

**Solutions**:
1. Check repository settings → Actions → Permissions
2. Add explicit permissions to workflow:
```yaml
permissions:
  contents: read
  pull-requests: write
```

### Problem: Secrets Not Available

**Solutions**:
1. Verify secret name matches (case-sensitive)
2. Secrets not available in PRs from forks
3. Use `${{ secrets.SECRET_NAME }}` syntax
4. Check repository/organization secrets in settings

### Problem: Workflow Times Out

**Solutions**:
```yaml
jobs:
  test:
    timeout-minutes: 30  # Set explicit timeout
    steps:
      - name: Long running task
        timeout-minutes: 15  # Per-step timeout
```

## Code Quality Issues

### Problem: Black Formatting Fails

**Solution**:
```bash
# Fix formatting
black examples/

# Check what would change
black --check --diff examples/
```

### Problem: Flake8 Linting Errors

**Common Errors**:
- `E501`: Line too long → Use shorter lines or ignore
- `F401`: Unused import → Remove or use `# noqa: F401`
- `E203`: Whitespace before ':' → Configure flake8 to ignore

**Solution**:
```bash
# See specific errors
flake8 examples/ --show-source

# Ignore specific error
flake8 examples/ --ignore=E501
```

### Problem: Tests Failing Locally But Pass in CI

**Possible Causes**:
1. Different Python versions
2. Missing dependencies
3. Environment differences
4. Test order dependencies

**Solutions**:
```bash
# Match CI Python version
pyenv install 3.11
pyenv local 3.11

# Run tests in random order
pytest --random-order

# Run with verbose output
pytest -vv
```

## Performance Issues

### Problem: Workflows Taking Too Long

**Solutions**:
1. **Cache dependencies**:
```yaml
- uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
```

2. **Use matrix builds to parallelize**:
```yaml
strategy:
  matrix:
    python-version: ['3.9', '3.10', '3.11']
```

3. **Skip unnecessary steps**:
```yaml
- name: Deploy
  if: github.ref == 'refs/heads/main'
```

### Problem: Large Repository

**Solutions**:
```bash
# Use shallow clone
git clone --depth 1 <repo-url>

# In GitHub Actions:
- uses: actions/checkout@v4
  with:
    fetch-depth: 1
```

## Common Error Messages

### `fatal: not a git repository`
**Solution**: Initialize git or run from repository root
```bash
git init
```

### `error: failed to push some refs`
**Solution**: Pull latest changes first
```bash
git pull --rebase origin main
git push
```

### `Permission denied (publickey)`
**Solution**: Check SSH key configuration
```bash
ssh -T git@github.com
ssh-keygen -t ed25519 -C "your_email@example.com"
```

### `refusing to merge unrelated histories`
**Solution**: Allow unrelated histories
```bash
git pull origin main --allow-unrelated-histories
```

## Getting Help

1. **Check workflow logs**: View detailed errors in GitHub Actions tab
2. **Search GitHub Discussions**: Many issues already documented
3. **GitHub Actions Community**: https://github.community/c/code-to-cloud/github-actions/41
4. **Stack Overflow**: Tag questions with `github-actions`
5. **Repository Issues**: Report bugs in the appropriate repo

## Debug Tips

### Enable Debug Logging in GitHub Actions

```bash
# Set repository secret
ACTIONS_STEP_DEBUG = true
ACTIONS_RUNNER_DEBUG = true
```

### Test Workflows Locally

```bash
# Install act
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run workflows locally
act push
act pull_request
```

### Git Debug

```bash
# Verbose output
git -v <command>

# Trace execution
GIT_TRACE=1 git <command>
```
