# Tests for Git Version Control Tools

Comprehensive test suite for Git workflow automation tools.

## Running Tests

### Run All Tests

```bash
cd tests/
pytest
```

### Run Specific Test File

```bash
pytest test_git_workflow_automation.py
pytest test_branch_manager.py
pytest test_commit_analyzer.py
pytest test_best_practices_checker.py
```

### Run Specific Test

```bash
pytest test_git_workflow_automation.py::test_create_feature_branch
```

### Run with Coverage

```bash
pytest --cov=../solutions --cov-report=html --cov-report=term
```

View coverage report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Run Only Fast Tests

```bash
pytest -m "not slow"
```

### Verbose Output

```bash
pytest -v
pytest -vv  # Extra verbose
```

## Test Structure

```
tests/
├── README.md                           # This file
├── pytest.ini                          # Pytest configuration
├── conftest.py                         # Shared fixtures
├── test_git_workflow_automation.py     # Tests for git_workflow_automation.py
├── test_branch_manager.py              # Tests for branch_manager.py
├── test_commit_analyzer.py             # Tests for commit_analyzer.py
└── test_best_practices_checker.py      # Tests for git_best_practices_checker.py
```

## Test Coverage

### git_workflow_automation.py

**Covered Features:**
- ✓ Initialization and validation
- ✓ Get current branch
- ✓ List branches
- ✓ Create feature branches
- ✓ Create bugfix branches
- ✓ Create hotfix branches
- ✓ Branch name sanitization
- ✓ Smart commits (conventional commits)
- ✓ Commit with scope and body
- ✓ Breaking change commits
- ✓ Status summary
- ✓ Quick save functionality
- ✓ Branch merging
- ✓ Cleanup merged branches

**Test Count:** 15+ tests

### branch_manager.py

**Covered Features:**
- ✓ Initialization and validation
- ✓ Get branch information
- ✓ List branches by type
- ✓ Filter branches by type
- ✓ Create release branches
- ✓ Version validation
- ✓ Compare branches
- ✓ Merge strategies (merge, squash, rebase)
- ✓ Delete source after merge
- ✓ Find stale branches
- ✓ Branch protection

**Test Count:** 12+ tests

### commit_analyzer.py

**Covered Features:**
- ✓ Get commit history
- ✓ Filter commits (limit, author, date)
- ✓ Analyze commit messages
- ✓ Detect conventional commits
- ✓ Count commit types
- ✓ Detect scoped commits
- ✓ Detect breaking changes
- ✓ Contributor statistics
- ✓ Commit frequency analysis
- ✓ Find large commits
- ✓ Generate changelogs
- ✓ Export reports (JSON, Markdown)

**Test Count:** 13+ tests

### git_best_practices_checker.py

**Covered Features:**
- ✓ Check .gitignore existence
- ✓ Check .gitignore patterns
- ✓ Detect large files
- ✓ Detect sensitive files
- ✓ Analyze commit quality
- ✓ Check branch strategy
- ✓ Check README existence
- ✓ Check Git configuration
- ✓ Check installed hooks
- ✓ Check default branch
- ✓ Run all checks
- ✓ Generate reports

**Test Count:** 15+ tests

**Total Test Count:** 55+ tests

## Fixtures

### temp_git_repo

Creates a temporary Git repository for testing.

**Usage:**
```python
def test_something(temp_git_repo):
    git = GitWorkflowAutomation(str(temp_git_repo))
    # ... test code ...
```

**Features:**
- Initialized Git repository
- Configured user name and email
- Initial commit with README
- On 'main' branch
- Automatically cleaned up after test

### git_available

Checks if Git is available on the system.

**Usage:**
```python
def test_requires_git(git_available):
    # Test will be skipped if Git is not available
    pass
```

### clean_git_config

Provides isolated Git configuration for tests.

**Usage:**
```python
def test_config(clean_git_config):
    # Tests won't affect user's actual Git config
    pass
```

## Writing New Tests

### Test Template

```python
import pytest
from pathlib import Path
import sys

# Add solutions to path
sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))
from module_name import ClassName


def test_feature_description(temp_git_repo):
    """Test that feature does what it should."""
    # Arrange
    obj = ClassName(str(temp_git_repo))

    # Act
    result = obj.method()

    # Assert
    assert result == expected_value
```

### Best Practices

1. **One assertion per test** (when possible)
2. **Use descriptive test names**
3. **Follow AAA pattern** (Arrange, Act, Assert)
4. **Clean up resources** (use fixtures)
5. **Test edge cases**
6. **Test error conditions**
7. **Use parametrize for similar tests**

### Example: Parametrized Test

```python
@pytest.mark.parametrize("input,expected", [
    ("feature name", "feature/feature-name"),
    ("Bug Fix!", "bugfix/bug-fix-"),
    ("My-Feature", "feature/my-feature"),
])
def test_branch_name_sanitization(temp_git_repo, input, expected):
    git = GitWorkflowAutomation(str(temp_git_repo))
    result = git.create_feature_branch(input)
    assert result == expected
```

## Troubleshooting

### Tests Failing Due to Git Configuration

**Problem:** Tests fail because of user's Git configuration

**Solution:** Use the `clean_git_config` fixture

```python
def test_something(temp_git_repo, clean_git_config):
    # Test with clean config
    pass
```

### Tests Timing Out

**Problem:** Git operations take too long

**Solution:** Mark as slow test

```python
@pytest.mark.slow
def test_large_operation(temp_git_repo):
    # Long-running test
    pass
```

Run without slow tests:
```bash
pytest -m "not slow"
```

### Permission Errors on Windows

**Problem:** Git hooks not executable on Windows

**Solution:** Skip hook tests on Windows

```python
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="File permissions work differently on Windows"
)
def test_hook_installation(temp_git_repo):
    pass
```

### Temporary Files Not Cleaned Up

**Problem:** /tmp directory filling up

**Solution:** Use pytest's built-in `tmp_path` fixture

```python
def test_something(tmp_path):
    # tmp_path is automatically cleaned up
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")
```

## Continuous Integration

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install pytest pytest-cov
          pip install -r solutions/requirements.txt
      - name: Run tests
        run: |
          cd tests/
          pytest --cov=../solutions --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Adding New Tests

### Checklist for New Features

When adding a new feature to the tools:

- [ ] Write tests first (TDD)
- [ ] Test happy path
- [ ] Test error cases
- [ ] Test edge cases
- [ ] Test with different inputs
- [ ] Add integration test if needed
- [ ] Update this README
- [ ] Run full test suite
- [ ] Check code coverage

### Coverage Goals

- **Minimum:** 80% code coverage
- **Target:** 90% code coverage
- **Stretch:** 95% code coverage

Check current coverage:
```bash
pytest --cov=../solutions --cov-report=term-missing
```

## Performance Testing

### Benchmark Tests

For performance-critical features:

```python
import time

def test_performance(temp_git_repo):
    """Test that operation completes quickly."""
    git = GitWorkflowAutomation(str(temp_git_repo))

    start = time.time()
    git.get_status_summary()
    duration = time.time() - start

    assert duration < 1.0, f"Operation took {duration}s, expected <1s"
```

## Integration Tests

Mark integration tests that require more setup:

```python
@pytest.mark.integration
def test_full_workflow(temp_git_repo):
    """Test complete workflow end-to-end."""
    # Full workflow test
    pass
```

Run only unit tests:
```bash
pytest -m "not integration"
```

Run only integration tests:
```bash
pytest -m integration
```

## Test Data

### Creating Test Repositories

```python
def create_test_repo_with_history(path):
    """Helper to create repo with commit history."""
    # Initialize
    subprocess.run(["git", "init"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=path,
        check=True,
    )

    # Create commits
    for i in range(5):
        file = path / f"file{i}.txt"
        file.write_text(f"content {i}")
        subprocess.run(["git", "add", "."], cwd=path, check=True)
        subprocess.run(
            ["git", "commit", "-m", f"feat: add file {i}"],
            cwd=path,
            check=True,
        )
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
- [Testing Git Operations](https://git-scm.com/book/en/v2/Git-Tools-Testing)

## Contributing Tests

1. Fork the repository
2. Create a feature branch: `git checkout -b test/new-feature`
3. Write tests following the patterns above
4. Ensure all tests pass: `pytest`
5. Submit a pull request

---

**Happy Testing!** 🧪
