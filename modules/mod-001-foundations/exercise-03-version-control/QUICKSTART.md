# Quick Start Guide - Git Version Control Exercise

Get up and running with the Git version control tools in 5 minutes.

## Prerequisites

- Python 3.8+
- Git 2.0+
- A terminal/command line

## Installation

### 1. Navigate to Exercise Directory

```bash
cd exercise-03-version-control
```

### 2. Verify Setup

```bash
python3 solutions/verify_setup.py
```

You should see all checks passing.

### 3. Install Dependencies (Optional)

```bash
pip install -r solutions/requirements.txt
```

## Quick Test Drive

### Test 1: Git Workflow Automation

```bash
# Initialize a test repository
mkdir ~/test-git-repo
cd ~/test-git-repo
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Create initial commit
echo "# Test Repo" > README.md
git add .
git commit -m "Initial commit"
git branch -M main

# Create a feature branch
python3 /path/to/solutions/git_workflow_automation.py branch feature "add authentication"

# Make some changes
echo "New feature" > feature.txt
git add .

# Smart commit
python3 /path/to/solutions/git_workflow_automation.py commit feat "add authentication" --scope auth

# Check status
python3 /path/to/solutions/git_workflow_automation.py status
```

### Test 2: Branch Manager

```bash
# Still in your test repo
cd ~/test-git-repo

# List branches by type
python3 /path/to/solutions/branch_manager.py list

# Get info about a branch
python3 /path/to/solutions/branch_manager.py info main

# Compare branches
python3 /path/to/solutions/branch_manager.py compare feature/add-authentication main
```

### Test 3: Commit Analyzer

```bash
# Analyze your commits
python3 /path/to/solutions/commit_analyzer.py messages

# Get contributor stats
python3 /path/to/solutions/commit_analyzer.py contributors

# Generate a changelog
python3 /path/to/solutions/commit_analyzer.py changelog 1.0.0 --output CHANGELOG.md
```

### Test 4: Best Practices Checker

```bash
# Check your repository
python3 /path/to/solutions/git_best_practices_checker.py

# Should give you a score and recommendations
```

### Test 5: Git Hooks

```bash
# Install pre-commit hook
cp /path/to/solutions/pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Install post-commit hook
cp /path/to/solutions/post-commit-hook.sh .git/hooks/post-commit
chmod +x .git/hooks/post-commit

# Test the hooks
echo "test" > test.txt
git add .
git commit -m "test: testing hooks"
# You should see the hooks running!
```

## Run Tests

```bash
cd tests/

# Install pytest first if not already installed
pip install pytest

# Run all tests
pytest -v

# Run specific test file
pytest test_git_workflow_automation.py -v

# Run with coverage
pip install pytest-cov
pytest --cov=../solutions --cov-report=html
```

## Common Use Cases

### Scenario 1: Start a New Feature

```bash
# 1. Create feature branch
./solutions/git_workflow_automation.py branch feature "user-dashboard"

# 2. Make changes
# ... edit files ...

# 3. Commit with proper format
./solutions/git_workflow_automation.py commit feat "add user dashboard" \
    --scope ui \
    --body "Implemented responsive dashboard with widgets"

# 4. Push to remote
git push -u origin feature/user-dashboard

# 5. Create PR via GitHub/GitLab UI
```

### Scenario 2: Fix a Bug

```bash
# 1. Create bugfix branch
./solutions/git_workflow_automation.py branch bugfix "login-error"

# 2. Fix the bug
# ... edit files ...

# 3. Commit
./solutions/git_workflow_automation.py commit fix "resolve login timeout" --scope auth

# 4. Push and create PR
git push -u origin bugfix/login-error
```

### Scenario 3: Prepare a Release

```bash
# 1. Create release branch
./solutions/branch_manager.py release 2.0.0 --base develop

# 2. Generate changelog
./solutions/commit_analyzer.py changelog 2.0.0 --since v1.9.0

# 3. Check best practices
./solutions/git_best_practices_checker.py

# 4. Merge to main (via PR)
# 5. Tag the release
git tag -a v2.0.0 -m "Release 2.0.0"
git push origin v2.0.0
```

### Scenario 4: Analyze Project Health

```bash
# Check commit quality
./solutions/commit_analyzer.py messages --limit 100

# Get contributor stats
./solutions/commit_analyzer.py contributors --since "3 months ago"

# Check frequency
./solutions/commit_analyzer.py frequency --days 90

# Export report
./solutions/commit_analyzer.py report health-report.json --format json

# Check best practices
./solutions/git_best_practices_checker.py --output compliance-report.txt
```

### Scenario 5: Clean Up Old Branches

```bash
# Find stale branches
./solutions/branch_manager.py stale --days 30

# Clean up merged branches (dry run first)
./solutions/git_workflow_automation.py cleanup

# Actually delete them
./solutions/git_workflow_automation.py cleanup --execute
```

## Tips for Daily Use

### 1. Add to PATH

Add the solutions directory to your PATH for easy access:

```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH="$PATH:/path/to/exercise-03-version-control/solutions"

# Now you can run from anywhere:
git_workflow_automation.py status
branch_manager.py list
```

### 2. Create Aliases

Create shell aliases for common commands:

```bash
# Add to ~/.bashrc or ~/.zshrc
alias gwf="python3 /path/to/git_workflow_automation.py"
alias gbm="python3 /path/to/branch_manager.py"
alias gca="python3 /path/to/commit_analyzer.py"

# Usage:
gwf branch feature "new-feature"
gbm list --type feature
gca messages
```

### 3. Use with Git Aliases

Integrate with Git aliases:

```bash
git config --global alias.smart-commit '!f() { python3 /path/to/git_workflow_automation.py commit "$@"; }; f'
git config --global alias.check-practices '!python3 /path/to/git_best_practices_checker.py'

# Usage:
git smart-commit feat "add feature" --scope api
git check-practices
```

## Learning Path

1. **Week 1: Basics**
   - Use git_workflow_automation.py for all Git operations
   - Practice conventional commits
   - Install and use Git hooks

2. **Week 2: Branch Management**
   - Use branch_manager.py for branch operations
   - Practice different merge strategies
   - Learn branch naming conventions

3. **Week 3: Analysis**
   - Use commit_analyzer.py to review your work
   - Generate changelogs for your projects
   - Improve commit message quality

4. **Week 4: Best Practices**
   - Run git_best_practices_checker.py on all projects
   - Fix identified issues
   - Share reports with team

## Troubleshooting

### "Command not found"

Make sure you're using the full path or have added to PATH:

```bash
python3 /full/path/to/solutions/script_name.py
```

### "Not a git repository"

Make sure you're in a Git repository:

```bash
cd /path/to/your/repo
git status  # Should work if in a repo
```

### "Permission denied"

Make scripts executable:

```bash
chmod +x solutions/*.py solutions/*.sh
```

### Tests Failing

Make sure pytest is installed:

```bash
pip install pytest
cd tests/
pytest -v
```

## Next Steps

1. **Complete the Exercise**: Follow the main README.md for full exercises
2. **Read Documentation**: Check solutions/README.md for detailed docs
3. **Explore Examples**: Look at examples/ directory for real-world scenarios
4. **Practice Daily**: Use the tools in your daily work
5. **Contribute**: Improve the tools and share your enhancements

## Resources

- Main README: [README.md](README.md)
- Solutions Documentation: [solutions/README.md](solutions/README.md)
- Test Documentation: [tests/README.md](tests/README.md)
- Branch Strategies: [examples/branch-strategies.md](examples/branch-strategies.md)
- Workflow Examples: [examples/workflow-examples.md](examples/workflow-examples.md)

## Getting Help

If you run into issues:

1. Check the verification script: `python3 solutions/verify_setup.py`
2. Read the error messages carefully
3. Check the documentation in solutions/README.md
4. Look at test files for usage examples
5. Review the example workflows

---

**Happy Git-ing!** 🚀
