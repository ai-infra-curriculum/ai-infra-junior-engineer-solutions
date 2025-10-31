# Git Version Control - Solutions

Comprehensive Git workflow automation tools for AI infrastructure engineers.

## Overview

This directory contains production-ready Python scripts and Git hooks to automate and enforce best practices in Git workflows. These tools are specifically designed for junior AI infrastructure engineers working on ML/AI projects.

## Tools Included

### 1. Git Workflow Automation (`git_workflow_automation.py`)

Automates common Git workflows with standardized patterns.

**Features:**
- Create feature, bugfix, and hotfix branches with proper naming
- Smart commits following conventional commits standard
- Automated branch merging with configurable strategies
- Cleanup merged branches
- Sync with remote repositories
- Quick WIP saves

**Usage:**

```bash
# Create a feature branch
python git_workflow_automation.py branch feature "add user authentication" --base main

# Create conventional commit
python git_workflow_automation.py commit feat "add login endpoint" --scope auth

# Merge branches
python git_workflow_automation.py merge feature/add-auth main

# Clean up merged branches
python git_workflow_automation.py cleanup --execute

# Get repository status
python git_workflow_automation.py status

# Quick save work in progress
python git_workflow_automation.py quicksave --message "WIP: working on API"
```

**Examples:**

```bash
# Start a new feature
./git_workflow_automation.py branch feature "ml-model-serving"
# Creates: feature/ml-model-serving from main

# Commit with conventional format
./git_workflow_automation.py commit feat "implement model loading" \
    --scope serving \
    --body "Add caching and error handling"

# Sync with remote
./git_workflow_automation.py sync

# List all branches
./git_workflow_automation.py branch list
```

### 2. Branch Manager (`branch_manager.py`)

Advanced branch management and lifecycle operations.

**Features:**
- List branches by type (feature, bugfix, hotfix, release)
- Get detailed branch information
- Find stale branches
- Create release branches with semantic versioning
- Compare branches
- Merge with strategies (merge, squash, rebase)
- Branch protection
- Visualize branch tree

**Usage:**

```bash
# List all feature branches
python branch_manager.py list --type feature

# Get branch information
python branch_manager.py info feature/new-api

# Find stale branches (>30 days)
python branch_manager.py stale --days 30

# Create release branch
python branch_manager.py release 1.0.0 --base develop

# Compare two branches
python branch_manager.py compare feature/api-v2 main

# Merge with squash strategy
python branch_manager.py merge feature/api-v2 main --strategy squash --delete

# Protect a branch
python branch_manager.py protect main

# Visualize branch tree
python branch_manager.py tree
```

**Examples:**

```bash
# Find and clean up old branches
./branch_manager.py stale --days 60
# Lists branches not updated in 60 days

# Create a release
./branch_manager.py release 2.1.0
# Creates: release/2.1.0 from develop

# Compare before merging
./branch_manager.py compare feature/new-feature main
# Shows commits ahead/behind, files changed
```

### 3. Commit Analyzer (`commit_analyzer.py`)

Analyze commit history and generate insights.

**Features:**
- Analyze commit message quality and patterns
- Get contributor statistics
- Analyze commit frequency patterns
- Find large commits
- Generate changelogs automatically
- Export comprehensive reports

**Usage:**

```bash
# Analyze commit messages
python commit_analyzer.py messages --since "2024-01-01" --limit 100

# Get contributor stats
python commit_analyzer.py contributors --since "30 days ago"

# Analyze commit frequency
python commit_analyzer.py frequency --days 30

# Find large commits
python commit_analyzer.py large --threshold 500

# Generate changelog
python commit_analyzer.py changelog 1.0.0 --since "v0.9.0" --output CHANGELOG.md

# Export comprehensive report
python commit_analyzer.py report analysis.json --format json
python commit_analyzer.py report analysis.md --format markdown
```

**Examples:**

```bash
# Check commit quality
./commit_analyzer.py messages --limit 50
# Shows conventional commit compliance, quality score

# Generate release changelog
./commit_analyzer.py changelog 2.0.0 --since "v1.9.0"
# Creates CHANGELOG.md with categorized changes

# Get team statistics
./commit_analyzer.py contributors --since "1 month ago"
# Shows commits, lines changed per contributor
```

### 4. Pre-commit Hook (`pre-commit-hook.sh`)

Validates code quality before commits.

**Features:**
- Checks for large files (>10MB)
- Detects sensitive data patterns (passwords, API keys, tokens)
- Validates Python syntax
- Checks code formatting (black)
- Runs linting (flake8)
- Detects debug statements
- Validates JSON/YAML files
- Checks for merge conflict markers

**Installation:**

```bash
# Copy to .git/hooks
cp pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Or create symlink
ln -s ../../solutions/pre-commit-hook.sh .git/hooks/pre-commit
```

**Bypass (when necessary):**

```bash
git commit --no-verify -m "emergency fix"
```

### 5. Post-commit Hook (`post-commit-hook.sh`)

Provides feedback and reminders after commits.

**Features:**
- Displays commit summary
- Logs commits to local file
- Validates conventional commit format
- Warns about unpushed commits
- Shows files changed and statistics
- Detects issue references
- Suggests next steps based on branch type
- Auto-generates release tags
- Tracks commit milestones

**Installation:**

```bash
cp post-commit-hook.sh .git/hooks/post-commit
chmod +x .git/hooks/post-commit
```

### 6. Best Practices Checker (`git_best_practices_checker.py`)

Audits repositories for Git best practices.

**Features:**
- Checks .gitignore existence and patterns
- Detects large files
- Finds potentially sensitive files
- Analyzes commit message quality
- Validates branch strategy
- Checks for README
- Validates Git configuration
- Checks installed hooks
- Validates default branch name
- Analyzes merge commit patterns
- Generates compliance score

**Usage:**

```bash
# Run all checks
python git_best_practices_checker.py

# Check specific repository
python git_best_practices_checker.py --repo /path/to/repo

# Export results as JSON
python git_best_practices_checker.py --json --output report.json

# Save report to file
python git_best_practices_checker.py --output report.txt
```

**Example Output:**

```
======================================================================
GIT BEST PRACTICES REPORT
======================================================================

Repository: /home/user/my-project
Score: 85.7%

ISSUES (must fix):
  ✗ .gitignore file is missing

WARNINGS (should fix):
  ⚠ Only 60% of commits follow conventional format
  ⚠ No Git hooks installed (consider adding pre-commit hooks)

PASSED CHECKS:
  ✓ No files larger than 10MB
  ✓ No obviously sensitive files detected
  ✓ Repository follows a branch strategy (Git Flow/GitHub Flow)
  ✓ README file exists
  ✓ Git user name and email configured
  ✓ Uses 'main' as default branch (modern convention)
  ✓ Clean commit history (few merge commits)
  ✓ Remote repository (origin) configured

======================================================================
⚠ GOOD - Some improvements recommended
======================================================================
```

## Installation

### Prerequisites

- Python 3.8+
- Git 2.0+

### Setup

1. Clone or navigate to the exercise directory:

```bash
cd exercise-03-version-control/solutions
```

2. Install optional dependencies:

```bash
pip install -r requirements.txt
```

3. Make scripts executable:

```bash
chmod +x *.py *.sh
```

4. Add to PATH (optional but recommended):

```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH="$PATH:/path/to/exercise-03-version-control/solutions"
```

## Quick Start

### Setting Up a New Repository

```bash
# Initialize repository
git init my-ai-project
cd my-ai-project

# Install hooks
cp ../solutions/pre-commit-hook.sh .git/hooks/pre-commit
cp ../solutions/post-commit-hook.sh .git/hooks/post-commit
chmod +x .git/hooks/*

# Create initial structure
mkdir -p src tests docs
touch README.md .gitignore

# Run best practices check
python ../solutions/git_best_practices_checker.py

# Create first feature
python ../solutions/git_workflow_automation.py branch feature "initial-setup"
```

### Daily Workflow

```bash
# Start work on new feature
python git_workflow_automation.py branch feature "add-model-serving"

# Make changes, then commit
python git_workflow_automation.py commit feat "add FastAPI endpoint" --scope api

# Check status
python git_workflow_automation.py status

# Sync with remote
python git_workflow_automation.py sync

# Create PR (via web interface or gh CLI)
```

### Before Release

```bash
# Analyze commits
python commit_analyzer.py messages --since "v1.0.0"

# Generate changelog
python commit_analyzer.py changelog 2.0.0 --since "v1.0.0"

# Create release branch
python branch_manager.py release 2.0.0

# Run best practices check
python git_best_practices_checker.py
```

## Configuration

### Git Configuration

Set up useful Git aliases:

```bash
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.lg "log --oneline --graph --all"
```

### Environment Variables

Some scripts support environment variables:

```bash
# Default base branch
export GIT_DEFAULT_BASE_BRANCH=main

# Enable verbose output
export GIT_VERBOSE=1
```

## Best Practices

### Commit Messages

Always use conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Tests
- `chore`: Maintenance

**Examples:**

```bash
# Good
git commit -m "feat(api): add authentication endpoint"
git commit -m "fix(model): handle edge case in preprocessing"
git commit -m "docs: update API documentation"

# Bad
git commit -m "changes"
git commit -m "fix stuff"
git commit -m "WIP"
```

### Branch Naming

Use consistent branch naming:

```
feature/description-of-feature
bugfix/description-of-bug
hotfix/description-of-urgent-fix
release/version-number
```

### Workflow

1. **Always branch from main/develop**
2. **Keep branches short-lived** (<1 week)
3. **Pull frequently** to avoid conflicts
4. **Squash when appropriate** to keep history clean
5. **Delete merged branches** to reduce clutter

## ML/AI Project Specific

### Large Files

Never commit large model files directly:

```bash
# Use .gitignore
echo "*.h5" >> .gitignore
echo "*.pkl" >> .gitignore
echo "*.pth" >> .gitignore

# Use Git LFS for models you must version
git lfs track "*.h5"
```

### Data Files

Keep data separate from code:

```bash
# In .gitignore
data/raw/**
data/processed/**
*.csv
*.parquet
```

### Notebooks

Handle Jupyter notebooks carefully:

```bash
# Clear outputs before committing
jupyter nbconvert --clear-output --inplace *.ipynb

# Or use nbstripout
pip install nbstripout
nbstripout --install
```

## Testing

Run the test suite:

```bash
cd ../tests
pytest -v
pytest --cov=../solutions --cov-report=html
```

## Troubleshooting

### Hooks Not Running

```bash
# Ensure hooks are executable
chmod +x .git/hooks/*

# Check hook path
ls -la .git/hooks/
```

### Python Not Found in Hooks

```bash
# Use full path in hook scripts
which python3
# Update shebang to full path
```

### Large File Already Committed

```bash
# Remove from history
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch path/to/large/file' \
  --prune-empty --tag-name-filter cat -- --all

# Or use BFG Repo Cleaner
```

## Advanced Usage

### Custom Pre-commit Checks

Edit `pre-commit-hook.sh` to add project-specific checks:

```bash
# Example: Check for specific function calls
if git diff --cached | grep -E "eval\s*\("; then
    echo "Error: eval() function not allowed"
    exit 1
fi
```

### Integration with CI/CD

Use the tools in your CI pipeline:

```yaml
# .github/workflows/checks.yml
name: Git Quality Checks

on: [push, pull_request]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Check best practices
        run: python solutions/git_best_practices_checker.py
```

### Automated Reporting

Generate weekly reports:

```bash
# Add to crontab
0 9 * * MON cd /path/to/repo && \
  python solutions/commit_analyzer.py report weekly-report.json
```

## Additional Resources

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [Semantic Versioning](https://semver.org/)

## Contributing

Found a bug or want to add a feature? Contributions welcome!

1. Create a feature branch
2. Make changes with tests
3. Ensure all checks pass
4. Submit a pull request

## License

MIT License - See parent repository for details

## Support

For issues or questions:
- Check the troubleshooting section
- Review test files for usage examples
- Consult the main README.md in exercise root
