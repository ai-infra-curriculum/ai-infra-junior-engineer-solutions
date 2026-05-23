# Step-by-Step Guide: Git and GitHub Fundamentals

## Overview
Master Git version control and GitHub collaboration including branching strategies, pull requests, code reviews, and team workflows for ML infrastructure projects.

## Phase 1: Git Basics (15 minutes)

### Initialize Repository and Basic Operations
```bash
# Create project directory
mkdir git-fundamentals
cd git-fundamentals

# Initialize git repository
git init

# Configure local settings
git config user.name "Your Name"
git config user.email "your.email@example.com"

# View configuration
git config --list --local
```

### Create and Track Files
```bash
# Create README
cat > README.md << 'EOF'
# Git Fundamentals

Learning Git and GitHub for ML infrastructure.
EOF

# Create Python file
cat > app.py << 'EOF'
def hello():
    print("Hello, Git!")

if __name__ == "__main__":
    hello()
EOF

# Check status
git status

# Add files to staging area
git add README.md
git add app.py
# OR add all files
git add .

# Check status again
git status

# Commit changes
git commit -m "Initial commit: Add README and app.py"

# View commit history
git log
git log --oneline
git log --graph --decorate --oneline
```

### Make Changes and Track History
```bash
# Modify app.py
cat >> app.py << 'EOF'

def goodbye():
    print("Goodbye, Git!")
EOF

# View changes before staging
git diff

# Stage changes
git add app.py

# View staged changes
git diff --staged

# Commit
git commit -m "Add goodbye function"

# View detailed history
git log --oneline --graph
git show HEAD  # Show last commit details
git show HEAD~1  # Show previous commit
```

**Validation**: Run `git log --oneline` and verify commits are tracked.

## Phase 2: Branching and Merging (15 minutes)

### Create and Switch Branches
```bash
# Create new branch
git branch feature/add-greeting

# List branches
git branch

# Switch to branch
git checkout feature/add-greeting
# OR (newer syntax)
git switch feature/add-greeting

# Create and switch in one command
git checkout -b feature/improve-messages
# OR
git switch -c feature/improve-messages
```

### Work on Feature Branch
```bash
# Switch to feature branch
git switch feature/add-greeting

# Make changes
cat > greetings.py << 'EOF'
def greet(name):
    return f"Hello, {name}!"

def farewell(name):
    return f"Goodbye, {name}!"
EOF

# Commit changes
git add greetings.py
git commit -m "Add greetings module"

# View branch history
git log --oneline --graph --all
```

### Merge Branches
```bash
# Switch to main branch
git switch main

# Merge feature branch
git merge feature/add-greeting

# View result
git log --oneline --graph

# Delete merged branch
git branch -d feature/add-greeting

# List remaining branches
git branch
```

### Handle Merge Conflicts
```bash
# Create two conflicting branches
git switch -c branch-a
echo "Line from branch A" >> app.py
git add app.py
git commit -m "Add line from branch A"

git switch main
git switch -c branch-b
echo "Line from branch B" >> app.py
git add app.py
git commit -m "Add line from branch B"

# Try to merge
git switch main
git merge branch-a  # Success
git merge branch-b  # Conflict!

# View conflict
cat app.py

# Resolve conflict manually
# Edit app.py to resolve conflict markers (<<<<, ====, >>>>)

# Mark as resolved
git add app.py
git commit -m "Merge branch-b with conflict resolution"

# Clean up
git branch -d branch-a branch-b
```

**Validation**: Successfully merge branches and resolve conflicts.

## Phase 3: GitHub Setup and Remote Repositories (15 minutes)

### Create GitHub Repository
```bash
# Go to github.com and create new repository
# Name: git-fundamentals
# Keep it public or private
# Don't initialize with README (we already have one)

# Add remote to local repository
git remote add origin https://github.com/yourusername/git-fundamentals.git

# Verify remote
git remote -v

# Push to GitHub
git push -u origin main

# View on GitHub
# Open: https://github.com/yourusername/git-fundamentals
```

### Clone Repository
```bash
# Clone your repository to different location
cd /tmp
git clone https://github.com/yourusername/git-fundamentals.git
cd git-fundamentals

# View remotes
git remote -v

# View branches (including remote)
git branch -a
```

### Fetch and Pull Changes
```bash
# Make changes on GitHub (edit README via web interface)

# Fetch changes (download but don't merge)
git fetch origin

# View what's new
git log origin/main

# Pull changes (fetch + merge)
git pull origin main
# OR simply
git pull

# Verify changes
cat README.md
```

**Validation**: Successfully push and pull from GitHub.

## Phase 4: Branching Strategies (15 minutes)

### Git Flow Workflow
```bash
# Main branches
git switch main
git switch -c develop

# Create feature branch from develop
git switch develop
git switch -c feature/ml-model

# Work on feature
cat > model.py << 'EOF'
import numpy as np

class SimpleModel:
    def __init__(self):
        self.trained = False

    def train(self, X, y):
        print("Training model...")
        self.trained = True

    def predict(self, X):
        if not self.trained:
            raise ValueError("Model not trained")
        return np.random.rand(len(X))
EOF

git add model.py
git commit -m "feat: Add simple ML model"

# Merge feature to develop
git switch develop
git merge feature/ml-model --no-ff  # Keep merge commit
git branch -d feature/ml-model

# Create release branch
git switch -c release/v1.0

# Make release preparations
echo "v1.0.0" > VERSION
git add VERSION
git commit -m "chore: Bump version to 1.0.0"

# Merge to main and tag
git switch main
git merge release/v1.0 --no-ff
git tag -a v1.0.0 -m "Release version 1.0.0"

# Merge back to develop
git switch develop
git merge release/v1.0 --no-ff
git branch -d release/v1.0

# Push everything
git push origin main develop --tags
```

### Trunk-Based Development (Simpler Alternative)
```bash
# Work on main with short-lived branches
git switch main
git pull

# Create short-lived feature branch
git switch -c add-validation

# Make quick changes
cat > validation.py << 'EOF'
def validate_input(data):
    if not data:
        raise ValueError("Empty data")
    return True
EOF

git add validation.py
git commit -m "Add input validation"

# Push and create PR immediately
git push -u origin add-validation

# After PR approval, merge and delete
git switch main
git pull
git branch -d add-validation
```

**Validation**: Practice both Git Flow and trunk-based workflows.

## Phase 5: Pull Requests and Code Review (15 minutes)

### Create Pull Request
```bash
# Create feature branch
git switch -c feature/add-tests
git push -u origin feature/add-tests

# Add tests
cat > test_model.py << 'EOF'
import pytest
from model import SimpleModel
import numpy as np

def test_model_initialization():
    model = SimpleModel()
    assert model.trained == False

def test_model_training():
    model = SimpleModel()
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    model.train(X, y)
    assert model.trained == True

def test_model_predict_untrained():
    model = SimpleModel()
    with pytest.raises(ValueError):
        model.predict(np.array([[1, 2]]))
EOF

git add test_model.py
git commit -m "test: Add model unit tests"
git push

# Go to GitHub and create Pull Request
# Title: "Add unit tests for SimpleModel"
# Description: Explain what tests were added
```

### Review Pull Request
```bash
# As reviewer, check out PR locally
git fetch origin
git switch -c review-tests origin/feature/add-tests

# Review code
cat test_model.py

# Run tests locally
pytest test_model.py -v

# If changes needed, add review comments on GitHub
# Request changes or approve

# As author, address feedback
# Make changes
echo "# Add more test coverage" >> test_model.py
git add test_model.py
git commit -m "test: Add additional test coverage"
git push

# Merge PR via GitHub interface
# Delete branch after merge
```

### PR Templates
Create `.github/pull_request_template.md`:
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tests pass locally

## Related Issues
Closes #(issue number)
```

**Validation**: Create, review, and merge a pull request.

## Phase 6: Advanced Git Techniques (10 minutes)

### Interactive Rebase
```bash
# Create messy commit history
git switch -c cleanup-commits
echo "1" >> file.txt && git add file.txt && git commit -m "add 1"
echo "2" >> file.txt && git add file.txt && git commit -m "add 2"
echo "3" >> file.txt && git add file.txt && git commit -m "add 3"
echo "fix" >> file.txt && git add file.txt && git commit -m "fix typo"

# View history
git log --oneline

# Interactive rebase to clean up
git rebase -i HEAD~4

# In editor, change 'pick' to:
# - 'squash' or 's' to combine commits
# - 'reword' or 'r' to change commit message
# - 'drop' or 'd' to remove commit

# Save and exit editor
```

### Cherry-Pick Commits
```bash
# Create branch with useful commit
git switch -c experimental
echo "useful feature" > feature.txt
git add feature.txt
git commit -m "Add useful feature"
COMMIT_HASH=$(git rev-parse HEAD)

# Switch to main and cherry-pick
git switch main
git cherry-pick $COMMIT_HASH

# Verify
git log --oneline
```

### Stash Changes
```bash
# Start work on something
echo "WIP" >> app.py
git status

# Need to switch branches but don't want to commit
git stash

# Verify working directory is clean
git status

# Switch branches, do other work
git switch develop

# Come back and restore work
git switch main
git stash pop

# List stashes
git stash list

# Apply specific stash
git stash apply stash@{0}
```

### Useful Git Aliases
```bash
# Add helpful aliases
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
git config --global alias.lg "log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"

# Use aliases
git st
git lg
```

### Commit Best Practices
Create `.gitmessage`:
```
# <type>(<scope>): <subject>
#
# <body>
#
# <footer>

# Type: feat, fix, docs, style, refactor, test, chore
# Scope: component or file name
# Subject: imperative, lowercase, no period
# Body: explain what and why vs. how
# Footer: breaking changes, issues closed
```

```bash
# Set as default commit template
git config --global commit.template ~/.gitmessage

# Example good commit
git commit -m "feat(model): Add batch prediction support

Implement batch processing for model predictions to improve
throughput when handling multiple requests.

- Add batch_predict() method
- Update API endpoint to accept arrays
- Add tests for batch processing

Closes #123"
```

**Validation**: Practice rebase, cherry-pick, and stash operations.

## Summary

You've mastered Git and GitHub fundamentals:
- **Git basics** including init, add, commit, and log operations
- **Branching and merging** with conflict resolution strategies
- **GitHub integration** for remote repositories and collaboration
- **Branching strategies** including Git Flow and trunk-based development
- **Pull requests** with code review workflows and templates
- **Advanced techniques** including rebase, cherry-pick, and stash

These skills enable effective collaboration on ML infrastructure projects with proper version control and team workflows.
