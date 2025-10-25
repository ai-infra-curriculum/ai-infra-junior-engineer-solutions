#!/bin/bash

# setup-git-hooks.sh
# Install Git hooks for local validation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Git Hooks Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if we're in a Git repository
if [ ! -d "$REPO_ROOT/.git" ]; then
    echo -e "${RED}Error: Not in a Git repository${NC}"
    echo "Please run this script from within a Git repository"
    exit 1
fi

# Create hooks directory if it doesn't exist
mkdir -p "$HOOKS_DIR"

# Install pre-commit hook
echo -e "${BLUE}Installing pre-commit hook...${NC}"
cp "$SCRIPT_DIR/pre-commit-hook.sh" "$HOOKS_DIR/pre-commit"
chmod +x "$HOOKS_DIR/pre-commit"
echo -e "${GREEN}✓ Pre-commit hook installed${NC}"

# Install commit-msg hook
echo -e "${BLUE}Installing commit-msg hook...${NC}"
cat > "$HOOKS_DIR/commit-msg" << 'EOF'
#!/bin/bash

# commit-msg hook
# Validates commit message format

COMMIT_MSG_FILE=$1
COMMIT_MSG=$(cat "$COMMIT_MSG_FILE")

# Check for conventional commit format
PATTERN="^(feat|fix|docs|style|refactor|test|chore|ci|perf|build|revert)(\(.+\))?!?: .+"

if ! echo "$COMMIT_MSG" | grep -qE "$PATTERN"; then
    echo "❌ Invalid commit message format!"
    echo ""
    echo "Commit message must follow Conventional Commits format:"
    echo "  <type>(<scope>): <description>"
    echo ""
    echo "Valid types: feat, fix, docs, style, refactor, test, chore, ci, perf, build, revert"
    echo ""
    echo "Examples:"
    echo "  feat(training): Add distributed training support"
    echo "  fix(api): Handle edge case in preprocessing"
    echo "  docs: Update README with new examples"
    echo ""
    exit 1
fi

echo "✓ Commit message format is valid"
EOF

chmod +x "$HOOKS_DIR/commit-msg"
echo -e "${GREEN}✓ Commit-msg hook installed${NC}"

# Install pre-push hook
echo -e "${BLUE}Installing pre-push hook...${NC}"
cat > "$HOOKS_DIR/pre-push" << 'EOF'
#!/bin/bash

# pre-push hook
# Run tests before pushing

set -e

echo "Running pre-push checks..."

# Run quick tests
if command -v pytest &> /dev/null; then
    echo "Running tests..."
    if ! pytest examples/ --maxfail=1 --quiet 2>/dev/null; then
        echo "❌ Tests failed! Push aborted."
        exit 1
    fi
    echo "✓ Tests passed"
fi

echo "✓ Pre-push checks passed"
EOF

chmod +x "$HOOKS_DIR/pre-push"
echo -e "${GREEN}✓ Pre-push hook installed${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Git hooks installed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Installed hooks:"
echo "  • pre-commit  - Code quality checks before commit"
echo "  • commit-msg  - Validate commit message format"
echo "  • pre-push    - Run tests before push"
echo ""
echo "To bypass hooks (not recommended):"
echo "  git commit --no-verify"
echo "  git push --no-verify"
echo ""

# Install Python dependencies for hooks
echo -e "${YELLOW}Checking Python dependencies...${NC}"

REQUIRED_PACKAGES="black isort flake8 pytest"
MISSING_PACKAGES=""

for package in $REQUIRED_PACKAGES; do
    if ! python -m pip show "$package" &>/dev/null; then
        MISSING_PACKAGES="$MISSING_PACKAGES $package"
    fi
done

if [ -n "$MISSING_PACKAGES" ]; then
    echo -e "${YELLOW}Missing packages:$MISSING_PACKAGES${NC}"
    echo ""
    read -p "Install missing packages? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install $MISSING_PACKAGES
        echo -e "${GREEN}✓ Dependencies installed${NC}"
    else
        echo -e "${YELLOW}⚠ Hooks may not work without required packages${NC}"
    fi
else
    echo -e "${GREEN}✓ All required packages are installed${NC}"
fi

echo ""
echo -e "${BLUE}Setup complete!${NC}"
