#!/bin/bash

# create-release.sh
# Helper script to create a new release

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Create Release${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo -e "${YELLOW}⚠ Warning: You're not on the main branch (currently on: $CURRENT_BRANCH)${NC}"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Release creation cancelled"
        exit 0
    fi
fi

# Check if working directory is clean
if ! git diff-index --quiet HEAD --; then
    echo -e "${RED}✗ Working directory is not clean${NC}"
    echo "Please commit or stash your changes before creating a release"
    exit 1
fi

# Get current version
CURRENT_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
echo "Current version: $CURRENT_TAG"
echo ""

# Prompt for new version
echo "Enter new version number (e.g., 1.2.3):"
read -r NEW_VERSION

# Validate version format
if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*)?$ ]]; then
    echo -e "${RED}✗ Invalid version format${NC}"
    echo "Version must follow semantic versioning (e.g., 1.0.0, 2.1.3-beta.1)"
    exit 1
fi

NEW_TAG="v${NEW_VERSION}"

# Check if tag already exists
if git rev-parse "$NEW_TAG" >/dev/null 2>&1; then
    echo -e "${RED}✗ Tag $NEW_TAG already exists${NC}"
    exit 1
fi

echo ""
echo "New version will be: $NEW_TAG"
echo ""

# Determine if this is a pre-release
IS_PRERELEASE="false"
if [[ "$NEW_VERSION" =~ (alpha|beta|rc) ]]; then
    IS_PRERELEASE="true"
    echo -e "${YELLOW}This will be marked as a pre-release${NC}"
    echo ""
fi

# Ask for release notes
echo "Enter release notes (press Ctrl+D when done):"
RELEASE_NOTES=$(cat)

if [ -z "$RELEASE_NOTES" ]; then
    echo -e "${RED}✗ Release notes cannot be empty${NC}"
    exit 1
fi

# Show summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Release Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Version: $NEW_TAG"
echo "Branch: $CURRENT_BRANCH"
echo "Pre-release: $IS_PRERELEASE"
echo ""
echo "Release Notes:"
echo "$RELEASE_NOTES"
echo ""

# Confirm
read -p "Create this release? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Release creation cancelled"
    exit 0
fi

# Run quality checks
echo ""
echo -e "${BLUE}➤ Running quality checks...${NC}"
if [ -x "scripts/check-code-quality.sh" ]; then
    if ! ./scripts/check-code-quality.sh examples/; then
        echo -e "${RED}✗ Quality checks failed${NC}"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Release creation cancelled"
            exit 1
        fi
    fi
else
    echo -e "${YELLOW}⚠ Quality check script not found (skipped)${NC}"
fi

# Run tests
echo ""
echo -e "${BLUE}➤ Running tests...${NC}"
if command -v pytest &> /dev/null; then
    if ! pytest examples/ --quiet; then
        echo -e "${RED}✗ Tests failed${NC}"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Release creation cancelled"
            exit 1
        fi
    else
        echo -e "${GREEN}✓ Tests passed${NC}"
    fi
else
    echo -e "${YELLOW}⚠ pytest not installed (skipped)${NC}"
fi

# Create tag
echo ""
echo -e "${BLUE}➤ Creating tag...${NC}"
git tag -a "$NEW_TAG" -m "Release $NEW_TAG

$RELEASE_NOTES"

echo -e "${GREEN}✓ Tag created: $NEW_TAG${NC}"

# Push tag
echo ""
read -p "Push tag to remote? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}➤ Pushing tag to remote...${NC}"
    git push origin "$NEW_TAG"
    echo -e "${GREEN}✓ Tag pushed to remote${NC}"
    echo ""
    echo "GitHub Actions will now:"
    echo "  1. Run tests"
    echo "  2. Build artifacts"
    echo "  3. Create GitHub release"
    echo "  4. Build and push Docker images"
    echo ""
    echo "Monitor the release workflow at:"
    echo "  https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/actions"
else
    echo ""
    echo "Tag created locally but not pushed."
    echo "To push later: git push origin $NEW_TAG"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Release $NEW_TAG created successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Show next steps
echo "Next steps:"
echo "  1. Monitor the GitHub Actions workflow"
echo "  2. Verify the release was created"
echo "  3. Announce the release to users"
echo "  4. Update documentation if needed"
echo ""

# Generate changelog (optional)
if command -v gh &> /dev/null; then
    echo "View release on GitHub:"
    gh release view "$NEW_TAG" --web 2>/dev/null || echo "  (Release will appear after workflow completes)"
fi
