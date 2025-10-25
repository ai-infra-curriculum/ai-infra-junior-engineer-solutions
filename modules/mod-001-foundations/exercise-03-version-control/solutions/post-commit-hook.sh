#!/bin/bash
#
# Post-commit Hook
#
# This hook runs after a successful commit to perform notifications,
# logging, and other post-commit actions.
#
# Install with:
#   cp post-commit-hook.sh .git/hooks/post-commit
#   chmod +x .git/hooks/post-commit
#

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get commit information
COMMIT_HASH=$(git rev-parse HEAD)
COMMIT_SHORT=$(git rev-parse --short HEAD)
COMMIT_MSG=$(git log -1 --pretty=%B)
COMMIT_AUTHOR=$(git log -1 --pretty=%an)
COMMIT_DATE=$(git log -1 --pretty=%ad)
BRANCH=$(git rev-parse --abbrev-ref HEAD)

# ============================================================================
# 1. Display commit summary
# ============================================================================
echo -e "\n${GREEN}✓ Commit successful!${NC}\n"
echo -e "${BLUE}Branch:${NC} $BRANCH"
echo -e "${BLUE}Commit:${NC} $COMMIT_SHORT"
echo -e "${BLUE}Author:${NC} $COMMIT_AUTHOR"
echo -e "${BLUE}Message:${NC} $COMMIT_MSG"

# ============================================================================
# 2. Log commit to local file (optional)
# ============================================================================
LOG_FILE=".git/commit-log.txt"
echo "[$COMMIT_DATE] $COMMIT_SHORT - $COMMIT_MSG (by $COMMIT_AUTHOR on $BRANCH)" >> "$LOG_FILE"

# ============================================================================
# 3. Check if commit follows conventional commits
# ============================================================================
if echo "$COMMIT_MSG" | grep -qE "^(feat|fix|docs|style|refactor|test|chore|perf|ci|build)(\(.+\))?!?:"; then
    echo -e "\n${GREEN}✓ Commit follows conventional commits format${NC}"
else
    echo -e "\n${YELLOW}⚠ Commit doesn't follow conventional commits format${NC}"
    echo -e "${YELLOW}Consider using: type(scope): description${NC}"
fi

# ============================================================================
# 4. Check for unpushed commits
# ============================================================================
UNPUSHED=$(git log @{u}.. --oneline 2>/dev/null | wc -l || echo "0")
if [ "$UNPUSHED" -gt 0 ]; then
    echo -e "\n${YELLOW}⚠ You have $UNPUSHED unpushed commit(s)${NC}"
    echo -e "${YELLOW}Run 'git push' to sync with remote${NC}"
fi

# ============================================================================
# 5. Show files changed in this commit
# ============================================================================
echo -e "\n${BLUE}Files changed:${NC}"
git diff-tree --no-commit-id --name-status -r HEAD | while read status file; do
    case $status in
        A) echo -e "  ${GREEN}+ $file${NC}" ;;
        M) echo -e "  ${YELLOW}~ $file${NC}" ;;
        D) echo -e "  ${RED}- $file${NC}" ;;
        *) echo "  $status $file" ;;
    esac
done

# ============================================================================
# 6. Show commit statistics
# ============================================================================
echo -e "\n${BLUE}Statistics:${NC}"
git show --stat HEAD | tail -n 1

# ============================================================================
# 7. Check for related issues/tickets in commit message
# ============================================================================
if echo "$COMMIT_MSG" | grep -qE "#[0-9]+|closes #[0-9]+|fixes #[0-9]+"; then
    ISSUES=$(echo "$COMMIT_MSG" | grep -oE "#[0-9]+" | tr '\n' ' ')
    echo -e "\n${GREEN}✓ References issue(s):${NC} $ISSUES"
fi

# ============================================================================
# 8. Remind about running tests (for certain file types)
# ============================================================================
CHANGED_FILES=$(git diff-tree --no-commit-id --name-only -r HEAD)

if echo "$CHANGED_FILES" | grep -qE "\.py$"; then
    echo -e "\n${YELLOW}💡 Reminder: Run tests before pushing${NC}"
    echo -e "${YELLOW}   pytest tests/          # Run all tests${NC}"
    echo -e "${YELLOW}   python -m unittest     # Alternative${NC}"
fi

# ============================================================================
# 9. Suggest next steps based on branch type
# ============================================================================
echo -e "\n${BLUE}Suggested next steps:${NC}"

case $BRANCH in
    feature/*)
        echo "  1. Continue development on this feature"
        echo "  2. Run tests: pytest tests/"
        echo "  3. Push when ready: git push origin $BRANCH"
        echo "  4. Create pull request when feature is complete"
        ;;
    bugfix/*)
        echo "  1. Verify the bug is fixed"
        echo "  2. Add/update tests"
        echo "  3. Push: git push origin $BRANCH"
        echo "  4. Create pull request for review"
        ;;
    hotfix/*)
        echo "  1. Test the hotfix thoroughly"
        echo "  2. Push: git push origin $BRANCH"
        echo "  3. Create URGENT pull request"
        echo "  4. Notify team about the hotfix"
        ;;
    main|master)
        echo -e "  ${YELLOW}⚠ You committed directly to $BRANCH${NC}"
        echo "  Consider using feature branches for development"
        ;;
    *)
        echo "  1. Push changes: git push origin $BRANCH"
        echo "  2. Run tests if not already done"
        ;;
esac

# ============================================================================
# 10. Optional: Send notification (webhook, Slack, etc.)
# ============================================================================
# Uncomment and configure if you want notifications
# WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
# if [ -n "$WEBHOOK_URL" ]; then
#     curl -X POST "$WEBHOOK_URL" \
#         -H 'Content-Type: application/json' \
#         -d "{\"text\":\"New commit by $COMMIT_AUTHOR: $COMMIT_MSG ($COMMIT_SHORT)\"}" \
#         2>/dev/null
# fi

# ============================================================================
# 11. Auto-generate tags for releases
# ============================================================================
if echo "$COMMIT_MSG" | grep -qE "^release:"; then
    VERSION=$(echo "$COMMIT_MSG" | grep -oE "[0-9]+\.[0-9]+\.[0-9]+")
    if [ -n "$VERSION" ]; then
        echo -e "\n${BLUE}Creating tag for release: v$VERSION${NC}"
        git tag -a "v$VERSION" -m "Release version $VERSION"
        echo -e "${GREEN}✓ Tag created: v$VERSION${NC}"
        echo -e "${YELLOW}Push tag with: git push origin v$VERSION${NC}"
    fi
fi

# ============================================================================
# 12. Update local commit counter
# ============================================================================
COUNTER_FILE=".git/commit-counter"
if [ -f "$COUNTER_FILE" ]; then
    COUNT=$(($(cat "$COUNTER_FILE") + 1))
else
    COUNT=1
fi
echo $COUNT > "$COUNTER_FILE"

if [ $((COUNT % 10)) -eq 0 ]; then
    echo -e "\n${GREEN}🎉 Milestone: $COUNT commits in this repository!${NC}"
fi

echo ""
