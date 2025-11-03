#!/bin/bash
#
# fix_permissions.sh - Automatically fix common permission issues
#
# Usage: ./fix_permissions.sh PROJECT_PATH
#

set -e
set -u

PROJECT_ROOT="${1:-.}"
[[ ! -d "$PROJECT_ROOT" ]] && echo "Error: Directory not found" && exit 1

echo "Fixing permissions in: $PROJECT_ROOT"

# Fix directories (755)
find "$PROJECT_ROOT" -type d -exec chmod 755 {} \; 2>/dev/null
echo "✓ Fixed directory permissions (755)"

# Fix files (644)
find "$PROJECT_ROOT" -type f -exec chmod 644 {} \; 2>/dev/null
echo "✓ Fixed file permissions (644)"

# Make scripts executable (755)
find "$PROJECT_ROOT" -type f -name "*.sh" -exec chmod 755 {} \; 2>/dev/null
find "$PROJECT_ROOT" -type f -name "*.py" -path "*/scripts/*" -exec chmod 755 {} \; 2>/dev/null
echo "✓ Made scripts executable (755)"

# Secure secrets (600 files, 700 dirs)
find "$PROJECT_ROOT" -type f \( -name "*secret*" -o -name "*password*" -o -name "*key*" -o -name "credentials.*" -o -name "*.pem" \) -exec chmod 600 {} \; 2>/dev/null
find "$PROJECT_ROOT" -type d -name "secrets" -exec chmod 700 {} \; 2>/dev/null
echo "✓ Secured sensitive files (600) and directories (700)"

# Set collaborative directories (775)
for dir in notebooks shared experiments datasets/processed models/checkpoints; do
    [[ -d "$PROJECT_ROOT/$dir" ]] && chmod 775 "$PROJECT_ROOT/$dir"
done
echo "✓ Set collaborative directories (775)"

echo ""
echo "Permissions fixed. Run audit to verify:"
echo "  ./audit_permissions.sh $PROJECT_ROOT"
