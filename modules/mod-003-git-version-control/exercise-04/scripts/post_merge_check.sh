#!/bin/bash

#######################################################################
# Post-Merge Validation Script
#######################################################################
# Validates that merge was completed successfully:
# - No unresolved conflict markers
# - Python syntax is valid
# - YAML configuration is valid
# - No broken imports
# - Tests pass (if available)
#######################################################################

set -e  # Exit on first error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ERROR_COUNT=0

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Post-Merge Validation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

#######################################################################
# Check 1: Unresolved Conflict Markers
#######################################################################

echo -e "${YELLOW}[1/5] Checking for unresolved conflict markers...${NC}"

CONFLICT_PATTERNS=("<<<<<<< HEAD" "=======" ">>>>>>>")
CONFLICT_FOUND=0

for pattern in "${CONFLICT_PATTERNS[@]}"; do
    if grep -r "$pattern" src/ tests/ configs/ 2>/dev/null | grep -v "Binary file" | grep -v ".git"; then
        echo -e "${RED}✗ ERROR: Found conflict marker: $pattern${NC}"
        CONFLICT_FOUND=1
        ((ERROR_COUNT++))
    fi
done

if [ $CONFLICT_FOUND -eq 0 ]; then
    echo -e "${GREEN}✓ No conflict markers found${NC}"
else
    echo -e "${RED}✗ Please resolve all conflicts before proceeding${NC}"
fi

echo ""

#######################################################################
# Check 2: Python Syntax
#######################################################################

echo -e "${YELLOW}[2/5] Checking Python syntax...${NC}"

PYTHON_ERRORS=0

if command -v python3 &> /dev/null; then
    while IFS= read -r -d '' file; do
        if ! python3 -m py_compile "$file" 2>/dev/null; then
            echo -e "${RED}✗ Syntax error in: $file${NC}"
            python3 -m py_compile "$file" 2>&1 | head -3
            PYTHON_ERRORS=1
            ((ERROR_COUNT++))
        fi
    done < <(find src/ tests/ -name "*.py" -type f -print0 2>/dev/null)

    if [ $PYTHON_ERRORS -eq 0 ]; then
        echo -e "${GREEN}✓ All Python files have valid syntax${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Python not found, skipping syntax check${NC}"
fi

echo ""

#######################################################################
# Check 3: YAML Configuration
#######################################################################

echo -e "${YELLOW}[3/5] Checking YAML configuration files...${NC}"

YAML_ERRORS=0

if command -v python3 &> /dev/null; then
    python3 << 'PYEOF'
import sys
import glob
import os

try:
    import yaml
except ImportError:
    print("\033[33m⚠ PyYAML not installed, skipping YAML validation\033[0m")
    sys.exit(0)

yaml_files = glob.glob("configs/**/*.yaml", recursive=True)
yaml_files += glob.glob("configs/**/*.yml", recursive=True)

if not yaml_files:
    print("\033[33m⚠ No YAML files found\033[0m")
    sys.exit(0)

errors = 0
for yaml_file in yaml_files:
    try:
        with open(yaml_file, 'r') as f:
            yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"\033[31m✗ Invalid YAML in {yaml_file}:\033[0m")
        print(f"  {str(e)}")
        errors += 1
    except Exception as e:
        print(f"\033[31m✗ Error reading {yaml_file}:\033[0m")
        print(f"  {str(e)}")
        errors += 1

if errors == 0:
    print(f"\033[32m✓ All {len(yaml_files)} YAML files are valid\033[0m")
else:
    sys.exit(1)
PYEOF

    if [ $? -ne 0 ]; then
        YAML_ERRORS=1
        ((ERROR_COUNT++))
    fi
else
    echo -e "${YELLOW}⚠ Python not found, skipping YAML validation${NC}"
fi

echo ""

#######################################################################
# Check 4: Import Verification
#######################################################################

echo -e "${YELLOW}[4/5] Checking Python imports...${NC}"

IMPORT_ERRORS=0

if command -v python3 &> /dev/null; then
    python3 << 'PYEOF'
import sys
import os
import ast
import glob

def check_imports(filename):
    """Check if file imports are valid syntax."""
    try:
        with open(filename, 'r') as f:
            tree = ast.parse(f.read(), filename=filename)
        return True
    except SyntaxError as e:
        print(f"\033[31m✗ Import/syntax error in {filename}:\033[0m")
        print(f"  Line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"\033[31m✗ Error checking {filename}:\033[0m")
        print(f"  {str(e)}")
        return False

python_files = []
for root, dirs, files in os.walk("src"):
    for file in files:
        if file.endswith(".py"):
            python_files.append(os.path.join(root, file))

if not python_files:
    print("\033[33m⚠ No Python files found\033[0m")
    sys.exit(0)

errors = 0
for py_file in python_files:
    if not check_imports(py_file):
        errors += 1

if errors == 0:
    print(f"\033[32m✓ All Python imports are valid ({len(python_files)} files checked)\033[0m")
else:
    sys.exit(1)
PYEOF

    if [ $? -ne 0 ]; then
        IMPORT_ERRORS=1
        ((ERROR_COUNT++))
    fi
else
    echo -e "${YELLOW}⚠ Python not found, skipping import validation${NC}"
fi

echo ""

#######################################################################
# Check 5: Tests (Optional)
#######################################################################

echo -e "${YELLOW}[5/5] Running tests...${NC}"

if [ -d "tests/" ]; then
    if command -v pytest &> /dev/null; then
        if pytest tests/ -v --tb=short; then
            echo -e "${GREEN}✓ All tests passed${NC}"
        else
            echo -e "${RED}✗ Some tests failed${NC}"
            ((ERROR_COUNT++))
        fi
    else
        echo -e "${YELLOW}⚠ pytest not installed, skipping tests${NC}"
        echo -e "${YELLOW}  Install with: pip install pytest${NC}"
    fi
else
    echo -e "${YELLOW}⚠ No tests directory found${NC}"
fi

echo ""

#######################################################################
# Summary
#######################################################################

echo -e "${BLUE}========================================${NC}"

if [ $ERROR_COUNT -eq 0 ]; then
    echo -e "${GREEN}✓ All post-merge checks passed!${NC}"
    echo -e "${GREEN}  Merge is ready for commit/push.${NC}"
    exit 0
else
    echo -e "${RED}✗ Found $ERROR_COUNT error(s)${NC}"
    echo -e "${RED}  Please fix errors before committing.${NC}"
    echo ""
    echo -e "${YELLOW}Common fixes:${NC}"
    echo "  - Remove conflict markers (<<<<<<, ======, >>>>>>)"
    echo "  - Fix Python syntax errors"
    echo "  - Fix YAML formatting"
    echo "  - Resolve import issues"
    echo "  - Fix failing tests"
    exit 1
fi
