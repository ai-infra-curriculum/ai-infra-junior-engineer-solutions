#!/bin/bash
#
# grep_logs.sh - Log Filtering with grep
#
# Description:
#   Demonstrates various grep techniques for filtering and searching log files.
#   Shows practical examples for finding errors, patterns, and extracting information.
#

set -euo pipefail

LOG_DIR="${1:-../sample_logs}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

section() { echo -e "\n${BLUE}=== $* ===${NC}\n"; }
subsection() { echo -e "${CYAN}$*${NC}"; }

section "Find all ERROR messages"
grep "ERROR" "$LOG_DIR/training.log"

section "Find ERROR or WARNING"
grep -E "ERROR|WARNING" "$LOG_DIR/training.log"

section "Count errors"
echo "Total errors in errors.log:"
grep -c "ERROR" "$LOG_DIR/errors.log"

echo ""
echo "Errors per file:"
for file in "$LOG_DIR"/*.log; do
    count=$(grep -c "ERROR" "$file" || echo 0)
    echo "  $(basename "$file"): $count"
done

section "Find errors with context (2 lines before and after)"
grep -C 2 "ERROR" "$LOG_DIR/training.log"

section "Case-insensitive search for 'cuda'"
grep -i "cuda" "$LOG_DIR/training.log"

section "Invert match (exclude INFO messages)"
subsection "Show only WARNING and ERROR lines:"
grep -v "INFO" "$LOG_DIR/training.log"

section "Find lines with accuracy > 0.5"
grep "accuracy: 0\.[5-9]" "$LOG_DIR/training.log"

section "Extract epoch numbers"
grep -o "Epoch [0-9]*" "$LOG_DIR/training.log"

section "Search multiple files recursively"
grep -r "ERROR" "$LOG_DIR/"

section "Show only filenames with matches"
grep -l "CUDA" "$LOG_DIR"/* || echo "No files with 'CUDA' found"

section "Highlight matches (with color)"
grep --color=always "ERROR" "$LOG_DIR/training.log"

section "Find lines with specific patterns"
subsection "Lines containing 'memory':"
grep -i "memory" "$LOG_DIR/errors.log"

subsection "Lines starting with timestamp and containing ERROR:"
grep "^[0-9].*ERROR" "$LOG_DIR/errors.log"

section "Advanced pattern matching"
subsection "Find GPU-related errors:"
grep -E "CUDA|GPU|memory" -i "$LOG_DIR"/*.log | grep -i error

subsection "Find authentication issues:"
grep -iE "auth.*fail|unauthorized|permission" "$LOG_DIR"/*.log

subsection "Find timeout issues:"
grep -i "timeout" "$LOG_DIR"/*.log

section "Grep with line numbers"
grep -n "ERROR" "$LOG_DIR/training.log"

section "Quiet mode - exit status only"
if grep -q "CUDA" "$LOG_DIR/training.log"; then
    echo "CUDA found in training.log"
else
    echo "CUDA not found in training.log"
fi

echo -e "\n${GREEN}Grep analysis complete!${NC}"
