#!/bin/bash
#
# navigate_examples.sh - Interactive Linux navigation demonstrations
#
# Usage: ./navigate_examples.sh
#
# This script provides interactive demonstrations of:
# - cd, pwd, ls commands
# - Absolute vs relative paths
# - Symbolic links
# - find and locate commands
# - Common navigation patterns
#

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

VERSION="1.0.0"
SCRIPT_NAME=$(basename "$0")

# Colors for output (if terminal supports it)
if [[ -t 1 ]]; then
    GREEN='\033[0;32m'
    BLUE='\033[0;34m'
    YELLOW='\033[1;33m'
    NC='\033[0m'  # No Color
else
    GREEN=''
    BLUE=''
    YELLOW=''
    NC=''
fi

# =============================================================================
# Helper Functions
# =============================================================================

# Print section header
section() {
    echo ""
    echo -e "${BLUE}=========================================="
    echo "$1"
    echo -e "==========================================${NC}"
    echo ""
}

# Print command being demonstrated
show_command() {
    echo -e "${GREEN}\$ $1${NC}"
}

# Wait for user to continue
wait_continue() {
    echo ""
    read -p "Press Enter to continue..." -r
    echo ""
}

# Run command and show output
demo_command() {
    local cmd="$1"
    show_command "$cmd"
    eval "$cmd"
    echo ""
}

# =============================================================================
# Create Demo Environment
# =============================================================================

setup_demo() {
    section "Setting Up Demo Environment"

    # Create temporary demo directory
    DEMO_DIR=$(mktemp -d)
    echo "Created temporary demo directory: $DEMO_DIR"

    # Create demo project structure
    mkdir -p "$DEMO_DIR/ml-project/data/raw"
    mkdir -p "$DEMO_DIR/ml-project/data/processed"
    mkdir -p "$DEMO_DIR/ml-project/models/checkpoints"
    mkdir -p "$DEMO_DIR/ml-project/models/production"
    mkdir -p "$DEMO_DIR/ml-project/src/training"
    mkdir -p "$DEMO_DIR/ml-project/src/evaluation"
    mkdir -p "$DEMO_DIR/ml-project/notebooks"

    # Create some demo files
    echo "# ML Project" > "$DEMO_DIR/ml-project/README.md"
    echo "print('train model')" > "$DEMO_DIR/ml-project/src/training/train.py"
    echo "print('evaluate model')" > "$DEMO_DIR/ml-project/src/evaluation/evaluate.py"
    echo "config: {}" > "$DEMO_DIR/ml-project/config.yaml"
    touch "$DEMO_DIR/ml-project/data/raw/dataset.csv"
    touch "$DEMO_DIR/ml-project/models/checkpoints/model_epoch_10.ckpt"

    # Create second project for comparison
    mkdir -p "$DEMO_DIR/another-project/src"
    echo "# Another Project" > "$DEMO_DIR/another-project/README.md"

    echo "Demo environment created!"
    echo ""
    echo "Directory structure:"
    tree -L 3 "$DEMO_DIR" 2>/dev/null || find "$DEMO_DIR" -type d | head -15

    wait_continue
}

# =============================================================================
# Basic Navigation Demos
# =============================================================================

demo_basic_navigation() {
    section "Demo 1: Basic Navigation Commands (cd, pwd, ls)"

    echo "Starting directory:"
    demo_command "pwd"

    echo "Change to demo directory:"
    cd "$DEMO_DIR"
    demo_command "cd $DEMO_DIR"
    demo_command "pwd"

    echo "List contents:"
    demo_command "ls"

    echo "List with details:"
    demo_command "ls -l"

    echo "List with human-readable sizes:"
    demo_command "ls -lh"

    echo "List including hidden files:"
    demo_command "ls -la"

    wait_continue
}

# =============================================================================
# Path Navigation Demos
# =============================================================================

demo_absolute_vs_relative() {
    section "Demo 2: Absolute vs Relative Paths"

    cd "$DEMO_DIR"

    echo "Current directory:"
    demo_command "pwd"

    echo "Navigate using ABSOLUTE path:"
    demo_command "cd $DEMO_DIR/ml-project/src/training"
    demo_command "pwd"

    echo "Go back to demo directory:"
    cd "$DEMO_DIR"
    demo_command "cd $DEMO_DIR"
    demo_command "pwd"

    echo "Navigate using RELATIVE path:"
    demo_command "cd ml-project/src/training"
    demo_command "pwd"

    echo "Go up one level (parent directory):"
    demo_command "cd .."
    demo_command "pwd"

    echo "Go up two levels:"
    demo_command "cd ../.."
    demo_command "pwd"

    echo "Go to home directory:"
    demo_command "cd ~"
    demo_command "pwd"

    echo "Return to previous directory:"
    cd "$DEMO_DIR"
    demo_command "cd -"
    demo_command "pwd"

    wait_continue
}

# =============================================================================
# Directory Navigation Patterns
# =============================================================================

demo_navigation_patterns() {
    section "Demo 3: Common Navigation Patterns"

    cd "$DEMO_DIR/ml-project"

    echo "Starting in project root:"
    demo_command "pwd"

    echo "Pattern 1: Navigate down the tree"
    demo_command "cd data/raw"
    demo_command "pwd"

    echo "Pattern 2: Navigate to sibling directory"
    demo_command "cd ../processed"
    demo_command "pwd"

    echo "Pattern 3: Navigate to cousin directory"
    demo_command "cd ../../models/checkpoints"
    demo_command "pwd"

    echo "Pattern 4: Navigate back to root"
    demo_command "cd ../.."
    demo_command "pwd"

    echo "Pattern 5: Navigate using absolute path"
    demo_command "cd $DEMO_DIR/ml-project/src/evaluation"
    demo_command "pwd"

    wait_continue
}

# =============================================================================
# File Finding Demos
# =============================================================================

demo_find_command() {
    section "Demo 4: Finding Files with find"

    cd "$DEMO_DIR"

    echo "Find all Python files:"
    demo_command "find . -name '*.py'"

    echo "Find all Python files with type filter:"
    demo_command "find . -type f -name '*.py'"

    echo "Find all directories named 'src':"
    demo_command "find . -type d -name 'src'"

    echo "Find files modified in the last 10 minutes:"
    demo_command "find . -type f -mmin -10"

    echo "Find files with specific pattern in name:"
    demo_command "find . -name '*model*'"

    echo "Find and execute command on each file:"
    demo_command "find . -name '*.py' -exec wc -l {} \;"

    wait_continue
}

# =============================================================================
# Symbolic Link Demos
# =============================================================================

demo_symbolic_links() {
    section "Demo 5: Symbolic Links"

    cd "$DEMO_DIR/ml-project"

    echo "Create a symbolic link to large dataset:"
    demo_command "ln -s $DEMO_DIR/ml-project/data/raw data-link"

    echo "List to see the symbolic link:"
    demo_command "ls -l"

    echo "The symbolic link points to:"
    demo_command "readlink data-link"

    echo "Navigate through the symbolic link:"
    demo_command "cd data-link"
    demo_command "pwd"
    demo_command "ls"

    echo "Go back:"
    cd "$DEMO_DIR/ml-project"
    demo_command "cd .."

    echo "Remove symbolic link:"
    demo_command "rm data-link"

    wait_continue
}

# =============================================================================
# Advanced Listing Demos
# =============================================================================

demo_advanced_listing() {
    section "Demo 6: Advanced Listing Options"

    cd "$DEMO_DIR"

    echo "List recursively with tree (if available):"
    if command -v tree &>/dev/null; then
        demo_command "tree -L 2 ml-project"
    else
        echo "tree command not available, using find instead:"
        demo_command "find ml-project -type d | head -10"
    fi

    echo "List only directories:"
    demo_command "ls -d ml-project/*/"

    echo "List sorted by modification time:"
    demo_command "ls -lt ml-project | head -5"

    echo "List sorted by size:"
    demo_command "ls -lSh ml-project | head -5"

    echo "List with full path:"
    demo_command "find ml-project -maxdepth 2 -type f"

    wait_continue
}

# =============================================================================
# Disk Usage Demos
# =============================================================================

demo_disk_usage() {
    section "Demo 7: Disk Usage"

    cd "$DEMO_DIR"

    echo "Show disk usage of directory:"
    demo_command "du -sh ml-project"

    echo "Show disk usage of subdirectories:"
    demo_command "du -h ml-project | tail -5"

    echo "Show disk usage summary for each subdirectory:"
    demo_command "du -h --max-depth=1 ml-project"

    echo "Show disk usage sorted by size:"
    demo_command "du -h ml-project | sort -rh | head -5"

    wait_continue
}

# =============================================================================
# Practical ML Workflow Demo
# =============================================================================

demo_ml_workflow() {
    section "Demo 8: Practical ML Project Workflow"

    cd "$DEMO_DIR/ml-project"

    echo "Step 1: Start in project root"
    demo_command "pwd"
    demo_command "ls"

    echo "Step 2: Check raw data"
    demo_command "cd data/raw"
    demo_command "ls -lh"

    echo "Step 3: Navigate to training code"
    demo_command "cd ../../src/training"
    demo_command "pwd"
    demo_command "cat train.py"

    echo "Step 4: Check model checkpoints"
    demo_command "cd ../../models/checkpoints"
    demo_command "ls -lh"

    echo "Step 5: Return to project root"
    demo_command "cd ../.."
    demo_command "pwd"

    echo "Step 6: Find all Python files in project"
    demo_command "find . -name '*.py'"

    echo "Step 7: Find model files"
    demo_command "find . -name '*.ckpt'"

    wait_continue
}

# =============================================================================
# Quick Tips
# =============================================================================

demo_quick_tips() {
    section "Demo 9: Quick Tips and Shortcuts"

    cd "$DEMO_DIR"

    cat << 'EOF'
Quick Tips:

1. Tab Completion
   - Type 'cd ml-' and press TAB
   - Linux will auto-complete to 'cd ml-project'

2. Command History
   - Press UP arrow to cycle through previous commands
   - Use 'history' to see command history
   - Use Ctrl+R to search command history

3. Useful Aliases (add to ~/.bashrc)
   alias ll='ls -lah'
   alias la='ls -A'
   alias ..='cd ..'
   alias ...='cd ../..'

4. Directory Stack
   - Use 'pushd <dir>' to save current directory and navigate
   - Use 'popd' to return to saved directory
   - Use 'dirs' to view directory stack

5. Wildcards
   - * matches any characters: ls *.py
   - ? matches single character: ls file?.txt
   - [] matches character range: ls file[0-9].txt

6. Path Shortcuts
   .   = current directory
   ..  = parent directory
   ~   = home directory
   -   = previous directory
   /   = root directory

7. Efficient Navigation
   cd ~           # Go to home
   cd -           # Toggle between two directories
   cd             # Also goes to home
   pwd            # Show current directory

8. Combining Commands
   cd /path && ls      # Navigate and list
   cd dir || mkdir dir # Navigate or create if doesn't exist

EOF

    wait_continue
}

# =============================================================================
# Cleanup
# =============================================================================

cleanup_demo() {
    section "Cleaning Up Demo Environment"

    echo "Removing temporary demo directory: $DEMO_DIR"
    rm -rf "$DEMO_DIR"
    echo "Cleanup complete!"
    echo ""
}

# =============================================================================
# Main Menu
# =============================================================================

show_menu() {
    cat << EOF
========================================
Linux Navigation Demonstrations
========================================

Choose a demonstration:

  1. Basic Navigation (cd, pwd, ls)
  2. Absolute vs Relative Paths
  3. Navigation Patterns
  4. Finding Files (find command)
  5. Symbolic Links
  6. Advanced Listing
  7. Disk Usage
  8. ML Project Workflow
  9. Quick Tips and Shortcuts

  a. Run ALL demonstrations
  q. Quit

EOF
}

# =============================================================================
# Main Script
# =============================================================================

main() {
    # Setup demo environment
    setup_demo

    # Interactive menu loop
    while true; do
        show_menu
        read -p "Enter choice: " choice

        case "$choice" in
            1) demo_basic_navigation ;;
            2) demo_absolute_vs_relative ;;
            3) demo_navigation_patterns ;;
            4) demo_find_command ;;
            5) demo_symbolic_links ;;
            6) demo_advanced_listing ;;
            7) demo_disk_usage ;;
            8) demo_ml_workflow ;;
            9) demo_quick_tips ;;
            a|A)
                demo_basic_navigation
                demo_absolute_vs_relative
                demo_navigation_patterns
                demo_find_command
                demo_symbolic_links
                demo_advanced_listing
                demo_disk_usage
                demo_ml_workflow
                demo_quick_tips
                ;;
            q|Q)
                break
                ;;
            *)
                echo "Invalid choice. Please try again."
                wait_continue
                ;;
        esac
    done

    # Cleanup
    cleanup_demo

    echo "Thank you for using the navigation demonstrations!"
    echo ""
}

# Run main function
main
