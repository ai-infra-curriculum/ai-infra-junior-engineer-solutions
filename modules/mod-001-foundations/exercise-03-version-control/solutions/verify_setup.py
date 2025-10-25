#!/usr/bin/env python3
"""
Verification script to check if Git version control exercise is properly set up.
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Tuple


def check_python_version() -> Tuple[bool, str]:
    """Check Python version."""
    if sys.version_info >= (3, 8):
        return True, f"Python {sys.version_info.major}.{sys.version_info.minor} OK"
    return False, f"Python {sys.version_info.major}.{sys.version_info.minor} - Need 3.8+"


def check_git_installed() -> Tuple[bool, str]:
    """Check if Git is installed."""
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        return True, f"Git installed: {result.stdout.strip()}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False, "Git is not installed"


def check_file_exists(file_path: Path, description: str) -> Tuple[bool, str]:
    """Check if a file exists."""
    if file_path.exists():
        return True, f"{description} exists"
    return False, f"{description} missing: {file_path}"


def check_files_structure(base_path: Path) -> List[Tuple[bool, str]]:
    """Check all required files exist."""
    checks = []

    # Solution files
    solutions_dir = base_path / "solutions"
    checks.append(check_file_exists(
        solutions_dir / "git_workflow_automation.py",
        "git_workflow_automation.py"
    ))
    checks.append(check_file_exists(
        solutions_dir / "branch_manager.py",
        "branch_manager.py"
    ))
    checks.append(check_file_exists(
        solutions_dir / "commit_analyzer.py",
        "commit_analyzer.py"
    ))
    checks.append(check_file_exists(
        solutions_dir / "git_best_practices_checker.py",
        "git_best_practices_checker.py"
    ))
    checks.append(check_file_exists(
        solutions_dir / "pre-commit-hook.sh",
        "pre-commit-hook.sh"
    ))
    checks.append(check_file_exists(
        solutions_dir / "post-commit-hook.sh",
        "post-commit-hook.sh"
    ))
    checks.append(check_file_exists(
        solutions_dir / "requirements.txt",
        "requirements.txt"
    ))
    checks.append(check_file_exists(
        solutions_dir / "README.md",
        "solutions/README.md"
    ))

    # Test files
    tests_dir = base_path / "tests"
    checks.append(check_file_exists(
        tests_dir / "test_git_workflow_automation.py",
        "test_git_workflow_automation.py"
    ))
    checks.append(check_file_exists(
        tests_dir / "test_branch_manager.py",
        "test_branch_manager.py"
    ))
    checks.append(check_file_exists(
        tests_dir / "test_commit_analyzer.py",
        "test_commit_analyzer.py"
    ))
    checks.append(check_file_exists(
        tests_dir / "test_best_practices_checker.py",
        "test_best_practices_checker.py"
    ))
    checks.append(check_file_exists(
        tests_dir / "conftest.py",
        "conftest.py"
    ))
    checks.append(check_file_exists(
        tests_dir / "pytest.ini",
        "pytest.ini"
    ))

    # Example files
    examples_dir = base_path / "examples"
    checks.append(check_file_exists(
        examples_dir / "sample-gitconfig",
        "sample-gitconfig"
    ))
    checks.append(check_file_exists(
        examples_dir / "gitignore-python-ml.txt",
        "gitignore-python-ml.txt"
    ))
    checks.append(check_file_exists(
        examples_dir / "branch-strategies.md",
        "branch-strategies.md"
    ))
    checks.append(check_file_exists(
        examples_dir / "workflow-examples.md",
        "workflow-examples.md"
    ))

    return checks


def check_scripts_executable(base_path: Path) -> List[Tuple[bool, str]]:
    """Check if scripts are executable."""
    checks = []
    solutions_dir = base_path / "solutions"

    for script in [
        "git_workflow_automation.py",
        "branch_manager.py",
        "commit_analyzer.py",
        "git_best_practices_checker.py",
        "pre-commit-hook.sh",
        "post-commit-hook.sh",
    ]:
        script_path = solutions_dir / script
        if script_path.exists():
            is_executable = script_path.stat().st_mode & 0o111
            if is_executable:
                checks.append((True, f"{script} is executable"))
            else:
                checks.append((False, f"{script} is not executable"))
        else:
            checks.append((False, f"{script} does not exist"))

    return checks


def test_import_modules(base_path: Path) -> List[Tuple[bool, str]]:
    """Test if Python modules can be imported."""
    checks = []
    solutions_dir = base_path / "solutions"

    # Add solutions to path
    sys.path.insert(0, str(solutions_dir))

    modules = [
        ("git_workflow_automation", "GitWorkflowAutomation"),
        ("branch_manager", "BranchManager"),
        ("commit_analyzer", "CommitAnalyzer"),
        ("git_best_practices_checker", "BestPracticesChecker"),
    ]

    for module_name, class_name in modules:
        try:
            module = __import__(module_name)
            if hasattr(module, class_name):
                checks.append((True, f"{module_name}.{class_name} imports successfully"))
            else:
                checks.append((False, f"{module_name} missing {class_name} class"))
        except Exception as e:
            checks.append((False, f"{module_name} import failed: {e}"))

    return checks


def main():
    """Run all verification checks."""
    print("=" * 70)
    print("Git Version Control Exercise - Setup Verification")
    print("=" * 70)
    print()

    # Get base path
    base_path = Path(__file__).parent.parent

    # Run all checks
    all_checks = []

    # System requirements
    print("System Requirements:")
    check, msg = check_python_version()
    print(f"  {'✓' if check else '✗'} {msg}")
    all_checks.append(check)

    check, msg = check_git_installed()
    print(f"  {'✓' if check else '✗'} {msg}")
    all_checks.append(check)

    print()

    # File structure
    print("File Structure:")
    file_checks = check_files_structure(base_path)
    for check, msg in file_checks:
        print(f"  {'✓' if check else '✗'} {msg}")
        all_checks.append(check)

    print()

    # Executable permissions
    print("Script Permissions:")
    exec_checks = check_scripts_executable(base_path)
    for check, msg in exec_checks:
        print(f"  {'✓' if check else '✗'} {msg}")
        all_checks.append(check)

    print()

    # Module imports
    print("Module Imports:")
    import_checks = test_import_modules(base_path)
    for check, msg in import_checks:
        print(f"  {'✓' if check else '✗'} {msg}")
        all_checks.append(check)

    print()
    print("=" * 70)

    # Summary
    passed = sum(all_checks)
    total = len(all_checks)
    percentage = (passed / total * 100) if total > 0 else 0

    print(f"Results: {passed}/{total} checks passed ({percentage:.1f}%)")

    if passed == total:
        print("✓ All checks passed! Setup is complete.")
        print()
        print("Next steps:")
        print("  1. Install dependencies: pip install -r solutions/requirements.txt")
        print("  2. Run tests: cd tests && pytest")
        print("  3. Try the tools: python solutions/git_workflow_automation.py --help")
        return 0
    else:
        print("✗ Some checks failed. Please review the errors above.")
        print()
        print("Common fixes:")
        print("  - Make scripts executable: chmod +x solutions/*.py solutions/*.sh")
        print("  - Install Git if missing: sudo apt install git (or brew install git)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
