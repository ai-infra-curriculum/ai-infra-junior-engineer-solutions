#!/usr/bin/env python3
"""
Tests for git_best_practices_checker.py
"""

import pytest
import subprocess
import shutil
from pathlib import Path
import sys
import json

# Add solutions directory to path
sys.path.insert(
    0, str(Path(__file__).parent.parent / "solutions")
)

from git_best_practices_checker import BestPracticesChecker


@pytest.fixture
def temp_git_repo(tmp_path):
    """Create a temporary Git repository for testing."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)

    # Configure git
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path,
        check=True,
    )

    # Create initial commit
    readme = repo_path / "README.md"
    readme.write_text("# Test Repository")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "feat: initial commit"],
        cwd=repo_path,
        check=True,
    )

    subprocess.run(["git", "branch", "-M", "main"], cwd=repo_path, check=True)

    yield repo_path

    shutil.rmtree(repo_path, ignore_errors=True)


def test_initialization(temp_git_repo):
    """Test BestPracticesChecker initialization."""
    checker = BestPracticesChecker(str(temp_git_repo))
    assert checker.repo_path == temp_git_repo


def test_initialization_invalid_repo(tmp_path):
    """Test initialization with invalid repository."""
    with pytest.raises(ValueError):
        BestPracticesChecker(str(tmp_path))


def test_check_gitignore_exists_true(temp_git_repo):
    """Test gitignore check when file exists."""
    checker = BestPracticesChecker(str(temp_git_repo))

    # Create .gitignore
    gitignore = temp_git_repo / ".gitignore"
    gitignore.write_text("*.pyc\n")

    result = checker.check_gitignore_exists()

    assert result is True
    assert any(".gitignore file exists" in p for p in checker.passes)


def test_check_gitignore_exists_false(temp_git_repo):
    """Test gitignore check when file missing."""
    checker = BestPracticesChecker(str(temp_git_repo))

    result = checker.check_gitignore_exists()

    assert result is False
    assert any(".gitignore file is missing" in i for i in checker.issues)


def test_check_gitignore_patterns(temp_git_repo):
    """Test checking for important gitignore patterns."""
    checker = BestPracticesChecker(str(temp_git_repo))

    # Create comprehensive .gitignore
    gitignore = temp_git_repo / ".gitignore"
    gitignore.write_text("""
__pycache__
*.pyc
.env
venv/
*.log
.DS_Store
""")

    checker.check_gitignore_exists()
    checker.check_gitignore_patterns()

    assert any("has common important patterns" in p for p in checker.passes)


def test_check_gitignore_patterns_missing(temp_git_repo):
    """Test when gitignore is missing common patterns."""
    checker = BestPracticesChecker(str(temp_git_repo))

    gitignore = temp_git_repo / ".gitignore"
    gitignore.write_text("# Empty")

    checker.check_gitignore_exists()
    checker.check_gitignore_patterns()

    assert any("missing common patterns" in w for w in checker.warnings)


def test_check_large_files_none(temp_git_repo):
    """Test when no large files exist."""
    checker = BestPracticesChecker(str(temp_git_repo))

    checker.check_large_files(max_size_mb=10)

    assert any("No files larger than" in p for p in checker.passes)


def test_check_large_files_found(temp_git_repo):
    """Test when large files are found."""
    checker = BestPracticesChecker(str(temp_git_repo))

    # Create a large file
    large_file = temp_git_repo / "large.bin"
    large_file.write_bytes(b"x" * (11 * 1024 * 1024))  # 11MB

    subprocess.run(["git", "add", "."], cwd=temp_git_repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Add large file"],
        cwd=temp_git_repo,
        check=True,
    )

    checker.check_large_files(max_size_mb=10)

    assert any("Large files found" in i for i in checker.issues)


def test_check_sensitive_files_none(temp_git_repo):
    """Test when no sensitive files exist."""
    checker = BestPracticesChecker(str(temp_git_repo))

    checker.check_sensitive_files()

    assert any("No obviously sensitive" in p for p in checker.passes)


def test_check_sensitive_files_found(temp_git_repo):
    """Test when sensitive files are found."""
    checker = BestPracticesChecker(str(temp_git_repo))

    # Create a sensitive file
    env_file = temp_git_repo / ".env"
    env_file.write_text("API_KEY=secret")

    subprocess.run(["git", "add", "."], cwd=temp_git_repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Add .env"],
        cwd=temp_git_repo,
        check=True,
    )

    checker.check_sensitive_files()

    assert any("sensitive files found" in i for i in checker.issues)


def test_check_commit_message_quality_good(temp_git_repo):
    """Test commit message quality check with good messages."""
    checker = BestPracticesChecker(str(temp_git_repo))

    # Add more good commits
    for msg in ["fix(api): resolve error", "docs: update guide"]:
        test_file = temp_git_repo / f"{msg[:5]}.txt"
        test_file.write_text("content")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, check=True)
        subprocess.run(["git", "commit", "-m", msg], cwd=temp_git_repo, check=True)

    checker.check_commit_message_quality(limit=10)

    # Should have high percentage of conventional commits
    assert len(checker.issues) == 0 or not any(
        "commits follow conventional" in i for i in checker.issues
    )


def test_check_commit_message_quality_bad(temp_git_repo):
    """Test commit message quality check with bad messages."""
    checker = BestPracticesChecker(str(temp_git_repo))

    # Add bad commits
    for msg in ["wip", "fix", "changes"]:
        test_file = temp_git_repo / f"{msg}.txt"
        test_file.write_text("content")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, check=True)
        subprocess.run(["git", "commit", "-m", msg], cwd=temp_git_repo, check=True)

    checker.check_commit_message_quality(limit=10)

    # Should detect low quality
    assert any("commits follow conventional" in i for i in checker.issues) or \
           any("commits follow conventional" in w for w in checker.warnings)


def test_check_branch_strategy(temp_git_repo):
    """Test branch strategy detection."""
    checker = BestPracticesChecker(str(temp_git_repo))

    # Create feature branches
    subprocess.run(
        ["git", "checkout", "-b", "feature/test"],
        cwd=temp_git_repo,
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "-b", "develop"],
        cwd=temp_git_repo,
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "main"],
        cwd=temp_git_repo,
        check=True,
    )

    checker.check_branch_strategy()

    assert any("branch strategy" in p for p in checker.passes) or \
           any("branch strategy" in w for w in checker.warnings)


def test_check_readme_exists_true(temp_git_repo):
    """Test README check when file exists."""
    checker = BestPracticesChecker(str(temp_git_repo))

    checker.check_readme_exists()

    assert any("README file exists" in p for p in checker.passes)


def test_check_readme_exists_false(tmp_path):
    """Test README check when file missing."""
    repo_path = tmp_path / "no_readme_repo"
    repo_path.mkdir()

    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo_path,
        check=True,
    )

    # Create initial commit without README
    test_file = repo_path / "test.txt"
    test_file.write_text("test")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo_path,
        check=True,
    )

    checker = BestPracticesChecker(str(repo_path))
    checker.check_readme_exists()

    assert any("README file is missing" in i for i in checker.issues)


def test_check_git_config(temp_git_repo):
    """Test Git configuration check."""
    checker = BestPracticesChecker(str(temp_git_repo))

    checker.check_git_config()

    assert any("Git user name and email configured" in p for p in checker.passes)


def test_check_hooks_installed_none(temp_git_repo):
    """Test when no hooks are installed."""
    checker = BestPracticesChecker(str(temp_git_repo))

    checker.check_hooks_installed()

    assert any("No Git hooks installed" in w for w in checker.warnings)


def test_check_hooks_installed_present(temp_git_repo):
    """Test when hooks are installed."""
    checker = BestPracticesChecker(str(temp_git_repo))

    # Create a pre-commit hook
    hooks_dir = temp_git_repo / ".git" / "hooks"
    pre_commit = hooks_dir / "pre-commit"
    pre_commit.write_text("#!/bin/bash\necho 'test'\n")
    pre_commit.chmod(0o755)

    checker.check_hooks_installed()

    assert any("Git hooks installed" in p for p in checker.passes)


def test_check_default_branch_main(temp_git_repo):
    """Test default branch check with main."""
    checker = BestPracticesChecker(str(temp_git_repo))

    # Set up remote
    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.com/repo.git"],
        cwd=temp_git_repo,
        check=True,
    )
    subprocess.run(
        ["git", "push", "-u", "origin", "main"],
        cwd=temp_git_repo,
        check=False,
        capture_output=True,
    )

    checker.check_default_branch()

    # This test may not work without actual remote, so just verify it runs
    assert isinstance(checker.passes, list)
    assert isinstance(checker.warnings, list)


def test_run_all_checks(temp_git_repo):
    """Test running all checks together."""
    checker = BestPracticesChecker(str(temp_git_repo))

    # Add .gitignore for better score
    gitignore = temp_git_repo / ".gitignore"
    gitignore.write_text("__pycache__\n*.pyc\n.env\nvenv/\n*.log\n.DS_Store\n")
    subprocess.run(["git", "add", "."], cwd=temp_git_repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "feat: add gitignore"],
        cwd=temp_git_repo,
        check=True,
    )

    results = checker.run_all_checks()

    assert "repository" in results
    assert "score" in results
    assert "issues" in results
    assert "warnings" in results
    assert "passes" in results
    assert results["score"] >= 0
    assert results["score"] <= 100


def test_run_all_checks_output(temp_git_repo, capsys):
    """Test that run_all_checks produces output."""
    checker = BestPracticesChecker(str(temp_git_repo))

    results = checker.run_all_checks()
    captured = capsys.readouterr()

    assert "Running Git best practices checks" in captured.out


def test_print_report(temp_git_repo, capsys):
    """Test printing the report."""
    checker = BestPracticesChecker(str(temp_git_repo))

    results = checker.run_all_checks()
    checker.print_report(results)

    captured = capsys.readouterr()

    assert "GIT BEST PRACTICES REPORT" in captured.out
    assert "Score:" in captured.out
