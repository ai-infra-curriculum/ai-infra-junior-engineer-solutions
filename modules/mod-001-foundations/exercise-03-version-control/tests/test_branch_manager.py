#!/usr/bin/env python3
"""
Tests for branch_manager.py
"""

import pytest
import subprocess
import shutil
from pathlib import Path
import sys

# Add solutions directory to path
sys.path.insert(
    0, str(Path(__file__).parent.parent / "solutions")
)

from branch_manager import BranchManager


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
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        check=True,
    )

    # Ensure we're on main branch
    subprocess.run(["git", "branch", "-M", "main"], cwd=repo_path, check=True)

    yield repo_path

    shutil.rmtree(repo_path, ignore_errors=True)


def test_initialization(temp_git_repo):
    """Test BranchManager initialization."""
    manager = BranchManager(str(temp_git_repo))
    assert manager.repo_path == temp_git_repo


def test_get_branch_info(temp_git_repo):
    """Test getting branch information."""
    manager = BranchManager(str(temp_git_repo))

    info = manager.get_branch_info("main")

    assert info["name"] == "main"
    assert "last_commit" in info
    assert "commit_count" in info
    assert info["commit_count"] >= 1


def test_get_branch_info_invalid_branch(temp_git_repo):
    """Test getting info for non-existent branch."""
    manager = BranchManager(str(temp_git_repo))

    with pytest.raises(ValueError):
        manager.get_branch_info("nonexistent")


def test_list_branches_by_type(temp_git_repo):
    """Test listing branches by type."""
    manager = BranchManager(str(temp_git_repo))

    # Create different types of branches
    subprocess.run(
        ["git", "checkout", "-b", "feature/test1"],
        cwd=temp_git_repo,
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "-b", "bugfix/test2"],
        cwd=temp_git_repo,
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "main"],
        cwd=temp_git_repo,
        check=True,
    )

    branches = manager.list_branches_by_type()

    assert "feature/test1" in branches["feature"]
    assert "bugfix/test2" in branches["bugfix"]
    assert "main" in branches["other"]


def test_list_branches_by_type_filtered(temp_git_repo):
    """Test filtering branches by type."""
    manager = BranchManager(str(temp_git_repo))

    subprocess.run(
        ["git", "checkout", "-b", "feature/test"],
        cwd=temp_git_repo,
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "main"],
        cwd=temp_git_repo,
        check=True,
    )

    branches = manager.list_branches_by_type("feature")

    assert "feature" in branches
    assert "bugfix" not in branches
    assert "feature/test" in branches["feature"]


def test_create_release_branch(temp_git_repo):
    """Test creating a release branch."""
    manager = BranchManager(str(temp_git_repo))

    # Create develop branch first
    subprocess.run(
        ["git", "checkout", "-b", "develop"],
        cwd=temp_git_repo,
        check=True,
    )

    branch_name = manager.create_release_branch("1.0.0", "develop")

    assert branch_name == "release/1.0.0"

    # Verify we're on the release branch
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=temp_git_repo,
        capture_output=True,
        text=True,
    )
    assert "release/1.0.0" in result.stdout


def test_create_release_branch_invalid_version(temp_git_repo):
    """Test creating release branch with invalid version."""
    manager = BranchManager(str(temp_git_repo))

    with pytest.raises(ValueError):
        manager.create_release_branch("v1.0", "main")


def test_compare_branches(temp_git_repo):
    """Test comparing two branches."""
    manager = BranchManager(str(temp_git_repo))

    # Create a feature branch with changes
    subprocess.run(
        ["git", "checkout", "-b", "feature/compare"],
        cwd=temp_git_repo,
        check=True,
    )

    test_file = temp_git_repo / "new.txt"
    test_file.write_text("new content")
    subprocess.run(["git", "add", "."], cwd=temp_git_repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Add new file"],
        cwd=temp_git_repo,
        check=True,
    )

    comparison = manager.compare_branches("feature/compare", "main")

    assert comparison["branch1"] == "feature/compare"
    assert comparison["branch2"] == "main"
    assert comparison["commits_ahead"] == 1
    assert comparison["commits_behind"] == 0
    assert comparison["files_changed"] >= 1


def test_merge_with_strategy_merge(temp_git_repo):
    """Test merge with merge strategy."""
    manager = BranchManager(str(temp_git_repo))

    # Create feature branch
    subprocess.run(
        ["git", "checkout", "-b", "feature/merge-test"],
        cwd=temp_git_repo,
        check=True,
    )

    test_file = temp_git_repo / "merge.txt"
    test_file.write_text("merge content")
    subprocess.run(["git", "add", "."], cwd=temp_git_repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Add merge file"],
        cwd=temp_git_repo,
        check=True,
    )

    # Merge
    manager.merge_with_strategy("feature/merge-test", "main", "merge")

    # Verify merge
    result = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=temp_git_repo,
        capture_output=True,
        text=True,
    )
    assert "Add merge file" in result.stdout


def test_merge_with_strategy_squash(temp_git_repo):
    """Test merge with squash strategy."""
    manager = BranchManager(str(temp_git_repo))

    # Create feature branch with multiple commits
    subprocess.run(
        ["git", "checkout", "-b", "feature/squash-test"],
        cwd=temp_git_repo,
        check=True,
    )

    for i in range(2):
        test_file = temp_git_repo / f"file{i}.txt"
        test_file.write_text(f"content {i}")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, check=True)
        subprocess.run(
            ["git", "commit", "-m", f"Commit {i}"],
            cwd=temp_git_repo,
            check=True,
        )

    # Merge with squash
    manager.merge_with_strategy("feature/squash-test", "main", "squash")

    # Verify squashed merge
    result = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=temp_git_repo,
        capture_output=True,
        text=True,
    )
    assert "squashed" in result.stdout.lower()


def test_merge_with_delete_source(temp_git_repo):
    """Test merging with source branch deletion."""
    manager = BranchManager(str(temp_git_repo))

    # Create feature branch
    subprocess.run(
        ["git", "checkout", "-b", "feature/delete-me"],
        cwd=temp_git_repo,
        check=True,
    )

    test_file = temp_git_repo / "delete.txt"
    test_file.write_text("delete content")
    subprocess.run(["git", "add", "."], cwd=temp_git_repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Add file"],
        cwd=temp_git_repo,
        check=True,
    )

    # Merge and delete
    manager.merge_with_strategy(
        "feature/delete-me", "main", "merge", delete_source=True
    )

    # Verify branch is deleted
    result = subprocess.run(
        ["git", "branch"],
        cwd=temp_git_repo,
        capture_output=True,
        text=True,
    )
    assert "feature/delete-me" not in result.stdout


def test_find_stale_branches(temp_git_repo):
    """Test finding stale branches."""
    manager = BranchManager(str(temp_git_repo))

    # This test is tricky as we can't easily fake old dates
    # Just verify the method runs without error
    stale = manager.find_stale_branches(days=1000)
    assert isinstance(stale, list)


def test_protect_branch(temp_git_repo):
    """Test branch protection setup."""
    manager = BranchManager(str(temp_git_repo))

    manager.protect_branch("main")

    # Verify pre-push hook was created
    hook_path = temp_git_repo / ".git" / "hooks" / "pre-push"
    assert hook_path.exists()
    assert hook_path.stat().st_mode & 0o111  # Executable
