#!/usr/bin/env python3
"""
Tests for git_workflow_automation.py
"""

import pytest
import subprocess
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add solutions directory to path
sys.path.insert(
    0, str(Path(__file__).parent.parent / "solutions")
)

from git_workflow_automation import GitWorkflowAutomation


@pytest.fixture
def temp_git_repo(tmp_path):
    """Create a temporary Git repository for testing."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Initialize git repo
    subprocess.run(
        ["git", "init"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

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

    # Create main branch if on master
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if "master" in result.stdout:
        subprocess.run(
            ["git", "branch", "-M", "main"],
            cwd=repo_path,
            check=True,
        )

    yield repo_path

    # Cleanup
    shutil.rmtree(repo_path, ignore_errors=True)


def test_initialization(temp_git_repo):
    """Test GitWorkflowAutomation initialization."""
    git = GitWorkflowAutomation(str(temp_git_repo))
    assert git.repo_path == temp_git_repo


def test_initialization_invalid_repo(tmp_path):
    """Test initialization with invalid repository."""
    with pytest.raises(ValueError):
        GitWorkflowAutomation(str(tmp_path))


def test_get_current_branch(temp_git_repo):
    """Test getting current branch."""
    git = GitWorkflowAutomation(str(temp_git_repo))
    branch = git.get_current_branch()
    assert branch == "main"


def test_get_all_branches(temp_git_repo):
    """Test getting all branches."""
    git = GitWorkflowAutomation(str(temp_git_repo))

    # Create a test branch
    subprocess.run(
        ["git", "checkout", "-b", "test-branch"],
        cwd=temp_git_repo,
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "main"],
        cwd=temp_git_repo,
        check=True,
    )

    branches = git.get_all_branches()
    assert "main" in branches
    assert "test-branch" in branches


def test_create_feature_branch(temp_git_repo):
    """Test creating a feature branch."""
    git = GitWorkflowAutomation(str(temp_git_repo))

    branch_name = git.create_feature_branch("new feature")
    assert branch_name == "feature/new-feature"
    assert git.get_current_branch() == "feature/new-feature"


def test_create_feature_branch_sanitizes_name(temp_git_repo):
    """Test feature branch name sanitization."""
    git = GitWorkflowAutomation(str(temp_git_repo))

    branch_name = git.create_feature_branch("Add User Auth!")
    assert branch_name == "feature/add-user-auth-"


def test_create_bugfix_branch(temp_git_repo):
    """Test creating a bugfix branch."""
    git = GitWorkflowAutomation(str(temp_git_repo))

    branch_name = git.create_bugfix_branch("login error")
    assert branch_name == "bugfix/login-error"
    assert git.get_current_branch() == "bugfix/login-error"


def test_create_hotfix_branch(temp_git_repo):
    """Test creating a hotfix branch."""
    git = GitWorkflowAutomation(str(temp_git_repo))

    branch_name = git.create_hotfix_branch("critical bug")
    assert branch_name == "hotfix/critical-bug"
    assert git.get_current_branch() == "hotfix/critical-bug"


def test_smart_commit_valid_type(temp_git_repo):
    """Test creating a conventional commit."""
    git = GitWorkflowAutomation(str(temp_git_repo))

    # Make a change
    test_file = temp_git_repo / "test.txt"
    test_file.write_text("test content")

    git.smart_commit("feat", "api", "add new endpoint")

    # Verify commit
    result = subprocess.run(
        ["git", "log", "-1", "--pretty=%s"],
        cwd=temp_git_repo,
        capture_output=True,
        text=True,
    )
    assert "feat(api): add new endpoint" in result.stdout


def test_smart_commit_invalid_type(temp_git_repo):
    """Test smart commit with invalid type."""
    git = GitWorkflowAutomation(str(temp_git_repo))

    with pytest.raises(ValueError):
        git.smart_commit("invalid", None, "test message")


def test_smart_commit_with_body(temp_git_repo):
    """Test commit with body."""
    git = GitWorkflowAutomation(str(temp_git_repo))

    test_file = temp_git_repo / "test.txt"
    test_file.write_text("test")

    git.smart_commit(
        "fix",
        "auth",
        "resolve login issue",
        body="Fixed bug in token validation",
    )

    result = subprocess.run(
        ["git", "log", "-1", "--pretty=%B"],
        cwd=temp_git_repo,
        capture_output=True,
        text=True,
    )
    assert "fix(auth): resolve login issue" in result.stdout
    assert "Fixed bug in token validation" in result.stdout


def test_smart_commit_breaking_change(temp_git_repo):
    """Test commit with breaking change."""
    git = GitWorkflowAutomation(str(temp_git_repo))

    test_file = temp_git_repo / "test.txt"
    test_file.write_text("test")

    git.smart_commit(
        "feat",
        "api",
        "change endpoint structure",
        breaking=True,
    )

    result = subprocess.run(
        ["git", "log", "-1", "--pretty=%B"],
        cwd=temp_git_repo,
        capture_output=True,
        text=True,
    )
    assert "feat(api)!: change endpoint structure" in result.stdout
    assert "BREAKING CHANGE" in result.stdout


def test_get_status_summary(temp_git_repo):
    """Test getting repository status summary."""
    git = GitWorkflowAutomation(str(temp_git_repo))

    # Create some changes
    test_file = temp_git_repo / "new_file.txt"
    test_file.write_text("new content")

    status = git.get_status_summary()

    assert status["branch"] == "main"
    assert status["untracked"] == 1
    assert isinstance(status["clean"], bool)


def test_quick_save(temp_git_repo):
    """Test quick save functionality."""
    git = GitWorkflowAutomation(str(temp_git_repo))

    # Make a change
    test_file = temp_git_repo / "wip.txt"
    test_file.write_text("work in progress")

    git.quick_save("WIP: testing feature")

    # Verify commit
    result = subprocess.run(
        ["git", "log", "-1", "--pretty=%s"],
        cwd=temp_git_repo,
        capture_output=True,
        text=True,
    )
    assert "WIP: testing feature" in result.stdout


def test_merge_branch(temp_git_repo):
    """Test merging branches."""
    git = GitWorkflowAutomation(str(temp_git_repo))

    # Create and switch to feature branch
    subprocess.run(
        ["git", "checkout", "-b", "feature/test"],
        cwd=temp_git_repo,
        check=True,
    )

    # Make a change
    test_file = temp_git_repo / "feature.txt"
    test_file.write_text("feature content")
    subprocess.run(["git", "add", "."], cwd=temp_git_repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Add feature"],
        cwd=temp_git_repo,
        check=True,
    )

    # Merge
    git.merge_branch("feature/test", "main", no_ff=True)

    # Verify we're on main
    assert git.get_current_branch() == "main"

    # Verify merge commit exists
    result = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=temp_git_repo,
        capture_output=True,
        text=True,
    )
    assert "Add feature" in result.stdout


def test_cleanup_merged_branches_dry_run(temp_git_repo):
    """Test cleanup in dry run mode."""
    git = GitWorkflowAutomation(str(temp_git_repo))

    # Create and merge a branch
    subprocess.run(
        ["git", "checkout", "-b", "feature/old"],
        cwd=temp_git_repo,
        check=True,
    )
    test_file = temp_git_repo / "old.txt"
    test_file.write_text("old feature")
    subprocess.run(["git", "add", "."], cwd=temp_git_repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Old feature"],
        cwd=temp_git_repo,
        check=True,
    )

    subprocess.run(
        ["git", "checkout", "main"],
        cwd=temp_git_repo,
        check=True,
    )
    subprocess.run(
        ["git", "merge", "feature/old"],
        cwd=temp_git_repo,
        check=True,
    )

    # Cleanup (dry run)
    deleted = git.cleanup_merged_branches(dry_run=True)

    # Branch should still exist
    branches = git.get_all_branches()
    assert "feature/old" in branches


def test_cleanup_merged_branches_execute(temp_git_repo):
    """Test actual cleanup of merged branches."""
    git = GitWorkflowAutomation(str(temp_git_repo))

    # Create and merge a branch
    subprocess.run(
        ["git", "checkout", "-b", "feature/to-delete"],
        cwd=temp_git_repo,
        check=True,
    )
    test_file = temp_git_repo / "delete.txt"
    test_file.write_text("to delete")
    subprocess.run(["git", "add", "."], cwd=temp_git_repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "To delete"],
        cwd=temp_git_repo,
        check=True,
    )

    subprocess.run(
        ["git", "checkout", "main"],
        cwd=temp_git_repo,
        check=True,
    )
    subprocess.run(
        ["git", "merge", "feature/to-delete"],
        cwd=temp_git_repo,
        check=True,
    )

    # Cleanup (execute)
    deleted = git.cleanup_merged_branches(dry_run=False)

    # Branch should be deleted
    branches = git.get_all_branches()
    assert "feature/to-delete" not in branches
    assert "feature/to-delete" in deleted
