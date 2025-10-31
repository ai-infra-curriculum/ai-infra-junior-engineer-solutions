#!/usr/bin/env python3
"""
Tests for commit_analyzer.py
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

from commit_analyzer import CommitAnalyzer


@pytest.fixture
def temp_git_repo(tmp_path):
    """Create a temporary Git repository with commit history."""
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

    # Create more commits with different types
    for i, msg in enumerate([
        "fix(auth): resolve login issue",
        "docs: update README",
        "refactor(api): simplify endpoint",
        "test: add unit tests",
        "bad commit message",
    ]):
        test_file = repo_path / f"file{i}.txt"
        test_file.write_text(f"content {i}")
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=repo_path,
            check=True,
        )

    subprocess.run(["git", "branch", "-M", "main"], cwd=repo_path, check=True)

    yield repo_path

    shutil.rmtree(repo_path, ignore_errors=True)


def test_initialization(temp_git_repo):
    """Test CommitAnalyzer initialization."""
    analyzer = CommitAnalyzer(str(temp_git_repo))
    assert analyzer.repo_path == temp_git_repo


def test_get_commits(temp_git_repo):
    """Test getting commit history."""
    analyzer = CommitAnalyzer(str(temp_git_repo))

    commits = analyzer.get_commits()

    assert len(commits) == 6  # Initial + 5 more
    assert all("hash" in c for c in commits)
    assert all("author" in c for c in commits)
    assert all("message" in c for c in commits)


def test_get_commits_with_limit(temp_git_repo):
    """Test getting commits with limit."""
    analyzer = CommitAnalyzer(str(temp_git_repo))

    commits = analyzer.get_commits(limit=3)

    assert len(commits) == 3


def test_get_commits_with_author(temp_git_repo):
    """Test filtering commits by author."""
    analyzer = CommitAnalyzer(str(temp_git_repo))

    commits = analyzer.get_commits(author="Test User")

    assert len(commits) > 0
    assert all(c["author"] == "Test User" for c in commits)


def test_analyze_commit_messages(temp_git_repo):
    """Test analyzing commit message quality."""
    analyzer = CommitAnalyzer(str(temp_git_repo))

    analysis = analyzer.analyze_commit_messages()

    assert analysis["total_commits"] == 6
    assert analysis["conventional_commits"] == 5  # All except "bad commit message"
    assert "feat" in analysis["commit_types"]
    assert "fix" in analysis["commit_types"]
    assert analysis["scoped_commits"] >= 2  # fix(auth) and refactor(api)
    assert analysis["quality_score"] > 0


def test_analyze_commit_messages_custom_commits(temp_git_repo):
    """Test analyzing specific commits."""
    analyzer = CommitAnalyzer(str(temp_git_repo))

    commits = analyzer.get_commits(limit=2)
    analysis = analyzer.analyze_commit_messages(commits)

    assert analysis["total_commits"] == 2


def test_get_contributor_stats(temp_git_repo):
    """Test getting contributor statistics."""
    analyzer = CommitAnalyzer(str(temp_git_repo))

    contributors = analyzer.get_contributor_stats()

    assert len(contributors) >= 1
    assert contributors[0]["author"] == "Test User"
    assert contributors[0]["commits"] == 6
    assert "lines_added" in contributors[0]
    assert "lines_deleted" in contributors[0]


def test_analyze_commit_frequency(temp_git_repo):
    """Test analyzing commit frequency."""
    analyzer = CommitAnalyzer(str(temp_git_repo))

    frequency = analyzer.analyze_commit_frequency(days=30)

    assert frequency["total_commits"] == 6
    assert frequency["commits_per_day"] >= 0
    assert "weekday_distribution" in frequency
    assert "hour_distribution" in frequency


def test_find_large_commits(temp_git_repo):
    """Test finding large commits."""
    analyzer = CommitAnalyzer(str(temp_git_repo))

    # Create a large commit
    large_file = temp_git_repo / "large.txt"
    large_file.write_text("x" * 1000 + "\n" * 600)  # 600+ lines
    subprocess.run(["git", "add", "."], cwd=temp_git_repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "feat: add large file"],
        cwd=temp_git_repo,
        check=True,
    )

    large_commits = analyzer.find_large_commits(threshold=500)

    assert len(large_commits) >= 1
    assert large_commits[0]["changes"] >= 500


def test_generate_changelog(temp_git_repo):
    """Test generating changelog."""
    analyzer = CommitAnalyzer(str(temp_git_repo))

    changelog = analyzer.generate_changelog("1.0.0")

    assert "Version 1.0.0" in changelog
    assert "Features" in changelog
    assert "Bug Fixes" in changelog
    assert "initial commit" in changelog


def test_generate_changelog_with_since(temp_git_repo):
    """Test generating changelog from specific point."""
    analyzer = CommitAnalyzer(str(temp_git_repo))

    # Get first commit hash
    commits = analyzer.get_commits()
    first_commit = commits[-1]["hash"]

    changelog = analyzer.generate_changelog("1.0.0", since=first_commit)

    assert "Version 1.0.0" in changelog


def test_export_report_json(temp_git_repo, tmp_path):
    """Test exporting report as JSON."""
    analyzer = CommitAnalyzer(str(temp_git_repo))

    output_file = tmp_path / "report.json"
    analyzer.export_report(str(output_file), format="json")

    assert output_file.exists()

    # Verify JSON is valid
    with open(output_file) as f:
        data = json.load(f)

    assert "commit_analysis" in data
    assert "contributors" in data
    assert "frequency" in data


def test_export_report_markdown(temp_git_repo, tmp_path):
    """Test exporting report as Markdown."""
    analyzer = CommitAnalyzer(str(temp_git_repo))

    output_file = tmp_path / "report.md"
    analyzer.export_report(str(output_file), format="markdown")

    assert output_file.exists()

    content = output_file.read_text()
    assert "# Git Repository Analysis Report" in content
    assert "Commit Message Quality" in content
    assert "Top Contributors" in content


def test_commit_type_detection(temp_git_repo):
    """Test detection of different commit types."""
    analyzer = CommitAnalyzer(str(temp_git_repo))

    analysis = analyzer.analyze_commit_messages()

    types = analysis["commit_types"]
    assert types["feat"] >= 1
    assert types["fix"] >= 1
    assert types["docs"] >= 1
    assert types["refactor"] >= 1
    assert types["test"] >= 1


def test_breaking_change_detection(temp_git_repo):
    """Test detection of breaking changes."""
    analyzer = CommitAnalyzer(str(temp_git_repo))

    # Create a breaking change commit
    test_file = temp_git_repo / "breaking.txt"
    test_file.write_text("breaking change")
    subprocess.run(["git", "add", "."], cwd=temp_git_repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "feat!: breaking API change\n\nBREAKING CHANGE: API restructured"],
        cwd=temp_git_repo,
        check=True,
    )

    analysis = analyzer.analyze_commit_messages()

    assert analysis["breaking_changes"] >= 1
