#!/usr/bin/env python3
"""
Git Workflow Automation Tool

Automates common Git workflows including branch creation, commits, and merges.
Designed for junior AI infrastructure engineers to standardize Git operations.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import json
import re
from datetime import datetime


class GitWorkflowAutomation:
    """Automate common Git workflows with best practices."""

    def __init__(self, repo_path: Optional[str] = None):
        """
        Initialize Git workflow automation.

        Args:
            repo_path: Path to Git repository. Defaults to current directory.
        """
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self._validate_git_repo()

    def _validate_git_repo(self) -> None:
        """Validate that the current directory is a Git repository."""
        git_dir = self.repo_path / ".git"
        if not git_dir.exists():
            raise ValueError(f"Not a git repository: {self.repo_path}")

    def _run_command(
        self, command: List[str], check: bool = True
    ) -> Tuple[int, str, str]:
        """
        Run a shell command and return the result.

        Args:
            command: Command to run as list of strings
            check: Whether to raise exception on failure

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        try:
            result = subprocess.run(
                command,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=check,
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except subprocess.CalledProcessError as e:
            if check:
                raise RuntimeError(f"Command failed: {' '.join(command)}\n{e.stderr}")
            return e.returncode, e.stdout, e.stderr

    def get_current_branch(self) -> str:
        """Get the current Git branch name."""
        _, stdout, _ = self._run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        return stdout

    def get_all_branches(self, remote: bool = False) -> List[str]:
        """
        Get all branch names.

        Args:
            remote: Include remote branches if True

        Returns:
            List of branch names
        """
        if remote:
            _, stdout, _ = self._run_command(["git", "branch", "-a"])
        else:
            _, stdout, _ = self._run_command(["git", "branch"])

        branches = []
        for line in stdout.split("\n"):
            branch = line.strip().lstrip("* ").strip()
            if branch and not branch.startswith("remotes/"):
                branches.append(branch)
        return branches

    def create_feature_branch(self, feature_name: str, base_branch: str = "main") -> str:
        """
        Create a new feature branch with naming convention.

        Args:
            feature_name: Name of the feature
            base_branch: Base branch to branch from

        Returns:
            Name of created branch
        """
        # Sanitize feature name
        sanitized_name = re.sub(r"[^a-zA-Z0-9-]", "-", feature_name.lower())
        branch_name = f"feature/{sanitized_name}"

        # Ensure we're on the base branch and up to date
        print(f"Checking out {base_branch}...")
        self._run_command(["git", "checkout", base_branch])

        print(f"Pulling latest changes from {base_branch}...")
        self._run_command(["git", "pull", "origin", base_branch])

        # Create and checkout new branch
        print(f"Creating branch: {branch_name}")
        self._run_command(["git", "checkout", "-b", branch_name])

        return branch_name

    def create_bugfix_branch(self, bug_name: str, base_branch: str = "main") -> str:
        """
        Create a new bugfix branch with naming convention.

        Args:
            bug_name: Name/description of the bug
            base_branch: Base branch to branch from

        Returns:
            Name of created branch
        """
        sanitized_name = re.sub(r"[^a-zA-Z0-9-]", "-", bug_name.lower())
        branch_name = f"bugfix/{sanitized_name}"

        self._run_command(["git", "checkout", base_branch])
        self._run_command(["git", "pull", "origin", base_branch])
        self._run_command(["git", "checkout", "-b", branch_name])

        return branch_name

    def create_hotfix_branch(self, hotfix_name: str, base_branch: str = "main") -> str:
        """
        Create a new hotfix branch for urgent production fixes.

        Args:
            hotfix_name: Name/description of the hotfix
            base_branch: Base branch to branch from

        Returns:
            Name of created branch
        """
        sanitized_name = re.sub(r"[^a-zA-Z0-9-]", "-", hotfix_name.lower())
        branch_name = f"hotfix/{sanitized_name}"

        self._run_command(["git", "checkout", base_branch])
        self._run_command(["git", "pull", "origin", base_branch])
        self._run_command(["git", "checkout", "-b", branch_name])

        return branch_name

    def smart_commit(
        self,
        commit_type: str,
        scope: Optional[str],
        message: str,
        body: Optional[str] = None,
        breaking: bool = False,
    ) -> None:
        """
        Create a commit following conventional commits standard.

        Args:
            commit_type: Type of commit (feat, fix, docs, etc.)
            scope: Scope of the change (optional)
            message: Commit message
            body: Extended commit body (optional)
            breaking: Whether this is a breaking change
        """
        valid_types = [
            "feat",
            "fix",
            "docs",
            "style",
            "refactor",
            "test",
            "chore",
            "perf",
            "ci",
            "build",
        ]

        if commit_type not in valid_types:
            raise ValueError(
                f"Invalid commit type: {commit_type}. Must be one of {valid_types}"
            )

        # Build commit message
        scope_str = f"({scope})" if scope else ""
        breaking_str = "!" if breaking else ""
        full_message = f"{commit_type}{scope_str}{breaking_str}: {message}"

        if body:
            full_message += f"\n\n{body}"

        if breaking:
            full_message += "\n\nBREAKING CHANGE: This commit contains breaking changes."

        # Stage all changes
        self._run_command(["git", "add", "."])

        # Create commit
        self._run_command(["git", "commit", "-m", full_message])
        print(f"Created commit: {full_message.split(chr(10))[0]}")

    def merge_branch(
        self, source_branch: str, target_branch: str, no_ff: bool = True
    ) -> None:
        """
        Merge a branch into target branch.

        Args:
            source_branch: Branch to merge from
            target_branch: Branch to merge into
            no_ff: Use --no-ff flag for merge commit
        """
        print(f"Switching to {target_branch}...")
        self._run_command(["git", "checkout", target_branch])

        print(f"Pulling latest changes...")
        self._run_command(["git", "pull", "origin", target_branch])

        print(f"Merging {source_branch} into {target_branch}...")
        merge_cmd = ["git", "merge", source_branch]
        if no_ff:
            merge_cmd.append("--no-ff")

        self._run_command(merge_cmd)
        print("Merge successful!")

    def cleanup_merged_branches(self, dry_run: bool = True) -> List[str]:
        """
        Clean up branches that have been merged.

        Args:
            dry_run: If True, only show what would be deleted

        Returns:
            List of deleted (or would-be-deleted) branches
        """
        # Get list of merged branches
        _, stdout, _ = self._run_command(["git", "branch", "--merged"])

        merged_branches = []
        for line in stdout.split("\n"):
            branch = line.strip().lstrip("* ").strip()
            # Don't delete main, master, develop, or current branch
            if branch and branch not in ["main", "master", "develop"]:
                merged_branches.append(branch)

        if not merged_branches:
            print("No merged branches to clean up.")
            return []

        print(f"\nFound {len(merged_branches)} merged branches:")
        for branch in merged_branches:
            print(f"  - {branch}")

        if dry_run:
            print("\nDry run mode. Use --no-dry-run to actually delete.")
            return merged_branches

        # Delete branches
        for branch in merged_branches:
            self._run_command(["git", "branch", "-d", branch])
            print(f"Deleted branch: {branch}")

        return merged_branches

    def sync_with_remote(self, branch: Optional[str] = None) -> None:
        """
        Sync local branch with remote.

        Args:
            branch: Branch to sync. Defaults to current branch.
        """
        if branch:
            self._run_command(["git", "checkout", branch])

        current = self.get_current_branch()
        print(f"Syncing branch: {current}")

        # Fetch all changes
        self._run_command(["git", "fetch", "origin"])

        # Pull with rebase
        self._run_command(["git", "pull", "--rebase", "origin", current])
        print("Sync complete!")

    def get_status_summary(self) -> dict:
        """
        Get a summary of repository status.

        Returns:
            Dictionary with status information
        """
        current_branch = self.get_current_branch()

        # Get status
        _, status_output, _ = self._run_command(["git", "status", "--porcelain"])

        # Count changes
        modified = len([l for l in status_output.split("\n") if l.startswith(" M")])
        added = len([l for l in status_output.split("\n") if l.startswith("A ")])
        deleted = len([l for l in status_output.split("\n") if l.startswith(" D")])
        untracked = len([l for l in status_output.split("\n") if l.startswith("??")])

        # Get unpushed commits
        returncode, ahead_output, _ = self._run_command(
            [
                "git",
                "rev-list",
                "--count",
                f"origin/{current_branch}..{current_branch}",
            ],
            check=False,
        )
        unpushed = int(ahead_output) if returncode == 0 else 0

        return {
            "branch": current_branch,
            "modified": modified,
            "added": added,
            "deleted": deleted,
            "untracked": untracked,
            "unpushed_commits": unpushed,
            "clean": status_output == "",
        }

    def quick_save(self, message: str = "WIP: Quick save") -> None:
        """
        Quickly save current work with a WIP commit.

        Args:
            message: Commit message (defaults to WIP)
        """
        self._run_command(["git", "add", "."])
        self._run_command(["git", "commit", "-m", message])
        print(f"Quick save created: {message}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Git Workflow Automation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--repo", help="Path to Git repository (default: current dir)")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Create branch commands
    branch_parser = subparsers.add_parser("branch", help="Branch operations")
    branch_subparsers = branch_parser.add_subparsers(dest="branch_command")

    # Feature branch
    feature_parser = branch_subparsers.add_parser(
        "feature", help="Create feature branch"
    )
    feature_parser.add_argument("name", help="Feature name")
    feature_parser.add_argument(
        "--base", default="main", help="Base branch (default: main)"
    )

    # Bugfix branch
    bugfix_parser = branch_subparsers.add_parser("bugfix", help="Create bugfix branch")
    bugfix_parser.add_argument("name", help="Bug name/description")
    bugfix_parser.add_argument(
        "--base", default="main", help="Base branch (default: main)"
    )

    # Hotfix branch
    hotfix_parser = branch_subparsers.add_parser("hotfix", help="Create hotfix branch")
    hotfix_parser.add_argument("name", help="Hotfix name/description")
    hotfix_parser.add_argument(
        "--base", default="main", help="Base branch (default: main)"
    )

    # List branches
    branch_subparsers.add_parser("list", help="List all branches")

    # Commit command
    commit_parser = subparsers.add_parser("commit", help="Create conventional commit")
    commit_parser.add_argument(
        "type",
        choices=[
            "feat",
            "fix",
            "docs",
            "style",
            "refactor",
            "test",
            "chore",
            "perf",
            "ci",
            "build",
        ],
        help="Commit type",
    )
    commit_parser.add_argument("message", help="Commit message")
    commit_parser.add_argument("--scope", help="Scope of the change")
    commit_parser.add_argument("--body", help="Extended commit body")
    commit_parser.add_argument(
        "--breaking", action="store_true", help="Mark as breaking change"
    )

    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge branches")
    merge_parser.add_argument("source", help="Source branch")
    merge_parser.add_argument("target", help="Target branch")
    merge_parser.add_argument(
        "--ff", action="store_true", help="Allow fast-forward merge"
    )

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up merged branches")
    cleanup_parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete branches (default is dry run)",
    )

    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Sync with remote")
    sync_parser.add_argument("--branch", help="Branch to sync (default: current)")

    # Status command
    subparsers.add_parser("status", help="Get repository status summary")

    # Quick save command
    quicksave_parser = subparsers.add_parser("quicksave", help="Quick WIP save")
    quicksave_parser.add_argument(
        "--message", default="WIP: Quick save", help="Commit message"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        git = GitWorkflowAutomation(args.repo)

        if args.command == "branch":
            if args.branch_command == "feature":
                branch = git.create_feature_branch(args.name, args.base)
                print(f"Created and switched to: {branch}")
            elif args.branch_command == "bugfix":
                branch = git.create_bugfix_branch(args.name, args.base)
                print(f"Created and switched to: {branch}")
            elif args.branch_command == "hotfix":
                branch = git.create_hotfix_branch(args.name, args.base)
                print(f"Created and switched to: {branch}")
            elif args.branch_command == "list":
                branches = git.get_all_branches()
                current = git.get_current_branch()
                print("Branches:")
                for branch in branches:
                    marker = "*" if branch == current else " "
                    print(f"  {marker} {branch}")
            else:
                branch_parser.print_help()

        elif args.command == "commit":
            git.smart_commit(
                args.type,
                args.scope,
                args.message,
                args.body,
                args.breaking,
            )

        elif args.command == "merge":
            git.merge_branch(args.source, args.target, no_ff=not args.ff)

        elif args.command == "cleanup":
            git.cleanup_merged_branches(dry_run=not args.execute)

        elif args.command == "sync":
            git.sync_with_remote(args.branch)

        elif args.command == "status":
            status = git.get_status_summary()
            print("\nRepository Status:")
            print(f"  Branch: {status['branch']}")
            print(f"  Modified: {status['modified']}")
            print(f"  Added: {status['added']}")
            print(f"  Deleted: {status['deleted']}")
            print(f"  Untracked: {status['untracked']}")
            print(f"  Unpushed commits: {status['unpushed_commits']}")
            print(f"  Clean: {status['clean']}")

        elif args.command == "quicksave":
            git.quick_save(args.message)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
