#!/usr/bin/env python3
"""
Branch Manager Tool

Advanced branch management for Git repositories including branch strategies,
automated merges, and branch lifecycle management.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import re


class BranchManager:
    """Manage Git branches with best practices for AI/ML projects."""

    def __init__(self, repo_path: Optional[str] = None):
        """
        Initialize branch manager.

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

    def get_branch_info(self, branch_name: str) -> Dict[str, any]:
        """
        Get detailed information about a branch.

        Args:
            branch_name: Name of the branch

        Returns:
            Dictionary with branch information
        """
        # Check if branch exists
        _, branches_output, _ = self._run_command(["git", "branch", "-a"])
        if branch_name not in branches_output:
            raise ValueError(f"Branch does not exist: {branch_name}")

        info = {"name": branch_name}

        # Get last commit info
        _, commit_hash, _ = self._run_command(
            ["git", "rev-parse", branch_name], check=False
        )
        info["last_commit"] = commit_hash

        # Get last commit date
        _, commit_date, _ = self._run_command(
            ["git", "log", "-1", "--format=%ci", branch_name], check=False
        )
        info["last_commit_date"] = commit_date

        # Get commit count
        _, commit_count, _ = self._run_command(
            ["git", "rev-list", "--count", branch_name], check=False
        )
        info["commit_count"] = int(commit_count) if commit_count else 0

        # Check if merged into main
        returncode, _, _ = self._run_command(
            ["git", "branch", "--merged", "main"], check=False
        )
        info["merged_to_main"] = returncode == 0 and branch_name in _

        # Get branch type from name
        if branch_name.startswith("feature/"):
            info["type"] = "feature"
        elif branch_name.startswith("bugfix/"):
            info["type"] = "bugfix"
        elif branch_name.startswith("hotfix/"):
            info["type"] = "hotfix"
        elif branch_name.startswith("release/"):
            info["type"] = "release"
        else:
            info["type"] = "other"

        return info

    def list_branches_by_type(self, branch_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List all branches grouped by type.

        Args:
            branch_type: Filter by specific type (feature, bugfix, hotfix, release)

        Returns:
            Dictionary mapping branch types to branch names
        """
        _, branches_output, _ = self._run_command(["git", "branch"])

        branches = {
            "feature": [],
            "bugfix": [],
            "hotfix": [],
            "release": [],
            "other": [],
        }

        for line in branches_output.split("\n"):
            branch = line.strip().lstrip("* ").strip()
            if not branch:
                continue

            if branch.startswith("feature/"):
                branches["feature"].append(branch)
            elif branch.startswith("bugfix/"):
                branches["bugfix"].append(branch)
            elif branch.startswith("hotfix/"):
                branches["hotfix"].append(branch)
            elif branch.startswith("release/"):
                branches["release"].append(branch)
            else:
                branches["other"].append(branch)

        if branch_type:
            return {branch_type: branches.get(branch_type, [])}

        return branches

    def find_stale_branches(self, days: int = 30) -> List[Dict[str, any]]:
        """
        Find branches that haven't been updated in specified days.

        Args:
            days: Number of days to consider a branch stale

        Returns:
            List of stale branch information
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        stale_branches = []

        _, branches_output, _ = self._run_command(["git", "branch"])

        for line in branches_output.split("\n"):
            branch = line.strip().lstrip("* ").strip()
            if not branch or branch in ["main", "master", "develop"]:
                continue

            # Get last commit date
            _, date_str, _ = self._run_command(
                ["git", "log", "-1", "--format=%ci", branch], check=False
            )

            if date_str:
                # Parse date (format: 2024-01-15 10:30:45 -0500)
                try:
                    commit_date = datetime.strptime(
                        date_str.split()[0], "%Y-%m-%d"
                    )
                    if commit_date < cutoff_date:
                        info = self.get_branch_info(branch)
                        info["days_old"] = (datetime.now() - commit_date).days
                        stale_branches.append(info)
                except ValueError:
                    continue

        return sorted(stale_branches, key=lambda x: x["days_old"], reverse=True)

    def create_release_branch(self, version: str, base_branch: str = "develop") -> str:
        """
        Create a release branch following semantic versioning.

        Args:
            version: Version number (e.g., 1.0.0)
            base_branch: Base branch to create from

        Returns:
            Name of created release branch
        """
        # Validate version format
        if not re.match(r"^\d+\.\d+\.\d+$", version):
            raise ValueError(
                f"Invalid version format: {version}. Use semantic versioning (e.g., 1.0.0)"
            )

        branch_name = f"release/{version}"

        # Check if release branch already exists
        _, branches, _ = self._run_command(["git", "branch"])
        if branch_name in branches:
            raise ValueError(f"Release branch already exists: {branch_name}")

        # Checkout base branch and pull
        self._run_command(["git", "checkout", base_branch])
        self._run_command(["git", "pull", "origin", base_branch])

        # Create release branch
        self._run_command(["git", "checkout", "-b", branch_name])

        print(f"Created release branch: {branch_name}")
        return branch_name

    def compare_branches(self, branch1: str, branch2: str) -> Dict[str, any]:
        """
        Compare two branches.

        Args:
            branch1: First branch
            branch2: Second branch

        Returns:
            Dictionary with comparison results
        """
        comparison = {
            "branch1": branch1,
            "branch2": branch2,
        }

        # Commits in branch1 but not in branch2
        _, ahead, _ = self._run_command(
            ["git", "rev-list", "--count", f"{branch2}..{branch1}"], check=False
        )
        comparison["commits_ahead"] = int(ahead) if ahead else 0

        # Commits in branch2 but not in branch1
        _, behind, _ = self._run_command(
            ["git", "rev-list", "--count", f"{branch1}..{branch2}"], check=False
        )
        comparison["commits_behind"] = int(behind) if behind else 0

        # Get diverged files
        _, diff_output, _ = self._run_command(
            ["git", "diff", "--name-only", branch1, branch2], check=False
        )
        comparison["different_files"] = (
            diff_output.split("\n") if diff_output else []
        )
        comparison["files_changed"] = len(comparison["different_files"])

        # Check if can fast-forward merge
        returncode, _, _ = self._run_command(
            ["git", "merge-base", "--is-ancestor", branch1, branch2], check=False
        )
        comparison["can_fast_forward"] = returncode == 0

        return comparison

    def merge_with_strategy(
        self,
        source_branch: str,
        target_branch: str,
        strategy: str = "merge",
        delete_source: bool = False,
    ) -> None:
        """
        Merge branches with specified strategy.

        Args:
            source_branch: Branch to merge from
            target_branch: Branch to merge into
            strategy: Merge strategy (merge, squash, rebase)
            delete_source: Delete source branch after merge
        """
        valid_strategies = ["merge", "squash", "rebase"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {strategy}. Use one of {valid_strategies}")

        # Checkout target branch
        self._run_command(["git", "checkout", target_branch])
        self._run_command(["git", "pull", "origin", target_branch])

        if strategy == "merge":
            # Standard merge with no-ff
            self._run_command(["git", "merge", "--no-ff", source_branch])
            print(f"Merged {source_branch} into {target_branch} (merge commit)")

        elif strategy == "squash":
            # Squash merge
            self._run_command(["git", "merge", "--squash", source_branch])
            self._run_command(
                ["git", "commit", "-m", f"Merge branch '{source_branch}' (squashed)"]
            )
            print(f"Merged {source_branch} into {target_branch} (squashed)")

        elif strategy == "rebase":
            # Rebase and merge
            self._run_command(["git", "checkout", source_branch])
            self._run_command(["git", "rebase", target_branch])
            self._run_command(["git", "checkout", target_branch])
            self._run_command(["git", "merge", source_branch])
            print(f"Merged {source_branch} into {target_branch} (rebased)")

        # Delete source branch if requested
        if delete_source and source_branch not in ["main", "master", "develop"]:
            self._run_command(["git", "branch", "-d", source_branch])
            print(f"Deleted source branch: {source_branch}")

    def protect_branch(self, branch_name: str) -> None:
        """
        Set up local branch protection rules.

        Args:
            branch_name: Branch to protect
        """
        # Create a pre-push hook to prevent direct pushes
        hooks_dir = self.repo_path / ".git" / "hooks"
        pre_push_hook = hooks_dir / "pre-push"

        protected_branches = ["main", "master", "develop"]
        if branch_name not in protected_branches:
            protected_branches.append(branch_name)

        hook_content = f"""#!/bin/bash
# Pre-push hook to protect branches

protected_branches=({' '.join(f'"{b}"' for b in protected_branches)})

current_branch=$(git symbolic-ref HEAD | sed -e 's,.*/\\(.*\\),\\1,')

for branch in "${{protected_branches[@]}}"; do
    if [ "$current_branch" = "$branch" ]; then
        echo "Direct pushes to $branch are not allowed!"
        echo "Please use a pull request workflow."
        exit 1
    fi
done

exit 0
"""

        pre_push_hook.write_text(hook_content)
        pre_push_hook.chmod(0o755)
        print(f"Protected branch: {branch_name}")
        print("Note: This is a local protection. Configure GitHub/GitLab for remote protection.")

    def visualize_branch_tree(self) -> None:
        """Display a visual tree of branches and their relationships."""
        print("\nBranch Tree:")
        print("=" * 60)
        self._run_command(
            [
                "git",
                "log",
                "--graph",
                "--oneline",
                "--all",
                "--decorate",
                "--max-count=20",
            ]
        )


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Advanced Git Branch Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--repo", help="Path to Git repository (default: current dir)")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List branches
    list_parser = subparsers.add_parser("list", help="List branches by type")
    list_parser.add_argument(
        "--type",
        choices=["feature", "bugfix", "hotfix", "release", "other"],
        help="Filter by branch type",
    )

    # Branch info
    info_parser = subparsers.add_parser("info", help="Get branch information")
    info_parser.add_argument("branch", help="Branch name")

    # Find stale branches
    stale_parser = subparsers.add_parser("stale", help="Find stale branches")
    stale_parser.add_argument(
        "--days", type=int, default=30, help="Days to consider stale (default: 30)"
    )

    # Create release branch
    release_parser = subparsers.add_parser("release", help="Create release branch")
    release_parser.add_argument("version", help="Version number (e.g., 1.0.0)")
    release_parser.add_argument(
        "--base", default="develop", help="Base branch (default: develop)"
    )

    # Compare branches
    compare_parser = subparsers.add_parser("compare", help="Compare two branches")
    compare_parser.add_argument("branch1", help="First branch")
    compare_parser.add_argument("branch2", help="Second branch")

    # Merge with strategy
    merge_parser = subparsers.add_parser("merge", help="Merge with strategy")
    merge_parser.add_argument("source", help="Source branch")
    merge_parser.add_argument("target", help="Target branch")
    merge_parser.add_argument(
        "--strategy",
        choices=["merge", "squash", "rebase"],
        default="merge",
        help="Merge strategy",
    )
    merge_parser.add_argument(
        "--delete", action="store_true", help="Delete source branch after merge"
    )

    # Protect branch
    protect_parser = subparsers.add_parser("protect", help="Protect a branch")
    protect_parser.add_argument("branch", help="Branch to protect")

    # Visualize
    subparsers.add_parser("tree", help="Visualize branch tree")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        manager = BranchManager(args.repo)

        if args.command == "list":
            branches = manager.list_branches_by_type(args.type)
            for branch_type, branch_list in branches.items():
                if branch_list:
                    print(f"\n{branch_type.upper()} branches:")
                    for branch in branch_list:
                        print(f"  - {branch}")

        elif args.command == "info":
            info = manager.get_branch_info(args.branch)
            print(f"\nBranch: {info['name']}")
            print(f"Type: {info['type']}")
            print(f"Last Commit: {info['last_commit']}")
            print(f"Last Commit Date: {info['last_commit_date']}")
            print(f"Total Commits: {info['commit_count']}")
            print(f"Merged to Main: {info['merged_to_main']}")

        elif args.command == "stale":
            stale = manager.find_stale_branches(args.days)
            if not stale:
                print(f"No stale branches found (older than {args.days} days)")
            else:
                print(f"\nStale branches (older than {args.days} days):")
                for branch in stale:
                    print(f"  - {branch['name']} ({branch['days_old']} days old)")

        elif args.command == "release":
            branch = manager.create_release_branch(args.version, args.base)
            print(f"Created and switched to: {branch}")

        elif args.command == "compare":
            comparison = manager.compare_branches(args.branch1, args.branch2)
            print(f"\nComparing {comparison['branch1']} with {comparison['branch2']}:")
            print(f"  Commits ahead: {comparison['commits_ahead']}")
            print(f"  Commits behind: {comparison['commits_behind']}")
            print(f"  Files changed: {comparison['files_changed']}")
            print(f"  Can fast-forward: {comparison['can_fast_forward']}")

        elif args.command == "merge":
            manager.merge_with_strategy(
                args.source, args.target, args.strategy, args.delete
            )

        elif args.command == "protect":
            manager.protect_branch(args.branch)

        elif args.command == "tree":
            manager.visualize_branch_tree()

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
