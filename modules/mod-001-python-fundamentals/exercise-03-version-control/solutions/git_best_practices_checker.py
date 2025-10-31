#!/usr/bin/env python3
"""
Git Best Practices Checker

Audits Git repositories for adherence to best practices including
.gitignore, commit conventions, branch strategies, and security.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
import json


class BestPracticesChecker:
    """Check Git repositories for best practices compliance."""

    def __init__(self, repo_path: Optional[str] = None):
        """
        Initialize best practices checker.

        Args:
            repo_path: Path to Git repository. Defaults to current directory.
        """
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self._validate_git_repo()
        self.issues = []
        self.warnings = []
        self.passes = []

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

    def check_gitignore_exists(self) -> bool:
        """Check if .gitignore file exists."""
        gitignore_path = self.repo_path / ".gitignore"
        if gitignore_path.exists():
            self.passes.append(".gitignore file exists")
            return True
        else:
            self.issues.append(".gitignore file is missing")
            return False

    def check_gitignore_patterns(self) -> None:
        """Check for common important .gitignore patterns."""
        gitignore_path = self.repo_path / ".gitignore"
        if not gitignore_path.exists():
            return

        content = gitignore_path.read_text()

        # Important patterns for Python/ML projects
        important_patterns = {
            "__pycache__": "Python cache directories",
            "*.pyc": "Python compiled files",
            ".env": "Environment files with secrets",
            "venv/": "Virtual environment",
            "*.log": "Log files",
            ".DS_Store": "macOS system files",
        }

        missing_patterns = []
        for pattern, description in important_patterns.items():
            if pattern not in content:
                missing_patterns.append(f"{pattern} ({description})")

        if missing_patterns:
            self.warnings.append(
                f".gitignore missing common patterns: {', '.join(missing_patterns)}"
            )
        else:
            self.passes.append(".gitignore has common important patterns")

    def check_large_files(self, max_size_mb: int = 10) -> None:
        """
        Check for large files in repository.

        Args:
            max_size_mb: Maximum file size in MB
        """
        _, output, _ = self._run_command(
            ["git", "ls-files"], check=False
        )

        large_files = []
        for file_path in output.split("\n"):
            if file_path:
                full_path = self.repo_path / file_path
                if full_path.exists():
                    size_mb = full_path.stat().st_size / (1024 * 1024)
                    if size_mb > max_size_mb:
                        large_files.append(f"{file_path} ({size_mb:.1f}MB)")

        if large_files:
            self.issues.append(
                f"Large files found (>{max_size_mb}MB): {', '.join(large_files[:5])}"
            )
        else:
            self.passes.append(f"No files larger than {max_size_mb}MB")

    def check_sensitive_files(self) -> None:
        """Check for potentially sensitive files."""
        _, output, _ = self._run_command(["git", "ls-files"], check=False)

        sensitive_patterns = [
            r"\.env$",
            r"\.pem$",
            r"\.key$",
            r"credentials",
            r"secret",
            r"password",
            r"id_rsa",
            r"\.p12$",
        ]

        sensitive_files = []
        for file_path in output.split("\n"):
            for pattern in sensitive_patterns:
                if re.search(pattern, file_path, re.IGNORECASE):
                    sensitive_files.append(file_path)
                    break

        if sensitive_files:
            self.issues.append(
                f"Potentially sensitive files found: {', '.join(sensitive_files[:5])}"
            )
        else:
            self.passes.append("No obviously sensitive files detected")

    def check_commit_message_quality(self, limit: int = 50) -> None:
        """
        Check recent commit messages for quality.

        Args:
            limit: Number of recent commits to check
        """
        _, output, _ = self._run_command(
            ["git", "log", f"-{limit}", "--pretty=%s"], check=False
        )

        messages = output.split("\n")
        conventional_pattern = re.compile(
            r"^(feat|fix|docs|style|refactor|test|chore|perf|ci|build)"
            r"(\([a-zA-Z0-9_-]+\))?"
            r"(!)?:\s+.+"
        )

        conventional_count = sum(
            1 for msg in messages if conventional_pattern.match(msg)
        )
        total = len(messages)

        if total == 0:
            return

        percentage = (conventional_count / total) * 100

        if percentage >= 80:
            self.passes.append(
                f"{percentage:.0f}% of recent commits follow conventional format"
            )
        elif percentage >= 50:
            self.warnings.append(
                f"Only {percentage:.0f}% of commits follow conventional format"
            )
        else:
            self.issues.append(
                f"Only {percentage:.0f}% of commits follow conventional format"
            )

        # Check for bad commit messages
        bad_patterns = [r"^wip$", r"^fix$", r"^update$", r"^changes$", r"^asdf"]
        bad_messages = []
        for msg in messages[:10]:  # Check most recent 10
            for pattern in bad_patterns:
                if re.match(pattern, msg.lower()):
                    bad_messages.append(msg)
                    break

        if bad_messages:
            self.warnings.append(
                f"Vague commit messages found: {', '.join(bad_messages[:3])}"
            )

    def check_branch_strategy(self) -> None:
        """Check if repository follows a branch strategy."""
        _, output, _ = self._run_command(["git", "branch", "-a"], check=False)

        branches = [
            line.strip().lstrip("* ").strip()
            for line in output.split("\n")
            if line.strip()
        ]

        # Look for branch strategy indicators
        has_develop = any("develop" in b for b in branches)
        has_feature_branches = any("feature/" in b for b in branches)
        has_release_branches = any("release/" in b for b in branches)

        strategy_score = sum([has_develop, has_feature_branches, has_release_branches])

        if strategy_score >= 2:
            self.passes.append("Repository follows a branch strategy (Git Flow/GitHub Flow)")
        elif strategy_score == 1:
            self.warnings.append("Partial branch strategy detected - consider formalizing")
        else:
            self.warnings.append("No clear branch strategy - consider implementing Git Flow")

    def check_readme_exists(self) -> None:
        """Check if README file exists."""
        readme_files = ["README.md", "README.rst", "README.txt", "README"]
        readme_exists = any((self.repo_path / f).exists() for f in readme_files)

        if readme_exists:
            self.passes.append("README file exists")
        else:
            self.issues.append("README file is missing")

    def check_git_config(self) -> None:
        """Check Git configuration."""
        # Check user name and email
        _, name, _ = self._run_command(["git", "config", "user.name"], check=False)
        _, email, _ = self._run_command(["git", "config", "user.email"], check=False)

        if name and email:
            self.passes.append("Git user name and email configured")
        else:
            self.issues.append("Git user name or email not configured")

        # Check for useful aliases
        _, aliases, _ = self._run_command(
            ["git", "config", "--get-regexp", "^alias\\."], check=False
        )

        if aliases:
            self.passes.append("Git aliases configured")
        else:
            self.warnings.append("No Git aliases configured (optional but helpful)")

    def check_hooks_installed(self) -> None:
        """Check if Git hooks are installed."""
        hooks_dir = self.repo_path / ".git" / "hooks"
        if not hooks_dir.exists():
            return

        # Check for common hooks
        hook_files = ["pre-commit", "commit-msg", "pre-push", "post-commit"]
        installed_hooks = []

        for hook in hook_files:
            hook_path = hooks_dir / hook
            if hook_path.exists() and hook_path.stat().st_size > 0:
                # Check if executable
                if hook_path.stat().st_mode & 0o111:
                    installed_hooks.append(hook)

        if installed_hooks:
            self.passes.append(f"Git hooks installed: {', '.join(installed_hooks)}")
        else:
            self.warnings.append("No Git hooks installed (consider adding pre-commit hooks)")

    def check_default_branch(self) -> None:
        """Check default branch name."""
        _, branch, _ = self._run_command(
            ["git", "symbolic-ref", "refs/remotes/origin/HEAD"], check=False
        )

        if "main" in branch:
            self.passes.append("Uses 'main' as default branch (modern convention)")
        elif "master" in branch:
            self.warnings.append("Uses 'master' as default branch (consider renaming to 'main')")
        else:
            self.warnings.append("Unclear default branch")

    def check_merge_commits(self) -> None:
        """Check for merge commits vs. squash/rebase."""
        _, output, _ = self._run_command(
            ["git", "log", "--merges", "--oneline", "-10"], check=False
        )

        merge_commits = len(output.split("\n")) if output else 0

        if merge_commits > 5:
            self.warnings.append(
                "Many merge commits detected - consider using rebase or squash"
            )
        else:
            self.passes.append("Clean commit history (few merge commits)")

    def check_remote_configured(self) -> None:
        """Check if remote repository is configured."""
        _, output, _ = self._run_command(["git", "remote", "-v"], check=False)

        if "origin" in output:
            self.passes.append("Remote repository (origin) configured")
        else:
            self.warnings.append("No remote repository configured")

    def run_all_checks(self) -> Dict[str, any]:
        """
        Run all checks and return results.

        Returns:
            Dictionary with check results
        """
        print("Running Git best practices checks...\n")

        self.check_gitignore_exists()
        self.check_gitignore_patterns()
        self.check_large_files()
        self.check_sensitive_files()
        self.check_commit_message_quality()
        self.check_branch_strategy()
        self.check_readme_exists()
        self.check_git_config()
        self.check_hooks_installed()
        self.check_default_branch()
        self.check_merge_commits()
        self.check_remote_configured()

        # Calculate score
        total_checks = len(self.issues) + len(self.warnings) + len(self.passes)
        score = (len(self.passes) / total_checks * 100) if total_checks > 0 else 0

        return {
            "repository": str(self.repo_path),
            "score": score,
            "issues": self.issues,
            "warnings": self.warnings,
            "passes": self.passes,
        }

    def print_report(self, results: Dict[str, any]) -> None:
        """
        Print formatted report.

        Args:
            results: Results dictionary from run_all_checks
        """
        print("=" * 70)
        print("GIT BEST PRACTICES REPORT")
        print("=" * 70)
        print(f"\nRepository: {results['repository']}")
        print(f"Score: {results['score']:.1f}%\n")

        if results["issues"]:
            print("\nISSUES (must fix):")
            for issue in results["issues"]:
                print(f"  ✗ {issue}")

        if results["warnings"]:
            print("\nWARNINGS (should fix):")
            for warning in results["warnings"]:
                print(f"  ⚠ {warning}")

        if results["passes"]:
            print("\nPASSED CHECKS:")
            for passed in results["passes"]:
                print(f"  ✓ {passed}")

        print("\n" + "=" * 70)

        if results["score"] >= 80:
            print("✓ EXCELLENT - Repository follows Git best practices!")
        elif results["score"] >= 60:
            print("⚠ GOOD - Some improvements recommended")
        else:
            print("✗ NEEDS IMPROVEMENT - Address issues above")

        print("=" * 70 + "\n")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Check Git repository for best practices compliance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--repo",
        help="Path to Git repository (default: current directory)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--output",
        help="Write results to file",
    )

    args = parser.parse_args()

    try:
        checker = BestPracticesChecker(args.repo)
        results = checker.run_all_checks()

        if args.json:
            output = json.dumps(results, indent=2)
            if args.output:
                Path(args.output).write_text(output)
                print(f"Results written to: {args.output}")
            else:
                print(output)
        else:
            checker.print_report(results)
            if args.output:
                # Save text report
                with open(args.output, "w") as f:
                    original_stdout = sys.stdout
                    sys.stdout = f
                    checker.print_report(results)
                    sys.stdout = original_stdout
                print(f"Report written to: {args.output}")

        # Exit with error code if score is too low
        if results["score"] < 60:
            return 1

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
