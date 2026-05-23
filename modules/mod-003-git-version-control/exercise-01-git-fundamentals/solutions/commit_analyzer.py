#!/usr/bin/env python3
"""
Commit Analyzer

Analyze Git commit history, generate reports, and extract insights
about development patterns and code quality.
"""

import argparse
import subprocess
import sys
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json


class CommitAnalyzer:
    """Analyze Git commit history and generate insights."""

    def __init__(self, repo_path: Optional[str] = None):
        """
        Initialize commit analyzer.

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

    def get_commits(
        self,
        branch: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        author: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Get commit history with filters.

        Args:
            branch: Branch to analyze (default: current)
            since: Start date (e.g., '2024-01-01' or '2 weeks ago')
            until: End date
            author: Filter by author
            limit: Maximum number of commits

        Returns:
            List of commit dictionaries
        """
        cmd = [
            "git",
            "log",
            "--format=%H|%an|%ae|%ai|%s|%b",
        ]

        if branch:
            cmd.append(branch)

        if since:
            cmd.append(f"--since={since}")

        if until:
            cmd.append(f"--until={until}")

        if author:
            cmd.append(f"--author={author}")

        if limit:
            cmd.append(f"-n{limit}")

        _, output, _ = self._run_command(cmd)

        commits = []
        for line in output.split("\n"):
            if not line:
                continue

            parts = line.split("|")
            if len(parts) >= 5:
                commits.append(
                    {
                        "hash": parts[0],
                        "author": parts[1],
                        "email": parts[2],
                        "date": parts[3],
                        "message": parts[4],
                        "body": parts[5] if len(parts) > 5 else "",
                    }
                )

        return commits

    def analyze_commit_messages(
        self, commits: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, any]:
        """
        Analyze commit message patterns and quality.

        Args:
            commits: List of commits to analyze (default: all commits)

        Returns:
            Dictionary with analysis results
        """
        if commits is None:
            commits = self.get_commits()

        analysis = {
            "total_commits": len(commits),
            "conventional_commits": 0,
            "commit_types": Counter(),
            "scoped_commits": 0,
            "breaking_changes": 0,
            "average_message_length": 0,
            "quality_score": 0,
        }

        # Conventional commit pattern
        conventional_pattern = re.compile(
            r"^(feat|fix|docs|style|refactor|test|chore|perf|ci|build)"
            r"(\([a-zA-Z0-9_-]+\))?"
            r"(!)?:\s+.+"
        )

        total_length = 0
        quality_points = 0

        for commit in commits:
            message = commit["message"]
            total_length += len(message)

            # Check conventional commits
            if conventional_pattern.match(message):
                analysis["conventional_commits"] += 1
                quality_points += 2

                # Extract type
                match = re.match(r"^(\w+)", message)
                if match:
                    commit_type = match.group(1)
                    analysis["commit_types"][commit_type] += 1

                # Check for scope
                if "(" in message and ")" in message:
                    analysis["scoped_commits"] += 1
                    quality_points += 1

                # Check for breaking changes
                if "!" in message or "BREAKING CHANGE" in commit["body"]:
                    analysis["breaking_changes"] += 1

            # Quality checks
            if len(message) > 10:  # Not too short
                quality_points += 1
            if len(message) < 72:  # Not too long
                quality_points += 1
            if not message.endswith("."):  # No period at end (convention)
                quality_points += 1

        if commits:
            analysis["average_message_length"] = total_length / len(commits)
            analysis["quality_score"] = (quality_points / (len(commits) * 5)) * 100

        return analysis

    def get_contributor_stats(
        self, since: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        Get statistics about contributors.

        Args:
            since: Start date for analysis

        Returns:
            List of contributor statistics
        """
        commits = self.get_commits(since=since)

        # Group by author
        author_stats = defaultdict(
            lambda: {
                "commits": 0,
                "lines_added": 0,
                "lines_deleted": 0,
                "files_changed": 0,
            }
        )

        for commit in commits:
            author = commit["author"]
            author_stats[author]["commits"] += 1

            # Get stats for this commit
            _, stats, _ = self._run_command(
                ["git", "show", "--stat", "--format=", commit["hash"]], check=False
            )

            for line in stats.split("\n"):
                if "file changed" in line or "files changed" in line:
                    # Parse: X files changed, Y insertions(+), Z deletions(-)
                    match = re.search(r"(\d+) insertion", line)
                    if match:
                        author_stats[author]["lines_added"] += int(match.group(1))

                    match = re.search(r"(\d+) deletion", line)
                    if match:
                        author_stats[author]["lines_deleted"] += int(match.group(1))

                    match = re.search(r"(\d+) file", line)
                    if match:
                        author_stats[author]["files_changed"] += int(match.group(1))

        # Convert to list and sort by commits
        result = []
        for author, stats in author_stats.items():
            stats["author"] = author
            stats["total_changes"] = stats["lines_added"] + stats["lines_deleted"]
            result.append(stats)

        return sorted(result, key=lambda x: x["commits"], reverse=True)

    def analyze_commit_frequency(
        self, days: int = 30
    ) -> Dict[str, any]:
        """
        Analyze commit frequency patterns.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with frequency analysis
        """
        since = f"{days} days ago"
        commits = self.get_commits(since=since)

        frequency = {
            "total_days": days,
            "total_commits": len(commits),
            "commits_per_day": 0,
            "most_active_day": None,
            "most_active_hour": None,
            "weekday_distribution": defaultdict(int),
            "hour_distribution": defaultdict(int),
        }

        if not commits:
            return frequency

        for commit in commits:
            # Parse date
            date_str = commit["date"]
            try:
                dt = datetime.strptime(date_str.split()[0], "%Y-%m-%d")
                weekday = dt.strftime("%A")
                frequency["weekday_distribution"][weekday] += 1

                # Extract hour
                time_str = date_str.split()[1]
                hour = int(time_str.split(":")[0])
                frequency["hour_distribution"][hour] += 1
            except (ValueError, IndexError):
                continue

        frequency["commits_per_day"] = len(commits) / days

        # Find most active day and hour
        if frequency["weekday_distribution"]:
            frequency["most_active_day"] = max(
                frequency["weekday_distribution"],
                key=frequency["weekday_distribution"].get,
            )

        if frequency["hour_distribution"]:
            frequency["most_active_hour"] = max(
                frequency["hour_distribution"],
                key=frequency["hour_distribution"].get,
            )

        return frequency

    def find_large_commits(self, threshold: int = 500) -> List[Dict[str, any]]:
        """
        Find commits with large changes.

        Args:
            threshold: Minimum number of lines changed

        Returns:
            List of large commits
        """
        commits = self.get_commits(limit=100)
        large_commits = []

        for commit in commits:
            _, stats, _ = self._run_command(
                ["git", "show", "--stat", "--format=", commit["hash"]], check=False
            )

            total_changes = 0
            for line in stats.split("\n"):
                match = re.search(r"(\d+) insertion", line)
                if match:
                    total_changes += int(match.group(1))

                match = re.search(r"(\d+) deletion", line)
                if match:
                    total_changes += int(match.group(1))

            if total_changes >= threshold:
                large_commits.append(
                    {
                        "hash": commit["hash"][:8],
                        "message": commit["message"],
                        "author": commit["author"],
                        "date": commit["date"],
                        "changes": total_changes,
                    }
                )

        return sorted(large_commits, key=lambda x: x["changes"], reverse=True)

    def generate_changelog(
        self, version: str, since: Optional[str] = None
    ) -> str:
        """
        Generate a changelog from commits.

        Args:
            version: Version number for changelog
            since: Start date/tag for changelog

        Returns:
            Formatted changelog string
        """
        commits = self.get_commits(since=since)

        # Group commits by type
        grouped = {
            "Features": [],
            "Bug Fixes": [],
            "Documentation": [],
            "Refactoring": [],
            "Performance": [],
            "Tests": [],
            "Other": [],
        }

        for commit in commits:
            message = commit["message"]

            if message.startswith("feat"):
                grouped["Features"].append(message)
            elif message.startswith("fix"):
                grouped["Bug Fixes"].append(message)
            elif message.startswith("docs"):
                grouped["Documentation"].append(message)
            elif message.startswith("refactor"):
                grouped["Refactoring"].append(message)
            elif message.startswith("perf"):
                grouped["Performance"].append(message)
            elif message.startswith("test"):
                grouped["Tests"].append(message)
            else:
                grouped["Other"].append(message)

        # Build changelog
        changelog = f"# Changelog - Version {version}\n\n"
        changelog += f"**Release Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n"

        for category, items in grouped.items():
            if items:
                changelog += f"## {category}\n\n"
                for item in items:
                    # Clean up commit message
                    clean_msg = re.sub(r"^(feat|fix|docs|refactor|perf|test)\(?\w*\)?!?:\s*", "", item)
                    changelog += f"- {clean_msg}\n"
                changelog += "\n"

        return changelog

    def export_report(self, output_file: str, format: str = "json") -> None:
        """
        Export comprehensive analysis report.

        Args:
            output_file: Output file path
            format: Output format (json, markdown, html)
        """
        # Gather all analyses
        commits = self.get_commits(limit=100)
        message_analysis = self.analyze_commit_messages(commits)
        contributors = self.get_contributor_stats(since="30 days ago")
        frequency = self.analyze_commit_frequency(30)
        large_commits = self.find_large_commits()

        report = {
            "generated_at": datetime.now().isoformat(),
            "repository": str(self.repo_path),
            "commit_analysis": message_analysis,
            "contributors": contributors,
            "frequency": frequency,
            "large_commits": large_commits[:10],
        }

        if format == "json":
            with open(output_file, "w") as f:
                json.dump(report, f, indent=2, default=str)

        elif format == "markdown":
            md = "# Git Repository Analysis Report\n\n"
            md += f"**Generated:** {report['generated_at']}\n\n"

            md += "## Commit Message Quality\n\n"
            md += f"- Total Commits: {message_analysis['total_commits']}\n"
            md += f"- Conventional Commits: {message_analysis['conventional_commits']}\n"
            md += f"- Quality Score: {message_analysis['quality_score']:.1f}%\n\n"

            md += "## Top Contributors (Last 30 Days)\n\n"
            for contrib in contributors[:5]:
                md += f"- **{contrib['author']}**: {contrib['commits']} commits, "
                md += f"+{contrib['lines_added']}/-{contrib['lines_deleted']} lines\n"

            md += f"\n## Commit Frequency\n\n"
            md += f"- Average commits/day: {frequency['commits_per_day']:.2f}\n"
            md += f"- Most active day: {frequency['most_active_day']}\n"
            md += f"- Most active hour: {frequency['most_active_hour']}:00\n"

            with open(output_file, "w") as f:
                f.write(md)

        print(f"Report exported to: {output_file}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Git Commit Analyzer and Reporter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--repo", help="Path to Git repository (default: current dir)")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Analyze messages
    msg_parser = subparsers.add_parser(
        "messages", help="Analyze commit message quality"
    )
    msg_parser.add_argument("--since", help="Start date (e.g., '2024-01-01')")
    msg_parser.add_argument("--limit", type=int, help="Limit number of commits")

    # Contributor stats
    contrib_parser = subparsers.add_parser("contributors", help="Get contributor stats")
    contrib_parser.add_argument(
        "--since", default="30 days ago", help="Start date (default: 30 days ago)"
    )

    # Frequency analysis
    freq_parser = subparsers.add_parser("frequency", help="Analyze commit frequency")
    freq_parser.add_argument(
        "--days", type=int, default=30, help="Number of days (default: 30)"
    )

    # Large commits
    large_parser = subparsers.add_parser("large", help="Find large commits")
    large_parser.add_argument(
        "--threshold", type=int, default=500, help="Minimum lines changed (default: 500)"
    )

    # Generate changelog
    changelog_parser = subparsers.add_parser("changelog", help="Generate changelog")
    changelog_parser.add_argument("version", help="Version number")
    changelog_parser.add_argument("--since", help="Start date/tag")
    changelog_parser.add_argument(
        "--output", default="CHANGELOG.md", help="Output file (default: CHANGELOG.md)"
    )

    # Export report
    report_parser = subparsers.add_parser("report", help="Export analysis report")
    report_parser.add_argument("output", help="Output file path")
    report_parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format (default: json)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        analyzer = CommitAnalyzer(args.repo)

        if args.command == "messages":
            commits = analyzer.get_commits(
                since=args.since, limit=args.limit
            )
            analysis = analyzer.analyze_commit_messages(commits)

            print("\nCommit Message Analysis:")
            print(f"  Total commits: {analysis['total_commits']}")
            print(f"  Conventional commits: {analysis['conventional_commits']}")
            print(f"  Scoped commits: {analysis['scoped_commits']}")
            print(f"  Breaking changes: {analysis['breaking_changes']}")
            print(f"  Avg message length: {analysis['average_message_length']:.1f}")
            print(f"  Quality score: {analysis['quality_score']:.1f}%")

            if analysis["commit_types"]:
                print("\n  Commit types:")
                for ctype, count in analysis["commit_types"].most_common():
                    print(f"    - {ctype}: {count}")

        elif args.command == "contributors":
            contributors = analyzer.get_contributor_stats(args.since)

            print(f"\nContributor Statistics (since {args.since}):")
            for contrib in contributors:
                print(f"\n  {contrib['author']}:")
                print(f"    Commits: {contrib['commits']}")
                print(f"    Lines added: {contrib['lines_added']}")
                print(f"    Lines deleted: {contrib['lines_deleted']}")
                print(f"    Files changed: {contrib['files_changed']}")

        elif args.command == "frequency":
            frequency = analyzer.analyze_commit_frequency(args.days)

            print(f"\nCommit Frequency Analysis (last {args.days} days):")
            print(f"  Total commits: {frequency['total_commits']}")
            print(f"  Commits per day: {frequency['commits_per_day']:.2f}")
            print(f"  Most active day: {frequency['most_active_day']}")
            print(f"  Most active hour: {frequency['most_active_hour']}:00")

            if frequency["weekday_distribution"]:
                print("\n  Weekday distribution:")
                for day, count in sorted(
                    frequency["weekday_distribution"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                ):
                    print(f"    {day}: {count}")

        elif args.command == "large":
            large = analyzer.find_large_commits(args.threshold)

            print(f"\nLarge Commits (>{args.threshold} lines changed):")
            for commit in large:
                print(f"\n  {commit['hash']} - {commit['changes']} lines")
                print(f"    {commit['message']}")
                print(f"    by {commit['author']} on {commit['date']}")

        elif args.command == "changelog":
            changelog = analyzer.generate_changelog(args.version, args.since)
            with open(args.output, "w") as f:
                f.write(changelog)
            print(f"Changelog written to: {args.output}")

        elif args.command == "report":
            analyzer.export_report(args.output, args.format)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
