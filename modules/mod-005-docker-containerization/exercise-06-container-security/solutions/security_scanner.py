#!/usr/bin/env python3
"""
Docker container security scanner and auditor.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SecurityIssue:
    """Security issue found in container."""
    severity: str
    category: str
    description: str
    recommendation: str


class SecurityScanner:
    """Scan containers and images for security issues."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _run_command(self, cmd: List[str]) -> tuple:
        """Run command and return result."""
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False, timeout=120
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            return 1, "", str(e)

    def scan_dockerfile(self, dockerfile_path: str) -> List[SecurityIssue]:
        """Scan Dockerfile for security issues."""
        issues = []
        dockerfile = Path(dockerfile_path)

        if not dockerfile.exists():
            print(f"✗ Dockerfile not found: {dockerfile_path}")
            return issues

        content = dockerfile.read_text()
        lines = content.split('\n')

        # Check for root user
        has_user = any("USER " in line and "USER root" not in line for line in lines)
        if not has_user:
            issues.append(SecurityIssue(
                severity="HIGH",
                category="User",
                description="Container runs as root user",
                recommendation="Add USER instruction with non-root user"
            ))

        # Check for latest tag
        if "FROM" in content and ":latest" in content:
            issues.append(SecurityIssue(
                severity="MEDIUM",
                category="Image",
                description="Using :latest tag for base image",
                recommendation="Pin specific version tags"
            ))

        # Check for HEALTHCHECK
        if "HEALTHCHECK" not in content:
            issues.append(SecurityIssue(
                severity="LOW",
                category="Monitoring",
                description="No HEALTHCHECK instruction",
                recommendation="Add HEALTHCHECK for container monitoring"
            ))

        # Check for secrets in ENV
        for line in lines:
            if "ENV" in line and any(word in line.upper() for word in ["PASSWORD", "SECRET", "KEY", "TOKEN"]):
                if "=" in line and len(line.split("=")[1].strip()) > 0:
                    issues.append(SecurityIssue(
                        severity="CRITICAL",
                        category="Secrets",
                        description="Possible secret in ENV instruction",
                        recommendation="Use secrets management, not ENV variables"
                    ))

        # Check for apt-get update without cleanup
        for i, line in enumerate(lines):
            if "apt-get update" in line:
                # Look for cleanup in next few lines
                has_cleanup = any(
                    "rm -rf /var/lib/apt/lists" in lines[j]
                    for j in range(i, min(i+5, len(lines)))
                )
                if not has_cleanup:
                    issues.append(SecurityIssue(
                        severity="LOW",
                        category="Size",
                        description="apt-get update without cleanup",
                        recommendation="Add 'rm -rf /var/lib/apt/lists/*'"
                    ))

        return issues

    def scan_image_trivy(self, image_name: str) -> bool:
        """Scan image with Trivy for vulnerabilities."""
        print(f"Scanning {image_name} with Trivy...")

        cmd = [
            "trivy", "image",
            "--severity", "HIGH,CRITICAL",
            "--format", "json",
            image_name
        ]

        returncode, stdout, stderr = self._run_command(cmd)

        if returncode == 0:
            try:
                results = json.loads(stdout)
                total_vulns = 0

                for result in results.get("Results", []):
                    vulns = result.get("Vulnerabilities", [])
                    total_vulns += len(vulns)

                    if self.verbose:
                        for vuln in vulns[:5]:  # Show first 5
                            print(f"  {vuln.get('Severity')}: {vuln.get('VulnerabilityID')}")

                print(f"✓ Found {total_vulns} HIGH/CRITICAL vulnerabilities")
                return True

            except json.JSONDecodeError:
                print("✗ Failed to parse Trivy results")
                return False
        else:
            print(f"✗ Trivy scan failed (may not be installed)")
            if self.verbose and stderr:
                print(f"Error: {stderr}")
            return False

    def check_container_security(self, container_name: str) -> List[SecurityIssue]:
        """Check running container for security issues."""
        issues = []

        # Check if running as root
        cmd = ["docker", "exec", container_name, "whoami"]
        returncode, stdout, _ = self._run_command(cmd)

        if returncode == 0 and stdout.strip() == "root":
            issues.append(SecurityIssue(
                severity="HIGH",
                category="User",
                description="Container running as root",
                recommendation="Run as non-root user"
            ))

        # Check for privileged mode
        cmd = ["docker", "inspect", container_name, "--format={{.HostConfig.Privileged}}"]
        returncode, stdout, _ = self._run_command(cmd)

        if returncode == 0 and stdout.strip() == "true":
            issues.append(SecurityIssue(
                severity="CRITICAL",
                category="Privileges",
                description="Container running in privileged mode",
                recommendation="Remove --privileged flag unless absolutely necessary"
            ))

        # Check capabilities
        cmd = ["docker", "inspect", container_name, "--format={{.HostConfig.CapAdd}}"]
        returncode, stdout, _ = self._run_command(cmd)

        if returncode == 0 and "SYS_ADMIN" in stdout:
            issues.append(SecurityIssue(
                severity="HIGH",
                category="Capabilities",
                description="Container has SYS_ADMIN capability",
                recommendation="Remove unnecessary capabilities"
            ))

        return issues

    def audit_dockerfile_best_practices(self, dockerfile_path: str) -> Dict[str, bool]:
        """Audit Dockerfile against best practices."""
        dockerfile = Path(dockerfile_path)
        if not dockerfile.exists():
            return {}

        content = dockerfile.read_text()

        checks = {
            "uses_specific_base_tag": ":latest" not in content,
            "has_user_instruction": "USER " in content,
            "has_healthcheck": "HEALTHCHECK" in content,
            "uses_copy_not_add": content.count("COPY") > content.count("ADD"),
            "has_labels": "LABEL" in content,
            "combines_run_commands": content.count("RUN") < 10,
            "cleans_apt_cache": "rm -rf /var/lib/apt/lists" in content or "apt" not in content,
            "sets_workdir": "WORKDIR" in content,
            "no_secrets_in_env": not any(
                word in content.upper() for word in ["PASSWORD=", "SECRET=", "API_KEY="]
            ),
            "uses_exec_form_cmd": 'CMD ["' in content or 'ENTRYPOINT ["' in content
        }

        return checks

    def generate_security_report(
        self,
        dockerfile_path: Optional[str] = None,
        image_name: Optional[str] = None,
        container_name: Optional[str] = None
    ) -> Dict:
        """Generate comprehensive security report."""
        report = {
            "timestamp": subprocess.run(
                ["date", "-Iseconds"],
                capture_output=True,
                text=True
            ).stdout.strip(),
            "issues": []
        }

        # Scan Dockerfile
        if dockerfile_path:
            print(f"\nScanning Dockerfile: {dockerfile_path}")
            issues = self.scan_dockerfile(dockerfile_path)
            report["dockerfile_issues"] = [i.__dict__ for i in issues]
            print(f"Found {len(issues)} Dockerfile issues")

            # Best practices
            practices = self.audit_dockerfile_best_practices(dockerfile_path)
            report["best_practices"] = practices
            passed = sum(1 for v in practices.values() if v)
            print(f"Best practices: {passed}/{len(practices)} passed")

        # Scan image
        if image_name:
            print(f"\nScanning image: {image_name}")
            self.scan_image_trivy(image_name)

        # Check container
        if container_name:
            print(f"\nChecking container: {container_name}")
            issues = self.check_container_security(container_name)
            report["container_issues"] = [i.__dict__ for i in issues]
            print(f"Found {len(issues)} container issues")

        return report

    def print_report(self, report: Dict) -> None:
        """Print security report."""
        print("\n" + "="*60)
        print("SECURITY REPORT")
        print("="*60)

        if "dockerfile_issues" in report:
            print("\nDockerfile Issues:")
            for issue in report["dockerfile_issues"]:
                print(f"  [{issue['severity']}] {issue['description']}")
                print(f"    → {issue['recommendation']}")

        if "best_practices" in report:
            print("\nBest Practices:")
            for check, passed in report["best_practices"].items():
                status = "✓" if passed else "✗"
                print(f"  {status} {check.replace('_', ' ').title()}")

        if "container_issues" in report:
            print("\nContainer Issues:")
            for issue in report["container_issues"]:
                print(f"  [{issue['severity']}] {issue['description']}")
                print(f"    → {issue['recommendation']}")

        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Docker security scanner")
    parser.add_argument("--dockerfile", help="Dockerfile path to scan")
    parser.add_argument("--image", help="Image name to scan")
    parser.add_argument("--container", help="Container name to check")
    parser.add_argument("--report", help="Output report to JSON file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if not (args.dockerfile or args.image or args.container):
        print("Error: Provide at least one of --dockerfile, --image, or --container")
        sys.exit(1)

    scanner = SecurityScanner(verbose=args.verbose)

    report = scanner.generate_security_report(
        dockerfile_path=args.dockerfile,
        image_name=args.image,
        container_name=args.container
    )

    scanner.print_report(report)

    if args.report:
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Report saved to {args.report}")

    # Exit with error if critical issues found
    critical_count = sum(
        1 for issue in report.get("dockerfile_issues", []) + report.get("container_issues", [])
        if issue.get("severity") == "CRITICAL"
    )

    sys.exit(1 if critical_count > 0 else 0)


if __name__ == "__main__":
    main()
