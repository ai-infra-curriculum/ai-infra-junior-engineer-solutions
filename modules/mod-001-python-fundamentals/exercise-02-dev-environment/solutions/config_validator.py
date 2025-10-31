#!/usr/bin/env python3
"""
Configuration Validator for AI Infrastructure Development

This tool validates development environment configurations including:
- Git configuration and credentials
- Docker configuration and resource limits
- Kubernetes config and contexts
- Cloud provider configurations (AWS, GCP, Azure)
- Python virtual environments
- IDE/Editor settings

Usage:
    python config_validator.py                  # Validate all configs
    python config_validator.py --git            # Check Git config only
    python config_validator.py --docker         # Check Docker config only
    python config_validator.py --export report  # Export validation report

Author: AI Infrastructure Curriculum Team
License: MIT
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
import configparser


class ValidationStatus(Enum):
    """Status of configuration validation."""
    VALID = "✅ VALID"
    INVALID = "❌ INVALID"
    WARNING = "⚠️  WARNING"
    NOT_FOUND = "📁 NOT FOUND"


class Colors:
    """ANSI color codes."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


@dataclass
class ValidationResult:
    """Result of a configuration validation."""
    component: str
    status: ValidationStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['status'] = self.status.value
        return result


class ConfigValidator:
    """Main configuration validator."""

    def __init__(self, verbose: bool = False):
        """
        Initialize validator.

        Args:
            verbose: Enable detailed output
        """
        self.verbose = verbose
        self.results: List[ValidationResult] = []
        self.home = Path.home()

    def _run_command(self, command: str) -> Tuple[bool, str, str]:
        """Execute shell command and return result."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            return (
                result.returncode == 0,
                result.stdout.strip(),
                result.stderr.strip()
            )
        except Exception as e:
            return False, "", str(e)

    def _read_file(self, path: Path) -> Optional[str]:
        """Safely read file contents."""
        try:
            return path.read_text()
        except Exception:
            return None

    def _parse_yaml(self, path: Path) -> Optional[Dict]:
        """Parse YAML file."""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception:
            return None

    def _parse_json(self, path: Path) -> Optional[Dict]:
        """Parse JSON file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def validate_git_config(self) -> ValidationResult:
        """Validate Git configuration."""
        gitconfig_path = self.home / '.gitconfig'

        if not gitconfig_path.exists():
            return ValidationResult(
                component="Git Configuration",
                status=ValidationStatus.NOT_FOUND,
                message="~/.gitconfig not found",
                recommendations=[
                    "Run: git config --global user.name 'Your Name'",
                    "Run: git config --global user.email 'your@email.com'"
                ]
            )

        config = configparser.ConfigParser()
        try:
            config.read(gitconfig_path)
        except Exception as e:
            return ValidationResult(
                component="Git Configuration",
                status=ValidationStatus.INVALID,
                message=f"Failed to parse .gitconfig: {e}"
            )

        issues = []
        warnings = []
        details = {}

        # Check required fields
        if 'user' in config:
            if 'name' in config['user']:
                details['user.name'] = config['user']['name']
            else:
                issues.append("user.name not set")

            if 'email' in config['user']:
                details['user.email'] = config['user']['email']
            else:
                issues.append("user.email not set")
        else:
            issues.append("user section missing")

        # Check recommended settings
        if 'core' in config:
            if 'editor' in config['core']:
                details['core.editor'] = config['core']['editor']
            else:
                warnings.append("core.editor not set (recommended)")

            if 'autocrlf' not in config['core']:
                warnings.append("core.autocrlf not set (recommended for cross-platform)")

        if 'credential' not in config:
            warnings.append("credential helper not configured")

        # Check for SSH keys
        ssh_dir = self.home / '.ssh'
        ssh_keys = []
        if ssh_dir.exists():
            for key_file in ['id_rsa', 'id_ed25519', 'id_ecdsa']:
                if (ssh_dir / key_file).exists():
                    ssh_keys.append(key_file)

        details['ssh_keys'] = ssh_keys if ssh_keys else None

        if not ssh_keys:
            warnings.append("No SSH keys found in ~/.ssh")

        recommendations = []
        if issues:
            recommendations.extend([f"Fix: {issue}" for issue in issues])
        if warnings:
            recommendations.extend([f"Consider: {warning}" for warning in warnings])

        if issues:
            status = ValidationStatus.INVALID
            message = f"Git config has {len(issues)} error(s)"
        elif warnings:
            status = ValidationStatus.WARNING
            message = f"Git config valid with {len(warnings)} recommendation(s)"
        else:
            status = ValidationStatus.VALID
            message = "Git configuration is properly set up"

        return ValidationResult(
            component="Git Configuration",
            status=status,
            message=message,
            details=details,
            recommendations=recommendations if recommendations else None
        )

    def validate_docker_config(self) -> ValidationResult:
        """Validate Docker configuration."""
        # Check if Docker is running
        success, stdout, stderr = self._run_command("docker info --format '{{json .}}'")

        if not success:
            return ValidationResult(
                component="Docker Configuration",
                status=ValidationStatus.INVALID,
                message="Docker daemon not accessible",
                recommendations=["Start Docker Desktop or docker service"]
            )

        try:
            docker_info = json.loads(stdout)
        except json.JSONDecodeError:
            return ValidationResult(
                component="Docker Configuration",
                status=ValidationStatus.INVALID,
                message="Failed to parse docker info"
            )

        details = {
            "server_version": docker_info.get("ServerVersion"),
            "operating_system": docker_info.get("OperatingSystem"),
            "architecture": docker_info.get("Architecture"),
            "cpus": docker_info.get("NCPU"),
            "memory_gb": round(docker_info.get("MemTotal", 0) / (1024**3), 2),
            "driver": docker_info.get("Driver")
        }

        warnings = []
        recommendations = []

        # Check resource limits
        memory_gb = details["memory_gb"]
        if memory_gb < 4:
            warnings.append(f"Low memory allocation ({memory_gb}GB)")
            recommendations.append("Increase Docker memory to at least 4GB for ML workloads")

        cpus = details["cpus"]
        if cpus < 2:
            warnings.append(f"Low CPU allocation ({cpus} CPUs)")
            recommendations.append("Allocate at least 2 CPUs to Docker")

        # Check for daemon.json config
        if sys.platform == "darwin":
            daemon_config_path = self.home / "Library/Group Containers/group.com.docker/settings.json"
        else:
            daemon_config_path = Path("/etc/docker/daemon.json")

        if daemon_config_path.exists():
            daemon_config = self._parse_json(daemon_config_path)
            if daemon_config:
                details['daemon_config'] = daemon_config

        if warnings:
            status = ValidationStatus.WARNING
            message = f"Docker configured with {len(warnings)} warning(s)"
        else:
            status = ValidationStatus.VALID
            message = "Docker is properly configured"

        return ValidationResult(
            component="Docker Configuration",
            status=status,
            message=message,
            details=details,
            recommendations=recommendations if recommendations else None
        )

    def validate_kubernetes_config(self) -> ValidationResult:
        """Validate Kubernetes configuration."""
        kubeconfig_path = self.home / '.kube' / 'config'

        if not kubeconfig_path.exists():
            return ValidationResult(
                component="Kubernetes Configuration",
                status=ValidationStatus.NOT_FOUND,
                message="~/.kube/config not found",
                recommendations=[
                    "Install kubectl and configure cluster access",
                    "Or use: kubectl config view"
                ]
            )

        kubeconfig = self._parse_yaml(kubeconfig_path)
        if not kubeconfig:
            return ValidationResult(
                component="Kubernetes Configuration",
                status=ValidationStatus.INVALID,
                message="Failed to parse kubeconfig"
            )

        details = {}
        warnings = []
        recommendations = []

        # Count contexts and clusters
        contexts = kubeconfig.get('contexts', [])
        clusters = kubeconfig.get('clusters', [])
        users = kubeconfig.get('users', [])

        details['contexts_count'] = len(contexts)
        details['clusters_count'] = len(clusters)
        details['users_count'] = len(users)

        # Get current context
        current_context = kubeconfig.get('current-context')
        details['current_context'] = current_context

        if not current_context:
            warnings.append("No current context set")
            recommendations.append("Set context with: kubectl config use-context <context-name>")

        # List context names
        context_names = [ctx['name'] for ctx in contexts]
        details['contexts'] = context_names

        # Check connectivity to current cluster
        if current_context:
            success, stdout, stderr = self._run_command("kubectl cluster-info")
            if success:
                details['cluster_accessible'] = True
            else:
                warnings.append("Cannot access current cluster")
                details['cluster_accessible'] = False
                recommendations.append("Verify cluster credentials and network connectivity")

        if warnings:
            status = ValidationStatus.WARNING
            message = f"Kubernetes config has {len(warnings)} warning(s)"
        else:
            status = ValidationStatus.VALID
            message = "Kubernetes configuration is valid"

        return ValidationResult(
            component="Kubernetes Configuration",
            status=status,
            message=message,
            details=details,
            recommendations=recommendations if recommendations else None
        )

    def validate_aws_config(self) -> ValidationResult:
        """Validate AWS CLI configuration."""
        aws_dir = self.home / '.aws'
        config_path = aws_dir / 'config'
        credentials_path = aws_dir / 'credentials'

        if not aws_dir.exists():
            return ValidationResult(
                component="AWS Configuration",
                status=ValidationStatus.NOT_FOUND,
                message="~/.aws directory not found",
                recommendations=["Run: aws configure"]
            )

        details = {}
        warnings = []
        recommendations = []

        # Check config file
        if config_path.exists():
            config = configparser.ConfigParser()
            try:
                config.read(config_path)
                profiles = [s.replace('profile ', '') for s in config.sections()]
                details['config_profiles'] = profiles
            except Exception as e:
                warnings.append(f"Failed to parse config: {e}")
        else:
            warnings.append("config file not found")

        # Check credentials file (don't read contents for security)
        if credentials_path.exists():
            creds = configparser.ConfigParser()
            try:
                creds.read(credentials_path)
                cred_profiles = creds.sections()
                details['credential_profiles'] = cred_profiles
                details['has_credentials'] = True

                # Verify credentials work
                success, stdout, stderr = self._run_command("aws sts get-caller-identity")
                if success:
                    try:
                        identity = json.loads(stdout)
                        details['account_id'] = identity.get('Account')
                        details['user_arn'] = identity.get('Arn')
                    except:
                        pass
                else:
                    warnings.append("AWS credentials not working")
                    recommendations.append("Run: aws configure")
            except Exception as e:
                warnings.append(f"Failed to parse credentials: {e}")
        else:
            warnings.append("credentials file not found")
            recommendations.append("Run: aws configure")

        if warnings:
            status = ValidationStatus.WARNING
            message = f"AWS config has {len(warnings)} warning(s)"
        else:
            status = ValidationStatus.VALID
            message = "AWS configuration is valid"

        return ValidationResult(
            component="AWS Configuration",
            status=status,
            message=message,
            details=details,
            recommendations=recommendations if recommendations else None
        )

    def validate_gcp_config(self) -> ValidationResult:
        """Validate Google Cloud SDK configuration."""
        gcloud_dir = self.home / '.config' / 'gcloud'

        if not gcloud_dir.exists():
            return ValidationResult(
                component="GCP Configuration",
                status=ValidationStatus.NOT_FOUND,
                message="gcloud not configured",
                recommendations=["Run: gcloud init"]
            )

        details = {}
        warnings = []
        recommendations = []

        # Get active configuration
        success, stdout, stderr = self._run_command("gcloud config list --format=json")
        if success:
            try:
                config = json.loads(stdout)
                details['active_config'] = config.get('core', {})

                project = config.get('core', {}).get('project')
                account = config.get('core', {}).get('account')

                if project:
                    details['project'] = project
                else:
                    warnings.append("No default project set")
                    recommendations.append("Run: gcloud config set project PROJECT_ID")

                if account:
                    details['account'] = account
                else:
                    warnings.append("No account configured")
                    recommendations.append("Run: gcloud auth login")

            except json.JSONDecodeError:
                warnings.append("Failed to parse gcloud config")

        # List configurations
        success, stdout, stderr = self._run_command("gcloud config configurations list --format=json")
        if success:
            try:
                configs = json.loads(stdout)
                details['configurations'] = [c.get('name') for c in configs]
            except:
                pass

        if warnings:
            status = ValidationStatus.WARNING
            message = f"GCP config has {len(warnings)} warning(s)"
        else:
            status = ValidationStatus.VALID
            message = "GCP configuration is valid"

        return ValidationResult(
            component="GCP Configuration",
            status=status,
            message=message,
            details=details,
            recommendations=recommendations if recommendations else None
        )

    def validate_python_venv(self) -> ValidationResult:
        """Check if running in virtual environment."""
        in_venv = sys.prefix != sys.base_prefix
        venv_path = os.environ.get('VIRTUAL_ENV')

        details = {
            'in_virtualenv': in_venv,
            'venv_path': venv_path,
            'sys_prefix': sys.prefix,
            'python_version': sys.version.split()[0]
        }

        if in_venv:
            status = ValidationStatus.VALID
            message = f"Running in virtual environment: {venv_path}"
            recommendations = None
        else:
            status = ValidationStatus.WARNING
            message = "Not running in a virtual environment"
            recommendations = [
                "Create venv: python -m venv venv",
                "Activate: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
            ]

        return ValidationResult(
            component="Python Virtual Environment",
            status=status,
            message=message,
            details=details,
            recommendations=recommendations
        )

    def validate_vscode_settings(self) -> ValidationResult:
        """Validate VS Code settings."""
        if sys.platform == "darwin":
            vscode_dir = self.home / 'Library' / 'Application Support' / 'Code'
        elif sys.platform == "win32":
            vscode_dir = self.home / 'AppData' / 'Roaming' / 'Code'
        else:
            vscode_dir = self.home / '.config' / 'Code'

        settings_path = vscode_dir / 'User' / 'settings.json'

        if not settings_path.exists():
            return ValidationResult(
                component="VS Code Settings",
                status=ValidationStatus.NOT_FOUND,
                message="VS Code settings not found",
                recommendations=["Open VS Code and configure settings"]
            )

        settings = self._parse_json(settings_path)
        if not settings:
            return ValidationResult(
                component="VS Code Settings",
                status=ValidationStatus.INVALID,
                message="Failed to parse VS Code settings"
            )

        details = {}
        recommendations = []

        # Check for recommended extensions
        recommended_settings = {
            'python.linting.enabled': True,
            'python.linting.pylintEnabled': False,
            'python.linting.flake8Enabled': True,
            'python.formatting.provider': 'black',
            'editor.formatOnSave': True,
            'files.autoSave': 'afterDelay'
        }

        for key, recommended_value in recommended_settings.items():
            actual_value = settings.get(key)
            details[key] = actual_value

            if actual_value != recommended_value:
                recommendations.append(f"Consider setting '{key}': {recommended_value}")

        if recommendations:
            status = ValidationStatus.WARNING
            message = f"VS Code has {len(recommendations)} recommended settings"
        else:
            status = ValidationStatus.VALID
            message = "VS Code settings are well configured"

        return ValidationResult(
            component="VS Code Settings",
            status=status,
            message=message,
            details=details,
            recommendations=recommendations if recommendations else None
        )

    def validate_ssh_config(self) -> ValidationResult:
        """Validate SSH configuration."""
        ssh_dir = self.home / '.ssh'
        config_path = ssh_dir / 'config'

        if not ssh_dir.exists():
            return ValidationResult(
                component="SSH Configuration",
                status=ValidationStatus.NOT_FOUND,
                message="~/.ssh directory not found",
                recommendations=["Create: mkdir -p ~/.ssh && chmod 700 ~/.ssh"]
            )

        details = {}
        warnings = []
        recommendations = []

        # Check directory permissions
        if oct(ssh_dir.stat().st_mode)[-3:] != '700':
            warnings.append("~/.ssh directory has incorrect permissions")
            recommendations.append("Fix: chmod 700 ~/.ssh")

        # Check for keys
        key_types = ['id_rsa', 'id_ed25519', 'id_ecdsa']
        found_keys = []

        for key_name in key_types:
            private_key = ssh_dir / key_name
            public_key = ssh_dir / f"{key_name}.pub"

            if private_key.exists():
                found_keys.append(key_name)

                # Check private key permissions
                if oct(private_key.stat().st_mode)[-3:] != '600':
                    warnings.append(f"{key_name} has incorrect permissions")
                    recommendations.append(f"Fix: chmod 600 ~/.ssh/{key_name}")

                # Check if public key exists
                if not public_key.exists():
                    warnings.append(f"Public key for {key_name} not found")

        details['keys_found'] = found_keys

        if not found_keys:
            warnings.append("No SSH keys found")
            recommendations.append("Generate key: ssh-keygen -t ed25519 -C 'your@email.com'")

        # Check SSH config file
        if config_path.exists():
            details['has_config'] = True
        else:
            details['has_config'] = False

        # Check known_hosts
        known_hosts = ssh_dir / 'known_hosts'
        if known_hosts.exists():
            details['has_known_hosts'] = True
        else:
            details['has_known_hosts'] = False

        if warnings:
            status = ValidationStatus.WARNING
            message = f"SSH config has {len(warnings)} warning(s)"
        else:
            status = ValidationStatus.VALID
            message = "SSH configuration is valid"

        return ValidationResult(
            component="SSH Configuration",
            status=status,
            message=message,
            details=details,
            recommendations=recommendations if recommendations else None
        )

    def run_all_validations(self) -> List[ValidationResult]:
        """Run all configuration validations."""
        validators = [
            self.validate_git_config,
            self.validate_docker_config,
            self.validate_kubernetes_config,
            self.validate_aws_config,
            self.validate_gcp_config,
            self.validate_python_venv,
            self.validate_vscode_settings,
            self.validate_ssh_config
        ]

        self.results = []
        for validator in validators:
            try:
                result = validator()
                self.results.append(result)
            except Exception as e:
                self.results.append(ValidationResult(
                    component=validator.__name__.replace('validate_', '').replace('_', ' ').title(),
                    status=ValidationStatus.INVALID,
                    message=f"Validation failed: {e}"
                ))

        return self.results

    def print_results(self):
        """Print validation results."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}  Configuration Validation Report{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")

        # Count statuses
        valid = sum(1 for r in self.results if r.status == ValidationStatus.VALID)
        invalid = sum(1 for r in self.results if r.status == ValidationStatus.INVALID)
        warning = sum(1 for r in self.results if r.status == ValidationStatus.WARNING)
        not_found = sum(1 for r in self.results if r.status == ValidationStatus.NOT_FOUND)

        for result in self.results:
            color = {
                ValidationStatus.VALID: Colors.GREEN,
                ValidationStatus.INVALID: Colors.RED,
                ValidationStatus.WARNING: Colors.YELLOW,
                ValidationStatus.NOT_FOUND: Colors.BLUE
            }.get(result.status, Colors.END)

            print(f"{color}{result.status.value}{Colors.END} {Colors.BOLD}{result.component}{Colors.END}")
            print(f"    {result.message}")

            if self.verbose and result.details:
                print(f"    {Colors.CYAN}Details:{Colors.END}")
                for key, value in result.details.items():
                    print(f"      {key}: {value}")

            if result.recommendations:
                print(f"    {Colors.YELLOW}Recommendations:{Colors.END}")
                for rec in result.recommendations:
                    print(f"      • {rec}")

            print()

        # Summary
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}Summary:{Colors.END}")
        print(f"  {Colors.GREEN}✅ Valid: {valid}{Colors.END}")
        print(f"  {Colors.YELLOW}⚠️  Warnings: {warning}{Colors.END}")
        print(f"  {Colors.RED}❌ Invalid: {invalid}{Colors.END}")
        print(f"  {Colors.BLUE}📁 Not Found: {not_found}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")

        return 0 if invalid == 0 else 1

    def export_report(self, format: str = 'json') -> str:
        """Export validation report."""
        report = {
            'results': [r.to_dict() for r in self.results],
            'summary': {
                'total': len(self.results),
                'valid': sum(1 for r in self.results if r.status == ValidationStatus.VALID),
                'invalid': sum(1 for r in self.results if r.status == ValidationStatus.INVALID),
                'warnings': sum(1 for r in self.results if r.status == ValidationStatus.WARNING),
                'not_found': sum(1 for r in self.results if r.status == ValidationStatus.NOT_FOUND)
            }
        }

        if format == 'json':
            return json.dumps(report, indent=2)
        elif format == 'yaml':
            return yaml.dump(report, default_flow_style=False)
        else:
            return str(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate development environment configurations"
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--git',
        action='store_true',
        help='Validate Git configuration only'
    )
    parser.add_argument(
        '--docker',
        action='store_true',
        help='Validate Docker configuration only'
    )
    parser.add_argument(
        '--k8s',
        action='store_true',
        help='Validate Kubernetes configuration only'
    )
    parser.add_argument(
        '--export',
        choices=['json', 'yaml'],
        help='Export report in specified format'
    )

    args = parser.parse_args()

    validator = ConfigValidator(verbose=args.verbose)

    # Run specific validation or all
    if args.git:
        result = validator.validate_git_config()
        validator.results = [result]
    elif args.docker:
        result = validator.validate_docker_config()
        validator.results = [result]
    elif args.k8s:
        result = validator.validate_kubernetes_config()
        validator.results = [result]
    else:
        validator.run_all_validations()

    if args.export:
        print(validator.export_report(args.export))
        return 0
    else:
        return validator.print_results()


if __name__ == "__main__":
    sys.exit(main())
