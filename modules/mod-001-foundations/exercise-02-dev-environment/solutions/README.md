# Development Environment Setup - Solutions

Complete implementation of development environment validation and setup tools for AI Infrastructure engineers.

## 📋 Overview

This solution provides comprehensive tools to:
- ✅ Validate your development environment
- 🔧 Install required dependencies automatically
- ⚙️ Verify configuration files
- 📊 Generate validation reports

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Check Your Environment

```bash
python check_environment.py
```

Expected output:
```
================================================================
  AI Infrastructure Development Environment Check
================================================================

System Information:
  os: Darwin (or Linux)
  python_version: 3.11.0

✅ PASS Python (3.11.0)
    Python 3.11.0 is installed (>= 3.11 required)

✅ PASS pip (23.0.1)
    pip is installed
...
```

### 3. Validate Configurations

```bash
python config_validator.py
```

### 4. Install Missing Tools (Optional)

```bash
# On macOS/Linux
chmod +x install_dependencies.sh
./install_dependencies.sh

# Dry run to see what would be installed
./install_dependencies.sh --check

# Install only essentials
./install_dependencies.sh --minimal

# Install everything
./install_dependencies.sh --full
```

## 📁 Files Included

### Core Tools

1. **check_environment.py** (620 lines)
   - Validates 14 different tools and configurations
   - Checks: Python, pip, Git, Docker, kubectl, Terraform, cloud CLIs, ML libraries
   - System checks: disk space, memory, CPU
   - Output formats: human-readable, JSON
   - Provides fix commands for issues

2. **config_validator.py** (790 lines)
   - Validates configuration files
   - Checks: Git config, Docker config, Kubernetes, AWS, GCP, SSH
   - Validates VS Code settings
   - Checks virtual environment usage
   - Exports reports in JSON/YAML

3. **install_dependencies.sh** (730 lines)
   - Cross-platform installer (macOS, Ubuntu, RHEL)
   - Installs: Python, Docker, kubectl, Terraform, cloud tools
   - Supports dry-run mode
   - Minimal/full installation modes
   - Automatic package manager detection

4. **requirements.txt**
   - Python dependencies for the tools
   - Optional dependencies clearly marked

### Tests

5. **tests/test_check_environment.py** (280+ lines)
   - Comprehensive test suite
   - Unit tests for all checker methods
   - Mock-based testing for external commands
   - Integration tests
   - >80% code coverage

## 🔧 Usage Examples

### Environment Checker

```bash
# Basic check
python check_environment.py

# Verbose output with details
python check_environment.py --verbose

# JSON output for automation
python check_environment.py --json > env-report.json

# Check and show fix commands
python check_environment.py -v
```

### Configuration Validator

```bash
# Validate all configurations
python config_validator.py

# Validate specific components
python config_validator.py --git
python config_validator.py --docker
python config_validator.py --k8s

# Verbose mode with details
python config_validator.py -v

# Export report
python config_validator.py --export json > config-report.json
python config_validator.py --export yaml > config-report.yaml
```

### Dependency Installer

```bash
# Standard installation
./install_dependencies.sh

# Check what would be installed (dry run)
./install_dependencies.sh --check

# Minimal installation (only essentials)
./install_dependencies.sh --minimal

# Full installation (including Terraform, cloud CLIs)
./install_dependencies.sh --full

# Skip specific tools
./install_dependencies.sh --no-python
./install_dependencies.sh --no-docker

# Install cloud tools
./install_dependencies.sh --cloud-tools
```

## 📊 Sample Output

### Environment Check Results

```
================================================================
  AI Infrastructure Development Environment Check
================================================================

✅ PASS Python (3.11.0)
    Python 3.11.0 is installed (>= 3.11 required)

✅ PASS pip (23.0.1)
    pip is installed

⚠️  WARN Git (2.40.0)
    Git installed but not fully configured
    Fix: git config --global user.name 'Your Name' && ...

❌ FAIL Docker
    Docker daemon not accessible
    Fix: Start Docker Desktop or docker service

================================================================
Summary:
  ✅ Passed: 8
  ⚠️  Warnings: 3
  ❌ Failed: 2
================================================================
```

### Configuration Validation

```
================================================================
  Configuration Validation Report
================================================================

✅ VALID Git Configuration
    Git configuration is properly set up
    Details:
      user.name: John Doe
      user.email: john@example.com

⚠️  WARNING Docker Configuration
    Docker configured with 1 warning(s)
    Recommendations:
      • Increase Docker memory to at least 4GB for ML workloads
```

## 🧪 Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=../ --cov-report=html

# Run specific test
pytest tests/test_check_environment.py::TestEnvironmentChecker::test_check_python -v
```

Expected output:
```
tests/test_check_environment.py::TestEnvironmentChecker::test_initialization PASSED
tests/test_check_environment.py::TestEnvironmentChecker::test_check_python PASSED
...
====== 35 passed in 2.5s ======
```

## 📝 What These Tools Check

### Environment Checker

| Component | What's Checked |
|-----------|---------------|
| Python | Version >= 3.11 recommended |
| pip | Installation and version |
| Git | Installation, version, config (user.name, user.email) |
| Docker | Installation, daemon running, version |
| Docker Compose | V1 or V2 installation |
| kubectl | Kubernetes CLI installation |
| Terraform | Installation (optional) |
| AWS CLI | Installation (optional) |
| gcloud | Google Cloud SDK (optional) |
| VS Code | Installation |
| Virtual Envs | venv or virtualenv available |
| ML Libraries | numpy, pandas, scikit-learn, etc. |
| Disk Space | >= 10GB free (20GB recommended) |
| RAM | >= 8GB recommended for ML |

### Configuration Validator

| Configuration | What's Validated |
|--------------|------------------|
| Git | .gitconfig, user settings, SSH keys |
| Docker | Daemon config, resource limits (CPU/memory) |
| Kubernetes | kubeconfig, contexts, cluster connectivity |
| AWS | ~/.aws/config, credentials, active account |
| GCP | gcloud config, project, account |
| Python | Virtual environment usage |
| VS Code | settings.json, recommended settings |
| SSH | Keys, permissions, config file |

## 🎯 Learning Objectives

After using these tools, you should understand:

1. **Environment Validation**
   - How to programmatically check tool installations
   - Version checking and compatibility
   - System resource validation

2. **Configuration Management**
   - Where configs are stored (~/.config, ~/.aws, etc.)
   - Proper file permissions (SSH keys)
   - Configuration file formats (YAML, JSON, INI)

3. **Automation**
   - Cross-platform scripting
   - Dry-run implementations
   - Error handling and recovery

4. **Testing**
   - Mocking external commands
   - Testing system-dependent code
   - Integration testing strategies

## 🔍 Troubleshooting

### Issue: "command not found" when running scripts

**Solution**:
```bash
# Make scripts executable
chmod +x check_environment.py
chmod +x config_validator.py
chmod +x install_dependencies.sh
```

### Issue: ModuleNotFoundError when running Python scripts

**Solution**:
```bash
# Install dependencies
pip install -r requirements.txt

# Or create virtual environment first
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: Permission denied when installing dependencies

**Solution**:
```bash
# On Linux, some installations need sudo
# The script will prompt for sudo when needed

# Or install with --user flag for Python packages
pip install --user -r requirements.txt
```

### Issue: Docker checks fail even though Docker is installed

**Solution**:
```bash
# Make sure Docker daemon is running
docker ps

# On Linux, add user to docker group
sudo usermod -aG docker $USER
# Log out and back in

# On macOS, start Docker Desktop
open -a Docker
```

## 📚 Additional Resources

- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)
- [Git Configuration](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup)
- [Docker Installation](https://docs.docker.com/get-docker/)
- [Kubernetes kubectl](https://kubernetes.io/docs/tasks/tools/)

## 🤝 Contributing

These tools are part of the AI Infrastructure curriculum. To improve them:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

---

**Next Steps**: After validating your environment, proceed to the exercises in this module to apply these concepts.
