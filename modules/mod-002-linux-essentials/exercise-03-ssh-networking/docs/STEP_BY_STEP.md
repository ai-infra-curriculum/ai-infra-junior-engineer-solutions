# Step-by-Step Implementation Guide: SSH & Networking for ML Infrastructure

## Overview

Master secure remote access and networking by building tools to manage SSH tunnels, check network health, and audit security for ML infrastructure. Learn to securely access remote GPU servers, troubleshoot connectivity issues, and maintain secure network configurations.

**Time**: 3-4 hours | **Difficulty**: Intermediate

---

## Prerequisites

```bash
# Verify SSH client
ssh -V

# Install network tools
sudo apt update
sudo apt install -y \
    openssh-client \
    openssh-server \
    netcat-openbsd \
    nmap \
    tcpdump \
    iptraf-ng \
    net-tools \
    curl \
    wget \
    dnsutils \
    traceroute

# Verify tools
which ssh scp rsync nc nmap netstat ss ip

# Generate SSH key if not exists
if [[ ! -f ~/.ssh/id_rsa ]]; then
    ssh-keygen -t rsa -b 4096 -C "ml-infrastructure"
fi
```

---

## Learning Objectives

By completing this exercise, you will be able to:

✅ Configure SSH keys and secure authentication
✅ Create and manage SSH tunnels for ML services
✅ Set up port forwarding for Jupyter and TensorBoard
✅ Monitor network connectivity and performance
✅ Troubleshoot common network issues
✅ Audit SSH security configurations
✅ Implement network security best practices

---

## Phase 1: SSH Tunnel Management Script (90 minutes)

### Step 1: Understanding SSH Tunneling

SSH tunnels allow secure access to remote services:

**Local Port Forwarding** (access remote service on local port):
```bash
# Forward remote Jupyter (8888) to local port 8888
ssh -L 8888:localhost:8888 user@gpu-server

# Now access at http://localhost:8888
```

**Remote Port Forwarding** (expose local service to remote):
```bash
# Make local service (port 5000) accessible on remote port 8080
ssh -R 8080:localhost:5000 user@remote-server
```

**Dynamic Port Forwarding** (SOCKS proxy):
```bash
# Create SOCKS proxy on local port 1080
ssh -D 1080 user@server

# Configure browser to use localhost:1080 as SOCKS proxy
```

### Step 2: Create manage_tunnels.sh

Start with the script structure:

```bash
#!/bin/bash
#
# manage_tunnels.sh - Manage SSH tunnels for ML infrastructure
#
# Description:
#   Create, monitor, and manage SSH tunnels for accessing
#   remote ML services (Jupyter, TensorBoard, MLflow, etc.)
#
# Usage:
#   ./manage_tunnels.sh [OPTIONS] COMMAND
#
# Commands:
#   start PROFILE     Start SSH tunnel using profile
#   stop PROFILE      Stop running tunnel
#   status            Show all running tunnels
#   list              List available profiles
#   create PROFILE    Create new tunnel profile
#
# Options:
#   -c, --config FILE  Configuration file (default: ~/.ml-tunnels.conf)
#   -v, --verbose      Verbose output
#   -h, --help         Show help
#

set -euo pipefail

# Configuration
readonly SCRIPT_NAME="$(basename "$0")"
readonly CONFIG_FILE="${HOME}/.ml-tunnels.conf"
readonly PID_DIR="${HOME}/.ml-tunnels/pids"
readonly LOG_DIR="${HOME}/.ml-tunnels/logs"

VERBOSE=false

# Logging
log() {
    local level="$1"
    shift
    echo "[$level] $*" >&2
}

log_verbose() {
    if [[ "$VERBOSE" == true ]]; then
        log "DEBUG" "$@"
    fi
}

# Initialize directories
init() {
    mkdir -p "$PID_DIR" "$LOG_DIR"

    if [[ ! -f "$CONFIG_FILE" ]]; then
        create_default_config
    fi
}
```

### Step 3: Implement Configuration Management

Add config file handling:

```bash
create_default_config() {
    cat > "$CONFIG_FILE" <<'EOF'
# ML Infrastructure SSH Tunnel Configuration
# Format: PROFILE_NAME|HOST|USER|LOCAL_PORT|REMOTE_HOST|REMOTE_PORT|DESCRIPTION

# Jupyter notebook on GPU server
jupyter-gpu1|gpu1.example.com|ml-user|8888|localhost|8888|Jupyter on GPU Server 1
jupyter-gpu2|gpu2.example.com|ml-user|8889|localhost|8888|Jupyter on GPU Server 2

# TensorBoard
tensorboard-gpu1|gpu1.example.com|ml-user|6006|localhost|6006|TensorBoard GPU 1

# MLflow tracking server
mlflow|mlflow.example.com|admin|5000|localhost|5000|MLflow Tracking Server

# Postgres database
postgres-db|db.example.com|dbadmin|5432|localhost|5432|ML Training Database

# Ray dashboard
ray-dashboard|ray-head.example.com|ray-user|8265|localhost|8265|Ray Cluster Dashboard
EOF

    log "INFO" "Created default config: $CONFIG_FILE"
    log "INFO" "Edit this file to add your servers"
}

read_profile() {
    local profile="$1"

    if [[ ! -f "$CONFIG_FILE" ]]; then
        log "ERROR" "Config file not found: $CONFIG_FILE"
        return 1
    fi

    local profile_line=$(grep -v "^#" "$CONFIG_FILE" | grep "^${profile}|" | head -1)

    if [[ -z "$profile_line" ]]; then
        log "ERROR" "Profile not found: $profile"
        return 1
    fi

    echo "$profile_line"
}

parse_profile() {
    local profile_line="$1"

    IFS='|' read -r PROFILE HOST USER LOCAL_PORT REMOTE_HOST REMOTE_PORT DESCRIPTION <<< "$profile_line"

    log_verbose "Parsed profile: $PROFILE"
    log_verbose "  Host: $HOST"
    log_verbose "  User: $USER"
    log_verbose "  Local Port: $LOCAL_PORT"
    log_verbose "  Remote: $REMOTE_HOST:$REMOTE_PORT"
}
```

### Step 4: Implement Tunnel Start Function

Add tunnel creation:

```bash
start_tunnel() {
    local profile="$1"

    log "INFO" "Starting tunnel: $profile"

    # Read and parse profile
    local profile_line=$(read_profile "$profile")
    if [[ $? -ne 0 ]]; then
        return 1
    fi

    parse_profile "$profile_line"

    # Check if tunnel already running
    if is_tunnel_running "$profile"; then
        log "WARNING" "Tunnel $profile is already running"
        return 0
    fi

    # Check if local port is available
    if is_port_in_use "$LOCAL_PORT"; then
        log "ERROR" "Local port $LOCAL_PORT is already in use"
        return 1
    fi

    # Create SSH tunnel
    local pid_file="$PID_DIR/${profile}.pid"
    local log_file="$LOG_DIR/${profile}.log"

    log "INFO" "Creating SSH tunnel: localhost:${LOCAL_PORT} -> ${HOST}:${REMOTE_HOST}:${REMOTE_PORT}"

    ssh -f -N \
        -L "${LOCAL_PORT}:${REMOTE_HOST}:${REMOTE_PORT}" \
        -o ServerAliveInterval=60 \
        -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes \
        "${USER}@${HOST}" \
        -E "$log_file"

    # Get PID of SSH process
    local tunnel_pid=$(pgrep -f "ssh.*${LOCAL_PORT}:${REMOTE_HOST}:${REMOTE_PORT}.*${USER}@${HOST}" | head -1)

    if [[ -z "$tunnel_pid" ]]; then
        log "ERROR" "Failed to start tunnel"
        return 1
    fi

    # Save PID
    echo "$tunnel_pid" > "$pid_file"

    log "SUCCESS" "Tunnel started (PID: $tunnel_pid)"
    log "INFO" "  Access at: http://localhost:${LOCAL_PORT}"
    log "INFO" "  Description: $DESCRIPTION"
    log "INFO" "  Log file: $log_file"

    return 0
}

is_port_in_use() {
    local port="$1"

    if command -v lsof &> /dev/null; then
        lsof -i ":$port" &> /dev/null
    elif command -v ss &> /dev/null; then
        ss -tuln | grep ":$port " &> /dev/null
    else
        netstat -tuln | grep ":$port " &> /dev/null
    fi
}

is_tunnel_running() {
    local profile="$1"
    local pid_file="$PID_DIR/${profile}.pid"

    if [[ ! -f "$pid_file" ]]; then
        return 1
    fi

    local pid=$(cat "$pid_file")

    if kill -0 "$pid" 2>/dev/null; then
        return 0
    else
        # PID file exists but process is dead, clean up
        rm -f "$pid_file"
        return 1
    fi
}
```

### Step 5: Implement Tunnel Stop Function

Add tunnel teardown:

```bash
stop_tunnel() {
    local profile="$1"

    log "INFO" "Stopping tunnel: $profile"

    if ! is_tunnel_running "$profile"; then
        log "WARNING" "Tunnel $profile is not running"
        return 0
    fi

    local pid_file="$PID_DIR/${profile}.pid"
    local pid=$(cat "$pid_file")

    log "INFO" "Killing SSH process (PID: $pid)"

    if kill "$pid" 2>/dev/null; then
        # Wait for process to exit
        local waited=0
        while kill -0 "$pid" 2>/dev/null && [[ $waited -lt 5 ]]; do
            sleep 1
            ((waited++))
        done

        if kill -0 "$pid" 2>/dev/null; then
            log "WARNING" "Process did not exit gracefully, forcing..."
            kill -9 "$pid" 2>/dev/null
        fi

        rm -f "$pid_file"
        log "SUCCESS" "Tunnel stopped"
        return 0
    else
        log "ERROR" "Failed to kill process $pid"
        rm -f "$pid_file"  # Clean up stale PID file
        return 1
    fi
}

stop_all_tunnels() {
    log "INFO" "Stopping all tunnels..."

    local count=0
    for pid_file in "$PID_DIR"/*.pid; do
        if [[ -f "$pid_file" ]]; then
            local profile=$(basename "$pid_file" .pid)
            if stop_tunnel "$profile"; then
                ((count++))
            fi
        fi
    done

    log "SUCCESS" "Stopped $count tunnel(s)"
}
```

### Step 6: Implement Status and List Functions

Add monitoring capabilities:

```bash
show_status() {
    log "INFO" "Active SSH Tunnels"
    log "INFO" "========================================"

    local count=0

    for pid_file in "$PID_DIR"/*.pid; do
        if [[ ! -f "$pid_file" ]]; then
            continue
        fi

        local profile=$(basename "$pid_file" .pid)
        local pid=$(cat "$pid_file")

        if kill -0 "$pid" 2>/dev/null; then
            ((count++))

            # Get profile details
            local profile_line=$(read_profile "$profile" 2>/dev/null || echo "")
            if [[ -n "$profile_line" ]]; then
                parse_profile "$profile_line"

                echo ""
                echo "Profile: $profile"
                echo "  PID: $pid"
                echo "  Local Port: $LOCAL_PORT"
                echo "  Remote: ${USER}@${HOST} -> ${REMOTE_HOST}:${REMOTE_PORT}"
                echo "  Description: $DESCRIPTION"
                echo "  Runtime: $(ps -p $pid -o etime= | tr -d ' ')"
                echo "  Access: http://localhost:${LOCAL_PORT}"
            else
                echo ""
                echo "Profile: $profile"
                echo "  PID: $pid (profile not found in config)"
            fi
        else
            # Clean up stale PID file
            rm -f "$pid_file"
        fi
    done

    echo ""
    if [[ $count -eq 0 ]]; then
        log "INFO" "No active tunnels"
    else
        log "INFO" "Total active tunnels: $count"
    fi
}

list_profiles() {
    log "INFO" "Available Tunnel Profiles"
    log "INFO" "========================================"

    if [[ ! -f "$CONFIG_FILE" ]]; then
        log "ERROR" "Config file not found: $CONFIG_FILE"
        return 1
    fi

    echo ""
    printf "%-20s %-15s %-10s %s\n" "PROFILE" "HOST" "PORT" "DESCRIPTION"
    echo "--------------------------------------------------------------------------------"

    while IFS='|' read -r profile host user local_port remote_host remote_port description; do
        if [[ "$profile" =~ ^#.*$ ]] || [[ -z "$profile" ]]; then
            continue
        fi

        local status="○"
        if is_tunnel_running "$profile"; then
            status="●"
        fi

        printf "%s %-18s %-15s %-10s %s\n" "$status" "$profile" "$host" "$local_port" "$description"
    done < "$CONFIG_FILE"

    echo ""
    echo "● = Running, ○ = Stopped"
}
```

### Step 7: Add Health Check Function

Monitor tunnel connectivity:

```bash
check_tunnel_health() {
    local profile="$1"

    if ! is_tunnel_running "$profile"; then
        log "ERROR" "Tunnel $profile is not running"
        return 1
    fi

    # Get profile details
    local profile_line=$(read_profile "$profile")
    parse_profile "$profile_line"

    log "INFO" "Checking health of tunnel: $profile"

    # Test if local port is listening
    if ! is_port_in_use "$LOCAL_PORT"; then
        log "ERROR" "Local port $LOCAL_PORT is not listening"
        return 1
    fi

    # Try to connect to local port
    if timeout 5 bash -c "cat < /dev/null > /dev/tcp/localhost/$LOCAL_PORT" 2>/dev/null; then
        log "SUCCESS" "Tunnel is healthy (port $LOCAL_PORT responding)"
        return 0
    else
        log "WARNING" "Cannot connect to local port $LOCAL_PORT"
        return 1
    fi
}

monitor_tunnels() {
    log "INFO" "Monitoring tunnels (press Ctrl+C to stop)..."

    while true; do
        clear
        echo "=========================================="
        echo "SSH Tunnel Monitor - $(date)"
        echo "=========================================="

        for pid_file in "$PID_DIR"/*.pid; do
            if [[ -f "$pid_file" ]]; then
                local profile=$(basename "$pid_file" .pid)

                echo ""
                echo "Profile: $profile"

                if check_tunnel_health "$profile"; then
                    echo "  Status: ✓ Healthy"
                else
                    echo "  Status: ✗ Unhealthy"

                    # Attempt restart
                    log "INFO" "Attempting to restart $profile..."
                    stop_tunnel "$profile"
                    sleep 2
                    start_tunnel "$profile"
                fi
            fi
        done

        sleep 30
    done
}
```

### Step 8: Add Main Function

Tie it all together:

```bash
usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS] COMMAND [ARGS]

Manage SSH tunnels for ML infrastructure services.

COMMANDS:
    start PROFILE       Start tunnel using profile
    stop PROFILE        Stop running tunnel
    stop-all            Stop all running tunnels
    status              Show all running tunnels
    list                List available profiles
    health PROFILE      Check tunnel health
    monitor             Monitor all tunnels (auto-restart on failure)
    edit                Edit configuration file

OPTIONS:
    -c, --config FILE   Configuration file (default: ~/.ml-tunnels.conf)
    -v, --verbose       Verbose output
    -h, --help          Show this help

EXAMPLES:
    # Start Jupyter tunnel
    $SCRIPT_NAME start jupyter-gpu1

    # Check status
    $SCRIPT_NAME status

    # Monitor and auto-restart
    $SCRIPT_NAME monitor

    # Edit config
    $SCRIPT_NAME edit

EOF
}

main() {
    # Parse options
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -*)
                log "ERROR" "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                break
                ;;
        esac
    done

    if [[ $# -eq 0 ]]; then
        usage
        exit 1
    fi

    init

    local command="$1"
    shift

    case "$command" in
        start)
            if [[ $# -eq 0 ]]; then
                log "ERROR" "Profile name required"
                exit 1
            fi
            start_tunnel "$1"
            ;;
        stop)
            if [[ $# -eq 0 ]]; then
                log "ERROR" "Profile name required"
                exit 1
            fi
            stop_tunnel "$1"
            ;;
        stop-all)
            stop_all_tunnels
            ;;
        status)
            show_status
            ;;
        list)
            list_profiles
            ;;
        health)
            if [[ $# -eq 0 ]]; then
                log "ERROR" "Profile name required"
                exit 1
            fi
            check_tunnel_health "$1"
            ;;
        monitor)
            monitor_tunnels
            ;;
        edit)
            ${EDITOR:-nano} "$CONFIG_FILE"
            ;;
        *)
            log "ERROR" "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

main "$@"
```

### Step 9: Test the Tunnel Manager

```bash
# List available profiles
./solutions/manage_tunnels.sh list

# Edit config to add your servers
./solutions/manage_tunnels.sh edit

# Start a tunnel
./solutions/manage_tunnels.sh start jupyter-gpu1

# Check status
./solutions/manage_tunnels.sh status

# Test the connection
curl http://localhost:8888

# Check health
./solutions/manage_tunnels.sh health jupyter-gpu1

# Stop tunnel
./solutions/manage_tunnels.sh stop jupyter-gpu1

# Start monitoring mode
./solutions/manage_tunnels.sh monitor
```

---

## Phase 2: Network Health Check Script (60 minutes)

### Summary

The `check_network_health.sh` script monitors network connectivity and performance.

**Features**:
- Ping remote hosts
- Check DNS resolution
- Test TCP port connectivity
- Measure latency and packet loss
- Detect network issues

**Core Implementation**:

```bash
check_host_reachable() {
    local host="$1"
    local count="${2:-4}"

    log "INFO" "Pinging $host..."

    if ping -c "$count" -W 2 "$host" &> /dev/null; then
        local avg_time=$(ping -c "$count" -W 2 "$host" | tail -1 | awk -F'/' '{print $5}')
        log "SUCCESS" "$host is reachable (avg: ${avg_time}ms)"
        return 0
    else
        log "ERROR" "$host is not reachable"
        return 1
    fi
}

check_dns_resolution() {
    local hostname="$1"

    log "INFO" "Resolving $hostname..."

    local ip=$(dig +short "$hostname" | head -1)

    if [[ -n "$ip" ]]; then
        log "SUCCESS" "$hostname resolves to $ip"
        echo "$ip"
        return 0
    else
        log "ERROR" "Failed to resolve $hostname"
        return 1
    fi
}

check_port_open() {
    local host="$1"
    local port="$2"
    local timeout="${3:-5}"

    log "INFO" "Checking $host:$port..."

    if timeout "$timeout" bash -c "cat < /dev/null > /dev/tcp/$host/$port" 2>/dev/null; then
        log "SUCCESS" "$host:$port is open"
        return 0
    else
        log "ERROR" "$host:$port is not accessible"
        return 1
    fi
}

measure_latency() {
    local host="$1"
    local count=10

    log "INFO" "Measuring latency to $host..."

    ping -c "$count" "$host" | tail -1 | awk -F'/' '{print "Min: "$4"ms, Avg: "$5"ms, Max: "$6"ms"}'
}

check_bandwidth() {
    local server="$1"

    log "INFO" "Testing bandwidth to $server..."

    # Download test (if curl/wget available)
    if command -v curl &> /dev/null; then
        local start=$(date +%s)
        curl -o /dev/null -s "$server/test-file" 2>&1
        local end=$(date +%s)
        local duration=$((end - start))

        echo "Download completed in ${duration}s"
    fi
}
```

---

## Phase 3: Security Audit Script (60 minutes)

### Summary

The `audit_security.sh` script checks SSH and network security configurations.

**Audit Checks**:
- SSH key strength
- SSH server configuration security
- Open ports analysis
- Firewall rules review
- Failed login attempts

**Implementation**:

```bash
audit_ssh_keys() {
    log "INFO" "Auditing SSH keys..."

    for key in ~/.ssh/id_*.pub; do
        if [[ -f "$key" ]]; then
            local key_type=$(ssh-keygen -l -f "$key" | awk '{print $4}' | tr -d '()')
            local key_bits=$(ssh-keygen -l -f "$key" | awk '{print $1}')

            echo "Key: $(basename "$key")"
            echo "  Type: $key_type"
            echo "  Bits: $key_bits"

            # Check key strength
            if [[ "$key_type" == "RSA" ]] && [[ $key_bits -lt 2048 ]]; then
                log "WARNING" "Weak RSA key detected (< 2048 bits)"
            fi
        fi
    done
}

audit_ssh_config() {
    local sshd_config="/etc/ssh/sshd_config"

    if [[ ! -f "$sshd_config" ]]; then
        log "WARNING" "SSHD config not found"
        return 1
    fi

    log "INFO" "Auditing SSH server configuration..."

    # Check for password authentication
    if grep -q "^PasswordAuthentication yes" "$sshd_config"; then
        log "WARNING" "Password authentication is enabled (recommend key-only)"
    fi

    # Check for root login
    if grep -q "^PermitRootLogin yes" "$sshd_config"; then
        log "WARNING" "Root login is permitted (recommend disabling)"
    fi

    # Check for X11 forwarding
    if grep -q "^X11Forwarding yes" "$sshd_config"; then
        log "INFO" "X11 forwarding is enabled"
    fi
}

scan_open_ports() {
    log "INFO" "Scanning open ports..."

    if command -v ss &> /dev/null; then
        ss -tuln
    elif command -v netstat &> /dev/null; then
        netstat -tuln
    else
        log "ERROR" "No network scanning tool available"
        return 1
    fi
}

check_failed_logins() {
    log "INFO" "Checking for failed SSH login attempts..."

    if [[ -f /var/log/auth.log ]]; then
        local failed_count=$(grep "Failed password" /var/log/auth.log | wc -l)
        log "INFO" "Failed login attempts: $failed_count"

        # Show top IPs with failures
        grep "Failed password" /var/log/auth.log | \
            awk '{print $(NF-3)}' | \
            sort | uniq -c | sort -rn | head -10
    fi
}
```

---

## Best Practices

1. **Use SSH keys, not passwords**: More secure and convenient
2. **Disable root login**: Use sudo for privileged access
3. **Use SSH config**: Simplifies connections and applies settings consistently
4. **Monitor tunnels**: Auto-restart failed tunnels
5. **Audit regularly**: Check for weak configurations
6. **Use fail2ban**: Automatically block brute-force attempts

---

## Next Steps

1. Integrate with systemd for persistent tunnels
2. Add Slack/email notifications for tunnel failures
3. Implement automatic failover between redundant servers
4. Create monitoring dashboards
5. Add support for ProxyJump and bastion hosts

---

## Resources

- [OpenSSH Manual](https://www.openssh.com/manual.html)
- [SSH Tunneling Guide](https://www.ssh.com/academy/ssh/tunneling)
- [Linux Network Administration](https://tldp.org/LDP/nag2/index.html)
- [Securing SSH](https://stribika.github.io/2015/01/04/secure-secure-shell.html)

---

**Congratulations!** You've built production-ready tools for managing remote ML infrastructure securely. 🚀
