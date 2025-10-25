# Exercise 03: SSH & Networking

## Overview

Master remote access and networking for ML infrastructure. Learn secure SSH practices, network configuration, troubleshooting, and remote management of training servers and GPU clusters.

## Learning Objectives

- ✅ Configure and use SSH for secure remote access
- ✅ Manage SSH keys and authentication
- ✅ Set up SSH tunneling and port forwarding
- ✅ Understand network fundamentals (TCP/IP, DNS, routing)
- ✅ Troubleshoot network connectivity issues
- ✅ Configure firewalls and security groups
- ✅ Monitor network performance
- ✅ Secure remote ML infrastructure access

## Topics Covered

### 1. SSH Fundamentals

#### Basic SSH Usage

```bash
# Connect to remote server
ssh username@hostname
ssh username@192.168.1.100

# Connect on specific port
ssh -p 2222 username@hostname

# Run single command
ssh username@hostname "ls -la /data"

# Execute local script on remote server
ssh username@hostname 'bash -s' < local_script.sh

# Copy files with SCP
scp file.txt username@hostname:/remote/path/
scp username@hostname:/remote/file.txt ./local/
scp -r directory/ username@hostname:/remote/path/

# Copy files with RSYNC (more efficient)
rsync -avz file.txt username@hostname:/remote/path/
rsync -avz --progress /data/ username@hostname:/backup/data/
rsync -avz -e "ssh -p 2222" file.txt username@hostname:/path/
```

#### SSH Configuration

```bash
# ~/.ssh/config - Client configuration
Host ml-server
    HostName 192.168.1.100
    User ml-user
    Port 22
    IdentityFile ~/.ssh/ml_server_key
    ForwardAgent yes
    ServerAliveInterval 60

Host gpu-cluster
    HostName gpu.example.com
    User admin
    ProxyJump bastion-host
    LocalForward 8888 localhost:8888  # Jupyter
    LocalForward 6006 localhost:6006  # TensorBoard

Host bastion-host
    HostName bastion.example.com
    User jumpuser
    IdentityFile ~/.ssh/bastion_key

# Usage
ssh ml-server           # Uses config settings
ssh gpu-cluster         # Connects through bastion
```

#### SSH Server Configuration

```bash
# /etc/ssh/sshd_config - Server configuration

# Basic settings
Port 22
AddressFamily any
ListenAddress 0.0.0.0

# Authentication
PermitRootLogin no
PubkeyAuthentication yes
PasswordAuthentication no
ChallengeResponseAuthentication no
UsePAM yes

# Security
MaxAuthTries 3
MaxSessions 10
LoginGraceTime 60
StrictModes yes

# Connection settings
ClientAliveInterval 300
ClientAliveCountMax 2
TCPKeepAlive yes

# Subsystems
Subsystem sftp /usr/lib/openssh/sftp-server

# Restart SSH after changes
sudo systemctl restart sshd
```

### 2. SSH Key Management

#### Generating SSH Keys

```bash
# Generate RSA key (4096-bit)
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# Generate Ed25519 key (recommended)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Generate key with custom filename
ssh-keygen -t ed25519 -f ~/.ssh/ml_server_key

# Generate key with passphrase
ssh-keygen -t ed25519 -C "ml-server" -f ~/.ssh/ml_key

# List fingerprints
ssh-keygen -l -f ~/.ssh/id_ed25519.pub
ssh-keygen -lv -f ~/.ssh/id_ed25519.pub  # Visual ASCII art

# Change passphrase
ssh-keygen -p -f ~/.ssh/id_ed25519
```

#### Managing Authorized Keys

```bash
# Copy public key to server
ssh-copy-id username@hostname
ssh-copy-id -i ~/.ssh/ml_key.pub username@hostname

# Manual key installation
cat ~/.ssh/id_ed25519.pub | ssh username@hostname "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"

# Set correct permissions
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub

# View authorized keys on server
cat ~/.ssh/authorized_keys

# Remove specific key
# Edit ~/.ssh/authorized_keys and delete the line
```

#### SSH Agent

```bash
# Start SSH agent
eval "$(ssh-agent -s)"

# Add key to agent
ssh-add ~/.ssh/id_ed25519
ssh-add ~/.ssh/ml_key

# List loaded keys
ssh-add -l
ssh-add -L  # Show full public keys

# Remove key from agent
ssh-add -d ~/.ssh/id_ed25519

# Remove all keys
ssh-add -D

# Agent forwarding (use with caution)
ssh -A username@hostname
```

### 3. SSH Tunneling and Port Forwarding

#### Local Port Forwarding

```bash
# Forward local port to remote
# Access remote service on local machine
ssh -L local_port:destination:destination_port username@hostname

# Forward Jupyter notebook (remote 8888 -> local 8888)
ssh -L 8888:localhost:8888 username@gpu-server

# Forward TensorBoard (remote 6006 -> local 6006)
ssh -L 6006:localhost:6006 username@ml-server

# Forward database (remote DB to local)
ssh -L 5432:localhost:5432 username@db-server

# Multiple port forwards
ssh -L 8888:localhost:8888 -L 6006:localhost:6006 username@ml-server

# Forward to different host
ssh -L 8080:internal-server:80 username@jump-host
```

#### Remote Port Forwarding

```bash
# Forward remote port to local
# Expose local service to remote network
ssh -R remote_port:localhost:local_port username@hostname

# Expose local web server on remote
ssh -R 8080:localhost:80 username@remote-server

# Expose local database
ssh -R 5432:localhost:5432 username@remote-server
```

#### Dynamic Port Forwarding (SOCKS Proxy)

```bash
# Create SOCKS proxy
ssh -D 8080 username@hostname

# Configure browser to use SOCKS proxy
# Host: localhost
# Port: 8080
# Type: SOCKS v5

# Use with curl
curl --socks5 localhost:8080 http://internal-site.com
```

#### SSH Tunneling for Jupyter

```bash
# On remote server
jupyter notebook --no-browser --port=8888

# On local machine
ssh -N -L 8888:localhost:8888 username@remote-server

# Access in browser: http://localhost:8888
```

### 4. Network Fundamentals

#### Network Interfaces

```bash
# List network interfaces
ip addr
ip link

# Show specific interface
ip addr show eth0

# Bring interface up/down
sudo ip link set eth0 up
sudo ip link set eth0 down

# Configure IP address
sudo ip addr add 192.168.1.100/24 dev eth0
sudo ip addr del 192.168.1.100/24 dev eth0

# Legacy commands (still widely used)
ifconfig
ifconfig eth0
ifconfig eth0 192.168.1.100 netmask 255.255.255.0
```

#### Routing

```bash
# Show routing table
ip route
ip route show
route -n

# Add route
sudo ip route add 192.168.2.0/24 via 192.168.1.1
sudo ip route add default via 192.168.1.1

# Delete route
sudo ip route del 192.168.2.0/24

# Trace route
traceroute google.com
tracepath google.com
mtr google.com  # Real-time traceroute
```

#### DNS Configuration

```bash
# DNS resolution
nslookup google.com
nslookup google.com 8.8.8.8  # Use specific DNS server

# Detailed DNS query
dig google.com
dig @8.8.8.8 google.com
dig google.com +short
dig google.com ANY  # All records

# Reverse DNS lookup
dig -x 8.8.8.8

# DNS configuration files
cat /etc/resolv.conf          # DNS servers
cat /etc/hosts                # Local hostname mapping
cat /etc/nsswitch.conf        # Name resolution order

# Test DNS resolution
host google.com
host -t MX google.com
```

### 5. Network Connectivity Testing

#### Basic Connectivity

```bash
# Ping
ping google.com
ping -c 4 192.168.1.1           # Send 4 packets
ping -i 0.5 google.com          # 0.5 second interval

# Check if host is up
ping -c 1 -W 1 192.168.1.1 && echo "Host is up" || echo "Host is down"

# Test TCP connection
nc -zv hostname 22              # Test SSH port
nc -zv hostname 80              # Test HTTP port
telnet hostname 22              # Interactive test

# Test UDP connection
nc -uzv hostname 53

# Check port availability
timeout 2 bash -c "</dev/tcp/hostname/22" && echo "Port 22 open"
```

#### Network Diagnostics

```bash
# Show active connections
netstat -tuln                   # Listening ports
netstat -tun                    # Active connections
netstat -tupln                  # With process info (needs sudo)

# Modern alternative: ss
ss -tuln                        # Listening ports
ss -tun                         # Active connections
ss -tupln                       # With process info
ss -s                           # Statistics summary

# Find process using port
sudo lsof -i :8080
sudo fuser 8080/tcp

# Network statistics
netstat -s                      # Protocol statistics
ss -s                           # Connection statistics
```

#### Bandwidth and Performance

```bash
# Download speed test
wget --output-document=/dev/null http://speedtest.com/test.bin
curl -o /dev/null http://speedtest.com/test.bin

# Network speed test (requires speedtest-cli)
speedtest-cli
speedtest-cli --simple

# Bandwidth monitoring
iftop                           # Interactive bandwidth usage
iftop -i eth0                   # Specific interface

# Network usage
nethogs                         # Per-process bandwidth
vnstat                          # Network statistics
vnstat -l                       # Live traffic
vnstat -h                       # Hourly stats

# Measure bandwidth between servers (iperf)
# On server
iperf3 -s

# On client
iperf3 -c server-ip
```

### 6. Firewall Configuration

#### UFW (Uncomplicated Firewall)

```bash
# Enable/disable firewall
sudo ufw enable
sudo ufw disable

# Status
sudo ufw status
sudo ufw status verbose
sudo ufw status numbered

# Default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow services
sudo ufw allow ssh
sudo ufw allow 22/tcp
sudo ufw allow 8888/tcp         # Jupyter
sudo ufw allow 6006/tcp         # TensorBoard

# Allow from specific IP
sudo ufw allow from 192.168.1.100
sudo ufw allow from 192.168.1.0/24

# Allow specific port from specific IP
sudo ufw allow from 192.168.1.100 to any port 22

# Deny connections
sudo ufw deny 23/tcp

# Delete rules
sudo ufw delete allow 80/tcp
sudo ufw delete 1               # By rule number

# Reset firewall
sudo ufw reset
```

#### iptables (Advanced)

```bash
# List rules
sudo iptables -L
sudo iptables -L -v -n

# Allow SSH
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HTTP/HTTPS
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow from specific IP
sudo iptables -A INPUT -s 192.168.1.100 -j ACCEPT

# Drop all other incoming
sudo iptables -P INPUT DROP

# Save rules
sudo iptables-save > /etc/iptables/rules.v4

# Restore rules
sudo iptables-restore < /etc/iptables/rules.v4
```

### 7. Network Security

#### SSL/TLS Certificates

```bash
# Check certificate
openssl s_client -connect example.com:443 -servername example.com

# View certificate details
openssl x509 -in cert.pem -text -noout

# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Check certificate expiration
echo | openssl s_client -connect example.com:443 2>/dev/null | openssl x509 -noout -dates
```

#### SSH Security Best Practices

```bash
# Disable password authentication
# /etc/ssh/sshd_config
PasswordAuthentication no
ChallengeResponseAuthentication no

# Disable root login
PermitRootLogin no

# Use non-standard port
Port 2222

# Limit user access
AllowUsers ml-user admin
DenyUsers guest

# Enable 2FA (requires google-authenticator-libpam)
# Install: sudo apt install libpam-google-authenticator
AuthenticationMethods publickey,keyboard-interactive

# Fail2ban for brute force protection
sudo apt install fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# Check banned IPs
sudo fail2ban-client status sshd
```

### 8. Remote ML Infrastructure Access

#### Accessing GPU Servers

```bash
# SSH config for GPU cluster
Host gpu-1
    HostName gpu1.ml.example.com
    User ml-engineer
    IdentityFile ~/.ssh/gpu_cluster_key
    LocalForward 8888 localhost:8888
    LocalForward 6006 localhost:6006
    ServerAliveInterval 60

Host gpu-2
    HostName gpu2.ml.example.com
    User ml-engineer
    IdentityFile ~/.ssh/gpu_cluster_key

# Connect and start Jupyter
ssh gpu-1 "jupyter notebook --no-browser --port=8888"

# Access in local browser: http://localhost:8888
```

#### Tmux for Persistent Sessions

```bash
# Install tmux
sudo apt install tmux

# Start new session
tmux new -s training

# Detach from session (Ctrl+B, then D)

# List sessions
tmux ls

# Attach to session
tmux attach -t training

# Kill session
tmux kill-session -t training

# Common workflow
ssh ml-server
tmux new -s experiment
# Start training
python train_model.py
# Detach with Ctrl+B, D
# Disconnect from SSH
# Later, reconnect
ssh ml-server
tmux attach -s experiment
# Training still running!
```

#### Screen (Alternative to Tmux)

```bash
# Start screen session
screen -S training

# Detach (Ctrl+A, then D)

# List sessions
screen -ls

# Reattach
screen -r training

# Kill session
screen -X -S training quit
```

---

## Project: Secure Remote ML Infrastructure

Build a secure remote access system for ML infrastructure.

### Requirements

**Components to Create:**
1. SSH configuration templates
2. Automated SSH key deployment
3. Port forwarding setup scripts
4. Network monitoring dashboard
5. Security audit script
6. Firewall configuration automation

**Security Requirements:**
- Key-based authentication only
- Fail2ban integration
- Port knocking option
- Connection logging and monitoring
- Automated security updates

### Implementation

See `solutions/` directory for complete implementations.

### Example Configurations

#### 1. SSH Configuration Template

```bash
# ~/.ssh/config

# Jump host / Bastion
Host bastion
    HostName bastion.ml.example.com
    User jumpuser
    IdentityFile ~/.ssh/bastion_key
    ServerAliveInterval 60
    ServerAliveCountMax 3

# ML Training Servers (via bastion)
Host ml-train-*
    User ml-engineer
    IdentityFile ~/.ssh/ml_cluster_key
    ProxyJump bastion
    ForwardAgent no
    ServerAliveInterval 60

Host ml-train-1
    HostName 10.0.1.101

Host ml-train-2
    HostName 10.0.1.102

Host ml-train-3
    HostName 10.0.1.103

# GPU Cluster
Host gpu-*
    User gpu-user
    IdentityFile ~/.ssh/gpu_key
    ProxyJump bastion
    LocalForward 8888 localhost:8888
    LocalForward 6006 localhost:6006

Host gpu-node-1
    HostName 10.0.2.101

Host gpu-node-2
    HostName 10.0.2.102

# Development server
Host dev
    HostName dev.ml.example.com
    User developer
    IdentityFile ~/.ssh/dev_key
    LocalForward 5432 localhost:5432
    LocalForward 3000 localhost:3000
```

#### 2. Automated Key Deployment

```bash
#!/bin/bash
# deploy_keys.sh - Deploy SSH keys to ML infrastructure

SERVERS=(
    "ml-train-1"
    "ml-train-2"
    "gpu-node-1"
    "gpu-node-2"
)

PUBLIC_KEY="~/.ssh/ml_cluster_key.pub"

for server in "${SERVERS[@]}"; do
    echo "Deploying key to $server..."
    ssh-copy-id -i "$PUBLIC_KEY" "$server"
done

echo "Key deployment complete!"
```

---

## Practice Problems

### Problem 1: SSH Tunnel Manager

Create a script that:
- Manages multiple SSH tunnels
- Auto-reconnects on disconnect
- Logs connection status
- Provides easy start/stop for tunnels

### Problem 2: Network Health Check

Create a script that:
- Tests connectivity to all ML servers
- Checks port availability
- Measures latency and bandwidth
- Generates health report

### Problem 3: Security Audit

Create a script that:
- Audits SSH configurations
- Checks for weak settings
- Scans for open ports
- Reports security issues
- Provides remediation recommendations

---

## Best Practices

### 1. SSH Security

```bash
# Use strong key algorithms
ssh-keygen -t ed25519

# Use separate keys for different purposes
~/.ssh/personal_key
~/.ssh/work_key
~/.ssh/ml_infrastructure_key

# Protect private keys
chmod 600 ~/.ssh/*_key

# Use SSH agent, not key files directly
ssh-add ~/.ssh/ml_key

# Never share private keys
# Only share public keys (.pub files)
```

### 2. Network Security

```bash
# Use firewall on all servers
sudo ufw enable
sudo ufw allow ssh

# Limit SSH access by IP
sudo ufw allow from 192.168.1.0/24 to any port 22

# Use fail2ban
sudo apt install fail2ban

# Monitor logs
sudo tail -f /var/log/auth.log
```

### 3. Remote Sessions

```bash
# Always use tmux or screen for long-running tasks
tmux new -s training
python train_model.py
# Detach: Ctrl+B, D

# Periodically check for disconnects
# Set ServerAliveInterval in SSH config

# Use logging for training scripts
python train.py 2>&1 | tee training.log
```

---

## Validation

Test your setup:

```bash
# Test SSH connection
ssh -v username@hostname

# Test key authentication
ssh -i ~/.ssh/key username@hostname

# Test port forward
ssh -L 8888:localhost:8888 username@hostname
# Access http://localhost:8888

# Test firewall
sudo ufw status
nmap localhost

# Test network connectivity
ping -c 4 hostname
nc -zv hostname 22

# Verify SSH configuration
ssh -G hostname
```

---

## Resources

- [OpenSSH Documentation](https://www.openssh.com/manual.html)
- [SSH Best Practices](https://www.ssh.com/academy/ssh/best-practices)
- [Linux Networking](https://linux-training.be/networking/)
- [UFW Documentation](https://help.ubuntu.com/community/UFW)
- [Fail2ban](https://www.fail2ban.org/)

---

## Next Steps

1. **Exercise 04: System Administration** - System management and automation
2. Set up your own secure SSH infrastructure
3. Practice with port forwarding for Jupyter/TensorBoard
4. Implement automated security monitoring
5. Learn network troubleshooting techniques

---

**Secure your ML infrastructure access! 🔒**
