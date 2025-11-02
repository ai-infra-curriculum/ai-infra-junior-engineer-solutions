# Implementation Guide: Linux Networking and Troubleshooting for ML Infrastructure

## Overview

Master essential Linux networking tools and troubleshooting techniques for managing ML infrastructure. Learn to diagnose connectivity issues, secure SSH access, configure firewalls, and analyze network performance.

**Estimated Time:** 90-120 minutes
**Difficulty:** Intermediate

## Prerequisites

- Linux system with sudo access
- Basic TCP/IP knowledge
- Completed Exercises 01-06

## Phase 1: Network Configuration Basics (20 minutes)

### Step 1.1: Inspect Network Interfaces

```bash
# Modern way: ip command
ip addr show
ip link show

# Show specific interface
ip addr show eth0

# Show routing table
ip route show

# Legacy commands (still useful)
ifconfig
route -n
```

**Expected Output:**
```
2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500
    inet 192.168.1.100/24 brd 192.168.1.255 scope global eth0
```

### Step 1.2: Check Network Connectivity

```bash
# Test basic connectivity
ping -c 4 8.8.8.8

# Test with specific interface
ping -I eth0 -c 4 google.com

# Check local connectivity
ping -c 4 192.168.1.1

# Check if network interface is up
ip link show eth0 | grep "state UP"
```

**Validation:**
- [ ] Can view network interfaces
- [ ] Can see IP addresses
- [ ] Can ping external hosts

### Step 1.3: DNS Resolution

```bash
# Test DNS resolution
nslookup google.com
dig google.com

# Check DNS servers
cat /etc/resolv.conf

# Test specific DNS server
dig @8.8.8.8 google.com

# Reverse DNS lookup
dig -x 8.8.8.8
```

## Phase 2: Connection Diagnostics (30 minutes)

### Step 2.1: Check Active Connections

```bash
# Modern way: ss command
ss -tuln  # All TCP/UDP listening sockets

# Show established connections
ss -tn state established

# Show listening ports
ss -tln

# Show process using port
sudo ss -tlnp | grep :8080

# Legacy: netstat
netstat -tuln
netstat -tnp
```

**Key flags:**
- `-t`: TCP
- `-u`: UDP
- `-l`: Listening
- `-n`: Numeric (don't resolve names)
- `-p`: Show process

### Step 2.2: Test Port Connectivity

```bash
# Install telnet (if needed)
sudo apt install -y telnet || sudo yum install -y telnet

# Test if port is open
telnet localhost 8080

# Better: use nc (netcat)
nc -zv localhost 8080
nc -zv google.com 80

# Test port range
nc -zv localhost 8000-8010

# HTTP request with curl
curl -v http://localhost:8080

# Test with timeout
timeout 5 nc -zv remote-host 22
```

### Step 2.3: Traceroute and Latency

```bash
# Trace route to destination
traceroute google.com

# Alternative: mtr (better tool)
sudo apt install -y mtr
mtr google.com

# Show network path
tracepath google.com

# Measure round-trip time
ping -c 10 google.com | tail -1
```

**Validation:**
- [ ] Can check listening ports
- [ ] Can test port connectivity
- [ ] Can trace network path

## Phase 3: SSH Configuration and Troubleshooting (25 minutes)

### Step 3.1: SSH Key-Based Authentication

```bash
# Generate SSH key pair
ssh-keygen -t ed25519 -C "ml-infrastructure"

# Or RSA (older but more compatible)
ssh-keygen -t rsa -b 4096 -C "ml-infrastructure"

# Copy key to remote server
ssh-copy-id user@remote-host

# Or manually
cat ~/.ssh/id_ed25519.pub | ssh user@remote-host "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"

# Test connection
ssh user@remote-host

# SSH with specific key
ssh -i ~/.ssh/custom_key user@remote-host
```

### Step 3.2: SSH Configuration File

```bash
# Create SSH config
cat > ~/.ssh/config << 'EOF'
# ML Training Servers
Host ml-train-*
    User mluser
    IdentityFile ~/.ssh/ml_training_key
    ForwardAgent yes
    ServerAliveInterval 60

# GPU Nodes
Host gpu-node-01
    HostName 192.168.10.101
    User mluser
    Port 2222
    IdentityFile ~/.ssh/gpu_access_key

Host gpu-node-02
    HostName 192.168.10.102
    User mluser
    Port 2222
    IdentityFile ~/.ssh/gpu_access_key

# Jump host pattern
Host internal-*
    ProxyJump bastion.company.com
    User mluser
EOF

chmod 600 ~/.ssh/config

# Now connect easily
ssh ml-train-01
ssh gpu-node-01
```

### Step 3.3: Troubleshoot SSH Issues

```bash
# Debug SSH connection
ssh -vvv user@host

# Common issues and fixes:

# 1. Permission denied
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys  # on remote

# 2. Host key verification failed
ssh-keygen -R remote-host

# 3. Connection timeout
# Check if SSH service is running on remote
sudo systemctl status sshd

# Check firewall
sudo ufw status
sudo iptables -L -n | grep 22
```

**Validation:**
- [ ] Generated SSH keys
- [ ] Can connect via SSH
- [ ] Created SSH config file
- [ ] Can troubleshoot SSH issues

## Phase 4: Firewall Configuration (30 minutes)

### Step 4.1: UFW (Uncomplicated Firewall)

```bash
# Install UFW (Ubuntu/Debian)
sudo apt install -y ufw

# Check status
sudo ufw status verbose

# Enable firewall
sudo ufw enable

# Allow SSH (IMPORTANT: do this first!)
sudo ufw allow ssh
sudo ufw allow 22/tcp

# Allow specific services
sudo ufw allow 8080/tcp  # ML API
sudo ufw allow 5432/tcp  # PostgreSQL
sudo ufw allow 6379/tcp  # Redis

# Allow from specific IP
sudo ufw allow from 192.168.1.100 to any port 5432

# Allow port range
sudo ufw allow 8000:8010/tcp

# Delete rule
sudo ufw delete allow 8080/tcp

# Show numbered rules
sudo ufw status numbered

# Delete by number
sudo ufw delete 3
```

### Step 4.2: Create ML Service Rules

```bash
cat > setup_firewall.sh << 'EOF'
#!/bin/bash
# Configure firewall for ML infrastructure

set -e

echo "Configuring UFW for ML services..."

# Reset UFW
sudo ufw --force reset

# Default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH
sudo ufw allow 22/tcp comment 'SSH'

# ML API endpoints
sudo ufw allow 8000:8010/tcp comment 'ML API servers'

# Database access (only from app servers)
sudo ufw allow from 192.168.10.0/24 to any port 5432 comment 'PostgreSQL from app network'
sudo ufw allow from 192.168.10.0/24 to any port 6379 comment 'Redis from app network'

# Monitoring
sudo ufw allow 9090/tcp comment 'Prometheus'
sudo ufw allow 3000/tcp comment 'Grafana'

# Enable firewall
sudo ufw --force enable

echo "Firewall configured. Status:"
sudo ufw status numbered
EOF

chmod +x setup_firewall.sh
# Run with: sudo ./setup_firewall.sh
```

### Step 4.3: iptables (Advanced)

```bash
# View current rules
sudo iptables -L -n -v

# Save current rules
sudo iptables-save > iptables_backup.txt

# Allow specific port
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT

# Allow from specific IP
sudo iptables -A INPUT -p tcp -s 192.168.1.100 --dport 5432 -j ACCEPT

# Block specific IP
sudo iptables -A INPUT -s 10.0.0.5 -j DROP

# Delete rule
sudo iptables -D INPUT -p tcp --dport 8080 -j ACCEPT

# Flush all rules (CAUTION!)
sudo iptables -F
```

**Validation:**
- [ ] UFW installed and configured
- [ ] Can allow/deny specific ports
- [ ] Can restrict access by IP
- [ ] Created ML service firewall rules

## Phase 5: Network Performance Analysis (25 minutes)

### Step 5.1: Bandwidth Testing

```bash
# Install iperf3
sudo apt install -y iperf3

# Server mode (on one machine)
iperf3 -s

# Client mode (on another machine)
iperf3 -c server-ip

# Test with specific parameters
iperf3 -c server-ip -t 30  # 30 second test
iperf3 -c server-ip -P 4   # 4 parallel streams

# Monitor bandwidth
sudo apt install -y iftop
sudo iftop -i eth0
```

### Step 5.2: Packet Capture with tcpdump

```bash
# Capture packets on interface
sudo tcpdump -i eth0

# Capture specific port
sudo tcpdump -i eth0 port 8080

# Capture and save to file
sudo tcpdump -i eth0 -w capture.pcap

# Read from file
tcpdump -r capture.pcap

# Filter HTTP traffic
sudo tcpdump -i eth0 'tcp port 80'

# Capture only SYN packets
sudo tcpdump -i eth0 'tcp[tcpflags] & tcp-syn != 0'

# Show packet contents
sudo tcpdump -i eth0 -X port 8080
```

### Step 5.3: Network Monitoring Script

```bash
cat > network_monitor.sh << 'EOF'
#!/bin/bash
# Monitor network connections and bandwidth

echo "=== Network Monitoring Dashboard ==="
echo "Time: $(date)"
echo ""

echo "Active Connections:"
ss -tn state established | tail -n +2 | wc -l
echo ""

echo "Listening Services:"
sudo ss -tlnp | grep LISTEN | awk '{print $4, $7}' | sort -u
echo ""

echo "Top Bandwidth Consumers:"
if command -v nethogs &> /dev/null; then
    sudo nethogs -t -d 5 | head -20
else
    echo "nethogs not installed: sudo apt install nethogs"
fi
echo ""

echo "Network Interface Statistics:"
ip -s link show
EOF

chmod +x network_monitor.sh
```

**Validation:**
- [ ] Can test bandwidth with iperf3
- [ ] Can capture packets with tcpdump
- [ ] Can monitor network usage

## Phase 6: Troubleshooting Common Issues (30 minutes)

### Step 6.1: "Cannot Reach Host" Troubleshooting

```bash
cat > diagnose_connectivity.sh << 'EOF'
#!/bin/bash
# Diagnose network connectivity issues

HOST="$1"
PORT="${2:-22}"

if [ -z "$HOST" ]; then
    echo "Usage: $0 <host> [port]"
    exit 1
fi

echo "=== Connectivity Diagnosis for $HOST:$PORT ==="
echo ""

# 1. DNS Resolution
echo "[1] DNS Resolution:"
if host "$HOST" > /dev/null 2>&1; then
    echo "✓ DNS resolves: $(host "$HOST" | grep "has address" | awk '{print $NF}')"
    IP=$(host "$HOST" | grep "has address" | awk '{print $NF}' | head -1)
else
    echo "✗ DNS resolution failed"
    exit 1
fi
echo ""

# 2. Ping Test
echo "[2] Ping Test:"
if ping -c 3 -W 2 "$IP" > /dev/null 2>&1; then
    echo "✓ Host is reachable"
else
    echo "✗ Host is not reachable (may be blocking ICMP)"
fi
echo ""

# 3. Port Connectivity
echo "[3] Port Connectivity Test:"
if timeout 5 bash -c "echo >/dev/tcp/$IP/$PORT" 2>/dev/null; then
    echo "✓ Port $PORT is open"
else
    echo "✗ Port $PORT is closed or filtered"
fi
echo ""

# 4. Traceroute
echo "[4] Network Path:"
traceroute -m 10 -w 2 "$IP" 2>&1 | head -10
echo ""

# 5. Check local firewall
echo "[5] Local Firewall:"
if command -v ufw &> /dev/null; then
    sudo ufw status | grep "$PORT"
fi
echo ""

echo "Diagnosis complete."
EOF

chmod +x diagnose_connectivity.sh
./diagnose_connectivity.sh google.com 443
```

### Step 6.2: "Port Already in Use" Fix

```bash
# Find process using port
sudo lsof -i :8080
# or
sudo ss -tlnp | grep :8080

# Kill process by port
sudo fuser -k 8080/tcp

# Or kill by PID
sudo kill -9 $(sudo lsof -t -i:8080)
```

### Step 6.3: "Slow Network" Diagnosis

```bash
cat > diagnose_slow_network.sh << 'EOF'
#!/bin/bash
# Diagnose slow network issues

HOST="$1"

echo "=== Network Performance Diagnosis ==="
echo ""

# Latency test
echo "[1] Latency Test:"
ping -c 10 "$HOST" | tail -3
echo ""

# MTR for detailed path analysis
echo "[2] Network Path Analysis:"
mtr -r -c 10 "$HOST"
echo ""

# DNS resolution time
echo "[3] DNS Resolution Time:"
time nslookup "$HOST" > /dev/null
echo ""

# Check for packet loss
echo "[4] Packet Loss Test:"
ping -c 100 -i 0.2 "$HOST" 2>&1 | grep "packet loss"
echo ""

echo "Diagnosis complete."
EOF

chmod +x diagnose_slow_network.sh
```

**Validation:**
- [ ] Can diagnose connectivity issues
- [ ] Can find processes using ports
- [ ] Can troubleshoot slow connections

## Common Issues and Solutions

### Issue 1: SSH Connection Refused

**Symptoms:**
```
ssh: connect to host X port 22: Connection refused
```

**Solutions:**
```bash
# Check if SSH service is running
sudo systemctl status sshd

# Start SSH service
sudo systemctl start sshd

# Check if firewall is blocking
sudo ufw allow 22
sudo iptables -L -n | grep 22

# Verify SSH is listening
sudo ss -tln | grep :22
```

### Issue 2: DNS Not Resolving

**Symptoms:**
```
ping: google.com: Name or service not known
```

**Solutions:**
```bash
# Check DNS configuration
cat /etc/resolv.conf

# Add Google DNS
echo "nameserver 8.8.8.8" | sudo tee -a /etc/resolv.conf

# Test DNS
nslookup google.com
dig google.com

# Restart network
sudo systemctl restart systemd-resolved
```

### Issue 3: Port Already in Use

**Symptoms:**
```
Error: Address already in use
```

**Solutions:**
```bash
# Find what's using the port
sudo lsof -i :8080

# Kill the process
sudo kill -9 <PID>

# Or use fuser
sudo fuser -k 8080/tcp

# Change port in application config
```

### Issue 4: Firewall Blocking Connections

**Symptoms:**
Connection timeouts to specific ports

**Solutions:**
```bash
# Check UFW status
sudo ufw status

# Allow specific port
sudo ufw allow 8080/tcp

# Check iptables
sudo iptables -L -n -v

# Temporarily disable for testing (CAUTION!)
sudo ufw disable
# Re-enable after testing
sudo ufw enable
```

## Best Practices Summary

### Network Configuration

✅ Use modern tools (`ip`, `ss`) over legacy (`ifconfig`, `netstat`)
✅ Document network topology
✅ Use static IPs for infrastructure servers
✅ Configure proper DNS resolution
✅ Set up proper hostname resolution

### SSH Security

✅ Use key-based authentication (not passwords)
✅ Disable root login
✅ Change default SSH port if internet-facing
✅ Use SSH config file for convenience
✅ Set up SSH jump hosts for internal access

### Firewall Management

✅ Default deny incoming, allow outgoing
✅ Always allow SSH before enabling firewall
✅ Use specific IP restrictions for databases
✅ Document firewall rules
✅ Test rules after changes

### Troubleshooting

✅ Start with basic connectivity (ping)
✅ Check DNS resolution
✅ Verify ports are open (nc, telnet)
✅ Check firewalls on both ends
✅ Use tcpdump for detailed analysis

### Performance

✅ Monitor bandwidth usage
✅ Check for packet loss
✅ Measure latency regularly
✅ Use MTR for path analysis
✅ Test during peak hours

## Completion Checklist

### Network Basics
- [ ] Can view network interfaces and IPs
- [ ] Can check routing tables
- [ ] Can test connectivity with ping
- [ ] Understand DNS resolution

### Diagnostics
- [ ] Can check active connections with ss
- [ ] Can test port connectivity
- [ ] Can trace network paths
- [ ] Can capture packets with tcpdump

### SSH
- [ ] Generated SSH keys
- [ ] Set up key-based authentication
- [ ] Created SSH config file
- [ ] Can troubleshoot SSH issues

### Firewall
- [ ] Configured UFW rules
- [ ] Can allow/deny specific ports
- [ ] Can restrict by IP address
- [ ] Understand iptables basics

### Troubleshooting
- [ ] Can diagnose connectivity issues
- [ ] Can find processes using ports
- [ ] Can troubleshoot DNS problems
- [ ] Can analyze network performance

## Next Steps

1. **Production Skills:**
   - Network monitoring (Prometheus, Grafana)
   - VPN setup for secure access
   - Load balancer configuration

2. **ML-Specific:**
   - Distributed training network setup
   - Model serving network optimization
   - Data transfer optimization

3. **Advanced Topics:**
   - Network segmentation
   - SDN (Software Defined Networking)
   - Service mesh (Istio, Linkerd)

## Resources

- [iproute2 Documentation](https://wiki.linuxfoundation.org/networking/iproute2)
- [UFW Guide](https://help.ubuntu.com/community/UFW)
- [SSH Best Practices](https://infosec.mozilla.org/guidelines/openssh)
- [tcpdump Tutorial](https://www.tcpdump.org/manpages/tcpdump.1.html)

Congratulations! You can now troubleshoot network issues in ML infrastructure.
