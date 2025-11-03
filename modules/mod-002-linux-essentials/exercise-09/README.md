# Exercise 09: Linux Networking and Troubleshooting for ML Infrastructure

Complete network configuration, security hardening, and troubleshooting solution for ML infrastructure.

## Overview

This exercise provides production-ready network configuration files and troubleshooting tools for managing ML infrastructure, including:

- **SSH Security Hardening**: Comprehensive SSH daemon and client configuration
- **Firewall Configuration**: UFW rules for ML services
- **Network Troubleshooting**: Three specialized diagnostic scripts
- **Internal DNS**: Complete hosts file for ML infrastructure
- **Documentation**: Detailed guides and best practices

## Directory Structure

```
exercise-09/
├── config/
│   ├── 99-hardening.conf     # SSH daemon hardening
│   ├── ssh_config            # SSH client configuration
│   ├── ufw_rules.txt         # Firewall rules
│   └── hosts                 # Internal hostname resolution
├── scripts/
│   ├── network_monitor.sh    # Continuous network monitoring
│   ├── debug_connection.sh   # Connection diagnostics
│   └── analyze_latency.sh    # Latency analysis
├── docs/
│   └── NETWORK_GUIDE.md      # Comprehensive network guide
└── README.md                 # This file
```

## Quick Start

### 1. SSH Hardening

```bash
# Backup current configuration
sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup

# Install hardened configuration
sudo cp config/99-hardening.conf /etc/ssh/sshd_config.d/

# Test configuration
sudo sshd -t

# Apply (reload SSH)
sudo systemctl reload sshd

# Install client configuration
cp config/ssh_config ~/.ssh/config
chmod 600 ~/.ssh/config
mkdir -p ~/.ssh/sockets
```

### 2. Firewall Setup

```bash
# Review firewall rules
cat config/ufw_rules.txt

# Apply rules (carefully!)
sudo bash config/ufw_rules.txt

# Verify
sudo ufw status verbose
```

### 3. Internal DNS

```bash
# Add internal hostnames
sudo tee -a /etc/hosts < config/hosts

# Test resolution
ping -c 2 ml-api.internal
```

### 4. Run Troubleshooting Tools

```bash
# Monitor network continuously
./scripts/network_monitor.sh -v

# Debug connection issues
./scripts/debug_connection.sh ml-api.internal 8080

# Analyze latency
./scripts/analyze_latency.sh ml-api.internal
```

## Configuration Files

### SSH Hardening (`99-hardening.conf`)

**Features:**
- Password authentication disabled (key-only)
- Root login disabled
- Strong cryptographic algorithms
- Session timeout (15 minutes)
- Rate limiting (MaxAuthTries: 3)
- Comprehensive logging

**Installation:**
```bash
sudo cp config/99-hardening.conf /etc/ssh/sshd_config.d/
sudo sshd -t  # Test
sudo systemctl reload sshd
```

**Key Settings:**
- `PasswordAuthentication no` - Force key-based auth
- `PermitRootLogin no` - No root access via SSH
- `MaxAuthTries 3` - Limit brute-force attempts
- `ClientAliveInterval 300` - 5-minute keep-alive
- Strong ciphers: ChaCha20-Poly1305, AES-256-GCM

**Compliance:**
- CIS Benchmark aligned
- PCI-DSS compliant
- NIST 800-53 controls

### SSH Client Config (`ssh_config`)

**Features:**
- Connection multiplexing (reuse connections)
- Host-specific configurations
- Jump host support (ProxyJump)
- Port forwarding templates
- Strong algorithm preferences

**Configuration Highlights:**
```ini
# Global defaults
Host *
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 600

# GPU nodes
Host gpu-node-*
    User mluser
    IdentityFile ~/.ssh/ml_platform_key

# Via bastion
Host internal-*
    ProxyJump bastion
```

**Usage Examples:**
```bash
# Direct connection
ssh gpu-node-1

# Via bastion (automatic)
ssh internal-gpu-1

# Port forwarding
ssh -L 8888:localhost:8888 gpu-node-1

# Multiple connections (instant via multiplexing)
ssh gpu-node-1  # First connection (authenticates)
ssh gpu-node-1  # Subsequent (instant, reuses connection)
```

### Firewall Rules (`ufw_rules.txt`)

**Features:**
- Default deny incoming
- Service-specific rules
- Network segmentation (internal/external)
- Rate limiting for SSH
- Comprehensive documentation

**Protected Services:**
- SSH (22) - Rate limited
- ML APIs (8080, 8000, 5000)
- Monitoring (Prometheus: 9090, Grafana: 3000)
- Databases (5432, 3306, 27017) - Internal only
- Caching (Redis: 6379) - Internal only

**Network Segmentation:**
```bash
# Public services
sudo ufw allow 8080/tcp comment 'ML Model API'

# Internal services (database)
sudo ufw allow from 192.168.1.0/24 to any port 5432 comment 'PostgreSQL internal'

# Rate limiting (SSH brute-force protection)
sudo ufw limit 22/tcp
```

**Management:**
```bash
# Status
sudo ufw status verbose
sudo ufw status numbered

# Delete rule
sudo ufw delete <number>

# Disable temporarily
sudo ufw disable

# Reload
sudo ufw reload
```

### Internal DNS (`hosts`)

**Features:**
- Complete ML infrastructure hostnames
- Service aliases for convenience
- Categorized by service type
- Documentation and usage examples

**Services Defined:**
- GPU Training Nodes (192.168.1.101-114)
- ML API Servers (192.168.1.200-203)
- Model Serving (TorchServe, TensorFlow Serving)
- Feature Store (Redis cluster)
- Databases (PostgreSQL, MongoDB)
- Monitoring Stack (Prometheus, Grafana)
- Development Tools (JupyterHub, GitLab)

**Usage:**
```bash
# Add to /etc/hosts
sudo tee -a /etc/hosts < config/hosts

# Test resolution
ping ml-api.internal
getent hosts feature-store.internal
```

## Troubleshooting Scripts

### 1. Network Monitor (`network_monitor.sh`)

**Purpose:** Continuous monitoring of network performance and service availability.

**Features:**
- Interface statistics (RX/TX errors, dropped packets)
- Connection statistics (ESTABLISHED, TIME_WAIT)
- Bandwidth monitoring
- Service connectivity tests
- Latency measurements
- DNS health checks
- Alert on thresholds

**Usage:**
```bash
# Basic monitoring
./scripts/network_monitor.sh -v

# Custom interface and interval
./scripts/network_monitor.sh -i enp0s3 -t 30

# Custom services
./scripts/network_monitor.sh -s "api:8080 db:5432"

# Background monitoring
nohup ./scripts/network_monitor.sh -l /var/log/ml-network.log &
```

**Configuration:**
```bash
# Environment variables
export NETWORK_INTERFACE=eth0
export CHECK_INTERVAL=60
export ALERT_THRESHOLD_LATENCY=100
export ALERT_EMAIL=ops@company.com
```

**Output Example:**
```
[2025-01-31 14:30:00] [INFO] Network monitoring started
[2025-01-31 14:30:00] [INFO] Connections: EST=150, TW=50, CW=2
[2025-01-31 14:30:00] [INFO] Bandwidth: RX=15.30MB/s, TX=8.50MB/s
[2025-01-31 14:30:00] [INFO] Service ml-api.internal:8080: OK
[2025-01-31 14:30:00] [WARN] High latency to feature-store.internal: 120ms
```

### 2. Connection Debugger (`debug_connection.sh`)

**Purpose:** Systematic diagnosis of connection issues.

**Features:**
- DNS resolution verification
- ICMP ping test
- Port connectivity test
- Traceroute analysis
- Local firewall check
- Service listening verification
- MTU/packet size testing
- Recommendations

**Usage:**
```bash
./scripts/debug_connection.sh <host> <port>

# Examples
./scripts/debug_connection.sh ml-api.internal 8080
./scripts/debug_connection.sh 192.168.1.100 5432
```

**Diagnostic Steps:**
1. **DNS Resolution** - Verify hostname resolves to IP
2. **Ping Test** - Check ICMP connectivity
3. **Port Test** - Verify port is open
4. **Traceroute** - Identify routing path
5. **Firewall** - Check for blocking rules
6. **Service Check** - Verify service is listening (if local)
7. **MTU Test** - Check for fragmentation issues

**Output Example:**
```
═══════════════════════════════════════════════════════════════
  Connection Debugging: ml-api.internal:8080
═══════════════════════════════════════════════════════════════

[1] DNS Resolution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Resolved: ml-api.internal → 192.168.1.200

[2] Ping Test (ICMP)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Host is reachable via ICMP
rtt min/avg/max/mdev = 0.5/1.2/2.1/0.3 ms

[3] Port 8080 Test
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Port 8080 is open and accessible
```

### 3. Latency Analyzer (`analyze_latency.sh`)

**Purpose:** Detailed latency analysis with statistical metrics.

**Features:**
- RTT measurement (100 pings)
- Latency distribution (P50, P95, P99)
- Jitter analysis (standard deviation)
- Packet loss testing
- Per-hop latency (mtr)
- Stability over time
- Performance assessment
- Recommendations

**Usage:**
```bash
./scripts/analyze_latency.sh <host>

# Examples
./scripts/analyze_latency.sh ml-api.internal
./scripts/analyze_latency.sh 192.168.1.100
```

**Metrics Collected:**
- **Min/Max/Avg Latency** - Basic statistics
- **Percentiles** - P50, P95, P99 for SLA tracking
- **Standard Deviation** - Jitter measurement
- **Packet Loss** - Network reliability
- **Trend Analysis** - Latency stability over time

**Output Example:**
```
═══════════════════════════════════════════════════════════════
  Latency Analysis for ml-api.internal
═══════════════════════════════════════════════════════════════

[2] Latency Distribution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Min:  0.5 ms
P50:  1.2 ms
P95:  2.5 ms
P99:  3.8 ms
Max:  5.2 ms
Avg:  1.4 ms

Assessment:
✓ Excellent - Very low latency (<20ms)

[3] Jitter Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Standard Deviation: 0.8 ms
✓ Excellent - Very consistent latency
```

## Testing

### Network Configuration Testing

```bash
# Test SSH configuration
sudo sshd -t

# Test key authentication
ssh -i ~/.ssh/ml_platform_key localhost

# Test firewall rules
sudo ufw status
nc -zv localhost 8080  # Should succeed
nc -zv localhost 5432  # Should fail (blocked)

# Test hostname resolution
ping -c 2 ml-api.internal
getent hosts feature-store.internal
```

### Script Testing

```bash
# Test network monitor (5 minute run)
timeout 300 ./scripts/network_monitor.sh -v -t 10

# Test connection debugger
./scripts/debug_connection.sh localhost 22
./scripts/debug_connection.sh google.com 443

# Test latency analyzer
./scripts/analyze_latency.sh localhost
```

## Common Network Issues and Solutions

### Issue 1: "Connection Refused"

**Symptoms:**
```
Connection refused
telnet: Unable to connect to remote host: Connection refused
```

**Diagnosis:**
```bash
# Check if service is running
sudo systemctl status <service>

# Check if port is listening
sudo ss -tlnp | grep <port>

# Check firewall
sudo ufw status | grep <port>
```

**Solutions:**
1. Start the service: `sudo systemctl start <service>`
2. Add firewall rule: `sudo ufw allow <port>/tcp`
3. Verify service is binding to correct interface

### Issue 2: "Connection Timeout"

**Symptoms:**
```
Connection timed out
ssh: connect to host <hostname> port 22: Connection timed out
```

**Diagnosis:**
```bash
# Check network connectivity
ping <hostname>

# Check route
traceroute <hostname>

# Debug connection
./scripts/debug_connection.sh <hostname> <port>
```

**Solutions:**
1. Check firewall on target: `sudo ufw status`
2. Check network path: `traceroute <hostname>`
3. Verify port number is correct
4. Check for intermediate firewalls/security groups

### Issue 3: "Permission Denied (publickey)"

**Symptoms:**
```
Permission denied (publickey).
```

**Diagnosis:**
```bash
# Check SSH key
ssh -vv -i ~/.ssh/key user@host

# Verify key is added
ssh-add -L

# Check authorized_keys on server
cat ~/.ssh/authorized_keys
```

**Solutions:**
1. Verify correct key: `ssh -i /path/to/key user@host`
2. Check key permissions: `chmod 600 ~/.ssh/id_rsa`
3. Add key to authorized_keys on server
4. Check SELinux: `restorecon -R ~/.ssh`

### Issue 4: "High Latency"

**Symptoms:**
- Slow API responses
- Training jobs timing out
- Model inference delays

**Diagnosis:**
```bash
# Analyze latency
./scripts/analyze_latency.sh <target>

# Check network path
mtr <target>

# Monitor bandwidth
./scripts/network_monitor.sh -v
```

**Solutions:**
1. Check for packet loss (fix network issues)
2. Identify bottleneck hop (traceroute/mtr)
3. Reduce geographic distance (use CDN/edge)
4. Increase bandwidth
5. Implement caching
6. Use compression

### Issue 5: "DNS Resolution Failed"

**Symptoms:**
```
Could not resolve hostname
Name or service not known
```

**Diagnosis:**
```bash
# Check DNS configuration
cat /etc/resolv.conf

# Test DNS
nslookup <hostname>
dig <hostname>

# Check /etc/hosts
cat /etc/hosts | grep <hostname>
```

**Solutions:**
1. Add to /etc/hosts: `echo "IP hostname" | sudo tee -a /etc/hosts`
2. Fix DNS servers in /etc/resolv.conf
3. Check internal DNS server accessibility
4. Use IP address instead of hostname

## Best Practices

### SSH Security

1. **Never use password authentication in production**
   - Use key-based auth only
   - Disable PasswordAuthentication

2. **Rotate SSH keys regularly**
   - Every 6-12 months
   - After staff changes

3. **Use different keys for different environments**
   - Dev, staging, production
   - Personal vs. shared access

4. **Monitor SSH access**
   - Review /var/log/auth.log
   - Alert on failed attempts
   - Use fail2ban

5. **Limit SSH access**
   - Use bastion hosts
   - Restrict to VPN/specific IPs
   - Use AllowUsers/AllowGroups

### Firewall Management

1. **Default deny**
   - Block all incoming by default
   - Only allow necessary services

2. **Network segmentation**
   - Databases: internal only
   - APIs: specific sources only
   - Admin access: VPN only

3. **Document all rules**
   - Comment each rule with purpose
   - Track business justification
   - Regular audits

4. **Test before applying**
   - Use staging environment
   - Keep console access ready
   - Test in maintenance window

5. **Monitor firewall logs**
   - Review blocked connections
   - Identify attack patterns
   - Alert on unusual activity

### Network Monitoring

1. **Continuous monitoring**
   - Always-on network monitoring
   - Alert on thresholds
   - Track trends over time

2. **Baseline performance**
   - Know normal latency/bandwidth
   - Set realistic alert thresholds
   - Monitor SLA compliance

3. **Proactive debugging**
   - Regular health checks
   - Test critical paths
   - Document known issues

4. **Capacity planning**
   - Monitor bandwidth usage
   - Plan for growth
   - Identify bottlenecks early

5. **Incident response**
   - Runbooks for common issues
   - Escalation procedures
   - Post-mortem analysis

## Production Deployment

### Pre-deployment Checklist

- [ ] SSH configuration tested in staging
- [ ] Firewall rules documented and reviewed
- [ ] Backup of current network configuration
- [ ] Console/IPMI access verified
- [ ] Monitoring scripts tested
- [ ] Team trained on new configuration
- [ ] Rollback plan prepared
- [ ] Maintenance window scheduled

### Deployment Steps

1. **Backup current configuration**
   ```bash
   sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup
   sudo ufw status numbered > /tmp/ufw_backup.txt
   ```

2. **Deploy SSH hardening**
   ```bash
   sudo cp config/99-hardening.conf /etc/ssh/sshd_config.d/
   sudo sshd -t
   sudo systemctl reload sshd
   ```

3. **Test SSH access** (from another terminal)
   ```bash
   ssh -i ~/.ssh/key user@host
   ```

4. **Deploy firewall rules** (carefully!)
   ```bash
   sudo bash config/ufw_rules.txt
   ```

5. **Deploy monitoring**
   ```bash
   nohup ./scripts/network_monitor.sh &
   ```

6. **Verify everything works**
   ```bash
   ./scripts/debug_connection.sh localhost 22
   ```

### Post-deployment Monitoring

**First 24 Hours:**
- Monitor SSH access logs
- Check firewall blocks
- Verify service accessibility
- Test from multiple sources

**First Week:**
- Review monitoring data
- Tune alert thresholds
- Address any issues
- Update documentation

**Ongoing:**
- Monthly configuration review
- Quarterly security audit
- Annual penetration testing
- Regular training updates

## Troubleshooting Workflow

```
Connection Issue
       |
       v
[1] DNS Resolution
   OK?  → [2] Ping Test
   NO → Check /etc/hosts, DNS servers
       |
       v
[2] Ping Test
   OK?  → [3] Port Test
   NO → Check routing, firewall
       |
       v
[3] Port Test
   OK?  → [4] Application Test
   NO → Check service, firewall
       |
       v
[4] Application Test
   OK?  → Monitor Performance
   NO → Check logs, service health
```

## Additional Resources

- [SSH Security Best Practices](https://www.ssh.com/academy/ssh/security)
- [UFW Essentials](https://www.digitalocean.com/community/tutorials/ufw-essentials-common-firewall-rules-and-commands)
- [Linux Networking](https://www.redhat.com/sysadmin/linux-networking-commands)
- [tcpdump Tutorial](https://danielmiessler.com/study/tcpdump/)
- [mtr Guide](https://www.linode.com/docs/guides/diagnosing-network-issues-with-mtr/)

## Support

For additional help:
- Review `docs/NETWORK_GUIDE.md` for detailed explanations
- Check script help: `./scripts/network_monitor.sh --help`
- Consult module instructor or TA

## License

Part of AI Infrastructure Junior Engineer curriculum.
