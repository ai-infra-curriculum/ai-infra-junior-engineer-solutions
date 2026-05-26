# Exercise 07: Linux Troubleshooting — Solution

## What the exercise asked for

Diagnose common Linux problems on ML servers: disk full,
process hung, port collision, network can't reach service,
permission denied, out-of-memory.

## The triage flow

When something's broken on a Linux box:

1. **What's the symptom?** (Error message? Slow? Silent?)
2. **What changed recently?** (Check `journalctl --since '1 hour ago'`.)
3. **Check the resources** (`top`, `df`, `free`).
4. **Check the logs** (`journalctl -u <service>` for relevant services).
5. **Check the network** (`ss`, `ping`, `curl`).
6. **Form a hypothesis. Test it.**

## Disk full

```bash
# How full?
df -h

# Where's the space going?
sudo du -h --max-depth=1 / 2>/dev/null | sort -h

# Or
ncdu /

# Common culprits on ML servers:
ls -lh /var/log/   # rotated logs
ls -lh ~/.cache/   # pip cache, huggingface cache
ls -lh /var/lib/docker/   # docker images + volumes
docker system df  # docker-specific breakdown
docker system prune -a  # nuclear cleanup

# Big files only:
find / -type f -size +1G 2>/dev/null
```

ML servers fill up because of: pip caches, HuggingFace caches,
old container images, training checkpoints, log files. Clean
those up.

## Process hung / runaway

```bash
# Top CPU consumers
top -o %CPU

# Top memory consumers (alternative)
ps aux --sort=-%mem | head

# A specific process
ps aux | grep python
top -p <PID>

# What is it doing?
strace -p <PID> -e trace=network
lsof -p <PID> | head    # open files, sockets

# Kill (escalating force)
kill <PID>             # SIGTERM, polite
kill -9 <PID>          # SIGKILL, immediate (avoid if possible)
```

For an ML server: training job that's stuck in I/O wait
usually means the data loader is slow. Look at `iotop`.

## Port already in use

```bash
# Who has port 8080?
sudo ss -lnp | grep :8080
# or
sudo lsof -i :8080

# Kill if needed, or move your service to a different port.
```

## Network can't reach service

```bash
# Is the service running?
ss -tlnp | grep :8080

# Local connectivity
curl -v http://localhost:8080/health

# From another host
nc -vz <host> 8080

# DNS
dig <hostname>
nslookup <hostname>

# Routing
ip route
traceroute <host>

# Firewall (don't forget about iptables / ufw)
sudo iptables -L
sudo ufw status
```

## Permission denied

```bash
# Who owns the file?
ls -la <file>

# What permissions?
stat <file>

# Am I the right user?
whoami
id

# Effective vs. real (sudo, capabilities)
ps -o pid,ruser,euser,comm -p <PID>

# Fix
sudo chown <user>:<group> <file>
sudo chmod 644 <file>
```

For containers, also check:
- SELinux / AppArmor restrictions.
- Container's user (often non-root).
- Mount permissions.

## Out of memory

```bash
# Memory state
free -h
cat /proc/meminfo | head -10

# Recent OOM events
dmesg -T | grep -i "out of memory"
journalctl -k --since "1 hour ago" | grep -i oom

# Per-process memory
ps aux --sort=-%mem | head -10
```

For ML workloads:
- Training jobs that hit OOM usually need smaller batch size
  or gradient accumulation.
- Inference OOM: model loaded multiple times (worker
  concurrency × model size > RAM).

## A useful one-liner per problem

```bash
# Disk full
df -h && sudo du -h --max-depth=1 / 2>/dev/null | sort -h | tail -10

# Top CPU + memory in one shot
ps aux --sort=-%cpu | head -5; ps aux --sort=-%mem | head -5

# Open ports
sudo ss -tlnp

# Recent errors in journald
journalctl -p err --since "1 hour ago"

# Network reachability
nc -vz <host> <port>
```

## Common mistakes

- Reaching for `sudo` before understanding why permission was
  denied.
- Killing processes with `-9` before trying `-15` (skips
  cleanup).
- "Reboot fixes it" without understanding what got fixed.
- Not checking `journalctl` first.

## Cross-references

- Exercise prompt:
  `ai-infra-junior-engineer-learning/lessons/mod-002-linux-essentials/exercises/exercise-07-troubleshooting.md`
- mod-009-monitoring-basics covers observability that catches
  these problems before they're 3 AM pages.
