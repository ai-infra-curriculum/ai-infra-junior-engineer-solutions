# Exercise 08: System Automation - Reflection Questions

## Part 1: Backup Strategies

### Q1: Why is automation critical for ML infrastructure backup, and what risks does manual backup introduce?

**Answer:**

Automation is critical for ML infrastructure backups for several reasons:

**Consistency and Reliability:**
- Manual backups depend on human memory and discipline
- Automation ensures backups run on schedule, even during holidays/weekends
- Eliminates "I forgot to run the backup" scenarios
- Consistent execution reduces variability in backup quality

**Risk Mitigation:**
- Manual backups are often skipped during busy periods
- Critical model versions may be lost if not backed up immediately
- Human error (wrong directory, incomplete backup) is common
- Automation provides audit trails and verification

**Scale:**
- ML infrastructure generates models continuously (training pipelines)
- Manual backup doesn't scale with model proliferation
- Automated systems can back up hundreds of models consistently

**Risks of Manual Backup:**
1. **Forgotten Backups**: Most common failure mode
2. **Incomplete Backups**: Missing files, wrong compression
3. **No Verification**: Manual processes rarely verify backup integrity
4. **Poor Documentation**: "Which backup has the v2.3 model?"
5. **Knowledge Silos**: Only one person knows the backup procedure
6. **Time Constraints**: Backups skipped when team is busy

**Real-world Example:**
A startup lost 2 months of model training work because backups were manual and skipped during a sprint. An automated system with S3 sync and checksums would have prevented this.

### Q2: Explain the 3-2-1 backup rule and how this exercise implements it.

**Answer:**

**The 3-2-1 Backup Rule:**
- **3** copies of your data (1 primary + 2 backups)
- **2** different storage media types
- **1** off-site/off-premises copy

**Implementation in This Exercise:**

1. **3 Copies:**
   - **Copy 1**: Original models in `/opt/ml/models` (primary)
   - **Copy 2**: Local backup in `/backup/models` (first backup)
   - **Copy 3**: S3 bucket with `--s3-sync` (second backup)

2. **2 Storage Media:**
   - **Local disk**: Fast access, immediate recovery
   - **S3/Cloud**: Durable storage (99.999999999% durability), survives hardware failure

3. **1 Off-site:**
   - **S3 bucket**: Different physical location, survives datacenter disasters
   - Optional: S3 cross-region replication for geo-redundancy

**Code Implementation:**
```bash
# Local backup (Copy 2, Media 1)
tar -czf "$BACKUP_PATH" "$MODEL_DIR"

# Verify integrity
sha256sum "$BACKUP_PATH" >> checksums.txt

# Off-site backup (Copy 3, Media 2)
if [ -n "$S3_BUCKET" ]; then
    aws s3 cp "$BACKUP_PATH" "s3://${S3_BUCKET}/models/" \
        --storage-class GLACIER  # Cost-effective archival
fi
```

**Why This Matters:**
- **Local failure** (disk crash): Recover from S3
- **Site failure** (datacenter fire): Recover from S3 in different region
- **Corruption**: Multiple copies increase recovery options
- **Ransomware**: Off-site backups survive encryption attacks

**Enhanced Implementation:**
For production, add:
- S3 versioning (multiple versions of same file)
- S3 lifecycle policies (auto-archive to Glacier)
- Cross-region replication (survive regional outages)

### Q3: What are the trade-offs between compression levels (1-9) for model backups?

**Answer:**

**Compression Levels Comparison:**

| Level | Speed | Size Reduction | CPU Usage | Best For |
|-------|-------|----------------|-----------|----------|
| 1 | Fastest | ~50-60% | Low | Large models, fast backups needed |
| 6 (default) | Balanced | ~65-75% | Medium | General purpose |
| 9 | Slowest | ~70-80% | High | Small models, storage critical |

**Detailed Analysis:**

**Level 1 (Fast):**
```bash
# Example: 10GB model
# Compression time: ~2 minutes
# Compressed size: ~5GB
# CPU usage: 1 core @ 50%

COMPRESSION_LEVEL=1 ./backup_models.sh
```
- **Pros**: Fast backups, low CPU impact
- **Cons**: Larger backup files, higher storage costs
- **Use case**: Frequent backups (hourly), large models (>10GB)

**Level 6 (Balanced - Default):**
```bash
# Example: 10GB model
# Compression time: ~5 minutes
# Compressed size: ~3GB
# CPU usage: 1 core @ 80%

COMPRESSION_LEVEL=6 ./backup_models.sh
```
- **Pros**: Good balance of speed and size
- **Cons**: Moderate CPU usage
- **Use case**: Daily backups, most scenarios

**Level 9 (Maximum):**
```bash
# Example: 10GB model
# Compression time: ~15 minutes
# Compressed size: ~2.5GB
# CPU usage: 1 core @ 100%

COMPRESSION_LEVEL=9 ./backup_models.sh
```
- **Pros**: Maximum space savings
- **Cons**: Slow, high CPU usage
- **Use case**: Archival, storage-constrained environments

**Model-Specific Considerations:**

**Text-based models** (weights, JSON):
- Compress well (70-90% reduction)
- Level 6-9 recommended
- Example: BERT model (500MB → 100MB at level 9)

**Binary models** (PyTorch .pt, TensorFlow .pb):
- Already compressed internally
- Level 1-3 sufficient
- Example: Pre-quantized model (minimal additional compression)

**Checkpoint directories** (many small files):
- tar overhead significant
- Level 6 balanced
- Consider separate compression for large tensors

**Real-world Example:**
```bash
# Daily backups: Fast turnaround
COMPRESSION_LEVEL=3 ./backup_models.sh

# Weekly archival: Maximum compression
COMPRESSION_LEVEL=9 ./backup_models.sh

# Critical pre-deployment: Fast, verified backup
COMPRESSION_LEVEL=1 VERIFY_BACKUP=true ./backup_models.sh
```

**Resource Impact:**
```bash
# Monitor compression resource usage
time COMPRESSION_LEVEL=1 tar -cz1f test.tar.gz models/  # ~2 min
time COMPRESSION_LEVEL=9 tar -cz9f test.tar.gz models/  # ~15 min

# Parallel compression (faster, more CPU)
tar -I 'pigz -p 4' -cf test.tar.gz models/  # 4 cores
```

**Recommendation:**
- **Default**: Level 6 (balanced)
- **Large models + SSD**: Level 1-3 (I/O bound anyway)
- **Small models + HDD**: Level 9 (compression faster than I/O)
- **S3 upload**: Level 9 (network is bottleneck, save bandwidth)

## Part 2: GPU Monitoring

### Q4: Why is GPU monitoring critical for ML infrastructure, and what are the key metrics to track?

**Answer:**

**Why GPU Monitoring is Critical:**

1. **Cost Management:**
   - GPUs are expensive ($2-10/hour on cloud, $10K-50K for hardware)
   - Underutilization wastes money
   - Monitoring identifies idle GPUs that can be freed

2. **Performance Optimization:**
   - Low GPU utilization indicates bottlenecks (data loading, preprocessing)
   - Memory patterns reveal inefficient batch sizes
   - Temperature spikes indicate thermal throttling

3. **Hardware Protection:**
   - Overheating shortens GPU lifespan
   - Power limit violations indicate insufficient PSU
   - Memory errors indicate failing hardware

4. **Debugging Training Issues:**
   - OOM errors correlate with memory usage spikes
   - Slow training correlates with GPU utilization < 80%
   - Multi-GPU training should show balanced load

**Key Metrics to Track:**

**1. Temperature (°C):**
```bash
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader
```
- **Normal**: 60-75°C under load
- **Warning**: 75-85°C (thermal throttling risk)
- **Critical**: >85°C (hardware damage risk)
- **Action**: Improve cooling, reduce workload

**2. GPU Utilization (%):**
```bash
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
```
- **Good**: 80-100% (GPU fully utilized)
- **Moderate**: 50-80% (some inefficiency)
- **Poor**: <50% (major bottleneck)
- **Causes**: Data loading, CPU preprocessing, I/O wait

**3. Memory Utilization (%):**
```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```
- **Normal**: 70-90% (efficient batch size)
- **Warning**: >95% (OOM risk)
- **Low**: <50% (batch size too small)
- **Action**: Adjust batch size, enable gradient checkpointing

**4. Power Draw (W):**
```bash
nvidia-smi --query-gpu=power.draw,power.limit --format=csv,noheader
```
- **Normal**: 80-100% of power limit
- **Low**: <50% (GPU not fully loaded)
- **Critical**: Hitting power limit (throttling)
- **Action**: Increase power limit if safe

**5. Memory Errors:**
```bash
nvidia-smi --query-gpu=ecc.errors.corrected.volatile.total --format=csv,noheader
```
- **Acceptable**: 0 errors
- **Warning**: >0 corrected errors (investigate)
- **Critical**: Uncorrected errors (replace GPU)

**6. Process Information:**
```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```
- Track which processes use GPUs
- Identify memory leaks (growing usage)
- Detect zombie processes

**Implementation in Exercise:**
```bash
# monitor_gpus.sh tracks all critical metrics
nvidia-smi --query-gpu=\
index,name,temperature.gpu,utilization.gpu,utilization.memory,\
memory.used,memory.total,power.draw,power.limit \
--format=csv,noheader,nounits

# Alert on thresholds
if [ "$temp" -gt "$TEMP_THRESHOLD" ]; then
    send_alert "GPU $idx temperature is ${temp}°C"
fi
```

**Historical Metrics (CSV Logging):**
```bash
# GPU metrics logged to CSV for analysis
timestamp,gpu_id,temp,util_gpu,util_mem,mem_used,mem_total,power

# Analyze trends
# - Average utilization over 24 hours
# - Temperature patterns (time of day)
# - Memory usage growth (leak detection)
```

**Advanced Monitoring:**
- **nvtop**: Real-time TUI dashboard
- **Prometheus exporter**: Time-series metrics
- **Grafana dashboards**: Visualization
- **DCGM**: NVIDIA datacenter GPU manager

**Real-world Impact:**
- Detected GPU underutilization (30%) → Fixed data loading → 3x faster training
- Temperature alerts prevented hardware failure → $10K GPU saved
- Memory monitoring identified leak → 2-day debugging session avoided

### Q5: Explain the purpose of alert cooldown and how it prevents alert fatigue.

**Answer:**

**Alert Fatigue:**
A phenomenon where excessive alerts cause responders to:
- Ignore alerts (desensitization)
- Disable alerting systems
- Miss critical issues in noise
- Experience stress and burnout

**The Problem Without Cooldown:**

```bash
# GPU temperature spikes to 85°C
# Without cooldown: Alert every 5 minutes

14:00 - ALERT: GPU 0 temperature 85°C
14:05 - ALERT: GPU 0 temperature 86°C  # Same issue
14:10 - ALERT: GPU 0 temperature 85°C  # Still same issue
14:15 - ALERT: GPU 0 temperature 87°C  # Getting annoying
14:20 - ALERT: GPU 0 temperature 85°C  # NOW IGNORED

# Result: 5 alerts in 20 minutes for the same problem
# Responder disables alerts → misses critical GPU failure at 15:00
```

**With Alert Cooldown:**

```bash
# Same scenario with 1-hour cooldown

14:00 - ALERT: GPU 0 temperature 85°C  [SENT]
14:05 - Alert suppressed (cooldown: 55 minutes remaining)
14:10 - Alert suppressed (cooldown: 50 minutes remaining)
15:00 - Alert suppressed (cooldown: 0 minutes remaining)
15:01 - ALERT: GPU 0 temperature still high  [SENT]

# Result: 2 alerts → acknowledged and investigated
```

**Implementation in monitor_gpus.sh:**

```bash
ALERT_COOLDOWN="${ALERT_COOLDOWN:-3600}"  # 1 hour default

should_alert() {
    local alert_key="$1"
    local state_file="${STATE_DIR}/${alert_key}"

    # Check if alert was sent recently
    if [ ! -f "$state_file" ]; then
        return 0  # No previous alert, send now
    fi

    local last_alert=$(cat "$state_file")
    local current_time=$(date +%s)
    local time_diff=$((current_time - last_alert))

    if [ $time_diff -gt $ALERT_COOLDOWN ]; then
        return 0  # Cooldown expired, send alert
    fi

    return 1  # Still in cooldown, suppress alert
}

# Usage
if [ "$temp" -gt "$TEMP_THRESHOLD" ]; then
    if should_alert "gpu${idx}_temp"; then
        send_alert "GPU $idx temperature is ${temp}°C"
        echo "$(date +%s)" > "${STATE_DIR}/gpu${idx}_temp"
    fi
fi
```

**How It Works:**

1. **First Alert**: No state file exists → alert sent → timestamp saved
2. **Subsequent Checks**: State file exists → compare timestamps
3. **Within Cooldown**: Suppress alert, log to monitoring
4. **After Cooldown**: Send new alert if issue persists

**Cooldown Strategies:**

**Per-Issue Cooldown:**
```bash
# Different cooldowns for different issues
alert_cooldown "gpu0_temp" 3600      # GPU temp: 1 hour
alert_cooldown "gpu0_memory" 600     # Memory: 10 minutes
alert_cooldown "gpu0_error" 0        # Errors: immediate
```

**Escalating Cooldown:**
```bash
# Increase cooldown after each alert
first_alert:  cooldown = 10 minutes
second_alert: cooldown = 30 minutes
third_alert:  cooldown = 1 hour
fourth_alert: cooldown = 24 hours
```

**Severity-based Cooldown:**
```bash
if [ "$temp" -lt 85 ]; then
    COOLDOWN=3600    # Warning: 1 hour
elif [ "$temp" -lt 90 ]; then
    COOLDOWN=1800    # Critical: 30 minutes
else
    COOLDOWN=300     # Emergency: 5 minutes
fi
```

**Benefits:**

1. **Reduces Noise**: One alert per hour vs. 12 alerts per hour
2. **Maintains Urgency**: Alerts remain actionable
3. **Prevents Desensitization**: Responders take alerts seriously
4. **Saves Resources**: Fewer emails, webhooks, pages
5. **Allows Investigation**: Time to diagnose before next alert

**Trade-offs:**

**Too Short** (< 15 minutes):
- Still produces alert fatigue
- Doesn't allow time for investigation
- Responders feel pressured

**Too Long** (> 4 hours):
- May miss escalating issues
- Doesn't re-alert if first alert missed
- Insufficient for critical systems

**Recommended Cooldowns:**

| Severity | Cooldown | Rationale |
|----------|----------|-----------|
| Info | 4 hours | FYI, not urgent |
| Warning | 1 hour | Time to investigate |
| Critical | 15-30 min | Regular updates |
| Emergency | 5 min | Continuous monitoring |

**Real-world Example:**
```bash
# Before cooldown implementation:
# - 500 alerts/day
# - 90% duplicate
# - Team disabled Slack integration
# - Missed critical disk failure

# After 1-hour cooldown:
# - 50 alerts/day
# - 95% actionable
# - Team responsive
# - Caught GPU failure early
```

**Advanced: State Management:**
```bash
# Track alert state for smart alerting
STATE_DIR=/var/lib/gpu-monitor

# Alert states
- NEW: Issue just detected
- ACKNOWLEDGED: Alert sent, cooldown active
- RESOLVED: Issue cleared, reset state
- RECURRING: Issue keeps reappearing

# Escalation after N occurrences
if [ "$(cat ${STATE_DIR}/gpu0_temp_count)" -gt 5 ]; then
    # Same issue 5 times → escalate to on-call
    send_page "GPU 0 persistent overheating"
fi
```

**Integration with On-call Systems:**
- PagerDuty: De-duplication key + cooldown
- Opsgenie: Alert policies with quiet hours
- Slack: Thread replies instead of new messages

## Part 3: Log Rotation

### Q6: Compare `copytruncate` vs `postrotate` with service reload. When should each be used?

**Answer:**

**Key Difference:**
- **copytruncate**: Copy log, then truncate original file (file handle stays open)
- **postrotate + reload**: Move log, then tell application to reopen new file

**Detailed Comparison:**

### copytruncate Method

**How It Works:**
```bash
/var/log/app.log {
    daily
    rotate 7
    copytruncate
}
```

1. Copy `app.log` to `app.log.1`
2. Truncate `app.log` to 0 bytes
3. Application continues writing to same file handle

**Diagram:**
```
Before rotation:
  app → /var/log/app.log (inode 12345, 500MB)

During copytruncate:
  app → /var/log/app.log (inode 12345, 500MB)  [COPY]
  Copy to app.log.1
  Truncate app.log → 0 bytes
  app → /var/log/app.log (inode 12345, 0 bytes)  [SAME FILE]

After rotation:
  app → /var/log/app.log (inode 12345, growing from 0)
  app.log.1 (500MB archived)
```

**Pros:**
- No application changes required
- Works with apps that keep file open
- No service disruption
- Simple configuration

**Cons:**
- **Data loss window**: Logs written during copy might be missed
- **Disk space**: Briefly needs 2x space (original + copy)
- **I/O intensive**: Copying large logs is slow
- **Not atomic**: Corruption possible if crash during copy

**Use Cases:**
- Applications that can't reopen logs (legacy apps)
- Scripts that keep file handles open (long-running)
- Monitoring scripts (like our GPU monitor)
- Cases where log continuity is critical

**Example from Exercise:**
```bash
# GPU Monitor Log
/var/log/gpu-monitor.log {
    daily
    copytruncate  # Script keeps file open
    minsize 10K
}

# Why: monitor_gpus.sh redirects stdout to file
# File handle stays open for entire execution
```

### postrotate + Service Reload Method

**How It Works:**
```bash
/var/log/app.log {
    daily
    rotate 7
    create 0640 app app
    postrotate
        systemctl reload app.service
    endscript
}
```

1. Move `app.log` to `app.log.1` (rename, instant)
2. Create new empty `app.log`
3. Run postrotate script (reload app)
4. Application reopens log file

**Diagram:**
```
Before rotation:
  app → /var/log/app.log (inode 12345, 500MB)

During rotation:
  mv app.log → app.log.1 (inode 12345, 500MB)
  app → [writing to inode 12345] (now called app.log.1)
  touch app.log (inode 67890, 0 bytes)
  reload app
  app → /var/log/app.log (inode 67890, 0 bytes)

After rotation:
  app → /var/log/app.log (inode 67890, growing)
  app.log.1 (inode 12345, 500MB archived)
```

**Pros:**
- **No data loss**: Atomic rename
- **Efficient**: Move is instant (no copying)
- **Clean separation**: Old log completely separate
- **Industry standard**: Most production apps use this

**Cons:**
- Requires application support (SIGHUP, USR1, or reload)
- Brief service interruption during reload
- Application must handle log reopening
- More complex configuration

**Use Cases:**
- Production web servers (nginx, apache)
- Application servers (gunicorn, uwsgi)
- System daemons (syslog, systemd-journald)
- Microservices with log reopening

**Example from Exercise:**
```bash
# ML API Logs
/var/log/ml-api/*.log {
    daily
    rotate 14
    create 0640 mluser mluser
    sharedscripts
    postrotate
        systemctl reload ml-api.service > /dev/null 2>&1 || true
    endscript
}

# Why: ML API service supports SIGHUP
# Reload is fast and reopens logs cleanly
```

**Signal-based Reloading:**
```bash
# Common patterns
postrotate
    # Systemd service reload
    systemctl reload app.service

    # PID-based signal
    kill -HUP $(cat /var/run/app.pid) || true

    # Multiple services
    systemctl reload nginx.service gunicorn.service

    # Graceful reload (USR1)
    kill -USR1 $(cat /var/run/nginx.pid)
endscript
```

### Decision Matrix

| Factor | copytruncate | postrotate + reload |
|--------|--------------|---------------------|
| **Data loss risk** | Small window | None (atomic) |
| **Disk space** | 2x during copy | Minimal |
| **Performance** | Slow (I/O bound) | Fast (rename) |
| **Application support** | Not required | Required |
| **Service disruption** | None | Brief reload |
| **Production use** | Rare | Common |

### Real-world Examples

**Scenario 1: Long-running Script**
```bash
#!/bin/bash
# Script that runs for hours
exec >> /var/log/my-script.log 2>&1

while true; do
    # Long-running work
    sleep 60
done

# Logrotate config: MUST use copytruncate
# Script can't reopen log mid-execution
```

**Scenario 2: Web Application**
```bash
# Flask/Django/Rails app
# Opens log file at startup
# Supports SIGHUP to reopen logs

# Logrotate config: postrotate + reload
# Clean, atomic, no data loss
```

**Scenario 3: Systemd Service**
```bash
# Modern systemd service
# Logs to journald (no files)
# No logrotate needed!

# But if logging to files:
[Service]
ExecReload=/bin/kill -HUP $MAINPID

# Logrotate config: postrotate + reload
```

**Hybrid Approach:**
```bash
# Critical logs: postrotate + reload
/var/log/ml-api/*.log {
    postrotate
        systemctl reload ml-api.service
    endscript
}

# Monitoring logs: copytruncate
/var/log/gpu-monitor.log {
    copytruncate
}

# Why: API logs are critical (no loss acceptable)
# Monitoring logs are best-effort (loss acceptable)
```

### Best Practices

1. **Prefer postrotate + reload** for production apps
2. **Use copytruncate** only when necessary
3. **Test reload** before deploying logrotate config
4. **Handle errors** in postrotate scripts:
   ```bash
   postrotate
       systemctl reload app.service > /dev/null 2>&1 || true
   endscript
   ```
5. **Use sharedscripts** to reload once for multiple logs:
   ```bash
   /var/log/ml-api/*.log {
       sharedscripts  # Reload once, not per log
       postrotate
           systemctl reload ml-api.service
       endscript
   }
   ```

### Performance Comparison

```bash
# Test file: 1GB log

# copytruncate
time: 5 seconds (I/O bound)
space: 2GB (original + copy)
data loss: ~100ms window

# postrotate + reload
time: <100ms (rename + reload)
space: 1GB (just the log)
data loss: 0ms (atomic)
```

**Recommendation for Exercise:**
- ML API logs: postrotate + reload (critical)
- GPU monitoring: copytruncate (script keeps file open)
- Health checks: copytruncate (same reason)
- Training logs: postrotate if possible (large files)

### Q7: Why is `delaycompress` recommended, and what problem does it solve?

**Answer:**

**The Problem: Compressed Logs During Active Analysis**

**Without delaycompress:**
```bash
/var/log/app.log {
    daily
    rotate 7
    compress
}
```

**Timeline:**
```
Day 1, 02:00: Rotation happens
  app.log → app.log.1
  app.log.1 → app.log.1.gz  [COMPRESSED IMMEDIATELY]

Day 1, 09:00: Engineer needs to check yesterday's logs
  cat /var/log/app.log.1  # ERROR: File not found
  zcat /var/log/app.log.1.gz  # Works, but slower
```

**With delaycompress:**
```bash
/var/log/app.log {
    daily
    rotate 7
    compress
    delaycompress  # Delay compression until next rotation
}
```

**Timeline:**
```
Day 1, 02:00: First rotation
  app.log → app.log.1  [UNCOMPRESSED]
  (no app.log.1.gz yet)

Day 1, 09:00: Engineer checks yesterday's logs
  cat /var/log/app.log.1  # Works! Fast access

Day 2, 02:00: Second rotation
  app.log → app.log.1  [NEW, UNCOMPRESSED]
  app.log.1 → app.log.2.gz  [COMPRESSED NOW]
```

**Key Insight:** Most recent rotated log stays uncompressed for 24 hours, allowing fast access.

**Problems Solved:**

**1. Fast Log Analysis:**
```bash
# Without delaycompress (compressed immediately)
time zcat app.log.1.gz | grep ERROR
# Real: 5.2 seconds (decompression overhead)

# With delaycompress (uncompressed for 24h)
time cat app.log.1 | grep ERROR
# Real: 0.3 seconds (direct file read)
```

**2. Tool Compatibility:**
```bash
# Many tools expect uncompressed logs
tail -f app.log.1           # Doesn't work with .gz
less app.log.1              # Works without .gz
awk '/pattern/' app.log.1   # Works without .gz

# With .gz, need special handling
zcat app.log.1.gz | less
zcat app.log.1.gz | awk '/pattern/'
```

**3. Script Simplicity:**
```bash
# Debugging script that checks last 24h
for log in app.log app.log.1; do
    grep ERROR "$log"  # Works with delaycompress
done

# Without delaycompress, need:
grep ERROR app.log
zcat app.log.1.gz | grep ERROR  # Different command!
```

**4. Incident Response:**
```bash
# Production incident at 10:00
# Need to check logs from 02:00-10:00

# With delaycompress:
cat /var/log/app.log.1    # Yesterday's full day
cat /var/log/app.log      # Today so far
# Fast, immediate access during incident

# Without delaycompress:
zcat /var/log/app.log.1.gz    # Slow, delays investigation
cat /var/log/app.log
```

**Implementation in Exercise:**

```bash
# All log configurations use delaycompress
/var/log/ml-api/*.log {
    daily
    rotate 14
    compress
    delaycompress  # Keep most recent rotation uncompressed
}

/var/log/gpu-monitor.log {
    daily
    rotate 30
    compress
    delaycompress
}
```

**File State Over Time:**

```
Day 0:
  app.log (active, growing)

Day 1, 02:00 (first rotation):
  app.log (new, empty)
  app.log.1 (yesterday, UNCOMPRESSED)

Day 2, 02:00 (second rotation):
  app.log (new, empty)
  app.log.1 (yesterday, UNCOMPRESSED)
  app.log.2.gz (2 days ago, COMPRESSED)

Day 3, 02:00 (third rotation):
  app.log (new, empty)
  app.log.1 (yesterday, UNCOMPRESSED)
  app.log.2.gz (2 days ago, COMPRESSED)
  app.log.3.gz (3 days ago, COMPRESSED)
```

**Key Pattern:** Only the most recent rotated log (app.log.1) is uncompressed.

**Space vs. Performance Trade-off:**

**Without delaycompress:**
- **Space**: Maximum savings (all rotated logs compressed)
- **Performance**: Slow access to most recent rotated log
- **Total space**: e.g., 100MB + 7×10MB = 170MB

**With delaycompress:**
- **Space**: One more uncompressed log (~10% more space)
- **Performance**: Fast access to most recent rotated log
- **Total space**: e.g., 100MB + 100MB + 6×10MB = 260MB

**Space Impact Example:**
```bash
# Without delaycompress
app.log      100MB (active)
app.log.1.gz  10MB (compressed)
app.log.2.gz  10MB
...
app.log.7.gz  10MB
Total: 170MB

# With delaycompress
app.log      100MB (active)
app.log.1    100MB (uncompressed, recent)
app.log.2.gz  10MB (compressed)
...
app.log.7.gz  10MB
Total: 260MB

# Extra cost: 90MB (one uncompressed log)
# Benefit: Fast access during debugging
```

**When NOT to Use delaycompress:**

**1. Storage-Constrained Systems:**
```bash
# Embedded systems, IoT devices
/var/log/app.log {
    daily
    rotate 3      # Short retention
    compress
    # NO delaycompress - need every byte
}
```

**2. Archive-Only Logs:**
```bash
# Logs only accessed for compliance audits
/var/log/audit.log {
    monthly
    rotate 120    # 10 years
    compress
    # NO delaycompress - never accessed immediately
}
```

**3. Very Large Logs:**
```bash
# 10GB daily logs on system with 100GB disk
/var/log/bigapp.log {
    daily
    rotate 7
    compress
    # NO delaycompress - can't spare 10GB
}
```

**Best Practices:**

1. **Use delaycompress by default** for application logs
2. **Skip delaycompress** for archival/compliance logs
3. **Combine with minsize** to avoid compressing tiny logs:
   ```bash
   compress
   delaycompress
   minsize 1M  # Don't rotate/compress logs < 1MB
   ```

4. **Monitor disk space** impact of delaycompress
5. **Document** why delaycompress is used

**Real-world Example:**

**Before delaycompress:**
- Incident at 03:00 AM
- Engineer needs to check logs from 23:00-03:00
- app.log.1.gz takes 30 seconds to search (large compressed file)
- Incident resolution delayed by slow log access

**After delaycompress:**
- Same incident at 03:00 AM
- app.log.1 is uncompressed
- grep through logs in 2 seconds
- Faster diagnosis, faster resolution

**Monitoring Integration:**
```bash
# Log monitoring tools expect uncompressed recent logs
logwatch /var/log/app.log.1      # Works with delaycompress
fail2ban-client set jail unban   # Parses uncompressed logs faster

# Without delaycompress, tools need compression support
# or wait until logs are rotated again (48h delay)
```

**Recommendation for Exercise:**
Use `delaycompress` for all logs except:
- GPU metrics CSV (set to `nocompress` anyway)
- Archival logs older than 30 days

## Part 4: Cron vs Systemd Timers

### Q8: What are the advantages of systemd timers over cron jobs? When would you still use cron?

**Answer:**

**Systemd Timers Advantages:**

**1. Integrated Logging:**
```bash
# Cron: Logs scattered or missing
# Output goes to email or manual redirection
0 2 * * * /path/script.sh >> /var/log/cron.log 2>&1

# View logs
tail -f /var/log/cron.log  # Manual log file

# Systemd: Integrated with journald
journalctl -u ml-backup.service -f
journalctl -u ml-backup.service --since "1 hour ago"
journalctl -u ml-backup.service -n 50 --no-pager

# Structured logging with metadata
# - Timestamp, hostname, service name
# - Exit codes, runtime, errors
```

**2. Resource Limits:**
```bash
# Cron: No resource control
# Script can consume all CPU/memory

# Systemd: Cgroup-based limits
[Service]
MemoryLimit=2G        # Max 2GB RAM
CPUQuota=50%          # Max 50% of one CPU
IOWeight=100          # I/O priority
TasksMax=50           # Max 50 processes
```

**3. Dependencies:**
```bash
# Cron: No dependency management
# Script runs even if required services are down

# Systemd: Service dependencies
[Unit]
After=network.target           # Wait for network
Requires=docker.service        # Need Docker running
BindsTo=postgresql.service     # If Postgres dies, stop this

[Service]
ExecStart=/path/script.sh
```

**4. Missed Run Handling:**
```bash
# Cron: Skips missed runs
# If system down at 02:00, backup never runs

# Systemd: Persistent timers catch up
[Timer]
OnCalendar=daily
Persistent=true    # Run if system was down
```

**Example scenario:**
```
Monday 02:00: Scheduled backup
Monday 01:00-08:00: Server maintenance (offline)

Cron: Backup missed, no notification
Systemd: Backup runs at 08:01 (after boot)
```

**5. Randomized Delays:**
```bash
# Cron: All servers run at exact time
# 100 servers backup at 02:00 → S3 throttled

# Systemd: Randomized start
[Timer]
OnCalendar=02:00
RandomizedDelaySec=600    # Spread over 10 minutes

# Server 1: 02:02
# Server 2: 02:07
# Server 3: 02:09
# Result: Load spread, no S3 throttling
```

**6. Flexible Scheduling:**
```bash
# Cron: Fixed time expressions
0 2 * * 0    # Sunday 2 AM (simple)

# Systemd: Calendar expressions
OnCalendar=Mon..Fri 09:00         # Weekdays 9 AM
OnCalendar=*-*-01 00:00           # First of month
OnCalendar=Sat *-*-1..7 18:00     # First Saturday 6 PM
OnCalendar=*:0/5                  # Every 5 minutes

# Multiple schedules
OnCalendar=Mon 09:00
OnCalendar=Fri 17:00
```

**7. Service Management:**
```bash
# Cron: Run-and-forget
# No status, no control

# Systemd: Full service control
systemctl start ml-backup.service    # Manual trigger
systemctl stop ml-backup.service     # Cancel running job
systemctl status ml-backup.service   # Check status
systemctl list-timers                # List all schedules

# Timer control
systemctl enable ml-backup.timer     # Start on boot
systemctl disable ml-backup.timer    # Don't start on boot
systemctl restart ml-backup.timer    # Reschedule
```

**8. Exit Code Handling:**
```bash
# Cron: Manual exit code checking
# No built-in retry or failure handling

# Systemd: Automatic retry
[Service]
Restart=on-failure
RestartSec=300             # Wait 5 min before retry
StartLimitBurst=3          # Max 3 retries
StartLimitIntervalSec=600  # Within 10 minutes
```

**9. Environment Management:**
```bash
# Cron: Limited environment
# PATH, HOME, USER set
# Manual environment loading required

# Systemd: Rich environment control
[Service]
Environment="MODEL_DIR=/opt/ml/models"
Environment="BACKUP_DIR=/backup/models"
EnvironmentFile=/etc/ml-backup.conf  # Load from file
```

**10. Monitoring Integration:**
```bash
# Cron: Manual monitoring setup
# Check log files, email alerts

# Systemd: Built-in monitoring
[Unit]
OnFailure=alert@%n.service    # Trigger alert on failure

# Status queries
systemctl is-active ml-backup.service  # running/failed
systemctl is-enabled ml-backup.timer   # enabled/disabled

# Prometheus integration
node_exporter includes systemd metrics
```

**When to Still Use Cron:**

**1. Simple, User-level Tasks:**
```bash
# User crontab for personal tasks
# No need for system-level complexity
crontab -e

# Backup personal files
0 2 * * * rsync ~/docs ~/backup/
```

**2. Legacy Systems:**
```bash
# Old systems without systemd
# RHEL/CentOS 6, Debian 7, etc.
# Cron is the only option
```

**3. Quick, Throwaway Tasks:**
```bash
# Temporary automation
# */5 * * * * curl -s https://api.example.com/health
# Easier than creating systemd files
```

**4. Cross-platform Scripts:**
```bash
# Scripts that run on multiple OSes
# macOS, BSD, old Linux → all have cron
# Systemd is Linux-specific (and recent)
```

**5. Minimal Systems:**
```bash
# Embedded systems, containers
# Alpine Linux uses OpenRC, not systemd
# Cron is lighter, more universal
```

**6. Shared Hosting:**
```bash
# No root access, can't create systemd units
# User crontab is only option
```

**Comparison Table:**

| Feature | Cron | Systemd Timers |
|---------|------|----------------|
| **Logging** | Manual (email, files) | journald (integrated) |
| **Resource limits** | None | cgroups (CPU, mem, I/O) |
| **Dependencies** | None | Full systemd integration |
| **Missed runs** | Skipped | Persistent option |
| **Randomization** | No | RandomizedDelaySec |
| **Calendar expressions** | Limited | Rich (Mon..Fri, etc.) |
| **Service control** | None | Full (start, stop, status) |
| **Retry on failure** | Manual | Automatic |
| **Environment** | Limited | Rich (files, variables) |
| **Monitoring** | Manual | Built-in (metrics, alerts) |
| **Complexity** | Low | Higher (2 files) |
| **Portability** | High (universal) | Low (Linux only) |
| **User-level** | Easy (crontab -e) | Harder (systemctl --user) |

**Real-world Decision:**

**Use Systemd Timers When:**
- Production ML infrastructure (this exercise)
- Need resource limits (prevent runaway jobs)
- Want integrated logging and monitoring
- System has systemd (modern Linux)
- Need dependency management
- Want automatic retry on failure

**Use Cron When:**
- Quick personal tasks
- Legacy systems without systemd
- Cross-platform scripts
- Shared hosting / no root access
- Minimal systems (embedded, containers)
- Simple, throwaway automation

**Migration Path:**
```bash
# Start with cron for prototyping
crontab -e
0 2 * * * /path/script.sh

# Move to systemd for production
# Better logging, resource control, monitoring
systemctl enable --now script.timer
```

**Exercise Recommendation:**

This exercise demonstrates **both approaches**:
- **config/crontab.example**: Learn cron syntax
- **config/systemd-timers.md**: Production-ready systemd

**For production ML infrastructure:**
→ Use **systemd timers** (better in every way except simplicity)

## Part 5: Production Deployment

### Q9: What are the key considerations when deploying automation to production?

**Answer:**

**1. Testing and Validation**

**Dry-run Mode:**
```bash
# Test before running in production
./backup_models.sh --dry-run
./cleanup_ml_artifacts.sh --dry-run

# Verify logic without making changes
# Check paths, permissions, thresholds
```

**Staging Environment:**
```bash
# Test in staging first
# Same configuration as production
# Smaller dataset

# Staging
MODEL_DIR=/staging/ml/models \
BACKUP_DIR=/staging/backup \
./backup_models.sh

# After 1 week of successful runs → Production
```

**Gradual Rollout:**
```bash
# Week 1: Single server, manual trigger
./backups/backup_models.sh

# Week 2: Single server, scheduled (daily)
crontab -e  # Add job

# Week 3: 3 servers, scheduled
ansible-playbook deploy-automation.yml --limit "ml-server-[1:3]"

# Week 4: All servers
ansible-playbook deploy-automation.yml
```

**2. Monitoring and Alerting**

**Success Monitoring:**
```bash
# Don't just monitor failures!
# Absence of success is also a problem

# Heartbeat monitoring
0 3 * * * /opt/automation/backup.sh && \
          curl https://hc-ping.com/YOUR-UUID

# If no ping received for 25 hours → alert
```

**Log Monitoring:**
```bash
# Automated log analysis
# Alert on unexpected patterns

# Monitor for errors
journalctl -u ml-backup.service | grep -i error

# Alert on high error rate
if [ $(journalctl -u ml-backup --since "1 hour ago" | grep -c ERROR) -gt 10 ]; then
    send_alert "High error rate in backup service"
fi
```

**Metrics Collection:**
```bash
# Track automation metrics
# - Runtime duration
# - Backup sizes
# - Cleanup freed space
# - GPU temperature trends

# Export to Prometheus
backup_duration_seconds{job="backup"} 234
backup_size_bytes{job="backup"} 5368709120
cleanup_freed_bytes{job="cleanup"} 10737418240
```

**3. Resource Management**

**Timing Optimization:**
```bash
# Schedule during low-traffic hours
# Avoid impacting production workloads

# Good: 2 AM (low traffic)
0 2 * * * /opt/automation/backup.sh

# Bad: 2 PM (peak traffic)
# Backup I/O competes with training jobs
```

**Resource Limits:**
```bash
# Prevent automation from impacting production

# Systemd limits
[Service]
CPUQuota=30%       # Max 30% of CPU
MemoryLimit=2G     # Max 2GB RAM
IOWeight=100       # Low I/O priority

# Nice/ionice for cron
0 2 * * * nice -n 19 ionice -c 3 /opt/automation/backup.sh
```

**Concurrent Job Prevention:**
```bash
# Prevent overlapping runs
# Long backup still running when next backup starts

# Flock (file locking)
0 2 * * * flock -n /var/lock/backup.lock /opt/automation/backup.sh

# Systemd (built-in)
# Won't start new instance if previous still running
```

**4. Error Handling and Recovery**

**Graceful Degradation:**
```bash
# Don't fail entire automation if one part fails
# Continue processing other tasks

# Bad
set -e  # Exit on any error
backup_models.sh    # If this fails...
cleanup_artifacts.sh  # ...this never runs

# Good
set -euo pipefail
if ! backup_models.sh; then
    log ERROR "Backup failed, but continuing"
fi
cleanup_artifacts.sh  # Runs regardless
```

**Retry Logic:**
```bash
# Transient failures should retry
# Network blips, temporary resource constraints

retry_with_backoff() {
    local max_attempts=3
    local attempt=1
    local delay=60

    while [ $attempt -le $max_attempts ]; do
        if "$@"; then
            return 0
        fi

        log WARN "Attempt $attempt failed, retrying in ${delay}s"
        sleep $delay
        delay=$((delay * 2))  # Exponential backoff
        attempt=$((attempt + 1))
    done

    return 1
}

# Usage
retry_with_backoff aws s3 cp backup.tar.gz s3://bucket/
```

**Idempotency:**
```bash
# Safe to run multiple times
# Second run has same effect as first

# Example: Backup script
# - Check if today's backup exists
# - If yes, skip (or verify)
# - If no, create backup

BACKUP_NAME="models_backup_$(date +%Y%m%d).tar.gz"
if [ -f "$BACKUP_DIR/$BACKUP_NAME" ]; then
    log INFO "Backup already exists, verifying"
    verify_backup "$BACKUP_DIR/$BACKUP_NAME"
    exit 0
fi
```

**5. Security Considerations**

**Principle of Least Privilege:**
```bash
# Don't run as root unless necessary
# Create dedicated user

# Create user
sudo useradd -r -s /bin/bash mlautomation

# Set ownership
sudo chown -R mlautomation:mlautomation /opt/ml-automation

# Run as user
[Service]
User=mlautomation
Group=mlautomation
```

**Secrets Management:**
```bash
# Don't hardcode credentials
# Use secret management systems

# Bad
S3_KEY="AKIAIOSFODNN7EXAMPLE"  # Hardcoded in script

# Good - Environment file
EnvironmentFile=/etc/ml-automation/secrets.conf
# secrets.conf (mode 0600, root-owned)
S3_KEY=AKIAIOSFODNN7EXAMPLE

# Better - Secret manager
S3_KEY=$(aws secretsmanager get-secret-value --secret-id ml/s3key --query SecretString --output text)

# Best - IAM roles
# No credentials needed, use instance role
```

**File Permissions:**
```bash
# Restrict access to automation files

# Scripts: rwxr-x---
chmod 750 /opt/ml-automation/*.sh

# Configs: rw-r-----
chmod 640 /opt/ml-automation/config/*

# Secrets: rw-------
chmod 600 /etc/ml-automation/secrets.conf
```

**6. Documentation and Runbooks**

**Operational Runbooks:**
```markdown
# Backup Failure Runbook

## Symptoms
- No backup created in /backup/models for >24 hours
- Alert: "Backup heartbeat missing"

## Diagnosis
1. Check service status: `systemctl status ml-backup.service`
2. Check logs: `journalctl -u ml-backup.service --since "24 hours ago"`
3. Check disk space: `df -h /backup`

## Resolution
1. If disk full: Run cleanup, expand disk
2. If S3 error: Check IAM permissions
3. If script error: Review logs, fix script

## Prevention
- Monitor disk space
- Set up disk space alerts (>85%)
- Test backups weekly
```

**Change Management:**
```bash
# Document all changes
# Version control for automation scripts

# Git repository
/opt/ml-automation/
├── .git/
├── CHANGELOG.md
├── scripts/
└── config/

# Track changes
git log --oneline backup_models.sh

# Rollback if needed
git revert HEAD
systemctl restart ml-backup.service
```

**7. Backup and Disaster Recovery**

**Backup the Automation:**
```bash
# Automation scripts are critical infrastructure
# Back them up too!

# Git repository (pushed to remote)
git push origin main

# Config backup
tar -czf /backup/automation-config-$(date +%Y%m%d).tar.gz \
    /opt/ml-automation \
    /etc/systemd/system/ml-*.{service,timer} \
    /etc/logrotate.d/ml-infrastructure
```

**Recovery Testing:**
```bash
# Regularly test recovery procedures

# Quarterly disaster recovery drill:
# 1. Delete automation scripts
# 2. Restore from backup
# 3. Verify functionality
# 4. Document time to recover (RTO)
```

**8. Capacity Planning**

**Growth Projections:**
```bash
# Track resource usage over time
# Plan for growth

# Backup sizes
2025-01: 500GB
2025-02: 600GB
2025-03: 750GB
# Projection: 1.2TB by 2025-06

# Action: Expand backup storage before hitting limit
```

**Retention Policy Adjustment:**
```bash
# Balance retention vs. cost

# Initial: 30 days full retention
RETENTION_DAYS=30

# 6 months later: Too expensive
# New policy:
# - 7 days: Full daily backups
# - 30 days: Weekly backups
# - 90 days: Monthly backups
```

**9. Communication and Coordination**

**Maintenance Windows:**
```bash
# Communicate automation changes
# Coordinate with team

# Email team before changes
Subject: [Maintenance] ML Backup Schedule Change
Body: Backup window moving from 2-3 AM to 1-2 AM on 2025-02-01
```

**Change Approvals:**
```bash
# Production changes require approval

# Change request:
- What: Update backup script to use S3 Glacier
- Why: Reduce storage costs by 50%
- When: 2025-02-01 02:00 AM
- Rollback: Revert to previous script version
- Risk: Low (tested in staging)
```

**10. Performance Optimization**

**Incremental Backups:**
```bash
# Full backup daily: Slow, expensive

# Better: Incremental backups
# - Full backup weekly
# - Incremental backup daily

# Day 0: Full backup (10GB)
# Day 1: Incremental (+500MB, only changes)
# Day 2: Incremental (+300MB)
# ...
# Day 7: Full backup (12GB)
```

**Parallel Processing:**
```bash
# Speed up automation with parallelism

# Sequential (slow)
backup_models.sh      # 10 min
backup_datasets.sh    # 15 min
backup_configs.sh     # 2 min
# Total: 27 min

# Parallel (fast)
backup_models.sh &
backup_datasets.sh &
backup_configs.sh &
wait
# Total: 15 min (limited by slowest)
```

**Pre-deployment Checklist:**

- [ ] Tested in staging environment
- [ ] Dry-run mode tested in production
- [ ] Monitoring and alerts configured
- [ ] Resource limits set (CPU, memory, I/O)
- [ ] Timing optimized (low-traffic hours)
- [ ] Error handling and retry logic implemented
- [ ] Idempotent (safe to re-run)
- [ ] Security reviewed (least privilege, no hardcoded secrets)
- [ ] Documentation complete (runbooks, change logs)
- [ ] Backup and recovery tested
- [ ] Capacity planning done
- [ ] Team notified and trained
- [ ] Rollback plan prepared
- [ ] Success metrics defined

**Post-deployment Monitoring:**

**First 24 Hours:**
- Watch logs continuously
- Verify first run succeeds
- Check resource usage

**First Week:**
- Review daily metrics
- Fine-tune thresholds
- Address any issues

**First Month:**
- Validate reliability (>99% success rate)
- Optimize performance
- Update documentation with learnings

**Ongoing:**
- Monthly review of metrics
- Quarterly disaster recovery test
- Annual capacity planning review

### Q10: How would you test this automation in a production environment safely?

**Answer:**

**1. Phased Testing Approach**

**Phase 1: Local Development**
```bash
# Test on laptop/workstation
# Use test data and directories

# Setup test environment
mkdir -p /tmp/test-{models,backup,experiments}
dd if=/dev/zero of=/tmp/test-models/model.pt bs=1M count=100

# Test each script
MODEL_DIR=/tmp/test-models \
BACKUP_DIR=/tmp/test-backup \
./backups/backup_models.sh --dry-run

# Verify output
ls -lh /tmp/test-backup/
cat /tmp/test-backup/checksums.txt
```

**Phase 2: Staging Environment**
```bash
# Mirror of production
# Same OS, same configs, smaller dataset

# Deploy to staging
ansible-playbook -i staging deploy-automation.yml

# Run manually
ssh staging-ml-01 'sudo /opt/ml-automation/backups/backup_models.sh'

# Check results
ssh staging-ml-01 'ls -lh /backup/models/'

# Schedule with short interval
# Run every hour for 24 hours
ssh staging-ml-01 'crontab -e'
# 0 * * * * /opt/ml-automation/backups/backup_models.sh
```

**Phase 3: Production Shadow Mode**
```bash
# Deploy to production, but don't modify critical data
# Read-only testing

# Test with dry-run
PROD_SERVER=prod-ml-01
ssh $PROD_SERVER '/opt/ml-automation/backups/backup_models.sh --dry-run'

# Write to test location
ssh $PROD_SERVER 'BACKUP_DIR=/tmp/test-backup /opt/ml-automation/backups/backup_models.sh'

# Verify no impact on production
# - Check CPU/memory usage during run
# - Verify production services unaffected
```

**Phase 4: Canary Deployment**
```bash
# Enable on one production server

# Server selection
CANARY_SERVER=prod-ml-01  # Lowest traffic server

# Deploy automation
ansible-playbook -i production deploy-automation.yml \
    --limit "$CANARY_SERVER"

# Enable timers
ssh $CANARY_SERVER 'sudo systemctl enable --now ml-backup.timer'

# Monitor for 1 week
# - Check logs daily
# - Verify backups created
# - Monitor resource usage
# - Check for alerts

# Success criteria:
# - 100% successful runs
# - <5% resource usage
# - No production impact
# - Backups verified restorable
```

**Phase 5: Progressive Rollout**
```bash
# Expand to more servers gradually

# Week 1: 1 server (canary)
# Week 2: 3 servers (10%)
# Week 3: 10 servers (30%)
# Week 4: All servers (100%)

# Rollout script
SERVERS=(prod-ml-{01..30})
BATCH_SIZE=3
for ((i=0; i<${#SERVERS[@]}; i+=BATCH_SIZE)); do
    batch=("${SERVERS[@]:i:BATCH_SIZE}")
    echo "Deploying to: ${batch[*]}"

    ansible-playbook -i production deploy-automation.yml \
        --limit "$(IFS=,; echo "${batch[*]}")"

    # Monitor for 1 day
    sleep $((24 * 3600))

    # Check for issues
    if ansible-playbook -i production check-automation.yml; then
        echo "Batch succeeded, continuing"
    else
        echo "Batch failed, rolling back"
        exit 1
    fi
done
```

**2. Safety Mechanisms**

**Dry-run Mode:**
```bash
# Implement --dry-run for all scripts

# backup_models.sh
if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY RUN] Would create: $BACKUP_PATH"
    echo "[DRY RUN] Would calculate checksum"
    echo "[DRY RUN] Would upload to S3"
    exit 0
fi

# Test in production without risk
./backup_models.sh --dry-run  # Safe, no changes

# cleanup_ml_artifacts.sh
if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY RUN] Would delete: $(find $EXPERIMENTS_DIR -mtime +30)"
    echo "[DRY RUN] Total space to free: 10GB"
    exit 0
fi
```

**Circuit Breakers:**
```bash
# Prevent catastrophic failures
# Stop if something looks wrong

# Example: Cleanup safety check
OLD_FILES=$(find $EXPERIMENTS_DIR -mtime +$RETENTION_DAYS)
FILE_COUNT=$(echo "$OLD_FILES" | wc -l)

# Safety: Don't delete more than 1000 files at once
if [ "$FILE_COUNT" -gt 1000 ]; then
    log ERROR "Safety check: Would delete $FILE_COUNT files (max 1000)"
    log ERROR "This seems wrong, aborting"
    send_alert "Cleanup safety check triggered"
    exit 1
fi

# Safety: Don't free more than 500GB at once
SPACE_TO_FREE=$(du -sb $OLD_FILES | awk '{sum+=$1} END {print sum}')
SPACE_TO_FREE_GB=$((SPACE_TO_FREE / 1024 / 1024 / 1024))

if [ "$SPACE_TO_FREE_GB" -gt 500 ]; then
    log ERROR "Safety check: Would free ${SPACE_TO_FREE_GB}GB (max 500GB)"
    exit 1
fi
```

**Confirmation Prompts:**
```bash
# Require confirmation for destructive operations
# (Only in interactive mode, skip in cron)

if [ -t 0 ] && [ "$FORCE" != "true" ]; then
    echo "About to delete $FILE_COUNT files freeing ${SPACE_GB}GB"
    read -p "Continue? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Aborted"
        exit 0
    fi
fi
```

**Rate Limiting:**
```bash
# Slow down automation to reduce impact

# Delete files slowly (not all at once)
while IFS= read -r file; do
    rm -f "$file"
    sleep 0.1  # 10 files/second (gentle on I/O)
done < <(find $EXPERIMENTS_DIR -mtime +30)
```

**3. Monitoring During Deployment**

**Real-time Metrics:**
```bash
# Monitor key metrics during rollout

# CPU usage
watch -n 5 'top -bn1 | head -20'

# Disk I/O
iostat -x 5

# Backup progress
watch -n 10 'ls -lht /backup/models/ | head -5'

# Service logs
journalctl -u ml-backup.service -f

# Application impact
# Monitor training job performance
# Check API response times
# Verify GPU utilization unchanged
```

**Automated Health Checks:**
```bash
# Run health checks after each automation run

# check-automation.yml (Ansible)
- name: Check automation health
  hosts: all
  tasks:
    - name: Verify backup created
      stat:
        path: "/backup/models/models_backup_{{ ansible_date_time.date }}.tar.gz"
      register: backup_file
      failed_when: not backup_file.stat.exists

    - name: Check service status
      systemd:
        name: ml-backup.timer
        state: started
      register: service
      failed_when: service.status != 'active'

    - name: Verify disk space
      shell: df / | tail -1 | awk '{print $5}' | sed 's/%//'
      register: disk_usage
      failed_when: disk_usage.stdout|int > 90

    - name: Check for errors in logs
      shell: journalctl -u ml-backup --since "1 hour ago" | grep -c ERROR || true
      register: error_count
      failed_when: error_count.stdout|int > 5
```

**4. Rollback Plan**

**Automated Rollback:**
```bash
# Detect issues and rollback automatically

# deploy-automation.yml (Ansible)
- name: Deploy automation
  hosts: all
  tasks:
    - name: Backup current scripts
      archive:
        path: /opt/ml-automation
        dest: /tmp/automation-backup-{{ ansible_date_time.epoch }}.tar.gz

    - name: Deploy new version
      copy:
        src: "{{ item }}"
        dest: /opt/ml-automation/
      with_fileglob:
        - "scripts/*"
      notify: restart timers

    - name: Wait for first run
      wait_for:
        timeout: 3600  # Wait up to 1 hour

    - name: Check if automation succeeded
      shell: systemctl is-failed ml-backup.service
      register: service_status
      failed_when: false
      changed_when: false

    - name: Rollback on failure
      when: service_status.rc != 0
      block:
        - name: Extract backup
          unarchive:
            src: /tmp/automation-backup-{{ ansible_date_time.epoch }}.tar.gz
            dest: /

        - name: Restart timers
          systemd:
            name: "{{ item }}"
            state: restarted
          loop:
            - ml-backup.timer
            - gpu-monitor.timer

        - name: Alert on rollback
          command: >
            curl -X POST https://hooks.slack.com/... \
            -d '{"text":"Automation deployment failed, rolled back"}'
```

**Manual Rollback:**
```bash
# Quick rollback procedure

# 1. Stop timers
ssh prod-ml-01 'sudo systemctl stop ml-backup.timer'

# 2. Restore previous version
ssh prod-ml-01 'cd /opt/ml-automation && git reset --hard HEAD~1'

# 3. Restart timers
ssh prod-ml-01 'sudo systemctl start ml-backup.timer'

# 4. Verify
ssh prod-ml-01 'systemctl status ml-backup.timer'
```

**5. Testing Specific Scenarios**

**Failure Scenarios:**
```bash
# Test how automation handles failures

# Scenario 1: Disk full
# Create test environment with limited disk
truncate -s 1G /tmp/test-disk.img
mkfs.ext4 /tmp/test-disk.img
mkdir -p /tmp/test-backup
sudo mount /tmp/test-disk.img /tmp/test-backup

# Run backup (should fail gracefully)
BACKUP_DIR=/tmp/test-backup ./backups/backup_models.sh

# Verify error handling
# - Check error logged
# - Check alert sent
# - Check exit code != 0

# Scenario 2: S3 unavailable
# Use invalid S3 bucket or disable network
S3_BUCKET=invalid-bucket-name ./backups/backup_models.sh --s3-sync

# Verify retry logic
# - 3 attempts with exponential backoff
# - Local backup still created
# - Alert sent after failure

# Scenario 3: GPU not available
# Disable GPU or run without nvidia-smi
PATH=/usr/bin ./monitoring/monitor_gpus.sh  # nvidia-smi not in PATH

# Verify graceful degradation
# - Logs "GPU not available"
# - Doesn't fail health check
# - Continues monitoring other metrics
```

**Load Testing:**
```bash
# Test automation under load

# Scenario: Run automation while training
# Start training job
ssh prod-ml-01 'python train.py &'

# Run backup
ssh prod-ml-01 '/opt/ml-automation/backups/backup_models.sh'

# Monitor impact
# - Training speed (should be <5% slower)
# - GPU utilization (should stay >90%)
# - Backup time (may be slower, but completes)

# Scenario: Multiple automations simultaneously
# Run all maintenance tasks at once
ssh prod-ml-01 '/opt/ml-automation/run_all_maintenance.sh'

# Monitor resource usage
# - CPU <50% (enforced by systemd limits)
# - Memory <2GB per task
# - I/O doesn't block training
```

**Recovery Testing:**
```bash
# Test backup restoration

# Create backup
./backups/backup_models.sh

# Simulate disaster (delete models)
rm -rf /opt/ml/models/*

# Restore from backup
LATEST_BACKUP=$(ls -t /backup/models/models_backup_*.tar.gz | head -1)
tar -xzf "$LATEST_BACKUP" -C /opt/ml/

# Verify restoration
# - All models present
# - Checksums match
# - Training can resume

# Measure recovery time
# RTO (Recovery Time Objective): <1 hour
# RPO (Recovery Point Objective): <24 hours
```

**6. Progressive Feature Enablement**

**Feature Flags:**
```bash
# Enable features gradually

# Week 1: Backups only (low risk)
RUN_HEALTH_CHECK=true \
RUN_GPU_MONITOR=false \
RUN_BACKUP=true \
RUN_CLEANUP=false \
./run_all_maintenance.sh

# Week 2: Add monitoring (moderate risk)
RUN_GPU_MONITOR=true

# Week 3: Add cleanup (high risk - destructive)
RUN_CLEANUP=true
EXPERIMENTS_RETENTION=90  # Conservative initially

# Week 4: Optimize retention
EXPERIMENTS_RETENTION=30  # Normal retention
```

**7. Success Criteria**

**Define clear success metrics before deployment:**

**Reliability:**
- 99%+ successful runs over 1 week
- <1 unplanned alert per week
- 0 data loss incidents

**Performance:**
- Backup completes in <30 minutes
- Automation uses <5% CPU during peak hours
- No impact on training job performance

**Resource Usage:**
- Disk usage stays <85%
- Backups consume <500GB storage
- Network usage <10% of bandwidth

**Operational:**
- 0 manual interventions required
- Recovery tested successfully
- Documentation complete and accurate

**If any criteria not met:** Pause rollout, investigate, fix, restart from Phase 1.

**8. Production Testing Checklist**

Before enabling automation in production:

- [ ] Tested in local development environment
- [ ] Tested in staging environment (1 week)
- [ ] Dry-run mode tested in production
- [ ] Safety mechanisms implemented (circuit breakers, rate limiting)
- [ ] Monitoring and alerts configured
- [ ] Rollback plan prepared and tested
- [ ] Team trained on automation behavior
- [ ] Documentation complete (runbooks, troubleshooting)
- [ ] Success criteria defined
- [ ] Failure scenarios tested
- [ ] Load testing completed
- [ ] Recovery testing completed
- [ ] Canary server selected and deployed
- [ ] Post-deployment monitoring plan ready
- [ ] On-call rotation notified
- [ ] Change management approval obtained

Only proceed to production after all items checked off.

## Conclusion

This exercise demonstrates production-ready ML infrastructure automation covering:
- Robust backup strategies with verification
- Comprehensive GPU health monitoring with alerting
- Intelligent log rotation for multiple log types
- Automated cleanup with safety mechanisms
- System health validation
- Task orchestration with error handling

All implemented with modern best practices:
- Dry-run modes for safe testing
- Configurable thresholds via environment variables
- Multiple scheduling options (cron and systemd)
- Integrated logging and monitoring
- Resource limits and safety checks
- Comprehensive documentation

These skills are essential for managing production ML infrastructure at scale.
