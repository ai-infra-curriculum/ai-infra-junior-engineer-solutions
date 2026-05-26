# Exercise 08: Linux System Automation — Solution

## What the exercise asked for

Automate routine ML-server operations: scheduled jobs (cron /
systemd timers), shell scripts that survive errors, basic
configuration management, and the boundary with proper
automation tools.

## Cron vs. systemd timers

Both schedule recurring tasks. Modern preference: systemd
timers (better logging, more flexible).

### Cron

```bash
# Edit your crontab
crontab -e

# Format: minute hour day month weekday command
# Run model retraining daily at 2 AM
0 2 * * * /opt/ml/retrain.sh >> /var/log/retrain.log 2>&1

# Run cleanup every hour
0 * * * * /opt/ml/cleanup-stale.sh

# Cron quirks:
#  - Minimal PATH; set explicit absolute paths.
#  - HOME=/root by default; export needed env vars.
#  - Errors go to email (if configured) and stderr; redirect.
```

### systemd timer

```ini
# /etc/systemd/system/ml-retrain.service
[Unit]
Description=ML model retrain

[Service]
Type=oneshot
User=ml-runner
WorkingDirectory=/opt/ml
ExecStart=/opt/ml/retrain.sh
```

```ini
# /etc/systemd/system/ml-retrain.timer
[Unit]
Description=Daily ML retrain

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ml-retrain.timer

# Check status
systemctl status ml-retrain.timer
systemctl list-timers --all

# View logs
journalctl -u ml-retrain.service --since "1 day ago"
```

systemd advantages: integrated logging, missed-run recovery
(`Persistent=true`), randomized delays
(`RandomizedDelaySec=300`), dependency management.

## Robust shell scripts

The bash defaults are unhelpful. Always start with:

```bash
#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
```

- `set -e`: exit on first error.
- `set -u`: error on unset variables.
- `set -o pipefail`: pipeline fails if any command fails.
- `IFS`: prevents word-splitting surprises with spaces.

### Sample robust ML runner

```bash
#!/usr/bin/env bash
set -euo pipefail

readonly LOG_FILE="/var/log/ml-retrain.log"
readonly LOCK_FILE="/var/run/ml-retrain.lock"

# Logging helper
log() {
    echo "[$(date -Iseconds)] $*" | tee -a "$LOG_FILE"
}

# Lock to prevent concurrent runs
if [[ -f "$LOCK_FILE" ]]; then
    log "ERROR: lock file exists, previous run may be in progress"
    exit 1
fi
touch "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"; log "Cleaned up lock"' EXIT

log "Starting ML retrain"

# Activate venv, run training, capture exit code
source /opt/ml/.venv/bin/activate
if python /opt/ml/train.py --config /opt/ml/configs/prod.yaml; then
    log "Training succeeded"
else
    log "ERROR: training failed (exit $?)"
    # Notify on-call here (e.g., curl to PagerDuty webhook)
    exit 1
fi

log "Done"
```

## Idempotent operations

Scripts that run repeatedly should be safe to run repeatedly.

```bash
# Bad: errors if directory exists
mkdir /opt/ml/models

# Good: idempotent
mkdir -p /opt/ml/models

# Bad: appends every run
echo "export ML_HOME=/opt/ml" >> ~/.bashrc

# Good: only adds if not present
grep -q "ML_HOME" ~/.bashrc || echo "export ML_HOME=/opt/ml" >> ~/.bashrc
```

## Where shell ends and proper tools begin

| Task | Shell appropriate? | Better tool |
|---|---|---|
| Run a single ML training job | ✓ | n/a |
| Schedule a recurring task | ✓ | systemd timers |
| Manage 5 servers | ✓ | n/a |
| Manage 50 servers | ✗ | Ansible / Terraform |
| Manage cluster state | ✗ | Kubernetes |
| Build a model pipeline | ✗ | Airflow / Prefect |
| Manage secrets | ✗ | Vault / cloud secret manager |
| Provision infrastructure | ✗ | Terraform |

Shell + cron is appropriate at small scale. Past that, the
operational cost of shell automation exceeds learning a real
tool.

## Common mistakes

- No `set -euo pipefail` (silent failures).
- Hardcoded paths that don't work in cron context.
- No locking (concurrent runs corrupt state).
- No logging (can't debug when it fails).
- Stuffing 500 lines of bash into a "quick script."

## Cross-references

- Exercise prompt:
  `ai-infra-junior-engineer-learning/lessons/mod-002-linux-essentials/exercises/exercise-08-system-automation.md`
- Beyond this exercise: `mod-006-kubernetes-intro` and the
  Engineer track's `mod-109` cover the "proper tools" mentioned
  above.
