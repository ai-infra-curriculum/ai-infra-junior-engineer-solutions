# Troubleshooting Guide

Common issues and solutions for the Airflow Workflow Monitoring exercise.

## Table of Contents

1. [Services Not Starting](#services-not-starting)
2. [DAGs Not Appearing](#dags-not-appearing)
3. [Tasks Failing](#tasks-failing)
4. [Metrics Not Showing](#metrics-not-showing)
5. [Performance Issues](#performance-issues)
6. [Database Issues](#database-issues)

---

## Services Not Starting

### Issue: Docker containers fail to start

**Symptoms:**
- `docker-compose up` fails
- Containers exit immediately
- Health checks fail

**Solutions:**

1. **Check Docker is running:**
   ```bash
   docker ps
   ```

2. **Check available disk space:**
   ```bash
   df -h
   ```
   Need at least 10GB free space

3. **Check port conflicts:**
   ```bash
   # Check if ports are already in use
   lsof -i :8080  # Airflow
   lsof -i :9090  # Prometheus
   lsof -i :3000  # Grafana
   ```

4. **Clean up existing containers:**
   ```bash
   cd docker
   docker-compose down -v
   docker-compose up -d
   ```

5. **Check logs:**
   ```bash
   docker-compose logs airflow-webserver
   docker-compose logs airflow-scheduler
   docker-compose logs postgres
   ```

### Issue: Permission denied errors

**Symptoms:**
- "Permission denied" in logs
- Airflow can't write to directories

**Solutions:**

1. **Check AIRFLOW_UID:**
   ```bash
   cat .env
   ```
   Should match your user ID: `echo $UID`

2. **Fix permissions:**
   ```bash
   sudo chown -R $USER:$USER logs/ plugins/ dags/
   ```

3. **Recreate with correct UID:**
   ```bash
   cd docker
   docker-compose down -v
   cd ..
   echo "AIRFLOW_UID=$(id -u)" > .env
   cd docker
   docker-compose up -d
   ```

---

## DAGs Not Appearing

### Issue: DAGs don't show in UI

**Symptoms:**
- DAG list is empty
- New DAGs not detected
- "No DAGs found" message

**Solutions:**

1. **Check DAG syntax:**
   ```bash
   docker-compose -f docker/docker-compose.yml exec airflow-webserver \
     python /opt/airflow/dags/ml_pipeline_dag.py
   ```

2. **Check for import errors:**
   ```bash
   docker-compose -f docker/docker-compose.yml exec airflow-webserver \
     airflow dags list-import-errors
   ```

3. **Verify DAG file location:**
   ```bash
   docker-compose -f docker/docker-compose.yml exec airflow-webserver \
     ls -la /opt/airflow/dags/
   ```

4. **Check scheduler logs:**
   ```bash
   docker-compose -f docker/docker-compose.yml logs airflow-scheduler | tail -50
   ```

5. **Force DAG refresh:**
   - In Airflow UI: Admin → Configuration → Refresh DAGs
   - Or restart scheduler:
     ```bash
     docker-compose -f docker/docker-compose.yml restart airflow-scheduler
     ```

### Issue: DAG shows "Import Error"

**Symptoms:**
- DAG appears with import error icon
- Error message in DAG details

**Solutions:**

1. **Check Python syntax:**
   ```bash
   python3 dags/ml_pipeline_dag.py
   ```

2. **Check imports:**
   - Ensure all imports are available
   - Check `sys.path` includes src directory

3. **Review error message:**
   - Click on DAG in UI
   - Read full error stack trace
   - Fix identified issue

---

## Tasks Failing

### Issue: Tasks fail immediately

**Symptoms:**
- Task turns red immediately
- Error in task logs
- Task retries repeatedly

**Solutions:**

1. **Check task logs:**
   - Click task in Graph view
   - Click "Log" button
   - Review error message

2. **Check XCom data:**
   - Click task
   - Click "XCom" tab
   - Verify data from previous tasks

3. **Common task errors:**

   **Missing XCom data:**
   ```python
   # Error: NoneType object has no attribute...
   # Solution: Check previous task completed successfully
   ```

   **Import errors:**
   ```python
   # Error: ModuleNotFoundError
   # Solution: Ensure src/ is in PYTHONPATH
   ```

   **Resource constraints:**
   ```bash
   # Check Docker resources
   docker stats
   ```

4. **Test task in isolation:**
   ```bash
   docker-compose -f docker/docker-compose.yml exec airflow-webserver \
     airflow tasks test ml_pipeline_dag download_data 2024-01-15
   ```

### Issue: Tasks stuck in "queued"

**Symptoms:**
- Tasks never start
- Remain in queued state
- No progress

**Solutions:**

1. **Check executor:**
   ```bash
   docker-compose -f docker/docker-compose.yml exec airflow-webserver \
     airflow config get-value core executor
   ```

2. **Check scheduler is running:**
   ```bash
   docker-compose -f docker/docker-compose.yml ps airflow-scheduler
   ```

3. **Check parallelism settings:**
   - Admin → Configuration
   - Look for `parallelism`, `dag_concurrency`

4. **Restart scheduler:**
   ```bash
   docker-compose -f docker/docker-compose.yml restart airflow-scheduler
   ```

---

## Metrics Not Showing

### Issue: No metrics in Prometheus

**Symptoms:**
- Prometheus shows no Airflow metrics
- Empty graphs
- No targets

**Solutions:**

1. **Check StatsD exporter:**
   ```bash
   docker-compose -f docker/docker-compose.yml logs statsd-exporter
   ```

2. **Verify Prometheus targets:**
   - Go to http://localhost:9090/targets
   - Check statsd-exporter is "UP"

3. **Check Airflow StatsD config:**
   ```bash
   docker-compose -f docker/docker-compose.yml exec airflow-webserver \
     airflow config get-value metrics statsd_on
   ```
   Should be "True"

4. **Test StatsD connection:**
   ```bash
   # From inside Airflow container
   docker-compose -f docker/docker-compose.yml exec airflow-webserver \
     ping statsd-exporter
   ```

5. **Restart services:**
   ```bash
   docker-compose -f docker/docker-compose.yml restart statsd-exporter prometheus
   ```

### Issue: Grafana can't connect to Prometheus

**Symptoms:**
- "Bad Gateway" error
- Can't add data source
- Connection timeout

**Solutions:**

1. **Check Prometheus URL:**
   - Should be: `http://prometheus:9090`
   - Not: `http://localhost:9090`

2. **Check network:**
   ```bash
   docker-compose -f docker/docker-compose.yml exec grafana \
     ping prometheus
   ```

3. **Verify Prometheus is accessible:**
   ```bash
   curl http://localhost:9090/api/v1/query?query=up
   ```

---

## Performance Issues

### Issue: Airflow UI is slow

**Symptoms:**
- Pages take long to load
- Timeouts
- Unresponsive

**Solutions:**

1. **Check Docker resources:**
   ```bash
   docker stats
   ```
   Increase Docker memory/CPU if needed

2. **Check database size:**
   ```bash
   docker-compose -f docker/docker-compose.yml exec postgres \
     psql -U airflow -d airflow -c "SELECT pg_size_pretty(pg_database_size('airflow'));"
   ```

3. **Clean old task instances:**
   ```bash
   docker-compose -f docker/docker-compose.yml exec airflow-webserver \
     airflow db clean --clean-before-timestamp "2024-01-01"
   ```

4. **Reduce log retention:**
   - Edit `docker/docker-compose.yml`
   - Add: `AIRFLOW__LOGGING__LOG_RETENTION_DAYS: 7`

### Issue: Tasks run slowly

**Symptoms:**
- Tasks take longer than expected
- High CPU usage
- Memory issues

**Solutions:**

1. **Check resource usage:**
   ```bash
   docker stats
   ```

2. **Increase parallelism:**
   - Edit DAG: `max_active_tasks`
   - Or system-wide in config

3. **Optimize task code:**
   - Profile slow tasks
   - Reduce logging
   - Optimize data processing

---

## Database Issues

### Issue: Database connection errors

**Symptoms:**
- "OperationalError: could not connect"
- Scheduler can't connect
- Webserver shows errors

**Solutions:**

1. **Check PostgreSQL is running:**
   ```bash
   docker-compose -f docker/docker-compose.yml ps postgres
   ```

2. **Check PostgreSQL logs:**
   ```bash
   docker-compose -f docker/docker-compose.yml logs postgres
   ```

3. **Test connection:**
   ```bash
   docker-compose -f docker/docker-compose.yml exec postgres \
     psql -U airflow -d airflow -c "SELECT 1;"
   ```

4. **Reset database:**
   ```bash
   cd docker
   docker-compose down -v
   docker-compose up -d
   ```
   **Warning:** This deletes all data!

### Issue: Database locked errors

**Symptoms:**
- "database is locked"
- SQLite errors (if using SQLite)

**Solution:**
- Ensure using PostgreSQL (configured in docker-compose.yml)
- Never use SQLite for production

---

## Quick Diagnostics

Run these commands to get a quick health check:

```bash
# Check all container status
docker-compose -f docker/docker-compose.yml ps

# Check logs for errors
docker-compose -f docker/docker-compose.yml logs --tail=100 | grep -i error

# List DAGs
docker-compose -f docker/docker-compose.yml exec airflow-webserver airflow dags list

# Check import errors
docker-compose -f docker/docker-compose.yml exec airflow-webserver airflow dags list-import-errors

# Test database connection
docker-compose -f docker/docker-compose.yml exec postgres psql -U airflow -d airflow -c "SELECT version();"

# Check Prometheus targets
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'
```

---

## Getting Help

If you're still stuck:

1. **Check logs thoroughly:**
   ```bash
   docker-compose -f docker/docker-compose.yml logs -f
   ```

2. **Search error messages:**
   - Google the exact error
   - Check Airflow documentation
   - Search Stack Overflow

3. **Verify versions:**
   ```bash
   docker-compose -f docker/docker-compose.yml exec airflow-webserver airflow version
   ```

4. **Reset everything:**
   ```bash
   cd docker
   docker-compose down -v
   cd ..
   ./scripts/setup.sh
   ./scripts/start.sh
   ```

---

## Prevention Tips

1. **Always check DAG syntax before deploying:**
   ```bash
   python dags/my_dag.py
   ```

2. **Use the test script regularly:**
   ```bash
   ./scripts/test.sh
   ```

3. **Monitor resources:**
   ```bash
   docker stats
   ```

4. **Keep Docker images updated:**
   ```bash
   docker-compose pull
   ```

5. **Regular maintenance:**
   ```bash
   # Clean old data weekly
   airflow db clean --clean-before-timestamp "$(date -d '7 days ago' +%Y-%m-%d)"
   ```
