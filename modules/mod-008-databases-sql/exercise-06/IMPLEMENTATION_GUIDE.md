# Implementation Guide: Transaction Isolation and Concurrency Control

## Overview

This guide provides step-by-step instructions for implementing transaction isolation and concurrency control for the ML Model Registry. You'll learn ACID properties, isolation levels, locking strategies, and deadlock handling.

**Estimated Time:** 3-4 hours
**Difficulty:** Intermediate

## Prerequisites

- Docker and Docker Compose installed
- Python 3.9+ installed
- Basic understanding of SQL and Python
- Basic understanding of databases and transactions

## Phase 1: Environment Setup (30 minutes)

### Step 1.1: Start PostgreSQL Database

```bash
# Navigate to exercise directory
cd modules/mod-008-databases-sql/exercise-06

# Start PostgreSQL with Docker Compose
docker-compose up -d

# Verify PostgreSQL is running
docker-compose ps

# Wait for PostgreSQL to be ready
docker-compose logs -f postgres
# Look for: "database system is ready to accept connections"
# Press Ctrl+C to exit logs
```

**Expected Output:**
```
Creating ml-registry-db ... done
Name                      State       Ports
ml-registry-db           Up          0.0.0.0:5432->5432/tcp
```

### Step 1.2: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep sqlalchemy
```

### Step 1.3: Initialize Database Schema

```bash
# Method 1: Using psql directly
psql -h localhost -U mluser -d model_registry -f sql/schema.sql
# Password: mlregistry123

# Method 2: Using Python
python -c "from src.db_connection import init_database; init_database()"

# Verify tables were created
psql -h localhost -U mluser -d model_registry -c "\dt"
```

**Expected Tables:**
```
 Schema |      Name       | Type  |  Owner
--------+-----------------+-------+---------
 public | experiments     | table | mluser
 public | model_metadata  | table | mluser
 public | model_versions  | table | mluser
 public | models          | table | mluser
```

### Step 1.4: Test Database Connection

```bash
python -c "from src.db_connection import test_connection; test_connection()"
```

**Expected Output:**
```
✓ Database connection successful
```

## Phase 2: ACID Properties (30 minutes)

### Step 2.1: Understand Atomicity

**Concept:** All operations in a transaction succeed or all fail.

**Implementation:**

Create `tests/test_acid_atomicity.py`:
```python
from src.db_connection import get_db_session
from sqlalchemy import text

def register_model_with_version(model_name: str, version: int):
    with get_db_session() as session:
        # Insert model
        result = session.execute(
            text("INSERT INTO models (name, created_by) VALUES (:name, 'system') RETURNING id"),
            {"name": model_name}
        )
        model_id = result.fetchone()[0]

        # Insert version (may fail, causing rollback)
        session.execute(
            text("INSERT INTO model_versions (model_id, version, framework) VALUES (:mid, :ver, 'pytorch')"),
            {"mid": model_id, "ver": version}
        )

# Test atomicity
register_model_with_version("test-model-1", 1)  # Succeeds
register_model_with_version("test-model-1", 1)  # Fails, rolls back model creation
```

**Test:**
```bash
python tests/test_acid_atomicity.py
```

**Expected:**
- First call creates both model and version
- Second call fails due to duplicate version
- Model "test-model-1" exists only once (not twice)

### Step 2.2: Understand Consistency

**Concept:** Database constraints prevent invalid states.

**Test:**
```python
# Try to create version for non-existent model
INSERT INTO model_versions (model_id, version) VALUES (99999, 1);
# Error: violates foreign key constraint
```

### Step 2.3: Run All ACID Tests

```bash
python src/transaction_examples.py
```

## Phase 3: Transaction Isolation Levels (1 hour)

### Step 3.1: READ COMMITTED (Default)

**Prevents:** Dirty reads
**Allows:** Non-repeatable reads, phantom reads

**Test Scenario:**
```python
# Writer: UPDATE (not committed) → sleep → COMMIT
# Reader: READ → sleep → READ again
# Expected: Second read sees updated value
```

**Run:**
```bash
python -c "from src.transaction_examples import demonstrate_read_committed; demonstrate_read_committed()"
```

**Expected Output:**
```
Writer: Updated (not committed)
Reader: First read = 'Credit card fraud detection model'
Writer: Committed
Reader: Second read = 'TEMP UPDATE'
✓ Non-repeatable read occurred (expected)
```

### Step 3.2: REPEATABLE READ

**Prevents:** Dirty reads, non-repeatable reads
**Allows:** Phantom reads

**Test Scenario:**
```python
# Reader: Start transaction → READ → sleep → READ again
# Writer: UPDATE and COMMIT
# Expected: Both reads see same value (snapshot isolation)
```

**Run:**
```bash
python -c "from src.transaction_examples import demonstrate_repeatable_read; demonstrate_repeatable_read()"
```

**Expected Output:**
```
Reader: First read = 'staging'
Writer: Updated stage to production
Reader: Second read = 'staging'
✓ Repeatable read: saw consistent snapshot
```

### Step 3.3: SERIALIZABLE

**Prevents:** All anomalies (dirty reads, non-repeatable reads, phantom reads)
**Cost:** Highest overhead, possible serialization failures

**Behavior:** Transactions execute as if serial (one after another)

### Step 3.4: Isolation Level Decision Matrix

| Use Case | Isolation Level | Reason |
|----------|----------------|---------|
| Model registration | READ COMMITTED | Simple writes, no read dependencies |
| Metric logging | READ COMMITTED | Independent inserts |
| Model comparison | REPEATABLE READ | Need consistent snapshot |
| Critical promotion | SERIALIZABLE | Absolute correctness required |

## Phase 4: Pessimistic Locking (45 minutes)

### Step 4.1: Understand SELECT FOR UPDATE

**Purpose:** Lock rows to prevent concurrent modifications

**Syntax:**
```sql
SELECT * FROM models WHERE id = 1 FOR UPDATE;
-- Row is now locked until transaction commits/rolls back
-- Other transactions will wait
```

### Step 4.2: Safe Version Number Generation

**Problem:** Race condition in version number generation

**Without Locking:**
```python
# Thread A: SELECT MAX(version) → 5
# Thread B: SELECT MAX(version) → 5
# Thread A: INSERT version 6
# Thread B: INSERT version 6  ← Duplicate!
```

**With Locking:**
```python
def register_new_model_version_safe(model_id, framework, artifact_uri):
    with get_db_session() as session:
        # Lock model row
        session.execute(
            text("SELECT id FROM models WHERE id = :mid FOR UPDATE"),
            {"mid": model_id}
        )

        # Get max version (protected by lock)
        result = session.execute(
            text("SELECT COALESCE(MAX(version), 0) FROM model_versions WHERE model_id = :mid"),
            {"mid": model_id}
        )
        next_version = result.fetchone()[0] + 1

        # Insert new version
        session.execute(
            text("INSERT INTO model_versions (model_id, version, framework, artifact_uri) VALUES (:mid, :ver, :fw, :uri)"),
            {"mid": model_id, "ver": next_version, "fw": framework, "uri": artifact_uri}
        )

        return next_version
```

### Step 4.3: Test Concurrent Version Creation

```bash
python -c "from src.transaction_examples import test_pessimistic_locking; test_pessimistic_locking()"
```

**Expected Output:**
```
[Thread-0] Acquired lock on model 1
[Thread-0] Generating version 4
[Thread-0] Created version 4
[Thread-1] Acquired lock on model 1  ← Waited for Thread-0
[Thread-1] Generating version 5
[Thread-1] Created version 5
[Thread-2] Acquired lock on model 1
[Thread-2] Generating version 6
[Thread-2] Created version 6

Final versions for model 1: [1, 2, 3, 4, 5, 6]
✓ No duplicate versions
```

## Phase 5: Optimistic Locking (45 minutes)

### Step 5.1: Understand Optimistic Locking

**Concept:** Don't lock, but detect conflicts at update time

**Implementation:** Use a version column

**Table Schema:**
```sql
CREATE TABLE model_metadata (
    model_version_id INTEGER PRIMARY KEY,
    tags JSONB,
    parameters JSONB,
    version_lock INTEGER DEFAULT 0  ← Version column
);
```

### Step 5.2: Update with Conflict Detection

```python
def update_model_metadata_optimistic(model_version_id, new_tags, new_params):
    with get_db_session() as session:
        # Read current version_lock
        result = session.execute(
            text("SELECT version_lock FROM model_metadata WHERE model_version_id = :mvid"),
            {"mvid": model_version_id}
        )
        current_lock = result.fetchone()[0]

        # Update only if version_lock unchanged
        result = session.execute(
            text("""
                UPDATE model_metadata
                SET tags = :tags, parameters = :params, version_lock = :new_lock
                WHERE model_version_id = :mvid AND version_lock = :old_lock
            """),
            {"mvid": model_version_id, "tags": json.dumps(new_tags),
             "params": json.dumps(new_params),
             "old_lock": current_lock, "new_lock": current_lock + 1}
        )

        if result.rowcount == 0:
            raise OptimisticLockException("Conflict detected")
```

### Step 5.3: Implement Retry Logic

```python
def update_with_retry(model_version_id, new_tags, new_params):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            update_model_metadata_optimistic(model_version_id, new_tags, new_params)
            return  # Success
        except OptimisticLockException:
            time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
    raise Exception("Max retries exceeded")
```

### Step 5.4: Test Optimistic Locking

```bash
python -c "from src.transaction_examples import test_optimistic_locking; test_optimistic_locking()"
```

**Expected Output:**
```
[Thread-0] Current version_lock: 0
[Thread-1] Current version_lock: 0
[Thread-2] Current version_lock: 0
[Thread-0] Updated metadata (new version_lock: 1)
Thread 0: Update succeeded on attempt 1
[Thread-1] Conflict on attempt 1 - retrying...
[Thread-1] Current version_lock: 1
[Thread-1] Updated metadata (new version_lock: 2)
Thread 1: Update succeeded on attempt 2
[Thread-2] Conflict on attempt 1 - retrying...
[Thread-2] Current version_lock: 2
[Thread-2] Updated metadata (new version_lock: 3)
Thread 2: Update succeeded on attempt 2

Final state: version_lock=3
✓ All updates applied sequentially
```

### Step 5.5: When to Use Each Strategy

**Pessimistic Locking:**
- ✅ High contention (many concurrent writes)
- ✅ Short transactions
- ✅ Critical operations (version generation, promotions)
- ❌ Can cause deadlocks
- ❌ Reduces concurrency

**Optimistic Locking:**
- ✅ Low contention (rare conflicts)
- ✅ Long-running transactions (user edits)
- ✅ Distributed systems
- ❌ Requires retry logic
- ❌ Wastes work on conflicts

## Phase 6: Deadlock Handling (30 minutes)

### Step 6.1: Create Deadlock Scenario

```python
# Thread A: Lock model 1 → Try to lock model 2
# Thread B: Lock model 2 → Try to lock model 1
# Result: Circular wait → Deadlock
```

**Run:**
```bash
python -c "from src.transaction_examples import demonstrate_deadlock; demonstrate_deadlock()"
```

**Expected Output:**
```
Thread A: Acquired lock on model 1
Thread B: Acquired lock on model 2
Thread A: Trying to lock model 2...  ← Waiting
Thread B: Trying to lock model 1...  ← Waiting
Thread B: DEADLOCK DETECTED - deadlock detected
Thread A: SUCCESS
```

PostgreSQL detects the deadlock and aborts one transaction.

### Step 6.2: Prevent with Lock Ordering

**Solution:** Always acquire locks in the same order (by ID)

```python
def lock_models_safely(model_ids):
    sorted_ids = sorted(model_ids)  # Always lock in ascending order
    for model_id in sorted_ids:
        session.execute(text(f"SELECT * FROM models WHERE id = {model_id} FOR UPDATE"))
```

**Test:**
```bash
python -c "from src.transaction_examples import demonstrate_lock_ordering; demonstrate_lock_ordering()"
```

**Expected Output:**
```
Thread A: Locking model 1...
Thread A: Acquired lock on model 1
Thread B: Locking model 1...  ← Waits for A
Thread A: Locking model 2...
Thread A: Acquired lock on model 2
Thread A: SUCCESS
Thread B: Acquired lock on model 1  ← A released locks
Thread B: Locking model 2...
Thread B: Acquired lock on model 2
Thread B: SUCCESS

✓ No deadlock occurred
```

### Step 6.3: Implement Retry on Deadlock

```python
def operation_with_deadlock_retry(max_retries=3):
    for attempt in range(max_retries):
        try:
            # Perform operation
            return success
        except Exception as e:
            if "deadlock" in str(e).lower():
                time.sleep(0.1 * (2 ** attempt))
            else:
                raise
    raise Exception("Max retries exceeded")
```

## Phase 7: Testing and Validation (30 minutes)

### Step 7.1: Run All Demonstrations

```bash
python src/transaction_examples.py
```

This runs all demonstrations in sequence:
1. ✅ ACID properties (atomicity, consistency)
2. ✅ Isolation levels (READ COMMITTED, REPEATABLE READ)
3. ✅ Pessimistic locking (safe version generation)
4. ✅ Optimistic locking (metadata updates with retries)
5. ✅ Deadlock creation and prevention

### Step 7.2: Run Performance Benchmarks

```bash
python tests/benchmark_locking.py
```

**Expected Results:**
- Pessimistic: ~1 op/s (sequential)
- Optimistic: ~3-4 op/s (concurrent, with retries)

### Step 7.3: Verify Database State

```bash
psql -h localhost -U mluser -d model_registry

# Check for duplicate versions (should be none)
SELECT model_id, version, COUNT(*)
FROM model_versions
GROUP BY model_id, version
HAVING COUNT(*) > 1;

# Check version_lock progression
SELECT model_version_id, version_lock FROM model_metadata ORDER BY version_lock DESC LIMIT 5;
```

## Common Issues and Solutions

### Issue 1: Connection Refused

**Symptoms:**
```
psycopg2.OperationalError: could not connect to server
```

**Solution:**
```bash
# Check if PostgreSQL is running
docker-compose ps

# Check logs
docker-compose logs postgres

# Restart PostgreSQL
docker-compose restart postgres
```

### Issue 2: Deadlock Not Detected

**Symptoms:**
Threads hang indefinitely

**Causes:**
- Not using FOR UPDATE
- Insufficient sleep time between threads
- Wrong lock acquisition order

**Solution:**
Verify you're using `SELECT ... FOR UPDATE` and proper timing.

### Issue 3: Optimistic Lock Always Fails

**Symptoms:**
All threads fail on first attempt

**Cause:**
Not incrementing version_lock correctly

**Solution:**
```sql
-- Ensure you're incrementing
UPDATE model_metadata SET version_lock = version_lock + 1 ...
```

## Best Practices Summary

### 1. Choose the Right Isolation Level

✅ Default to READ COMMITTED for most operations
✅ Use REPEATABLE READ for consistent reports
✅ Use SERIALIZABLE only when necessary
❌ Don't over-isolate (performance cost)

### 2. Implement Proper Locking

✅ Use pessimistic locking for high-contention writes
✅ Use optimistic locking for low-contention updates
✅ Always use lock ordering to prevent deadlocks
❌ Don't hold locks during expensive operations

### 3. Handle Failures Gracefully

✅ Implement retry logic for optimistic locks
✅ Implement retry logic for deadlocks
✅ Use exponential backoff
✅ Log all conflicts and deadlocks
❌ Don't ignore exceptions

## Completion Checklist

- [ ] PostgreSQL running and accessible
- [ ] Database schema created successfully
- [ ] ACID properties demonstrated
- [ ] All 4 isolation levels tested
- [ ] Pessimistic locking prevents race conditions
- [ ] Optimistic locking detects conflicts
- [ ] Deadlocks created and prevented
- [ ] All tests pass
- [ ] Performance benchmarks run
- [ ] Understand trade-offs of each approach

## Next Steps

1. **Add Monitoring**: Track transaction metrics
2. **Implement Distributed Locks**: Redis-based locking
3. **Add Circuit Breakers**: Handle database failures
4. **Implement Saga Pattern**: Multi-step workflows
5. **Optimize Connection Pool**: Tune for your workload

## Resources

- [PostgreSQL Transaction Isolation](https://www.postgresql.org/docs/current/transaction-iso.html)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org)
- [Designing Data-Intensive Applications](https://dataintensive.net/)

Congratulations! You've completed the Transaction Isolation and Concurrency Control exercise.
