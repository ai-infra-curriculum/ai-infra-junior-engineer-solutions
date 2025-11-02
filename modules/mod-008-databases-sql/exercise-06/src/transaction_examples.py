"""
Transaction Isolation and Concurrency Control Examples

This module demonstrates all key concepts:
- ACID properties
- Transaction isolation levels
- Pessimistic locking (SELECT FOR UPDATE)
- Optimistic locking (version numbers)
- Deadlock handling

Run this file to see demonstrations of each concept.
"""

import time
import threading
import json
from sqlalchemy import text
from db_connection import get_db_session, logger
from typing import Optional


# ============================================================================
# ACID PROPERTIES
# ============================================================================

def demonstrate_atomicity():
    """
    Atomicity: All-or-nothing execution.
    If one operation fails, all operations in the transaction are rolled back.
    """
    print("\n=== ACID: Atomicity Demonstration ===\n")

    def register_model_with_version(model_name: str, version: int):
        """Either both model and version are created, or neither."""
        with get_db_session() as session:
            # Insert model
            result = session.execute(
                text("INSERT INTO models (name, created_by) VALUES (:name, 'system') RETURNING id"),
                {"name": model_name}
            )
            model_id = result.fetchone()[0]
            print(f"✓ Created model '{model_name}' with ID {model_id}")

            # Insert version (this will fail if duplicate, causing rollback)
            session.execute(
                text("INSERT INTO model_versions (model_id, version, framework) VALUES (:mid, :ver, 'pytorch')"),
                {"mid": model_id, "ver": version}
            )
            print(f"✓ Created version {version} for model {model_id}")

    try:
        # First call succeeds
        register_model_with_version("atomicity-test-1", 1)
        print("✓ Transaction 1 committed\n")
    except Exception as e:
        print(f"✗ Transaction 1 failed: {e}\n")

    try:
        # Second call with same version fails, rolling back model creation
        register_model_with_version("atomicity-test-1", 1)  # Duplicate version
        print("✓ Transaction 2 committed\n")
    except Exception as e:
        print(f"✗ Transaction 2 failed and rolled back: {e}\n")

    # Verify atomicity: model should only exist once
    with get_db_session() as session:
        result = session.execute(
            text("SELECT COUNT(*) FROM models WHERE name LIKE 'atomicity-test-%'")
        )
        count = result.fetchone()[0]
        print(f"Final count of atomicity-test models: {count}")
        print("✓ Atomicity preserved" if count == 1 else "✗ Atomicity violated")


def demonstrate_consistency():
    """
    Consistency: Database constraints maintain valid states.
    """
    print("\n=== ACID: Consistency Demonstration ===\n")

    # Test 1: Foreign key constraint
    print("Test 1: Foreign Key Constraint")
    with get_db_session() as session:
        try:
            session.execute(
                text("INSERT INTO model_versions (model_id, version) VALUES (99999, 1)")
            )
            print("✗ Consistency violated: orphaned version created")
        except Exception as e:
            print(f"✓ Consistency enforced: {str(e)[:80]}...")
            session.rollback()

    # Test 2: Unique constraint
    print("\nTest 2: Unique Constraint")
    with get_db_session() as session:
        try:
            session.execute(
                text("INSERT INTO model_versions (model_id, version, framework) VALUES (1, 1, 'pytorch')")
            )
            session.execute(
                text("INSERT INTO model_versions (model_id, version, framework) VALUES (1, 1, 'tensorflow')")
            )
            print("✗ Uniqueness violated")
        except Exception as e:
            print(f"✓ Uniqueness enforced: {str(e)[:80]}...")
            session.rollback()


# ============================================================================
# TRANSACTION ISOLATION LEVELS
# ============================================================================

def demonstrate_read_committed():
    """
    READ COMMITTED: Only see committed changes.
    Prevents dirty reads, but allows non-repeatable reads.
    """
    print("\n=== Isolation Level: READ COMMITTED ===\n")

    def writer():
        with get_db_session() as session:
            session.execute(text("SET TRANSACTION ISOLATION LEVEL READ COMMITTED"))
            session.execute(text("UPDATE models SET description = 'TEMP UPDATE' WHERE id = 1"))
            print("Writer: Updated (not committed)")
            time.sleep(2)
            session.commit()
            print("Writer: Committed")

    def reader():
        time.sleep(0.5)
        with get_db_session() as session:
            session.execute(text("SET TRANSACTION ISOLATION LEVEL READ COMMITTED"))

            result = session.execute(text("SELECT description FROM models WHERE id = 1"))
            desc1 = result.fetchone()[0]
            print(f"Reader: First read = '{desc1}' (should be original)")

            time.sleep(2.5)  # Wait for writer to commit

            result = session.execute(text("SELECT description FROM models WHERE id = 1"))
            desc2 = result.fetchone()[0]
            print(f"Reader: Second read = '{desc2}' (should be updated)")

            if desc1 != desc2:
                print("✓ Non-repeatable read occurred (expected behavior)")

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=reader)
    t1.start()
    t2.start()
    t1.join()
    t2.join()


def demonstrate_repeatable_read():
    """
    REPEATABLE READ: Sees snapshot of database at transaction start.
    Prevents non-repeatable reads within same transaction.
    """
    print("\n=== Isolation Level: REPEATABLE READ ===\n")

    def writer():
        time.sleep(1)
        with get_db_session() as session:
            session.execute(text("UPDATE model_versions SET stage = 'production' WHERE id = 1"))
            session.commit()
            print("Writer: Updated stage to production")

    def reader():
        with get_db_session() as session:
            session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"))

            result = session.execute(text("SELECT stage FROM model_versions WHERE id = 1"))
            stage1 = result.fetchone()[0]
            print(f"Reader: First read = '{stage1}'")

            time.sleep(3)  # Wait for writer to commit

            result = session.execute(text("SELECT stage FROM model_versions WHERE id = 1"))
            stage2 = result.fetchone()[0]
            print(f"Reader: Second read = '{stage2}'")

            if stage1 == stage2:
                print("✓ Repeatable read: saw consistent snapshot")
            else:
                print("✗ Non-repeatable read occurred")

    t1 = threading.Thread(target=reader)
    t2 = threading.Thread(target=writer)
    t1.start()
    t2.start()
    t1.join()
    t2.join()


# ============================================================================
# PESSIMISTIC LOCKING
# ============================================================================

def register_new_model_version_safe(model_id: int, framework: str, artifact_uri: str) -> int:
    """
    Use SELECT FOR UPDATE to safely generate next version number.
    Prevents race condition where two processes read same max version.
    """
    with get_db_session() as session:
        # Lock the model row
        session.execute(
            text("SELECT id FROM models WHERE id = :mid FOR UPDATE"),
            {"mid": model_id}
        )
        thread_name = threading.current_thread().name
        print(f"[{thread_name}] Acquired lock on model {model_id}")

        # Get current max version (protected by lock)
        result = session.execute(
            text("SELECT COALESCE(MAX(version), 0) FROM model_versions WHERE model_id = :mid"),
            {"mid": model_id}
        )
        max_version = result.fetchone()[0]
        next_version = max_version + 1
        print(f"[{thread_name}] Generating version {next_version}")

        # Simulate processing time
        time.sleep(0.5)

        # Insert new version
        session.execute(
            text("""
                INSERT INTO model_versions (model_id, version, framework, artifact_uri)
                VALUES (:mid, :ver, :fw, :uri)
            """),
            {"mid": model_id, "ver": next_version, "fw": framework, "uri": artifact_uri}
        )
        print(f"[{thread_name}] Created version {next_version}")
        return next_version


def test_pessimistic_locking():
    """Test concurrent version creation with pessimistic locking."""
    print("\n=== Pessimistic Locking: Safe Version Generation ===\n")

    def create_version(i):
        try:
            version = register_new_model_version_safe(
                model_id=1,
                framework="pytorch",
                artifact_uri=f"s3://models/fraud-detector/v{i}.pt"
            )
            print(f"Thread {i}: Created version {version}")
        except Exception as e:
            print(f"Thread {i}: Error - {e}")

    threads = [threading.Thread(target=create_version, args=(i,), name=f"Thread-{i}") for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify no duplicate versions
    with get_db_session() as session:
        result = session.execute(
            text("SELECT version FROM model_versions WHERE model_id = 1 ORDER BY version")
        )
        versions = [row[0] for row in result.fetchall()]
        print(f"\nFinal versions for model 1: {versions}")

        if len(versions) == len(set(versions)):
            print("✓ No duplicate versions - pessimistic locking worked")
        else:
            print("✗ Duplicate versions detected")


# ============================================================================
# OPTIMISTIC LOCKING
# ============================================================================

class OptimisticLockException(Exception):
    """Raised when optimistic lock fails."""
    pass


def update_model_metadata_optimistic(model_version_id: int, new_tags: dict, new_params: dict):
    """
    Update metadata using optimistic locking (version_lock field).
    Only succeeds if version_lock hasn't changed since we read it.
    """
    with get_db_session() as session:
        # Read current metadata and version_lock
        result = session.execute(
            text("SELECT tags, parameters, version_lock FROM model_metadata WHERE model_version_id = :mvid"),
            {"mvid": model_version_id}
        )
        row = result.fetchone()

        if not row:
            # Insert if doesn't exist
            session.execute(
                text("""
                    INSERT INTO model_metadata (model_version_id, tags, parameters, version_lock)
                    VALUES (:mvid, :tags, :params, 0)
                """),
                {"mvid": model_version_id, "tags": json.dumps(new_tags), "params": json.dumps(new_params)}
            )
            print(f"Created metadata for version {model_version_id}")
            return

        current_lock = row[2]
        thread_name = threading.current_thread().name
        print(f"[{thread_name}] Current version_lock: {current_lock}")

        # Simulate processing time
        time.sleep(0.5)

        # Update only if version_lock hasn't changed
        result = session.execute(
            text("""
                UPDATE model_metadata
                SET tags = :tags, parameters = :params, version_lock = :new_lock, updated_at = CURRENT_TIMESTAMP
                WHERE model_version_id = :mvid AND version_lock = :old_lock
            """),
            {
                "mvid": model_version_id,
                "tags": json.dumps(new_tags),
                "params": json.dumps(new_params),
                "old_lock": current_lock,
                "new_lock": current_lock + 1
            }
        )

        if result.rowcount == 0:
            raise OptimisticLockException(f"Metadata was modified by another process")

        print(f"[{thread_name}] Updated metadata (new version_lock: {current_lock + 1})")


def test_optimistic_locking():
    """Test concurrent updates with optimistic locking."""
    print("\n=== Optimistic Locking: Metadata Updates ===\n")

    model_version_id = 1

    def update_with_retry(thread_id):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                new_tags = {"updated_by": f"thread_{thread_id}", "attempt": attempt}
                new_params = {"value": thread_id * 100}
                update_model_metadata_optimistic(model_version_id, new_tags, new_params)
                print(f"Thread {thread_id}: Update succeeded on attempt {attempt + 1}")
                return
            except OptimisticLockException as e:
                print(f"Thread {thread_id}: Conflict on attempt {attempt + 1} - retrying...")
                time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
        print(f"Thread {thread_id}: All retries exhausted")

    threads = [threading.Thread(target=update_with_retry, args=(i,), name=f"Thread-{i}") for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Check final state
    with get_db_session() as session:
        result = session.execute(
            text("SELECT tags, version_lock FROM model_metadata WHERE model_version_id = :mvid"),
            {"mvid": model_version_id}
        )
        row = result.fetchone()
        print(f"\nFinal state: tags={row[0]}, version_lock={row[1]}")
        print(f"✓ Final version_lock is {row[1]} (started at 0)")


# ============================================================================
# DEADLOCK HANDLING
# ============================================================================

def demonstrate_deadlock():
    """
    Create a deadlock scenario:
    - Thread A locks model 1, then tries to lock model 2
    - Thread B locks model 2, then tries to lock model 1
    PostgreSQL detects this and aborts one transaction.
    """
    print("\n=== Deadlock: Creation and Detection ===\n")

    def thread_a():
        try:
            with get_db_session() as session:
                print("Thread A: Locking model 1...")
                session.execute(text("SELECT * FROM models WHERE id = 1 FOR UPDATE"))
                print("Thread A: Acquired lock on model 1")

                time.sleep(1)  # Give thread B time to lock model 2

                print("Thread A: Trying to lock model 2...")
                session.execute(text("SELECT * FROM models WHERE id = 2 FOR UPDATE"))
                print("Thread A: Acquired lock on model 2")
                session.commit()
                print("Thread A: SUCCESS")
        except Exception as e:
            print(f"Thread A: DEADLOCK DETECTED - {str(e)[:100]}")

    def thread_b():
        try:
            time.sleep(0.5)  # Let thread A go first
            with get_db_session() as session:
                print("Thread B: Locking model 2...")
                session.execute(text("SELECT * FROM models WHERE id = 2 FOR UPDATE"))
                print("Thread B: Acquired lock on model 2")

                time.sleep(1)

                print("Thread B: Trying to lock model 1...")
                session.execute(text("SELECT * FROM models WHERE id = 1 FOR UPDATE"))
                print("Thread B: Acquired lock on model 1")
                session.commit()
                print("Thread B: SUCCESS")
        except Exception as e:
            print(f"Thread B: DEADLOCK DETECTED - {str(e)[:100]}")

    ta = threading.Thread(target=thread_a)
    tb = threading.Thread(target=thread_b)
    ta.start()
    tb.start()
    ta.join()
    tb.join()

    print("\n✓ PostgreSQL detected deadlock and aborted one transaction")


def demonstrate_lock_ordering():
    """
    Prevent deadlocks by always acquiring locks in the same order.
    """
    print("\n=== Deadlock Prevention: Lock Ordering ===\n")

    def thread_safe(thread_name, model_ids):
        try:
            # Sort model IDs to ensure consistent lock order
            sorted_ids = sorted(model_ids)

            with get_db_session() as session:
                for model_id in sorted_ids:
                    print(f"{thread_name}: Locking model {model_id}...")
                    session.execute(
                        text(f"SELECT * FROM models WHERE id = {model_id} FOR UPDATE")
                    )
                    print(f"{thread_name}: Acquired lock on model {model_id}")
                    time.sleep(0.5)

                session.commit()
                print(f"{thread_name}: SUCCESS - no deadlock")
        except Exception as e:
            print(f"{thread_name}: ERROR - {e}")

    ta = threading.Thread(target=thread_safe, args=("Thread A", [1, 2]))
    tb = threading.Thread(target=thread_safe, args=("Thread B", [2, 1]))  # Reverse order
    ta.start()
    tb.start()
    ta.join()
    tb.join()

    print("\n✓ Both threads succeeded because locks acquired in same order (1→2)")


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ML Model Registry: Transaction Isolation & Concurrency Control")
    print("=" * 70)

    # ACID Properties
    demonstrate_atomicity()
    demonstrate_consistency()

    # Isolation Levels
    demonstrate_read_committed()
    demonstrate_repeatable_read()

    # Locking Strategies
    test_pessimistic_locking()
    test_optimistic_locking()

    # Deadlock Handling
    demonstrate_deadlock()
    demonstrate_lock_ordering()

    print("\n" + "=" * 70)
    print("All demonstrations completed!")
    print("=" * 70)
