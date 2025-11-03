"""
Tests for async ML pipeline implementation.
"""

import pytest
import asyncio
import time
from typing import List, Dict
from dataclasses import dataclass


# ============================================================================
# Test Data Structures
# ============================================================================

@dataclass
class Sample:
    """Data sample for testing."""
    id: int
    data: List[float]
    processed: bool = False
    predicted: bool = False
    prediction: float = 0.0


# ============================================================================
# Pipeline Components
# ============================================================================

class TestAsyncPipeline:
    """Test async pipeline operations."""

    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size

    async def load_data(self, num_samples: int) -> List[Sample]:
        """Load data asynchronously."""
        await asyncio.sleep(0.01)
        samples = [
            Sample(id=i, data=[float(i) * 0.1 + j * 0.01 for j in range(5)])
            for i in range(num_samples)
        ]
        return samples

    async def preprocess_sample(self, sample: Sample) -> Sample:
        """Preprocess single sample."""
        await asyncio.sleep(0.005)
        sample.processed = True
        sample.data = [x / 10.0 for x in sample.data]
        return sample

    async def preprocess_batch(self, samples: List[Sample]) -> List[Sample]:
        """Preprocess batch of samples."""
        tasks = [self.preprocess_sample(s) for s in samples]
        return await asyncio.gather(*tasks)

    async def predict_sample(self, sample: Sample) -> Sample:
        """Run inference on single sample."""
        await asyncio.sleep(0.005)
        sample.predicted = True
        sample.prediction = sum(sample.data)
        return sample

    async def predict_batch(self, samples: List[Sample]) -> List[Sample]:
        """Run inference on batch."""
        tasks = [self.predict_sample(s) for s in samples]
        return await asyncio.gather(*tasks)


# ============================================================================
# Data Loading Tests
# ============================================================================

@pytest.mark.asyncio
async def test_load_data():
    """Test async data loading."""
    pipeline = TestAsyncPipeline()
    samples = await pipeline.load_data(50)

    assert len(samples) == 50
    assert all(isinstance(s, Sample) for s in samples)
    assert samples[0].id == 0
    assert samples[-1].id == 49


@pytest.mark.asyncio
async def test_load_multiple_datasets():
    """Test loading multiple datasets concurrently."""
    pipeline = TestAsyncPipeline()

    datasets = await asyncio.gather(
        pipeline.load_data(20),
        pipeline.load_data(30),
        pipeline.load_data(40)
    )

    assert len(datasets) == 3
    assert len(datasets[0]) == 20
    assert len(datasets[1]) == 30
    assert len(datasets[2]) == 40


# ============================================================================
# Preprocessing Tests
# ============================================================================

@pytest.mark.asyncio
async def test_preprocess_single_sample():
    """Test preprocessing a single sample."""
    pipeline = TestAsyncPipeline()
    sample = Sample(id=1, data=[1.0, 2.0, 3.0, 4.0, 5.0])

    processed = await pipeline.preprocess_sample(sample)

    assert processed.processed is True
    assert processed.data == [0.1, 0.2, 0.3, 0.4, 0.5]


@pytest.mark.asyncio
async def test_preprocess_batch():
    """Test preprocessing a batch of samples."""
    pipeline = TestAsyncPipeline()
    samples = [Sample(id=i, data=[float(i)] * 5) for i in range(10)]

    processed = await pipeline.preprocess_batch(samples)

    assert len(processed) == 10
    assert all(s.processed for s in processed)


@pytest.mark.asyncio
async def test_preprocess_performance():
    """Test that batch preprocessing is concurrent."""
    pipeline = TestAsyncPipeline()
    samples = [Sample(id=i, data=[1.0] * 5) for i in range(20)]

    # Sequential
    start = time.time()
    for sample in samples[:20]:
        await pipeline.preprocess_sample(Sample(id=sample.id, data=sample.data))
    sequential_time = time.time() - start

    # Concurrent batch
    start = time.time()
    await pipeline.preprocess_batch(samples)
    concurrent_time = time.time() - start

    # Concurrent should be significantly faster
    speedup = sequential_time / concurrent_time
    assert speedup >= 10.0


# ============================================================================
# Inference Tests
# ============================================================================

@pytest.mark.asyncio
async def test_predict_single_sample():
    """Test prediction on a single sample."""
    pipeline = TestAsyncPipeline()
    sample = Sample(id=1, data=[1.0, 2.0, 3.0, 4.0, 5.0], processed=True)

    predicted = await pipeline.predict_sample(sample)

    assert predicted.predicted is True
    assert predicted.prediction == 15.0


@pytest.mark.asyncio
async def test_predict_batch():
    """Test prediction on a batch of samples."""
    pipeline = TestAsyncPipeline()
    samples = [
        Sample(id=i, data=[float(i)] * 5, processed=True)
        for i in range(10)
    ]

    predicted = await pipeline.predict_batch(samples)

    assert len(predicted) == 10
    assert all(s.predicted for s in predicted)
    assert predicted[5].prediction == 25.0  # 5.0 * 5


@pytest.mark.asyncio
async def test_predict_performance():
    """Test that batch prediction is concurrent."""
    pipeline = TestAsyncPipeline()
    samples = [Sample(id=i, data=[1.0] * 5, processed=True) for i in range(20)]

    start = time.time()
    await pipeline.predict_batch(samples)
    batch_time = time.time() - start

    # Should complete in roughly the time of one prediction (0.005s)
    # plus overhead, not 20 predictions (0.1s)
    assert batch_time < 0.05


# ============================================================================
# Full Pipeline Tests
# ============================================================================

@pytest.mark.asyncio
async def test_full_pipeline_flow():
    """Test complete pipeline from load to prediction."""
    pipeline = TestAsyncPipeline(batch_size=10)

    # Load
    samples = await pipeline.load_data(30)
    assert len(samples) == 30

    # Preprocess in batches
    batches = [samples[i:i+10] for i in range(0, 30, 10)]
    preprocessed = []
    for batch in batches:
        batch_result = await pipeline.preprocess_batch(batch)
        preprocessed.extend(batch_result)

    assert len(preprocessed) == 30
    assert all(s.processed for s in preprocessed)

    # Predict in batches
    batches = [preprocessed[i:i+10] for i in range(0, 30, 10)]
    predicted = []
    for batch in batches:
        batch_result = await pipeline.predict_batch(batch)
        predicted.extend(batch_result)

    assert len(predicted) == 30
    assert all(s.predicted for s in predicted)
    assert all(s.prediction > 0 for s in predicted)


@pytest.mark.asyncio
async def test_pipeline_with_different_batch_sizes():
    """Test pipeline with various batch sizes."""
    batch_sizes = [5, 10, 20]
    num_samples = 50

    for batch_size in batch_sizes:
        pipeline = TestAsyncPipeline(batch_size=batch_size)
        samples = await pipeline.load_data(num_samples)

        # Process in batches
        batches = [samples[i:i+batch_size] for i in range(0, num_samples, batch_size)]
        all_processed = []

        for batch in batches:
            processed = await pipeline.preprocess_batch(batch)
            all_processed.extend(processed)

        assert len(all_processed) == num_samples


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.asyncio
async def test_pipeline_with_failing_samples():
    """Test pipeline handles failures gracefully."""
    class RobustPipeline(TestAsyncPipeline):
        async def preprocess_sample(self, sample: Sample) -> Dict:
            try:
                await asyncio.sleep(0.005)
                if sample.id % 5 == 0:  # Every 5th sample fails
                    raise ValueError(f"Sample {sample.id} failed")
                sample.processed = True
                return {"sample": sample, "success": True}
            except Exception as e:
                return {"sample": sample, "success": False, "error": str(e)}

    pipeline = RobustPipeline()
    samples = [Sample(id=i, data=[1.0] * 5) for i in range(20)]

    tasks = [pipeline.preprocess_sample(s) for s in samples]
    results = await asyncio.gather(*tasks)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    assert len(successful) == 16  # 20 - 4 failures (0, 5, 10, 15)
    assert len(failed) == 4


@pytest.mark.asyncio
async def test_pipeline_with_return_exceptions():
    """Test pipeline using return_exceptions."""
    class FailingPipeline(TestAsyncPipeline):
        async def preprocess_sample(self, sample: Sample) -> Sample:
            await asyncio.sleep(0.005)
            if sample.id == 5:
                raise ValueError("Sample 5 failed")
            sample.processed = True
            return sample

    pipeline = FailingPipeline()
    samples = [Sample(id=i, data=[1.0] * 5) for i in range(10)]

    tasks = [pipeline.preprocess_sample(s) for s in samples]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = [r for r in results if isinstance(r, Sample)]
    errors = [r for r in results if isinstance(r, Exception)]

    assert len(successful) == 9
    assert len(errors) == 1


# ============================================================================
# Performance Benchmark Tests
# ============================================================================

@pytest.mark.asyncio
async def test_pipeline_performance_improvement():
    """Test that async pipeline provides significant speedup."""
    pipeline = TestAsyncPipeline()
    num_samples = 50

    samples = await pipeline.load_data(num_samples)

    # Sequential processing
    start = time.time()
    for sample in samples[:10]:  # Test with 10 samples
        s = Sample(id=sample.id, data=sample.data)
        await pipeline.preprocess_sample(s)
        await pipeline.predict_sample(s)
    sequential_time = time.time() - start

    # Concurrent batch processing
    start = time.time()
    batch = samples[:10]
    processed = await pipeline.preprocess_batch(batch)
    predicted = await pipeline.predict_batch(processed)
    concurrent_time = time.time() - start

    speedup = sequential_time / concurrent_time
    assert speedup >= 5.0  # Should be ~10x but account for overhead


@pytest.mark.asyncio
async def test_pipeline_throughput():
    """Test pipeline throughput."""
    pipeline = TestAsyncPipeline(batch_size=20)
    num_samples = 100

    start = time.time()

    samples = await pipeline.load_data(num_samples)

    batches = [samples[i:i+20] for i in range(0, num_samples, 20)]

    all_processed = []
    for batch in batches:
        processed = await pipeline.preprocess_batch(batch)
        all_processed.extend(processed)

    all_predicted = []
    for batch in [all_processed[i:i+20] for i in range(0, len(all_processed), 20)]:
        predicted = await pipeline.predict_batch(batch)
        all_predicted.extend(predicted)

    elapsed = time.time() - start

    throughput = num_samples / elapsed
    assert throughput >= 100  # Should process at least 100 samples/sec


# ============================================================================
# Concurrent Pipeline Operations Tests
# ============================================================================

@pytest.mark.asyncio
async def test_multiple_pipelines_concurrent():
    """Test running multiple pipeline instances concurrently."""
    pipeline1 = TestAsyncPipeline()
    pipeline2 = TestAsyncPipeline()
    pipeline3 = TestAsyncPipeline()

    results = await asyncio.gather(
        pipeline1.load_data(30),
        pipeline2.load_data(40),
        pipeline3.load_data(50)
    )

    assert len(results) == 3
    assert len(results[0]) == 30
    assert len(results[1]) == 40
    assert len(results[2]) == 50


@pytest.mark.asyncio
async def test_pipeline_stages_concurrent():
    """Test overlapping pipeline stages."""
    pipeline = TestAsyncPipeline()

    # Start loading next batch while processing current batch
    samples_batch1 = await pipeline.load_data(20)

    # Process batch1 and load batch2 concurrently
    processed_batch1, samples_batch2 = await asyncio.gather(
        pipeline.preprocess_batch(samples_batch1),
        pipeline.load_data(20)
    )

    assert len(processed_batch1) == 20
    assert len(samples_batch2) == 20
    assert all(s.processed for s in processed_batch1)


# ============================================================================
# Resource Management Tests
# ============================================================================

@pytest.mark.asyncio
async def test_pipeline_cleanup():
    """Test proper cleanup of pipeline resources."""
    pipeline = TestAsyncPipeline()

    samples = await pipeline.load_data(10)
    processed = await pipeline.preprocess_batch(samples)
    predicted = await pipeline.predict_batch(processed)

    # Verify all samples were processed and predicted
    assert all(s.processed and s.predicted for s in predicted)
    assert len(predicted) == 10


@pytest.mark.asyncio
async def test_pipeline_with_semaphore():
    """Test rate limiting pipeline operations with semaphore."""
    semaphore = asyncio.Semaphore(5)  # Max 5 concurrent operations

    async def limited_process(sample: Sample) -> Sample:
        async with semaphore:
            await asyncio.sleep(0.01)
            sample.processed = True
            return sample

    samples = [Sample(id=i, data=[1.0] * 5) for i in range(20)]
    tasks = [limited_process(s) for s in samples]
    results = await asyncio.gather(*tasks)

    assert len(results) == 20
    assert all(s.processed for s in results)
