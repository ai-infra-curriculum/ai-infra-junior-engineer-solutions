#!/usr/bin/env python3
"""
Async ML Pipeline

Demonstrates building a complete asynchronous ML data pipeline.
"""

import asyncio
import time
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Sample:
    """Data sample for ML pipeline"""
    id: int
    data: List[float]
    processed: bool = False
    predicted: bool = False
    prediction: float = 0.0


class AsyncMLPipeline:
    """Asynchronous ML pipeline for efficient data processing"""

    def __init__(self, batch_size: int = 32):
        """
        Initialize pipeline.

        Args:
            batch_size: Number of samples to process in each batch
        """
        self.batch_size = batch_size

    async def load_data(self, num_samples: int) -> List[Sample]:
        """
        Load data asynchronously.

        Args:
            num_samples: Number of samples to load

        Returns:
            List of samples
        """
        print(f"  → Loading {num_samples} samples...")
        await asyncio.sleep(0.5)  # Simulate I/O

        samples = [
            Sample(id=i, data=[float(i) * 0.1 + j * 0.01 for j in range(10)])
            for i in range(num_samples)
        ]

        print(f"  ✓ Loaded {len(samples)} samples")
        return samples

    async def preprocess_sample(self, sample: Sample) -> Sample:
        """
        Preprocess single sample.

        Args:
            sample: Sample to preprocess

        Returns:
            Preprocessed sample
        """
        await asyncio.sleep(0.01)  # Simulate processing
        sample.processed = True
        # Normalize data
        sample.data = [x / 10.0 for x in sample.data]
        return sample

    async def preprocess_batch(self, samples: List[Sample]) -> List[Sample]:
        """
        Preprocess batch of samples concurrently.

        Args:
            samples: Batch of samples

        Returns:
            Preprocessed samples
        """
        tasks = [self.preprocess_sample(s) for s in samples]
        return await asyncio.gather(*tasks)

    async def predict_sample(self, sample: Sample) -> Sample:
        """
        Run inference on single sample.

        Args:
            sample: Sample to predict

        Returns:
            Sample with prediction
        """
        await asyncio.sleep(0.02)  # Simulate inference
        sample.predicted = True
        # Simple prediction (sum of features)
        sample.prediction = sum(sample.data)
        return sample

    async def predict_batch(self, samples: List[Sample]) -> List[Sample]:
        """
        Run inference on batch concurrently.

        Args:
            samples: Batch of samples

        Returns:
            Samples with predictions
        """
        tasks = [self.predict_sample(s) for s in samples]
        return await asyncio.gather(*tasks)

    async def run_pipeline(self, num_samples: int) -> Dict[str, Any]:
        """
        Run complete async pipeline.

        Args:
            num_samples: Number of samples to process

        Returns:
            Pipeline results with metrics
        """
        print("\n" + "=" * 70)
        print("Running Async ML Pipeline")
        print("=" * 70 + "\n")

        start_time = time.time()

        # Step 1: Load data
        samples = await self.load_data(num_samples)
        load_time = time.time() - start_time

        # Step 2: Preprocess in batches
        print(f"  → Preprocessing {len(samples)} samples in batches of {self.batch_size}...")
        preprocess_start = time.time()

        batches = [samples[i:i+self.batch_size]
                  for i in range(0, len(samples), self.batch_size)]

        preprocessed = []
        for i, batch in enumerate(batches, 1):
            batch_result = await self.preprocess_batch(batch)
            preprocessed.extend(batch_result)
            if i % 5 == 0 or i == len(batches):
                print(f"    Batch {i}/{len(batches)} complete")

        preprocess_time = time.time() - preprocess_start
        print(f"  ✓ Preprocessed {len(preprocessed)} samples")

        # Step 3: Predict in batches
        print(f"  → Running inference on {len(preprocessed)} samples...")
        predict_start = time.time()

        predicted = []
        for i, batch in enumerate([preprocessed[i:i+self.batch_size]
                                   for i in range(0, len(preprocessed), self.batch_size)], 1):
            batch_result = await self.predict_batch(batch)
            predicted.extend(batch_result)
            if i % 5 == 0 or i == len(batches):
                print(f"    Batch {i}/{len(batches)} complete")

        predict_time = time.time() - predict_start
        print(f"  ✓ Predicted {len(predicted)} samples")

        total_time = time.time() - start_time

        return {
            "total_samples": len(predicted),
            "time_total": total_time,
            "time_load": load_time,
            "time_preprocess": preprocess_time,
            "time_predict": predict_time,
            "throughput": len(predicted) / total_time,
            "avg_prediction": sum(s.prediction for s in predicted) / len(predicted)
        }


async def compare_batch_sizes():
    """Compare performance with different batch sizes"""
    print("\n" + "=" * 70)
    print("Batch Size Performance Comparison")
    print("=" * 70 + "\n")

    num_samples = 200
    batch_sizes = [16, 32, 64, 128]

    results = []
    for batch_size in batch_sizes:
        print(f"Testing batch_size={batch_size}")
        pipeline = AsyncMLPipeline(batch_size=batch_size)
        result = await pipeline.run_pipeline(num_samples)
        results.append((batch_size, result))
        print()

    print("=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print(f"{'Batch Size':<12} {'Total Time':<12} {'Throughput':<15}")
    print("-" * 70)

    for batch_size, result in results:
        print(f"{batch_size:<12} {result['time_total']:<12.2f} "
              f"{result['throughput']:<15.1f} samples/sec")

    print()


async def demonstrate_pipeline_stages():
    """Demonstrate pipeline stage breakdown"""
    print("\n" + "=" * 70)
    print("Pipeline Stage Analysis")
    print("=" * 70 + "\n")

    pipeline = AsyncMLPipeline(batch_size=32)
    result = await pipeline.run_pipeline(100)

    print("\n" + "=" * 70)
    print("Stage Timing Breakdown")
    print("=" * 70)
    print(f"Load:       {result['time_load']:.3f}s "
          f"({100 * result['time_load']/result['time_total']:.1f}%)")
    print(f"Preprocess: {result['time_preprocess']:.3f}s "
          f"({100 * result['time_preprocess']/result['time_total']:.1f}%)")
    print(f"Predict:    {result['time_predict']:.3f}s "
          f"({100 * result['time_predict']/result['time_total']:.1f}%)")
    print(f"Total:      {result['time_total']:.3f}s")
    print()
    print(f"Throughput: {result['throughput']:.1f} samples/second")
    print(f"Avg prediction: {result['avg_prediction']:.3f}")
    print()


async def main():
    """Run all demonstrations"""
    print("\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  Async ML Pipeline".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    # Run basic pipeline
    await demonstrate_pipeline_stages()

    # Compare batch sizes
    await compare_batch_sizes()

    # Summary
    print("=" * 70)
    print("Key Takeaways")
    print("=" * 70)
    print("1. Async pipeline processes data concurrently within batches")
    print("2. Batch size affects performance (too small=overhead, too large=memory)")
    print("3. Each stage can be optimized independently")
    print("4. Concurrent preprocessing and inference provide significant speedup")
    print("5. Monitor stage timing to identify bottlenecks")
    print("=" * 70)
    print()


if __name__ == "__main__":
    asyncio.run(main())
