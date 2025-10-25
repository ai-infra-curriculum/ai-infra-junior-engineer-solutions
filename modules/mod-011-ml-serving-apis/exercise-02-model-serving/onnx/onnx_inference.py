"""ONNX Runtime inference."""

import time
import numpy as np
import onnxruntime as ort


class ONNXPredictor:
    """ONNX Runtime predictor."""

    def __init__(self, model_path, use_gpu=False):
        """
        Initialize ONNX predictor.

        Args:
            model_path: Path to ONNX model
            use_gpu: Whether to use GPU acceleration
        """
        self.model_path = model_path

        # Set execution providers
        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # Create inference session
        self.session = ort.InferenceSession(
            model_path,
            providers=providers
        )

        # Get model info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"Model loaded: {model_path}")
        print(f"Input: {self.input_name}")
        print(f"Output: {self.output_name}")
        print(f"Providers: {self.session.get_providers()}")

    def predict(self, data):
        """
        Run inference.

        Args:
            data: Input data (numpy array or list)

        Returns:
            Predictions
        """
        # Convert to numpy array if needed
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)

        # Ensure correct shape
        if len(data.shape) == 1:
            data = data.reshape(1, -1)

        # Run inference
        result = self.session.run(
            [self.output_name],
            {self.input_name: data}
        )

        return result[0]

    def predict_batch(self, batch_data):
        """
        Run batch inference.

        Args:
            batch_data: Batch of input data

        Returns:
            Batch predictions
        """
        data = np.array(batch_data, dtype=np.float32)
        return self.predict(data)

    def benchmark(self, n_iterations=1000):
        """
        Benchmark inference performance.

        Args:
            n_iterations: Number of iterations

        Returns:
            Performance metrics
        """
        # Generate random data
        data = np.random.rand(1, 5).astype(np.float32)

        # Warmup
        for _ in range(10):
            self.predict(data)

        # Benchmark
        latencies = []
        for _ in range(n_iterations):
            start = time.time()
            self.predict(data)
            latencies.append((time.time() - start) * 1000)  # ms

        return {
            "mean_ms": np.mean(latencies),
            "median_ms": np.median(latencies),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
            "throughput_rps": 1000 / np.mean(latencies)
        }


def main():
    """Demo ONNX inference."""
    print("="*50)
    print("ONNX Runtime Inference Demo")
    print("="*50)

    # Load model
    predictor = ONNXPredictor("model_sklearn.onnx")

    # Single prediction
    print("\n1. Single Prediction")
    print("-"*50)
    data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
    result = predictor.predict(data)
    print(f"Input: {data}")
    print(f"Output: {result}")

    # Batch prediction
    print("\n2. Batch Prediction")
    print("-"*50)
    batch_data = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0, 5.0, 6.0],
        [3.0, 4.0, 5.0, 6.0, 7.0]
    ]
    results = predictor.predict_batch(batch_data)
    print(f"Batch size: {len(batch_data)}")
    print(f"Results shape: {results.shape}")
    print(f"Results: {results}")

    # Benchmark
    print("\n3. Performance Benchmark")
    print("-"*50)
    metrics = predictor.benchmark(n_iterations=1000)
    print(f"Mean latency: {metrics['mean_ms']:.3f} ms")
    print(f"Median latency: {metrics['median_ms']:.3f} ms")
    print(f"P95 latency: {metrics['p95_ms']:.3f} ms")
    print(f"P99 latency: {metrics['p99_ms']:.3f} ms")
    print(f"Throughput: {metrics['throughput_rps']:.0f} req/sec")

    print("\n" + "="*50)
    print("Demo complete!")
    print("="*50)


if __name__ == "__main__":
    main()
