"""Model Comparison

This script compares different LLM models in terms of size, speed, and quality.
It helps understand the trade-offs when selecting models for deployment.
"""

import logging
import time
import psutil
from typing import Dict, Any, List, Optional
from transformers import pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare different LLM models."""

    def __init__(self, device: int = -1):
        """Initialize the model comparator.

        Args:
            device: Device to use (-1 for CPU, 0+ for GPU)
        """
        self.device = device
        self.process = psutil.Process()

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def test_model(
        self,
        model_name: str,
        test_prompt: str = "Hello, I am",
        max_length: int = 50
    ) -> Dict[str, Any]:
        """Test a single model and collect metrics.

        Args:
            model_name: Name of the model to test
            test_prompt: Prompt to use for testing
            max_length: Maximum generation length

        Returns:
            Dictionary with model metrics
        """
        print(f"\n{'=' * 80}")
        print(f"Testing: {model_name}")
        print(f"{'=' * 80}")

        metrics = {
            "model_name": model_name,
            "success": False,
            "error": None
        }

        try:
            # Measure memory before loading
            mem_before = self.get_memory_usage()

            # Load model
            logger.info(f"Loading model: {model_name}")
            load_start = time.time()

            generator = pipeline(
                'text-generation',
                model=model_name,
                device=self.device
            )

            load_time = time.time() - load_start
            mem_after = self.get_memory_usage()
            model_memory = mem_after - mem_before

            logger.info(f"Model loaded in {load_time:.2f}s, using {model_memory:.2f} MB")

            # Run inference test
            logger.info("Running inference test")
            inference_start = time.time()

            result = generator(
                test_prompt,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )

            inference_time = time.time() - inference_start
            generated_text = result[0]['generated_text']

            # Calculate tokens per second (rough estimate)
            estimated_tokens = max_length - len(test_prompt.split())
            tokens_per_second = estimated_tokens / inference_time if inference_time > 0 else 0

            # Update metrics
            metrics.update({
                "success": True,
                "load_time_seconds": load_time,
                "memory_mb": model_memory,
                "inference_time_seconds": inference_time,
                "tokens_per_second": tokens_per_second,
                "generated_text": generated_text,
                "output_length": len(generated_text)
            })

            # Print results
            print(f"✓ Model loaded successfully")
            print(f"  Load time: {load_time:.2f}s")
            print(f"  Memory usage: {model_memory:.2f} MB")
            print(f"  Inference time: {inference_time:.2f}s")
            print(f"  Tokens/second: {tokens_per_second:.1f}")
            print(f"  Output preview: {generated_text[:100]}...")

        except Exception as e:
            logger.error(f"Error testing model {model_name}: {str(e)}")
            metrics["error"] = str(e)
            print(f"✗ Error: {str(e)}")

        return metrics

    def compare_models(
        self,
        model_list: List[str],
        test_prompt: str = "The future of artificial intelligence",
        max_length: int = 50
    ) -> List[Dict[str, Any]]:
        """Compare multiple models.

        Args:
            model_list: List of model names to compare
            test_prompt: Prompt to use for testing
            max_length: Maximum generation length

        Returns:
            List of metrics for each model
        """
        print("=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        print(f"Test Prompt: '{test_prompt}'")
        print(f"Max Length: {max_length}")
        print(f"Models to test: {len(model_list)}")

        results = []

        for i, model_name in enumerate(model_list, 1):
            print(f"\n[{i}/{len(model_list)}] Testing {model_name}...")

            metrics = self.test_model(
                model_name=model_name,
                test_prompt=test_prompt,
                max_length=max_length
            )
            results.append(metrics)

            # Cool down between models
            if i < len(model_list):
                time.sleep(2)

        return results

    def print_comparison_summary(self, results: List[Dict[str, Any]]):
        """Print a summary comparison table.

        Args:
            results: List of model metrics
        """
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)

        # Filter successful results
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        if successful:
            print("\nSuccessful Models:")
            print("-" * 80)
            print(f"{'Model':<20} {'Memory (MB)':<15} {'Load (s)':<12} {'Inference (s)':<15} {'Tokens/s':<10}")
            print("-" * 80)

            for result in successful:
                print(
                    f"{result['model_name']:<20} "
                    f"{result['memory_mb']:<15.2f} "
                    f"{result['load_time_seconds']:<12.2f} "
                    f"{result['inference_time_seconds']:<15.2f} "
                    f"{result['tokens_per_second']:<10.1f}"
                )

            # Find best performers
            print("\n" + "-" * 80)
            print("Best Performers:")

            fastest_load = min(successful, key=lambda x: x['load_time_seconds'])
            print(f"  Fastest load: {fastest_load['model_name']} ({fastest_load['load_time_seconds']:.2f}s)")

            fastest_inference = min(successful, key=lambda x: x['inference_time_seconds'])
            print(f"  Fastest inference: {fastest_inference['model_name']} ({fastest_inference['inference_time_seconds']:.2f}s)")

            smallest_memory = min(successful, key=lambda x: x['memory_mb'])
            print(f"  Smallest memory: {smallest_memory['model_name']} ({smallest_memory['memory_mb']:.2f} MB)")

            highest_throughput = max(successful, key=lambda x: x['tokens_per_second'])
            print(f"  Highest throughput: {highest_throughput['model_name']} ({highest_throughput['tokens_per_second']:.1f} tokens/s)")

        if failed:
            print("\n" + "-" * 80)
            print(f"Failed Models ({len(failed)}):")
            for result in failed:
                print(f"  - {result['model_name']}: {result['error']}")

        print("\n" + "=" * 80)
        print("Recommendations:")
        print("=" * 80)

        if successful:
            print("\nFor different use cases:")
            print("  - Low latency: Choose models with fast inference time")
            print("  - Limited memory: Choose models with small memory footprint")
            print("  - High throughput: Choose models with high tokens/second")
            print("  - Production: Balance memory, speed, and quality")

            print("\nGeneral tips:")
            print("  - Smaller models (distilgpt2) are faster but less capable")
            print("  - Larger models (gpt2-medium/large) are slower but more coherent")
            print("  - GPU acceleration can improve speed 10-50x")
            print("  - Consider model quantization to reduce memory usage")


def main():
    """Main execution function."""
    print("=" * 80)
    print("LLM MODEL COMPARISON")
    print("=" * 80)

    # Initialize comparator
    comparator = ModelComparator(device=-1)

    # Define models to compare
    # Note: Larger models may be very slow on CPU
    models_to_test = [
        "gpt2",              # 124M parameters (~500MB)
        "distilgpt2",        # 82M parameters (~350MB) - faster, smaller
        # Uncomment if you have enough resources and patience:
        # "gpt2-medium",     # 355M parameters (~1.5GB) - slower on CPU
    ]

    print(f"\nModels selected for comparison:")
    for model in models_to_test:
        print(f"  - {model}")

    # Run comparison
    results = comparator.compare_models(
        model_list=models_to_test,
        test_prompt="The future of artificial intelligence",
        max_length=50
    )

    # Print summary
    comparator.print_comparison_summary(results)

    print("\n" + "=" * 80)
    print("Comparison Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
