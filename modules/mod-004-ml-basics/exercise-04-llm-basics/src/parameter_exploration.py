"""Parameter Exploration

This script explores how different parameters affect LLM text generation.
It demonstrates the effects of temperature, max_length, top_k, and top_p.
"""

import logging
from typing import Dict, Any
from transformers import pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ParameterExplorer:
    """Explore LLM generation parameters and their effects."""

    def __init__(self, model_name: str = 'gpt2', device: int = -1):
        """Initialize the parameter explorer.

        Args:
            model_name: Model to use for exploration
            device: Device to use (-1 for CPU, 0+ for GPU)
        """
        logger.info(f"Initializing ParameterExplorer with model: {model_name}")
        self.generator = pipeline('text-generation', model=model_name, device=device)
        self.model_name = model_name

    def explore_temperature(self, prompt: str, max_length: int = 30) -> Dict[str, str]:
        """Explore the effect of temperature on generation.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length

        Returns:
            Dictionary mapping temperature values to generated text
        """
        print("\n" + "=" * 80)
        print("EXPERIMENT 1: Temperature Effect")
        print("=" * 80)
        print(f"Prompt: '{prompt}'")
        print(f"Max Length: {max_length}")
        print("-" * 80)

        temperatures = [0.1, 0.5, 0.7, 1.0, 1.5]
        results = {}

        for temp in temperatures:
            logger.info(f"Testing temperature: {temp}")
            result = self.generator(
                prompt,
                max_length=max_length,
                temperature=temp,
                num_return_sequences=1,
                do_sample=True
            )
            generated = result[0]['generated_text']
            results[f"temp_{temp}"] = generated

            print(f"\nTemperature={temp} ({'deterministic' if temp <= 0.3 else 'creative' if temp >= 1.2 else 'balanced'}):")
            print(f"{generated}")
            print("-" * 80)

        return results

    def explore_max_length(self, prompt: str, temperature: float = 0.7) -> Dict[str, str]:
        """Explore the effect of max_length on generation.

        Args:
            prompt: Input prompt
            temperature: Temperature to use

        Returns:
            Dictionary mapping length values to generated text
        """
        print("\n" + "=" * 80)
        print("EXPERIMENT 2: Max Length Effect")
        print("=" * 80)
        print(f"Prompt: '{prompt}'")
        print(f"Temperature: {temperature}")
        print("-" * 80)

        lengths = [20, 40, 60, 80, 100]
        results = {}

        for length in lengths:
            logger.info(f"Testing max_length: {length}")
            result = self.generator(
                prompt,
                max_length=length,
                temperature=temperature,
                num_return_sequences=1,
                do_sample=True
            )
            generated = result[0]['generated_text']
            results[f"length_{length}"] = generated

            print(f"\nMax Length={length}:")
            print(f"{generated}")
            print("-" * 80)

        return results

    def explore_top_k(self, prompt: str, max_length: int = 40) -> Dict[str, str]:
        """Explore the effect of top_k on generation.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length

        Returns:
            Dictionary mapping top_k values to generated text
        """
        print("\n" + "=" * 80)
        print("EXPERIMENT 3: Top-K Effect")
        print("=" * 80)
        print(f"Prompt: '{prompt}'")
        print(f"Max Length: {max_length}")
        print("-" * 80)

        top_k_values = [10, 30, 50, 100]
        results = {}

        for top_k in top_k_values:
            logger.info(f"Testing top_k: {top_k}")
            result = self.generator(
                prompt,
                max_length=max_length,
                temperature=0.7,
                top_k=top_k,
                num_return_sequences=1,
                do_sample=True
            )
            generated = result[0]['generated_text']
            results[f"top_k_{top_k}"] = generated

            print(f"\nTop-K={top_k}:")
            print(f"{generated}")
            print("-" * 80)

        return results

    def explore_top_p(self, prompt: str, max_length: int = 40) -> Dict[str, str]:
        """Explore the effect of top_p (nucleus sampling) on generation.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length

        Returns:
            Dictionary mapping top_p values to generated text
        """
        print("\n" + "=" * 80)
        print("EXPERIMENT 4: Top-P (Nucleus Sampling) Effect")
        print("=" * 80)
        print(f"Prompt: '{prompt}'")
        print(f"Max Length: {max_length}")
        print("-" * 80)

        top_p_values = [0.5, 0.7, 0.9, 0.95, 1.0]
        results = {}

        for top_p in top_p_values:
            logger.info(f"Testing top_p: {top_p}")
            result = self.generator(
                prompt,
                max_length=max_length,
                temperature=0.7,
                top_p=top_p,
                top_k=0,  # Disable top_k to isolate top_p effect
                num_return_sequences=1,
                do_sample=True
            )
            generated = result[0]['generated_text']
            results[f"top_p_{top_p}"] = generated

            print(f"\nTop-P={top_p}:")
            print(f"{generated}")
            print("-" * 80)

        return results

    def compare_sampling_strategies(self, prompt: str, max_length: int = 40):
        """Compare different sampling strategies.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length
        """
        print("\n" + "=" * 80)
        print("EXPERIMENT 5: Sampling Strategy Comparison")
        print("=" * 80)
        print(f"Prompt: '{prompt}'")
        print(f"Max Length: {max_length}")
        print("-" * 80)

        strategies = [
            {"name": "Greedy (no sampling)", "do_sample": False},
            {"name": "Temperature sampling (0.7)", "do_sample": True, "temperature": 0.7},
            {"name": "Top-K sampling (50)", "do_sample": True, "top_k": 50, "temperature": 1.0},
            {"name": "Top-P sampling (0.9)", "do_sample": True, "top_p": 0.9, "top_k": 0, "temperature": 1.0},
            {"name": "Combined (temp=0.8, top_k=50, top_p=0.95)", "do_sample": True, "temperature": 0.8, "top_k": 50, "top_p": 0.95}
        ]

        for strategy in strategies:
            name = strategy.pop("name")
            logger.info(f"Testing strategy: {name}")

            result = self.generator(
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                **strategy
            )
            generated = result[0]['generated_text']

            print(f"\nStrategy: {name}")
            print(f"Generated: {generated}")
            print("-" * 80)


def main():
    """Main execution function."""
    print("=" * 80)
    print("LLM Parameter Exploration")
    print("=" * 80)

    # Initialize explorer
    explorer = ParameterExplorer(model_name='gpt2', device=-1)

    # Run experiments
    base_prompt = "The future of AI is"

    # Experiment 1: Temperature
    explorer.explore_temperature(base_prompt, max_length=30)

    # Experiment 2: Max Length
    explorer.explore_max_length(base_prompt, temperature=0.7)

    # Experiment 3: Top-K
    explorer.explore_top_k(base_prompt, max_length=40)

    # Experiment 4: Top-P
    explorer.explore_top_p(base_prompt, max_length=40)

    # Experiment 5: Sampling Strategies
    explorer.compare_sampling_strategies(base_prompt, max_length=40)

    print("\n" + "=" * 80)
    print("Parameter Exploration Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("- Temperature controls randomness (lower = more deterministic)")
    print("- Max length controls output length")
    print("- Top-K limits vocabulary to top K probable tokens")
    print("- Top-P (nucleus sampling) uses cumulative probability threshold")
    print("- Different strategies can be combined for fine-tuned control")


if __name__ == '__main__':
    main()
