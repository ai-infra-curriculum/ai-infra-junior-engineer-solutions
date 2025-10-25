"""Basic Text Generation

This script demonstrates basic LLM text generation using Hugging Face Transformers.
It shows how to load a model, generate text, and work with multiple outputs.
"""

import logging
from typing import List, Dict, Any
from transformers import pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_generator(model_name: str = 'gpt2', device: int = -1):
    """Initialize text generation pipeline.

    Args:
        model_name: Name of the model to load from Hugging Face
        device: Device to use (-1 for CPU, 0+ for GPU)

    Returns:
        Initialized text generation pipeline
    """
    logger.info(f"Loading model: {model_name}")
    generator = pipeline(
        'text-generation',
        model=model_name,
        device=device
    )
    logger.info("Model loaded successfully!")
    return generator


def generate_text(
    generator,
    prompt: str,
    max_length: int = 50,
    num_sequences: int = 3,
    temperature: float = 0.7,
    top_k: int = 50
) -> List[Dict[str, Any]]:
    """Generate text using the LLM.

    Args:
        generator: Text generation pipeline
        prompt: Input text prompt
        max_length: Maximum total tokens (prompt + generation)
        num_sequences: Number of different variations to generate
        temperature: Randomness (0=deterministic, 2=very creative)
        top_k: Consider top K probable tokens at each step

    Returns:
        List of generated text results
    """
    logger.info(f"Generating text for prompt: '{prompt}'")

    results = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=num_sequences,
        temperature=temperature,
        top_k=top_k,
        do_sample=True  # Enable sampling for creative outputs
    )

    logger.info(f"Generated {len(results)} variations")
    return results


def main():
    """Main execution function."""
    print("=" * 80)
    print("LLM Basic Text Generation Demo")
    print("=" * 80)

    # Initialize the model
    generator = initialize_generator(model_name='gpt2', device=-1)

    # Define prompt
    prompt = "Machine learning is"
    print(f"\nPrompt: {prompt}")
    print("-" * 80)

    # Generate text
    results = generate_text(
        generator=generator,
        prompt=prompt,
        max_length=50,
        num_sequences=3,
        temperature=0.7,
        top_k=50
    )

    # Display results
    for i, result in enumerate(results, 1):
        print(f"\nGeneration {i}:")
        print(result['generated_text'])
        print("-" * 80)

    # Additional examples
    print("\n" + "=" * 80)
    print("Additional Examples")
    print("=" * 80)

    example_prompts = [
        "Kubernetes is a",
        "The future of artificial intelligence",
        "Docker containers provide"
    ]

    for prompt in example_prompts:
        print(f"\nPrompt: {prompt}")
        results = generate_text(
            generator=generator,
            prompt=prompt,
            max_length=40,
            num_sequences=1,
            temperature=0.8
        )
        print(f"Generated: {results[0]['generated_text']}")


if __name__ == '__main__':
    main()
