"""
Data loading utilities for sentiment classifier.
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def load_dataset(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and split sentiment dataset.

    Args:
        data_path: Path to CSV file with 'text' and 'label' columns
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, test_df)

    Expected CSV format:
        text,label
        "Great product!",1
        "Terrible experience.",0
    """
    # Check if file exists
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Please create a CSV file with 'text' and 'label' columns."
        )

    # Load data
    df = pd.read_csv(data_path)

    # Validate columns
    required_columns = ["text", "label"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            f"CSV must contain {required_columns} columns. "
            f"Found: {df.columns.tolist()}"
        )

    # Remove missing values
    df = df.dropna(subset=required_columns)

    # Split data
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],  # Maintain label distribution
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment classification."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
    ):
        """
        Initialize sentiment dataset.

        Args:
            texts: List of text samples
            labels: List of labels (0 or 1)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single item from dataset.

        Args:
            idx: Index of item

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # Tokenize text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }
