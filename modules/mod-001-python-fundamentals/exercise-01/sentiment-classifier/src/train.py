"""
Training script for sentiment classifier.

Usage:
    python src/train.py --config configs/training_config.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)

from utils.data_loader import load_dataset, SentimentDataset
from utils.metrics import compute_metrics

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override with environment variables if present
    config["model_name"] = os.getenv("MODEL_NAME", config["model_name"])
    config["batch_size"] = int(os.getenv("BATCH_SIZE", config["batch_size"]))
    config["learning_rate"] = float(os.getenv("LEARNING_RATE", config["learning_rate"]))
    config["num_epochs"] = int(os.getenv("NUM_EPOCHS", config["num_epochs"]))
    config["random_seed"] = int(os.getenv("RANDOM_SEED", config.get("random_seed", 42)))

    return config


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader)


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_predictions, all_labels)
    return metrics


def main(args: argparse.Namespace) -> None:
    """Main training function."""
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Configuration loaded: {config}")

    # Set seed for reproducibility
    set_seed(config["random_seed"])
    logger.info(f"Random seed set to {config['random_seed']}")

    # Set device
    device = torch.device(os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    # Load tokenizer and model
    logger.info(f"Loading model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=2,  # Binary sentiment classification
    )
    model.to(device)

    # Load and prepare data
    data_path = os.getenv("DATA_PATH", config.get("data_path", "./data/sentiment_data.csv"))
    logger.info(f"Loading data from: {data_path}")

    train_df, val_df = load_dataset(data_path, test_size=0.2)

    train_dataset = SentimentDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        max_length=int(os.getenv("MAX_LENGTH", 128)),
    )

    val_dataset = SentimentDataset(
        val_df["text"].tolist(),
        val_df["label"].tolist(),
        tokenizer,
        max_length=int(os.getenv("MAX_LENGTH", 128)),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
    )

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    total_steps = len(train_loader) * config["num_epochs"]
    warmup_steps = int(os.getenv("WARMUP_STEPS", 500))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Training loop
    logger.info("Starting training...")
    best_accuracy = 0.0

    for epoch in range(config["num_epochs"]):
        logger.info(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        logger.info(f"Train loss: {train_loss:.4f}")

        # Evaluate
        metrics = evaluate(model, val_loader, device)
        logger.info(f"Validation metrics: {metrics}")

        # Save best model
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            output_dir = Path(os.getenv("MODEL_OUTPUT_PATH", "./models"))
            output_dir.mkdir(parents=True, exist_ok=True)

            model_path = output_dir / "sentiment_model.pth"
            torch.save(model.state_dict(), model_path)
            logger.info(f"✓ Best model saved to {model_path} (accuracy: {best_accuracy:.4f})")

    logger.info("\n" + "="*50)
    logger.info(f"Training complete! Best accuracy: {best_accuracy:.4f}")
    logger.info("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration file",
    )

    args = parser.parse_args()
    main(args)
