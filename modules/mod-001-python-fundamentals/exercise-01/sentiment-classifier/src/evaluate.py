"""
Evaluation script for sentiment classifier.

Usage:
    python src/evaluate.py --model-path models/sentiment_model.pth
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Dict

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.data_loader import load_dataset, SentimentDataset
from utils.metrics import compute_metrics, plot_confusion_matrix

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def predict(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[List[int], List[int]]:
    """Make predictions on dataset."""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
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

    return all_predictions, all_labels


def main(args: argparse.Namespace) -> None:
    """Main evaluation function."""
    # Set device
    device = torch.device(os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    # Load model
    model_name = os.getenv("MODEL_NAME", "distilbert-base-uncased")
    logger.info(f"Loading model architecture: {model_name}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    # Load trained weights
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return

    logger.info(f"Loading model weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load test data
    data_path = os.getenv("DATA_PATH", "./data/sentiment_data.csv")
    logger.info(f"Loading test data from: {data_path}")

    _, test_df = load_dataset(data_path, test_size=0.2)

    test_dataset = SentimentDataset(
        test_df["text"].tolist(),
        test_df["label"].tolist(),
        tokenizer,
        max_length=int(os.getenv("MAX_LENGTH", 128)),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(os.getenv("BATCH_SIZE", 32)),
        shuffle=False,
    )

    logger.info(f"Test samples: {len(test_dataset)}")

    # Make predictions
    logger.info("Running evaluation...")
    predictions, labels = predict(model, test_loader, device)

    # Compute metrics
    metrics = compute_metrics(predictions, labels)

    # Display results
    logger.info("\n" + "="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1']:.4f}")
    logger.info("="*50)

    # Plot confusion matrix
    if args.plot:
        output_path = Path("./models/confusion_matrix.png")
        plot_confusion_matrix(labels, predictions, output_path)
        logger.info(f"✓ Confusion matrix saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sentiment classifier")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/sentiment_model.pth",
        help="Path to trained model",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot confusion matrix",
    )

    args = parser.parse_args()
    main(args)
