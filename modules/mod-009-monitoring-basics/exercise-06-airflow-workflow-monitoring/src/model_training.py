"""
Model Training Tasks

Contains all model training related tasks:
- Feature engineering
- Model training
- Model evaluation
"""

import time
import random
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def feature_engineering(**context):
    """
    Create additional features from preprocessed data.

    Creates:
    - Aggregation features
    - Interaction features
    - Temporal features
    - Domain-specific transformations

    Returns:
        dict: Feature engineering results
    """
    logger.info("=" * 60)
    logger.info("Starting feature engineering...")
    logger.info("=" * 60)

    # Get data from previous task
    ti = context['ti']
    preprocessing_results = ti.xcom_pull(task_ids='preprocess_data')

    if not preprocessing_results:
        raise ValueError("No preprocessing results found")

    num_base_features = preprocessing_results['num_features']
    logger.info(f"Base features: {num_base_features}")

    # Create aggregation features
    logger.info("Creating aggregation features...")
    time.sleep(1)
    agg_features = random.randint(10, 20)
    logger.info(f"  - Sum, mean, max, min aggregations")
    logger.info(f"  - Generated {agg_features} features")

    # Create interaction features
    logger.info("Creating interaction features...")
    time.sleep(1)
    interaction_features = random.randint(15, 25)
    logger.info(f"  - Polynomial combinations")
    logger.info(f"  - Generated {interaction_features} features")

    # Create temporal features
    logger.info("Creating temporal features...")
    time.sleep(1)
    temporal_features = random.randint(5, 10)
    logger.info(f"  - Day of week, month, quarter")
    logger.info(f"  - Time since last event")
    logger.info(f"  - Generated {temporal_features} features")

    # Create domain-specific features
    logger.info("Creating domain-specific features...")
    time.sleep(1)
    domain_features = random.randint(8, 15)
    logger.info(f"  - Custom business logic features")
    logger.info(f"  - Generated {domain_features} features")

    total_features = (
        num_base_features +
        agg_features +
        interaction_features +
        temporal_features +
        domain_features
    )

    feature_results = {
        'base_features': num_base_features,
        'aggregation_features': agg_features,
        'interaction_features': interaction_features,
        'temporal_features': temporal_features,
        'domain_features': domain_features,
        'total_features': total_features,
        'feature_importance_available': True
    }

    logger.info("=" * 60)
    logger.info("Feature engineering completed")
    logger.info(f"Total features: {total_features}")
    logger.info(f"Features added: {total_features - num_base_features}")
    logger.info("=" * 60)

    return feature_results


def train_model(**context):
    """
    Train machine learning model.

    Steps:
    - Initialize model
    - Train on training set
    - Generate training metrics
    - Save model artifacts

    Returns:
        dict: Training results including metrics and model info
    """
    logger.info("=" * 60)
    logger.info("Starting model training...")
    logger.info("=" * 60)

    # Get data from previous tasks
    ti = context['ti']
    feature_results = ti.xcom_pull(task_ids='feature_engineering')
    preprocessing_results = ti.xcom_pull(task_ids='preprocess_data')

    if not feature_results or not preprocessing_results:
        raise ValueError("Missing required data from previous tasks")

    train_samples = preprocessing_results['train_samples']
    num_features = feature_results['total_features']

    logger.info(f"Training samples: {train_samples:,}")
    logger.info(f"Features: {num_features}")

    # Model configuration
    model_type = "GradientBoostingClassifier"
    logger.info(f"Model type: {model_type}")
    logger.info("Hyperparameters:")
    logger.info("  - n_estimators: 100")
    logger.info("  - max_depth: 5")
    logger.info("  - learning_rate: 0.1")
    logger.info("  - min_samples_split: 20")

    # Simulate training
    logger.info("")
    logger.info("Training progress:")

    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        time.sleep(0.5)  # Simulate training time
        train_loss = 1.0 - (epoch / num_epochs * 0.7)  # Decreasing loss
        logger.info(f"  Epoch {epoch}/{num_epochs} - Loss: {train_loss:.4f}")

    logger.info("")
    logger.info("Training completed!")

    # Generate metrics
    training_time_minutes = random.uniform(30, 60)
    train_accuracy = random.uniform(0.92, 0.98)
    val_accuracy = random.uniform(0.88, 0.94)

    logger.info("")
    logger.info("Training metrics:")
    logger.info(f"  - Training accuracy: {train_accuracy:.4f}")
    logger.info(f"  - Validation accuracy: {val_accuracy:.4f}")
    logger.info(f"  - Training time: {training_time_minutes:.1f} minutes")

    # Model artifacts
    model_size_mb = random.randint(200, 600)
    logger.info("")
    logger.info("Model artifacts:")
    logger.info(f"  - Model size: {model_size_mb}MB")
    logger.info(f"  - Format: pickle")
    logger.info(f"  - Location: /models/model_{context.get('ds', 'latest')}.pkl")

    training_results = {
        'model_type': model_type,
        'train_samples': train_samples,
        'num_features': num_features,
        'train_accuracy': round(train_accuracy, 4),
        'validation_accuracy': round(val_accuracy, 4),
        'training_time_minutes': round(training_time_minutes, 2),
        'model_size_mb': model_size_mb,
        'n_estimators': 100,
        'max_depth': 5,
        'model_path': f"/models/model_{context.get('ds', 'latest')}.pkl"
    }

    logger.info("=" * 60)
    logger.info("Model training completed successfully")
    logger.info("=" * 60)

    return training_results


def evaluate_model(**context):
    """
    Evaluate model performance on test set.

    Metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - AUC-ROC
    - Confusion matrix

    Raises:
        ValueError: If model doesn't meet minimum performance thresholds

    Returns:
        dict: Evaluation metrics
    """
    logger.info("=" * 60)
    logger.info("Starting model evaluation...")
    logger.info("=" * 60)

    # Get training results
    ti = context['ti']
    training_results = ti.xcom_pull(task_ids='train_model')
    preprocessing_results = ti.xcom_pull(task_ids='preprocess_data')

    if not training_results or not preprocessing_results:
        raise ValueError("Missing required data from previous tasks")

    val_samples = preprocessing_results['validation_samples']
    logger.info(f"Evaluating on {val_samples:,} validation samples...")

    # Simulate evaluation
    time.sleep(2)

    # Generate metrics (slightly varied from training)
    base_accuracy = training_results['validation_accuracy']
    accuracy = base_accuracy + random.uniform(-0.02, 0.01)
    precision = accuracy + random.uniform(-0.02, 0.02)
    recall = accuracy + random.uniform(-0.02, 0.02)
    f1_score = 2 * (precision * recall) / (precision + recall)
    auc_roc = accuracy + random.uniform(0, 0.05)

    # Ensure values are in valid range
    accuracy = max(0, min(1, accuracy))
    precision = max(0, min(1, precision))
    recall = max(0, min(1, recall))
    f1_score = max(0, min(1, f1_score))
    auc_roc = max(0, min(1, auc_roc))

    logger.info("")
    logger.info("Evaluation Metrics:")
    logger.info("=" * 40)
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1 Score:  {f1_score:.4f}")
    logger.info(f"AUC-ROC:   {auc_roc:.4f}")
    logger.info("=" * 40)

    # Confusion matrix (simulated)
    logger.info("")
    logger.info("Confusion Matrix:")
    tp = int(val_samples * 0.4 * recall)
    fn = int(val_samples * 0.4 * (1 - recall))
    tn = int(val_samples * 0.6 * precision)
    fp = int(val_samples * 0.6 * (1 - precision))

    logger.info("              Predicted")
    logger.info("              Pos    Neg")
    logger.info(f"Actual Pos  {tp:5d}  {fn:5d}")
    logger.info(f"       Neg  {fp:5d}  {tn:5d}")

    # Quality gates
    logger.info("")
    logger.info("Quality Gates:")

    min_accuracy = 0.88
    min_f1 = 0.85

    if accuracy < min_accuracy:
        logger.error(f"FAILED: Accuracy {accuracy:.4f} < {min_accuracy}")
        raise ValueError(
            f"Model accuracy {accuracy:.4f} below threshold {min_accuracy}"
        )

    if f1_score < min_f1:
        logger.error(f"FAILED: F1 Score {f1_score:.4f} < {min_f1}")
        raise ValueError(
            f"Model F1 score {f1_score:.4f} below threshold {min_f1}"
        )

    logger.info(f"  - Accuracy >= {min_accuracy}: PASSED")
    logger.info(f"  - F1 Score >= {min_f1}: PASSED")

    evaluation_results = {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1_score, 4),
        'auc_roc': round(auc_roc, 4),
        'confusion_matrix': {
            'true_positives': tp,
            'false_negatives': fn,
            'true_negatives': tn,
            'false_positives': fp
        },
        'test_samples': val_samples,
        'quality_gates_passed': True,
        'min_accuracy_threshold': min_accuracy,
        'min_f1_threshold': min_f1
    }

    logger.info("")
    logger.info("=" * 60)
    logger.info("Model evaluation PASSED")
    logger.info("Model is ready for deployment")
    logger.info("=" * 60)

    return evaluation_results
