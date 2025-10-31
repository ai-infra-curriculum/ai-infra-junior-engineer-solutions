#!/usr/bin/env python3
"""End-to-end ML training pipeline integrating all components."""

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingestion import DataIngester
from src.data.validation import DataValidator
from src.data.preprocessing import DataPreprocessor
from src.features.engineering import FeatureEngineer
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(
    data_path: str,
    experiment_name: str,
    model_type: str = "random_forest",
    tune_hyperparameters: bool = False
):
    """Run complete training pipeline.

    Args:
        data_path: Path to training data (CSV, S3, or database)
        experiment_name: MLflow experiment name
        model_type: Type of model to train
        tune_hyperparameters: Whether to tune hyperparameters
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="end-to-end-pipeline"):
        print("=" * 70)
        print("MLOPS END-TO-END TRAINING PIPELINE")
        print("=" * 70)

        # Step 1: Data Ingestion
        print("\n[1/7] Data Ingestion...")
        logger.info(f"Loading data from: {data_path}")
        ingester = DataIngester()
        raw_data = ingester.load(data_path)
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("data_rows", len(raw_data))
        mlflow.log_param("data_columns", len(raw_data.columns))
        print(f"  ✓ Loaded {len(raw_data)} rows, {len(raw_data.columns)} columns")

        # Step 2: Data Validation
        print("\n[2/7] Data Validation...")
        logger.info("Validating data quality")
        validator = DataValidator(
            max_missing_ratio=0.3,
            max_duplicate_ratio=0.1
        )
        is_valid, issues = validator.validate(raw_data)

        # Log validation results
        mlflow.log_metric("validation_errors", sum(1 for i in issues if i.severity == "error"))
        mlflow.log_metric("validation_warnings", sum(1 for i in issues if i.severity == "warning"))

        if not is_valid:
            print("  ✗ Data validation failed!")
            print(validator.get_validation_report(issues))
            raise ValueError("Data validation failed. Cannot proceed with training.")

        print(f"  ✓ Data validation passed ({len(issues)} warnings)")

        # Step 3: Data Preprocessing
        print("\n[3/7] Data Preprocessing...")
        logger.info("Preprocessing data")
        preprocessor = DataPreprocessor()
        clean_data = preprocessor.process(
            raw_data,
            remove_duplicates=True,
            handle_missing=True,
            encode_categorical=True,
            scale_features=False  # Will scale in feature engineering
        )
        print(f"  ✓ Preprocessed {len(clean_data)} rows")

        # Step 4: Feature Engineering
        print("\n[4/7] Feature Engineering...")
        logger.info("Engineering features")

        # Separate features and target
        if 'target' in clean_data.columns:
            y = clean_data['target']
            X = clean_data.drop('target', axis=1)
        else:
            # Assume last column is target
            y = clean_data.iloc[:, -1]
            X = clean_data.iloc[:, :-1]

        engineer = FeatureEngineer()
        X_engineered = engineer.transform(
            X,
            create_polynomials=False,
            create_interactions=False,
            apply_pca_transform=False
        )

        mlflow.log_param("n_features", X_engineered.shape[1])
        print(f"  ✓ Engineered {X_engineered.shape[1]} features")

        # Step 5: Model Training
        print("\n[5/7] Model Training...")
        logger.info(f"Training {model_type} model")
        trainer = ModelTrainer(
            experiment_name=experiment_name,
            model_type=model_type
        )

        model, train_metrics = trainer.train(
            X_engineered,
            y,
            test_size=0.2,
            tune_hyperparameters=tune_hyperparameters,
            cv_folds=5 if tune_hyperparameters else 3
        )

        print(f"  ✓ Model trained - Accuracy: {train_metrics['accuracy']:.4f}")

        # Step 6: Model Evaluation
        print("\n[6/7] Model Evaluation...")
        logger.info("Evaluating model performance")
        evaluator = ModelEvaluator()

        # Check if model meets production criteria
        is_ready, readiness_message = evaluator.check_production_readiness(
            train_metrics,
            accuracy_threshold=0.75,
            precision_threshold=0.70,
            recall_threshold=0.70
        )

        print(f"  {readiness_message}")
        mlflow.log_metric("production_ready", 1 if is_ready else 0)

        # Step 7: Model Registration
        print("\n[7/7] Model Registration...")
        if is_ready:
            # Register model in MLflow
            model_name = f"{model_type.title()}Classifier"
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=model_name
            )
            print(f"  ✓ Model registered as '{model_name}'")
        else:
            print("  ✗ Model not registered (does not meet production criteria)")

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        # Return metrics for testing
        return {
            "is_valid": is_valid,
            "is_production_ready": is_ready,
            "metrics": train_metrics
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run end-to-end ML training pipeline"
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to training data (CSV file, S3 URI, or database connection string)"
    )
    parser.add_argument(
        "--experiment-name",
        default="end-to-end",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--model-type",
        default="random_forest",
        choices=["random_forest", "gradient_boosting", "logistic"],
        help="Type of model to train"
    )
    parser.add_argument(
        "--tune-hyperparameters",
        action="store_true",
        help="Enable hyperparameter tuning"
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            data_path=args.data_path,
            experiment_name=args.experiment_name,
            model_type=args.model_type,
            tune_hyperparameters=args.tune_hyperparameters
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
