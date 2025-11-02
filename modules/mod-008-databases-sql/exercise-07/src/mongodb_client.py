"""
MongoDB Client for ML Platform

Handles flexible, document-based ML metadata with varying schemas.
Use cases:
- Model configurations (different frameworks have different parameters)
- Experiment tracking with nested parameters
- Rapidly evolving data structures
- Hierarchical metadata
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError, PyMongoError
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB configuration
MONGO_URL = "mongodb://mluser:mlpass123@localhost:27017/"
DATABASE_NAME = "ml_metadata"

# Create MongoDB client
client = MongoClient(MONGO_URL)
db = client[DATABASE_NAME]


def test_connection() -> bool:
    """Test MongoDB connection."""
    try:
        info = client.server_info()
        logger.info(f"✓ MongoDB connected: v{info['version']}")
        return True
    except Exception as e:
        logger.error(f"✗ MongoDB connection failed: {e}")
        return False


# ============================================================================
# COLLECTION SETUP
# ============================================================================

def setup_collections():
    """
    Create collections with schema validation and indexes.
    MongoDB allows flexible schemas but can enforce basic validation.
    """
    # Collection: model_configs
    try:
        db.create_collection("model_configs", validator={
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["model_id", "model_name", "framework"],
                "properties": {
                    "model_id": {"bsonType": "string"},
                    "model_name": {"bsonType": "string"},
                    "framework": {"bsonType": "string"},
                    "config": {"bsonType": "object"},  # Flexible nested config
                    "metrics": {"bsonType": "object"},
                    "created_at": {"bsonType": "date"},
                    "updated_at": {"bsonType": "date"}
                }
            }
        })
        logger.info("✓ Created model_configs collection with validation")
    except Exception:
        logger.info("model_configs collection already exists")

    # Collection: experiments
    try:
        db.create_collection("experiments")
        logger.info("✓ Created experiments collection")
    except Exception:
        logger.info("experiments collection already exists")

    # Collection: feature_metadata
    try:
        db.create_collection("feature_metadata")
        logger.info("✓ Created feature_metadata collection")
    except Exception:
        logger.info("feature_metadata collection already exists")

    # Create indexes
    db.model_configs.create_index("model_id", unique=True)
    db.model_configs.create_index("framework")
    db.model_configs.create_index([("metrics.accuracy", DESCENDING)])
    db.model_configs.create_index([("model_name", ASCENDING), ("created_at", DESCENDING)])

    db.experiments.create_index([("model_id", ASCENDING), ("experiment_date", DESCENDING)])
    db.experiments.create_index("status")
    db.experiments.create_index([("metrics.accuracy", DESCENDING)])

    db.feature_metadata.create_index("dataset_name", unique=True)

    logger.info("✓ MongoDB indexes created")


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

def insert_model_config(config: Dict[str, Any]) -> str:
    """
    Insert model configuration with flexible schema.
    Different frameworks can have completely different config structures.
    """
    try:
        # Add timestamps
        config["created_at"] = datetime.utcnow()
        config["updated_at"] = datetime.utcnow()

        result = db.model_configs.insert_one(config)
        logger.info(f"✓ Inserted model config: {config['model_id']}")
        return str(result.inserted_id)
    except DuplicateKeyError:
        logger.warning(f"Model config '{config['model_id']}' already exists")
        return None
    except Exception as e:
        logger.error(f"Failed to insert model config: {e}")
        raise


def get_model_config(model_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve model configuration by ID."""
    config = db.model_configs.find_one({"model_id": model_id}, {"_id": 0})
    if config:
        logger.info(f"✓ Retrieved config for {model_id}")
        # Convert datetime objects to strings for JSON serialization
        if "created_at" in config:
            config["created_at"] = config["created_at"].isoformat()
        if "updated_at" in config:
            config["updated_at"] = config["updated_at"].isoformat()
    return config


def find_models_by_framework(framework: str) -> List[Dict[str, Any]]:
    """Find all models using a specific framework."""
    models = list(db.model_configs.find({"framework": framework}, {"_id": 0}))
    logger.info(f"✓ Found {len(models)} models using {framework}")
    return models


def find_high_accuracy_models(min_accuracy: float = 0.90) -> List[Dict[str, Any]]:
    """
    Query nested fields with dot notation.
    MongoDB excels at querying deeply nested documents.
    """
    models = list(
        db.model_configs.find(
            {"metrics.accuracy": {"$gte": min_accuracy}},
            {"_id": 0}
        ).sort("metrics.accuracy", DESCENDING)
    )
    logger.info(f"✓ Found {len(models)} models with accuracy >= {min_accuracy}")
    return models


def update_model_metrics(model_id: str, new_metrics: Dict[str, Any]):
    """
    MongoDB's flexible schema allows adding new fields without schema migration.
    Try doing this in SQL without ALTER TABLE!
    """
    result = db.model_configs.update_one(
        {"model_id": model_id},
        {
            "$set": {
                **{f"metrics.{k}": v for k, v in new_metrics.items()},
                "updated_at": datetime.utcnow()
            }
        }
    )

    if result.matched_count > 0:
        logger.info(f"✓ Updated metrics for {model_id}: {list(new_metrics.keys())}")
    else:
        logger.warning(f"Model {model_id} not found")

    return result.modified_count


def add_model_tags(model_id: str, tags: List[str]):
    """Add tags to model using MongoDB's array operators."""
    result = db.model_configs.update_one(
        {"model_id": model_id},
        {
            "$addToSet": {"tags": {"$each": tags}},  # $addToSet prevents duplicates
            "$set": {"updated_at": datetime.utcnow()}
        }
    )
    logger.info(f"✓ Added tags to {model_id}: {tags}")
    return result.modified_count


# ============================================================================
# AGGREGATION PIPELINES
# ============================================================================

def aggregate_metrics_by_framework() -> List[Dict[str, Any]]:
    """
    MongoDB aggregation pipeline for complex analysis.
    Similar to SQL GROUP BY but more flexible.
    """
    pipeline = [
        {
            "$group": {
                "_id": "$framework",
                "avg_accuracy": {"$avg": "$metrics.accuracy"},
                "max_accuracy": {"$max": "$metrics.accuracy"},
                "min_accuracy": {"$min": "$metrics.accuracy"},
                "count": {"$sum": 1},
                "models": {"$push": "$model_name"}
            }
        },
        {
            "$sort": {"avg_accuracy": -1}
        }
    ]

    results = list(db.model_configs.aggregate(pipeline))
    logger.info("✓ Aggregated metrics by framework")
    return results


def get_model_distribution_by_task() -> List[Dict[str, Any]]:
    """Count models by task type (if specified in config)."""
    pipeline = [
        {
            "$group": {
                "_id": "$config.task",
                "count": {"$sum": 1},
                "frameworks": {"$addToSet": "$framework"}
            }
        },
        {
            "$sort": {"count": -1}
        }
    ]

    results = list(db.model_configs.aggregate(pipeline))
    return results


# ============================================================================
# SAMPLE DATA
# ============================================================================

def insert_sample_configs():
    """Insert sample model configurations with diverse structures."""

    # PyTorch CNN model
    pytorch_config = {
        "model_id": "fraud-detector-v1",
        "model_name": "Fraud Detector CNN",
        "framework": "pytorch",
        "config": {
            "architecture": {
                "layers": [
                    {"type": "conv2d", "filters": 32, "kernel_size": 3, "activation": "relu"},
                    {"type": "maxpool", "pool_size": 2},
                    {"type": "conv2d", "filters": 64, "kernel_size": 3, "activation": "relu"},
                    {"type": "maxpool", "pool_size": 2},
                    {"type": "flatten"},
                    {"type": "dense", "units": 128, "activation": "relu", "dropout": 0.5},
                    {"type": "dense", "units": 1, "activation": "sigmoid"}
                ]
            },
            "training": {
                "optimizer": "adam",
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 50,
                "early_stopping": {
                    "patience": 5,
                    "monitor": "val_loss"
                }
            },
            "data_preprocessing": {
                "normalization": "minmax",
                "augmentation": ["rotation", "flip", "zoom"],
                "class_weight": {0: 1.0, 1: 10.0}  # Handle class imbalance
            }
        },
        "metrics": {
            "accuracy": 0.9845,
            "precision": 0.9712,
            "recall": 0.9834,
            "f1_score": 0.9772,
            "auc_roc": 0.9923,
            "inference_latency_ms": 15.3,
            "model_size_mb": 245.6
        },
        "tags": ["computer-vision", "classification", "fraud-detection", "production"]
    }

    # Hugging Face Transformer model
    transformer_config = {
        "model_id": "sentiment-analyzer-v2",
        "model_name": "BERT Sentiment Classifier",
        "framework": "huggingface",
        "config": {
            "base_model": "bert-base-uncased",
            "tokenizer": {
                "max_length": 512,
                "padding": "max_length",
                "truncation": True,
                "return_attention_mask": True
            },
            "fine_tuning": {
                "learning_rate": 2e-5,
                "warmup_steps": 500,
                "weight_decay": 0.01,
                "num_train_epochs": 3,
                "gradient_accumulation_steps": 2
            },
            "task": "text-classification",
            "num_labels": 3,
            "label_mapping": {
                0: "negative",
                1: "neutral",
                2: "positive"
            }
        },
        "metrics": {
            "accuracy": 0.9234,
            "f1_macro": 0.9187,
            "f1_weighted": 0.9245,
            "inference_latency_ms": 87.5,
            "model_size_mb": 438.0
        },
        "deployment": {
            "quantization": "int8",
            "onnx_export": True,
            "serving_framework": "triton",
            "max_batch_size": 64
        },
        "tags": ["nlp", "sentiment-analysis", "transformers", "production"]
    }

    # XGBoost model
    xgboost_config = {
        "model_id": "churn-predictor-v3",
        "model_name": "Customer Churn XGBoost",
        "framework": "xgboost",
        "config": {
            "booster": "gbtree",
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 6,
            "learning_rate": 0.3,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "feature_importance_type": "gain"
        },
        "feature_engineering": {
            "categorical_encoding": "target_encoding",
            "missing_value_strategy": "median_imputation",
            "feature_selection": {
                "method": "recursive_feature_elimination",
                "n_features": 50
            },
            "feature_scaling": None  # XGBoost doesn't require scaling
        },
        "metrics": {
            "accuracy": 0.8934,
            "auc_roc": 0.8934,
            "precision": 0.8234,
            "recall": 0.8678,
            "f1_score": 0.8451,
            "inference_latency_ms": 3.2,
            "model_size_mb": 38.9
        },
        "tags": ["tabular", "classification", "churn-prediction", "xgboost"]
    }

    # TensorFlow RNN model
    tensorflow_rnn = {
        "model_id": "timeseries-forecaster-v1",
        "model_name": "LSTM Time Series Forecaster",
        "framework": "tensorflow",
        "config": {
            "architecture": {
                "type": "sequential",
                "layers": [
                    {"type": "lstm", "units": 128, "return_sequences": True, "input_shape": [30, 10]},
                    {"type": "dropout", "rate": 0.2},
                    {"type": "lstm", "units": 64, "return_sequences": False},
                    {"type": "dropout", "rate": 0.2},
                    {"type": "dense", "units": 32, "activation": "relu"},
                    {"type": "dense", "units": 1, "activation": "linear"}
                ]
            },
            "task": "time-series-forecasting",
            "training": {
                "optimizer": "adam",
                "learning_rate": 0.001,
                "batch_size": 64,
                "epochs": 100,
                "validation_split": 0.2
            },
            "preprocessing": {
                "lookback_window": 30,
                "normalization": "standardization",
                "feature_columns": ["price", "volume", "moving_avg_7d", "moving_avg_30d"]
            }
        },
        "metrics": {
            "mse": 0.0234,
            "rmse": 0.1529,
            "mae": 0.1123,
            "mape": 5.67,  # Mean Absolute Percentage Error
            "inference_latency_ms": 12.4,
            "model_size_mb": 156.3
        },
        "tags": ["time-series", "forecasting", "lstm", "rnn"]
    }

    configs = [pytorch_config, transformer_config, xgboost_config, tensorflow_rnn]

    for config in configs:
        try:
            insert_model_config(config)
        except Exception as e:
            logger.warning(f"Skipping {config['model_id']}: {e}")

    logger.info(f"✓ Inserted {len(configs)} sample model configurations")


# ============================================================================
# EXPERIMENTS COLLECTION
# ============================================================================

def insert_experiment(
    model_id: str,
    experiment_name: str,
    hyperparameters: Dict[str, Any],
    metrics: Dict[str, float],
    status: str = "completed"
) -> str:
    """
    Insert experiment tracking record.
    MongoDB excels at storing varying experiment configurations.
    """
    experiment = {
        "model_id": model_id,
        "experiment_name": experiment_name,
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "status": status,
        "experiment_date": datetime.utcnow(),
        "created_at": datetime.utcnow()
    }

    result = db.experiments.insert_one(experiment)
    logger.info(f"✓ Inserted experiment: {experiment_name}")
    return str(result.inserted_id)


def get_best_experiment_for_model(model_id: str, metric: str = "accuracy") -> Optional[Dict[str, Any]]:
    """Find the best-performing experiment for a model based on a specific metric."""
    experiment = db.experiments.find_one(
        {"model_id": model_id, "status": "completed"},
        {"_id": 0},
        sort=[(f"metrics.{metric}", DESCENDING)]
    )
    return experiment


# ============================================================================
# COMPARISON WITH SQL
# ============================================================================

def demonstrate_schema_flexibility():
    """
    Demonstrate MongoDB's schema flexibility advantage.
    Adding new fields doesn't require schema migration.
    """
    print("\n=== MongoDB Schema Flexibility Demo ===\n")

    # Add completely new fields to existing documents
    db.model_configs.update_one(
        {"model_id": "fraud-detector-v1"},
        {
            "$set": {
                "explainability": {
                    "method": "shap",
                    "feature_importances": {
                        "transaction_amount": 0.45,
                        "merchant_category": 0.32,
                        "time_of_day": 0.23
                    }
                },
                "monitoring": {
                    "drift_detection": True,
                    "alert_threshold": 0.05
                }
            }
        }
    )

    print("✓ Added 'explainability' and 'monitoring' fields to fraud-detector-v1")
    print("  (No ALTER TABLE needed!)")

    # Query the updated document
    config = get_model_config("fraud-detector-v1")
    print(f"\n  New fields present: {list(config.get('explainability', {}).keys())}")


if __name__ == "__main__":
    print("="*70)
    print("MongoDB Client - ML Platform")
    print("="*70)

    # Test connection
    test_connection()

    # Setup collections
    setup_collections()

    # Insert sample data
    insert_sample_configs()

    # Demo: Find models by framework
    print("\n=== Models using PyTorch ===")
    pytorch_models = find_models_by_framework("pytorch")
    for model in pytorch_models:
        print(f"  {model['model_name']:<40} | Acc: {model['metrics'].get('accuracy', 'N/A')}")

    # Demo: High accuracy models
    print("\n=== Models with Accuracy >= 0.90 ===")
    high_acc = find_high_accuracy_models(0.90)
    for model in high_acc:
        acc = model['metrics'].get('accuracy', 0)
        print(f"  {model['model_name']:<40} | Acc: {acc:.4f} | {model['framework']}")

    # Demo: Aggregation
    print("\n=== Average Accuracy by Framework ===")
    framework_stats = aggregate_metrics_by_framework()
    for stat in framework_stats:
        print(f"  {stat['_id']:<15} | Avg: {stat['avg_accuracy']:.4f} | Count: {stat['count']} | Models: {', '.join(stat['models'][:2])}")

    # Demo: Schema flexibility
    demonstrate_schema_flexibility()

    # Demo: Update metrics
    print("\n=== Adding New Metrics ===")
    update_model_metrics("sentiment-analyzer-v2", {
        "cpu_inference_latency_ms": 42.1,
        "gpu_inference_latency_ms": 12.3,
        "memory_footprint_mb": 892.4
    })

    print("\n" + "="*70)
    print("✓ MongoDB demonstration complete")
    print("="*70)
