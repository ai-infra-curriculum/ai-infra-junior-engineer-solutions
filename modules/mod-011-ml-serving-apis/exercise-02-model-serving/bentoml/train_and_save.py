"""Train and save model to BentoML model store."""

import bentoml
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train_model():
    """Train a simple classification model."""
    print("Generating training data...")

    # Generate synthetic data
    np.random.seed(42)
    X = np.random.rand(1000, 5).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, accuracy


def save_to_bentoml(model, accuracy):
    """Save model to BentoML model store."""
    print("\nSaving model to BentoML...")

    saved_model = bentoml.sklearn.save_model(
        "my_classifier",
        model,
        signatures={
            "predict": {
                "batchable": True,
                "batch_dim": 0,
            },
            "predict_proba": {
                "batchable": True,
                "batch_dim": 0,
            }
        },
        labels={
            "framework": "sklearn",
            "task": "classification",
            "model_type": "random_forest"
        },
        metadata={
            "accuracy": accuracy,
            "n_estimators": 100,
            "max_depth": 10,
            "trained_on": "2025-10-24"
        },
        custom_objects={
            "feature_names": ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"]
        }
    )

    print(f"✓ Model saved: {saved_model}")
    print(f"  Tag: {saved_model.tag}")
    print(f"  Path: {saved_model.path}")

    return saved_model


if __name__ == "__main__":
    # Train model
    model, accuracy = train_model()

    # Save to BentoML
    saved_model = save_to_bentoml(model, accuracy)

    print("\n" + "="*50)
    print("Model ready for serving!")
    print("="*50)
    print("\nNext steps:")
    print("1. Build Bento: bentoml build")
    print("2. Serve locally: bentoml serve service:svc --reload")
    print("3. Containerize: bentoml containerize ml_classifier_service:latest")
