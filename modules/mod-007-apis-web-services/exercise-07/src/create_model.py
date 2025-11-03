"""Create a dummy ML model for testing.

This script creates a simple Random Forest model for demonstration.
In production, you would load your actual trained model.
"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def create_and_save_model(output_path='model.pkl'):
    """Create a dummy model and save it.

    Args:
        output_path: Path where to save the model
    """
    print("Creating dummy ML model...")

    # Generate synthetic dataset
    # 10 features as expected by the API
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        noise=10,
        random_state=42
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    print("Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model performance:")
    print(f"  MSE: {mse:.2f}")
    print(f"  R²:  {r2:.4f}")

    # Save model
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"✅ Model saved to {output_path}")
    print(f"   Model size: {get_file_size(output_path):.2f} MB")


def get_file_size(file_path):
    """Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        Size in MB
    """
    import os
    return os.path.getsize(file_path) / (1024 * 1024)


if __name__ == '__main__':
    create_and_save_model('model.pkl')
