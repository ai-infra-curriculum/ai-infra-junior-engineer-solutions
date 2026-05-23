"""BentoML service for multi-framework model serving."""

import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, JSON
from typing import Dict, List

# Load model from BentoML model store
sklearn_model_ref = bentoml.sklearn.get("my_classifier:latest")
sklearn_runner = sklearn_model_ref.to_runner()

# Create BentoML service
svc = bentoml.Service(
    "ml_classifier_service",
    runners=[sklearn_runner]
)


@svc.api(
    input=NumpyNdarray(dtype="float32", shape=(-1, 5)),
    output=JSON()
)
async def predict_numpy(input_array: np.ndarray) -> Dict:
    """
    Prediction endpoint accepting NumPy array.

    Args:
        input_array: Input features as NumPy array

    Returns:
        Prediction result
    """
    # Run prediction
    result = await sklearn_runner.predict.async_run(input_array)

    # Get probabilities
    probabilities = await sklearn_runner.predict_proba.async_run(input_array)

    return {
        "predictions": result.tolist(),
        "probabilities": probabilities.tolist(),
        "model_version": sklearn_model_ref.tag.version
    }


@svc.api(input=JSON(), output=JSON())
async def predict_json(input_data: Dict) -> Dict:
    """
    Prediction endpoint accepting JSON.

    Args:
        input_data: Dictionary with "features" key

    Returns:
        Prediction result
    """
    # Extract features
    features = input_data.get("features")

    if features is None:
        return {"error": "Missing 'features' in request"}

    # Convert to numpy array
    features_array = np.array([features], dtype=np.float32)

    # Run prediction
    result = await sklearn_runner.predict.async_run(features_array)
    probabilities = await sklearn_runner.predict_proba.async_run(features_array)

    return {
        "prediction": int(result[0]),
        "probability": probabilities[0].tolist(),
        "confidence": float(max(probabilities[0])),
        "model_version": sklearn_model_ref.tag.version
    }


@svc.api(input=JSON(), output=JSON())
async def predict_batch(input_data: Dict) -> Dict:
    """
    Batch prediction endpoint.

    Args:
        input_data: Dictionary with "instances" key containing list of feature arrays

    Returns:
        Batch prediction results
    """
    instances = input_data.get("instances")

    if instances is None:
        return {"error": "Missing 'instances' in request"}

    # Convert to numpy array
    features_array = np.array(instances, dtype=np.float32)

    # Run batch prediction
    results = await sklearn_runner.predict.async_run(features_array)
    probabilities = await sklearn_runner.predict_proba.async_run(features_array)

    # Format response
    predictions = []
    for i in range(len(results)):
        predictions.append({
            "prediction": int(results[i]),
            "probability": probabilities[i].tolist(),
            "confidence": float(max(probabilities[i]))
        })

    return {
        "predictions": predictions,
        "batch_size": len(predictions),
        "model_version": sklearn_model_ref.tag.version
    }


@svc.api(input=JSON(), output=JSON())
async def health() -> Dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": sklearn_model_ref.tag.version
    }
