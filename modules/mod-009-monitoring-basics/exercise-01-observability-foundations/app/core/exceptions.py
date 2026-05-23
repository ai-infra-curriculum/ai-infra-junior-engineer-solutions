"""Custom exceptions for the inference gateway."""

class InferenceGatewayException(Exception):
    """Base exception for all inference gateway errors."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ModelNotLoadedException(InferenceGatewayException):
    """Raised when model is not loaded."""

    def __init__(self, message: str = "Model not loaded"):
        super().__init__(message, status_code=503)


class InvalidInputException(InferenceGatewayException):
    """Raised when input validation fails."""

    def __init__(self, message: str = "Invalid input"):
        super().__init__(message, status_code=400)


class InferenceTimeoutException(InferenceGatewayException):
    """Raised when inference times out."""

    def __init__(self, message: str = "Inference timeout"):
        super().__init__(message, status_code=504)


class QueueFullException(InferenceGatewayException):
    """Raised when inference queue is full."""

    def __init__(self, message: str = "Inference queue is full"):
        super().__init__(message, status_code=503)


class ModelInferenceException(InferenceGatewayException):
    """Raised when model inference fails."""

    def __init__(self, message: str = "Model inference failed"):
        super().__init__(message, status_code=500)
