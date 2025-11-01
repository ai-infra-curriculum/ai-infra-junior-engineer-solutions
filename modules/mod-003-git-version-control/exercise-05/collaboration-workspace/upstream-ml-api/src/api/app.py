"""ML Inference API - Main Application"""

from fastapi import FastAPI

app = FastAPI(
    title="ML Inference API",
    description="Production ML inference service",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {
        "name": "ML Inference API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}
