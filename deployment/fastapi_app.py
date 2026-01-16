"""
ISL Recognition - FastAPI Backend
=================================
REST API for Indian Sign Language character recognition.

Run with:
    uvicorn deployment.fastapi_app:app --reload --host 0.0.0.0 --port 8000

Or:
    python -m deployment.fastapi_app
"""

import os
import io
import sys
from pathlib import Path
from typing import Optional, List
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
from PIL import Image

from deployment.inference import ISLRecognizer, IMG_SIZE


# ============================================================================
# FastAPI App Configuration
# ============================================================================
app = FastAPI(
    title="ISL Recognition API",
    description="""
    🤟 **Indian Sign Language Character Recognition API**
    
    This API provides endpoints for recognizing ISL characters from images.
    Supports both Custom CNN and MobileNetV2 (Transfer Learning) models.
    
    ## Features
    - Single image prediction
    - Top-K predictions with confidence scores
    - Batch prediction support
    - Model selection (CNN/MobileNetV2)
    
    ## Models
    - **MobileNetV2** (default): Higher accuracy, transfer learning-based
    - **CNN**: Lightweight custom model
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Response Models
# ============================================================================
class PredictionResponse(BaseModel):
    """Single prediction response."""
    success: bool
    prediction: str
    confidence: float
    model_type: str
    timestamp: str


class TopKPredictionResponse(BaseModel):
    """Top-K predictions response."""
    success: bool
    predictions: List[dict]
    model_type: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_type: str
    num_classes: int
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = False
    error: str
    timestamp: str


# ============================================================================
# Model Loading
# ============================================================================
# Global recognizer instances
recognizers = {}

def get_recognizer(model_type: str = "mobilenet") -> ISLRecognizer:
    """Get or create a recognizer instance."""
    if model_type not in recognizers:
        try:
            recognizers[model_type] = ISLRecognizer(model_type)
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
    return recognizers[model_type]


# ============================================================================
# API Endpoints
# ============================================================================
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ISL Recognition API</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #2c3e50; }
            .emoji { font-size: 48px; }
            a { color: #3498db; text-decoration: none; }
            a:hover { text-decoration: underline; }
            .endpoints { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-top: 20px; }
            code { background: #e9ecef; padding: 2px 6px; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="emoji">🤟</div>
        <h1>Indian Sign Language Recognition API</h1>
        <p>Welcome to the ISL Recognition API. Use the endpoints below to recognize ISL characters from images.</p>
        
        <div class="endpoints">
            <h3>📚 Documentation</h3>
            <ul>
                <li><a href="/docs">Swagger UI (Interactive)</a></li>
                <li><a href="/redoc">ReDoc (Alternative)</a></li>
            </ul>
            
            <h3>🔗 Endpoints</h3>
            <ul>
                <li><code>GET /health</code> - Health check</li>
                <li><code>POST /predict</code> - Single image prediction</li>
                <li><code>POST /predict/topk</code> - Top-K predictions</li>
                <li><code>GET /classes</code> - List all classes</li>
            </ul>
        </div>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
async def health_check(model_type: str = Query("mobilenet", enum=["cnn", "mobilenet"])):
    """
    Health check endpoint.
    
    Returns the status of the API and model information.
    """
    try:
        recognizer = get_recognizer(model_type)
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_type=model_type,
            num_classes=recognizer.num_classes,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_type=model_type,
            num_classes=0,
            timestamp=datetime.now().isoformat()
        )


@app.get("/classes")
async def get_classes(model_type: str = Query("mobilenet", enum=["cnn", "mobilenet"])):
    """
    Get list of all ISL character classes.
    """
    recognizer = get_recognizer(model_type)
    return {
        "success": True,
        "classes": recognizer.classes,
        "count": recognizer.num_classes,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Image file to classify"),
    model_type: str = Query("mobilenet", enum=["cnn", "mobilenet"])
):
    """
    Predict ISL character from an uploaded image.
    
    - **file**: Image file (JPEG, PNG, etc.)
    - **model_type**: Model to use ('cnn' or 'mobilenet')
    
    Returns the predicted character and confidence score.
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected an image."
        )
    
    try:
        # Read and process image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Get prediction
        recognizer = get_recognizer(model_type)
        label, confidence, _ = recognizer.predict(img)
        
        return PredictionResponse(
            success=True,
            prediction=label,
            confidence=round(confidence, 2),
            model_type=model_type,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/topk", response_model=TopKPredictionResponse)
async def predict_topk(
    file: UploadFile = File(..., description="Image file to classify"),
    k: int = Query(5, ge=1, le=35, description="Number of top predictions"),
    model_type: str = Query("mobilenet", enum=["cnn", "mobilenet"])
):
    """
    Get top-K predictions for an uploaded image.
    
    - **file**: Image file (JPEG, PNG, etc.)
    - **k**: Number of top predictions to return (1-35)
    - **model_type**: Model to use ('cnn' or 'mobilenet')
    
    Returns a list of predictions sorted by confidence.
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected an image."
        )
    
    try:
        # Read and process image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Get predictions
        recognizer = get_recognizer(model_type)
        top_k = recognizer.predict_top_k(img, k=k)
        
        predictions = [
            {"label": label, "confidence": round(conf, 2)}
            for label, conf in top_k
        ]
        
        return TopKPredictionResponse(
            success=True,
            predictions=predictions,
            model_type=model_type,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ============================================================================
# Exception Handlers
# ============================================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": f"Internal server error: {str(exc)}",
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================================
# Run Server
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("ISL Recognition - FastAPI Server")
    print("=" * 60)
    print("\nStarting server...")
    print("API Docs: http://localhost:8000/docs")
    print("Health:   http://localhost:8000/health")
    print()
    
    uvicorn.run(
        "deployment.fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
