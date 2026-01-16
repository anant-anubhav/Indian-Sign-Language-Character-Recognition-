# ISL Recognition - Deployment Guide

This directory contains the deployment files for the Indian Sign Language (ISL) Recognition project.

## 📁 Structure

```
deployment/
├── __init__.py           # Package init
├── inference.py          # Core inference module (ISLRecognizer class)
├── fastapi_app.py        # FastAPI REST API backend
├── streamlit_app.py      # Streamlit web interface
├── requirements.txt      # Deployment dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# From project root
uv pip install -r deployment/requirements.txt
```

### 2. Run FastAPI Backend

```bash
# From project root
python -m deployment.fastapi_app

# Or using uvicorn directly
uvicorn deployment.fastapi_app:app --reload --host 0.0.0.0 --port 8000
```

**API Endpoints:**
- `GET /` - Home page
- `GET /docs` - Swagger UI documentation
- `GET /health` - Health check
- `GET /classes` - List all ISL classes
- `POST /predict` - Single image prediction
- `POST /predict/topk` - Top-K predictions

### 3. Run Streamlit Frontend

```bash
# From project root
streamlit run deployment/streamlit_app.py
```

The Streamlit app will open in your browser at `http://localhost:8501`.

## 📊 Features

### FastAPI Backend
- RESTful API for ISL character recognition
- Swagger UI documentation at `/docs`
- Support for both CNN and MobileNetV2 models
- CORS enabled for frontend integration
- Health check and model info endpoints

### Streamlit Frontend
- Interactive web interface
- Image upload with drag-and-drop
- Real-time predictions with confidence scores
- Top-K predictions visualization
- Model selection (CNN/MobileNetV2)
- Responsive design

## 🔧 Configuration

### Model Paths
Models are loaded from `saved_models/` directory:
- `saved_models/cnn_model_final.keras`
- `saved_models/mobilenet_model_final.keras`
- `saved_models/class_labels.pkl`

### Image Size
Default image size is 128x128 pixels (configurable in `inference.py`).

## 📦 Deployment Options

### Local Development
```bash
# FastAPI with auto-reload
uvicorn deployment.fastapi_app:app --reload

# Streamlit
streamlit run deployment/streamlit_app.py
```

### Production
```bash
# FastAPI (production)
uvicorn deployment.fastapi_app:app --host 0.0.0.0 --port 8000 --workers 4

# Streamlit (production)
streamlit run deployment/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker (Coming Soon)
Docker configuration for containerized deployment.

## 📝 API Usage Examples

### Python
```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Predict
with open("image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict", files=files)
    print(response.json())
```

### cURL
```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

## 🤝 Integration

The inference module can be used directly in your Python code:

```python
from deployment.inference import ISLRecognizer, predict_sign

# Using the class
recognizer = ISLRecognizer(model_type="mobilenet")
label, confidence, all_probs = recognizer.predict("image.jpg")
print(f"Prediction: {label} ({confidence:.1f}%)")

# Using the convenience function
label, confidence = predict_sign("image.jpg")
print(f"Prediction: {label} ({confidence:.1f}%)")
```
