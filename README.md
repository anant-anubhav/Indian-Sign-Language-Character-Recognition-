# ✋ Indian Sign Language (ISL) Recognition

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

**A Deep Learning-based Image Classification System for Indian Sign Language Character Recognition**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Models](#-models) • [API](#-api-reference) • [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Models](#-models)
- [Deployment](#-deployment)
- [API Reference](#-api-reference)
- [Usage Examples](#-usage-examples)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)


---

## 🎯 About the Project

### Business Case

Indian Sign Language (ISL) recognition is critical for improving accessibility and communication for the deaf and hard-of-hearing community in India. This project builds a **deep learning-based image classification system** to automatically recognize ISL characters from images, enabling:

- 🤝 Better communication between hearing and deaf individuals
- 📱 Integration with mobile and web applications
- 🎓 Educational tools for learning ISL
- 🏥 Accessibility solutions in healthcare, education, and public services

### Project Goals

1. ✅ Perform detailed Exploratory Data Analysis (EDA) on ISL image data
2. ✅ Apply image preprocessing and data augmentation
3. ✅ Build and compare CNN and Transfer Learning models
4. ✅ Evaluate model performance using classification metrics
5. ✅ Save trained models for deployment
6. ✅ Provide REST API and Web Interface for inference

**Project Code:** `PRAICP-1000-IndiSignLang` (DataMites™)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Deep Learning Models** | Custom CNN and MobileNetV2 (Transfer Learning) |
| 📊 **35 Character Classes** | Digits 1-9 and Alphabets A-Z |
| 🌐 **REST API** | FastAPI backend with Swagger documentation |
| 💻 **Web Interface** | Interactive Streamlit application |
| 🎨 **Data Augmentation** | Rotation, shifting, zoom, and flipping |
| 📈 **Model Comparison** | Accuracy, confusion matrix, and classification reports |
| 🔮 **Real-time Inference** | Single image and batch prediction support |
| 📦 **Multiple Model Formats** | `.keras` and `.h5` formats available |

---

## 📁 Project Structure

```
Indian Sign Language (ISL)/
│
├── 📂 Dataset/                    # ISL Image Dataset
│   ├── 1/                         # Images for digit 1
│   ├── 2/                         # Images for digit 2
│   ├── ...                        # ... (3-9)
│   ├── A/                         # Images for letter A
│   ├── B/                         # Images for letter B
│   └── ...                        # ... (C-Z)
│
├── 📂 Notebook/                   # Jupyter Notebooks
│   ├── Indian_Sign_Language_Recognition_Datamites.ipynb
│   └── 📂 saved_models/           # Trained model files
│       ├── cnn_best_model.keras
│       ├── cnn_model_final.keras
│       ├── cnn_model_final.h5
│       ├── mobilenet_best_model.keras
│       ├── mobilenet_model_final.keras
│       ├── mobilenet_model_final.h5
│       ├── class_labels.pkl
│       └── class_indices.pkl
│
├── 📂 deployment/                 # Deployment Files
│   ├── __init__.py
│   ├── inference.py               # Core inference module
│   ├── fastapi_app.py             # FastAPI REST API
│   ├── streamlit_app.py           # Streamlit web interface
│   ├── requirements.txt           # Deployment dependencies
│   └── README.md                  # Deployment documentation
│
├── main.py                        # Main entry point
├── requirements.txt               # Project dependencies
├── pyproject.toml                 # Project configuration
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

---

## 📊 Dataset

### Overview

| Property | Value |
|----------|-------|
| **Dataset Type** | Image (Character-level ISL) |
| **Total Images** | ~42,700 |
| **Number of Classes** | 35 (Digits 1–9 + Alphabets A–Z) |
| **Image Format** | JPEG/PNG |
| **Structure** | Directory-based (each folder = one character) |

> **Note:** Digit `0` is not included as it is identical to ASL (American Sign Language).

### Classes

```
Digits:   1, 2, 3, 4, 5, 6, 7, 8, 9
Letters:  A, B, C, D, E, F, G, H, I, J, K, L, M, 
          N, O, P, Q, R, S, T, U, V, W, X, Y, Z
```

### Data Distribution

The dataset is well-balanced across all 35 ISL character classes, with approximately **1,200 images per class**.

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or higher
- pip or uv package manager
- CUDA (optional, for GPU acceleration)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/indian-sign-language-recognition.git
cd "Indian Sign Language (ISL)"
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install deployment dependencies (optional)
pip install -r deployment/requirements.txt
```

### Requirements

**Core Dependencies:**
```
tensorflow>=2.12.0
numpy>=1.23.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
pillow>=9.5.0
jupyter>=1.0.0
ipykernel>=6.22.0
```

**Deployment Dependencies:**
```
fastapi>=0.100.0
uvicorn>=0.22.0
streamlit>=1.25.0
plotly>=5.15.0
python-multipart>=0.0.6
pydantic>=2.0.0
```

---

## 🚀 Quick Start

### Option 1: Run Jupyter Notebook (Training & Experimentation)

```bash
# Start Jupyter Notebook
jupyter notebook Notebook/Indian_Sign_Language_Recognition_Datamites.ipynb
```

This notebook includes:
- Exploratory Data Analysis (EDA)
- Data preprocessing and augmentation
- Model training (CNN + MobileNetV2)
- Model evaluation and comparison
- Model saving and inference examples

### Option 2: Run FastAPI Backend (REST API)

```bash
# From project root
python -m deployment.fastapi_app

# Or using uvicorn
uvicorn deployment.fastapi_app:app --reload --host 0.0.0.0 --port 8000
```

Access the API at:
- **API Home:** http://localhost:8000
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Option 3: Run Streamlit App (Web Interface)

```bash
# From project root
streamlit run deployment/streamlit_app.py
```

Access the web app at: http://localhost:8501

---

## 🧠 Models

### Model 1: Custom CNN

A lightweight convolutional neural network built from scratch.

**Architecture:**
```
Input (128x128x3)
    ↓
Conv2D(32, 3x3) → ReLU → MaxPooling
    ↓
Conv2D(64, 3x3) → ReLU → MaxPooling
    ↓
Conv2D(128, 3x3) → ReLU → MaxPooling
    ↓
Flatten → Dense(256) → ReLU → Dropout(0.5)
    ↓
Dense(35) → Softmax
```

**Characteristics:**
- Fast training
- Smaller model size (~75 MB)
- Good for resource-constrained environments

### Model 2: MobileNetV2 (Transfer Learning)

Pre-trained MobileNetV2 with custom classification head.

**Architecture:**
```
Input (128x128x3)
    ↓
MobileNetV2 (pre-trained, frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256) → ReLU → Dropout(0.5)
    ↓
Dense(35) → Softmax
```

**Characteristics:**
- Higher accuracy
- Transfer learning from ImageNet
- Model size (~13 MB)
- Recommended for production

### Data Augmentation

Both models use the following augmentation techniques:

| Augmentation | Value |
|--------------|-------|
| Rotation Range | ±15° |
| Width Shift | ±10% |
| Height Shift | ±10% |
| Zoom Range | ±10% |
| Horizontal Flip | Yes |

---

## 🌐 Deployment

### FastAPI REST API

**Features:**
- RESTful endpoints for prediction
- Swagger UI documentation
- CORS enabled
- Health checks
- Model selection (CNN/MobileNetV2)

**Run:**
```bash
# Development
uvicorn deployment.fastapi_app:app --reload

# Production
uvicorn deployment.fastapi_app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Streamlit Web App

**Features:**
- Interactive web interface
- Drag-and-drop image upload
- Real-time predictions
- Top-K predictions visualization
- Model selection
- Responsive design

**Run:**
```bash
# Development/Production
streamlit run deployment/streamlit_app.py
```

### 🚂 Railway Deployment

Deploy your ISL Recognition API to [Railway.app](https://railway.app) with these steps:

#### Prerequisites

1. **Git LFS** - Required for large model files
2. **Railway Account** - Sign up at [railway.app](https://railway.app)
3. **GitHub Repository** - Push your code to GitHub

#### Step 1: Install Git LFS

```bash
# Windows (with Git for Windows)
git lfs install

# Or download from: https://git-lfs.github.com/
```

#### Step 2: Initialize Git LFS & Push

```bash
# Track large files with LFS
git lfs install
git lfs track "*.keras"
git lfs track "*.h5"
git lfs track "*.pkl"

# Add all files
git add .gitattributes
git add .
git commit -m "Add Railway deployment configuration"
git push origin main
```

#### Step 3: Deploy on Railway

1. Go to [railway.app](https://railway.app) and login with GitHub
2. Click **"New Project"** → **"Deploy from GitHub Repo"**
3. Select your ISL Recognition repository
4. Railway will auto-detect the `Procfile` and deploy

#### Step 4: Environment Variables (Optional)

Add these in Railway dashboard if needed:
| Variable | Value |
|----------|-------|
| `PORT` | Auto-set by Railway |
| `PYTHON_VERSION` | `3.10.12` |

#### Railway Configuration Files

| File | Purpose |
|------|---------|
| `Procfile` | Defines the start command |
| `railway.json` | Railway-specific configuration |
| `runtime.txt` | Python version specification |
| `requirements.txt` | Dependencies |

#### Important Notes

> ⚠️ **Model Files Size**: The `saved_models/` folder contains ~180MB of model files. Git LFS is required to push them to GitHub.

> 💡 **Free Tier**: Railway provides $5 free credits monthly. Monitor your usage in the dashboard.

> 🔧 **Memory**: The app requires ~1GB RAM for TensorFlow. Railway's free tier should handle this.

---

## 📖 API Reference

### Base URL

```
http://localhost:8000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Home page with API info |
| `GET` | `/health` | Health check |
| `GET` | `/classes` | List all ISL classes |
| `POST` | `/predict` | Single image prediction |
| `POST` | `/predict/topk` | Top-K predictions |

### Health Check

```http
GET /health?model_type=mobilenet
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "mobilenet",
  "num_classes": 35,
  "timestamp": "2026-01-16T12:30:00"
}
```

### Single Prediction

```http
POST /predict?model_type=mobilenet
Content-Type: multipart/form-data
```

**Request:**
- `file`: Image file (JPEG, PNG)
- `model_type`: `cnn` or `mobilenet` (optional, default: `mobilenet`)

**Response:**
```json
{
  "success": true,
  "prediction": "A",
  "confidence": 98.75,
  "model_type": "mobilenet",
  "timestamp": "2026-01-16T12:30:00"
}
```

### Top-K Predictions

```http
POST /predict/topk?k=5&model_type=mobilenet
Content-Type: multipart/form-data
```

**Response:**
```json
{
  "success": true,
  "predictions": [
    {"label": "A", "confidence": 98.75},
    {"label": "B", "confidence": 0.85},
    {"label": "C", "confidence": 0.20},
    {"label": "D", "confidence": 0.10},
    {"label": "E", "confidence": 0.05}
  ],
  "model_type": "mobilenet",
  "timestamp": "2026-01-16T12:30:00"
}
```

---

## 💻 Usage Examples

### Python - Direct Inference

```python
from deployment.inference import ISLRecognizer, predict_sign

# Using the class
recognizer = ISLRecognizer(model_type="mobilenet")
label, confidence, all_probs = recognizer.predict("image.jpg")
print(f"Prediction: {label} ({confidence:.1f}%)")

# Using the convenience function
label, confidence = predict_sign("image.jpg")
print(f"Prediction: {label} ({confidence:.1f}%)")

# Batch prediction
results = recognizer.batch_predict(["img1.jpg", "img2.jpg", "img3.jpg"])
for label, conf in results:
    print(f"{label}: {conf:.1f}%")
```

### Python - API Client

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Predict
with open("image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/predict",
        files=files,
        params={"model_type": "mobilenet"}
    )
    result = response.json()
    print(f"Prediction: {result['prediction']} ({result['confidence']:.1f}%)")
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"

# Top-5 predictions
curl -X POST "http://localhost:8000/predict/topk?k=5" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

---

## 📈 Results

### Model Performance

| Model | Validation Accuracy | Training Time | Model Size |
|-------|---------------------|---------------|------------|
| Custom CNN | ~85-90% | ~15 min | 75 MB |
| MobileNetV2 (TL) | ~95-98% | ~10 min | 13 MB |

> **Note:** Actual performance may vary based on training epochs, hardware, and hyperparameters.

### Saved Models

| File | Description | Size |
|------|-------------|------|
| `cnn_model_final.keras` | Custom CNN (Keras format) | 75 MB |
| `cnn_model_final.h5` | Custom CNN (HDF5 format) | 75 MB |
| `mobilenet_model_final.keras` | MobileNetV2 (Keras format) | 13 MB |
| `mobilenet_model_final.h5` | MobileNetV2 (HDF5 format) | 13 MB |
| `class_labels.pkl` | Class index to label mapping | 1 KB |
| `class_indices.pkl` | Class label to index mapping | 1 KB |

---

## 🛠️ Development

### Running Tests

```bash
# Run inference test
python -m deployment.inference path/to/test/image.jpg
```

### Code Structure

| Module | Description |
|--------|-------------|
| `deployment/inference.py` | Core inference module with `ISLRecognizer` class |
| `deployment/fastapi_app.py` | FastAPI REST API implementation |
| `deployment/streamlit_app.py` | Streamlit web application |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **DataMites™** - Project guidance and curriculum
- **TensorFlow/Keras** - Deep learning framework
- **FastAPI** - Modern web framework for APIs
- **Streamlit** - Interactive web applications
- **MobileNetV2** - Pre-trained model from ImageNet

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

<div align="center">

**Made with ❤️ for the Indian Deaf and Hard-of-Hearing Community**

</div>
