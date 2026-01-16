"""
ISL Recognition - Gradio App for Hugging Face Spaces
=====================================================
Interactive web interface for Indian Sign Language character recognition.

Deploy to Hugging Face Spaces:
1. Create a new Space on huggingface.co
2. Select "Gradio" as the SDK
3. Upload this app.py and requirements.txt
4. Upload saved_models/ folder
"""

import gradio as gr
import numpy as np
from PIL import Image
import pickle
from pathlib import Path

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    raise ImportError("TensorFlow is required. Install with: pip install tensorflow")


# ============================================================================
# Configuration
# ============================================================================
MODEL_DIR = Path("saved_models")
IMG_SIZE = (128, 128)

# Model paths
MOBILENET_PATH = MODEL_DIR / "mobilenet_model_final.keras"
CNN_PATH = MODEL_DIR / "cnn_model_final.keras"
CLASS_LABELS_PATH = MODEL_DIR / "class_labels.pkl"


# ============================================================================
# Custom CSS for Enhanced UI
# ============================================================================
CUSTOM_CSS = """
/* Main container */
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

/* Header styling */
.header-container {
    text-align: center;
    padding: 2rem 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    margin-bottom: 2rem;
    color: white;
}

.header-title {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}

.header-subtitle {
    font-size: 1.1rem;
    opacity: 0.9;
    margin-top: 0.5rem;
}

/* Prediction result card */
.result-card {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    color: white;
    margin: 1rem 0;
    box-shadow: 0 10px 40px rgba(17, 153, 142, 0.3);
}

.result-label {
    font-size: 4rem;
    font-weight: 800;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}

.result-confidence {
    font-size: 1.5rem;
    opacity: 0.95;
    margin-top: 0.5rem;
}

/* Stats cards */
.stats-container {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}

.stat-card {
    flex: 1;
    background: #1a1a2e;
    padding: 1rem;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #333;
}

.stat-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #667eea;
}

.stat-label {
    font-size: 0.85rem;
    color: #888;
    margin-top: 0.25rem;
}

/* Info section */
.info-section {
    background: #1a1a2e;
    padding: 1.5rem;
    border-radius: 12px;
    margin-top: 1.5rem;
    border: 1px solid #333;
}

.info-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #fff;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
}

.info-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #ccc;
}

/* Classes grid */
.classes-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1rem;
}

.class-badge {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9rem;
}

/* Footer */
.footer {
    text-align: center;
    padding: 1.5rem;
    color: #666;
    font-size: 0.85rem;
    border-top: 1px solid #333;
    margin-top: 2rem;
}
"""


# ============================================================================
# Load Model and Labels
# ============================================================================
def load_model_and_labels(model_type="mobilenet"):
    """Load the specified model and class labels."""
    model_path = MOBILENET_PATH if model_type == "mobilenet" else CNN_PATH
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = load_model(str(model_path))
    
    with open(CLASS_LABELS_PATH, 'rb') as f:
        class_labels = pickle.load(f)
    
    return model, class_labels


# Load model at startup
print("Loading MobileNetV2 model...")
model, class_labels = load_model_and_labels("mobilenet")
print(f"Model loaded! Classes: {len(class_labels)}")


# ============================================================================
# Prediction Function
# ============================================================================
def predict_sign(image):
    """Predict ISL character from an image."""
    if image is None:
        return None, create_placeholder_html()
    
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    else:
        image = image.convert("RGB")
    
    # Resize to model input size
    image = image.resize(IMG_SIZE)
    
    # Preprocess
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Get top predictions
    top_indices = np.argsort(predictions)[-5:][::-1]
    
    results = {}
    for idx in top_indices:
        label = class_labels[idx]
        confidence = float(predictions[idx])
        results[label] = confidence
    
    # Get the top prediction
    top_label = class_labels[top_indices[0]]
    top_confidence = predictions[top_indices[0]] * 100
    
    # Create result HTML
    result_html = create_result_html(top_label, top_confidence, results)
    
    return results, result_html


def create_placeholder_html():
    """Create placeholder HTML when no image is uploaded."""
    return """
    <div style="text-align: center; padding: 3rem; color: #888;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🤚</div>
        <p style="font-size: 1.2rem;">Upload an ISL gesture image to see predictions</p>
    </div>
    """


def create_result_html(label, confidence, all_results):
    """Create beautiful result HTML."""
    # Determine confidence color
    if confidence >= 80:
        bg_gradient = "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)"
    elif confidence >= 50:
        bg_gradient = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
    else:
        bg_gradient = "linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%)"
    
    return f"""
    <div style="background: {bg_gradient}; padding: 2rem; border-radius: 16px; text-align: center; color: white; box-shadow: 0 10px 40px rgba(0,0,0,0.3);">
        <p style="font-size: 1rem; opacity: 0.9; margin: 0;">Predicted Character</p>
        <p style="font-size: 5rem; font-weight: 800; margin: 0.5rem 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">{label}</p>
        <p style="font-size: 1.3rem; opacity: 0.95;">Confidence: {confidence:.1f}%</p>
    </div>
    
    <div style="display: flex; gap: 1rem; margin-top: 1.5rem;">
        <div style="flex: 1; background: #1a1a2e; padding: 1rem; border-radius: 12px; text-align: center; border: 1px solid #333;">
            <p style="font-size: 1.5rem; font-weight: 700; color: #667eea; margin: 0;">{label}</p>
            <p style="font-size: 0.85rem; color: #888; margin: 0.25rem 0 0 0;">Prediction</p>
        </div>
        <div style="flex: 1; background: #1a1a2e; padding: 1rem; border-radius: 12px; text-align: center; border: 1px solid #333;">
            <p style="font-size: 1.5rem; font-weight: 700; color: #38ef7d; margin: 0;">{confidence:.1f}%</p>
            <p style="font-size: 0.85rem; color: #888; margin: 0.25rem 0 0 0;">Confidence</p>
        </div>
        <div style="flex: 1; background: #1a1a2e; padding: 1rem; border-radius: 12px; text-align: center; border: 1px solid #333;">
            <p style="font-size: 1.5rem; font-weight: 700; color: #764ba2; margin: 0;">MobileNetV2</p>
            <p style="font-size: 0.85rem; color: #888; margin: 0.25rem 0 0 0;">Model</p>
        </div>
    </div>
    """


# ============================================================================
# Gradio Interface
# ============================================================================
with gr.Blocks(css=CUSTOM_CSS, title="ISL Recognition") as demo:
    
    # Header
    gr.HTML("""
    <div class="header-container">
        <h1 class="header-title">🤟 Indian Sign Language Recognition</h1>
        <p class="header-subtitle">Deep Learning powered ISL character recognition</p>
    </div>
    """)
    
    with gr.Row():
        # Left Column - Input
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Image")
            input_image = gr.Image(
                label="ISL Gesture Image",
                type="pil",
                height=350,
                sources=["upload", "webcam", "clipboard"]
            )
            predict_btn = gr.Button(
                "🔮 Analyze Gesture",
                variant="primary",
                size="lg"
            )
            
            # Quick info
            gr.Markdown("""
            **Tips for best results:**
            - Use a clear image of the hand gesture
            - Ensure good lighting
            - Hand should be the main focus
            """)
        
        # Right Column - Results
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Prediction Results")
            result_html = gr.HTML(create_placeholder_html())
            
            with gr.Accordion("📊 Detailed Predictions", open=False):
                output_label = gr.Label(
                    label="Top 5 Predictions",
                    num_top_classes=5
                )
    
    # Info Section
    gr.HTML("""
    <div class="info-section">
        <div class="info-title">ℹ️ About This App</div>
        <div class="info-grid">
            <div class="info-item">📊 <strong>Dataset:</strong>&nbsp;~42,700 images</div>
            <div class="info-item">🔤 <strong>Classes:</strong>&nbsp;35 (1-9, A-Z)</div>
            <div class="info-item">🧠 <strong>Model:</strong>&nbsp;MobileNetV2</div>
            <div class="info-item">🎯 <strong>Accuracy:</strong>&nbsp;95-98%</div>
        </div>
        <div style="margin-top: 1rem;">
            <p style="color: #888; font-size: 0.9rem;">Supported Classes:</p>
            <div class="classes-grid">
                <span class="class-badge">1-9</span>
                <span class="class-badge">A-Z</span>
            </div>
        </div>
    </div>
    """)
    
    # Footer
    gr.HTML("""
    <div class="footer">
        <p>🎓 ISL Recognition Project | Built with TensorFlow & Gradio</p>
    </div>
    """)
    
    # Event handlers
    predict_btn.click(
        fn=predict_sign,
        inputs=input_image,
        outputs=[output_label, result_html]
    )
    
    input_image.change(
        fn=predict_sign,
        inputs=input_image,
        outputs=[output_label, result_html]
    )


# ============================================================================
# Launch
# ============================================================================
if __name__ == "__main__":
    demo.launch()
