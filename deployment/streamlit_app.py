"""
ISL Recognition - Streamlit Frontend
=====================================
Interactive web interface for Indian Sign Language character recognition.

Run with:
    streamlit run deployment/streamlit_app.py

Or:
    python -m streamlit run deployment/streamlit_app.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import cv2

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="ISL Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# Custom CSS
# ============================================================================
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Prediction result styling */
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    
    .prediction-label {
        font-size: 4rem;
        font-weight: bold;
        margin: 0;
    }
    
    .confidence-score {
        font-size: 1.5rem;
        opacity: 0.9;
    }
    
    /* Live detection styling */
    .live-prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin: 0.5rem 0;
    }
    
    .live-label {
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Image upload area */
    .uploadedFile {
        border: 2px dashed #667eea !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Model Loading
# ============================================================================
@st.cache_resource
def load_recognizer(model_type: str):
    """Load and cache the ISL recognizer."""
    try:
        from deployment.inference import ISLRecognizer
        return ISLRecognizer(model_type)
    except FileNotFoundError as e:
        st.error(f"Model not found: {e}")
        st.info("Please train the model first using the Jupyter notebook.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# ============================================================================
# Helper Functions
# ============================================================================
def create_confidence_chart(predictions: list) -> go.Figure:
    """Create a horizontal bar chart for predictions."""
    labels = [p[0] for p in predictions]
    confidences = [p[1] for p in predictions]
    
    colors = ['#38ef7d' if i == 0 else '#667eea' for i in range(len(labels))]
    
    fig = go.Figure(go.Bar(
        x=confidences,
        y=labels,
        orientation='h',
        marker_color=colors,
        text=[f'{c:.1f}%' for c in confidences],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Top Predictions",
        xaxis_title="Confidence (%)",
        yaxis_title="Character",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(range=[0, 100]),
        yaxis=dict(categoryorder='total ascending')
    )
    
    return fig


# ============================================================================
# Live Webcam Detection
# ============================================================================
def live_detection_page(recognizer, model_type: str, top_k: int):
    """Live webcam ISL detection page using Streamlit's camera input."""
    st.header("Live Webcam Detection")
    st.markdown("Capture an image of your ISL hand gesture for recognition.")
    
    # Camera input - Streamlit's built-in camera widget
    camera_image = st.camera_input("Show your ISL hand gesture")
    
    if camera_image is not None:
        # Convert to PIL Image
        image = Image.open(camera_image).convert("RGB")
        
        # Create two columns for display
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Captured Image", width="stretch")
        
        with col2:
            with st.spinner("Analyzing..."):
                try:
                    # Get prediction
                    label, confidence, all_probs = recognizer.predict(image)
                    top_predictions = recognizer.predict_top_k(image, k=top_k)
                    
                    # Display main prediction
                    st.markdown(f"""
                    <div class="live-prediction-box">
                        <p class="live-label">{label}</p>
                        <p class="confidence-score">Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display metrics
                    metric_cols = st.columns(3)
                    metric_cols[0].metric("Predicted", label)
                    metric_cols[1].metric("Confidence", f"{confidence:.1f}%")
                    metric_cols[2].metric("Model", model_type.upper())
                    
                    # Display confidence chart
                    st.plotly_chart(
                        create_confidence_chart(top_predictions),
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
    else:
        # Placeholder when no image captured
        st.markdown("""
        <div style="background: #f0f2f6; padding: 3rem; border-radius: 12px; text-align: center;">
            <h3>No Image Captured</h3>
            <p>Click the camera button above to capture your hand gesture</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Instructions
        st.markdown("""
        ### How to use:
        1. Click the **camera button** above to open your webcam
        2. Position your hand showing an ISL gesture
        3. Click **Take Photo** to capture
        4. View the prediction results
        """)


# ============================================================================
# Image Upload Page
# ============================================================================
def upload_image_page(recognizer, model_type: str, top_k: int):
    """Image upload prediction page."""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an ISL hand gesture image",
            type=["jpg", "jpeg", "png"],
            help="Upload an image of an ISL hand gesture"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", width="stretch")
            
            # Image info
            st.caption(f"Image size: {image.size[0]} × {image.size[1]} pixels")
    
    with col2:
        st.header("Prediction")
        
        if uploaded_file is not None:
            with st.spinner("Analyzing image..."):
                try:
                    # Get prediction
                    label, confidence, all_probs = recognizer.predict(image)
                    top_predictions = recognizer.predict_top_k(image, k=top_k)
                    
                    # Display main prediction
                    st.markdown(f"""
                    <div class="prediction-box">
                        <p class="prediction-label">{label}</p>
                        <p class="confidence-score">Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display metrics
                    metric_cols = st.columns(3)
                    metric_cols[0].metric("Predicted", label)
                    metric_cols[1].metric("Confidence", f"{confidence:.1f}%")
                    metric_cols[2].metric("Model", model_type.upper())
                    
                    # Display confidence chart
                    st.plotly_chart(
                        create_confidence_chart(top_predictions),
                        width="stretch"
                    )
                    
                    # Detailed predictions table
                    with st.expander("View All Predictions"):
                        import pandas as pd
                        
                        sorted_probs = sorted(
                            all_probs.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )
                        
                        df = pd.DataFrame(
                            sorted_probs,
                            columns=["Character", "Confidence (%)"]
                        )
                        df["Confidence (%)"] = df["Confidence (%)"].round(2)
                        
                        st.dataframe(
                            df,
                            width="stretch",
                            hide_index=True
                        )
                
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        else:
            # Placeholder when no image uploaded
            st.info("Please upload an image to get started")
            
            # Show example predictions
            st.markdown("### Example Classes")
            example_classes = ["A", "B", "C", "1", "2", "3"]
            cols = st.columns(len(example_classes))
            for i, cls in enumerate(example_classes):
                cols[i].markdown(f"<div style='text-align:center; font-size:2rem; padding:1rem; background:#f0f2f6; border-radius:8px;'><b>{cls}</b></div>", unsafe_allow_html=True)


# ============================================================================
# Main Application
# ============================================================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>ISL Recognition</h1>
        <p>Indian Sign Language Character Recognition using Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Mode selection
        mode = st.radio(
            "Detection Mode",
            options=["Upload Image", "Live Webcam"],
            help="Choose between uploading an image or using live webcam"
        )
        
        st.divider()
        
        # Model selection
        model_type = st.selectbox(
            "Select Model",
            options=["mobilenet", "cnn"],
            format_func=lambda x: "MobileNetV2 (Recommended)" if x == "mobilenet" else "Custom CNN",
            help="MobileNetV2 generally provides higher accuracy"
        )
        
        # Top-K setting
        top_k = st.slider(
            "Top-K Predictions",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of top predictions to display"
        )
        
        st.divider()
        
        # Model info
        st.header("Model Info")
        recognizer = load_recognizer(model_type)
        
        if recognizer:
            st.metric("Classes", recognizer.num_classes)
            st.metric("Model Type", model_type.upper())
            
            with st.expander("View All Classes"):
                classes = recognizer.classes
                cols = st.columns(5)
                for i, cls in enumerate(classes):
                    cols[i % 5].write(f"**{cls}**")
        
        st.divider()
        
        # Instructions
        st.header("Instructions")
        if mode == "Upload Image":
            st.markdown("""
            1. **Upload** an ISL hand gesture image
            2. **View** the prediction and confidence
            3. **Analyze** the top-K predictions chart
            
            **Supported formats:** JPG, PNG, JPEG
            """)
        else:
            st.markdown("""
            1. **Start** the camera
            2. **Position** your hand in the blue box
            3. **Show** ISL gestures for detection
            
            **Tip:** Good lighting improves accuracy
            """)
        
        st.divider()
        
        # About
        st.header("About")
        st.markdown("""
        This app uses deep learning to recognize
        Indian Sign Language (ISL) characters.
        
        **Dataset:** 35 classes (1-9, A-Z)
        
        **Models:**
        - Custom CNN
        - MobileNetV2 (Transfer Learning)
        """)
    
    # Main content
    if recognizer is None:
        st.warning("Model not loaded. Please check the sidebar for details.")
        return
    
    # Render selected mode
    if mode == "Upload Image":
        upload_image_page(recognizer, model_type, top_k)
    else:
        live_detection_page(recognizer, model_type, top_k)
    
    # Footer
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    col1.markdown("**Project:** ISL Recognition (DataMites)")
    col2.markdown(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    col3.markdown("**Version:** 1.1.0")


# ============================================================================
# Run App
# ============================================================================
if __name__ == "__main__":
    main()
