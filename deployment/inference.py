"""
ISL Recognition - Inference Module
==================================
Core inference functions for Indian Sign Language character recognition.
"""

import os
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, List, Dict, Any

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "saved_models"

# Default model paths
CNN_MODEL_PATH = MODEL_DIR / "cnn_model_final.keras"
MOBILENET_MODEL_PATH = MODEL_DIR / "mobilenet_model_final.keras"
CLASS_LABELS_PATH = MODEL_DIR / "class_labels.pkl"
CLASS_INDICES_PATH = MODEL_DIR / "class_indices.pkl"

# Image configuration
IMG_SIZE = (128, 128)


# ============================================================================
# Model Loading
# ============================================================================
class ISLRecognizer:
    """
    Indian Sign Language Character Recognizer.
    
    Supports both CNN and MobileNetV2 (Transfer Learning) models.
    """
    
    def __init__(self, model_type: str = "mobilenet"):
        """
        Initialize the ISL Recognizer.
        
        Args:
            model_type: 'cnn' or 'mobilenet' (default: 'mobilenet')
        """
        self.model_type = model_type
        self.model = None
        self.class_labels = None
        self.class_indices = None
        self._load_model()
        self._load_labels()
    
    def _load_model(self):
        """Load the specified model."""
        if self.model_type == "cnn":
            model_path = CNN_MODEL_PATH
        else:
            model_path = MOBILENET_MODEL_PATH
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please train the model first using the Jupyter notebook."
            )
        
        print(f"Loading {self.model_type.upper()} model...")
        self.model = load_model(str(model_path))
        print(f"Model loaded successfully!")
    
    def _load_labels(self):
        """Load class labels and indices."""
        if CLASS_LABELS_PATH.exists():
            with open(CLASS_LABELS_PATH, 'rb') as f:
                self.class_labels = pickle.load(f)
        else:
            raise FileNotFoundError(
                f"Class labels not found at {CLASS_LABELS_PATH}. "
                "Please train the model first."
            )
        
        if CLASS_INDICES_PATH.exists():
            with open(CLASS_INDICES_PATH, 'rb') as f:
                self.class_indices = pickle.load(f)
    
    def preprocess_image(self, img_input) -> np.ndarray:
        """
        Preprocess an image for prediction.
        
        Args:
            img_input: Can be a file path (str/Path), PIL Image, or numpy array
        
        Returns:
            Preprocessed numpy array ready for prediction
        """
        if isinstance(img_input, (str, Path)):
            # Load from file path
            img = image.load_img(str(img_input), target_size=IMG_SIZE)
            img_array = image.img_to_array(img)
        elif hasattr(img_input, 'read'):
            # File-like object (e.g., uploaded file)
            from PIL import Image
            img = Image.open(img_input).convert('RGB')
            img = img.resize(IMG_SIZE)
            img_array = np.array(img, dtype=np.float32)
        elif isinstance(img_input, np.ndarray):
            # Already a numpy array
            from PIL import Image
            if img_input.shape[:2] != IMG_SIZE:
                img = Image.fromarray(img_input.astype('uint8')).resize(IMG_SIZE)
                img_array = np.array(img, dtype=np.float32)
            else:
                img_array = img_input.astype(np.float32)
        else:
            # Assume PIL Image
            img = img_input.convert('RGB').resize(IMG_SIZE)
            img_array = np.array(img, dtype=np.float32)
        
        # Normalize and add batch dimension
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, img_input) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict ISL character from an image.
        
        Args:
            img_input: Image path, PIL Image, or numpy array
        
        Returns:
            Tuple of (predicted_label, confidence_percentage, all_probabilities)
        """
        img_array = self.preprocess_image(img_input)
        
        # Get predictions
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get predicted class
        predicted_idx = int(np.argmax(predictions))
        predicted_label = self.class_labels[predicted_idx]
        confidence = float(predictions[predicted_idx]) * 100
        
        # Get all probabilities
        all_probs = {
            self.class_labels[i]: float(predictions[i]) * 100 
            for i in range(len(predictions))
        }
        
        return predicted_label, confidence, all_probs
    
    def predict_top_k(self, img_input, k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top-k predictions for an image.
        
        Args:
            img_input: Image path, PIL Image, or numpy array
            k: Number of top predictions to return
        
        Returns:
            List of (label, confidence) tuples sorted by confidence
        """
        _, _, all_probs = self.predict(img_input)
        
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_probs[:k]
    
    def batch_predict(self, img_inputs: List) -> List[Tuple[str, float]]:
        """
        Perform batch prediction on multiple images.
        
        Args:
            img_inputs: List of image paths, PIL Images, or numpy arrays
        
        Returns:
            List of (predicted_label, confidence) tuples
        """
        results = []
        
        # Prepare batch
        images = []
        for img_input in img_inputs:
            img_array = self.preprocess_image(img_input)
            images.append(img_array[0])  # Remove batch dimension
        
        images = np.array(images)
        
        # Batch predict
        predictions = self.model.predict(images, verbose=0)
        
        for pred in predictions:
            predicted_idx = int(np.argmax(pred))
            predicted_label = self.class_labels[predicted_idx]
            confidence = float(pred[predicted_idx]) * 100
            results.append((predicted_label, confidence))
        
        return results
    
    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return len(self.class_labels)
    
    @property
    def classes(self) -> List[str]:
        """Return list of class names."""
        return list(self.class_indices.keys()) if self.class_indices else []


# ============================================================================
# Convenience Functions
# ============================================================================
_default_recognizer = None

def get_recognizer(model_type: str = "mobilenet") -> ISLRecognizer:
    """
    Get or create a singleton recognizer instance.
    
    Args:
        model_type: 'cnn' or 'mobilenet'
    
    Returns:
        ISLRecognizer instance
    """
    global _default_recognizer
    if _default_recognizer is None or _default_recognizer.model_type != model_type:
        _default_recognizer = ISLRecognizer(model_type)
    return _default_recognizer


def predict_sign(img_input, model_type: str = "mobilenet") -> Tuple[str, float]:
    """
    Quick prediction function.
    
    Args:
        img_input: Image path, PIL Image, or numpy array
        model_type: 'cnn' or 'mobilenet'
    
    Returns:
        Tuple of (predicted_label, confidence_percentage)
    """
    recognizer = get_recognizer(model_type)
    label, confidence, _ = recognizer.predict(img_input)
    return label, confidence


# ============================================================================
# CLI Testing
# ============================================================================
if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("ISL Recognition - Inference Test")
    print("=" * 60)
    
    # Initialize recognizer
    recognizer = ISLRecognizer(model_type="mobilenet")
    print(f"\nNumber of classes: {recognizer.num_classes}")
    print(f"Classes: {recognizer.classes[:10]}... (showing first 10)")
    
    # Test with an image if provided
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        print(f"\nTesting with image: {test_image}")
        
        label, confidence, _ = recognizer.predict(test_image)
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2f}%")
        
        print("\nTop 5 Predictions:")
        for label, conf in recognizer.predict_top_k(test_image, k=5):
            print(f"• {label}: {conf:.2f}%")
    else:
        print("\nUsage: python inference.py <image_path>")
