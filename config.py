# config.py
"""
Centralized configuration for the gesture recognition system.
All paths, filenames, and constants defined here.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS AND DIRECTORIES
# ============================================================================

# Base project directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "gesture_dataset"
MODELS_DIR = PROJECT_ROOT / "models"
EXPORTS_DIR = PROJECT_ROOT / "exports"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, EXPORTS_DIR, LOGS_DIR, CONFIGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# ============================================================================
# MODEL FILES
# ============================================================================

# Training models
MODEL_CHECKPOINT = MODELS_DIR / "best_model.keras"
FINAL_MODEL = MODELS_DIR / "gesture_model.keras"
TFLITE_MODEL = EXPORTS_DIR / "gesture_model.tflite"
QUANTIZED_TFLITE_MODEL = EXPORTS_DIR / "gesture_model_quantized.tflite"

# Model metadata
MODEL_METADATA = MODELS_DIR / "model_metadata.json"
CONTRACT_EXPORT = CONFIGS_DIR / "gesture_contract.json"

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Data collection
SEQUENCE_LENGTH = 30  # Frames per sequence
MIN_SAMPLES_PER_GESTURE = 150  # Minimum samples needed per gesture
TARGET_SAMPLES_PER_GESTURE = 300  # Ideal samples per gesture

# Training parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Data augmentation
USE_AUGMENTATION = True
AUGMENTATION_FACTOR = 2  # Multiply dataset by this factor

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================

# Real-time inference
INFERENCE_CONFIDENCE_THRESHOLD = 0.7
PREDICTION_HISTORY_LENGTH = 5  # For majority voting
MIN_FRAMES_FOR_INFERENCE = SEQUENCE_LENGTH

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_FPS = 20  # Target frames per second for processing

# ============================================================================
# GESTURE CONFIGURATION
# ============================================================================

# Import gesture classes (must be done after config definition)
from gesture_action_contract import GESTURE_CLASSES, NUM_GESTURES

# ============================================================================
# VALIDATION AND CHECKS
# ============================================================================

def validate_paths():
    """Validate that all required paths and directories exist."""
    required_dirs = [DATA_DIR, MODELS_DIR, EXPORTS_DIR]
    
    for directory in required_dirs:
        if not directory.exists():
            print(f"⚠️  Creating directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
    
    return True

def get_model_path(model_type="best"):
    """
    Get the path to a specific model file.
    
    Args:
        model_type: "best" (checkpoint), "final", "tflite", or "quantized"
    
    Returns:
        Path object to the model file
    """
    model_paths = {
        "best": MODEL_CHECKPOINT,
        "final": FINAL_MODEL,
        "tflite": TFLITE_MODEL,
        "quantized": QUANTIZED_TFLITE_MODEL,
    }
    
    if model_type not in model_paths:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Available: {list(model_paths.keys())}"
        )
    
    return model_paths[model_type]

def check_model_exists(model_type="best", raise_error=False):
    """
    Check if a model file exists.
    
    Args:
        model_type: Type of model to check
        raise_error: If True, raise FileNotFoundError when missing
    
    Returns:
        bool: True if model exists
    """
    model_path = get_model_path(model_type)
    
    if not model_path.exists():
        if raise_error:
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Run training first: python main.py train"
            )
        return False
    
    return True

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

# Detect environment
def is_production():
    """Check if running in production environment."""
    return os.getenv("ENVIRONMENT", "development").lower() == "production"

def is_colab():
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

# Set environment-specific settings
if is_colab():
    # Colab-specific settings
    CAMERA_INDEX = 0  # Webcam in Colab
    DATA_DIR = Path("/content/gesture_dataset")
    MODELS_DIR = Path("/content/models")
    
elif is_production():
    # Production settings
    INFERENCE_CONFIDENCE_THRESHOLD = 0.8  # Higher threshold in production
    USE_AUGMENTATION = False  # No augmentation in production

# Initialize on import
validate_paths()