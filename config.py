"""
Centralized configuration for the gesture recognition system.
All paths, filenames, and constants defined here.

CRITICAL FIXES APPLIED:
1. Added FEATURE_CONFIG with deterministic settings
2. Added feature_dimension to MODEL_METADATA
3. Ensured idle features are consistently enabled everywhere
4. Added configuration hash for validation
"""

import os
import sys
from pathlib import Path
import json
from dataclasses import dataclass
from typing import Dict, Any

# ============================================================================
# FEATURE ENGINEERING CONFIGURATION (CRITICAL FIX)
# ============================================================================

@dataclass
class FeatureConfig:
    """Configuration for feature extraction - MUST BE CONSISTENT ACROSS TRAINING/INFERENCE."""
    # Visibility thresholds
    min_visibility: float = 0.5
    angle_visibility: float = 0.3
    
    # Normalization
    normalize_by_torso: bool = True
    reference_torso_length: float = 0.5
    
    # Velocity calculation
    assume_fps: float = 30.0
    velocity_smoothing: float = 0.3
    
    # Feature selection
    compute_angles: bool = True
    compute_distances: bool = True
    compute_torso_lean: bool = True
    compute_velocities: bool = True
    compute_height: bool = True
    
    # Jogging-specific features
    compute_jogging_features: bool = True
    compute_periodicity: bool = True
    compute_alternation: bool = True
    
    # IDLE FEATURES - CRITICAL: MUST BE TRUE FOR CONSISTENCY
    compute_idle_features: bool = True  # Now explicitly True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'min_visibility': self.min_visibility,
            'angle_visibility': self.angle_visibility,
            'normalize_by_torso': self.normalize_by_torso,
            'reference_torso_length': self.reference_torso_length,
            'assume_fps': self.assume_fps,
            'velocity_smoothing': self.velocity_smoothing,
            'compute_angles': self.compute_angles,
            'compute_distances': self.compute_distances,
            'compute_torso_lean': self.compute_torso_lean,
            'compute_velocities': self.compute_velocities,
            'compute_height': self.compute_height,
            'compute_jogging_features': self.compute_jogging_features,
            'compute_periodicity': self.compute_periodicity,
            'compute_alternation': self.compute_alternation,
            'compute_idle_features': self.compute_idle_features,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureConfig':
        """Create config from dictionary."""
        return cls(**data)
    
    def get_hash(self) -> str:
        """Get unique hash for this configuration."""
        import hashlib
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def print_summary(self):
        """Print configuration summary."""
        print("\n🔧 FEATURE CONFIGURATION SUMMARY:")
        print("=" * 50)
        for key, value in self.to_dict().items():
            print(f"  {key:25s}: {value}")
        print(f"  {'config_hash':25s}: {self.get_hash()}")
        print("=" * 50)

# Instantiate the feature configuration (CRITICAL: Used everywhere)
FEATURE_CONFIG = FeatureConfig(
    compute_idle_features=True,  # MUST MATCH TRAINING - NO CONDITIONAL LOGIC
    # ... other settings can be customized here
)

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

# Feature-specific directories
FEATURES_DIR = MODELS_DIR / "features"
NORMALIZER_DIR = MODELS_DIR / "normalizers"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, EXPORTS_DIR, LOGS_DIR, CONFIGS_DIR, 
                  FEATURES_DIR, NORMALIZER_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# ============================================================================
# MODEL FILES
# ============================================================================

# Training models
MODEL_CHECKPOINT = MODELS_DIR / "best_model.keras"
FINAL_MODEL = MODELS_DIR / "gesture_model.keras"
TFLITE_MODEL = EXPORTS_DIR / "gesture_model.tflite"
QUANTIZED_TFLITE_MODEL = EXPORTS_DIR / "gesture_model_quantized.tflite"

# Model metadata (CRITICAL FIX: Store feature dimension and config)
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"
FEATURE_CONFIG_FILE = CONFIGS_DIR / "feature_config.json"
NORMALIZER_FILE = NORMALIZER_DIR / "feature_normalizer.npz"
CONTRACT_EXPORT = CONFIGS_DIR / "gesture_contract.json"

# Feature dimension placeholder (will be set during training)
FEATURE_DIMENSION: int = None  # This MUST be set during training and used for inference

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Data collection
SEQUENCE_LENGTH = 30  # Frames per sequence
MIN_SAMPLES_PER_GESTURE = 100  # Minimum samples needed per gesture
TARGET_SAMPLES_PER_GESTURE = 200  # Ideal samples per gesture

# Idle class configuration (CRITICAL FOR FALSE POSITIVE PREVENTION)
IDLE_SAMPLES_MULTIPLIER = 1.5  # Idle should have 1.5x more samples than max active gesture

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

# Different thresholds for idle vs active gestures
IDLE_CONFIDENCE_THRESHOLD = 0.6  # Lower threshold for idle (more lenient)
ACTIVE_CONFIDENCE_THRESHOLD = 0.7  # Higher threshold for active gestures

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_FPS = 20  # Target frames per second for processing

# ============================================================================
# GESTURE CONFIGURATION (VERIFIED)
# ============================================================================

# IMPORTANT: Import gesture classes from contract AFTER config is defined
# This ensures the contract is validated before use
try:
    from gesture_action_contract import GESTURE_CLASSES, NUM_GESTURES
except ImportError as e:
    print(f"❌ Failed to import gesture classes: {e}")
    print("   Make sure gesture_action_contract.py exists and defines GESTURE_CLASSES")
    sys.exit(1)

# ============================================================================
# MODEL METADATA STRUCTURE (CRITICAL FIX)
# ============================================================================

# Initialize model metadata structure
# This will be populated during training and validated during inference
MODEL_METADATA = {
    'feature_dimension': None,  # Will be set during training
    'feature_config': FEATURE_CONFIG.to_dict(),  # Store the exact config used
    'config_hash': FEATURE_CONFIG.get_hash(),  # For validation
    'classes': GESTURE_CLASSES,
    'sequence_length': SEQUENCE_LENGTH,
    'timestamp': None,  # Will be set during training
}

def save_model_metadata(additional_data: Dict[str, Any] = None):
    """
    Save model metadata to file.
    
    Args:
        additional_data: Additional metadata to include
    """
    metadata = MODEL_METADATA.copy()
    
    # Add feature dimension (must be set by this point)
    if FEATURE_DIMENSION is None:
        raise RuntimeError("FEATURE_DIMENSION must be set before saving metadata")
    
    metadata['feature_dimension'] = FEATURE_DIMENSION
    metadata['timestamp'] = get_timestamp()
    
    # Add any additional data
    if additional_data:
        metadata.update(additional_data)
    
    # Save to file
    with open(MODEL_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Also save feature config separately
    with open(FEATURE_CONFIG_FILE, 'w') as f:
        json.dump(FEATURE_CONFIG.to_dict(), f, indent=2)
    
    print(f"✅ Model metadata saved to {MODEL_METADATA_FILE}")
    print(f"   Feature dimension: {FEATURE_DIMENSION}")
    print(f"   Config hash: {FEATURE_CONFIG.get_hash()}")
    print(f"   Classes: {GESTURE_CLASSES}")

def load_model_metadata() -> Dict[str, Any]:
    """
    Load model metadata from file.
    
    Returns:
        Dictionary with model metadata
        
    Raises:
        FileNotFoundError: If metadata file doesn't exist
    """
    if not MODEL_METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Model metadata file not found: {MODEL_METADATA_FILE}\n"
            f"Train a model first: python main.py train"
        )
    
    with open(MODEL_METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    # Update global feature dimension
    global FEATURE_DIMENSION
    FEATURE_DIMENSION = metadata.get('feature_dimension')
    
    if FEATURE_DIMENSION is None:
        raise ValueError("Feature dimension not found in metadata")
    
    print(f"✅ Model metadata loaded")
    print(f"   Feature dimension: {FEATURE_DIMENSION}")
    print(f"   Config hash: {metadata.get('config_hash', 'N/A')}")
    print(f"   Classes: {metadata.get('classes', [])}")
    
    return metadata

def validate_metadata_consistency(metadata: Dict[str, Any]) -> bool:
    """
    Validate that loaded metadata is consistent with current configuration.
    
    Args:
        metadata: Loaded metadata dictionary
        
    Returns:
        True if validation passes
    """
    print("🔍 Validating metadata consistency...")
    
    issues = []
    
    # 1. Check feature config hash
    current_hash = FEATURE_CONFIG.get_hash()
    saved_hash = metadata.get('config_hash')
    
    if current_hash != saved_hash:
        issues.append(
            f"❌ Feature configuration hash mismatch:\n"
            f"   Current: {current_hash}\n"
            f"   Saved:   {saved_hash}\n"
            f"   This will cause silent model degradation!"
        )
    else:
        print(f"   ✅ Feature config hash matches: {current_hash}")
    
    # 2. Check classes
    saved_classes = metadata.get('classes', [])
    if saved_classes != GESTURE_CLASSES:
        issues.append(
            f"❌ Gesture classes mismatch:\n"
            f"   Current: {GESTURE_CLASSES}\n"
            f"   Saved:   {saved_classes}"
        )
    else:
        print(f"   ✅ Gesture classes match: {len(GESTURE_CLASSES)} classes")
    
    # 3. Check feature dimension is set
    feature_dim = metadata.get('feature_dimension')
    if feature_dim is None:
        issues.append("❌ Feature dimension not found in metadata")
    else:
        print(f"   ✅ Feature dimension: {feature_dim}")
    
    # 4. Check idle feature consistency (CRITICAL)
    saved_config = metadata.get('feature_config', {})
    saved_idle = saved_config.get('compute_idle_features', False)
    current_idle = FEATURE_CONFIG.compute_idle_features
    
    if saved_idle != current_idle:
        issues.append(
            f"❌ IDLE FEATURE CONFIGURATION MISMATCH (CRITICAL):\n"
            f"   Saved: compute_idle_features={saved_idle}\n"
            f"   Current: compute_idle_features={current_idle}\n"
            f"   This will cause dimension mismatch and model failure!"
        )
    else:
        print(f"   ✅ Idle features consistent: {current_idle}")
    
    if issues:
        print("\n" + "=" * 60)
        print("METADATA VALIDATION FAILED:")
        for issue in issues:
            print(f"\n{issue}")
        print("\n" + "=" * 60)
        return False
    
    print("✅ All metadata validations passed!")
    return True

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def verify_idle_class():
    """Verify that idle class is properly configured."""
    print("🔍 Verifying idle class configuration...")
    
    issues = []
    
    # 1. Check if idle is in GESTURE_CLASSES
    if 'idle' not in GESTURE_CLASSES:
        issues.append("❌ 'idle' not found in GESTURE_CLASSES")
    else:
        print(f"   ✅ 'idle' found in GESTURE_CLASSES at index {GESTURE_CLASSES.index('idle')}")
    
    # 2. Check NUM_GESTURES matches GESTURE_CLASSES length
    if NUM_GESTURES != len(GESTURE_CLASSES):
        issues.append(f"❌ NUM_GESTURES ({NUM_GESTURES}) doesn't match GESTURE_CLASSES length ({len(GESTURE_CLASSES)})")
    else:
        print(f"   ✅ NUM_GESTURES ({NUM_GESTURES}) matches GESTURE_CLASSES length")
    
    # 3. Check if idle directory exists
    idle_dir = DATA_DIR / "idle"
    if not idle_dir.exists():
        print(f"   ⚠️  Idle directory not found: {idle_dir}")
        print(f"      Creating directory...")
        idle_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"   ✅ Idle directory exists: {idle_dir}")
    
    # 4. Check other gesture directories
    print(f"   📁 Checking gesture directories...")
    for gesture in GESTURE_CLASSES:
        if gesture != 'idle':  # Skip idle, we already checked it
            gesture_dir = DATA_DIR / gesture
            if not gesture_dir.exists():
                print(f"      ⚠️  Missing directory for '{gesture}': {gesture_dir}")
    
    # 5. Verify feature config has idle features enabled
    if not FEATURE_CONFIG.compute_idle_features:
        issues.append("❌ FeatureConfig.compute_idle_features is False! Must be True for consistency")
    else:
        print(f"   ✅ FeatureConfig.compute_idle_features is True (correct)")
    
    if issues:
        print("\n❌ IDLE CLASS CONFIGURATION ISSUES:")
        for issue in issues:
            print(f"   {issue}")
        print("\n   Fix these issues before training!")
        return False
    
    print("\n✅ Idle class configuration verified successfully!")
    return True

def get_required_idle_samples(active_gesture_count: int = None) -> int:
    """
    Calculate required idle samples based on active gestures.
    
    Args:
        active_gesture_count: Count of samples in the most frequent active gesture.
                              If None, uses TARGET_SAMPLES_PER_GESTURE.
    
    Returns:
        Required number of idle samples
    """
    if active_gesture_count is None:
        active_gesture_count = TARGET_SAMPLES_PER_GESTURE
    
    return int(active_gesture_count * IDLE_SAMPLES_MULTIPLIER)

def get_confidence_threshold(gesture_name: str) -> float:
    """Get appropriate confidence threshold based on gesture type."""
    if gesture_name == "idle":
        return IDLE_CONFIDENCE_THRESHOLD
    else:
        return ACTIVE_CONFIDENCE_THRESHOLD

def get_gesture_index(gesture_name: str) -> int:
    """
    Get the index of a gesture in GESTURE_CLASSES.
    
    Args:
        gesture_name: Name of the gesture
        
    Returns:
        Integer index (0-based)
        
    Raises:
        ValueError: If gesture_name is not in GESTURE_CLASSES
    """
    try:
        return GESTURE_CLASSES.index(gesture_name)
    except ValueError:
        raise ValueError(
            f"Gesture '{gesture_name}' not found in GESTURE_CLASSES. "
            f"Available gestures: {GESTURE_CLASSES}"
        )

def get_gesture_name(index: int) -> str:
    """
    Get the name of a gesture from its index.
    
    Args:
        index: Index of the gesture
        
    Returns:
        Gesture name
        
    Raises:
        IndexError: If index is out of bounds
    """
    if index < 0 or index >= NUM_GESTURES:
        raise IndexError(
            f"Gesture index {index} out of bounds. "
            f"Valid indices: 0-{NUM_GESTURES-1}"
        )
    return GESTURE_CLASSES[index]

def print_gesture_summary():
    """Print a summary of all gesture classes."""
    print("\n" + "=" * 60)
    print("GESTURE CLASSES SUMMARY")
    print("=" * 60)
    print(f"Total gestures: {NUM_GESTURES}")
    print(f"Active gestures: {NUM_GESTURES - 1} (excluding idle)")
    print(f"Idle class: {'✅ Present' if 'idle' in GESTURE_CLASSES else '❌ Missing'}")
    print("\nAll gesture classes:")
    for i, gesture in enumerate(GESTURE_CLASSES):
        marker = "⭐ " if gesture == "idle" else "  "
        print(f"  {marker}{i:2d}. {gesture}")
    print("=" * 60)

def get_timestamp() -> str:
    """Get current timestamp string."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ============================================================================
# VALIDATION AND CHECKS
# ============================================================================

def validate_paths():
    """Validate that all required paths and directories exist."""
    required_dirs = [DATA_DIR, MODELS_DIR, EXPORTS_DIR, FEATURES_DIR, NORMALIZER_DIR]
    
    for directory in required_dirs:
        if not directory.exists():
            print(f"⚠️  Creating directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
    
    return True

def validate_dataset_structure():
    """Validate the dataset directory structure."""
    print("\n🔍 Validating dataset structure...")
    
    # Check if data directory exists
    if not DATA_DIR.exists():
        print(f"❌ Data directory not found: {DATA_DIR}")
        return False
    
    # Check for gesture directories
    missing_dirs = []
    for gesture in GESTURE_CLASSES:
        gesture_dir = DATA_DIR / gesture
        if not gesture_dir.exists():
            missing_dirs.append(gesture)
    
    if missing_dirs:
        print(f"⚠️  Missing directories for gestures: {missing_dirs}")
        print(f"   Creating missing directories...")
        for gesture in missing_dirs:
            (DATA_DIR / gesture).mkdir(parents=True, exist_ok=True)
    
    print("✅ Dataset structure validated")
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

def check_metadata_exists(raise_error=False):
    """
    Check if model metadata exists.
    
    Args:
        raise_error: If True, raise FileNotFoundError when missing
    
    Returns:
        bool: True if metadata exists
    """
    if not MODEL_METADATA_FILE.exists():
        if raise_error:
            raise FileNotFoundError(
                f"Model metadata file not found: {MODEL_METADATA_FILE}\n"
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

def is_testing():
    """Check if running in test mode."""
    return os.getenv("TEST_MODE", "false").lower() == "true"

# Set environment-specific settings
if is_colab():
    # Colab-specific settings
    print("🔧 Google Colab environment detected")
    CAMERA_INDEX = 0  # Webcam in Colab
    DATA_DIR = Path("/content/gesture_dataset")
    MODELS_DIR = Path("/content/models")
    EXPORTS_DIR = Path("/content/exports")
    FEATURES_DIR = Path("/content/models/features")
    NORMALIZER_DIR = Path("/content/models/normalizers")
    
elif is_production():
    # Production settings
    print("🔧 Production environment detected")
    INFERENCE_CONFIDENCE_THRESHOLD = 0.8  # Higher threshold in production
    USE_AUGMENTATION = False  # No augmentation in production
    IDLE_CONFIDENCE_THRESHOLD = 0.5  # Even more lenient in production
    ACTIVE_CONFIDENCE_THRESHOLD = 0.75  # Higher threshold for active gestures

elif is_testing():
    # Testing settings
    print("🔧 Testing environment detected")
    MIN_SAMPLES_PER_GESTURE = 10  # Lower for testing
    TARGET_SAMPLES_PER_GESTURE = 20  # Lower for testing
    EPOCHS = 5  # Fewer epochs for testing

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_system():
    """Initialize and validate the entire system."""
    print("\n" + "=" * 60)
    print("SYSTEM INITIALIZATION")
    print("=" * 60)
    
    # 1. Validate paths
    validate_paths()
    
    # 2. Validate dataset structure
    validate_dataset_structure()
    
    # 3. Verify idle class configuration
    if not verify_idle_class():
        print("\n❌ System initialization failed!")
        return False
    
    # 4. Print gesture summary
    print_gesture_summary()
    
    # 5. Print feature configuration
    FEATURE_CONFIG.print_summary()
    
    # 6. Validate contract consistency
    try:
        from gesture_action_contract import GestureActionContract
        GestureActionContract.validate_consistency()
        print("✅ Gesture contract validation passed")
    except Exception as e:
        print(f"❌ Gesture contract validation failed: {e}")
        return False
    
    print("\n✅ System initialization complete!")
    print("=" * 60)
    
    return True

# ============================================================================
# CONSTANT VERIFICATION
# ============================================================================

# Verify critical constants are set
assert SEQUENCE_LENGTH > 0, "SEQUENCE_LENGTH must be positive"
assert MIN_SAMPLES_PER_GESTURE > 0, "MIN_SAMPLES_PER_GESTURE must be positive"
assert IDLE_SAMPLES_MULTIPLIER >= 1.0, "IDLE_SAMPLES_MULTIPLIER must be >= 1.0"
assert NUM_GESTURES > 0, "NUM_GESTURES must be positive"
assert FEATURE_CONFIG.compute_idle_features, "compute_idle_features MUST be True for consistency"

# ============================================================================
# AUTO-INITIALIZATION
# ============================================================================

# Auto-initialize on import (unless testing)
if not is_testing():
    try:
        initialize_system()
    except Exception as e:
        print(f"⚠️  System initialization warning: {e}")
        print("   Some features may not work correctly")