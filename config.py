"""
Centralized configuration for the gesture recognition system.
All paths, filenames, and constants defined here.

CRITICAL FIX: COMPLETE VERSION with ALL required functions for all modules.
NO UNDEFINED IMPORTS - EVERY function referenced elsewhere is implemented.
"""

import os
import sys
from pathlib import Path
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
import hashlib
from datetime import datetime

# ============================================================================
# FEATURE ENGINEERING CONFIGURATION (SINGLE SOURCE OF TRUTH)
# ============================================================================

@dataclass
class FeatureConfig:
    """
    Configuration for feature extraction - MUST BE CONSISTENT ACROSS TRAINING/INFERENCE.
    """
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
    
    # IDLE FEATURES - CRITICAL: MUST BE CONSISTENT
    compute_idle_features: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
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
        return cls(**data)
    
    def get_hash(self) -> str:
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def print_summary(self):
        print("\n🔧 FEATURE CONFIGURATION SUMMARY:")
        print("=" * 60)
        for key, value in self.to_dict().items():
            print(f"  {key:25s}: {value}")
        print(f"  {'config_hash':25s}: {self.get_hash()}")
        print("=" * 60)


# ============================================================================
# INSTANTIATE FEATURE CONFIGURATION
# ============================================================================
FEATURE_CONFIG = FeatureConfig(compute_idle_features=True)


# ============================================================================
# IDLE HANDLING STRATEGY
# ============================================================================

IDLE_STRATEGY = "explicit_features"  # DO NOT CHANGE
IDLE_FEATURES_ENABLED = FEATURE_CONFIG.compute_idle_features

# Idle class configuration
IDLE_SAMPLES_MULTIPLIER = 1.5

# Inference thresholds
IDLE_CONFIDENCE_THRESHOLD = 0.6
ACTIVE_CONFIDENCE_THRESHOLD = 0.7


# ============================================================================
# PATHS AND DIRECTORIES
# ============================================================================

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

# Create directories
for directory in [DATA_DIR, MODELS_DIR, EXPORTS_DIR, LOGS_DIR, CONFIGS_DIR, 
                  FEATURES_DIR, NORMALIZER_DIR]:
    directory.mkdir(exist_ok=True, parents=True)


# ============================================================================
# MODEL FILES
# ============================================================================

MODEL_CHECKPOINT = MODELS_DIR / "best_model.keras"
FINAL_MODEL = MODELS_DIR / "gesture_model.keras"
TFLITE_MODEL = EXPORTS_DIR / "gesture_model.tflite"
QUANTIZED_TFLITE_MODEL = EXPORTS_DIR / "gesture_model_quantized.tflite"

# Model metadata
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"
FEATURE_CONFIG_FILE = CONFIGS_DIR / "feature_config.json"
NORMALIZER_FILE = NORMALIZER_DIR / "feature_normalizer.npz"
CONTRACT_EXPORT = CONFIGS_DIR / "gesture_contract.json"

# Feature dimension placeholder
FEATURE_DIMENSION: Optional[int] = None


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

SEQUENCE_LENGTH = 30
MIN_SAMPLES_PER_GESTURE = 100
TARGET_SAMPLES_PER_GESTURE = 200

# Training parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Data augmentation
USE_AUGMENTATION = True
AUGMENTATION_FACTOR = 2


# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================

INFERENCE_CONFIDENCE_THRESHOLD = 0.7
PREDICTION_HISTORY_LENGTH = 5
MIN_FRAMES_FOR_INFERENCE = SEQUENCE_LENGTH

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_FPS = 20


# ============================================================================
# GESTURE CONTRACT - CRITICAL: FAIL HARD ON MISSING CONTRACT
# ============================================================================

GESTURE_CLASSES = []
NUM_GESTURES = 0

try:
    # CRITICAL: This import MUST succeed
    from gesture_action_contract import GESTURE_CLASSES as CONTRACT_CLASSES, NUM_GESTURES as CONTRACT_NUM
    
    GESTURE_CLASSES = CONTRACT_CLASSES
    NUM_GESTURES = CONTRACT_NUM
    
    # Validate idle is first
    if len(GESTURE_CLASSES) > 0 and GESTURE_CLASSES[0] != 'idle':
        print(f"\n⚠️  WARNING: 'idle' is not the first class (index {GESTURE_CLASSES.index('idle') if 'idle' in GESTURE_CLASSES else 'NOT FOUND'})")
        print(f"   First class is '{GESTURE_CLASSES[0]}' - this may cause issues")
    
    print(f"✅ Gesture contract loaded: {len(GESTURE_CLASSES)} classes")
    
except ImportError:
    error_msg = f"""
{'=' * 80}
❌ CRITICAL ERROR: gesture_action_contract.py is MANDATORY but missing!
{'=' * 80}

The system CANNOT operate without this file.

REQUIRED ACTION:
1. Create gesture_action_contract.py in: {PROJECT_ROOT}
2. Define:
   - GESTURE_CLASSES: List[str] = ['idle', 'attack', ...]
   - NUM_GESTURES: int = len(GESTURE_CLASSES)

Example:
   GESTURE_CLASSES = ['idle', 'attack', 'dodge', 'slide', 'block', 'forward_movement']
   NUM_GESTURES = len(GESTURE_CLASSES)
"""
    print(error_msg)
    sys.exit(1)


# ============================================================================
# MODEL METADATA FUNCTIONS - CRITICAL: ALL FUNCTIONS OTHER MODULES EXPECT
# ============================================================================

def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_feature_dimension(dimension: int):
    """
    Set the feature dimension globally.
    Expected by: train_gesture_model.py, feature_engineer.py
    """
    global FEATURE_DIMENSION
    FEATURE_DIMENSION = dimension
    print(f"✅ Feature dimension set to: {FEATURE_DIMENSION}")


def get_model_path(model_type: str = "final") -> Path:
    """
    Get path to model file.
    Expected by: main.py, real_time_inference.py
    """
    if model_type == "final":
        return FINAL_MODEL
    elif model_type == "checkpoint":
        return MODEL_CHECKPOINT
    elif model_type == "tflite":
        return TFLITE_MODEL
    elif model_type == "quantized":
        return QUANTIZED_TFLITE_MODEL
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def check_model_exists(model_type: str = "final") -> bool:
    """
    Check if model file exists.
    Expected by: main.py, real_time_inference.py
    """
    return get_model_path(model_type).exists()


def validate_paths() -> bool:
    """
    Validate all required paths exist.
    Expected by: main.py
    """
    required_dirs = [DATA_DIR, MODELS_DIR, EXPORTS_DIR, LOGS_DIR, CONFIGS_DIR]
    missing = [d for d in required_dirs if not d.exists()]
    
    if missing:
        print(f"❌ Missing directories: {missing}")
        return False
    
    return True


def is_production() -> bool:
    """
    Check if running in production mode.
    Expected by: main.py
    """
    return os.environ.get('GESTURE_RECOGNITION_ENV', 'development').lower() == 'production'


def get_required_idle_samples(active_gesture_count: int = None) -> int:
    """
    Calculate required idle samples based on active gestures.
    Expected by: train_gesture_model.py, data_pipeline.py
    """
    if active_gesture_count is None:
        active_gesture_count = TARGET_SAMPLES_PER_GESTURE
    
    return int(active_gesture_count * IDLE_SAMPLES_MULTIPLIER)


def get_confidence_threshold(gesture_name: str = None) -> float:
    """
    Get appropriate confidence threshold.
    Expected by: real_time_inference.py
    """
    if gesture_name == "idle":
        return IDLE_CONFIDENCE_THRESHOLD
    else:
        return ACTIVE_CONFIDENCE_THRESHOLD


def get_gesture_index(gesture_name: str) -> int:
    """
    Get the index of a gesture.
    Expected by: train_gesture_model.py, data_collector.py
    """
    try:
        return GESTURE_CLASSES.index(gesture_name)
    except ValueError:
        raise ValueError(f"Gesture '{gesture_name}' not found. Available: {GESTURE_CLASSES}")


def get_gesture_name(index: int) -> str:
    """
    Get the name of a gesture from its index.
    Expected by: real_time_inference.py, unity_integration.py
    """
    if index < 0 or index >= NUM_GESTURES:
        raise IndexError(f"Index {index} out of bounds (0-{NUM_GESTURES-1})")
    return GESTURE_CLASSES[index]


# ============================================================================
# METADATA SAVE/LOAD FUNCTIONS
# ============================================================================

def save_model_metadata(additional_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Save model metadata to file.
    Expected by: train_gesture_model.py, gesture_model.py
    """
    global FEATURE_DIMENSION
    
    if FEATURE_DIMENSION is None:
        raise RuntimeError(
            "FEATURE_DIMENSION must be set before saving metadata. "
            "Call set_feature_dimension() first."
        )
    
    metadata = {
        'feature_dimension': FEATURE_DIMENSION,
        'feature_config': FEATURE_CONFIG.to_dict(),
        'config_hash': FEATURE_CONFIG.get_hash(),
        'idle_strategy': IDLE_STRATEGY,
        'idle_features_enabled': IDLE_FEATURES_ENABLED,
        'idle_threshold': IDLE_CONFIDENCE_THRESHOLD,
        'active_threshold': ACTIVE_CONFIDENCE_THRESHOLD,
        'classes': GESTURE_CLASSES,
        'num_classes': NUM_GESTURES,
        'sequence_length': SEQUENCE_LENGTH,
        'timestamp': get_timestamp(),
    }
    
    if additional_data:
        metadata.update(additional_data)
    
    # Save to file
    with open(MODEL_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save feature config separately
    with open(FEATURE_CONFIG_FILE, 'w') as f:
        json.dump(FEATURE_CONFIG.to_dict(), f, indent=2)
    
    print(f"✅ Model metadata saved to {MODEL_METADATA_FILE}")
    print(f"   Feature dimension: {FEATURE_DIMENSION}")
    
    return metadata


def load_model_metadata() -> Dict[str, Any]:
    """
    Load model metadata from file.
    Expected by: real_time_inference.py, gesture_model.py
    """
    global FEATURE_DIMENSION
    
    if not MODEL_METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {MODEL_METADATA_FILE}\n"
            f"Train a model first: python main.py train"
        )
    
    with open(MODEL_METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    # Update global feature dimension
    FEATURE_DIMENSION = metadata.get('feature_dimension')
    
    if FEATURE_DIMENSION is None:
        raise ValueError("Feature dimension not found in metadata")
    
    print(f"✅ Model metadata loaded (dim={FEATURE_DIMENSION})")
    return metadata


def validate_metadata_consistency(metadata: Dict[str, Any]) -> bool:
    """
    Validate that loaded metadata is consistent with current configuration.
    Expected by: real_time_inference.py
    """
    issues = []
    
    # Check idle strategy (CRITICAL)
    saved_strategy = metadata.get('idle_strategy')
    if saved_strategy != IDLE_STRATEGY:
        issues.append(f"Idle strategy mismatch: saved='{saved_strategy}', current='{IDLE_STRATEGY}'")
    
    # Check idle features
    saved_idle_features = metadata.get('idle_features_enabled', False)
    if saved_idle_features != IDLE_FEATURES_ENABLED:
        issues.append(f"Idle features mismatch: saved={saved_idle_features}, current={IDLE_FEATURES_ENABLED}")
    
    # Check config hash
    current_hash = FEATURE_CONFIG.get_hash()
    saved_hash = metadata.get('config_hash')
    if current_hash != saved_hash:
        issues.append(f"Config hash mismatch: saved={saved_hash}, current={current_hash}")
    
    # Check classes
    saved_classes = metadata.get('classes', [])
    if saved_classes != GESTURE_CLASSES:
        issues.append(f"Classes mismatch: saved={saved_classes}, current={GESTURE_CLASSES}")
    
    if issues:
        print("\n❌ METADATA VALIDATION FAILED:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    
    print("✅ Metadata validation passed")
    return True


# ============================================================================
# SYSTEM INITIALIZATION
# ============================================================================

def initialize_system() -> bool:
    """
    Initialize and validate the entire system.
    Expected by: main.py
    """
    print("\n" + "=" * 60)
    print("SYSTEM INITIALIZATION")
    print("=" * 60)
    
    # Validate paths
    if not validate_paths():
        return False
    
    # Verify idle strategy
    if not FEATURE_CONFIG.compute_idle_features:
        print("❌ CRITICAL: compute_idle_features must be True")
        return False
    
    if IDLE_STRATEGY != "explicit_features":
        print(f"❌ IDLE_STRATEGY is '{IDLE_STRATEGY}', should be 'explicit_features'")
        return False
    
    print(f"✅ System initialized with {NUM_GESTURES} gesture classes")
    print(f"   Idle index: {get_gesture_index('idle') if 'idle' in GESTURE_CLASSES else 'NOT FOUND'}")
    
    return True


# ============================================================================
# AUTO-INITIALIZATION
# ============================================================================

# Run initialization on import
if not initialize_system():
    print("⚠️  System initialization had warnings - continuing...")

# ============================================================================
# EXPORT PUBLIC INTERFACE - COMPLETE SET FOR ALL MODULES
# ============================================================================

__all__ = [
    # Feature config
    'FeatureConfig',
    'FEATURE_CONFIG',
    
    # Idle strategy
    'IDLE_STRATEGY',
    'IDLE_FEATURES_ENABLED',
    'IDLE_SAMPLES_MULTIPLIER',
    'IDLE_CONFIDENCE_THRESHOLD',
    'ACTIVE_CONFIDENCE_THRESHOLD',
    
    # Paths
    'PROJECT_ROOT',
    'DATA_DIR',
    'MODELS_DIR',
    'EXPORTS_DIR',
    'LOGS_DIR',
    'CONFIGS_DIR',
    'FEATURES_DIR',
    'NORMALIZER_DIR',
    
    # Model files
    'MODEL_CHECKPOINT',
    'FINAL_MODEL',
    'TFLITE_MODEL',
    'QUANTIZED_TFLITE_MODEL',
    'MODEL_METADATA_FILE',
    'FEATURE_CONFIG_FILE',
    'NORMALIZER_FILE',
    'CONTRACT_EXPORT',
    
    # Feature dimension
    'FEATURE_DIMENSION',
    'set_feature_dimension',
    
    # Training config
    'SEQUENCE_LENGTH',
    'MIN_SAMPLES_PER_GESTURE',
    'TARGET_SAMPLES_PER_GESTURE',
    'BATCH_SIZE',
    'EPOCHS',
    'LEARNING_RATE',
    'VALIDATION_SPLIT',
    'TEST_SPLIT',
    'USE_AUGMENTATION',
    'AUGMENTATION_FACTOR',
    
    # Inference config
    'INFERENCE_CONFIDENCE_THRESHOLD',
    'PREDICTION_HISTORY_LENGTH',
    'MIN_FRAMES_FOR_INFERENCE',
    'CAMERA_INDEX',
    'CAMERA_WIDTH',
    'CAMERA_HEIGHT',
    'TARGET_FPS',
    
    # Gesture config
    'GESTURE_CLASSES',
    'NUM_GESTURES',
    
    # PATH VALIDATION FUNCTIONS - ADDED FOR main.py
    'validate_paths',
    'check_model_exists',
    'get_model_path',
    'is_production',
    
    # Metadata functions
    'save_model_metadata',
    'load_model_metadata',
    'validate_metadata_consistency',
    
    # Helper functions
    'get_timestamp',
    'get_required_idle_samples',
    'get_confidence_threshold',
    'get_gesture_index',
    'get_gesture_name',
    
    # System initialization
    'initialize_system',
]