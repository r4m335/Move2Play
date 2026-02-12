"""
Main training script for gesture recognition with idle class handling.
Prevents false positives by ensuring idle has sufficient samples.

CRITICAL FIXES APPLIED:
1. ✅ Feature dimension is determined BEFORE model building
2. ✅ Model metadata is saved with feature dimension and config hash
3. ✅ Feature engineering configuration is consistent across training/inference
4. ✅ Global FEATURE_DIMENSION properly updated via config.set_feature_dimension()
5. ✅ Normalizer saved with hash-based filename for inference consistency
6. ✅ No variable shadowing - imports are properly referenced
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple, Dict, Any, List, Optional
import json
from pathlib import Path
import tensorflow as tf

# Import from config - UPDATED WITH FEATURE CONFIG
from config import (
    GESTURE_CLASSES, NUM_GESTURES, SEQUENCE_LENGTH,
    FINAL_MODEL, TFLITE_MODEL, MIN_SAMPLES_PER_GESTURE,
    get_required_idle_samples, IDLE_SAMPLES_MULTIPLIER,
    FEATURE_CONFIG, FEATURE_DIMENSION, save_model_metadata,
    MODEL_METADATA_FILE, MODEL_CHECKPOINT,
    set_feature_dimension,
    get_gesture_index,
    NORMALIZER_DIR,  # CRITICAL: Import normalizer directory
)

# Import feature engineering - FIXED VERSION
try:
    from feature_engineer import FeatureEngineer, FeatureNormalizer, ModelMetadata
    FEATURE_ENGINEER_AVAILABLE = True
except ImportError:
    print("⚠️  FeatureEngineer not available. Using raw landmarks.")
    FEATURE_ENGINEER_AVAILABLE = False

from data_pipeline import GestureDataPipeline
from gesture_model import GestureModel


class GestureTrainer:
    """
    Handles training with feature engineering integration.
    Ensures consistent feature dimensions between training and inference.
    """
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.feature_normalizer: Optional[FeatureNormalizer] = None
        self.feature_dim: Optional[int] = None
        self.normalizer_path: Optional[Path] = None  # Store hash-based path
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        print("✅ GestureTrainer initialized")
        print(f"   Using feature engineering: {FEATURE_ENGINEER_AVAILABLE}")
    
    def setup_feature_engineering(self) -> bool:
        """
        Setup feature engineering pipeline.
        This must be done BEFORE loading/processing data.
        
        CRITICAL FIX: 
        - Uses config.set_feature_dimension() to update global FEATURE_DIMENSION
        - Does NOT shadow the imported name
        
        Returns:
            True if setup successful
        """
        if not FEATURE_ENGINEER_AVAILABLE:
            print("⚠️  Feature engineering not available, using raw landmarks")
            return True
        
        try:
            # Create feature engineer with config from config.py
            self.feature_engineer = FeatureEngineer(FEATURE_CONFIG)
            
            # Get feature dimension (CRITICAL)
            self.feature_dim = self.feature_engineer.get_feature_dimension()
            
            # CRITICAL FIX: Update global feature dimension using the setter function
            set_feature_dimension(self.feature_dim)
            
            print(f"✅ Feature engineering setup complete")
            print(f"   Feature dimension: {self.feature_dim}")
            print(f"   Config hash: {FEATURE_CONFIG.get_hash()}")
            print(f"   Global FEATURE_DIMENSION updated: {FEATURE_DIMENSION}")
            
            # Create normalizer
            self.feature_normalizer = FeatureNormalizer()
            
            return True
            
        except Exception as e:
            print(f"❌ Feature engineering setup failed: {e}")
            return False
    
    def extract_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract features from landmarks using feature engineer.
        Falls back to raw landmarks if feature engineering not available.
        
        Args:
            landmarks: Raw landmarks of shape (N, SEQUENCE_LENGTH, 33, 4)
            
        Returns:
            Features of shape (N, SEQUENCE_LENGTH, feature_dim)
        """
        if self.feature_engineer is None:
            # Fallback: use raw landmarks (flattened)
            N, T, L, D = landmarks.shape
            return landmarks.reshape(N, T, -1)
        
        # Extract features for each sequence
        N = landmarks.shape[0]
        features_list = []
        
        print(f"   Extracting features for {N} sequences...")
        for i in range(N):
            # Extract features for each sequence
            sequence_features = self.feature_engineer.extract_features(landmarks[i])
            
            # Validate feature dimension
            if sequence_features.shape[1] != self.feature_dim:
                raise ValueError(
                    f"Feature dimension mismatch at sequence {i}: "
                    f"expected {self.feature_dim}, got {sequence_features.shape[1]}"
                )
            
            features_list.append(sequence_features)
        
        features = np.array(features_list, dtype=np.float32)
        print(f"   ✓ Features shape: {features.shape}")
        
        return features
    
    def normalize_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Normalize features using feature normalizer.
        
        Args:
            features: Features to normalize
            fit: If True, fit the normalizer on these features
            
        Returns:
            Normalized features
        """
        if self.feature_normalizer is None:
            return features
        
        try:
            if fit:
                print("   Fitting feature normalizer...")
                self.feature_normalizer.fit(features, FEATURE_CONFIG)
            
            print("   Normalizing features...")
            normalized = self.feature_normalizer.transform(features)
            print(f"   ✓ Normalized features shape: {normalized.shape}")
            
            return normalized
            
        except Exception as e:
            print(f"❌ Feature normalization failed: {e}")
            return features
    
    def save_normalizer(self):
        """
        Save normalizer with hash-based filename for inference consistency.
        
        CRITICAL FIX: Inference expects normalizer_<config_hash>.npz, not feature_normalizer.npz
        """
        if self.feature_normalizer is None or not self.feature_normalizer.is_fitted:
            print("⚠️  Normalizer not fitted, skipping save")
            return
        
        try:
            # Get config hash for filename
            config_hash = FEATURE_CONFIG.get_hash()
            
            # CRITICAL FIX: Use hash-based filename for inference consistency
            normalizer_filename = f"normalizer_{config_hash}.npz"
            self.normalizer_path = NORMALIZER_DIR / normalizer_filename
            
            # Ensure directory exists
            self.normalizer_path.parent.mkdir(exist_ok=True, parents=True)
            
            # Save normalizer
            self.feature_normalizer.save(str(self.normalizer_path))
            print(f"   ✓ Normalizer saved to: {self.normalizer_path}")
            print(f"   ✓ Config hash: {config_hash}")
            
            # Also save a symlink/latest pointer for convenience (optional)
            latest_path = NORMALIZER_DIR / "normalizer_latest.npz"
            try:
                if latest_path.exists():
                    latest_path.unlink()
                # Create symlink if supported, otherwise copy
                try:
                    latest_path.symlink_to(self.normalizer_path)
                    print(f"   ✓ Created symlink: {latest_path} -> {normalizer_filename}")
                except (OSError, AttributeError):
                    # Symlink not supported, copy the file
                    import shutil
                    shutil.copy2(self.normalizer_path, latest_path)
                    print(f"   ✓ Copied to latest: {latest_path}")
            except Exception as e:
                print(f"   ⚠️  Could not create latest pointer: {e}")
            
        except Exception as e:
            print(f"❌ Failed to save normalizer: {e}")
    
    def load_and_preprocess_data(self) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Load data, extract features, and prepare datasets.
        
        Returns:
            ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        print("\n1️⃣  Loading and preprocessing data...")
        
        # Step 1: Load raw data
        pipeline = GestureDataPipeline(
            data_dir=self.args.data_dir,
            sequence_length=self.args.sequence_length,
            random_seed=42
        )
        
        # Get raw landmarks
        raw_data = pipeline.load_all_sequences()
        if raw_data is None or len(raw_data[0]) == 0:
            raise ValueError("No data found. Run data collection first.")
        
        # Separate into train/val/test
        (X_raw_train, y_train), (X_raw_val, y_val), (X_raw_test, y_test) = pipeline.prepare_datasets(
            test_size=0.2,
            val_size=0.1,
            stratify=True,
            save_splits=True
        )
        
        print(f"   ✓ Raw training samples: {len(X_raw_train)}")
        print(f"   ✓ Raw validation samples: {len(X_raw_val)}")
        print(f"   ✓ Raw test samples: {len(X_raw_test)}")
        
        # Step 2: Extract features
        print("\n2️⃣  Extracting features...")
        X_train = self.extract_features(X_raw_train)
        X_val = self.extract_features(X_raw_val)
        X_test = self.extract_features(X_raw_test)
        
        # Step 3: Normalize features (fit on training data only)
        print("\n3️⃣  Normalizing features...")
        X_train_norm = self.normalize_features(X_train, fit=True)
        X_val_norm = self.normalize_features(X_val, fit=False)
        X_test_norm = self.normalize_features(X_test, fit=False)
        
        # CRITICAL FIX: Save normalizer with hash-based filename
        self.save_normalizer()
        
        return (X_train_norm, y_train), (X_val_norm, y_val), (X_test_norm, y_test)
    
    def analyze_dataset_distribution(self,
                                   pipeline: GestureDataPipeline, 
                                   X_train: np.ndarray, 
                                   y_train: np.ndarray,
                                   X_val: np.ndarray, 
                                   y_val: np.ndarray,
                                   X_test: np.ndarray,
                                   y_test: np.ndarray) -> Dict[str, Any]:
        """
        Analyze and print dataset distribution with emphasis on idle class.
        """
        # Convert one-hot to integer
        y_train_int = np.argmax(y_train, axis=1)
        y_val_int = np.argmax(y_val, axis=1)
        y_test_int = np.argmax(y_test, axis=1)
        
        # Find idle index
        idle_idx = GESTURE_CLASSES.index("idle")
        
        # Count samples per class
        train_counts = np.bincount(y_train_int, minlength=NUM_GESTURES)
        val_counts = np.bincount(y_val_int, minlength=NUM_GESTURES)
        test_counts = np.bincount(y_test_int, minlength=NUM_GESTURES)
        
        # Find max among non-idle classes
        non_idle_indices = [i for i in range(NUM_GESTURES) if i != idle_idx]
        max_non_idle_train = max(train_counts[non_idle_indices]) if non_idle_indices else 0
        required_idle = int(max_non_idle_train * IDLE_SAMPLES_MULTIPLIER)
        
        print("\n" + "=" * 60)
        print("DATASET DISTRIBUTION ANALYSIS")
        print("=" * 60)
        print(f"{'Gesture':<15} {'Train':<8} {'Val':<8} {'Test':<8} {'Status':<10}")
        print("-" * 60)
        
        for i, gesture in enumerate(GESTURE_CLASSES):
            train_count = train_counts[i]
            val_count = val_counts[i]
            test_count = test_counts[i]
            
            # Check if meets minimum requirements
            if gesture == "idle":
                status = "✅ GOOD" if train_count >= required_idle else "⚠️  NEEDS MORE"
                marker = "⭐ "
            else:
                status = "✅ GOOD" if train_count >= MIN_SAMPLES_PER_GESTURE else "⚠️  NEEDS MORE"
                marker = ""
            
            print(f"{marker}{gesture:<13} {train_count:<8} {val_count:<8} {test_count:<8} {status:<10}")
        
        print("-" * 60)
        
        # Idle specific analysis
        idle_train_count = train_counts[idle_idx]
        print(f"\n📊 IDLE CLASS ANALYSIS:")
        print(f"   Idle samples: {idle_train_count}")
        print(f"   Max active gesture: {max_non_idle_train}")
        print(f"   Target ratio: {IDLE_SAMPLES_MULTIPLIER:.1f}x")
        print(f"   Current ratio: {idle_train_count/max_non_idle_train:.2f}x" if max_non_idle_train > 0 else "   No active gestures")
        
        if idle_train_count < required_idle:
            print(f"⚠️  WARNING: Idle needs {required_idle - idle_train_count} more samples!")
            print(f"   Run: python main.py collect_idle")
        
        return {
            'train_counts': train_counts.tolist(),
            'val_counts': val_counts.tolist(),
            'test_counts': test_counts.tolist(),
            'idle_idx': idle_idx,
            'required_idle': required_idle,
            'has_enough_idle': idle_train_count >= required_idle,
            'idle_count': idle_train_count,
            'max_active_count': max_non_idle_train
        }
    
    def compute_class_weights(self, 
                            y_train_int: np.ndarray, 
                            idle_idx: int,
                            analysis: Dict[str, Any]) -> Dict[int, float]:
        """
        Compute class weights with special handling for idle class.
        """
        # Basic balanced weights
        unique_classes = np.unique(y_train_int)
        weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y_train_int
        )
        
        # Create weight dictionary
        weight_dict = {int(cls): float(w) for cls, w in zip(unique_classes, weights)}
        
        # If idle is under-represented, increase its weight
        if not analysis['has_enough_idle']:
            idle_count = analysis['idle_count']
            max_active = analysis['max_active_count']
            
            if idle_count > 0 and max_active > 0:
                # Increase weight proportionally to the shortage
                shortage_ratio = max_active / idle_count
                adjustment = min(shortage_ratio, 3.0)  # Cap at 3x to avoid overfitting
                weight_dict[idle_idx] *= adjustment
                
                print(f"\n⚖️  Adjusted idle class weight by {adjustment:.2f}x")
                print(f"   New idle weight: {weight_dict[idle_idx]:.3f}")
        
        return weight_dict
    
    def select_architecture(self,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_val: np.ndarray,
                           y_val: np.ndarray,
                           dataset_size: int) -> str:
        """
        Select the best model architecture based on dataset characteristics.
        """
        print("\n" + "=" * 60)
        print("ARCHITECTURE SELECTION")
        print("=" * 60)
        
        # Simple rule-based selection
        if dataset_size < 1000:
            print("   Small dataset (<1000 samples) -> Using CNN (simpler)")
            return "cnn"
        elif dataset_size < 5000:
            print("   Medium dataset -> Using CNN+LSTM (good temporal modeling)")
            return "cnn_lstm"
        else:
            print("   Large dataset -> Using Attention (best temporal patterns)")
            return "attention"
    
    def save_training_metadata(self,
                              model: 'GestureModel',
                              history: tf.keras.callbacks.History,
                              metrics: Dict[str, Any],
                              analysis: Dict[str, Any]):
        """
        Save comprehensive training metadata for inference validation.
        """
        print("\n💾 Saving training metadata...")
        
        # CRITICAL: Include normalizer hash in metadata for inference
        config_hash = FEATURE_CONFIG.get_hash()
        normalizer_filename = f"normalizer_{config_hash}.npz"
        
        # Additional metadata
        additional_data = {
            'training_samples': len(history.history.get('loss', [])),
            'final_train_loss': float(history.history.get('loss', [-1])[-1]),
            'final_val_loss': float(history.history.get('val_loss', [-1])[-1]),
            'final_train_accuracy': float(history.history.get('accuracy', [-1])[-1]),
            'final_val_accuracy': float(history.history.get('val_accuracy', [-1])[-1]),
            'test_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                            for k, v in metrics.items()},
            'dataset_analysis': analysis,
            'architecture': model.architecture,
            'input_shape': model.input_shape,
            'sequence_length': SEQUENCE_LENGTH,
            'idle_class_index': GESTURE_CLASSES.index('idle'),
            'has_feature_engineering': FEATURE_ENGINEER_AVAILABLE,
            'feature_dimension': FEATURE_DIMENSION,  # CRITICAL: This is now properly set
            'config_hash': config_hash,
            'normalizer_file': normalizer_filename,  # CRITICAL: Store for inference
            'normalizer_path': str(self.normalizer_path) if self.normalizer_path else None,
        }
        
        # Save metadata
        save_model_metadata(additional_data)
        
        # Also save feature engineer config separately
        if self.feature_engineer:
            feature_config_path = Path("models/features/feature_config.json")
            feature_config_path.parent.mkdir(exist_ok=True, parents=True)
            
            with open(feature_config_path, 'w') as f:
                json.dump(FEATURE_CONFIG.to_dict(), f, indent=2)
            
            print(f"   ✓ Feature config saved to: {feature_config_path}")
        
        print("✅ Training metadata saved")
        print(f"   ✓ Config hash: {config_hash}")
        print(f"   ✓ Normalizer: {normalizer_filename}")


def train_model(args: argparse.Namespace) -> Tuple['GestureModel', Dict[str, Any]]:
    """
    Main training function with feature engineering integration.
    
    Returns:
        Tuple of (trained_model, training_metrics)
    """
    print("=" * 60)
    print("GESTURE RECOGNITION MODEL TRAINING")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Using feature engineering: {FEATURE_ENGINEER_AVAILABLE}")
    print(f"Feature dimension (initial): {FEATURE_DIMENSION}")
    print("=" * 60)
    
    # Create trainer
    trainer = GestureTrainer(args)
    
    # Step 1: Setup feature engineering (MUST BE FIRST)
    print("\n🔧 Setting up feature engineering...")
    if not trainer.setup_feature_engineering():
        print("⚠️  Feature engineering setup failed, continuing with raw data")
    
    # Step 2: Load and preprocess data with features
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = trainer.load_and_preprocess_data()
    
    # Step 3: Analyze distribution
    print("\n📊 Analyzing dataset distribution...")
    from data_pipeline import GestureDataPipeline
    pipeline = GestureDataPipeline(data_dir=args.data_dir, sequence_length=args.sequence_length)
    
    y_train_int = np.argmax(y_train, axis=1)
    analysis = trainer.analyze_dataset_distribution(
        pipeline, X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # Step 4: Apply data augmentation if requested
    if args.augment and X_train.shape[0] > 0:
        print("\n🎨 Applying data augmentation...")
        from data_pipeline import TemporalAugmenter
        
        # Only augment training data
        X_train_aug, y_train_aug = TemporalAugmenter.augment_batch(
            X_train, y_train_int,
            augment_factor=2,
            methods=['noise', 'scaling', 'crop'],
            gesture_names=GESTURE_CLASSES
        )
        
        # Convert back to one-hot
        from tensorflow.keras.utils import to_categorical
        y_train = to_categorical(y_train_aug, NUM_GESTURES)
        X_train = X_train_aug
        
        print(f"   ✓ Augmented training samples: {len(X_train)}")
    
    # Step 5: Compute class weights
    print("\n⚖️  Computing class weights...")
    weight_dict = trainer.compute_class_weights(y_train_int, analysis['idle_idx'], analysis)
    
    print("\n   Final class weights:")
    for i, gesture in enumerate(GESTURE_CLASSES):
        weight = weight_dict.get(i, 1.0)
        marker = "⭐ " if gesture == "idle" else ""
        if gesture in ['run', 'jogging']:
            marker = "🏃 " if gesture == 'run' else "🏃‍♂️ "
        print(f"   {marker}{gesture:<15}: {weight:.3f}")
    
    # Step 6: Select and build model
    print("\n🏗️  Building model architecture...")
    
    # Determine feature dimension
    if trainer.feature_dim:
        feature_dim = trainer.feature_dim
    else:
        # Fallback: use flattened landmarks
        feature_dim = X_train.shape[2]
    
    print(f"   Input shape: ({SEQUENCE_LENGTH}, {feature_dim})")
    print(f"   Global FEATURE_DIMENSION: {FEATURE_DIMENSION}")
    
    architecture = trainer.select_architecture(
        X_train, y_train, X_val, y_val, len(X_train)
    )
    
    model = GestureModel(
        input_shape=(SEQUENCE_LENGTH, feature_dim),
        num_classes=NUM_GESTURES,
        architecture=architecture,
        dataset_size=len(X_train)
    )
    
    # Compile model
    model.compile(learning_rate=args.learning_rate)
    
    # Print model summary
    print("\n   Model Architecture:")
    model.summary()
    
    # Step 7: Train model
    print(f"\n🚀 Training model for {args.epochs} epochs...")
    print("   (Press Ctrl+C to stop early and save best model)")
    
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weights=weight_dict,
        checkpoint_path=MODEL_CHECKPOINT
    )
    
    # Step 8: Evaluate on test set
    print("\n🧪 Evaluating on test set...")
    metrics = model.evaluate(X_test, y_test)
    
    # Step 9: Save training metadata
    trainer.save_training_metadata(model, history, metrics, analysis)
    
    # Step 10: Print results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"📊 Overall Metrics:")
    print(f"   Test Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"   Test Precision: {metrics.get('precision', 0):.4f}")
    print(f"   Test Recall:    {metrics.get('recall', 0):.4f}")
    print(f"   Test F1-Score:  {metrics.get('f1_score', 0):.4f}")
    
    # Per-class accuracy
    per_class_acc = metrics.get('per_class_accuracy', [])
    if per_class_acc:
        print(f"\n🎯 Per-Class Accuracy (Idle is critical!):")
        for i, gesture in enumerate(GESTURE_CLASSES):
            if i < len(per_class_acc):
                acc = per_class_acc[i]
                status = "✅" if acc > 0.8 else "⚠️ " if acc > 0.6 else "❌"
                marker = "⭐ " if gesture == "idle" else ""
                if gesture in ['run', 'jogging']:
                    marker = "🏃 " if gesture == 'run' else "🏃‍♂️ "
                print(f"   {marker}{gesture:<15}: {acc:.3f} {status}")
    
    # Step 11: Save final model
    print(f"\n💾 Saving final model...")
    model.save_model(FINAL_MODEL)
    
    # Step 12: Convert to TFLite
    print(f"\n📱 Converting to TFLite...")
    model.convert_to_tflite(
        output_path=TFLITE_MODEL,
        quantize=True,
        optimization_level=2
    )
    
    # Step 13: Plot training history
    print(f"\n📈 Generating training plots...")
    model.plot_training_history(save_path=FINAL_MODEL.parent / "training_history.png")
    
    # Step 14: Show sample predictions
    print(f"\n📋 Sample predictions from test set:")
    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        sample = X_test[idx:idx+1]
        true_idx = np.argmax(y_test[idx])
        true_class = GESTURE_CLASSES[true_idx]
        
        pred_idx, confidence, probs = model.predict_sequence(sample[0])
        pred_class = GESTURE_CLASSES[pred_idx]
        
        # Format confidence bars
        conf_bar = "█" * int(confidence * 10)
        
        print(f"   Sample {i+1}:")
        print(f"     True: {true_class:<15} Predicted: {pred_class:<15}")
        print(f"     Confidence: {confidence:.3f} [{conf_bar:<10}]")
        
        # Show top 3 predictions if not correct
        if pred_idx != true_idx:
            top_indices = np.argsort(probs)[-3:][::-1]
            print(f"     Top predictions:")
            for j, top_idx in enumerate(top_indices):
                print(f"       {j+1}. {GESTURE_CLASSES[top_idx]:<15} {probs[top_idx]:.3f}")
        print()
    
    return model, metrics


def validate_feature_consistency():
    """
    Validate that feature engineering is consistent across the system.
    This should be called before training.
    """
    print("\n🔍 Validating feature engineering consistency...")
    
    # Check that feature config has idle features enabled
    if not FEATURE_CONFIG.compute_idle_features:
        print("❌ CRITICAL: FEATURE_CONFIG.compute_idle_features is False!")
        print("   This will cause dimension mismatches between training and inference.")
        print("   Set compute_idle_features=True in config.py")
        return False
    
    print("✅ Feature configuration validated")
    print(f"   compute_idle_features: {FEATURE_CONFIG.compute_idle_features}")
    print(f"   config_hash: {FEATURE_CONFIG.get_hash()}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Train gesture recognition model with feature engineering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_gesture_model.py                      # Train with defaults
  python train_gesture_model.py --epochs 50          # Train for 50 epochs
  python train_gesture_model.py --no_augment         # Disable augmentation
  python train_gesture_model.py --data_dir ./my_data # Custom data directory
        """
    )
    
    parser.add_argument('--data_dir', 
                       default='gesture_dataset',
                       help='Directory containing gesture data')
    parser.add_argument('--epochs', 
                       type=int, 
                       default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', 
                       type=int, 
                       default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', 
                       type=float, 
                       default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--augment', 
                       action='store_true', 
                       default=True,
                       help='Apply data augmentation (default: True)')
    parser.add_argument('--no_augment', 
                       action='store_false', 
                       dest='augment',
                       help='Disable data augmentation')
    parser.add_argument('--sequence_length', 
                       type=int, 
                       default=30,
                       help='Sequence length for temporal data')
    
    args = parser.parse_args()
    
    try:
        # Validate feature consistency before training
        if not validate_feature_consistency():
            return
        
        # Check if we should retrain or continue training
        if MODEL_CHECKPOINT.exists():
            print(f"\n⚠️  Found existing model checkpoint: {MODEL_CHECKPOINT}")
            response = input("Continue training from checkpoint? (y/n): ")
            if response.lower() != 'y':
                print("Starting fresh training...")
        
        # Train model
        model, metrics = train_model(args)
        
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Model saved to: {FINAL_MODEL}")
        print(f"TFLite model saved to: {TFLITE_MODEL}")
        print(f"Test accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"Feature dimension: {FEATURE_DIMENSION}")
        print(f"Config hash: {FEATURE_CONFIG.get_hash()}")
        
        # Verify normalizer was saved with hash-based name
        config_hash = FEATURE_CONFIG.get_hash()
        normalizer_path = NORMALIZER_DIR / f"normalizer_{config_hash}.npz"
        if normalizer_path.exists():
            print(f"✅ Normalizer saved with hash: {normalizer_path.name}")
        else:
            print(f"⚠️  Normalizer not found at expected path: {normalizer_path}")
        
        # Give recommendations based on results
        per_class_acc = metrics.get('per_class_accuracy', [])
        if per_class_acc:
            idle_idx = GESTURE_CLASSES.index("idle")
            idle_acc = per_class_acc[idle_idx] if idle_idx < len(per_class_acc) else 0
            
            if idle_acc < 0.8:
                print(f"\n⚠️  RECOMMENDATION: Idle accuracy is low ({idle_acc:.3f})")
                print(f"   Collect more idle data: python main.py collect_idle")
            
            # Check for any class with very low accuracy
            for i, acc in enumerate(per_class_acc):
                if acc < 0.6 and i != idle_idx:
                    print(f"\n⚠️  RECOMMENDATION: {GESTURE_CLASSES[i]} accuracy is low ({acc:.3f})")
                    print(f"   Collect more data: python main.py collect --gestures {GESTURE_CLASSES[i]}")
        
        # Verify metadata was saved
        if MODEL_METADATA_FILE.exists():
            print(f"\n✅ Model metadata saved for inference validation")
            print(f"   Feature dimension in metadata: {FEATURE_DIMENSION}")
            print(f"   Normalizer file: normalizer_{config_hash}.npz")
            print(f"   Run: python main.py demo  # For real-time testing")
        else:
            print(f"\n⚠️  Warning: Model metadata not saved")
            print(f"   Inference may fail due to configuration mismatches")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        print("The best model so far has been saved.")
        
        # Try to save metadata even if interrupted
        try:
            if FEATURE_DIMENSION:
                print(f"\n💾 Saving partial metadata...")
                config_hash = FEATURE_CONFIG.get_hash()
                save_model_metadata({
                    'status': 'interrupted',
                    'feature_dimension': FEATURE_DIMENSION,
                    'config_hash': config_hash,
                    'normalizer_file': f"normalizer_{config_hash}.npz"
                })
        except Exception as e:
            print(f"   Failed to save metadata: {e}")
            
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()