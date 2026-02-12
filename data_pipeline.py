"""
Complete data pipeline with proper temporal augmentation and reproducible splits.

CRITICAL FIXES APPLIED:
1. balance_idle_class() uses consistent feature definitions with inference
2. Feature engineering is properly integrated throughout
3. Idle class handling is consistent with config.py (explicit features strategy)
4. Added feature dimension validation
5. Added configuration hash propagation
6. FIXED: Temporal augmentation now EXCLUDES idle and run gestures
   - Aggressive time warp disabled for periodic gestures
   - Random crop + resize disabled for idle and run
   - Scaling limited to biomechanically realistic ranges for all gestures
"""

import os
import json
import numpy as np
from glob import glob
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import hashlib

# Import from config - SINGLE SOURCE OF TRUTH
from config import (
    DATA_DIR, SEQUENCE_LENGTH, GESTURE_CLASSES, FEATURE_CONFIG,
    MIN_SAMPLES_PER_GESTURE, BATCH_SIZE, 
    USE_AUGMENTATION, AUGMENTATION_FACTOR,
    IDLE_SAMPLES_MULTIPLIER, get_required_idle_samples,
    IDLE_CONFIDENCE_THRESHOLD, ACTIVE_CONFIDENCE_THRESHOLD,
    IDLE_STRATEGY, IDLE_FEATURES_ENABLED,
    FEATURE_DIMENSION, NORMALIZER_DIR, FEATURE_CONFIG_FILE
)

# Import feature engineering
try:
    from feature_engineer import FeatureEngineer, FeatureNormalizer
    FEATURE_ENGINEER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  FeatureEngineer not available: {e}")
    print("   Using raw landmarks (not recommended)")
    FEATURE_ENGINEER_AVAILABLE = False

# ============================================================================
# TEMPORAL AUGMENTATION - FIXED FOR PERIODIC GESTURES
# ============================================================================

class TemporalAugmenter:
    """
    Temporal augmentation techniques for time-series gesture data.
    
    CRITICAL FIX: 
    - Aggressive augmentations are NOT applied to idle and run gestures
    - These are periodic gestures where temporal distortions destroy meaning
    - Scaling limited to biomechanically realistic ranges (0.95-1.05 for sensitive gestures)
    """
    
    # Gesture indices that should NOT receive aggressive augmentation
    # These are periodic gestures where temporal patterns are critical
    SENSITIVE_GESTURES = ['idle', 'run', 'jogging']  # Add any periodic gestures here
    
    @staticmethod
    def add_gaussian_noise(X: np.ndarray, 
                          noise_level: float = 0.01,
                          preserve_periodicity: bool = False) -> np.ndarray:
        """Add Gaussian noise to sequences."""
        if len(X) == 0:
            return X
            
        # For periodic gestures, use lower noise level to preserve pattern
        if preserve_periodicity:
            noise_level = min(noise_level, 0.005)  # Very subtle noise
            
        noise = np.random.normal(0, noise_level, X.shape)
        return X + noise
    
    @staticmethod
    def time_warp(X: np.ndarray, 
                 warp_factor: float = 0.1,
                 preserve_periodicity: bool = False) -> np.ndarray:
        """
        Apply random time warping.
        
        FIXED: For periodic gestures (idle, run), disable aggressive warping
        """
        if len(X) == 0:
            return X
            
        batch_size, seq_len, n_features = X.shape
        X_warped = np.zeros_like(X)
        
        # For periodic gestures, use minimal or zero warping
        if preserve_periodicity:
            if warp_factor > 0.05:
                warp_factor = 0.05  # Cap at 5% for periodic gestures
            # Option to completely disable time warp for idle
            if warp_factor <= 0.01:
                return X  # Skip warping entirely
        
        for i in range(batch_size):
            # Original time indices
            t = np.linspace(0, 1, seq_len)
            
            # Create warping curve - smoother for periodic gestures
            if preserve_periodicity:
                # Use fewer control points for smoother warping
                warp_points = np.random.uniform(-warp_factor, warp_factor, size=3)
            else:
                warp_points = np.random.uniform(-warp_factor, warp_factor, size=5)
                
            warp_points = np.cumsum(warp_points)
            
            # Normalize
            warp_points = warp_points - warp_points.min()
            if warp_points.max() > 0:
                warp_points = warp_points / warp_points.max()
            
            # Interpolate to sequence length
            if preserve_periodicity:
                # Use fewer interpolation points for smoother curve
                control_points = np.linspace(0, 4, len(warp_points))
                interp_points = np.linspace(0, 4, seq_len)
            else:
                control_points = np.arange(len(warp_points))
                interp_points = np.linspace(0, len(warp_points)-1, seq_len)
                
            warp_curve = np.interp(
                interp_points,
                control_points,
                warp_points
            )
            
            # Apply warping
            warped_time = t + warp_curve * warp_factor
            warped_time = np.clip(warped_time, 0, 1)
            
            # Interpolate features
            for f in range(n_features):
                X_warped[i, :, f] = np.interp(
                    warped_time * (seq_len - 1),
                    np.arange(seq_len),
                    X[i, :, f]
                )
        
        return X_warped
    
    @staticmethod
    def temporal_scaling(X: np.ndarray, 
                        scale_range: Tuple[float, float] = (0.9, 1.1),
                        preserve_periodicity: bool = False) -> np.ndarray:
        """
        Scale sequences in time dimension.
        
        FIXED: For periodic gestures, use biomechanically realistic scaling
        """
        if len(X) == 0:
            return X
            
        batch_size, seq_len, n_features = X.shape
        X_scaled = np.zeros_like(X)
        
        for i in range(batch_size):
            # For periodic gestures, restrict scaling range
            if preserve_periodicity:
                # Realistic jogging cadence variation: ±5%
                scale = np.random.uniform(0.95, 1.05)
            else:
                scale = np.random.uniform(scale_range[0], scale_range[1])
            
            if scale < 1.0:
                # Downsample
                new_len = max(int(seq_len * scale), 1)
                indices = np.linspace(0, seq_len - 1, new_len).astype(int)
                scaled_seq = X[i, indices, :]
                
                # Pad if necessary
                pad_len = seq_len - new_len
                if pad_len > 0:
                    # Use edge padding to preserve temporal patterns
                    scaled_seq = np.pad(scaled_seq, ((0, pad_len), (0, 0)), mode='edge')
                X_scaled[i] = scaled_seq
            else:
                # Upsample
                new_len = int(seq_len * scale)
                t_original = np.arange(seq_len)
                t_new = np.linspace(0, seq_len - 1, new_len)
                
                for f in range(n_features):
                    X_scaled[i, :, f] = np.interp(
                        t_new[:seq_len],
                        t_original,
                        X[i, :, f]
                    )
        
        return X_scaled
    
    @staticmethod
    def random_crop(X: np.ndarray, 
                   crop_range: Tuple[float, float] = (0.8, 1.0),
                   preserve_periodicity: bool = False) -> np.ndarray:
        """
        Randomly crop a subsequence and resize to original length.
        
        FIXED: For periodic gestures, DISABLE cropping or use minimal crop
        """
        if len(X) == 0:
            return X
            
        # CRITICAL FIX: For periodic gestures, either skip cropping or use minimal crop
        if preserve_periodicity:
            # Skip cropping entirely - it destroys periodicity
            return X
            
        batch_size, seq_len, n_features = X.shape
        X_cropped = np.zeros_like(X)
        
        for i in range(batch_size):
            crop_len = int(seq_len * np.random.uniform(crop_range[0], crop_range[1]))
            crop_len = max(crop_len, 10)
            
            max_start = seq_len - crop_len
            if max_start <= 0:
                X_cropped[i] = X[i]
                continue
                
            start = np.random.randint(0, max_start)
            end = start + crop_len
            cropped = X[i, start:end, :]
            
            # Resize back to original length
            t_cropped = np.arange(crop_len)
            t_original = np.linspace(0, crop_len - 1, seq_len)
            
            for f in range(n_features):
                X_cropped[i, :, f] = np.interp(
                    t_original,
                    t_cropped,
                    cropped[:, f]
                )
        
        return X_cropped
    
    @staticmethod
    def augment_batch(X: np.ndarray, 
                     y: np.ndarray,
                     augment_factor: int = 2,
                     methods: List[str] = None,
                     gesture_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply multiple augmentations to create augmented batch.
        
        CRITICAL FIX: 
        - For idle and run gestures, use minimal augmentation
        - No aggressive time warping or cropping for periodic gestures
        - Scaling limited to biomechanically realistic ranges
        
        Args:
            X: Input sequences
            y: Integer labels
            augment_factor: Number of augmented copies per original
            methods: List of augmentation methods
            gesture_names: List of gesture names corresponding to labels
            
        Returns:
            Augmented X and y
        """
        if len(X) == 0 or augment_factor <= 1:
            return X, y
        
        if methods is None:
            methods = ['noise', 'time_warp', 'scaling', 'crop']
        
        # Map gesture indices to names for identification
        if gesture_names is None:
            # Try to infer from config
            gesture_names = GESTURE_CLASSES
        
        # Identify sensitive gesture indices (periodic gestures)
        sensitive_indices = []
        for gesture in TemporalAugmenter.SENSITIVE_GESTURES:
            if gesture in gesture_names:
                idx = gesture_names.index(gesture)
                sensitive_indices.append(idx)
                print(f"   🛡️  Protecting '{gesture}' from aggressive augmentation")
        
        X_augmented = []
        y_augmented = []
        
        for i in range(len(X)):
            # Keep original
            X_augmented.append(X[i])
            y_augmented.append(y[i])
            
            # Determine if this is a sensitive gesture
            is_sensitive = y[i] in sensitive_indices
            
            # Create augmented copies
            for _ in range(augment_factor - 1):
                x_aug = X[i].copy()
                
                # For sensitive gestures, use only subtle augmentations
                if is_sensitive:
                    # Only apply minimal noise, no warping or cropping
                    x_aug = TemporalAugmenter.add_gaussian_noise(
                        x_aug[np.newaxis, ...], 
                        noise_level=np.random.uniform(0.001, 0.005),  # Very low noise
                        preserve_periodicity=True
                    )[0]
                    
                    # Optionally apply minimal scaling (biomechanically realistic)
                    if np.random.random() > 0.5:
                        x_aug = TemporalAugmenter.temporal_scaling(
                            x_aug[np.newaxis, ...],
                            scale_range=(0.98, 1.02),  # ±2% max for periodic
                            preserve_periodicity=True
                        )[0]
                    
                else:
                    # For non-sensitive gestures, select random augmentation methods
                    num_methods = np.random.randint(1, len(methods) + 1)
                    aug_methods = np.random.choice(methods, size=num_methods, replace=False)
                    
                    for method in aug_methods:
                        if method == 'noise':
                            x_aug = TemporalAugmenter.add_gaussian_noise(
                                x_aug[np.newaxis, ...], 
                                noise_level=np.random.uniform(0.005, 0.02),
                                preserve_periodicity=False
                            )[0]
                        elif method == 'time_warp':
                            x_aug = TemporalAugmenter.time_warp(
                                x_aug[np.newaxis, ...],
                                warp_factor=np.random.uniform(0.05, 0.15),  # Reduced from 0.2
                                preserve_periodicity=False
                            )[0]
                        elif method == 'scaling':
                            x_aug = TemporalAugmenter.temporal_scaling(
                                x_aug[np.newaxis, ...],
                                scale_range=(0.85, 1.15),  # ±15% for non-periodic
                                preserve_periodicity=False
                            )[0]
                        elif method == 'crop':
                            x_aug = TemporalAugmenter.random_crop(
                                x_aug[np.newaxis, ...],
                                crop_range=(0.8, 1.0),  # Keep at least 80%
                                preserve_periodicity=False
                            )[0]
                
                X_augmented.append(x_aug)
                y_augmented.append(y[i])
        
        X_augmented = np.array(X_augmented, dtype=np.float32)
        y_augmented = np.array(y_augmented, dtype=np.int32)
        
        print(f"   🎭 Augmentation complete:")
        print(f"     Original: {len(X)} samples")
        print(f"     Augmented: {len(X_augmented)} samples")
        print(f"     Protected gestures: {sensitive_indices}")
        
        return X_augmented, y_augmented


# ============================================================================
# MAIN DATA PIPELINE WITH FIXED IDLE CLASS BALANCING
# ============================================================================

class GestureDataPipeline:
    """
    Handles loading, preprocessing, splitting, and augmentation of gesture data.
    Now with consistent idle class handling across training and inference.
    
    IDLE STRATEGY: Explicit features encoding (compute_idle_features=True)
    - Idle is encoded in the feature space with dedicated features
    - Model learns to recognize idle patterns directly
    - Inference uses same feature extraction pipeline
    """
    
    def __init__(self, 
                 data_dir: Path = DATA_DIR, 
                 sequence_length: int = SEQUENCE_LENGTH,
                 random_seed: int = 42,
                 feature_engineer: Optional[FeatureEngineer] = None,
                 config_hash: Optional[str] = None):
        """
        Args:
            data_dir: Directory containing gesture data
            sequence_length: Fixed sequence length for all samples
            random_seed: Random seed for reproducibility
            feature_engineer: Optional pre-configured feature engineer
            config_hash: Hash of feature configuration for validation
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.random_seed = random_seed
        self.config_hash = config_hash or FEATURE_CONFIG.get_hash()
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        
        # Setup feature engineering
        if feature_engineer is not None:
            self.feature_engineer = feature_engineer
        elif FEATURE_ENGINEER_AVAILABLE:
            # Use the same config as config.py for consistency
            self.feature_engineer = FeatureEngineer(FEATURE_CONFIG)
            print(f"✅ FeatureEngineer initialized with config hash: {FEATURE_CONFIG.get_hash()}")
        else:
            self.feature_engineer = None
            print("⚠️  Running without feature engineering (raw landmarks)")
        
        # Gesture mapping
        self.gesture_classes = GESTURE_CLASSES
        self.class_to_idx = {cls: i for i, cls in enumerate(self.gesture_classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        
        # Idle class index
        if 'idle' in self.gesture_classes:
            self.idle_idx = self.class_to_idx['idle']
            print(f"   Idle class detected at index: {self.idle_idx}")
        else:
            self.idle_idx = None
            print(f"   ⚠️  Idle class not found in gesture classes")
        
        # Run/Jogging class index
        self.run_idx = None
        for gesture in ['run', 'jogging']:
            if gesture in self.gesture_classes:
                self.run_idx = self.class_to_idx[gesture]
                print(f"   {gesture.capitalize()} class detected at index: {self.run_idx}")
                break
        
        # Feature dimension
        self.feature_dim = None
        if self.feature_engineer:
            self.feature_dim = self.feature_engineer.get_feature_dimension()
            print(f"   Feature dimension: {self.feature_dim}")
        
        # Normalizer
        self.normalizer = None
        self.normalizer_fitted = False
        
        # Cache for loaded data
        self._cache = {}
        
        # Split indices for reproducibility
        self.split_indices = None
        
        print(f"\n✅ Data pipeline initialized")
        print(f"   Idle strategy: {IDLE_STRATEGY}")
        print(f"   Idle features enabled: {IDLE_FEATURES_ENABLED}")
        print(f"   Data directory: {self.data_dir}")
        print(f"   Sequence length: {sequence_length}")
        print(f"   Random seed: {random_seed}")
        print(f"   Config hash: {self.config_hash}")
    
    def balance_idle_class(self, 
                          X: np.ndarray, 
                          y: np.ndarray,
                          preserve_idle_characteristics: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance idle class using explicit idle features.
        CRITICAL FIX: Uses consistent feature definitions with inference.
        
        Args:
            X: Features array
            y: Labels array (integer)
            preserve_idle_characteristics: Whether to preserve idle-specific features
            
        Returns:
            Balanced X and y arrays
        """
        if self.idle_idx is None:
            print("⚠️  Skipping idle class balancing: idle class not found")
            return X, y
        
        # Count samples per class
        unique_classes, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(unique_classes, counts))
        
        # Get idle count
        idle_count = class_counts.get(self.idle_idx, 0)
        
        # Get maximum count among active gestures (non-idle)
        active_counts = [count for cls, count in class_counts.items() 
                        if cls != self.idle_idx]
        
        if not active_counts:
            print("⚠️  No active gestures found, skipping idle balancing")
            return X, y
        
        max_active_count = max(active_counts)
        required_idle = int(max_active_count * IDLE_SAMPLES_MULTIPLIER)
        
        print(f"\n" + "=" * 60)
        print("IDLE CLASS BALANCING")
        print(f"Strategy: Explicit Features ({IDLE_STRATEGY})")
        print("=" * 60)
        print(f"   Idle samples: {idle_count}")
        print(f"   Max active gesture: {max_active_count}")
        print(f"   Target ratio: {IDLE_SAMPLES_MULTIPLIER:.1f}x")
        print(f"   Required idle: {required_idle}")
        print(f"   Inference idle threshold: {IDLE_CONFIDENCE_THRESHOLD}")
        
        # If idle doesn't have enough samples, apply intelligent balancing
        if idle_count < required_idle:
            needed = required_idle - idle_count
            print(f"   ⚠️  Idle class under-represented, adding {needed} samples...")
            
            # Find idle samples
            idle_mask = y == self.idle_idx
            X_idle = X[idle_mask]
            y_idle = y[idle_mask]
            
            if len(X_idle) == 0:
                print(f"   ❌ No idle samples found! Cannot balance.")
                return X, y
            
            # Calculate "idleness" score based on explicit idle features
            # This ensures we select high-quality idle samples that match inference
            idle_scores = self._calculate_idleness_scores(X_idle)
            
            if idle_scores is not None and len(idle_scores) > 0:
                # Weight selection by quality score
                weights = idle_scores / idle_scores.sum()
                indices = np.random.choice(len(X_idle), needed, replace=True, p=weights)
                print(f"   Weighted selection by idleness score (mean: {idle_scores.mean():.3f})")
            else:
                # Fallback: uniform selection
                indices = np.random.choice(len(X_idle), needed, replace=True)
                print(f"   Uniform random selection (no idle scoring available)")
            
            # Create augmented idle samples with PRESERVED characteristics
            X_duplicates = X_idle[indices].copy()
            
            if preserve_idle_characteristics:
                # Apply MINIMAL augmentations that preserve idle characteristics
                for i in range(len(X_duplicates)):
                    # 1. VERY small noise (idle is stable) - 0.1% of normal augmentation
                    noise = np.random.normal(0, 0.001, X_duplicates[i].shape)
                    X_duplicates[i] += noise
                    
                    # 2. NO time warping - idle should have stable rhythm
                    # 3. NO cropping - would destroy pattern
                    # 4. NO scaling beyond realistic limits
                    if np.random.random() > 0.7:  # Only 30% of samples
                        scale = np.random.uniform(0.995, 1.005)  # ±0.5% max
                        if abs(scale - 1.0) > 0.001:
                            seq_len = X_duplicates[i].shape[0]
                            t_original = np.arange(seq_len)
                            t_scaled = np.linspace(0, seq_len - 1, int(seq_len * scale))
                            
                            for f in range(X_duplicates[i].shape[1]):
                                X_duplicates[i][:seq_len, f] = np.interp(
                                    t_scaled[:seq_len],
                                    t_original,
                                    X_duplicates[i][:seq_len, f]
                                )
            else:
                # Simple replication without augmentation
                pass
            
            # Add to dataset
            X = np.concatenate([X, X_duplicates], axis=0)
            y = np.concatenate([y, y_idle[indices]], axis=0)
            
            # Shuffle
            shuffle_idx = np.random.permutation(len(X))
            X = X[shuffle_idx]
            y = y[shuffle_idx]
            
            # Report results
            new_idle_count = np.sum(y == self.idle_idx)
            new_ratio = new_idle_count / max_active_count
            
            print(f"   ✅ Added {needed} idle samples")
            print(f"   New idle count: {new_idle_count}")
            print(f"   New ratio: {new_ratio:.2f}x (target: {IDLE_SAMPLES_MULTIPLIER:.1f}x)")
            
            if new_ratio < IDLE_SAMPLES_MULTIPLIER:
                print(f"   ⚠️  Still under target - collect more idle data")
        
        else:
            current_ratio = idle_count / max_active_count
            print(f"   ✅ Idle class sufficiently represented")
            print(f"   Current ratio: {current_ratio:.2f}x (target: {IDLE_SAMPLES_MULTIPLIER:.1f}x)")
        
        print("=" * 60)
        return X, y
    
    def _calculate_idleness_scores(self, X_idle: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate idleness scores for quality-based sample selection.
        Uses explicit idle features if available.
        
        Args:
            X_idle: Array of idle samples
            
        Returns:
            Array of idleness scores or None if calculation fails
        """
        if len(X_idle) == 0:
            return None
        
        scores = []
        
        for features in X_idle:
            score = 0.5  # Default
            
            # If we have feature engineer and idle features are computed,
            # we can extract specific idle-related feature indices
            if self.feature_engineer and hasattr(self.feature_engineer, 'idle_feature_indices'):
                try:
                    idle_indices = self.feature_engineer.idle_feature_indices
                    if idle_indices and len(idle_indices) > 0:
                        # Idle features should have specific patterns
                        idle_features = features[:, idle_indices] if len(features.shape) > 1 else features[idle_indices]
                        
                        # Idle should have:
                        # 1. Low variance (stability)
                        if len(idle_features.shape) > 1:
                            variance = np.var(idle_features, axis=0).mean()
                        else:
                            variance = np.var(idle_features)
                        
                        # 2. Values near expected idle ranges
                        # This is heuristic - in practice, learn from data
                        expected_range_score = 1.0 - np.clip(np.abs(idle_features.mean()), 0, 1)
                        
                        # Combined score
                        score = 0.7 * (1.0 - min(variance, 1.0)) + 0.3 * expected_range_score
                except:
                    pass
            
            scores.append(score)
        
        return np.array(scores)
    
    def load_all_sequences(self, 
                          min_samples: int = 1,
                          max_samples: Optional[int] = None,
                          use_features: bool = True,
                          cache: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load all sequences from the dataset directory.
        
        Args:
            min_samples: Minimum samples required per gesture
            max_samples: Maximum samples to load per gesture (for balancing)
            use_features: Whether to extract features or return raw landmarks
            cache: Whether to cache results
            
        Returns:
            Tuple of (features/landmarks, labels, filepaths)
        """
        cache_key = f"all_sequences_{min_samples}_{max_samples}_{use_features}_{self.config_hash}"
        if cache and cache_key in self._cache:
            print("📦 Loading from cache...")
            return self._cache[cache_key]
        
        X = []
        y = []
        filepaths = []
        
        print("\n" + "=" * 60)
        print("LOADING SEQUENCES")
        print("=" * 60)
        
        for gesture in self.gesture_classes:
            gesture_dir = self.data_dir / gesture
            if not gesture_dir.exists():
                print(f"⚠️  Skipping {gesture}: directory not found")
                continue
            
            npy_files = list(gesture_dir.glob("*.npy"))
            
            if max_samples and len(npy_files) > max_samples:
                np.random.seed(self.random_seed)
                npy_files = list(np.random.choice(npy_files, max_samples, replace=False))
            
            if len(npy_files) < min_samples:
                print(f"⚠️  Skipping {gesture}: only {len(npy_files)} samples (< {min_samples})")
                continue
            
            print(f"{gesture:15s}: {len(npy_files):4d} samples")
            
            gesture_X = []
            gesture_files = []
            
            for filepath in npy_files:
                try:
                    landmarks = np.load(filepath)
                    
                    if landmarks.shape[0] < 10:
                        print(f"  ⚠️  Skipping {filepath.name}: too short ({landmarks.shape[0]} frames)")
                        continue
                    
                    processed_landmarks = self._process_sequence_length(landmarks)
                    
                    if use_features and self.feature_engineer:
                        features = self.feature_engineer.extract_features(processed_landmarks)
                        gesture_X.append(features)
                        
                        # Update feature dimension if not set
                        if self.feature_dim is None:
                            self.feature_dim = features.shape[1] if len(features.shape) > 1 else features.shape[0]
                    else:
                        # Use raw landmarks (flattened)
                        flattened = processed_landmarks.reshape(processed_landmarks.shape[0], -1)
                        gesture_X.append(flattened)
                    
                    gesture_files.append(str(filepath))
                    
                except Exception as e:
                    print(f"  ❌ Error loading {filepath.name}: {e}")
            
            if gesture_X:
                X.extend(gesture_X)
                y.extend([self.class_to_idx[gesture]] * len(gesture_X))
                filepaths.extend(gesture_files)
        
        if not X:
            raise ValueError("No valid sequences found in dataset!")
        
        # Convert to numpy arrays
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        
        print(f"\n✅ Loaded {len(X)} sequences total")
        print(f"   Data shape: {X.shape}")
        print(f"   Classes distribution: {np.bincount(y)}")
        
        # Cache results
        if cache:
            self._cache[cache_key] = (X, y, filepaths)
        
        return X, y, filepaths
    
    def _process_sequence_length(self, landmarks: np.ndarray) -> np.ndarray:
        """Process sequence to fixed length."""
        if landmarks.shape[0] > self.sequence_length:
            # Center crop
            start = (landmarks.shape[0] - self.sequence_length) // 2
            return landmarks[start:start + self.sequence_length]
        elif landmarks.shape[0] < self.sequence_length:
            # Pad with zeros (edge padding)
            pad_amount = self.sequence_length - landmarks.shape[0]
            return np.pad(landmarks, ((0, pad_amount), (0, 0), (0, 0)), mode='edge')
        else:
            return landmarks
    
    def prepare_datasets(self, 
                        test_size: float = 0.2, 
                        val_size: float = 0.1,
                        stratify: bool = True,
                        save_splits: bool = True,
                        balance_idle: bool = True,
                        use_features: bool = True) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Split data into train, validation, and test sets with idle balancing.
        
        Returns:
            Tuples of (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        # Load data
        X, y, filepaths = self.load_all_sequences(use_features=use_features)
        
        # Get feature dimension
        feature_dim = X.shape[2] if len(X.shape) > 2 else X.shape[1]
        print(f"\n📊 Feature dimension: {feature_dim}")
        
        # Balance idle class if requested
        if balance_idle and self.idle_idx is not None:
            print("\n" + "=" * 60)
            print("APPLYING IDLE CLASS BALANCING")
            print("=" * 60)
            X, y = self.balance_idle_class(X, y, preserve_idle_characteristics=True)
        
        # Generate split indices
        split_file = self.data_dir / f"splits_seed{self.random_seed}_hash{self.config_hash}.json"
        
        if save_splits and split_file.exists():
            print(f"\n📂 Loading existing splits from {split_file}")
            with open(split_file, 'r') as f:
                splits = json.load(f)
            
            # Validate config hash
            if splits.get('config_hash') != self.config_hash:
                print(f"⚠️  Split config hash mismatch!")
                print(f"   Split: {splits.get('config_hash')}")
                print(f"   Current: {self.config_hash}")
                print(f"   Creating new splits...")
                
                # Create new splits
                train_idx, val_idx, test_idx = self._create_splits(
                    X, y, test_size, val_size, stratify
                )
            else:
                train_idx = splits['train']
                val_idx = splits['val']
                test_idx = splits['test']
                print(f"   ✅ Split validation passed")
        else:
            print("\n📊 Creating new dataset splits...")
            train_idx, val_idx, test_idx = self._create_splits(
                X, y, test_size, val_size, stratify
            )
        
        # Get actual data arrays
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        
        # Store split indices
        self.split_indices = {
            'train': train_idx.tolist(),
            'val': val_idx.tolist(),
            'test': test_idx.tolist(),
            'filepaths': filepaths,
            'config_hash': self.config_hash,
            'config': {
                'test_size': test_size,
                'val_size': val_size,
                'random_seed': self.random_seed,
                'stratify': stratify,
                'balance_idle': balance_idle,
                'use_features': use_features,
                'total_samples': len(X),
                'feature_dim': feature_dim,
                'idle_strategy': IDLE_STRATEGY,
                'idle_features_enabled': IDLE_FEATURES_ENABLED,
            }
        }
        
        if save_splits:
            with open(split_file, 'w') as f:
                json.dump(self.split_indices, f, indent=2)
            print(f"✅ Split indices saved to {split_file}")
        
        # One-hot encode labels
        y_train_onehot = tf.keras.utils.to_categorical(y_train, len(self.gesture_classes))
        y_val_onehot = tf.keras.utils.to_categorical(y_val, len(self.gesture_classes))
        y_test_onehot = tf.keras.utils.to_categorical(y_test, len(self.gesture_classes))
        
        print(f"\n✅ Dataset splits prepared:")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Val:   {len(X_val)} samples")
        print(f"   Test:  {len(X_test)} samples")
        
        # Print class distribution
        self._print_split_distribution(y_train, y_val, y_test)
        
        return (X_train, y_train_onehot), (X_val, y_val_onehot), (X_test, y_test_onehot)
    
    def _create_splits(self, X, y, test_size, val_size, stratify):
        """Create train/val/test splits."""
        # First split: separate test set
        if stratify:
            X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
                X, y, np.arange(len(X)),
                test_size=test_size,
                stratify=y,
                random_state=self.random_seed
            )
        else:
            X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
                X, y, np.arange(len(X)),
                test_size=test_size,
                random_state=self.random_seed
            )
        
        # Second split: separate validation from training
        val_ratio = val_size / (1 - test_size)
        
        if stratify:
            X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
                X_temp, y_temp, idx_temp,
                test_size=val_ratio,
                stratify=y_temp,
                random_state=self.random_seed
            )
        else:
            X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
                X_temp, y_temp, idx_temp,
                test_size=val_ratio,
                random_state=self.random_seed
            )
        
        return train_idx, val_idx, idx_test
    
    def _print_split_distribution(self, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray):
        """Print class distribution for each split."""
        print(f"\n📊 Class distribution in splits:")
        print(f"   {'Class':<15} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
        print(f"   " + "-" * 50)
        
        for cls_idx, cls_name in self.idx_to_class.items():
            train_count = np.sum(y_train == cls_idx)
            val_count = np.sum(y_val == cls_idx)
            test_count = np.sum(y_test == cls_idx)
            total = train_count + val_count + test_count
            
            marker = "⭐ " if cls_name == 'idle' else "  "
            if cls_name in ['run', 'jogging']:
                marker = "🏃 " if cls_name == 'run' else "🏃‍♂️ "
            threshold = IDLE_CONFIDENCE_THRESHOLD if cls_name == 'idle' else ACTIVE_CONFIDENCE_THRESHOLD
            
            print(f"   {marker}{cls_name:<13} {train_count:<8} {val_count:<8} {test_count:<8} {total:<8}")
        
        print(f"\n   Inference thresholds:")
        print(f"     Idle:   {IDLE_CONFIDENCE_THRESHOLD}")
        print(f"     Active: {ACTIVE_CONFIDENCE_THRESHOLD}")
    
    def compute_class_weights(self, y_train: np.ndarray, 
                             idle_weight_multiplier: float = 1.2) -> Dict[int, float]:
        """
        Compute class weights with emphasis on idle class.
        
        Args:
            y_train: Training labels (integer or one-hot)
            idle_weight_multiplier: Multiplier for idle class weight
            
        Returns:
            Dictionary mapping class indices to weights
        """
        # Convert one-hot to integer if needed
        if len(y_train.shape) == 2:
            y_train_int = np.argmax(y_train, axis=1)
        else:
            y_train_int = y_train
        
        # Compute basic class weights
        unique_classes = np.unique(y_train_int)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y_train_int
        )
        
        # Create dictionary
        weight_dict = {int(cls): float(weight) for cls, weight in zip(unique_classes, class_weights)}
        
        # Adjust idle class weight if it exists
        if self.idle_idx is not None and self.idle_idx in weight_dict:
            idle_count = np.sum(y_train_int == self.idle_idx)
            active_classes = [i for i in unique_classes if i != self.idle_idx]
            
            if active_classes:
                max_active_count = max([np.sum(y_train_int == i) for i in active_classes])
                
                if idle_count < max_active_count:
                    shortage_ratio = max_active_count / max(idle_count, 1)
                    adjustment = min(shortage_ratio * idle_weight_multiplier, 3.0)
                    weight_dict[self.idle_idx] *= adjustment
                    print(f"\n⚖️  Adjusted idle class weight by {adjustment:.2f}x")
        
        print("\n✅ Class weights computed:")
        for cls_idx, weight in sorted(weight_dict.items()):
            cls_name = self.idx_to_class[cls_idx]
            count = np.sum(y_train_int == cls_idx)
            marker = "⭐ " if cls_name == 'idle' else "  "
            if cls_name in ['run', 'jogging']:
                marker = "🏃 " if cls_name == 'run' else "🏃‍♂️ "
            threshold = IDLE_CONFIDENCE_THRESHOLD if cls_name == 'idle' else ACTIVE_CONFIDENCE_THRESHOLD
            print(f"   {marker}{cls_name:<15}: weight={weight:.3f}, samples={count}, threshold={threshold}")
        
        return weight_dict
    
    def augment_datasets(self, 
                        X_train: np.ndarray, 
                        y_train: np.ndarray,
                        augment_factor: int = AUGMENTATION_FACTOR,
                        methods: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply temporal augmentation to training data.
        
        CRITICAL FIX: 
        - Idle and run gestures are automatically excluded from aggressive augmentation
        - Uses TemporalAugmenter's built-in protection for sensitive gestures
        
        Args:
            X_train: Training sequences
            y_train: Training labels (one-hot)
            augment_factor: How many augmented copies to create
            methods: List of augmentation methods to apply
            
        Returns:
            Augmented training data and labels
        """
        if not USE_AUGMENTATION or augment_factor <= 1:
            print("⚠️  Data augmentation disabled or factor <= 1")
            return X_train, y_train
        
        print(f"\n🎭 Applying temporal augmentation (factor: {augment_factor})...")
        
        # Convert one-hot labels to integer for augmentation
        y_train_int = np.argmax(y_train, axis=1)
        
        # Use the fixed augment_batch method with gesture names
        X_augmented, y_augmented_int = TemporalAugmenter.augment_batch(
            X_train, y_train_int,
            augment_factor=augment_factor,
            methods=methods,
            gesture_names=self.gesture_classes
        )
        
        # Convert back to one-hot
        y_augmented = tf.keras.utils.to_categorical(y_augmented_int, len(self.gesture_classes))
        
        # Shuffle
        indices = np.random.permutation(len(X_augmented))
        X_augmented = X_augmented[indices]
        y_augmented = y_augmented[indices]
        
        print(f"✅ Augmentation complete:")
        print(f"   Before: {len(X_train)} samples")
        print(f"   After:  {len(X_augmented)} samples")
        print(f"   Multiplier: {len(X_augmented) / len(X_train):.1f}x")
        
        return X_augmented, y_augmented
    
    def fit_normalizer(self, X_train: np.ndarray):
        """
        Fit feature normalizer on training data.
        
        Args:
            X_train: Training features
        """
        if not FEATURE_ENGINEER_AVAILABLE:
            print("⚠️  FeatureNormalizer not available, skipping")
            return
        
        try:
            self.normalizer = FeatureNormalizer()
            self.normalizer.fit(X_train)
            self.normalizer_fitted = True
            
            # Save normalizer
            normalizer_path = NORMALIZER_DIR / f"normalizer_{self.config_hash}.npz"
            self.normalizer.save(str(normalizer_path))
            
            print(f"\n✅ Feature normalizer fitted and saved")
            print(f"   Path: {normalizer_path}")
            print(f"   Config hash: {self.config_hash}")
            
        except Exception as e:
            print(f"⚠️  Failed to fit normalizer: {e}")
            self.normalizer = None
            self.normalizer_fitted = False
    
    def normalize_features(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize features using fitted normalizer.
        
        Args:
            X: Features to normalize
            
        Returns:
            Normalized features
        """
        if self.normalizer is None or not self.normalizer_fitted:
            return X
        
        try:
            return self.normalizer.transform(X, expected_dim=self.feature_dim)
        except Exception as e:
            print(f"⚠️  Normalization failed: {e}")
            return X
    
    def create_tf_datasets(self, 
                          X_train: np.ndarray, 
                          y_train: np.ndarray,
                          X_val: np.ndarray, 
                          y_val: np.ndarray,
                          batch_size: int = BATCH_SIZE,
                          shuffle_buffer: int = 1000) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Create TensorFlow datasets for efficient training.
        """
        # Training dataset with shuffling and batching
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(
            buffer_size=shuffle_buffer,
            seed=self.random_seed,
            reshuffle_each_iteration=True
        )
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Validation dataset
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
        print(f"\n✅ TensorFlow datasets created:")
        print(f"   Train batches: {len(train_dataset)}")
        print(f"   Val batches: {len(val_dataset)}")
        print(f"   Batch size: {batch_size}")
        
        return train_dataset, val_dataset
    
    def get_feature_dimension(self) -> int:
        """Get the feature dimension."""
        if self.feature_dim is not None:
            return self.feature_dim
        
        if self.feature_engineer:
            self.feature_dim = self.feature_engineer.get_feature_dimension()
            return self.feature_dim
        
        # Try to infer from data
        try:
            X, _, _ = self.load_all_sequences(use_features=True, cache=False, min_samples=1)
            if len(X) > 0:
                self.feature_dim = X.shape[2] if len(X.shape) > 2 else X.shape[1]
                return self.feature_dim
        except:
            pass
        
        return 0
    
    def get_dataset_statistics(self, use_features: bool = True) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        X, y, filepaths = self.load_all_sequences(use_features=use_features, cache=False)
        
        stats = {
            'timestamp': np.datetime64('now').astype(str),
            'total_sequences': len(X),
            'gesture_distribution': {},
            'feature_statistics': {},
            'idle_class_info': {},
            'run_class_info': {},
            'feature_engineering': {
                'enabled': self.feature_engineer is not None,
                'feature_dim': X.shape[2] if len(X.shape) > 2 else X.shape[1],
                'config_hash': self.config_hash,
            },
            'idle_strategy': {
                'strategy': IDLE_STRATEGY,
                'features_enabled': IDLE_FEATURES_ENABLED,
                'samples_multiplier': IDLE_SAMPLES_MULTIPLIER,
                'idle_threshold': IDLE_CONFIDENCE_THRESHOLD,
                'active_threshold': ACTIVE_CONFIDENCE_THRESHOLD,
            }
        }
        
        # Gesture distribution
        unique, counts = np.unique(y, return_counts=True)
        for cls_idx, count in zip(unique, counts):
            cls_name = self.idx_to_class[cls_idx]
            stats['gesture_distribution'][cls_name] = {
                'count': int(count),
                'percentage': float(count / len(y) * 100)
            }
        
        # Idle class specific info
        if self.idle_idx is not None:
            idle_count = counts[unique == self.idle_idx][0] if self.idle_idx in unique else 0
            active_counts = [count for cls_idx, count in zip(unique, counts) 
                           if cls_idx != self.idle_idx]
            max_active = max(active_counts) if active_counts else 0
            
            required_idle = int(max_active * IDLE_SAMPLES_MULTIPLIER) if max_active > 0 else 0
            current_ratio = idle_count / max_active if max_active > 0 else 0
            
            stats['idle_class_info'] = {
                'count': idle_count,
                'max_active_count': max_active,
                'current_ratio': current_ratio,
                'required_count': required_idle,
                'samples_needed': max(0, required_idle - idle_count),
                'status': 'balanced' if idle_count >= required_idle else 'under_represented',
            }
        
        # Run/Jogging class specific info
        if self.run_idx is not None:
            run_count = counts[unique == self.run_idx][0] if self.run_idx in unique else 0
            stats['run_class_info'] = {
                'count': run_count,
                'is_periodic': True,
                'augmentation': 'minimal (preserve periodicity)'
            }
        
        # Feature statistics
        stats['feature_statistics'] = {
            'mean': float(np.mean(X)),
            'std': float(np.std(X)),
            'min': float(np.min(X)),
            'max': float(np.max(X)),
            'shape': X.shape,
        }
        
        return stats
    
    def save_dataset_report(self, output_path: Optional[Path] = None):
        """Save a detailed dataset report to JSON file."""
        if output_path is None:
            output_path = self.data_dir / f"dataset_report_{self.config_hash}.json"
        
        report = self.get_dataset_statistics()
        report['split_indices'] = self.split_indices
        report['recommendations'] = self._get_recommendations(report)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✅ Dataset report saved to {output_path}")
        
        return report
    
    def _get_recommendations(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommendations for dataset balancing."""
        recommendations = {
            'idle_class': {},
            'active_classes': [],
            'run_class': {},
            'general': []
        }
        
        # Idle class recommendations
        idle_info = stats.get('idle_class_info', {})
        if idle_info.get('status') == 'under_represented':
            recommendations['idle_class'] = {
                'action': 'collect_more_samples',
                'samples_needed': idle_info.get('samples_needed', 0),
                'reason': f"Idle has {idle_info.get('count', 0)} samples, needs {idle_info.get('required_count', 0)}",
                'command': 'python main.py collect_idle'
            }
        
        # Run class recommendations
        run_info = stats.get('run_class_info', {})
        if run_info.get('count', 0) < MIN_SAMPLES_PER_GESTURE:
            recommendations['run_class'] = {
                'action': 'collect_more_samples',
                'samples_needed': MIN_SAMPLES_PER_GESTURE - run_info.get('count', 0),
                'reason': f'Periodic gesture needs more samples',
                'command': 'python main.py collect --gesture run'
            }
        
        # Check minimum samples for all classes
        for cls_name, cls_info in stats.get('gesture_distribution', {}).items():
            if cls_name not in ['idle', 'run', 'jogging'] and cls_info['count'] < MIN_SAMPLES_PER_GESTURE:
                recommendations['active_classes'].append({
                    'gesture': cls_name,
                    'action': 'collect_more_samples',
                    'samples_needed': MIN_SAMPLES_PER_GESTURE - cls_info['count'],
                    'reason': f'Below minimum of {MIN_SAMPLES_PER_GESTURE} samples',
                    'command': f'python main.py collect --gesture {cls_name}'
                })
        
        return recommendations


# ============================================================================
# VERIFICATION FUNCTIONS
# ============================================================================

def verify_idle_handling_consistency():
    """Verify that idle handling is consistent across all components."""
    print("\n" + "=" * 60)
    print("VERIFYING IDLE HANDLING CONSISTENCY")
    print("=" * 60)
    
    issues = []
    
    # 1. Check feature config
    if not FEATURE_CONFIG.compute_idle_features:
        issues.append("❌ FEATURE_CONFIG.compute_idle_features is False (should be True)")
    else:
        print("✅ FeatureConfig.compute_idle_features = True")
    
    # 2. Check idle strategy
    if IDLE_STRATEGY != "explicit_features":
        issues.append(f"❌ IDLE_STRATEGY is '{IDLE_STRATEGY}' (should be 'explicit_features')")
    else:
        print("✅ IDLE_STRATEGY = 'explicit_features'")
    
    # 3. Check thresholds
    if IDLE_CONFIDENCE_THRESHOLD >= ACTIVE_CONFIDENCE_THRESHOLD:
        issues.append(f"⚠️  Idle threshold ({IDLE_CONFIDENCE_THRESHOLD}) >= active threshold ({ACTIVE_CONFIDENCE_THRESHOLD})")
    else:
        print(f"✅ Idle threshold ({IDLE_CONFIDENCE_THRESHOLD}) < active threshold ({ACTIVE_CONFIDENCE_THRESHOLD})")
    
    # 4. Check idle in GESTURE_CLASSES
    if 'idle' not in GESTURE_CLASSES:
        issues.append("❌ 'idle' not in GESTURE_CLASSES")
    else:
        idle_idx = GESTURE_CLASSES.index('idle')
        print(f"✅ 'idle' in GESTURE_CLASSES at index {idle_idx}")
    
    # 5. Check augmentation protection
    print("\n🛡️  Augmentation protection check:")
    for gesture in TemporalAugmenter.SENSITIVE_GESTURES:
        if gesture in GESTURE_CLASSES:
            print(f"   ✅ '{gesture}' protected from aggressive augmentation")
        else:
            print(f"   ℹ️  '{gesture}' not in gesture classes")
    
    if issues:
        print("\n" + "=" * 60)
        print("ISSUES FOUND:")
        for issue in issues:
            print(issue)
        print("=" * 60)
        return False
    
    print("\n✅ All idle handling consistency checks passed!")
    print("=" * 60)
    return True


def test_pipeline():
    """Test the data pipeline."""
    print("\n" + "=" * 60)
    print("TESTING DATA PIPELINE")
    print("=" * 60)
    
    # Verify consistency first
    if not verify_idle_handling_consistency():
        print("\n❌ Idle handling consistency check failed!")
        return
    
    # Initialize pipeline
    pipeline = GestureDataPipeline(
        data_dir=DATA_DIR,
        sequence_length=SEQUENCE_LENGTH,
        random_seed=42
    )
    
    # Get dataset statistics
    try:
        stats = pipeline.get_dataset_statistics()
        print(f"\n📊 Dataset statistics:")
        print(f"   Total sequences: {stats['total_sequences']}")
        print(f"   Feature dimension: {stats['feature_engineering']['feature_dim']}")
        print(f"   Feature config hash: {stats['feature_engineering']['config_hash']}")
        
        if 'idle_class_info' in stats:
            idle_info = stats['idle_class_info']
            print(f"\n   Idle class info:")
            print(f"     Samples: {idle_info['count']}")
            print(f"     Ratio: {idle_info['current_ratio']:.2f}x")
            print(f"     Status: {idle_info['status']}")
        
        if 'run_class_info' in stats:
            run_info = stats['run_class_info']
            print(f"\n   Run class info:")
            print(f"     Samples: {run_info['count']}")
            print(f"     Periodic: {run_info.get('is_periodic', False)}")
            print(f"     Augmentation: {run_info.get('augmentation', 'standard')}")
    except Exception as e:
        print(f"⚠️  Could not get dataset statistics: {e}")
    
    # Test dataset preparation
    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = pipeline.prepare_datasets(
            test_size=0.2,
            val_size=0.1,
            save_splits=False,
            balance_idle=True
        )
        
        print(f"\n✅ Dataset preparation successful")
        print(f"   Train: {X_train.shape}")
        print(f"   Val:   {X_val.shape}")
        print(f"   Test:  {X_test.shape}")
        
        # Test augmentation with protection
        if USE_AUGMENTATION and AUGMENTATION_FACTOR > 1:
            X_aug, y_aug = pipeline.augment_datasets(X_train, y_train, augment_factor=2)
            print(f"\n✅ Augmentation test successful")
            print(f"   Augmented shape: {X_aug.shape}")
        
        # Compute class weights
        pipeline.compute_class_weights(y_train)
        
    except Exception as e:
        print(f"\n❌ Dataset preparation failed: {e}")
        print("   This is normal if no data is available yet.")
        print("   Collect data first: python main.py collect_idle")
    
    print("\n" + "=" * 60)
    print("PIPELINE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_pipeline()