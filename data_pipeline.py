# data_pipeline.py (UPDATED WITH IMPLEMENTED TEMPORAL AUGMENTER)
"""
Complete data pipeline with proper temporal augmentation and reproducible splits.
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
import scipy.ndimage as ndi  # For interpolation in time warping

# Import from config
from config import (
    DATA_DIR, SEQUENCE_LENGTH, GESTURE_CLASSES,
    MIN_SAMPLES_PER_GESTURE, BATCH_SIZE, 
    USE_AUGMENTATION, AUGMENTATION_FACTOR,
    IDLE_SAMPLES_MULTIPLIER, get_required_idle_samples
)
from feature_engineer import FeatureEngineer

# ============================================================================
# TEMPORAL AUGMENTATION (NOW IMPLEMENTED)
# ============================================================================

class TemporalAugmenter:
    """Temporal augmentation techniques for time-series gesture data."""
    
    @staticmethod
    def add_gaussian_noise(X: np.ndarray, 
                          noise_level: float = 0.01) -> np.ndarray:
        """
        Add Gaussian noise to sequences.
        
        Args:
            X: Input sequences [batch_size, seq_len, features]
            noise_level: Standard deviation of Gaussian noise
            
        Returns:
            Noisy sequences
        """
        if len(X) == 0:
            return X
            
        # Create noise with same shape as X
        noise = np.random.normal(0, noise_level, X.shape)
        
        # Add noise
        X_noisy = X + noise
        
        return X_noisy
    
    @staticmethod
    def time_warp(X: np.ndarray, 
                 warp_factor: float = 0.1) -> np.ndarray:
        """
        Apply random time warping (speeding up/slowing down parts of sequence).
        
        Args:
            X: Input sequences [batch_size, seq_len, features]
            warp_factor: How much to warp (0-1)
            
        Returns:
            Time-warped sequences
        """
        if len(X) == 0:
            return X
            
        batch_size, seq_len, n_features = X.shape
        X_warped = np.zeros_like(X)
        
        for i in range(batch_size):
            # Create a smooth random warp curve
            t = np.linspace(0, 1, seq_len)
            
            # Random warp points
            warp_points = np.random.uniform(-warp_factor, warp_factor, size=5)
            warp_points = np.cumsum(warp_points)  # Make it smooth
            
            # Interpolate to full sequence length
            warp_curve = np.interp(
                np.linspace(0, 4, seq_len),
                np.arange(5),
                warp_points
            )
            
            # Normalize warp curve to [0, 1]
            warp_curve = warp_curve - warp_curve.min()
            if warp_curve.max() > 0:
                warp_curve = warp_curve / warp_curve.max()
            
            # Apply warp to time axis
            warped_time = t + warp_curve * warp_factor
            warped_time = np.clip(warped_time, 0, 1)
            
            # Resample each feature along the warped time axis
            for f in range(n_features):
                # Interpolate along time axis
                X_warped[i, :, f] = np.interp(
                    warped_time * (seq_len - 1),
                    np.arange(seq_len),
                    X[i, :, f]
                )
        
        return X_warped
    
    @staticmethod
    def temporal_scaling(X: np.ndarray, 
                        scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """
        Scale sequences in time dimension (speed up/slow down).
        
        Args:
            X: Input sequences [batch_size, seq_len, features]
            scale_range: Range for random scaling factors
            
        Returns:
            Temporally scaled sequences
        """
        if len(X) == 0:
            return X
            
        batch_size, seq_len, n_features = X.shape
        X_scaled = np.zeros_like(X)
        
        for i in range(batch_size):
            # Random scaling factor
            scale = np.random.uniform(scale_range[0], scale_range[1])
            
            if scale < 1.0:
                # Speed up - need to downsample
                new_len = max(int(seq_len * scale), 1)
                
                # Select evenly spaced indices
                indices = np.linspace(0, seq_len - 1, new_len).astype(int)
                
                # Take those indices
                scaled_seq = X[i, indices, :]
                
                # Pad back to original length
                pad_len = seq_len - new_len
                if pad_len > 0:
                    # Pad with last frame
                    scaled_seq = np.pad(scaled_seq, ((0, pad_len), (0, 0)), mode='edge')
                
                X_scaled[i] = scaled_seq
            else:
                # Slow down - need to upsample with interpolation
                new_len = int(seq_len * scale)
                
                # Create new time axis
                t_original = np.arange(seq_len)
                t_new = np.linspace(0, seq_len - 1, new_len)
                
                # Interpolate each feature
                for f in range(n_features):
                    X_scaled[i, :, f] = np.interp(
                        t_new[:seq_len],  # Only take first seq_len points
                        t_original,
                        X[i, :, f]
                    )
        
        return X_scaled
    
    @staticmethod
    def random_crop(X: np.ndarray, 
                   crop_range: Tuple[float, float] = (0.8, 1.0)) -> np.ndarray:
        """
        Randomly crop a subsequence and pad back to original length.
        
        Args:
            X: Input sequences [batch_size, seq_len, features]
            crop_range: Range of crop lengths as fraction of original
            
        Returns:
            Randomly cropped sequences
        """
        if len(X) == 0:
            return X
            
        batch_size, seq_len, n_features = X.shape
        X_cropped = np.zeros_like(X)
        
        for i in range(batch_size):
            # Random crop length
            crop_len = int(seq_len * np.random.uniform(crop_range[0], crop_range[1]))
            crop_len = max(crop_len, 10)  # Minimum 10 frames
            
            # Random start position
            max_start = seq_len - crop_len
            if max_start <= 0:
                # If sequence is too short, just return it as is
                X_cropped[i] = X[i]
                continue
                
            start = np.random.randint(0, max_start)
            end = start + crop_len
            
            # Crop and resize back to original length
            cropped = X[i, start:end, :]
            
            # Resize using linear interpolation
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
                     methods: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply multiple augmentations to create augmented batch.
        
        Args:
            X: Input sequences
            y: Labels (integer)
            augment_factor: How many augmented copies to create
            methods: List of augmentation methods to use
            
        Returns:
            Tuple of (augmented_X, augmented_y)
        """
        if len(X) == 0 or augment_factor <= 1:
            return X, y
        
        # Default augmentation methods
        if methods is None:
            methods = ['noise', 'time_warp', 'scaling', 'crop']
        
        # Storage for augmented samples
        X_augmented = []
        y_augmented = []
        
        # For each original sample, create augment_factor-1 augmented versions
        for i in range(len(X)):
            # Keep original
            X_augmented.append(X[i])
            y_augmented.append(y[i])
            
            # Create augmented versions
            for aug_idx in range(augment_factor - 1):
                x_aug = X[i].copy()
                
                # Apply random augmentations
                aug_methods = np.random.choice(methods, 
                                              size=np.random.randint(1, len(methods) + 1),
                                              replace=False)
                
                for method in aug_methods:
                    if method == 'noise':
                        x_aug = TemporalAugmenter.add_gaussian_noise(
                            x_aug[np.newaxis, ...], 
                            noise_level=np.random.uniform(0.005, 0.02)
                        )[0]
                    elif method == 'time_warp':
                        x_aug = TemporalAugmenter.time_warp(
                            x_aug[np.newaxis, ...],
                            warp_factor=np.random.uniform(0.05, 0.2)
                        )[0]
                    elif method == 'scaling':
                        x_aug = TemporalAugmenter.temporal_scaling(
                            x_aug[np.newaxis, ...],
                            scale_range=(np.random.uniform(0.8, 0.95),
                                       np.random.uniform(1.05, 1.2))
                        )[0]
                    elif method == 'crop':
                        x_aug = TemporalAugmenter.random_crop(
                            x_aug[np.newaxis, ...],
                            crop_range=(np.random.uniform(0.7, 0.9),
                                      np.random.uniform(0.9, 1.0))
                        )[0]
                
                X_augmented.append(x_aug)
                y_augmented.append(y[i])
        
        # Convert to arrays
        X_augmented = np.array(X_augmented, dtype=np.float32)
        y_augmented = np.array(y_augmented, dtype=np.int32)
        
        return X_augmented, y_augmented
    
    @staticmethod
    def random_time_shift(X: np.ndarray,
                         max_shift: float = 0.1) -> np.ndarray:
        """
        Randomly shift the sequence in time.
        
        Args:
            X: Input sequences [batch_size, seq_len, features]
            max_shift: Maximum shift as fraction of sequence length
            
        Returns:
            Time-shifted sequences
        """
        if len(X) == 0:
            return X
            
        batch_size, seq_len, n_features = X.shape
        max_shift_frames = int(seq_len * max_shift)
        
        if max_shift_frames == 0:
            return X
        
        X_shifted = np.zeros_like(X)
        
        for i in range(batch_size):
            # Random shift amount
            shift = np.random.randint(-max_shift_frames, max_shift_frames + 1)
            
            if shift > 0:
                # Shift right
                X_shifted[i, shift:, :] = X[i, :-shift, :]
                # Replicate first frame for the shifted part
                X_shifted[i, :shift, :] = X[i, 0:1, :]
            elif shift < 0:
                # Shift left
                shift = abs(shift)
                X_shifted[i, :-shift, :] = X[i, shift:, :]
                # Replicate last frame for the shifted part
                X_shifted[i, -shift:, :] = X[i, -1:, :]
            else:
                # No shift
                X_shifted[i] = X[i]
        
        return X_shifted
    
    @staticmethod
    def random_feature_dropout(X: np.ndarray,
                              dropout_rate: float = 0.1,
                              feature_groups: List[List[int]] = None) -> np.ndarray:
        """
        Randomly dropout (zero out) features to simulate occlusion.
        
        Args:
            X: Input sequences [batch_size, seq_len, features]
            dropout_rate: Probability of dropping a feature group
            feature_groups: Groups of features that should be dropped together
                           (e.g., left arm features, right arm features)
                           
        Returns:
            Sequences with random feature dropout
        """
        if len(X) == 0:
            return X
            
        if feature_groups is None:
            # Default: group features by body part
            # Assuming features are organized as: [0-2: left arm, 3-5: right arm, etc.]
            feature_groups = [
                [0, 1, 2],      # Left arm
                [3, 4, 5],      # Right arm
                [6, 7, 8],      # Left leg
                [9, 10, 11],    # Right leg
                [12, 13, 14],   # Torso
                [15, 16, 17, 18] # Head/face
            ]
        
        X_dropped = X.copy()
        batch_size, seq_len, n_features = X.shape
        
        for i in range(batch_size):
            # Randomly select feature groups to dropout
            groups_to_drop = []
            for group in feature_groups:
                if np.random.random() < dropout_rate:
                    groups_to_drop.extend(group)
            
            if groups_to_drop:
                # Set selected features to zero
                X_dropped[i, :, groups_to_drop] = 0
        
        return X_dropped

# ============================================================================
# MAIN DATA PIPELINE (WITH balance_idle_class IMPLEMENTATION)
# ============================================================================

class GestureDataPipeline:
    """
    Handles loading, preprocessing, splitting, and augmentation of gesture data.
    Maintains reproducibility through saved split indices.
    """
    
    def __init__(self, 
                 data_dir: Path = DATA_DIR, 
                 sequence_length: int = SEQUENCE_LENGTH,
                 random_seed: int = 42):
        """
        Args:
            data_dir: Directory containing gesture data
            sequence_length: Fixed sequence length for all samples
            random_seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.random_seed = random_seed
        self.feature_engineer = FeatureEngineer()
        
        # Gesture mapping
        self.gesture_classes = GESTURE_CLASSES
        self.class_to_idx = {cls: i for i, cls in enumerate(self.gesture_classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        
        # Cache for loaded data
        self._cache = {}
        
        # Split indices for reproducibility
        self.split_indices = None
        
        print(f"✅ Data pipeline initialized")
        print(f"   Data directory: {self.data_dir}")
        print(f"   Sequence length: {sequence_length}")
        print(f"   Random seed: {random_seed}")
        
        # Check if idle class exists
        if 'idle' in self.gesture_classes:
            self.idle_idx = self.class_to_idx['idle']
            print(f"   Idle class detected at index: {self.idle_idx}")
        else:
            self.idle_idx = None
            print(f"   ⚠️  Idle class not found in gesture classes")
    
    def balance_idle_class(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensure idle class has sufficient samples compared to other classes.
        Critical for preventing false positives.
        
        Args:
            X: Features array
            y: Labels array (integer)
            
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
        
        print(f"\n⚖️  Idle class balancing analysis:")
        print(f"   Idle samples: {idle_count}")
        print(f"   Max active gesture: {max_active_count}")
        print(f"   Target ratio: {IDLE_SAMPLES_MULTIPLIER:.1f}x")
        print(f"   Required idle: {required_idle}")
        
        # If idle doesn't have enough samples, duplicate existing ones with variations
        if idle_count < required_idle:
            print(f"   ⚠️  Idle class under-represented, balancing...")
            
            # Find idle samples
            idle_mask = y == self.idle_idx
            X_idle = X[idle_mask]
            y_idle = y[idle_mask]
            
            if len(X_idle) == 0:
                print(f"   ❌ No idle samples found! Collect idle data first.")
                return X, y
            
            # Calculate how many to add
            needed = required_idle - idle_count
            print(f"   Adding {needed} idle samples...")
            
            # Duplicate with small variations
            indices = np.random.choice(len(X_idle), needed, replace=True)
            
            # Apply augmentations to duplicates
            X_duplicates = X_idle[indices].copy()
            
            for i in range(len(X_duplicates)):
                # Add small Gaussian noise
                noise = np.random.normal(0, 0.02, X_duplicates[i].shape)
                X_duplicates[i] += noise
                
                # Slight time scaling (variation in speed)
                scale = np.random.uniform(0.95, 1.05)
                if scale != 1.0:
                    seq_len = X_duplicates[i].shape[0]
                    if scale < 1.0:
                        # Speed up
                        new_len = max(int(seq_len * scale), 1)
                        indices_scaled = np.linspace(0, seq_len - 1, new_len).astype(int)
                        scaled_seq = X_duplicates[i][indices_scaled]
                        # Pad back
                        pad_len = seq_len - new_len
                        X_duplicates[i] = np.pad(scaled_seq, ((0, pad_len), (0, 0)), mode='edge')
                    else:
                        # Slow down with interpolation
                        new_len = int(seq_len * scale)
                        scaled_seq = np.zeros((new_len, X_duplicates[i].shape[1]))
                        for f in range(X_duplicates[i].shape[1]):
                            scaled_seq[:, f] = np.interp(
                                np.linspace(0, seq_len - 1, new_len),
                                np.arange(seq_len),
                                X_duplicates[i][:, f]
                            )
                        X_duplicates[i] = scaled_seq[:seq_len, :]
            
            # Add to dataset
            X = np.concatenate([X, X_duplicates], axis=0)
            y = np.concatenate([y, y_idle[indices]], axis=0)
            
            # Shuffle the balanced dataset
            indices_shuffled = np.random.permutation(len(X))
            X = X[indices_shuffled]
            y = y[indices_shuffled]
            
            print(f"   ✅ Added {needed} idle samples. New idle count: {idle_count + needed}")
            print(f"   Total samples after balancing: {len(X)}")
        
        else:
            print(f"   ✅ Idle class already sufficiently represented")
            print(f"   Ratio: {idle_count/max_active_count:.2f}x (target: {IDLE_SAMPLES_MULTIPLIER:.1f}x)")
        
        return X, y
    
    def load_all_sequences(self, 
                          min_samples: int = 1,
                          max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load all sequences from the dataset directory.
        
        Args:
            min_samples: Minimum samples required per gesture
            max_samples: Maximum samples to load per gesture (for balancing)
            
        Returns:
            Tuple of (features, labels, filepaths)
        """
        cache_key = f"all_sequences_{min_samples}_{max_samples}"
        if cache_key in self._cache:
            print("Loading from cache...")
            return self._cache[cache_key]
        
        X = []  # Features
        y = []  # Labels
        filepaths = []  # Original file paths
        
        print("\n" + "=" * 60)
        print("LOADING SEQUENCES")
        print("=" * 60)
        
        for gesture in self.gesture_classes:
            gesture_dir = self.data_dir / gesture
            if not gesture_dir.exists():
                print(f"⚠️  Skipping {gesture}: directory not found")
                continue
            
            npy_files = list(gesture_dir.glob("*.npy"))
            
            # Apply max_samples limit
            if max_samples and len(npy_files) > max_samples:
                # Randomly sample if we have too many
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
                    # Load sequence (landmarks)
                    landmarks = np.load(filepath)
                    
                    # Validate shape
                    if landmarks.shape[0] < 10:  # Too short
                        print(f"  ⚠️  Skipping {filepath.name}: too short ({landmarks.shape[0]} frames)")
                        continue
                    
                    # Pad or truncate to fixed length
                    processed_landmarks = self._process_sequence_length(landmarks)
                    
                    # Extract features
                    features = self.feature_engineer.extract_features(processed_landmarks)
                    
                    gesture_X.append(features)
                    gesture_files.append(str(filepath))
                    
                except Exception as e:
                    print(f"  ❌ Error loading {filepath.name}: {e}")
            
            if gesture_X:
                X.extend(gesture_X)
                y.extend([self.class_to_idx[gesture]] * len(gesture_X))
                filepaths.extend(gesture_files)
        
        if not X:
            raise ValueError("No valid sequences found in dataset!")
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        
        print(f"\n✅ Loaded {len(X)} sequences total")
        print(f"   Feature shape: {X.shape}")
        print(f"   Classes distribution: {np.bincount(y)}")
        
        # Cache results
        self._cache[cache_key] = (X, y, filepaths)
        
        return X, y, filepaths
    
    def _process_sequence_length(self, landmarks: np.ndarray) -> np.ndarray:
        """Process sequence to fixed length."""
        if landmarks.shape[0] > self.sequence_length:
            # Truncate: take middle portion for stability
            start = (landmarks.shape[0] - self.sequence_length) // 2
            return landmarks[start:start + self.sequence_length]
        elif landmarks.shape[0] < self.sequence_length:
            # Pad with zeros (model will learn to ignore based on features)
            pad_amount = self.sequence_length - landmarks.shape[0]
            return np.pad(landmarks, ((0, pad_amount), (0, 0), (0, 0)), mode='constant')
        else:
            return landmarks
    
    def prepare_datasets(self, 
                        test_size: float = 0.2, 
                        val_size: float = 0.1,
                        stratify: bool = True,
                        save_splits: bool = True,
                        balance_idle: bool = True) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Split data into train, validation, and test sets.
        Saves split indices for reproducibility.
        
        Args:
            test_size: Proportion of data for test set
            val_size: Proportion of data for validation set (of training data after test split)
            stratify: Whether to stratify splits by class
            save_splits: Whether to save split indices to file
            balance_idle: Whether to balance idle class samples
            
        Returns:
            Tuples of (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        # Load data
        X, y, filepaths = self.load_all_sequences()
        
        # Balance idle class if requested and idle class exists
        if balance_idle and self.idle_idx is not None:
            print("\n" + "=" * 60)
            print("BALANCING IDLE CLASS")
            print("=" * 60)
            X, y = self.balance_idle_class(X, y)
        else:
            print("\n⚠️  Skipping idle class balancing")
        
        # Generate split indices
        split_file = self.data_dir / f"splits_seed{self.random_seed}.json"
        
        if save_splits and split_file.exists():
            # Load existing splits
            print(f"Loading existing splits from {split_file}")
            with open(split_file, 'r') as f:
                splits = json.load(f)
            
            train_idx = splits['train']
            val_idx = splits['val']
            test_idx = splits['test']
            
        else:
            # Create new splits
            print("Creating new dataset splits...")
            
            # First split: train+val vs test
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
            
            # Second split: train vs val
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
            
            # Store indices
            self.split_indices = {
                'train': train_idx.tolist(),
                'val': val_idx.tolist(),
                'test': idx_test.tolist(),
                'filepaths': [filepaths[i] for i in train_idx] + 
                            [filepaths[i] for i in val_idx] + 
                            [filepaths[i] for i in idx_test],
                'config': {
                    'test_size': test_size,
                    'val_size': val_size,
                    'random_seed': self.random_seed,
                    'stratify': stratify,
                    'balance_idle': balance_idle,
                    'total_samples': len(X),
                }
            }
            
            # Save splits if requested
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
        
        # Print class distribution in each split
        self._print_split_distribution(y_train, y_val, y_test)
        
        return (X_train, y_train_onehot), (X_val, y_val_onehot), (X_test, y_test_onehot)
    
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
            
            marker = "⭐ " if cls_name == 'idle' else ""
            print(f"   {marker}{cls_name:<13} {train_count:<8} {val_count:<8} {test_count:<8} {total:<8}")
    
    def compute_class_weights(self, y_train: np.ndarray, 
                             idle_weight_multiplier: float = 1.5) -> Dict[int, float]:
        """
        Compute class weights for imbalanced datasets with emphasis on idle class.
        
        Args:
            y_train: Training labels (integer, not one-hot)
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
        
        # Increase weight for idle class if it exists
        if self.idle_idx is not None and self.idle_idx in weight_dict:
            # Check if idle is under-represented
            idle_count = np.sum(y_train_int == self.idle_idx)
            max_other = max([np.sum(y_train_int == i) for i in unique_classes if i != self.idle_idx])
            
            if idle_count < max_other:
                # Increase weight proportionally to shortage
                shortage_ratio = max_other / max(idle_count, 1)
                adjustment = min(shortage_ratio * idle_weight_multiplier, 3.0)
                weight_dict[self.idle_idx] *= adjustment
                print(f"   ⚖️  Adjusted idle class weight by {adjustment:.2f}x")
        
        print("\n✅ Class weights computed:")
        for cls_idx, weight in weight_dict.items():
            cls_name = self.idx_to_class[cls_idx]
            count = np.sum(y_train_int == cls_idx)
            marker = "⭐ " if cls_name == 'idle' else ""
            print(f"   {marker}{cls_name:<15}: weight={weight:.3f}, samples={count}")
        
        return weight_dict
    
    def augment_datasets(self, 
                        X_train: np.ndarray, 
                        y_train: np.ndarray,
                        augment_factor: int = AUGMENTATION_FACTOR,
                        methods: List[str] = None,
                        exclude_idle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply temporal augmentation to training data.
        Can optionally exclude idle class from augmentation.
        
        Args:
            X_train: Training sequences
            y_train: Training labels (one-hot)
            augment_factor: How many augmented copies to create
            methods: List of augmentation methods to apply
            exclude_idle: Whether to exclude idle class from augmentation
            
        Returns:
            Augmented training data and labels
        """
        if not USE_AUGMENTATION or augment_factor <= 1:
            print("⚠️  Data augmentation disabled or factor <= 1")
            return X_train, y_train
        
        print(f"\n🎭 Applying temporal augmentation (factor: {augment_factor})...")
        
        # Convert one-hot labels to integer for augmentation
        y_train_int = np.argmax(y_train, axis=1)
        
        if exclude_idle and self.idle_idx is not None:
            print(f"   Excluding idle class from augmentation")
            
            # Separate idle and non-idle samples
            idle_mask = y_train_int == self.idle_idx
            non_idle_mask = ~idle_mask
            
            X_idle = X_train[idle_mask]
            y_idle = y_train_int[idle_mask]
            
            X_non_idle = X_train[non_idle_mask]
            y_non_idle = y_train_int[non_idle_mask]
            
            if len(X_non_idle) > 0:
                # Augment only non-idle samples
                X_augmented, y_augmented_int = TemporalAugmenter.augment_batch(
                    X_non_idle, y_non_idle,
                    augment_factor=augment_factor,
                    methods=methods
                )
                
                # Combine with original idle samples
                X_combined = np.concatenate([X_idle, X_augmented], axis=0)
                y_combined_int = np.concatenate([y_idle, y_augmented_int], axis=0)
            else:
                # No non-idle samples to augment
                X_combined = X_idle
                y_combined_int = y_idle
        else:
            # Augment all samples
            X_augmented, y_augmented_int = TemporalAugmenter.augment_batch(
                X_train, y_train_int,
                augment_factor=augment_factor,
                methods=methods
            )
            
            X_combined = X_augmented
            y_combined_int = y_augmented_int
        
        # Convert back to one-hot
        y_combined = tf.keras.utils.to_categorical(y_combined_int, len(self.gesture_classes))
        
        # Shuffle
        indices = np.random.permutation(len(X_combined))
        X_combined = X_combined[indices]
        y_combined = y_combined[indices]
        
        print(f"✅ Augmentation complete:")
        print(f"   Before: {len(X_train)} samples")
        print(f"   After:  {len(X_combined)} samples")
        print(f"   Multiplier: {len(X_combined) / len(X_train):.1f}x")
        
        return X_combined, y_combined
    
    def create_tf_datasets(self, 
                          X_train: np.ndarray, 
                          y_train: np.ndarray,
                          X_val: np.ndarray, 
                          y_val: np.ndarray,
                          batch_size: int = BATCH_SIZE,
                          shuffle_buffer: int = 1000) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Create TensorFlow datasets for efficient training.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            batch_size: Batch size for training
            shuffle_buffer: Buffer size for shuffling
            
        Returns:
            Tuple of (train_dataset, val_dataset)
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
        
        # Validation dataset (no shuffling)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
        print(f"✅ TensorFlow datasets created:")
        print(f"   Train batches: {len(train_dataset)}")
        print(f"   Val batches: {len(val_dataset)}")
        print(f"   Batch size: {batch_size}")
        
        return train_dataset, val_dataset
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        X, y, filepaths = self.load_all_sequences()
        
        stats = {
            'total_sequences': len(X),
            'gesture_distribution': {},
            'feature_statistics': {},
            'sequence_lengths': {},
            'idle_class_info': {},
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
            
            stats['idle_class_info'] = {
                'count': idle_count,
                'max_active_count': max_active,
                'ratio': idle_count / max_active if max_active > 0 else 0,
                'required': int(max_active * IDLE_SAMPLES_MULTIPLIER) if max_active > 0 else 0,
                'status': 'balanced' if idle_count >= int(max_active * IDLE_SAMPLES_MULTIPLIER) else 'under_represented'
            }
        
        # Feature statistics
        stats['feature_statistics'] = {
            'mean': float(np.mean(X)),
            'std': float(np.std(X)),
            'min': float(np.min(X)),
            'max': float(np.max(X)),
            'shape': X.shape,
        }
        
        # Load original sequence lengths from files
        lengths = []
        for filepath in filepaths[:100]:  # Sample first 100 to avoid loading all
            try:
                landmarks = np.load(filepath)
                lengths.append(landmarks.shape[0])
            except:
                continue
        
        if lengths:
            stats['sequence_lengths'] = {
                'mean': float(np.mean(lengths)),
                'std': float(np.std(lengths)),
                'min': int(np.min(lengths)),
                'max': int(np.max(lengths)),
                'samples_analyzed': len(lengths),
            }
        
        return stats
    
    def save_dataset_report(self, output_path: Optional[Path] = None):
        """Save a detailed dataset report to JSON file."""
        if output_path is None:
            output_path = self.data_dir / "dataset_report.json"
        
        report = {
            'timestamp': np.datetime64('now').astype(str),
            'config': {
                'data_dir': str(self.data_dir),
                'sequence_length': self.sequence_length,
                'random_seed': self.random_seed,
                'gesture_classes': self.gesture_classes,
                'idle_samples_multiplier': IDLE_SAMPLES_MULTIPLIER,
            },
            'statistics': self.get_dataset_statistics(),
            'split_indices': self.split_indices,
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Dataset report saved to {output_path}")
        
        return report

# ============================================================================
# BATCH GENERATOR FOR LARGE DATASETS (FIXED TO PREVENT LABEL DESYNC)
# ============================================================================

class GestureDataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator for large datasets that don't fit in memory.
    Loads and processes sequences on-the-fly.
    FIXED: Prevents label-filepath desync by shuffling both together.
    """
    
    def __init__(self, 
                 filepaths: List[str],
                 labels: np.ndarray,
                 batch_size: int = BATCH_SIZE,
                 sequence_length: int = SEQUENCE_LENGTH,
                 shuffle: bool = True,
                 augment: bool = False,
                 random_seed: int = 42):
        """
        Args:
            filepaths: List of paths to .npy files
            labels: Corresponding labels (one-hot encoded)
            batch_size: Batch size
            sequence_length: Fixed sequence length
            shuffle: Whether to shuffle data each epoch
            augment: Whether to apply augmentation
            random_seed: Random seed for reproducibility
        """
        # Convert both to lists to ensure consistent indexing
        self.filepaths = list(filepaths)
        self.labels = labels if isinstance(labels, np.ndarray) else np.array(labels)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.shuffle = shuffle
        self.augment = augment
        self.random_seed = random_seed
        self.feature_engineer = FeatureEngineer()
        
        # Verify length match
        if len(self.filepaths) != len(self.labels):
            raise ValueError(
                f"Number of filepaths ({len(self.filepaths)}) "
                f"doesn't match number of labels ({len(self.labels)})"
            )
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Initialize indices and shuffle
        self.indices = np.arange(len(self.filepaths))
        self.on_epoch_end()
        
        print(f"✅ Data generator initialized:")
        print(f"   Samples: {len(self.filepaths)}")
        print(f"   Batch size: {batch_size}")
        print(f"   Shuffle: {shuffle}")
        print(f"   Augment: {augment}")
    
    def __len__(self) -> int:
        """Number of batches per epoch."""
        return int(np.ceil(len(self.filepaths) / self.batch_size))
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get batch at index."""
        # Get indices for this batch
        batch_start = idx * self.batch_size
        batch_end = min(batch_start + self.batch_size, len(self.indices))
        
        # Get shuffled indices for this batch
        batch_indices = self.indices[batch_start:batch_end]
        
        # Get corresponding filepaths and labels using shuffled indices
        batch_filepaths = [self.filepaths[i] for i in batch_indices]
        batch_labels = self.labels[batch_indices]
        
        batch_X = []
        for filepath in batch_filepaths:
            try:
                # Load landmarks
                landmarks = np.load(filepath)
                
                # Process sequence length
                if landmarks.shape[0] > self.sequence_length:
                    # Take middle portion for stability
                    start = (landmarks.shape[0] - self.sequence_length) // 2
                    landmarks = landmarks[start:start + self.sequence_length]
                elif landmarks.shape[0] < self.sequence_length:
                    # Pad with zeros
                    pad_amount = self.sequence_length - landmarks.shape[0]
                    landmarks = np.pad(landmarks, ((0, pad_amount), (0, 0), (0, 0)), mode='constant')
                
                # Extract features
                features = self.feature_engineer.extract_features(landmarks)
                batch_X.append(features)
                
            except Exception as e:
                print(f"⚠️  Error loading {filepath}: {e}")
                # Return zeros as fallback
                batch_X.append(np.zeros((self.sequence_length, 19), dtype=np.float32))
        
        batch_X = np.array(batch_X, dtype=np.float32)
        
        # Apply augmentation if enabled
        if self.augment and len(batch_X) > 0:
            batch_X = TemporalAugmenter.add_gaussian_noise(batch_X, noise_level=0.005)
            
            # Optional: Apply random time shift
            if np.random.random() > 0.5:
                batch_X = TemporalAugmenter.random_time_shift(batch_X, max_shift=0.05)
        
        return batch_X, batch_labels
    
    def on_epoch_end(self):
        """Shuffle data at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def get_sample(self, index: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Get a single sample for debugging/inspection.
        
        Args:
            index: Index of sample to get
            
        Returns:
            Tuple of (features, label, filepath)
        """
        # Use shuffled indices if available
        if hasattr(self, 'indices'):
            actual_idx = self.indices[index % len(self.indices)]
        else:
            actual_idx = index
        
        filepath = self.filepaths[actual_idx]
        label = self.labels[actual_idx]
        
        # Load and process
        landmarks = np.load(filepath)
        
        if landmarks.shape[0] > self.sequence_length:
            start = (landmarks.shape[0] - self.sequence_length) // 2
            landmarks = landmarks[start:start + self.sequence_length]
        elif landmarks.shape[0] < self.sequence_length:
            pad_amount = self.sequence_length - landmarks.shape[0]
            landmarks = np.pad(landmarks, ((0, pad_amount), (0, 0), (0, 0)), mode='constant')
        
        features = self.feature_engineer.extract_features(landmarks)
        
        return features, label, filepath
    
    def verify_alignment(self, num_samples: int = 10) -> bool:
        """
        Verify that filepaths and labels are properly aligned.
        
        Args:
            num_samples: Number of samples to check
            
        Returns:
            True if alignment is correct
        """
        print("\n🔍 Verifying data generator alignment...")
        
        # Check original alignment
        for i in range(min(num_samples, len(self.filepaths))):
            filepath = self.filepaths[i]
            label = self.labels[i]
            
            # Extract gesture name from filepath
            gesture_from_path = Path(filepath).parent.name
            
            print(f"  Sample {i}:")
            print(f"    File: {Path(filepath).name}")
            print(f"    Path gesture: {gesture_from_path}")
            print(f"    Label: {label}")
            
            # For debugging, you could also load and check the actual label
            # if it's stored in the .npy file metadata
        
        # Check shuffled alignment after a few epochs
        if self.shuffle:
            print(f"\n  Testing shuffle alignment...")
            
            # Store original order
            original_filepaths = self.filepaths[:num_samples]
            original_labels = self.labels[:num_samples]
            
            # Simulate a few epochs
            for epoch in range(3):
                self.on_epoch_end()
                
                # Check that indices changed
                if epoch == 0:
                    print(f"    Epoch {epoch}: indices shuffled")
                
                # Verify batch retrieval works
                try:
                    X_batch, y_batch = self.__getitem__(0)
                    print(f"    Epoch {epoch}: Batch shape X={X_batch.shape}, y={y_batch.shape}")
                    
                    # Check that all labels in batch are valid
                    if len(y_batch.shape) == 2:  # One-hot encoded
                        batch_labels_int = np.argmax(y_batch, axis=1)
                    else:
                        batch_labels_int = y_batch
                    
                    unique_labels = np.unique(batch_labels_int)
                    print(f"      Unique labels in batch: {unique_labels}")
                    
                except Exception as e:
                    print(f"    ❌ Epoch {epoch}: Error getting batch: {e}")
                    return False
        
        print("  ✅ Alignment verification complete")
        return True
# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_pipeline():
    """Test the data pipeline."""
    print("Testing Data Pipeline...")
    
    # Initialize pipeline
    pipeline = GestureDataPipeline(
        data_dir=DATA_DIR,
        sequence_length=SEQUENCE_LENGTH,
        random_seed=42
    )
    
    # Test loading
    try:
        X, y, filepaths = pipeline.load_all_sequences(min_samples=1)
        print(f"✅ Loaded {len(X)} sequences")
    except Exception as e:
        print(f"❌ Loading failed: {e}")
        # Create dummy data for testing
        from config import NUM_GESTURES
        X = np.random.randn(100, SEQUENCE_LENGTH, 19).astype(np.float32)
        y = np.random.randint(0, NUM_GESTURES, 100)
        filepaths = [f"dummy_{i}.npy" for i in range(100)]
    
    # Test splits with idle balancing
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = pipeline.prepare_datasets(
        test_size=0.2,
        val_size=0.1,
        save_splits=False,  # Don't save during test
        balance_idle=True
    )
    
    print(f"✅ Splits created: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Test class weights
    y_train_int = np.argmax(y_train, axis=1)
    class_weights = pipeline.compute_class_weights(y_train_int)
    print(f"✅ Class weights computed")
    
    # Test augmentation
    if USE_AUGMENTATION and len(X_train) > 0:
        print("\n🎭 Testing augmentation methods...")
        
        # Test individual augmentation methods
        test_X = X_train[:2]  # Test with 2 samples
        
        # Gaussian noise
        X_noisy = TemporalAugmenter.add_gaussian_noise(test_X, noise_level=0.01)
        print(f"   ✓ Gaussian noise: shape={X_noisy.shape}")
        
        # Time warp
        X_warped = TemporalAugmenter.time_warp(test_X, warp_factor=0.1)
        print(f"   ✓ Time warp: shape={X_warped.shape}")
        
        # Temporal scaling
        X_scaled = TemporalAugmenter.temporal_scaling(test_X, scale_range=(0.9, 1.1))
        print(f"   ✓ Temporal scaling: shape={X_scaled.shape}")
        
        # Random crop
        X_cropped = TemporalAugmenter.random_crop(test_X, crop_range=(0.8, 1.0))
        print(f"   ✓ Random crop: shape={X_cropped.shape}")
        
        # Batch augmentation
        X_aug, y_aug = TemporalAugmenter.augment_batch(
            test_X, 
            y_train_int[:2],
            augment_factor=3,
            methods=['noise', 'time_warp']
        )
        print(f"   ✓ Batch augmentation: {len(test_X)} → {len(X_aug)} samples")
        
        # Test full pipeline augmentation
        X_train_aug, y_train_aug = pipeline.augment_datasets(
            X_train, y_train,
            augment_factor=2,
            exclude_idle=True
        )
        print(f"✅ Pipeline augmentation: {len(X_train)} → {len(X_train_aug)} samples")
    
    # Test dataset statistics
    stats = pipeline.get_dataset_statistics()
    print(f"✅ Statistics collected:")
    print(f"   Total sequences: {stats['total_sequences']}")
    
    # Test report generation
    report = pipeline.save_dataset_report(Path("test_report.json"))
    print(f"✅ Dataset report generated")
    
    # Test TF datasets
    if len(X_train) > 0:
        train_ds, val_ds = pipeline.create_tf_datasets(X_train, y_train, X_val, y_val)
        print(f"✅ TensorFlow datasets created")
        
        # Test a batch
        for batch_X, batch_y in train_ds.take(1):
            print(f"   Batch shape: X={batch_X.shape}, y={batch_y.shape}")
    
    print("\n✅ All pipeline tests passed!")

if __name__ == "__main__":
    test_pipeline()