# data_pipeline.py
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

# Import from config
from config import (
    DATA_DIR, SEQUENCE_LENGTH, GESTURE_CLASSES,
    MIN_SAMPLES_PER_GESTURE, BATCH_SIZE, 
    USE_AUGMENTATION, AUGMENTATION_FACTOR
)
from feature_engineer import FeatureEngineer

# ============================================================================
# TEMPORAL AUGMENTATION
# ============================================================================

class TemporalAugmenter:
    """
    Augmentation techniques specifically designed for temporal sequences.
    Applies transformations in the time domain, not spatial domain.
    """
    
    @staticmethod
    def add_gaussian_noise(X: np.ndarray, 
                          noise_level: float = 0.01) -> np.ndarray:
        """
        Add Gaussian noise to sequences.
        
        Args:
            X: Input sequences (batch, time, features)
            noise_level: Standard deviation of Gaussian noise
            
        Returns:
            Augmented sequences with noise
        """
        noise = np.random.normal(0, noise_level, X.shape).astype(np.float32)
        return X + noise
    
    @staticmethod
    def time_warp(X: np.ndarray, 
                 warp_factor: float = 0.1) -> np.ndarray:
        """
        Apply time warping by varying speed along the sequence.
        
        Args:
            X: Input sequences (batch, time, features)
            warp_factor: Maximum warp factor (0.0-0.3)
            
        Returns:
            Time-warped sequences
        """
        batch_size, seq_len, n_features = X.shape
        
        # Generate random warp for each sequence
        warped_sequences = []
        for i in range(batch_size):
            # Create non-linear time mapping
            original_time = np.linspace(0, 1, seq_len)
            
            # Random warp points
            warp_points = np.random.uniform(-warp_factor, warp_factor, 3)
            warp_points = np.cumsum(warp_points)  # Ensure monotonic
            
            # Interpolate warp
            warped_time = original_time + np.interp(
                original_time, 
                np.linspace(0, 1, len(warp_points)), 
                warp_points
            )
            
            # Clip to [0, 1] and rescale
            warped_time = np.clip(warped_time, 0, 1)
            warped_time = warped_time / warped_time[-1] if warped_time[-1] > 0 else warped_time
            
            # Resample sequence
            warped_seq = np.zeros((seq_len, n_features))
            for f in range(n_features):
                warped_seq[:, f] = np.interp(
                    np.linspace(0, 1, seq_len),
                    warped_time,
                    X[i, :, f]
                )
            
            warped_sequences.append(warped_seq)
        
        return np.array(warped_sequences, dtype=np.float32)
    
    @staticmethod
    def temporal_scaling(X: np.ndarray, 
                        scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """
        Scale sequences in time dimension (speed up/slow down).
        
        Args:
            X: Input sequences (batch, time, features)
            scale_range: Min and max scaling factors
            
        Returns:
            Temporally scaled sequences
        """
        batch_size, seq_len, n_features = X.shape
        
        scaled_sequences = []
        for i in range(batch_size):
            # Random scale factor
            scale = np.random.uniform(scale_range[0], scale_range[1])
            
            # Create new time axis
            if scale < 1.0:  # Speed up (shorter)
                new_len = max(int(seq_len * scale), 1)
                # Sample fewer points
                indices = np.linspace(0, seq_len - 1, new_len).astype(int)
                scaled_seq = X[i, indices, :]
                # Pad back to original length
                pad_len = seq_len - new_len
                scaled_seq = np.pad(scaled_seq, ((0, pad_len), (0, 0)), mode='edge')
            else:  # Slow down (longer)
                new_len = int(seq_len * scale)
                # Resample with linear interpolation
                scaled_seq = np.zeros((new_len, n_features))
                for f in range(n_features):
                    scaled_seq[:, f] = np.interp(
                        np.linspace(0, seq_len - 1, new_len),
                        np.arange(seq_len),
                        X[i, :, f]
                    )
                # Crop to original length
                scaled_seq = scaled_seq[:seq_len, :]
            
            scaled_sequences.append(scaled_seq)
        
        return np.array(scaled_sequences, dtype=np.float32)
    
    @staticmethod
    def random_crop(X: np.ndarray, 
                   crop_range: Tuple[float, float] = (0.8, 1.0)) -> np.ndarray:
        """
        Randomly crop sequences and resize back to original length.
        
        Args:
            X: Input sequences (batch, time, features)
            crop_range: Min and max crop factors
            
        Returns:
            Randomly cropped and resized sequences
        """
        batch_size, seq_len, n_features = X.shape
        
        cropped_sequences = []
        for i in range(batch_size):
            # Random crop factor
            crop_factor = np.random.uniform(crop_range[0], crop_range[1])
            crop_len = int(seq_len * crop_factor)
            
            # Random start position
            start = np.random.randint(0, seq_len - crop_len + 1)
            
            # Crop
            cropped = X[i, start:start + crop_len, :]
            
            # Resize back to original length with linear interpolation
            resized = np.zeros((seq_len, n_features))
            for f in range(n_features):
                resized[:, f] = np.interp(
                    np.linspace(0, crop_len - 1, seq_len),
                    np.arange(crop_len),
                    cropped[:, f]
                )
            
            cropped_sequences.append(resized)
        
        return np.array(cropped_sequences, dtype=np.float32)
    
    @staticmethod
    def augment_batch(X: np.ndarray, 
                     y: np.ndarray,
                     augment_factor: int = 2,
                     methods: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply multiple augmentation methods to a batch.
        
        Args:
            X: Input sequences
            y: Labels
            augment_factor: How many augmented copies to create
            methods: List of augmentation methods to apply
            
        Returns:
            Augmented X and y
        """
        if methods is None:
            methods = ['noise', 'scaling', 'crop']
        
        X_augmented = [X]
        y_augmented = [y]
        
        for _ in range(augment_factor - 1):
            X_aug = X.copy()
            
            # Apply random subset of augmentations
            selected_methods = np.random.choice(
                methods, 
                size=np.random.randint(1, len(methods) + 1),
                replace=False
            )
            
            for method in selected_methods:
                if method == 'noise':
                    X_aug = TemporalAugmenter.add_gaussian_noise(X_aug, noise_level=0.005)
                elif method == 'scaling':
                    X_aug = TemporalAugmenter.temporal_scaling(X_aug, scale_range=(0.9, 1.1))
                elif method == 'crop':
                    X_aug = TemporalAugmenter.random_crop(X_aug, crop_range=(0.85, 0.95))
                elif method == 'warp':
                    X_aug = TemporalAugmenter.time_warp(X_aug, warp_factor=0.1)
            
            X_augmented.append(X_aug)
            y_augmented.append(y)
        
        X_combined = np.concatenate(X_augmented, axis=0)
        y_combined = np.concatenate(y_augmented, axis=0)
        
        # Shuffle
        indices = np.random.permutation(len(X_combined))
        return X_combined[indices], y_combined[indices]

# ============================================================================
# MAIN DATA PIPELINE
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
                        save_splits: bool = True) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Split data into train, validation, and test sets.
        Saves split indices for reproducibility.
        
        Args:
            test_size: Proportion of data for test set
            val_size: Proportion of data for validation set (of training data after test split)
            stratify: Whether to stratify splits by class
            save_splits: Whether to save split indices to file
            
        Returns:
            Tuples of (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        # Load data
        X, y, filepaths = self.load_all_sequences()
        
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
        
        return (X_train, y_train_onehot), (X_val, y_val_onehot), (X_test, y_test_onehot)
    
    def compute_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights for imbalanced datasets.
        
        Args:
            y_train: Training labels (integer, not one-hot)
            
        Returns:
            Dictionary mapping class indices to weights
        """
        # Convert one-hot to integer if needed
        if len(y_train.shape) == 2:
            y_train_int = np.argmax(y_train, axis=1)
        else:
            y_train_int = y_train
        
        # Compute class weights
        unique_classes = np.unique(y_train_int)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y_train_int
        )
        
        # Create dictionary
        weight_dict = {int(cls): float(weight) for cls, weight in zip(unique_classes, class_weights)}
        
        print("\n✅ Class weights computed:")
        for cls_idx, weight in weight_dict.items():
            cls_name = self.idx_to_class[cls_idx]
            count = np.sum(y_train_int == cls_idx)
            print(f"   {cls_name:15s}: weight={weight:.2f}, samples={count}")
        
        return weight_dict
    
    def augment_datasets(self, 
                        X_train: np.ndarray, 
                        y_train: np.ndarray,
                        augment_factor: int = AUGMENTATION_FACTOR,
                        methods: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply temporal augmentation to training data.
        
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
        
        # Augment
        X_augmented, y_augmented_int = TemporalAugmenter.augment_batch(
            X_train, y_train_int,
            augment_factor=augment_factor,
            methods=methods
        )
        
        # Convert back to one-hot
        y_augmented = tf.keras.utils.to_categorical(y_augmented_int, len(self.gesture_classes))
        
        print(f"✅ Augmentation complete:")
        print(f"   Before: {len(X_train)} samples")
        print(f"   After:  {len(X_augmented)} samples")
        print(f"   Multiplier: {len(X_augmented) / len(X_train):.1f}x")
        
        return X_augmented, y_augmented
    
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
        }
        
        # Gesture distribution
        unique, counts = np.unique(y, return_counts=True)
        for cls_idx, count in zip(unique, counts):
            cls_name = self.idx_to_class[cls_idx]
            stats['gesture_distribution'][cls_name] = {
                'count': int(count),
                'percentage': float(count / len(y) * 100)
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
            },
            'statistics': self.get_dataset_statistics(),
            'split_indices': self.split_indices,
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Dataset report saved to {output_path}")
        
        return report

# ============================================================================
# BATCH GENERATOR FOR LARGE DATASETS
# ============================================================================

class GestureDataGenerator(keras.utils.Sequence):
    """
    Custom data generator for large datasets that don't fit in memory.
    Loads and processes sequences on-the-fly.
    """
    
    def __init__(self, 
                 filepaths: List[str],
                 labels: np.ndarray,
                 batch_size: int = BATCH_SIZE,
                 sequence_length: int = SEQUENCE_LENGTH,
                 shuffle: bool = True,
                 augment: bool = False):
        """
        Args:
            filepaths: List of paths to .npy files
            labels: Corresponding labels (one-hot encoded)
            batch_size: Batch size
            sequence_length: Fixed sequence length
            shuffle: Whether to shuffle data each epoch
            augment: Whether to apply augmentation
        """
        self.filepaths = filepaths
        self.labels = labels
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.shuffle = shuffle
        self.augment = augment
        self.feature_engineer = FeatureEngineer()
        
        self.on_epoch_end()
    
    def __len__(self) -> int:
        """Number of batches per epoch."""
        return int(np.ceil(len(self.filepaths) / self.batch_size))
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get batch at index."""
        batch_start = idx * self.batch_size
        batch_end = min(batch_start + self.batch_size, len(self.filepaths))
        
        batch_filepaths = self.filepaths[batch_start:batch_end]
        batch_labels = self.labels[batch_start:batch_end]
        
        batch_X = []
        for filepath in batch_filepaths:
            # Load landmarks
            landmarks = np.load(filepath)
            
            # Process sequence length
            if landmarks.shape[0] > self.sequence_length:
                landmarks = landmarks[:self.sequence_length]
            elif landmarks.shape[0] < self.sequence_length:
                pad_amount = self.sequence_length - landmarks.shape[0]
                landmarks = np.pad(landmarks, ((0, pad_amount), (0, 0), (0, 0)), mode='constant')
            
            # Extract features
            features = self.feature_engineer.extract_features(landmarks)
            batch_X.append(features)
        
        batch_X = np.array(batch_X, dtype=np.float32)
        
        # Apply augmentation if enabled
        if self.augment and len(batch_X) > 0:
            batch_X = TemporalAugmenter.add_gaussian_noise(batch_X, noise_level=0.005)
        
        return batch_X, batch_labels
    
    def on_epoch_end(self):
        """Shuffle data at the end of each epoch."""
        if self.shuffle:
            indices = np.arange(len(self.filepaths))
            np.random.shuffle(indices)
            self.filepaths = [self.filepaths[i] for i in indices]
            self.labels = self.labels[indices]

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
        X = np.random.randn(100, SEQUENCE_LENGTH, 19).astype(np.float32)
        y = np.random.randint(0, NUM_GESTURES, 100)
        filepaths = [f"dummy_{i}.npy" for i in range(100)]
    
    # Test splits
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = pipeline.prepare_datasets(
        test_size=0.2,
        val_size=0.1,
        save_splits=True
    )
    
    print(f"✅ Splits created: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Test class weights
    y_train_int = np.argmax(y_train, axis=1)
    class_weights = pipeline.compute_class_weights(y_train_int)
    print(f"✅ Class weights computed: {class_weights}")
    
    # Test augmentation
    if USE_AUGMENTATION and len(X_train) > 0:
        X_train_aug, y_train_aug = pipeline.augment_datasets(
            X_train, y_train,
            augment_factor=2
        )
        print(f"✅ Augmentation applied: {len(X_train_aug)} samples")
    
    # Test dataset statistics
    stats = pipeline.get_dataset_statistics()
    print(f"✅ Statistics collected:")
    print(f"   Total sequences: {stats['total_sequences']}")
    
    # Test report generation
    report = pipeline.save_dataset_report()
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