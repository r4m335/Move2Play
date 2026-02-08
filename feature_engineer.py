# feature_engineer.py
"""
Robust feature engineering with occlusion handling, normalized distances, and proper velocity calculation.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    # Visibility thresholds
    min_visibility: float = 0.5  # Minimum visibility to consider a joint
    angle_visibility: float = 0.3  # Lower threshold for angles (more tolerant)
    
    # Normalization
    normalize_by_torso: bool = True  # Normalize distances by torso length
    reference_torso_length: float = 0.5  # Default torso length for fallback
    
    # Velocity calculation
    assume_fps: float = 30.0  # Assumed FPS for velocity calculation
    velocity_smoothing: float = 0.3  # EMA smoothing factor for velocities
    
    # Feature selection
    compute_angles: bool = True
    compute_distances: bool = True
    compute_torso_lean: bool = True
    compute_velocities: bool = True
    compute_height: bool = True  # Body height relative to torso

class FeatureEngineer:
    """
    Transforms raw landmarks into robust engineered features.
    Handles occlusion, normalizes by body scale, and computes proper velocities.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Args:
            config: Feature extraction configuration
        """
        self.config = config or FeatureConfig()
        
        # Landmark indices (MediaPipe Pose 33 landmarks)
        self.indices = {
            'nose': 0,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_pinky': 17, 'right_pinky': 18,
            'left_index': 19, 'right_index': 20,
            'left_thumb': 21, 'right_thumb': 22,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 30,
            'left_foot_index': 31, 'right_foot_index': 32,
        }
        
        # State for velocity smoothing
        self.prev_velocities: Optional[np.ndarray] = None
        self.prev_frame_time: Optional[float] = None
        
        # Cache for torso length
        self.cached_torso_length: Optional[float] = None
        
        print(f"✅ FeatureEngineer initialized")
        print(f"   Visibility thresholds: angle={self.config.angle_visibility}, "
              f"general={self.config.min_visibility}")
        print(f"   Normalize by torso: {self.config.normalize_by_torso}")
        print(f"   Assumed FPS: {self.config.assume_fps}")
    
    def extract_features(self, 
                        landmarks_sequence: np.ndarray,
                        timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract robust features from a sequence of landmarks.
        
        Args:
            landmarks_sequence: Shape (T, 33, 4) where 4 = [x, y, z, visibility]
            timestamps: Optional array of timestamps for each frame (seconds)
            
        Returns:
            Feature array of shape (T, F) where F = feature count
        """
        T = len(landmarks_sequence)
        features_list = []
        
        # Reset state for new sequence
        self.prev_velocities = None
        self.prev_frame_time = None
        self.cached_torso_length = None
        
        for t in range(T):
            frame_features = []
            landmarks = landmarks_sequence[t]
            
            # Get current timestamp (or estimate from FPS)
            if timestamps is not None and t < len(timestamps):
                current_time = timestamps[t]
                delta_time = current_time - self.prev_frame_time if self.prev_frame_time else 1.0/self.config.assume_fps
                self.prev_frame_time = current_time
            else:
                # Estimate time based on assumed FPS
                delta_time = 1.0 / self.config.assume_fps
            
            # Extract positions and visibility
            positions = landmarks[:, :3]
            visibility = landmarks[:, 3]
            
            # 1. Joint angles (with occlusion handling)
            if self.config.compute_angles:
                angles = self._calculate_joint_angles_robust(positions, visibility)
                frame_features.extend(angles)
            
            # 2. Normalized relative distances
            if self.config.compute_distances:
                torso_length = self._get_torso_length(positions, visibility)
                distances = self._calculate_normalized_distances(positions, visibility, torso_length)
                frame_features.extend(distances)
            
            # 3. Torso lean angle
            if self.config.compute_torso_lean:
                torso_angle = self._calculate_torso_lean_robust(positions, visibility)
                frame_features.append(torso_angle)
            
            # 4. Body height relative to torso
            if self.config.compute_height:
                height_ratio = self._calculate_body_height(positions, visibility, torso_length)
                frame_features.append(height_ratio)
            
            # 5. Limb velocities (proper time-based)
            if self.config.compute_velocities and t > 0:
                prev_landmarks = landmarks_sequence[t-1]
                velocities = self._calculate_velocities_time_based(
                    prev_landmarks, landmarks, delta_time
                )
                frame_features.extend(velocities)
                
                # Apply smoothing
                if self.prev_velocities is not None:
                    velocities = self.config.velocity_smoothing * velocities + \
                                (1 - self.config.velocity_smoothing) * self.prev_velocities
                
                self.prev_velocities = velocities
            else:
                # Pad with zeros for first frame
                velocity_features = 8  # 4 points * 2 (speed + direction)
                frame_features.extend([0.0] * velocity_features)
                self.prev_velocities = np.zeros(velocity_features)
            
            # 6. Visibility mask (binary indicators for key joints)
            visibility_features = self._extract_visibility_features(visibility)
            frame_features.extend(visibility_features)
            
            features_list.append(frame_features)
        
        features_array = np.array(features_list, dtype=np.float32)
        
        # Log feature statistics
        if len(features_list) > 0:
            self._log_feature_stats(features_array)
        
        return features_array
    
    def _get_torso_length(self, positions: np.ndarray, visibility: np.ndarray) -> float:
        """
        Calculate torso length (shoulder to hip center) with visibility handling.
        Caches result for efficiency.
        """
        if self.cached_torso_length is not None:
            return self.cached_torso_length
        
        left_shoulder_idx = self.indices['left_shoulder']
        right_shoulder_idx = self.indices['right_shoulder']
        left_hip_idx = self.indices['left_hip']
        right_hip_idx = self.indices['right_hip']
        
        # Calculate shoulder center (if visible)
        shoulder_center = np.zeros(3)
        shoulder_count = 0
        
        for idx in [left_shoulder_idx, right_shoulder_idx]:
            if visibility[idx] >= self.config.min_visibility:
                shoulder_center += positions[idx]
                shoulder_count += 1
        
        if shoulder_count == 0:
            # Fallback: use default torso length
            self.cached_torso_length = self.config.reference_torso_length
            return self.cached_torso_length
        
        shoulder_center /= shoulder_count
        
        # Calculate hip center (if visible)
        hip_center = np.zeros(3)
        hip_count = 0
        
        for idx in [left_hip_idx, right_hip_idx]:
            if visibility[idx] >= self.config.min_visibility:
                hip_center += positions[idx]
                hip_count += 1
        
        if hip_count == 0:
            # Fallback: use shoulder position with offset
            hip_center = shoulder_center - np.array([0, 0.3, 0])  # Approximate offset
        else:
            hip_center /= hip_count
        
        # Calculate torso length
        torso_vector = shoulder_center - hip_center
        torso_length = np.linalg.norm(torso_vector)
        
        # Cache and return
        self.cached_torso_length = max(torso_length, 0.01)  # Avoid zero
        return self.cached_torso_length
    
    def _calculate_joint_angles_robust(self, positions: np.ndarray, visibility: np.ndarray) -> List[float]:
        """
        Calculate joint angles with occlusion handling.
        Returns 0 for angles involving occluded joints.
        """
        angles = []
        
        # Define joint triplets (proximal, joint, distal)
        joint_triplets = [
            # Elbow angles
            ('left_shoulder', 'left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow', 'right_wrist'),
            # Knee angles
            ('left_hip', 'left_knee', 'left_ankle'),
            ('right_hip', 'right_knee', 'right_ankle'),
            # Shoulder angles (relative to torso)
            ('left_elbow', 'left_shoulder', 'right_shoulder'),
            ('right_elbow', 'right_shoulder', 'left_shoulder'),
            # Hip angles
            ('left_knee', 'left_hip', 'right_hip'),
            ('right_knee', 'right_hip', 'left_hip'),
        ]
        
        for proximal_name, joint_name, distal_name in joint_triplets:
            proximal_idx = self.indices[proximal_name]
            joint_idx = self.indices[joint_name]
            distal_idx = self.indices[distal_name]
            
            # Check visibility (more tolerant for angles)
            if (visibility[proximal_idx] >= self.config.angle_visibility and
                visibility[joint_idx] >= self.config.angle_visibility and
                visibility[distal_idx] >= self.config.angle_visibility):
                
                v1 = positions[proximal_idx] - positions[joint_idx]
                v2 = positions[distal_idx] - positions[joint_idx]
                
                angle = self._angle_between_vectors(v1, v2)
                angles.append(angle)
            else:
                # Joint is occluded, use 0 (neutral position)
                angles.append(0.0)
        
        return angles
    
    def _calculate_normalized_distances(self, 
                                       positions: np.ndarray, 
                                       visibility: np.ndarray,
                                       torso_length: float) -> List[float]:
        """
        Calculate distances between key points, normalized by torso length.
        """
        distances = []
        
        # Define distance pairs to calculate
        distance_pairs = [
            # Hand to shoulder
            ('left_wrist', 'left_shoulder'),
            ('right_wrist', 'right_shoulder'),
            # Hand to opposite shoulder (cross-body)
            ('left_wrist', 'right_shoulder'),
            ('right_wrist', 'left_shoulder'),
            # Foot to hip
            ('left_ankle', 'left_hip'),
            ('right_ankle', 'right_hip'),
            # Foot to opposite hip
            ('left_ankle', 'right_hip'),
            ('right_ankle', 'left_hip'),
            # Hand to hand
            ('left_wrist', 'right_wrist'),
            # Foot to foot
            ('left_ankle', 'right_ankle'),
            # Shoulder width
            ('left_shoulder', 'right_shoulder'),
            # Hip width
            ('left_hip', 'right_hip'),
        ]
        
        for point1_name, point2_name in distance_pairs:
            idx1 = self.indices[point1_name]
            idx2 = self.indices[point2_name]
            
            if (visibility[idx1] >= self.config.min_visibility and
                visibility[idx2] >= self.config.min_visibility):
                
                # Calculate raw distance
                raw_distance = np.linalg.norm(positions[idx1] - positions[idx2])
                
                # Normalize by torso length
                if self.config.normalize_by_torso and torso_length > 0:
                    normalized_distance = raw_distance / torso_length
                else:
                    normalized_distance = raw_distance
                
                distances.append(normalized_distance)
            else:
                # Joints occluded, use 0 distance
                distances.append(0.0)
        
        return distances
    
    def _calculate_torso_lean_robust(self, positions: np.ndarray, visibility: np.ndarray) -> float:
        """
        Calculate torso lean angle with occlusion handling.
        Returns 0 if key joints are not visible.
        """
        left_shoulder_idx = self.indices['left_shoulder']
        right_shoulder_idx = self.indices['right_shoulder']
        left_hip_idx = self.indices['left_hip']
        right_hip_idx = self.indices['right_hip']
        
        # Need at least one shoulder and one hip visible
        shoulder_visible = (visibility[left_shoulder_idx] >= self.config.min_visibility or
                           visibility[right_shoulder_idx] >= self.config.min_visibility)
        hip_visible = (visibility[left_hip_idx] >= self.config.min_visibility or
                      visibility[right_hip_idx] >= self.config.min_visibility)
        
        if not shoulder_visible or not hip_visible:
            return 0.0
        
        # Calculate centers using visible points
        shoulder_center = np.zeros(3)
        shoulder_count = 0
        for idx in [left_shoulder_idx, right_shoulder_idx]:
            if visibility[idx] >= self.config.min_visibility:
                shoulder_center += positions[idx]
                shoulder_count += 1
        shoulder_center /= max(shoulder_count, 1)
        
        hip_center = np.zeros(3)
        hip_count = 0
        for idx in [left_hip_idx, right_hip_idx]:
            if visibility[idx] >= self.config.min_visibility:
                hip_center += positions[idx]
                hip_count += 1
        hip_center /= max(hip_count, 1)
        
        # Calculate torso vector
        torso_vector = shoulder_center - hip_center
        
        # Project to frontal plane (x-z) for side-to-side lean
        torso_2d = np.array([torso_vector[0], torso_vector[2]])
        
        if np.linalg.norm(torso_2d) < 1e-6:
            return 0.0
        
        # Reference vertical vector in frontal plane
        vertical_2d = np.array([0, 1])  # Pointing forward
        
        # Calculate angle
        cos_angle = np.dot(torso_2d, vertical_2d) / (np.linalg.norm(torso_2d) * np.linalg.norm(vertical_2d))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        # Determine direction (positive = lean right, negative = lean left)
        if torso_vector[0] > 0:
            angle = -angle
        
        return angle
    
    def _calculate_body_height(self, 
                              positions: np.ndarray, 
                              visibility: np.ndarray,
                              torso_length: float) -> float:
        """
        Calculate approximate body height relative to torso length.
        Uses highest visible point (head/shoulders) and lowest visible point (feet).
        """
        # Head points (nose is highest typically visible point)
        head_indices = [self.indices['nose']]
        
        # Foot points
        foot_indices = [
            self.indices['left_ankle'], self.indices['right_ankle'],
            self.indices['left_heel'], self.indices['right_heel'],
        ]
        
        # Find highest visible head point
        head_height = -float('inf')
        for idx in head_indices:
            if visibility[idx] >= self.config.min_visibility:
                head_height = max(head_height, positions[idx, 1])  # y-coordinate
        
        # Find lowest visible foot point
        foot_height = float('inf')
        for idx in foot_indices:
            if visibility[idx] >= self.config.min_visibility:
                foot_height = min(foot_height, positions[idx, 1])  # y-coordinate
        
        # Calculate height
        if head_height > -float('inf') and foot_height < float('inf'):
            body_height = abs(head_height - foot_height)
            
            # Normalize by torso length
            if self.config.normalize_by_torso and torso_length > 0:
                return body_height / torso_length
            else:
                return body_height
        else:
            # Not enough points visible
            return 0.0
    
    def _calculate_velocities_time_based(self,
                                        prev_landmarks: np.ndarray,
                                        curr_landmarks: np.ndarray,
                                        delta_time: float) -> List[float]:
        """
        Calculate velocities of key points with proper time scaling.
        Returns both speed and directional components.
        """
        velocities = []
        key_points = [
            'left_wrist', 'right_wrist',
            'left_ankle', 'right_ankle',
            'left_shoulder', 'right_shoulder',
            'left_hip', 'right_hip'
        ]
        
        for point in key_points:
            idx = self.indices[point]
            
            # Extract positions and visibility
            prev_pos = prev_landmarks[idx, :3]
            curr_pos = curr_landmarks[idx, :3]
            prev_vis = prev_landmarks[idx, 3]
            curr_vis = curr_landmarks[idx, 3]
            
            # Only calculate if both frames have reasonable visibility
            if prev_vis >= self.config.min_visibility and curr_vis >= self.config.min_visibility:
                # Calculate displacement
                displacement = curr_pos - prev_pos
                
                # Scale by time to get velocity
                if delta_time > 0:
                    velocity = displacement / delta_time
                else:
                    velocity = displacement * self.config.assume_fps
                
                # Speed (magnitude)
                speed = np.linalg.norm(velocity)
                
                # Normalized directional components (x, y)
                if speed > 1e-6:
                    dir_x = velocity[0] / speed
                    dir_y = velocity[1] / speed
                else:
                    dir_x = 0.0
                    dir_y = 0.0
                
                velocities.extend([speed, dir_x, dir_y])
            else:
                # Point occluded, use zeros
                velocities.extend([0.0, 0.0, 0.0])
        
        return velocities
    
    def _extract_visibility_features(self, visibility: np.ndarray) -> List[float]:
        """
        Extract binary visibility indicators for key joints.
        Helps model learn to handle occlusion.
        """
        key_joints = [
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
        ]
        
        visibility_features = []
        for joint in key_joints:
            idx = self.indices[joint]
            is_visible = 1.0 if visibility[idx] >= self.config.min_visibility else 0.0
            visibility_features.append(is_visible)
        
        # Also add overall visibility score
        overall_visibility = np.mean([visibility[self.indices[joint]] for joint in key_joints])
        visibility_features.append(overall_visibility)
        
        return visibility_features
    
    def _angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors in radians."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)
    
    def _log_feature_stats(self, features: np.ndarray):
        """Log basic statistics about extracted features."""
        if len(features) == 0:
            return
        
        print(f"📊 Feature statistics:")
        print(f"   Shape: {features.shape}")
        print(f"   Mean: {np.mean(features):.4f}")
        print(f"   Std: {np.std(features):.4f}")
        print(f"   Min: {np.min(features):.4f}")
        print(f"   Max: {np.max(features):.4f}")
        
        # Check for NaN or Inf
        has_nan = np.any(np.isnan(features))
        has_inf = np.any(np.isinf(features))
        
        if has_nan or has_inf:
            print(f"   ⚠️  WARNING: Features contain NaN/Inf values!")
            if has_nan:
                nan_count = np.sum(np.isnan(features))
                print(f"      NaN count: {nan_count}")
            if has_inf:
                inf_count = np.sum(np.isinf(features))
                print(f"      Inf count: {inf_count}")
    
    def get_feature_dimension(self) -> int:
        """
        Calculate the expected feature dimension.
        Useful for model initialization.
        """
        # Create dummy landmarks
        dummy_landmarks = np.zeros((1, 33, 4), dtype=np.float32)
        dummy_landmarks[:, :, 3] = 1.0  # All visible
        
        # Extract features for single frame
        features = self.extract_features(dummy_landmarks)
        
        return features.shape[1]
    
    def validate_features(self, features: np.ndarray) -> bool:
        """
        Validate extracted features for common issues.
        
        Returns:
            True if features are valid
        """
        if len(features) == 0:
            print("❌ No features extracted")
            return False
        
        # Check for NaN
        if np.any(np.isnan(features)):
            print("❌ Features contain NaN values")
            return False
        
        # Check for Inf
        if np.any(np.isinf(features)):
            print("❌ Features contain Inf values")
            return False
        
        # Check for extreme values (likely errors)
        if np.any(np.abs(features) > 1000):
            print("⚠️  Features contain extreme values (>1000)")
            # Not necessarily invalid, but worth noting
        
        # Check feature dimension consistency
        expected_dim = self.get_feature_dimension()
        if features.shape[1] != expected_dim:
            print(f"❌ Feature dimension mismatch: expected {expected_dim}, got {features.shape[1]}")
            return False
        
        return True

# ============================================================================
# FEATURE NORMALIZATION
# ============================================================================

class FeatureNormalizer:
    """
    Normalizes features to consistent range for better model training.
    Can fit on training data and transform new data.
    """
    
    def __init__(self, method: str = 'standard'):
        """
        Args:
            method: 'standard' (z-score), 'minmax', or 'robust' (median/IQR)
        """
        self.method = method
        self.fitted = False
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.median = None
        self.iqr = None
        
    def fit(self, X: np.ndarray):
        """Fit normalizer on training data."""
        if len(X.shape) != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        
        if self.method == 'standard':
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            # Avoid division by zero
            self.std = np.where(self.std < 1e-8, 1.0, self.std)
            
        elif self.method == 'minmax':
            self.min = np.min(X, axis=0)
            self.max = np.max(X, axis=0)
            # Avoid division by zero
            range_vals = self.max - self.min
            range_vals = np.where(range_vals < 1e-8, 1.0, range_vals)
            self.range = range_vals
            
        elif self.method == 'robust':
            self.median = np.median(X, axis=0)
            q75 = np.percentile(X, 75, axis=0)
            q25 = np.percentile(X, 25, axis=0)
            self.iqr = q75 - q25
            # Avoid division by zero
            self.iqr = np.where(self.iqr < 1e-8, 1.0, self.iqr)
        
        self.fitted = True
        print(f"✅ FeatureNormalizer fitted with method: {self.method}")
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters."""
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before transformation")
        
        if self.method == 'standard':
            return (X - self.mean) / self.std
        elif self.method == 'minmax':
            return (X - self.min) / self.range
        elif self.method == 'robust':
            return (X - self.median) / self.iqr
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_normalized: np.ndarray) -> np.ndarray:
        """Inverse transform back to original scale."""
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before inverse transformation")
        
        if self.method == 'standard':
            return X_normalized * self.std + self.mean
        elif self.method == 'minmax':
            return X_normalized * self.range + self.min
        elif self.method == 'robust':
            return X_normalized * self.iqr + self.median
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_feature_engineering():
    """Test the feature engineering pipeline."""
    print("Testing Feature Engineering...")
    
    # Create test landmarks (2 frames, 33 landmarks, 4 features each)
    np.random.seed(42)
    T = 2
    landmarks = np.random.randn(T, 33, 4).astype(np.float32)
    
    # Set visibility (some joints occluded)
    landmarks[:, :, 3] = np.random.uniform(0.3, 1.0, (T, 33))
    
    # Test with default config
    engineer = FeatureEngineer()
    features = engineer.extract_features(landmarks)
    
    print(f"✅ Features extracted: shape {features.shape}")
    print(f"   Expected dimension: {engineer.get_feature_dimension()}")
    
    # Validate features
    is_valid = engineer.validate_features(features)
    print(f"✅ Feature validation: {is_valid}")
    
    # Test feature normalizer
    normalizer = FeatureNormalizer(method='standard')
    
    # Create more data for fitting
    more_features = np.random.randn(100, features.shape[1])
    normalized = normalizer.fit_transform(more_features)
    
    print(f"✅ Feature normalization:")
    print(f"   Original mean: {np.mean(more_features, axis=0)[:3]}")
    print(f"   Normalized mean: {np.mean(normalized, axis=0)[:3]}")
    print(f"   Normalized std: {np.std(normalized, axis=0)[:3]}")
    
    # Test with custom config
    config = FeatureConfig(
        min_visibility=0.7,
        normalize_by_torso=True,
        assume_fps=20.0,
        compute_angles=True,
        compute_velocities=True
    )
    
    custom_engineer = FeatureEngineer(config)
    custom_features = custom_engineer.extract_features(landmarks)
    
    print(f"\n✅ Custom config test:")
    print(f"   Feature shape: {custom_features.shape}")
    
    # Test with timestamps
    timestamps = np.array([0.0, 0.05])  # 20 FPS
    timed_features = engineer.extract_features(landmarks, timestamps)
    print(f"✅ Timed features extracted: shape {timed_features.shape}")
    
    print("\n✅ All feature engineering tests passed!")

if __name__ == "__main__":
    test_feature_engineering()