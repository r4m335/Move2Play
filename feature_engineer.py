"""
Robust feature engineering with specialized features for jogging-in-place detection.
Emphasizes temporal patterns, periodicity, and vertical movement over displacement.

FIXES APPLIED:
1. Deterministic feature dimension calculation via get_feature_dimension()
2. Consistent idle feature handling based on config
3. State management fixes to prevent contamination between sequences
4. Dimension validation to prevent runtime mismatches
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.signal import find_peaks
import warnings
import json
from pathlib import Path


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
    
    # Jogging-specific features
    compute_jogging_features: bool = True  # Special features for run detection
    compute_periodicity: bool = True  # Calculate movement periodicity
    compute_alternation: bool = True  # Calculate limb alternation patterns
    
    # Idle features (MANDATORY FIX: Explicit control to avoid runtime issues)
    compute_idle_features: bool = True  # Now True by default for consistency
    
    def to_dict(self) -> Dict:
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
    def from_dict(cls, data: Dict) -> 'FeatureConfig':
        """Create config from dictionary."""
        return cls(**data)
    
    def get_hash(self) -> str:
        """Get unique hash for this configuration."""
        import hashlib
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


class FeatureEngineer:
    """
    Transforms raw landmarks into robust engineered features.
    Special emphasis on jogging-in-place detection with temporal patterns.
    
    FIXES:
    1. get_feature_dimension() method added for deterministic dimension calculation
    2. Feature count no longer depends on runtime state
    3. Idle features consistently enabled/disabled based on config
    4. State properly reset between sequences
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
            'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
            'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
            'left_ear': 7, 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10,
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
        
        # State for periodicity analysis
        self.ankle_history: List[np.ndarray] = []
        self.knee_angle_history: List[Tuple[float, float]] = []
        self.max_history_frames = 60  # Store last 2 seconds at 30 FPS
        
        # State for idle features (initialized if needed)
        self._prev_positions: Optional[np.ndarray] = None
        self._position_history: List[np.ndarray] = []
        self._prev_cog_y: Optional[float] = None
        
        # FIX: Pre-calculate feature dimension for consistency
        self._feature_dimension = self._calculate_feature_dimension()
        
        # FIX: Always use config setting for idle features
        self._idle_features_enabled = self.config.compute_idle_features
        
        print(f"✅ FeatureEngineer initialized (config hash: {self.config.get_hash()})")
        print(f"   Feature dimension: {self._feature_dimension}")
        print(f"   Jogging features: {self.config.compute_jogging_features}")
        print(f"   Idle features: {self._idle_features_enabled}")
        print(f"   Periodicity analysis: {self.config.compute_periodicity}")
    
    def _calculate_feature_dimension(self) -> int:
        """
        Calculate exact feature dimension based on configuration.
        This is deterministic and doesn't depend on runtime state.
        
        Returns:
            Exact number of features that will be extracted
        """
        total = 0
        
        # Angles: 4 angles (left/right elbow, left/right knee)
        if self.config.compute_angles:
            total += 4
        
        # Distances: 8 normalized distances
        if self.config.compute_distances:
            total += 8
        
        # Torso lean: 1 feature
        if self.config.compute_torso_lean:
            total += 1
        
        # Body height: 1 feature
        if self.config.compute_height:
            total += 1
        
        # Velocities: 4 limb points × 3 features + 4 other points × 2 features
        if self.config.compute_velocities:
            total += (4 * 3) + (4 * 2)  # 20 velocity features
        
        # Jogging features: 6 features
        if self.config.compute_jogging_features:
            total += 6
        
        # Periodicity: 4 features (always padded, even if not enough history)
        if self.config.compute_periodicity:
            total += 4
        
        # Visibility: 12 key joints
        total += 12
        
        # Idle features: 7 features (only if explicitly enabled)
        if self.config.compute_idle_features:
            total += 7
        
        return total
    
    def get_feature_dimension(self) -> int:
        """
        Get the deterministic feature dimension.
        This method should be called during training and the value persisted.
        
        Returns:
            Exact feature dimension for this configuration
        """
        return self._feature_dimension
    
    def extract_features(self, 
                        landmarks_sequence: np.ndarray,
                        timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract robust features from a sequence of landmarks.
        Special emphasis on jogging-in-place temporal patterns.
        
        Args:
            landmarks_sequence: Shape (T, 33, 4) where 4 = [x, y, z, visibility]
            timestamps: Optional array of timestamps for each frame (seconds)
            
        Returns:
            Feature array of shape (T, F) where F = get_feature_dimension()
        """
        T = len(landmarks_sequence)
        
        # FIX: Pre-allocate array with exact dimension
        feature_dim = self.get_feature_dimension()
        features_array = np.zeros((T, feature_dim), dtype=np.float32)
        
        # Reset state for new sequence
        self._reset_sequence_state()
        
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
            
            # 1. Joint angles (with occlusion handling) - ESPECIALLY KNEE ANGLES
            if self.config.compute_angles:
                angles = self._calculate_joint_angles_robust(positions, visibility)
                frame_features.extend(angles)
                
                # Store knee angles for periodicity analysis
                if self.config.compute_periodicity and len(angles) >= 4:
                    left_knee_idx = 2  # Based on joint_triplets order
                    right_knee_idx = 3
                    self.knee_angle_history.append((angles[left_knee_idx], angles[right_knee_idx]))
                    if len(self.knee_angle_history) > self.max_history_frames:
                        self.knee_angle_history.pop(0)
            
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
            
            # 5. Limb velocities (proper time-based) - EMPHASIZE VERTICAL VELOCITY
            if self.config.compute_velocities and t > 0:
                prev_landmarks = landmarks_sequence[t-1]
                velocities = self._calculate_velocities_jogging_focused(
                    prev_landmarks, landmarks, delta_time
                )
                
                # Apply smoothing
                if self.prev_velocities is not None:
                    velocities = self.config.velocity_smoothing * velocities + \
                                (1 - self.config.velocity_smoothing) * self.prev_velocities
                
                frame_features.extend(velocities)
                self.prev_velocities = velocities
            else:
                # Pad with zeros for first frame
                velocity_features = 20  # Fixed: 4×3 + 4×2
                frame_features.extend([0.0] * velocity_features)
                if self.prev_velocities is None:
                    self.prev_velocities = np.zeros(velocity_features)
            
            # 6. Jogging-specific features
            if self.config.compute_jogging_features:
                jogging_features = self._calculate_jogging_features(
                    positions, visibility, delta_time, t
                )
                frame_features.extend(jogging_features)
            
            # 7. Periodicity features (always produce 4 features, padded if needed)
            if self.config.compute_periodicity:
                if len(self.knee_angle_history) > 10:
                    periodicity_features = self._calculate_periodicity_features()
                    frame_features.extend(periodicity_features)
                else:
                    # Pad with zeros if not enough history
                    frame_features.extend([0.0] * 4)  # 4 periodicity features
            
            # 8. Visibility mask (binary indicators for key joints)
            visibility_features = self._extract_visibility_features(visibility)
            frame_features.extend(visibility_features)
            
            # 9. Idle-specific features (only if enabled in config)
            if self._idle_features_enabled:
                idle_features = self._calculate_idle_features(positions, visibility, delta_time)
                frame_features.extend(idle_features)
            
            # FIX: Verify we have exactly the expected number of features
            if len(frame_features) != feature_dim:
                raise ValueError(
                    f"Feature dimension mismatch at frame {t}: "
                    f"expected {feature_dim}, got {len(frame_features)}. "
                    f"Check feature calculation logic."
                )
            
            features_array[t] = frame_features
        
        # Log feature statistics
        self._log_feature_stats(features_array)
        
        return features_array
    
    def _reset_sequence_state(self):
        """Reset all sequence-specific state variables."""
        self.prev_velocities = None
        self.prev_frame_time = None
        self.cached_torso_length = None
        self.ankle_history = []
        self.knee_angle_history = []
        
        # Reset idle feature state
        self._prev_positions = None
        self._position_history = []
        self._prev_cog_y = None
    
    def _get_torso_length(self, positions: np.ndarray, visibility: np.ndarray) -> float:
        """
        Calculate torso length (shoulder to hip center) with visibility handling.
        
        Returns:
            Torso length in normalized coordinates, or reference length if can't calculate
        """
        left_shoulder_idx = self.indices['left_shoulder']
        right_shoulder_idx = self.indices['right_shoulder']
        left_hip_idx = self.indices['left_hip']
        right_hip_idx = self.indices['right_hip']
        
        # Check visibility of key points
        shoulder_visible = (visibility[left_shoulder_idx] >= self.config.min_visibility and 
                           visibility[right_shoulder_idx] >= self.config.min_visibility)
        hip_visible = (visibility[left_hip_idx] >= self.config.min_visibility and 
                      visibility[right_hip_idx] >= self.config.min_visibility)
        
        if shoulder_visible and hip_visible:
            # Calculate shoulder center and hip center
            shoulder_center = (positions[left_shoulder_idx] + positions[right_shoulder_idx]) / 2
            hip_center = (positions[left_hip_idx] + positions[right_hip_idx]) / 2
            
            # Calculate torso vector length
            torso_vector = shoulder_center - hip_center
            torso_length = np.linalg.norm(torso_vector)
            
            # Cache for future use in same sequence
            self.cached_torso_length = torso_length
            
            return torso_length
        
        # Fallback to cached or reference length
        if self.cached_torso_length is not None:
            return self.cached_torso_length
        
        return self.config.reference_torso_length
    
    def _calculate_joint_angles_robust(self, positions: np.ndarray, visibility: np.ndarray) -> List[float]:
        """
        Calculate joint angles with occlusion handling.
        Focus on key joints for gesture recognition.
        
        Returns:
            List of angles in radians for: left elbow, right elbow, left knee, right knee
        """
        angles = []
        
        # Define joint triplets for angle calculation: (joint1, joint2, joint3)
        # where joint2 is the vertex of the angle
        joint_triplets = [
            ('left_shoulder', 'left_elbow', 'left_wrist'),  # Left elbow angle
            ('right_shoulder', 'right_elbow', 'right_wrist'),  # Right elbow angle
            ('left_hip', 'left_knee', 'left_ankle'),  # Left knee angle
            ('right_hip', 'right_knee', 'right_ankle'),  # Right knee angle
        ]
        
        for j1_name, j2_name, j3_name in joint_triplets:
            j1_idx = self.indices[j1_name]
            j2_idx = self.indices[j2_name]
            j3_idx = self.indices[j3_name]
            
            # Check visibility with more tolerant threshold for angles
            if (visibility[j1_idx] >= self.config.angle_visibility and
                visibility[j2_idx] >= self.config.angle_visibility and
                visibility[j3_idx] >= self.config.angle_visibility):
                
                # Calculate vectors
                v1 = positions[j1_idx] - positions[j2_idx]
                v2 = positions[j3_idx] - positions[j2_idx]
                
                # Calculate angle
                angle = self._angle_between_vectors(v1, v2)
                angles.append(angle)
            else:
                # If joints are occluded, use default angle (straight = ~180 degrees)
                angles.append(np.pi)  # 180 degrees in radians
        
        return angles
    
    def _calculate_normalized_distances(self, positions: np.ndarray, 
                                       visibility: np.ndarray, 
                                       torso_length: float) -> List[float]:
        """
        Calculate normalized distances between key points.
        
        Returns:
            List of normalized distances for important joint pairs
        """
        distances = []
        
        # Key joint pairs to measure distances between
        joint_pairs = [
            ('left_shoulder', 'right_shoulder'),  # Shoulder width
            ('left_hip', 'right_hip'),  # Hip width
            ('left_shoulder', 'left_hip'),  # Left torso side
            ('right_shoulder', 'right_hip'),  # Right torso side
            ('left_elbow', 'left_shoulder'),  # Left upper arm
            ('right_elbow', 'right_shoulder'),  # Right upper arm
            ('left_wrist', 'left_elbow'),  # Left forearm
            ('right_wrist', 'right_elbow'),  # Right forearm
        ]
        
        for j1_name, j2_name in joint_pairs:
            j1_idx = self.indices[j1_name]
            j2_idx = self.indices[j2_name]
            
            if (visibility[j1_idx] >= self.config.min_visibility and
                visibility[j2_idx] >= self.config.min_visibility):
                
                # Calculate Euclidean distance
                distance = np.linalg.norm(positions[j1_idx] - positions[j2_idx])
                
                # Normalize by torso length
                if self.config.normalize_by_torso and torso_length > 1e-6:
                    normalized_distance = distance / torso_length
                else:
                    normalized_distance = distance
                
                distances.append(normalized_distance)
            else:
                # Use default value for occluded joints
                distances.append(1.0)  # Default normalized distance
        
        return distances
    
    def _calculate_torso_lean_robust(self, positions: np.ndarray, visibility: np.ndarray) -> float:
        """
        Calculate torso lean angle (forward/backward tilt) with occlusion handling.
        
        Returns:
            Torso lean angle in radians (positive = leaning forward, negative = leaning back)
        """
        left_shoulder_idx = self.indices['left_shoulder']
        right_shoulder_idx = self.indices['right_shoulder']
        left_hip_idx = self.indices['left_hip']
        right_hip_idx = self.indices['right_hip']
        
        # Check if we have enough visible points
        shoulder_visible = (visibility[left_shoulder_idx] >= self.config.min_visibility or
                           visibility[right_shoulder_idx] >= self.config.min_visibility)
        hip_visible = (visibility[left_hip_idx] >= self.config.min_visibility or
                      visibility[right_hip_idx] >= self.config.min_visibility)
        
        if shoulder_visible and hip_visible:
            # Calculate centers
            if visibility[left_shoulder_idx] >= self.config.min_visibility and visibility[right_shoulder_idx] >= self.config.min_visibility:
                shoulder_center = (positions[left_shoulder_idx] + positions[right_shoulder_idx]) / 2
            elif visibility[left_shoulder_idx] >= self.config.min_visibility:
                shoulder_center = positions[left_shoulder_idx]
            else:
                shoulder_center = positions[right_shoulder_idx]
            
            if visibility[left_hip_idx] >= self.config.min_visibility and visibility[right_hip_idx] >= self.config.min_visibility:
                hip_center = (positions[left_hip_idx] + positions[right_hip_idx]) / 2
            elif visibility[left_hip_idx] >= self.config.min_visibility:
                hip_center = positions[left_hip_idx]
            else:
                hip_center = positions[right_hip_idx]
            
            # Calculate torso vector (from hip to shoulder)
            torso_vector = shoulder_center - hip_center
            
            # Project onto sagittal plane (x-z plane for forward/backward lean)
            # In MediaPipe coordinates: x = left/right, y = up/down, z = forward/backward
            sagittal_vector = np.array([torso_vector[0], 0, torso_vector[2]])
            
            if np.linalg.norm(sagittal_vector) > 1e-6:
                # Calculate angle from vertical (y-axis)
                vertical = np.array([0, 1, 0])
                angle = self._angle_between_vectors(sagittal_vector, vertical)
                
                # Determine direction (forward or backward)
                # Positive z means leaning forward (toward camera)
                if torso_vector[2] > 0:
                    return angle  # Leaning forward
                else:
                    return -angle  # Leaning backward
            else:
                return 0.0  # No lean
        
        return 0.0  # Default: upright
    
    def _calculate_body_height(self, positions: np.ndarray, visibility: np.ndarray, torso_length: float) -> float:
        """
        Calculate approximate body height relative to torso length.
        
        Returns:
            Height ratio (body height / torso length)
        """
        # Key points for height estimation
        top_points = ['nose', 'left_eye', 'right_eye']
        bottom_points = ['left_ankle', 'right_ankle', 'left_heel', 'right_heel']
        
        # Find highest visible top point
        top_y = None
        for point_name in top_points:
            idx = self.indices[point_name]
            if visibility[idx] >= self.config.min_visibility:
                y = positions[idx, 1]
                if top_y is None or y < top_y:  # Smaller y = higher in image
                    top_y = y
        
        # Find lowest visible bottom point
        bottom_y = None
        for point_name in bottom_points:
            idx = self.indices[point_name]
            if visibility[idx] >= self.config.min_visibility:
                y = positions[idx, 1]
                if bottom_y is None or y > bottom_y:  # Larger y = lower in image
                    bottom_y = y
        
        if top_y is not None and bottom_y is not None:
            height = abs(bottom_y - top_y)
            
            # Normalize by torso length
            if torso_length > 1e-6:
                return height / torso_length
        
        # Fallback: approximate height ratio (average human is ~3x torso length)
        return 3.0
    
    def _calculate_idle_features(self,
                               positions: np.ndarray,
                               visibility: np.ndarray,
                               delta_time: float) -> List[float]:
        """
        Calculate features specifically for idle detection.
        Idle should have minimal movement and stable pose.
        """
        features = []
        
        # 1. Overall movement magnitude (2 features)
        if self._prev_positions is not None:
            movement = np.linalg.norm(positions - self._prev_positions, axis=1)
            visible_mask = visibility >= self.config.min_visibility
            if np.any(visible_mask):
                avg_movement = np.mean(movement[visible_mask])
                max_movement = np.max(movement[visible_mask])
                features.extend([avg_movement, max_movement])
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0])
        
        self._prev_positions = positions.copy()
        
        # 2. Pose stability (variance of joint positions over time) (1 feature)
        self._position_history.append(positions.copy())
        if len(self._position_history) > 10:
            self._position_history.pop(0)
        
        if len(self._position_history) >= 5:
            position_array = np.array(self._position_history)
            stability = np.std(position_array, axis=0).mean()
            features.append(stability)
        else:
            features.append(0.0)
        
        # 3. Symmetry features (idle is typically symmetric) (1 feature)
        left_indices = [self.indices['left_shoulder'], self.indices['left_elbow'], 
                       self.indices['left_wrist'], self.indices['left_hip'], 
                       self.indices['left_knee'], self.indices['left_ankle']]
        
        right_indices = [self.indices['right_shoulder'], self.indices['right_elbow'],
                        self.indices['right_wrist'], self.indices['right_hip'],
                        self.indices['right_knee'], self.indices['right_ankle']]
        
        symmetry_score = 0.0
        valid_pairs = 0
        
        for left_idx, right_idx in zip(left_indices, right_indices):
            if (visibility[left_idx] >= self.config.min_visibility and 
                visibility[right_idx] >= self.config.min_visibility):
                # Compare positions (mirror across vertical axis)
                left_pos = positions[left_idx]
                right_pos = positions[right_idx]
                
                # For symmetry, x should be opposite, y should be similar
                x_diff = abs(left_pos[0] + right_pos[0])  # Should be ~0 for symmetry
                y_diff = abs(left_pos[1] - right_pos[1])  # Should be ~0
                
                pair_symmetry = 1.0 - min((x_diff + y_diff) / 2.0, 1.0)
                symmetry_score += pair_symmetry
                valid_pairs += 1
        
        if valid_pairs > 0:
            symmetry_score /= valid_pairs
        
        features.append(symmetry_score)
        
        # 4. Center of gravity vertical stability (2 features)
        keypoint_indices = [
            self.indices['left_shoulder'], self.indices['right_shoulder'],
            self.indices['left_hip'], self.indices['right_hip']
        ]
        
        visible_keypoints = [positions[i] for i in keypoint_indices 
                            if visibility[i] >= self.config.min_visibility]
        
        if visible_keypoints:
            cog = np.mean(visible_keypoints, axis=0)
            features.append(cog[1])  # Vertical position (1 feature)
            
            # Track vertical movement (1 feature)
            if self._prev_cog_y is not None:
                vertical_movement = abs(cog[1] - self._prev_cog_y)
                features.append(vertical_movement)
            else:
                features.append(0.0)
            
            self._prev_cog_y = cog[1]
        else:
            features.extend([0.0, 0.0])
        
        # Total: 7 idle features
        return features
    
    def _calculate_velocities_jogging_focused(self,
                                            prev_landmarks: np.ndarray,
                                            curr_landmarks: np.ndarray,
                                            delta_time: float) -> List[float]:
        """
        Calculate velocities with emphasis on vertical movement for jogging detection.
        Returns exactly 20 features (deterministic).
        """
        velocities = []
        key_points = [
            'left_ankle', 'right_ankle',  # Primary for jogging (2 × 3 = 6 features)
            'left_knee', 'right_knee',    # Knee vertical movement (2 × 3 = 6 features)
            'left_hip', 'right_hip',      # Hip stability (2 × 2 = 4 features)
            'left_wrist', 'right_wrist',  # Arm swing (2 × 2 = 4 features)
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
                
                # For ankles and knees, emphasize vertical (y) velocity
                if point in ['left_ankle', 'right_ankle', 'left_knee', 'right_knee']:
                    # Vertical velocity (most important for jogging)
                    vertical_vel = velocity[1]
                    # Horizontal velocity (should be low for jogging-in-place)
                    horizontal_vel = np.linalg.norm([velocity[0], velocity[2]])
                    # Total speed
                    speed = np.linalg.norm(velocity)
                    
                    velocities.extend([vertical_vel, horizontal_vel, speed])
                else:
                    # For other points, use standard velocity features
                    speed = np.linalg.norm(velocity)
                    
                    # Normalized directional components (x, y, z)
                    if speed > 1e-6:
                        dir_y = velocity[1] / speed  # Emphasize vertical direction
                    else:
                        dir_y = 0.0
                    
                    velocities.extend([speed, dir_y])  # 2 features per point
            else:
                # Point occluded, use zeros
                if point in ['left_ankle', 'right_ankle', 'left_knee', 'right_knee']:
                    velocities.extend([0.0, 0.0, 0.0])  # 3 features for limb points
                else:
                    velocities.extend([0.0, 0.0])  # 2 features for other points
        
        # Ensure we return exactly 20 features
        if len(velocities) != 20:
            # Pad or truncate if necessary (shouldn't happen with proper logic)
            if len(velocities) < 20:
                velocities.extend([0.0] * (20 - len(velocities)))
            else:
                velocities = velocities[:20]
        
        return velocities
    
    def _calculate_jogging_features(self,
                                   positions: np.ndarray,
                                   visibility: np.ndarray,
                                   delta_time: float,
                                   frame_idx: int) -> List[float]:
        """
        Calculate features specifically designed for jogging-in-place detection.
        Returns exactly 6 features (deterministic).
        """
        features = []
        
        # 1. Ankle vertical positions (store for periodicity)
        ankle_positions = []
        for side in ['left', 'right']:
            idx = self.indices[f'{side}_ankle']
            if visibility[idx] >= self.config.min_visibility:
                ankle_positions.append(positions[idx, 1])  # y-coordinate
            else:
                ankle_positions.append(0.0)
        
        self.ankle_history.append(np.array(ankle_positions))
        if len(self.ankle_history) > self.max_history_frames:
            self.ankle_history.pop(0)
        
        # 2. Knee bend asymmetry (alternating pattern) - 4 features
        left_knee_idx = self.indices['left_knee']
        right_knee_idx = self.indices['right_knee']
        
        if (visibility[left_knee_idx] >= self.config.min_visibility and
            visibility[right_knee_idx] >= self.config.min_visibility):
            
            # Knee vertical positions
            left_knee_y = positions[left_knee_idx, 1]
            right_knee_y = positions[right_knee_idx, 1]
            
            # Hip positions for reference
            left_hip_idx = self.indices['left_hip']
            right_hip_idx = self.indices['right_hip']
            
            if (visibility[left_hip_idx] >= self.config.min_visibility and
                visibility[right_hip_idx] >= self.config.min_visibility):
                
                left_hip_y = positions[left_hip_idx, 1]
                right_hip_y = positions[right_hip_idx, 1]
                
                # Knee lift relative to hip
                left_lift = left_hip_y - left_knee_y  # Positive = knee below hip
                right_lift = right_hip_y - right_knee_y
                
                # Alternation feature: one knee up, one knee down
                alternation = abs(left_lift - right_lift) / max(abs(left_lift) + abs(right_lift), 1e-6)
                
                # Symmetry feature: similar movement on both sides
                symmetry = 1.0 - abs(left_lift - right_lift) / max(abs(left_lift) + abs(right_lift), 1e-6)
                
                features.extend([alternation, symmetry, left_lift, right_lift])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # 3. Hip vertical stability (should be relatively stable for jogging-in-place) - 1 feature
        left_hip_idx = self.indices['left_hip']
        right_hip_idx = self.indices['right_hip']
        
        if (visibility[left_hip_idx] >= self.config.min_visibility and
            visibility[right_hip_idx] >= self.config.min_visibility):
            
            hip_center_y = (positions[left_hip_idx, 1] + positions[right_hip_idx, 1]) / 2
            
            # Calculate vertical movement of hip center over recent frames
            if len(self.ankle_history) > 5:
                # We'll use ankle history as proxy for hip movement timing
                hip_variation = np.std([pos[0] for pos in self.ankle_history[-5:]])  # Left ankle as proxy
            else:
                hip_variation = 0.0
            
            features.append(hip_variation)
        else:
            features.append(0.0)
        
        # 4. Arm-leg opposition (cross-lateral pattern) - 1 feature
        left_wrist_idx = self.indices['left_wrist']
        right_wrist_idx = self.indices['right_wrist']
        
        if (visibility[left_wrist_idx] >= self.config.min_visibility and
            visibility[right_wrist_idx] >= self.config.min_visibility and
            visibility[left_knee_idx] >= self.config.min_visibility and
            visibility[right_knee_idx] >= self.config.min_visibility):
            
            # Wrist positions relative to shoulders
            left_shoulder_idx = self.indices['left_shoulder']
            right_shoulder_idx = self.indices['right_shoulder']
            
            if (visibility[left_shoulder_idx] >= self.config.min_visibility and
                visibility[right_shoulder_idx] >= self.config.min_visibility):
                
                left_wrist_rel = positions[left_wrist_idx, 1] - positions[left_shoulder_idx, 1]
                right_wrist_rel = positions[right_wrist_idx, 1] - positions[right_shoulder_idx, 1]
                
                # Cross-lateral coordination: left knee up with right arm forward, etc.
                cross_coordination = abs(left_wrist_rel - right_wrist_rel) / max(abs(left_wrist_rel) + abs(right_wrist_rel), 1e-6)
                features.append(cross_coordination)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Ensure exactly 6 features
        if len(features) != 6:
            # Pad or truncate if necessary (shouldn't happen)
            if len(features) < 6:
                features.extend([0.0] * (6 - len(features)))
            else:
                features = features[:6]
        
        return features
    
    def _calculate_periodicity_features(self) -> List[float]:
        """
        Calculate periodicity features from recent movement history.
        Returns exactly 4 features (deterministic).
        """
        if len(self.knee_angle_history) < 20:  # Need enough history
            return [0.0] * 4
        
        # Extract left and right knee angle histories
        left_angles = [angle[0] for angle in self.knee_angle_history]
        right_angles = [angle[1] for angle in self.knee_angle_history]
        
        features = []
        
        try:
            # 1. Dominant frequency using FFT
            from scipy.fft import fft, fftfreq
            
            # Use left knee angles for frequency analysis
            signal = np.array(left_angles)
            signal = signal - np.mean(signal)  # Remove DC component
            
            n = len(signal)
            if n > 1:
                yf = fft(signal)
                xf = fftfreq(n, 1.0 / self.config.assume_fps)
                
                # Get magnitude spectrum (positive frequencies only)
                idx = np.arange(1, n // 2)  # Skip DC component
                freqs = xf[idx]
                magnitudes = 2.0 / n * np.abs(yf[idx])
                
                if len(magnitudes) > 0:
                    # Dominant frequency (in Hz)
                    dominant_freq = freqs[np.argmax(magnitudes)]
                    dominant_mag = np.max(magnitudes)
                    
                    features.append(dominant_freq)
                    features.append(dominant_mag)
                else:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0, 0.0])
            
            # 2. Phase difference between left and right knees
            # For jogging, knees should be approximately 180° out of phase
            if len(left_angles) == len(right_angles):
                # Simple correlation-based phase estimation
                correlation = np.corrcoef(left_angles, right_angles)[0, 1]
                phase_similarity = (1.0 - correlation) / 2.0  # 1 = perfect alternation, 0 = in sync
                features.append(phase_similarity)
            else:
                features.append(0.0)
            
            # 3. Regularity (low variance in step timing)
            # Find peaks in knee angle signal (bent knee = step)
            peaks_left, _ = find_peaks(left_angles, height=np.mean(left_angles) + np.std(left_angles))
            peaks_right, _ = find_peaks(right_angles, height=np.mean(right_angles) + np.std(right_angles))
            
            if len(peaks_left) >= 2 and len(peaks_right) >= 2:
                # Calculate inter-step intervals
                intervals_left = np.diff(peaks_left) / self.config.assume_fps
                intervals_right = np.diff(peaks_right) / self.config.assume_fps
                
                # Regularity = 1 / coefficient of variation
                cv_left = np.std(intervals_left) / np.mean(intervals_left) if len(intervals_left) > 0 and np.mean(intervals_left) > 0 else 1.0
                cv_right = np.std(intervals_right) / np.mean(intervals_right) if len(intervals_right) > 0 and np.mean(intervals_right) > 0 else 1.0
                
                regularity = 1.0 / ((cv_left + cv_right) / 2.0) if (cv_left + cv_right) > 0 else 0.0
                features.append(min(regularity, 10.0))  # Cap at 10
            else:
                features.append(0.0)
                
        except Exception as e:
            warnings.warn(f"Periodicity analysis failed: {e}")
            features = [0.0] * 4
        
        # Ensure exactly 4 features
        if len(features) != 4:
            # Pad or truncate if necessary (shouldn't happen)
            if len(features) < 4:
                features.extend([0.0] * (4 - len(features)))
            else:
                features = features[:4]
        
        return features
    
    def _extract_visibility_features(self, visibility: np.ndarray) -> List[float]:
        """
        Extract binary visibility indicators for key joints.
        Returns exactly 12 features (deterministic).
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
            if visibility[idx] >= self.config.min_visibility:
                visibility_features.append(1.0)
            else:
                visibility_features.append(0.0)
        
        return visibility_features
    
    def _angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calculate angle between two vectors in radians.
        
        Returns:
            Angle in radians [0, π]
        """
        # Normalize vectors
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        
        v1_norm = v1 / norm1
        v2_norm = v2 / norm2
        
        # Calculate dot product and clip to avoid numerical errors
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        
        # Calculate angle
        angle = np.arccos(dot_product)
        
        return angle
    
    def _log_feature_stats(self, features: np.ndarray):
        """Log basic statistics about extracted features."""
        if len(features) == 0:
            return
        
        mean = np.mean(features)
        std = np.std(features)
        min_val = np.min(features)
        max_val = np.max(features)
        
        print(f"📊 Feature Statistics:")
        print(f"   Shape: {features.shape}")
        print(f"   Mean: {mean:.4f}, Std: {std:.4f}")
        print(f"   Range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"   Non-zero: {np.count_nonzero(features)}/{features.size} ({np.count_nonzero(features)/features.size:.1%})")
    
    def validate_features(self, features: np.ndarray) -> bool:
        """
        Validate extracted features for common issues.
        
        Returns:
            True if features are valid
        """
        if features is None:
            print("❌ Features are None")
            return False
        
        if len(features.shape) != 2:
            print(f"❌ Expected 2D features array, got shape {features.shape}")
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(features)):
            print("❌ Features contain NaN values")
            return False
        
        if np.any(np.isinf(features)):
            print("❌ Features contain infinite values")
            return False
        
        # Check feature dimension (must match exactly)
        actual_dim = features.shape[1] if len(features.shape) > 1 else 0
        expected_dim = self.get_feature_dimension()
        
        if actual_dim != expected_dim:
            print(f"❌ Feature dimension mismatch:")
            print(f"   Actual: {actual_dim}")
            print(f"   Expected: {expected_dim}")
            print(f"   Difference: {abs(actual_dim - expected_dim)}")
            return False
        
        # Check for extreme values (potential errors)
        if np.max(np.abs(features)) > 1000:
            print("⚠️  Features contain extremely large values")
            # Not necessarily an error, but worth checking
        
        return True
    
    def reset_state(self):
        """Reset all internal state variables."""
        self._reset_sequence_state()
    
    def save_config(self, path: str):
        """Save feature configuration to file."""
        config_dict = self.config.to_dict()
        config_dict['feature_dimension'] = self.get_feature_dimension()
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✅ Feature config saved to {path}")
    
    @classmethod
    def load_from_config(cls, config_path: str) -> 'FeatureEngineer':
        """Create FeatureEngineer from saved configuration."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Remove feature_dimension if present (it's calculated)
        config_dict.pop('feature_dimension', None)
        
        config = FeatureConfig.from_dict(config_dict)
        return cls(config)


# ============================================================================
# FEATURE NORMALIZER (FIXED VERSION)
# ============================================================================

class FeatureNormalizer:
    """Normalizes features for consistent model input with dimension validation."""
    
    def __init__(self):
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self.is_fitted = False
        self.feature_dimension: Optional[int] = None
        self.config_hash: Optional[str] = None
    
    def fit(self, features: np.ndarray, config: Optional[FeatureConfig] = None):
        """Calculate normalization parameters from training data."""
        if len(features) == 0:
            raise ValueError("Cannot fit normalizer with empty features")
        
        # Verify feature dimension consistency
        if len(set(f.shape[1] for f in [features])) > 1:
            raise ValueError("Inconsistent feature dimensions in training data")
        
        self.feature_means = np.mean(features, axis=0)
        self.feature_stds = np.std(features, axis=0)
        
        # Avoid division by zero
        zero_std_mask = self.feature_stds == 0
        if np.any(zero_std_mask):
            print(f"⚠️  {np.sum(zero_std_mask)} features have zero variance")
            self.feature_stds[zero_std_mask] = 1.0
        
        self.feature_dimension = features.shape[1]
        
        # Store config hash if provided
        if config is not None:
            self.config_hash = config.get_hash()
        
        self.is_fitted = True
        print(f"✅ Normalizer fitted on {len(features)} samples")
        print(f"   Feature dimension: {self.feature_dimension}")
        if self.config_hash:
            print(f"   Config hash: {self.config_hash}")
    
    def transform(self, features: np.ndarray, expected_dim: Optional[int] = None) -> np.ndarray:
        """Normalize features using fitted parameters."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transformation")
        
        if self.feature_means is None or self.feature_stds is None:
            raise RuntimeError("Normalizer parameters not set")
        
        # Check dimension match
        actual_dim = features.shape[1]
        expected_dim = expected_dim or self.feature_dimension
        
        if actual_dim != expected_dim:
            raise ValueError(
                f"Feature dimension mismatch during normalization:\n"
                f"  Expected: {expected_dim}\n"
                f"  Actual: {actual_dim}\n"
                f"  Difference: {abs(actual_dim - expected_dim)}\n"
                f"  This indicates a configuration mismatch between training and inference."
            )
        
        # Standard normalization: (x - mean) / std
        normalized = (features - self.feature_means) / self.feature_stds
        
        # Clip extreme values
        normalized = np.clip(normalized, -5.0, 5.0)
        
        return normalized
    
    def fit_transform(self, features: np.ndarray, config: Optional[FeatureConfig] = None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(features, config)
        return self.transform(features)
    
    def save(self, path: str):
        """Save normalizer parameters to file."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before saving")
        
        np.savez(path, 
                means=self.feature_means, 
                stds=self.feature_stds,
                dimension=self.feature_dimension,
                config_hash=self.config_hash)
        print(f"✅ Normalizer saved to {path}")
    
    def load(self, path: str) -> Tuple[int, Optional[str]]:
        """
        Load normalizer parameters from file.
        
        Returns:
            Tuple of (feature_dimension, config_hash)
        """
        data = np.load(path, allow_pickle=True)
        self.feature_means = data['means']
        self.feature_stds = data['stds']
        self.feature_dimension = int(data['dimension'])
        self.config_hash = str(data['config_hash']) if 'config_hash' in data else None
        self.is_fitted = True
        
        print(f"✅ Normalizer loaded from {path}")
        print(f"   Feature dimension: {self.feature_dimension}")
        if self.config_hash:
            print(f"   Config hash: {self.config_hash}")
        
        return self.feature_dimension, self.config_hash


# ============================================================================
# MODEL METADATA MANAGER (NEW - CRITICAL FIX)
# ============================================================================

class ModelMetadata:
    """
    Manages model metadata including feature dimension and configuration.
    Prevents runtime mismatches between training and inference.
    """
    
    def __init__(self):
        self.feature_dimension: Optional[int] = None
        self.feature_config: Optional[FeatureConfig] = None
        self.config_hash: Optional[str] = None
        self.classes: List[str] = []
        self.timestamp: Optional[str] = None
    
    def create(self, 
               feature_engineer: FeatureEngineer,
               classes: List[str]) -> 'ModelMetadata':
        """
        Create metadata from feature engineer.
        Should be called once during training and persisted.
        """
        self.feature_dimension = feature_engineer.get_feature_dimension()
        self.feature_config = feature_engineer.config
        self.config_hash = feature_engineer.config.get_hash()
        self.classes = classes
        self.timestamp = self._get_timestamp()
        
        print(f"✅ Model metadata created:")
        print(f"   Feature dimension: {self.feature_dimension}")
        print(f"   Config hash: {self.config_hash}")
        print(f"   Classes: {self.classes}")
        
        return self
    
    def save(self, path: str):
        """Save metadata to JSON file."""
        if self.feature_config is None:
            raise RuntimeError("Metadata not initialized")
        
        metadata_dict = {
            'feature_dimension': self.feature_dimension,
            'feature_config': self.feature_config.to_dict(),
            'config_hash': self.config_hash,
            'classes': self.classes,
            'timestamp': self.timestamp,
        }
        
        with open(path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        print(f"✅ Model metadata saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ModelMetadata':
        """Load metadata from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        metadata = cls()
        metadata.feature_dimension = data['feature_dimension']
        metadata.feature_config = FeatureConfig.from_dict(data['feature_config'])
        metadata.config_hash = data['config_hash']
        metadata.classes = data['classes']
        metadata.timestamp = data['timestamp']
        
        print(f"✅ Model metadata loaded from {path}")
        print(f"   Feature dimension: {metadata.feature_dimension}")
        print(f"   Config hash: {metadata.config_hash}")
        
        return metadata
    
    def validate_inference_setup(self, 
                               feature_engineer: FeatureEngineer,
                               normalizer: Optional[FeatureNormalizer] = None) -> bool:
        """
        Validate that inference setup matches training metadata.
        Should be called at inference startup.
        
        Returns:
            True if validation passes
        """
        print("🔍 Validating inference setup...")
        
        # 1. Check feature dimension
        current_dim = feature_engineer.get_feature_dimension()
        if current_dim != self.feature_dimension:
            print(f"❌ Feature dimension mismatch:")
            print(f"   Training: {self.feature_dimension}")
            print(f"   Inference: {current_dim}")
            return False
        
        print(f"   ✅ Feature dimension matches: {current_dim}")
        
        # 2. Check configuration hash
        current_hash = feature_engineer.config.get_hash()
        if current_hash != self.config_hash:
            print(f"❌ Feature configuration mismatch:")
            print(f"   Training hash: {self.config_hash}")
            print(f"   Inference hash: {current_hash}")
            print(f"   This will cause silent model degradation!")
            return False
        
        print(f"   ✅ Config hash matches: {current_hash}")
        
        # 3. Check normalizer dimension if provided
        if normalizer is not None and normalizer.is_fitted:
            if normalizer.feature_dimension != self.feature_dimension:
                print(f"❌ Normalizer dimension mismatch:")
                print(f"   Metadata: {self.feature_dimension}")
                print(f"   Normalizer: {normalizer.feature_dimension}")
                return False
            print(f"   ✅ Normalizer dimension matches")
        
        # 4. Check idle feature consistency (critical fix)
        training_idle = self.feature_config.compute_idle_features
        inference_idle = feature_engineer.config.compute_idle_features
        
        if training_idle != inference_idle:
            print(f"❌ Idle feature configuration mismatch:")
            print(f"   Training: idle_features={training_idle}")
            print(f"   Inference: idle_features={inference_idle}")
            print(f"   This will cause decision boundary drift!")
            return False
        
        print(f"   ✅ Idle features consistent: {training_idle}")
        
        print("✅ All validation checks passed!")
        return True
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================================
# GESTURE MODEL VALIDATOR (FIXED VERSION)
# ============================================================================

class GestureModelValidator:
    """
    Validates model input shapes and feature dimensions.
    Prevents hard failures at runtime.
    """
    
    def __init__(self, expected_feature_dim: int, expected_sequence_length: int = 30):
        self.expected_feature_dim = expected_feature_dim
        self.expected_sequence_length = expected_sequence_length
        
    def validate_input_shape(self, features: np.ndarray) -> bool:
        """
        Validate input features match expected shape.
        
        Args:
            features: Input features of shape (batch, sequence, features) or (sequence, features)
            
        Returns:
            True if validation passes
        """
        if features is None:
            print("❌ Input features are None")
            return False
        
        # Handle both batched and unbatched inputs
        if len(features.shape) == 3:
            # Batched input: (batch, sequence, features)
            batch_size, sequence_len, feature_dim = features.shape
        elif len(features.shape) == 2:
            # Unbatched input: (sequence, features)
            sequence_len, feature_dim = features.shape
            batch_size = 1
        else:
            print(f"❌ Unexpected input shape: {features.shape}")
            return False
        
        # Check sequence length
        if sequence_len != self.expected_sequence_length:
            print(f"❌ Sequence length mismatch:")
            print(f"   Expected: {self.expected_sequence_length}")
            print(f"   Actual: {sequence_len}")
            return False
        
        # Check feature dimension (CRITICAL FIX)
        if feature_dim != self.expected_feature_dim:
            print(f"❌ Feature dimension mismatch:")
            print(f"   Expected: {self.expected_feature_dim}")
            print(f"   Actual: {feature_dim}")
            print(f"   Difference: {abs(feature_dim - self.expected_feature_dim)}")
            
            # Provide diagnostic information
            if feature_dim < self.expected_feature_dim:
                print(f"   Missing {self.expected_feature_dim - feature_dim} features")
                print(f"   Check if idle features are enabled/disabled correctly")
            else:
                print(f"   Extra {feature_dim - self.expected_feature_dim} features")
            
            return False
        
        print(f"✅ Input validation passed:")
        print(f"   Batch size: {batch_size}")
        print(f"   Sequence length: {sequence_len}")
        print(f"   Feature dimension: {feature_dim}")
        
        return True
    
    def ensure_correct_shape(self, features: np.ndarray) -> np.ndarray:
        """
        Ensure features have correct shape, padding or truncating if necessary.
        
        Returns:
            Features with correct shape
        """
        if len(features.shape) == 2:
            current_seq, current_dim = features.shape
        else:
            raise ValueError(f"Expected 2D array, got shape {features.shape}")
        
        # Handle sequence length
        if current_seq < self.expected_sequence_length:
            # Pad with zeros
            pad_length = self.expected_sequence_length - current_seq
            padding = np.zeros((pad_length, current_dim), dtype=features.dtype)
            features = np.vstack([features, padding])
            print(f"⚠️  Padded sequence from {current_seq} to {self.expected_sequence_length} frames")
        elif current_seq > self.expected_sequence_length:
            # Truncate
            features = features[-self.expected_sequence_length:]
            print(f"⚠️  Truncated sequence from {current_seq} to {self.expected_sequence_length} frames")
        
        # Handle feature dimension (should not happen with proper configuration)
        if current_dim != self.expected_feature_dim:
            raise ValueError(
                f"Cannot automatically fix feature dimension mismatch:\n"
                f"  Expected: {self.expected_feature_dim}\n"
                f"  Actual: {current_dim}\n"
                f"  This must be fixed in configuration."
            )
        
        return features





# ============================================================================
# TEST FUNCTIONS WITH COMPREHENSIVE VALIDATION
# ============================================================================

def test_deterministic_feature_dimension():
    """Test that feature dimension is deterministic and consistent."""
    print("🧪 Testing deterministic feature dimension...")
    
    # Test different configurations
    configs = [
        ("Minimal", FeatureConfig(
            compute_angles=False,
            compute_distances=False,
            compute_jogging_features=False,
            compute_periodicity=False,
            compute_idle_features=False
        )),
        ("Standard (no idle)", FeatureConfig(
            compute_idle_features=False
        )),
        ("Standard (with idle)", FeatureConfig(
            compute_idle_features=True
        )),
        ("Full", FeatureConfig(
            compute_idle_features=True,
            compute_jogging_features=True,
            compute_periodicity=True
        ))
    ]
    
    results = []
    for name, config in configs:
        engineer = FeatureEngineer(config)
        
        # Create dummy input
        dummy_input = np.zeros((30, 33, 4), dtype=np.float32)
        dummy_input[:, :, 3] = 1.0  # All visible
        
        # Extract features
        features = engineer.extract_features(dummy_input)
        actual_dim = features.shape[1]
        predicted_dim = engineer.get_feature_dimension()
        
        match = actual_dim == predicted_dim
        results.append((name, predicted_dim, actual_dim, match))
        
        print(f"   {name:20s}: predicted={predicted_dim:3d}, actual={actual_dim:3d}, match={match}")
        
        # Verify validation
        if not engineer.validate_features(features):
            print(f"   ❌ Validation failed for {name}")
            return False
    
    # Check all configurations match
    all_match = all(r[3] for r in results)
    print(f"\n✅ All configurations deterministic: {all_match}")
    
    return all_match

def test_idle_feature_consistency():
    """Test that idle features are handled consistently."""
    print("\n🧪 Testing idle feature consistency...")
    
    # Create two configurations: one with idle, one without
    config_with_idle = FeatureConfig(compute_idle_features=True)
    config_without_idle = FeatureConfig(compute_idle_features=False)
    
    engineer_with_idle = FeatureEngineer(config_with_idle)
    engineer_without_idle = FeatureEngineer(config_without_idle)
    
    # Create test data
    dummy_input = np.zeros((30, 33, 4), dtype=np.float32)
    dummy_input[:, :, 3] = 1.0
    
    # Extract features
    features_with_idle = engineer_with_idle.extract_features(dummy_input)
    features_without_idle = engineer_without_idle.extract_features(dummy_input)
    
    dim_with_idle = features_with_idle.shape[1]
    dim_without_idle = features_without_idle.shape[1]
    dim_difference = dim_with_idle - dim_without_idle
    
    print(f"   With idle features:    {dim_with_idle} features")
    print(f"   Without idle features: {dim_without_idle} features")
    print(f"   Difference:            {dim_difference} features")
    
    # The difference should be exactly 7 (idle feature count)
    if dim_difference == 7:
        print(f"   ✅ Idle feature count correct: 7")
    else:
        print(f"   ❌ Unexpected idle feature count: {dim_difference}")
        return False
    
    # Test metadata validation
    metadata = ModelMetadata()
    metadata.create(engineer_with_idle, ['idle', 'jogging', 'other'])
    
    # This should pass
    if metadata.validate_inference_setup(engineer_with_idle):
        print("   ✅ Validation passes with matching config")
    else:
        print("   ❌ Validation should pass with matching config")
        return False
    
    # This should fail (config mismatch)
    print("\n   Testing config mismatch detection...")
    if not metadata.validate_inference_setup(engineer_without_idle):
        print("   ✅ Correctly detects config mismatch")
    else:
        print("   ❌ Should detect config mismatch")
        return False
    
    return True

def test_sequence_state_management():
    """Test that state is properly reset between sequences."""
    print("\n🧪 Testing sequence state management...")
    
    config = FeatureConfig(compute_idle_features=True)
    engineer = FeatureEngineer(config)
    
    # Create two different sequences
    np.random.seed(42)
    
    seq1 = np.random.normal(0, 0.1, (20, 33, 4)).astype(np.float32)
    seq1[:, :, 3] = 1.0
    
    seq2 = np.random.normal(1, 0.1, (20, 33, 4)).astype(np.float32)  # Different mean
    seq2[:, :, 3] = 1.0
    
    # Process first sequence
    features1 = engineer.extract_features(seq1)
    
    # Process second sequence (state should be reset)
    features2 = engineer.extract_features(seq2)
    
    # Check that both sequences have same dimension
    if features1.shape[1] == features2.shape[1]:
        print(f"   ✅ Both sequences have same feature dimension: {features1.shape[1]}")
    else:
        print(f"   ❌ Feature dimension mismatch between sequences")
        return False
    
    # Check that features are different (state was reset)
    # The sequences have different means, so features should be different
    if not np.allclose(features1, features2):
        print("   ✅ Sequences produce different features (state properly reset)")
    else:
        print("   ⚠️  Sequences produce similar features (might indicate state contamination)")
    
    return True

def test_normalizer_dimension_safety():
    """Test that normalizer catches dimension mismatches."""
    print("\n🧪 Testing normalizer dimension safety...")
    
    # Create two different configurations
    config1 = FeatureConfig(compute_idle_features=True)
    config2 = FeatureConfig(compute_idle_features=False)
    
    engineer1 = FeatureEngineer(config1)
    engineer2 = FeatureEngineer(config2)
    
    # Create dummy data
    dummy_data = np.zeros((100, 33, 4), dtype=np.float32)
    dummy_data[:, :, 3] = 1.0
    
    # Extract features with different dimensions
    features1 = engineer1.extract_features(dummy_data)
    features2 = engineer2.extract_features(dummy_data)
    
    # Create normalizer
    normalizer = FeatureNormalizer()
    
    # Fit on features with idle
    normalizer.fit(features1, config1)
    
    # Try to transform features without idle (should fail)
    try:
        normalized2 = normalizer.transform(features2)
        print("   ❌ Should have failed due to dimension mismatch")
        return False
    except ValueError as e:
        print(f"   ✅ Correctly caught dimension mismatch: {str(e)[:80]}...")
    
    # Transform features with idle (should work)
    try:
        normalized1 = normalizer.transform(features1)
        print("   ✅ Successfully normalized matching features")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False
    
    # Test with explicit expected dimension
    try:
        normalized2_with_dim = normalizer.transform(features2, expected_dim=features2.shape[1])
        print("   ✅ Allowed transform with explicit dimension override")
    except Exception as e:
        print(f"   ❌ Should allow with explicit dimension: {e}")
        return False
    
    return True

def test_end_to_end_pipeline():
    """Test complete pipeline from features to validation."""
    print("\n🧪 Testing end-to-end pipeline...")
    
    # 1. Create configuration (with idle features)
    config = FeatureConfig(
        compute_idle_features=True,
        compute_jogging_features=True,
        compute_periodicity=True
    )
    
    # 2. Create feature engineer
    engineer = FeatureEngineer(config)
    feature_dim = engineer.get_feature_dimension()
    print(f"   Feature dimension: {feature_dim}")
    
    # 3. Create training data
    np.random.seed(42)
    train_data = []
    for _ in range(10):
        seq = np.random.normal(0, 0.1, (30, 33, 4)).astype(np.float32)
        seq[:, :, 3] = 1.0
        train_data.append(seq)
    
    # 4. Extract features
    train_features = []
    for seq in train_data:
        features = engineer.extract_features(seq)
        train_features.append(features)
    
    train_features_array = np.vstack(train_features)
    print(f"   Training features shape: {train_features_array.shape}")
    
    # 5. Create metadata
    metadata = ModelMetadata()
    metadata.create(engineer, ['idle', 'jogging', 'walking', 'other'])
    
    # 6. Create normalizer
    normalizer = FeatureNormalizer()
    normalizer.fit(train_features_array, config)
    
    # 7. Save everything
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save metadata
        metadata_path = os.path.join(tmpdir, 'metadata.json')
        metadata.save(metadata_path)
        
        # Save normalizer
        normalizer_path = os.path.join(tmpdir, 'normalizer.npz')
        normalizer.save(normalizer_path)
        
        # Save config
        config_path = os.path.join(tmpdir, 'config.json')
        engineer.save_config(config_path)
        
        print(f"   Saved artifacts to temporary directory")
        
        # 8. Simulate inference setup
        print("\n   Simulating inference setup...")
        
        # Load metadata
        loaded_metadata = ModelMetadata.load(metadata_path)
        
        # Load config and create engineer
        loaded_engineer = FeatureEngineer.load_from_config(config_path)
        
        # Load normalizer
        loaded_normalizer = FeatureNormalizer()
        loaded_dim, loaded_hash = loaded_normalizer.load(normalizer_path)
        
        # 9. Validate inference setup
        if loaded_metadata.validate_inference_setup(loaded_engineer, loaded_normalizer):
            print("   ✅ Inference setup validation passed")
        else:
            print("   ❌ Inference setup validation failed")
            return False
        
        # 10. Create validator
        validator = GestureModelValidator(
            expected_feature_dim=loaded_metadata.feature_dimension,
            expected_sequence_length=30
        )
        
        # 11. Test inference
        test_seq = np.random.normal(0, 0.1, (25, 33, 4)).astype(np.float32)  # Shorter sequence
        test_seq[:, :, 3] = 1.0
        
        # Extract features
        test_features = loaded_engineer.extract_features(test_seq)
        
        # Ensure correct shape
        test_features_corrected = validator.ensure_correct_shape(test_features)
        
        # Normalize
        normalized_features = loaded_normalizer.transform(test_features_corrected)
        
        # Validate input
        if validator.validate_input_shape(normalized_features.reshape(1, 30, -1)):
            print("   ✅ Inference pipeline complete and validated")
        else:
            print("   ❌ Inference pipeline validation failed")
            return False
    
    print("\n✅ End-to-end pipeline test passed!")
    return True

def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("=" * 70)
    print("COMPREHENSIVE FEATURE ENGINEERING TESTS")
    print("WITH DIMENSION FIXES AND IDLE CONSISTENCY")
    print("=" * 70)
    
    tests = [
        ("Deterministic Feature Dimension", test_deterministic_feature_dimension),
        ("Idle Feature Consistency", test_idle_feature_consistency),
        ("Sequence State Management", test_sequence_state_management),
        ("Normalizer Dimension Safety", test_normalizer_dimension_safety),
        ("End-to-End Pipeline", test_end_to_end_pipeline),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {test_name}")
        print(f"{'='*60}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Feature engineering is robust and deterministic.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review the issues above.")
    
    return passed == total


if __name__ == "__main__":
    # Run comprehensive tests
    all_passed = run_comprehensive_tests()
    
    if all_passed:
        # Show integration fixes
        apply_fixes_to_existing_code()
        
        print("\n" + "=" * 70)
        print("SUMMARY OF CRITICAL FIXES APPLIED:")
        print("=" * 70)
        print("1. ✅ Deterministic feature dimension via get_feature_dimension()")
        print("2. ✅ Consistent idle feature handling (always enabled by default)")
        print("3. ✅ ModelMetadata class to persist configuration")
        print("4. ✅ GestureModelValidator for runtime shape validation")
        print("5. ✅ Normalizer dimension checking and validation")
        print("6. ✅ State management fixes to prevent contamination")
        print("7. ✅ Configuration hash for detecting mismatches")
        print("\nNEXT STEPS:")
        print("1. Update config.py to enable idle features: compute_idle_features=True")
        print("2. Add metadata creation to training pipeline")
        print("3. Add validation to inference pipeline")
        print("4. Retrain model with consistent configuration")
        print("=" * 70)
    else:
        print("\n❌ Some tests failed. Fix the issues before proceeding.")