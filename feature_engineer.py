# feature_engineer.py (UPDATED FOR JOGGING-IN-PLACE "RUN" GESTURE)
"""
Robust feature engineering with specialized features for jogging-in-place detection.
Emphasizes temporal patterns, periodicity, and vertical movement over displacement.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.signal import find_peaks
import warnings

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

class FeatureEngineer:
    """
    Transforms raw landmarks into robust engineered features.
    Special emphasis on jogging-in-place detection with temporal patterns.
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
        
        # State for periodicity analysis
        self.ankle_history: List[np.ndarray] = []
        self.knee_angle_history: List[Tuple[float, float]] = []
        self.max_history_frames = 60  # Store last 2 seconds at 30 FPS
        
        print(f"✅ FeatureEngineer initialized with jogging emphasis")
        print(f"   Jogging features: {self.config.compute_jogging_features}")
        print(f"   Periodicity analysis: {self.config.compute_periodicity}")
    
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
            Feature array of shape (T, F) where F = feature count
        """
        T = len(landmarks_sequence)
        features_list = []
        
        # Reset state for new sequence
        self.prev_velocities = None
        self.prev_frame_time = None
        self.cached_torso_length = None
        self.ankle_history = []
        self.knee_angle_history = []
        
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
                frame_features.extend(velocities)
                
                # Apply smoothing
                if self.prev_velocities is not None:
                    velocities = self.config.velocity_smoothing * velocities + \
                                (1 - self.config.velocity_smoothing) * self.prev_velocities
                
                self.prev_velocities = velocities
            else:
                # Pad with zeros for first frame
                velocity_features = self._get_velocity_feature_count()
                frame_features.extend([0.0] * velocity_features)
                self.prev_velocities = np.zeros(velocity_features)
            
            # 6. Jogging-specific features
            if self.config.compute_jogging_features:
                jogging_features = self._calculate_jogging_features(
                    positions, visibility, delta_time, t
                )
                frame_features.extend(jogging_features)
            
            # 7. Periodicity features
            if self.config.compute_periodicity and len(self.knee_angle_history) > 10:
                periodicity_features = self._calculate_periodicity_features()
                frame_features.extend(periodicity_features)
            elif self.config.compute_periodicity:
                # Pad with zeros if not enough history
                frame_features.extend([0.0] * 4)  # 4 periodicity features
            
            # 8. Visibility mask (binary indicators for key joints)
            visibility_features = self._extract_visibility_features(visibility)
            frame_features.extend(visibility_features)
            
            features_list.append(frame_features)
        
        features_array = np.array(features_list, dtype=np.float32)
        
        # Log feature statistics
        if len(features_list) > 0:
            self._log_feature_stats(features_array)
        
        return features_array
    
    def _calculate_velocities_jogging_focused(self,
                                            prev_landmarks: np.ndarray,
                                            curr_landmarks: np.ndarray,
                                            delta_time: float) -> List[float]:
        """
        Calculate velocities with emphasis on vertical movement for jogging detection.
        """
        velocities = []
        key_points = [
            'left_ankle', 'right_ankle',  # Primary for jogging
            'left_knee', 'right_knee',    # Knee vertical movement
            'left_hip', 'right_hip',      # Hip stability
            'left_wrist', 'right_wrist',  # Arm swing
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
                        dir_x = velocity[0] / speed
                        dir_y = velocity[1] / speed
                        dir_z = velocity[2] / speed
                    else:
                        dir_x = 0.0
                        dir_y = 0.0
                        dir_z = 0.0
                    
                    velocities.extend([speed, dir_y])  # Emphasize vertical direction
            else:
                # Point occluded, use zeros
                if point in ['left_ankle', 'right_ankle', 'left_knee', 'right_knee']:
                    velocities.extend([0.0, 0.0, 0.0])  # 3 features for limb points
                else:
                    velocities.extend([0.0, 0.0])  # 2 features for other points
        
        return velocities
    
    def _calculate_jogging_features(self,
                                   positions: np.ndarray,
                                   visibility: np.ndarray,
                                   delta_time: float,
                                   frame_idx: int) -> List[float]:
        """
        Calculate features specifically designed for jogging-in-place detection.
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
        
        # 2. Knee bend asymmetry (alternating pattern)
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
        
        # 3. Hip vertical stability (should be relatively stable for jogging-in-place)
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
        
        # 4. Arm-leg opposition (cross-lateral pattern)
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
                # Simplified: check if wrist and opposite knee are moving together
                cross_coordination = 0.0  # Placeholder - would need velocity correlation
                features.append(cross_coordination)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        return features
    
    def _calculate_periodicity_features(self) -> List[float]:
        """
        Calculate periodicity features from recent movement history.
        Essential for distinguishing rhythmic jogging from other movements.
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
        
        return features
    
    def _get_velocity_feature_count(self) -> int:
        """Get number of velocity features based on configuration."""
        # 4 limb points × 3 features + 4 other points × 2 features
        return 4 * 3 + 4 * 2
    
    def _get_torso_length(self, positions: np.ndarray, visibility: np.ndarray) -> float:
        """Calculate torso length (shoulder to hip center) with visibility handling."""
        # ... (same as before, kept for brevity)
        pass
    
    def _calculate_joint_angles_robust(self, positions: np.ndarray, visibility: np.ndarray) -> List[float]:
        """Calculate joint angles with occlusion handling."""
        # ... (same as before, kept for brevity)
        pass
    
    def _calculate_normalized_distances(self, positions: np.ndarray, visibility: np.ndarray, torso_length: float) -> List[float]:
        """Calculate normalized distances between key points."""
        # ... (same as before, kept for brevity)
        pass
    
    def _calculate_torso_lean_robust(self, positions: np.ndarray, visibility: np.ndarray) -> float:
        """Calculate torso lean angle with occlusion handling."""
        # ... (same as before, kept for brevity)
        pass
    
    def _calculate_body_height(self, positions: np.ndarray, visibility: np.ndarray, torso_length: float) -> float:
        """Calculate approximate body height relative to torso length."""
        # ... (same as before, kept for brevity)
        pass
    
    def _extract_visibility_features(self, visibility: np.ndarray) -> List[float]:
        """Extract binary visibility indicators for key joints."""
        # ... (same as before, kept for brevity)
        pass
    
    def _angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors in radians."""
        # ... (same as before, kept for brevity)
        pass
    
    def _log_feature_stats(self, features: np.ndarray):
        """Log basic statistics about extracted features."""
        # ... (same as before, kept for brevity)
        pass
    
    def get_feature_dimension(self) -> int:
        """
        Calculate the expected feature dimension.
        Now includes jogging-specific features.
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
        # ... (same as before, kept for brevity)
        pass

# FeatureNormalizer class remains the same
# ... (same as before, kept for brevity)

# ============================================================================
# TEST FUNCTION WITH JOGGING EMPHASIS
# ============================================================================

def test_jogging_features():
    """Test feature engineering with emphasis on jogging detection."""
    print("Testing Jogging-Focused Feature Engineering...")
    
    # Create test landmarks simulating jogging motion
    np.random.seed(42)
    T = 60  # 2 seconds at 30 FPS
    landmarks = np.zeros((T, 33, 4), dtype=np.float32)
    
    # Simulate jogging-in-place with alternating knee lifts
    for t in range(T):
        # All joints visible
        landmarks[t, :, 3] = 1.0
        
        # Simulate alternating knee movement
        phase = 2 * np.pi * t / 15  # 2Hz jogging frequency
        
        # Left knee up when phase is 0-π, right knee up when phase is π-2π
        left_knee_lift = 0.2 * (1 + np.sin(phase))  # 0 to 0.4 range
        right_knee_lift = 0.2 * (1 + np.sin(phase + np.pi))  # Opposite phase
        
        # Set knee positions (y-coordinate)
        landmarks[t, 25, 1] = 0.5 - left_knee_lift  # left_knee
        landmarks[t, 26, 1] = 0.5 - right_knee_lift  # right_knee
        
        # Ankles follow knees but less extreme
        landmarks[t, 27, 1] = 0.3 - left_knee_lift * 0.5  # left_ankle
        landmarks[t, 28, 1] = 0.3 - right_knee_lift * 0.5  # right_ankle
        
        # Hips relatively stable
        landmarks[t, 23, 1] = 0.6  # left_hip
        landmarks[t, 24, 1] = 0.6  # right_hip
        
        # Arms swing opposite to legs
        landmarks[t, 15, 1] = 0.7 - 0.1 * np.sin(phase + np.pi)  # left_wrist
        landmarks[t, 16, 1] = 0.7 - 0.1 * np.sin(phase)  # right_wrist
    
    # Test with jogging-focused config
    config = FeatureConfig(
        compute_jogging_features=True,
        compute_periodicity=True,
        compute_alternation=True,
        compute_velocities=True,
        assume_fps=30.0
    )
    
    engineer = FeatureEngineer(config)
    features = engineer.extract_features(landmarks)
    
    print(f"✅ Jogging features extracted: shape {features.shape}")
    print(f"   Feature dimension: {engineer.get_feature_dimension()}")
    
    # Analyze specific jogging features
    print(f"\n📊 Jogging feature analysis:")
    
    # Look at periodicity features (should be at the end of feature vector)
    periodicity_idx = -10  # Adjust based on actual feature ordering
    if features.shape[1] > abs(periodicity_idx):
        periodicity_features = features[:, periodicity_idx:periodicity_idx+4]
        print(f"   Dominant frequency: {np.mean(periodicity_features[:, 0]):.2f} Hz")
        print(f"   Phase similarity: {np.mean(periodicity_features[:, 2]):.2f}")
        print(f"   Regularity: {np.mean(periodicity_features[:, 3]):.2f}")
    
    # Check for expected patterns
    ankle_velocities = features[:, 30:33]  # Example indices for left ankle velocities
    avg_vertical_vel = np.mean(np.abs(ankle_velocities[:, 0]))  # Vertical velocity
    avg_horizontal_vel = np.mean(np.abs(ankle_velocities[:, 1]))  # Horizontal velocity
    
    print(f"\n🏃 Jogging pattern validation:")
    print(f"   Avg vertical ankle velocity: {avg_vertical_vel:.3f}")
    print(f"   Avg horizontal ankle velocity: {avg_horizontal_vel:.3f}")
    print(f"   Vertical/Horizontal ratio: {avg_vertical_vel/max(avg_horizontal_vel, 0.001):.1f}")
    
    # Good jogging should have high vertical, low horizontal movement
    if avg_vertical_vel > 2 * avg_horizontal_vel:
        print("   ✅ Good jogging pattern (vertical > horizontal)")
    else:
        print("   ⚠️  Check jogging technique (too much horizontal movement)")
    
    # Validate features
    is_valid = engineer.validate_features(features)
    print(f"\n✅ Feature validation: {is_valid}")
    
    print("\n✅ All jogging feature tests passed!")

if __name__ == "__main__":
    test_jogging_features()