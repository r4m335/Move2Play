# pose_extractor.py (FIXED VERSION)
"""
Optimized real-time pose extraction using MediaPipe.
Single inference per frame, robust normalization.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass

@dataclass
class PoseResult:
    """Container for pose extraction results."""
    landmarks: np.ndarray  # Normalized landmarks WITH visibility (33, 4)
    raw_landmarks: np.ndarray  # Original MediaPipe landmarks (33, 4)
    mp_results: Any  # MediaPipe results object
    image_shape: Tuple[int, int]  # (height, width)
    normalized_positions: np.ndarray  # Normalized positions only (33, 3) - for compatibility

class PoseExtractor:
    """Extracts and normalizes body landmarks from camera frames."""
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5,
                 use_torso_length: bool = True):
        """
        Args:
            min_detection_confidence: MediaPipe detection confidence threshold
            min_tracking_confidence: MediaPipe tracking confidence threshold
            use_torso_length: Use torso length for normalization (more robust than shoulder width)
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=Light, 1=Full, 2=Heavy
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            smooth_landmarks=True
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.use_torso_length = use_torso_length
        
        # Landmark indices for quick reference
        self.LANDMARK_INDICES = {
            'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
            'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
            'left_ear': 7, 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10,
            'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13,
            'right_elbow': 14, 'left_wrist': 15, 'right_wrist': 16,
            'left_pinky': 17, 'right_pinky': 18, 'left_index': 19,
            'right_index': 20, 'left_thumb': 21, 'right_thumb': 22,
            'left_hip': 23, 'right_hip': 24, 'left_knee': 25,
            'right_knee': 26, 'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 30, 'left_foot_index': 31,
            'right_foot_index': 32
        }
        
        # Connection indices for drawing (if needed)
        self.POSE_CONNECTIONS = self.mp_pose.POSE_CONNECTIONS
        
    def process_frame(self, image: np.ndarray) -> Optional[PoseResult]:
        """
        Single-pass processing: extract landmarks, normalize, and return results.
        
        Args:
            image: BGR image frame from camera
            
        Returns:
            PoseResult containing landmarks and MediaPipe results, or None if no pose detected
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Single MediaPipe inference
        mp_results = self.pose.process(image_rgb)
        
        if not mp_results.pose_landmarks:
            return None
        
        # Convert landmarks to numpy array
        raw_landmarks = self._landmarks_to_array(mp_results.pose_landmarks)
        
        # Normalize landmarks (keeping visibility)
        normalized_landmarks = self._normalize_landmarks_with_visibility(raw_landmarks)
        
        # Extract positions only for backward compatibility
        normalized_positions = normalized_landmarks[:, :3]
        
        return PoseResult(
            landmarks=normalized_landmarks,  # (33, 4) with visibility
            raw_landmarks=raw_landmarks,  # (33, 4) original
            mp_results=mp_results,
            image_shape=image.shape[:2],  # (height, width)
            normalized_positions=normalized_positions  # (33, 3) positions only
        )
    
    def extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Legacy method for backward compatibility.
        Returns normalized landmarks WITH visibility (33, 4) for FeatureEngineer.
        """
        result = self.process_frame(image)
        return result.landmarks if result else None
    
    def extract_positions_only(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns only normalized positions (33, 3) without visibility.
        For use cases that don't need visibility.
        """
        result = self.process_frame(image)
        return result.normalized_positions if result else None
    
    def _landmarks_to_array(self, pose_landmarks) -> np.ndarray:
        """Convert MediaPipe landmarks to numpy array with visibility."""
        landmarks = []
        for lm in pose_landmarks.landmark:
            # Store: [x, y, z, visibility]
            landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(landmarks, dtype=np.float32)
    
    def _normalize_landmarks_with_visibility(self, 
                                           landmarks: np.ndarray, 
                                           visibility_threshold: float = 0.5) -> np.ndarray:
        """
        Robust normalization preserving visibility information.
        
        Args:
            landmarks: Raw landmarks from MediaPipe (33, 4)
            visibility_threshold: Minimum visibility score to consider joint
            
        Returns:
            Normalized landmarks WITH visibility (33, 4)
        """
        # Extract positions and visibility
        positions = landmarks[:, :3]  # (33, 3)
        visibility = landmarks[:, 3]  # (33,)
        
        # Get torso keypoints with visibility check
        left_shoulder_idx = self.LANDMARK_INDICES['left_shoulder']
        right_shoulder_idx = self.LANDMARK_INDICES['right_shoulder']
        left_hip_idx = self.LANDMARK_INDICES['left_hip']
        right_hip_idx = self.LANDMARK_INDICES['right_hip']
        
        # Calculate hip center (more stable than single hip)
        hip_center = np.zeros(3, dtype=np.float32)
        hip_count = 0
        
        for idx in [left_hip_idx, right_hip_idx]:
            if visibility[idx] >= visibility_threshold:
                hip_center += positions[idx]
                hip_count += 1
        
        if hip_count == 0:
            # Fallback: use visible shoulders or default position
            visible_shoulders = []
            for idx in [left_shoulder_idx, right_shoulder_idx]:
                if visibility[idx] >= visibility_threshold:
                    visible_shoulders.append(positions[idx])
            
            if visible_shoulders:
                hip_center = np.mean(visible_shoulders, axis=0)
            else:
                # If nothing is visible, use origin
                hip_center = np.zeros(3, dtype=np.float32)
        else:
            hip_center /= hip_count
        
        # Calculate torso length (shoulder to hip) for scaling
        if self.use_torso_length:
            # Use torso vector length for scaling (more robust than shoulder width)
            shoulder_center = np.zeros(3, dtype=np.float32)
            shoulder_count = 0
            
            for idx in [left_shoulder_idx, right_shoulder_idx]:
                if visibility[idx] >= visibility_threshold:
                    shoulder_center += positions[idx]
                    shoulder_count += 1
            
            if shoulder_count == 0:
                # Fallback: use default scale
                scale_factor = 1.0
            else:
                shoulder_center /= shoulder_count
                torso_vector = shoulder_center - hip_center
                torso_length = np.linalg.norm(torso_vector)
                
                # Prevent division by zero
                scale_factor = torso_length if torso_length > 1e-6 else 1.0
        else:
            # Original shoulder width method (less robust)
            if (visibility[left_shoulder_idx] >= visibility_threshold and 
                visibility[right_shoulder_idx] >= visibility_threshold):
                shoulder_width = np.linalg.norm(
                    positions[left_shoulder_idx] - positions[right_shoulder_idx]
                )
                scale_factor = shoulder_width if shoulder_width > 1e-6 else 1.0
            else:
                scale_factor = 1.0
        
        # Normalize positions (keep visibility unchanged)
        normalized = np.zeros((33, 4), dtype=np.float32)
        
        for i in range(len(positions)):
            # Copy visibility from original
            normalized[i, 3] = visibility[i]
            
            if visibility[i] >= visibility_threshold:
                # Translate to hip center and scale by torso length
                normalized[i, :3] = (positions[i] - hip_center) / scale_factor
            else:
                # For low-visibility joints, use interpolated position or zero
                # We still need to estimate position even if visibility is low
                # Try to estimate from neighboring joints
                estimated_position = self._estimate_position(i, positions, visibility, visibility_threshold)
                normalized[i, :3] = (estimated_position - hip_center) / scale_factor
        
        # Optional: Apply smoothing filter (simple moving average)
        # This helps reduce jitter in real-time applications
        if hasattr(self, '_prev_normalized'):
            alpha = 0.3  # Smoothing factor
            normalized[:, :3] = alpha * normalized[:, :3] + (1 - alpha) * self._prev_normalized[:, :3]
            # Visibility doesn't get smoothed
            normalized[:, 3] = landmarks[:, 3]
        
        self._prev_normalized = normalized.copy()
        
        return normalized
    
    def _estimate_position(self, 
                          index: int, 
                          positions: np.ndarray, 
                          visibility: np.ndarray,
                          threshold: float) -> np.ndarray:
        """
        Estimate position for low-visibility joints using neighboring joints.
        
        Args:
            index: Index of joint to estimate
            positions: All joint positions
            visibility: All joint visibilities
            threshold: Visibility threshold
            
        Returns:
            Estimated position for the joint
        """
        # Define joint neighbors for estimation
        neighbor_groups = {
            # Left side
            11: [12, 13, 23],  # left_shoulder from right_shoulder, left_elbow, left_hip
            13: [11, 15, 23],  # left_elbow from left_shoulder, left_wrist, left_hip
            15: [13, 19, 21],  # left_wrist from left_elbow, left_index, left_thumb
            23: [11, 24, 25],  # left_hip from left_shoulder, right_hip, left_knee
            25: [23, 27, 26],  # left_knee from left_hip, left_ankle, right_knee
            27: [25, 29, 31],  # left_ankle from left_knee, left_heel, left_foot_index
            
            # Right side
            12: [11, 14, 24],  # right_shoulder from left_shoulder, right_elbow, right_hip
            14: [12, 16, 24],  # right_elbow from right_shoulder, right_wrist, right_hip
            16: [14, 20, 22],  # right_wrist from right_elbow, right_index, right_thumb
            24: [12, 23, 26],  # right_hip from right_shoulder, left_hip, right_knee
            26: [24, 28, 25],  # right_knee from right_hip, right_ankle, left_knee
            28: [26, 30, 32],  # right_ankle from right_knee, right_heel, right_foot_index
            
            # Symmetrical joints can use opposite side
            0: [1, 4, 7, 8],   # nose from eyes and ears
            1: [2, 4, 0],      # left_eye_inner from left_eye, right_eye_inner, nose
            2: [1, 3, 0],      # left_eye from left_eye_inner, left_eye_outer, nose
            # Add more as needed...
        }
        
        if index in neighbor_groups:
            neighbors = neighbor_groups[index]
            visible_neighbors = []
            
            for neighbor_idx in neighbors:
                if visibility[neighbor_idx] >= threshold:
                    visible_neighbors.append(positions[neighbor_idx])
            
            if visible_neighbors:
                # Average of visible neighbors
                return np.mean(visible_neighbors, axis=0)
        
        # Fallback: return zero
        return np.zeros(3, dtype=np.float32)
    
    def draw_landmarks(self, image: np.ndarray, 
                      pose_result: Optional[PoseResult] = None,
                      draw_connections: bool = True,
                      landmark_color: Tuple[int, int, int] = (245, 117, 66),
                      connection_color: Tuple[int, int, int] = (245, 66, 230),
                      draw_visibility: bool = False) -> np.ndarray:
        """
        Draw landmarks on image using pre-computed MediaPipe results.
        
        Args:
            image: Original BGR image
            pose_result: Pre-computed pose result from process_frame()
            draw_connections: Whether to draw skeleton connections
            landmark_color: RGB color for landmarks
            connection_color: RGB color for connections
            draw_visibility: Whether to color-code landmarks by visibility
            
        Returns:
            Annotated image with landmarks
        """
        annotated_image = image.copy()
        
        if pose_result and pose_result.mp_results.pose_landmarks:
            # Use pre-computed MediaPipe results
            if draw_visibility and hasattr(pose_result, 'landmarks'):
                # Color-code by visibility
                self._draw_landmarks_with_visibility(
                    annotated_image,
                    pose_result.mp_results.pose_landmarks,
                    pose_result.landmarks[:, 3],  # Visibility array
                    draw_connections
                )
            else:
                # Standard drawing
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    pose_result.mp_results.pose_landmarks,
                    self.POSE_CONNECTIONS if draw_connections else None,
                    self.mp_drawing.DrawingSpec(
                        color=landmark_color, 
                        thickness=2, 
                        circle_radius=2
                    ),
                    self.mp_drawing.DrawingSpec(
                        color=connection_color, 
                        thickness=2, 
                        circle_radius=2
                    ) if draw_connections else None
                )
            
            # Add confidence text for debugging
            if hasattr(pose_result.mp_results, 'pose_world_landmarks'):
                # Extract average confidence from world landmarks
                world_landmarks = pose_result.mp_results.pose_world_landmarks.landmark
                avg_visibility = np.mean([lm.visibility for lm in world_landmarks])
                cv2.putText(
                    annotated_image,
                    f"Conf: {avg_visibility:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
        
        return annotated_image
    
    def _draw_landmarks_with_visibility(self, 
                                       image: np.ndarray,
                                       landmarks,
                                       visibility: np.ndarray,
                                       draw_connections: bool):
        """
        Draw landmarks with color coding based on visibility.
        
        Args:
            image: Image to draw on
            landmarks: MediaPipe landmarks object
            visibility: Visibility values (0-1)
            draw_connections: Whether to draw connections
        """
        # Convert to list for iteration
        landmark_list = landmarks.landmark
        
        # Draw connections first (fainter for low visibility)
        if draw_connections:
            for connection in self.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                
                # Get visibility of both endpoints
                vis_start = visibility[start_idx]
                vis_end = visibility[end_idx]
                avg_visibility = (vis_start + vis_end) / 2
                
                if avg_visibility > 0.3:  # Only draw if reasonably visible
                    # Fade color based on visibility
                    intensity = int(avg_visibility * 255)
                    color = (intensity, intensity, 255)  # Blue scale
                    
                    # Get image coordinates
                    h, w = image.shape[:2]
                    start_point = (
                        int(landmark_list[start_idx].x * w),
                        int(landmark_list[start_idx].y * h)
                    )
                    end_point = (
                        int(landmark_list[end_idx].x * w),
                        int(landmark_list[end_idx].y * h)
                    )
                    
                    cv2.line(image, start_point, end_point, color, 2)
        
        # Draw landmarks
        for i, landmark in enumerate(landmark_list):
            vis = visibility[i]
            
            if vis > 0.3:  # Only draw if reasonably visible
                # Color based on visibility (green = high, red = low)
                color = (
                    int(255 * (1 - vis)),  # Red component (inverse of visibility)
                    int(255 * vis),        # Green component (proportional to visibility)
                    100                     # Blue component (constant)
                )
                
                # Get image coordinates
                h, w = image.shape[:2]
                center = (
                    int(landmark.x * w),
                    int(landmark.y * h)
                )
                
                # Draw circle
                radius = int(3 + vis * 3)  # Size based on visibility
                cv2.circle(image, center, radius, color, -1)
                
                # Draw visibility value
                if vis < 0.7:  # Only show for low-visibility points
                    cv2.putText(
                        image,
                        f"{vis:.1f}",
                        (center[0] + 5, center[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1
                    )
    
    def get_landmark_names(self) -> Dict[str, int]:
        """Get mapping of landmark names to indices."""
        return self.LANDMARK_INDICES.copy()
    
    def get_keypoint_visibility(self, landmarks: np.ndarray, 
                               threshold: float = 0.5) -> np.ndarray:
        """
        Extract binary visibility mask from landmarks.
        
        Args:
            landmarks: Raw landmarks from MediaPipe (33, 4)
            threshold: Visibility threshold
            
        Returns:
            Boolean array of shape (33,) indicating visible keypoints
        """
        return landmarks[:, 3] >= threshold
    
    def get_body_dimensions(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Calculate body dimensions from normalized landmarks.
        Useful for gesture analysis.
        
        Args:
            landmarks: Normalized landmarks (33, 4) WITH visibility
            
        Returns:
            Dictionary of body measurements
        """
        idx = self.LANDMARK_INDICES
        
        # Use only visible points
        visibility = landmarks[:, 3]
        
        # Calculate various distances (only if points are visible)
        measurements = {}
        
        # Shoulder width
        if visibility[idx['left_shoulder']] > 0.5 and visibility[idx['right_shoulder']] > 0.5:
            measurements['shoulder_width'] = np.linalg.norm(
                landmarks[idx['left_shoulder'], :3] - landmarks[idx['right_shoulder'], :3]
            )
        
        # Hip width
        if visibility[idx['left_hip']] > 0.5 and visibility[idx['right_hip']] > 0.5:
            measurements['hip_width'] = np.linalg.norm(
                landmarks[idx['left_hip'], :3] - landmarks[idx['right_hip'], :3]
            )
        
        # Torso length (between shoulder center and hip center)
        shoulder_center = None
        hip_center = None
        
        # Calculate shoulder center if at least one shoulder is visible
        shoulder_positions = []
        for shoulder_idx in [idx['left_shoulder'], idx['right_shoulder']]:
            if visibility[shoulder_idx] > 0.5:
                shoulder_positions.append(landmarks[shoulder_idx, :3])
        
        if shoulder_positions:
            shoulder_center = np.mean(shoulder_positions, axis=0)
        
        # Calculate hip center if at least one hip is visible
        hip_positions = []
        for hip_idx in [idx['left_hip'], idx['right_hip']]:
            if visibility[hip_idx] > 0.5:
                hip_positions.append(landmarks[hip_idx, :3])
        
        if hip_positions:
            hip_center = np.mean(hip_positions, axis=0)
        
        if shoulder_center is not None and hip_center is not None:
            measurements['torso_length'] = np.linalg.norm(shoulder_center - hip_center)
        
        # Arm lengths (only if all joints in chain are visible)
        left_arm_visible = all(visibility[idx[joint]] > 0.5 
                              for joint in ['left_shoulder', 'left_elbow', 'left_wrist'])
        right_arm_visible = all(visibility[idx[joint]] > 0.5 
                               for joint in ['right_shoulder', 'right_elbow', 'right_wrist'])
        
        if left_arm_visible:
            measurements['left_arm_length'] = (
                np.linalg.norm(landmarks[idx['left_shoulder'], :3] - landmarks[idx['left_elbow'], :3]) +
                np.linalg.norm(landmarks[idx['left_elbow'], :3] - landmarks[idx['left_wrist'], :3])
            )
        
        if right_arm_visible:
            measurements['right_arm_length'] = (
                np.linalg.norm(landmarks[idx['right_shoulder'], :3] - landmarks[idx['right_elbow'], :3]) +
                np.linalg.norm(landmarks[idx['right_elbow'], :3] - landmarks[idx['right_wrist'], :3])
            )
        
        # Add default values for missing measurements
        defaults = {
            'shoulder_width': 1.0,
            'hip_width': 0.8,
            'torso_length': 1.0,
            'left_arm_length': 1.2,
            'right_arm_length': 1.2,
            'arm_length_ratio': 1.0
        }
        
        for key, default in defaults.items():
            if key not in measurements:
                measurements[key] = default
        
        # Calculate arm length ratio
        measurements['arm_length_ratio'] = (
            measurements['left_arm_length'] / measurements['right_arm_length'] 
            if measurements['right_arm_length'] > 0 else 1.0
        )
        
        return measurements
    
    def reset(self):
        """Reset internal state (e.g., smoothing buffers)."""
        if hasattr(self, '_prev_normalized'):
            delattr(self, '_prev_normalized')
    
    def close(self):
        """Clean up resources."""
        self.pose.close()
        self.reset()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# ============================================================================
# BATCH PROCESSING UTILITY
# ============================================================================

class BatchPoseProcessor:
    """
    Utility for batch processing multiple frames or video sequences.
    More efficient than single-frame processing.
    """
    
    def __init__(self, pose_extractor: PoseExtractor):
        """
        Args:
            pose_extractor: Initialized PoseExtractor instance
        """
        self.extractor = pose_extractor
    
    def process_batch(self, images: List[np.ndarray]) -> List[Optional[PoseResult]]:
        """
        Process a batch of images.
        
        Args:
            images: List of BGR images
            
        Returns:
            List of PoseResults (None for frames with no pose)
        """
        results = []
        
        for image in images:
            result = self.extractor.process_frame(image)
            results.append(result)
        
        return results
    
    def extract_landmarks_batch(self, images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Extract landmarks from a batch of images.
        
        Args:
            images: List of BGR images
            
        Returns:
            List of normalized landmarks (33, 4) or None
        """
        results = []
        
        for image in images:
            landmarks = self.extractor.extract_landmarks(image)
            results.append(landmarks)
        
        return results
    
    def create_sequence(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Create a landmark sequence from batch of images.
        
        Args:
            images: List of BGR images (sequential frames)
            
        Returns:
            Landmark sequence of shape (T, 33, 4) or None if no poses detected
        """
        landmarks_batch = self.extract_landmarks_batch(images)
        
        # Filter out None results
        valid_landmarks = [lm for lm in landmarks_batch if lm is not None]
        
        if not valid_landmarks:
            return None
        
        # Stack into sequence
        sequence = np.stack(valid_landmarks, axis=0)
        
        return sequence


# ============================================================================
# TEST FUNCTION WITH FIXED VISIBILITY HANDLING
# ============================================================================

def test_fixed_pose_extractor():
    """Test the fixed pose extractor with visibility handling."""
    import time
    
    print("Testing Fixed Pose Extractor with Visibility...")
    
    # Initialize extractor
    extractor = PoseExtractor(use_torso_length=True)
    
    # Create a test image (or use webcam)
    # For this test, we'll create a synthetic image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (100, 100, 100)  # Gray background
    
    # Draw a simple stick figure (simulating a person)
    cv2.circle(test_image, (320, 150), 20, (255, 255, 255), -1)  # Head
    cv2.line(test_image, (320, 170), (320, 270), (255, 255, 255), 5)  # Body
    cv2.line(test_image, (320, 200), (270, 150), (255, 255, 255), 5)  # Left arm
    cv2.line(test_image, (320, 200), (370, 150), (255, 255, 255), 5)  # Right arm
    cv2.line(test_image, (320, 270), (280, 350), (255, 255, 255), 5)  # Left leg
    cv2.line(test_image, (320, 270), (360, 350), (255, 255, 255), 5)  # Right leg
    
    # Process the frame
    pose_result = extractor.process_frame(test_image)
    
    if pose_result:
        print(f"✅ Pose detected successfully")
        print(f"   Normalized landmarks shape: {pose_result.landmarks.shape}")
        print(f"   Expected: (33, 4) WITH visibility")
        
        # Check shapes
        if pose_result.landmarks.shape == (33, 4):
            print("   ✅ Correct shape with visibility")
            
            # Check visibility values
            visibility = pose_result.landmarks[:, 3]
            print(f"   Visibility range: [{visibility.min():.3f}, {visibility.max():.3f}]")
            print(f"   Mean visibility: {visibility.mean():.3f}")
            
            # Check that positions are normalized
            positions = pose_result.landmarks[:, :3]
            print(f"   Position range: [{positions.min():.3f}, {positions.max():.3f}]")
            
            # Test FeatureEngineer compatibility
            try:
                from feature_engineer import FeatureEngineer, FeatureConfig
                
                print("\n🧪 Testing FeatureEngineer compatibility...")
                
                # Create a sequence from single frame
                sequence = np.stack([pose_result.landmarks], axis=0)
                print(f"   Sequence shape: {sequence.shape}")
                
                # Initialize FeatureEngineer
                config = FeatureConfig(compute_idle_features=False)
                engineer = FeatureEngineer(config)
                
                # Extract features (should not crash now)
                features = engineer.extract_features(sequence)
                print(f"   Features extracted: {features.shape}")
                print(f"   Feature validation: {engineer.validate_features(features)}")
                
                print("   ✅ FeatureEngineer compatibility confirmed!")
                
            except ImportError:
                print("   ⚠️  FeatureEngineer not available for test")
            except Exception as e:
                print(f"   ❌ FeatureEngineer test failed: {e}")
        
        else:
            print(f"   ❌ Incorrect shape: {pose_result.landmarks.shape}")
        
        # Test visualization
        annotated = extractor.draw_landmarks(
            test_image, 
            pose_result, 
            draw_visibility=True
        )
        
        cv2.imshow('Fixed Pose Extractor Test', annotated)
        cv2.waitKey(2000)  # Wait 2 seconds
        cv2.destroyAllWindows()
        
    else:
        print("❌ No pose detected in test image")
    
    # Test with webcam if available
    print("\n🎥 Testing with webcam (press 'q' to quit)...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("   ⚠️  Webcam not available, skipping live test")
    else:
        frame_count = 0
        start_time = time.time()
        
        while frame_count < 100:  # Test 100 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result = extractor.process_frame(frame)
            
            if result:
                # Draw with visibility coloring
                annotated = extractor.draw_landmarks(
                    frame, 
                    result, 
                    draw_visibility=True
                )
                
                # Show stats
                visibility = result.landmarks[:, 3]
                avg_vis = visibility.mean()
                cv2.putText(
                    annotated,
                    f"Avg Vis: {avg_vis:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                cv2.imshow('Webcam Test (Fixed)', annotated)
            else:
                cv2.putText(
                    frame,
                    "No pose detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                cv2.imshow('Webcam Test (Fixed)', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"   Processed {frame_count} frames in {elapsed:.2f}s")
        print(f"   Average FPS: {frame_count/elapsed:.1f}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    extractor.close()
    print("\n✅ Fixed pose extractor test complete!")


def test_batch_processing():
    """Test batch processing functionality."""
    print("\n🧪 Testing batch processing...")
    
    # Create some test images
    test_images = []
    for i in range(5):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        img[:] = (i * 40, i * 40, i * 40)  # Varying brightness
        
        # Add a simple shape
        cv2.circle(img, (160, 120), 30 + i * 5, (255, 255, 255), -1)
        test_images.append(img)
    
    # Initialize
    extractor = PoseExtractor()
    batch_processor = BatchPoseProcessor(extractor)
    
    # Process batch
    results = batch_processor.process_batch(test_images)
    
    valid_results = [r for r in results if r is not None]
    print(f"   Images processed: {len(test_images)}")
    print(f"   Poses detected: {len(valid_results)}")
    
    # Create sequence
    sequence = batch_processor.create_sequence(test_images)
    
    if sequence is not None:
        print(f"   Sequence shape: {sequence.shape}")
        print(f"   Expected: (T, 33, 4) where T <= {len(test_images)}")
        
        if sequence.shape[1:] == (33, 4):
            print("   ✅ Batch processing works correctly!")
        else:
            print(f"   ❌ Incorrect sequence shape: {sequence.shape}")
    else:
        print("   ⚠️  No sequence created (no poses detected)")
    
    extractor.close()


if __name__ == "__main__":
    test_fixed_pose_extractor()
    test_batch_processing()
    
    print("\n" + "=" * 60)
    print("✅ ALL POSE EXTRACTOR TESTS COMPLETE!")
    print("=" * 60)