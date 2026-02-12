"""
MediaPipe 0.10+ Tasks API Pose Extractor for Windows.
Uses the modern PoseLandmarker API (supports Python 3.10+ on Windows).
Single inference per frame, robust normalization.

CRITICAL FIXES:
1. ✅ VIDEO mode requires detect_for_video() with timestamp
2. ✅ Correct Image class: mp.Image, not vision.Image
3. ✅ Monotonic real timestamps (not assumed FPS) for VIDEO mode stability
4. ✅ Removed frame_count - timestamps based on actual elapsed time
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union
from dataclasses import dataclass
import os
import time

# MediaPipe Tasks API (0.10+)
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    MP_TASKS_AVAILABLE = True
except ImportError:
    print("⚠️  MediaPipe Tasks API not available. Install: pip install mediapipe")
    MP_TASKS_AVAILABLE = False


@dataclass
class PoseResult:
    """Container for pose extraction results."""
    landmarks: np.ndarray  # Normalized landmarks WITH visibility (33, 4)
    raw_landmarks: np.ndarray  # Original MediaPipe landmarks (33, 4)
    mp_results: Any  # MediaPipe results object
    image_shape: Tuple[int, int]  # (height, width)
    normalized_positions: np.ndarray  # Normalized positions only (33, 3) - for compatibility
    timestamp_ms: int  # Monotonic real timestamp for video mode


class PoseExtractor:
    """
    Extracts and normalizes body landmarks using MediaPipe Tasks API.
    Compatible with Windows Python 3.10+ (MediaPipe 0.10+).
    """
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5,
                 model_complexity: int = 1,
                 use_torso_length: bool = True,
                 model_path: Optional[str] = None):
        """
        Initialize PoseLandmarker with Tasks API.
        
        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            min_presence_confidence: Minimum confidence for presence
            model_complexity: 0=light, 1=full, 2=heavy
            use_torso_length: Use torso length for normalization
            model_path: Optional path to custom model file
        """
        if not MP_TASKS_AVAILABLE:
            raise ImportError(
                "MediaPipe Tasks API not installed.\n"
                "Install with: pip install mediapipe\n"
                "Requires MediaPipe 0.10.0+ for Windows Python 3.10+"
            )
        
        self.use_torso_length = use_torso_length
        self.min_visibility = min_detection_confidence
        self.model_complexity = model_complexity
        
        # CRITICAL FIX: Store start time for monotonic timestamps
        self._start_time = time.time()
        
        # Initialize PoseLandmarker with Tasks API
        try:
            # Determine model path
            if model_path is None:
                # Use bundled model based on complexity
                if model_complexity == 0:
                    model_name = "pose_landmarker_lite.task"
                elif model_complexity == 1:
                    model_name = "pose_landmarker_full.task"
                else:
                    model_name = "pose_landmarker_heavy.task"
                
                # Check if model exists locally, otherwise use bundled
                self.model_path = self._get_model_path(model_name)
            else:
                self.model_path = model_path
            
            # Create PoseLandmarker options
            options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=self.model_path),
                running_mode=vision.RunningMode.VIDEO,  # VIDEO mode for real-time
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=min_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
                num_poses=1
            )
            
            # Create landmarker
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            print(f"✅ PoseLandmarker initialized with model: {os.path.basename(self.model_path)}")
            print(f"   Running mode: VIDEO (requires detect_for_video with real timestamps)")
            
        except Exception as e:
            print(f"❌ Failed to initialize PoseLandmarker: {e}")
            print("\nTROUBLESHOOTING:")
            print("1. MediaPipe 0.10+ requires the .task model files")
            print("2. Run this to download models:")
            print("   python -c \"from mediapipe.tasks import python; from mediapipe.tasks.python import vision;")
            print("   options = vision.PoseLandmarkerOptions(")
            print("       base_options=python.BaseOptions(model_asset_path='pose_landmarker_full.task'),")
            print("       running_mode=vision.RunningMode.VIDEO));")
            print("   vision.PoseLandmarker.create_from_options(options)\"")
            raise
        
        # Landmark indices (same as MediaPipe Pose)
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
        
        # Connection pairs for drawing
        self.POSE_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (27, 31),
            (29, 31), (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
        ]
        
        # Smoothing buffer
        self._prev_normalized = None
    
    def _get_model_path(self, model_name: str) -> str:
        """
        Get path to model file. Downloads if not present.
        
        Args:
            model_name: Name of the model file
            
        Returns:
            Path to model file
        """
        # Check common locations
        possible_paths = [
            model_name,
            os.path.join("models", model_name),
            os.path.join("mediapipe", "models", model_name),
            os.path.join(os.path.dirname(__file__), "models", model_name),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found model at: {path}")
                return path
        
        # If not found, try to download using MediaPipe's helper
        print(f"⚠️  Model '{model_name}' not found. Attempting to download...")
        
        try:
            # Create models directory
            os.makedirs("models", exist_ok=True)
            model_path = os.path.join("models", model_name)
            
            # MediaPipe 0.10+ can download models automatically
            options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.VIDEO
            )
            # This will trigger download if needed
            _ = vision.PoseLandmarker.create_from_options(options)
            
            print(f"✅ Model downloaded to: {model_path}")
            return model_path
            
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            print("\nPlease download manually:")
            print(f"1. Visit: https://storage.googleapis.com/mediapipe-models/pose_landmarker/{model_name}")
            print(f"2. Save to: {os.path.abspath('models')}/")
            raise
    
    def process_frame(self, image: np.ndarray) -> Optional[PoseResult]:
        """
        Single-pass processing using PoseLandmarker Tasks API in VIDEO mode.
        
        CRITICAL FIXES:
        1. Uses detect_for_video() with timestamp_ms (required for VIDEO mode)
        2. Uses mp.Image, not vision.Image (correct class)
        3. Uses monotonic real timestamps (not assumed FPS) for stability
        
        Args:
            image: BGR image frame from camera
            
        Returns:
            PoseResult containing landmarks and results, or None if no pose detected
        """
        if not MP_TASKS_AVAILABLE:
            return None
        
        try:
            # Convert BGR to RGB (MediaPipe Tasks expects RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # CRITICAL FIX: Use mp.Image, NOT vision.Image
            # vision.Image does not exist in MediaPipe Tasks API
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # CRITICAL FIX: VIDEO mode requires monotonic real timestamps
            # Calculate milliseconds since start - this handles any FPS variation
            timestamp_ms = int((time.time() - self._start_time) * 1000)
            
            # CRITICAL FIX: Use detect_for_video() for VIDEO mode
            detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
                return None
            
            # Get first pose landmarks
            pose_landmarks = detection_result.pose_landmarks[0]
            
            # Convert to numpy array with visibility
            raw_landmarks = self._landmarks_to_array(pose_landmarks)
            
            # Normalize landmarks (keeping visibility)
            normalized_landmarks = self._normalize_landmarks_with_visibility(raw_landmarks)
            
            # Extract positions only for backward compatibility
            normalized_positions = normalized_landmarks[:, :3]
            
            return PoseResult(
                landmarks=normalized_landmarks,  # (33, 4) with visibility
                raw_landmarks=raw_landmarks,     # (33, 4) original
                mp_results=detection_result,
                image_shape=image.shape[:2],      # (height, width)
                normalized_positions=normalized_positions,  # (33, 3) positions only
                timestamp_ms=timestamp_ms         # Monotonic real timestamp
            )
            
        except Exception as e:
            print(f"⚠️  Frame processing error: {e}")
            return None
    
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
        """
        result = self.process_frame(image)
        return result.normalized_positions if result else None
    
    def _landmarks_to_array(self, pose_landmarks) -> np.ndarray:
        """
        Convert MediaPipe landmarks to numpy array with visibility.
        
        Args:
            pose_landmarks: NormalizedLandmarkList from detection_result
            
        Returns:
            Array of shape (33, 4) with [x, y, z, visibility]
        """
        landmarks = []
        for lm in pose_landmarks:
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
            # Use torso vector length for scaling
            shoulder_center = np.zeros(3, dtype=np.float32)
            shoulder_count = 0
            
            for idx in [left_shoulder_idx, right_shoulder_idx]:
                if visibility[idx] >= visibility_threshold:
                    shoulder_center += positions[idx]
                    shoulder_count += 1
            
            if shoulder_count == 0:
                scale_factor = 1.0
            else:
                shoulder_center /= shoulder_count
                torso_vector = shoulder_center - hip_center
                torso_length = np.linalg.norm(torso_vector)
                scale_factor = torso_length if torso_length > 1e-6 else 1.0
        else:
            # Original shoulder width method
            if (visibility[left_shoulder_idx] >= visibility_threshold and 
                visibility[right_shoulder_idx] >= visibility_threshold):
                shoulder_width = np.linalg.norm(
                    positions[left_shoulder_idx] - positions[right_shoulder_idx]
                )
                scale_factor = shoulder_width if shoulder_width > 1e-6 else 1.0
            else:
                scale_factor = 1.0
        
        # Normalize positions
        normalized = np.zeros((33, 4), dtype=np.float32)
        
        for i in range(len(positions)):
            # Copy visibility from original
            normalized[i, 3] = visibility[i]
            
            if visibility[i] >= visibility_threshold:
                # Translate to hip center and scale
                normalized[i, :3] = (positions[i] - hip_center) / scale_factor
            else:
                # Estimate position for low-visibility joints
                estimated_position = self._estimate_position(i, positions, visibility, visibility_threshold)
                normalized[i, :3] = (estimated_position - hip_center) / scale_factor
        
        # Apply temporal smoothing
        if self._prev_normalized is not None:
            alpha = 0.3  # Smoothing factor
            normalized[:, :3] = alpha * normalized[:, :3] + (1 - alpha) * self._prev_normalized[:, :3]
        
        self._prev_normalized = normalized.copy()
        
        return normalized
    
    def _estimate_position(self, 
                          index: int, 
                          positions: np.ndarray, 
                          visibility: np.ndarray,
                          threshold: float) -> np.ndarray:
        """
        Estimate position for low-visibility joints using neighboring joints.
        """
        # Define joint neighbors
        neighbor_groups = {
            11: [12, 13, 23],  # left_shoulder
            13: [11, 15, 23],  # left_elbow
            15: [13, 19, 21],  # left_wrist
            23: [11, 24, 25],  # left_hip
            25: [23, 27, 26],  # left_knee
            27: [25, 29, 31],  # left_ankle
            12: [11, 14, 24],  # right_shoulder
            14: [12, 16, 24],  # right_elbow
            16: [14, 20, 22],  # right_wrist
            24: [12, 23, 26],  # right_hip
            26: [24, 28, 25],  # right_knee
            28: [26, 30, 32],  # right_ankle
            0: [1, 4, 7, 8],   # nose
            1: [2, 4, 0],      # left_eye_inner
            2: [1, 3, 0],      # left_eye
        }
        
        if index in neighbor_groups:
            neighbors = neighbor_groups[index]
            visible_neighbors = []
            
            for neighbor_idx in neighbors:
                if visibility[neighbor_idx] >= threshold:
                    visible_neighbors.append(positions[neighbor_idx])
            
            if visible_neighbors:
                return np.mean(visible_neighbors, axis=0)
        
        return np.zeros(3, dtype=np.float32)
    
    def draw_landmarks(self, 
                      image: np.ndarray, 
                      pose_result: Optional[PoseResult] = None,
                      draw_connections: bool = True,
                      landmark_color: Tuple[int, int, int] = (245, 117, 66),
                      connection_color: Tuple[int, int, int] = (245, 66, 230),
                      draw_visibility: bool = False) -> np.ndarray:
        """
        Draw landmarks on image.
        """
        annotated_image = image.copy()
        
        if pose_result and pose_result.mp_results.pose_landmarks:
            # Get pose landmarks from result
            pose_landmarks = pose_result.mp_results.pose_landmarks[0]
            
            if draw_visibility and hasattr(pose_result, 'landmarks'):
                # Color-code by visibility
                self._draw_landmarks_with_visibility(
                    annotated_image,
                    pose_landmarks,
                    pose_result.landmarks[:, 3],
                    draw_connections
                )
            else:
                # Standard drawing
                h, w = image.shape[:2]
                
                # Draw connections
                if draw_connections:
                    for connection in self.POSE_CONNECTIONS:
                        start_idx, end_idx = connection
                        
                        start_point = pose_landmarks[start_idx]
                        end_point = pose_landmarks[end_idx]
                        
                        # Convert normalized coordinates to pixel coordinates
                        start_pixel = (int(start_point.x * w), int(start_point.y * h))
                        end_pixel = (int(end_point.x * w), int(end_point.y * h))
                        
                        cv2.line(annotated_image, start_pixel, end_pixel, connection_color, 2)
                
                # Draw landmarks
                for landmark in pose_landmarks:
                    pixel_x = int(landmark.x * w)
                    pixel_y = int(landmark.y * h)
                    
                    cv2.circle(annotated_image, (pixel_x, pixel_y), 5, landmark_color, -1)
            
            # Add confidence text
            avg_visibility = np.mean(pose_result.landmarks[:, 3])
            cv2.putText(
                annotated_image,
                f"Conf: {avg_visibility:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Add timestamp info
            if hasattr(pose_result, 'timestamp_ms'):
                cv2.putText(
                    annotated_image,
                    f"Time: {pose_result.timestamp_ms}ms",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1
                )
        
        return annotated_image
    
    def _draw_landmarks_with_visibility(self, 
                                       image: np.ndarray,
                                       landmarks,
                                       visibility: np.ndarray,
                                       draw_connections: bool):
        """
        Draw landmarks with color coding based on visibility.
        """
        h, w = image.shape[:2]
        
        # Draw connections
        if draw_connections:
            for connection in self.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                
                if start_idx < len(visibility) and end_idx < len(visibility):
                    vis_start = visibility[start_idx]
                    vis_end = visibility[end_idx]
                    avg_visibility = (vis_start + vis_end) / 2
                    
                    if avg_visibility > 0.3:
                        intensity = int(avg_visibility * 255)
                        color = (intensity, intensity, 255)
                        
                        start_point = landmarks[start_idx]
                        end_point = landmarks[end_idx]
                        
                        start_pixel = (int(start_point.x * w), int(start_point.y * h))
                        end_pixel = (int(end_point.x * w), int(end_point.y * h))
                        
                        cv2.line(image, start_pixel, end_pixel, color, 2)
        
        # Draw landmarks
        for i, landmark in enumerate(landmarks):
            if i < len(visibility):
                vis = visibility[i]
                
                if vis > 0.3:
                    color = (
                        int(255 * (1 - vis)),
                        int(255 * vis),
                        100
                    )
                    
                    pixel_x = int(landmark.x * w)
                    pixel_y = int(landmark.y * h)
                    
                    radius = int(3 + vis * 3)
                    cv2.circle(image, (pixel_x, pixel_y), radius, color, -1)
                    
                    if vis < 0.7:
                        cv2.putText(
                            image,
                            f"{vis:.1f}",
                            (pixel_x + 5, pixel_y - 5),
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
        """Extract binary visibility mask from landmarks."""
        return landmarks[:, 3] >= threshold
    
    def get_body_dimensions(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate body dimensions from normalized landmarks."""
        idx = self.LANDMARK_INDICES
        visibility = landmarks[:, 3]
        
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
        
        # Torso length
        shoulder_center = None
        hip_center = None
        
        shoulder_positions = []
        for shoulder_idx in [idx['left_shoulder'], idx['right_shoulder']]:
            if visibility[shoulder_idx] > 0.5:
                shoulder_positions.append(landmarks[shoulder_idx, :3])
        
        if shoulder_positions:
            shoulder_center = np.mean(shoulder_positions, axis=0)
        
        hip_positions = []
        for hip_idx in [idx['left_hip'], idx['right_hip']]:
            if visibility[hip_idx] > 0.5:
                hip_positions.append(landmarks[hip_idx, :3])
        
        if hip_positions:
            hip_center = np.mean(hip_positions, axis=0)
        
        if shoulder_center is not None and hip_center is not None:
            measurements['torso_length'] = np.linalg.norm(shoulder_center - hip_center)
        
        # Arm lengths
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
        
        # Default values
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
        
        measurements['arm_length_ratio'] = (
            measurements['left_arm_length'] / measurements['right_arm_length'] 
            if measurements['right_arm_length'] > 0 else 1.0
        )
        
        return measurements
    
    def reset(self):
        """Reset internal state (smoothing buffers and start time)."""
        self._prev_normalized = None
        self._start_time = time.time()  # Reset timestamp baseline
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()
        self.reset()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================================
# BATCH PROCESSING UTILITY
# ============================================================================

class BatchPoseProcessor:
    """Batch processing for multiple frames."""
    
    def __init__(self, pose_extractor: PoseExtractor):
        self.extractor = pose_extractor
    
    def process_batch(self, images: List[np.ndarray]) -> List[Optional[PoseResult]]:
        results = []
        for image in images:
            result = self.extractor.process_frame(image)
            results.append(result)
        return results
    
    def extract_landmarks_batch(self, images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        results = []
        for image in images:
            landmarks = self.extractor.extract_landmarks(image)
            results.append(landmarks)
        return results
    
    def create_sequence(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        landmarks_batch = self.extract_landmarks_batch(images)
        valid_landmarks = [lm for lm in landmarks_batch if lm is not None]
        
        if not valid_landmarks:
            return None
        
        return np.stack(valid_landmarks, axis=0)


# ============================================================================
# QUICK INSTALLATION SCRIPT
# ============================================================================

def install_mediapipe_tasks():
    """Helper function to guide MediaPipe Tasks installation."""
    print("\n" + "=" * 60)
    print("📦 MEDIAPIPE TASKS INSTALLATION GUIDE (Windows Python 3.10+)")
    print("=" * 60)
    print("\n1. Install MediaPipe:")
    print("   pip install --upgrade mediapipe")
    print("\n2. Download pose landmarker model:")
    print("   python -c \"from mediapipe.tasks import python; from mediapipe.tasks.python import vision;")
    print("   options = vision.PoseLandmarkerOptions(")
    print("       base_options=python.BaseOptions(model_asset_path='pose_landmarker_full.task'),")
    print("       running_mode=vision.RunningMode.VIDEO);")
    print("   vision.PoseLandmarker.create_from_options(options)\"")
    print("\n3. Verify installation:")
    print("   python -c \"import mediapipe as mp; print(f'MediaPipe {mp.__version__}')\"")
    print("\n" + "=" * 60)


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_pose_extractor():
    """Test the Tasks API pose extractor."""
    if not MP_TASKS_AVAILABLE:
        install_mediapipe_tasks()
        return
    
    print("🧪 Testing PoseLandmarker (MediaPipe Tasks API)...")
    print("   Running mode: VIDEO")
    print("   Using: detect_for_video() with REAL timestamps")
    print("   Using: mp.Image (correct class)")
    print("   Timestamps: Monotonic (time since start)")
    
    try:
        # Initialize extractor
        extractor = PoseExtractor(
            model_complexity=1,  # Full model
            use_torso_length=True
        )
        
        # Test with webcam
        print("\n🎥 Testing with webcam (press 'q' to quit)...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("   ⚠️  Webcam not available")
            return
        
        frame_count = 0
        start_time = time.time()
        
        while frame_count < 100:  # Test 100 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result = extractor.process_frame(frame)
            
            if result:
                # Draw landmarks
                annotated = extractor.draw_landmarks(
                    frame,
                    result,
                    draw_visibility=True
                )
                
                # Show stats
                visibility = result.landmarks[:, 3]
                avg_vis = visibility.mean()
                fps = frame_count / (time.time() - start_time + 0.001)
                
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f} | Vis: {avg_vis:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                cv2.putText(
                    annotated,
                    f"Time: {result.timestamp_ms}ms",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1
                )
                
                cv2.imshow('MediaPipe Tasks API (VIDEO mode)', annotated)
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
                cv2.imshow('MediaPipe Tasks API (VIDEO mode)', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        elapsed = time.time() - start_time
        
        print(f"   Processed {frame_count} frames in {elapsed:.2f}s")
        print(f"   Average FPS: {frame_count/elapsed:.1f}")
        print(f"   Final timestamp: {int(elapsed * 1000)}ms")
        
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        
        print("\n✅ Pose extractor test complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        install_mediapipe_tasks()


if __name__ == "__main__":
    test_pose_extractor()