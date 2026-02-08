# pose_extractor.py
"""
Optimized real-time pose extraction using MediaPipe.
Single inference per frame, robust normalization.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class PoseResult:
    """Container for pose extraction results."""
    landmarks: np.ndarray  # Normalized landmarks (33, 4)
    raw_landmarks: np.ndarray  # Original MediaPipe landmarks (33, 4)
    mp_results: Any  # MediaPipe results object
    image_shape: Tuple[int, int]  # (height, width)

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
        
        # Normalize landmarks
        normalized_landmarks = self._normalize_landmarks_robust(raw_landmarks)
        
        return PoseResult(
            landmarks=normalized_landmarks,
            raw_landmarks=raw_landmarks,
            mp_results=mp_results,
            image_shape=image.shape[:2]  # (height, width)
        )
    
    def extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Legacy method for backward compatibility.
        Returns only normalized landmarks.
        """
        result = self.process_frame(image)
        return result.landmarks if result else None
    
    def _landmarks_to_array(self, pose_landmarks) -> np.ndarray:
        """Convert MediaPipe landmarks to numpy array."""
        landmarks = []
        for lm in pose_landmarks.landmark:
            # Store: [x, y, z, visibility]
            landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(landmarks)
    
    def _normalize_landmarks_robust(self, landmarks: np.ndarray, 
                                   visibility_threshold: float = 0.5) -> np.ndarray:
        """
        Robust normalization using torso length and handling low-visibility joints.
        
        Args:
            landmarks: Raw landmarks from MediaPipe (33, 4)
            visibility_threshold: Minimum visibility score to consider joint
            
        Returns:
            Normalized landmarks (33, 3) - visibility dropped for model input
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
        hip_center = np.zeros(3)
        hip_count = 0
        
        for idx in [left_hip_idx, right_hip_idx]:
            if visibility[idx] >= visibility_threshold:
                hip_center += positions[idx]
                hip_count += 1
        
        if hip_count == 0:
            # Fallback: use visible shoulders or default position
            hip_center = np.mean(positions[[left_shoulder_idx, right_shoulder_idx]], axis=0)
        else:
            hip_center /= hip_count
        
        # Calculate torso length (shoulder to hip)
        if self.use_torso_length:
            # Use torso vector length for scaling (more robust than shoulder width)
            shoulder_center = np.zeros(3)
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
        
        # Normalize positions
        normalized = np.zeros((33, 3), dtype=np.float32)
        
        for i in range(len(positions)):
            if visibility[i] >= visibility_threshold:
                # Translate to hip center and scale by torso length
                normalized[i] = (positions[i] - hip_center) / scale_factor
            else:
                # For low-visibility joints, use interpolated position or zero
                # Zero is fine since model will learn to ignore based on features
                normalized[i] = np.zeros(3)
        
        # Optional: Apply smoothing filter (simple moving average)
        # This helps reduce jitter in real-time applications
        if hasattr(self, '_prev_normalized'):
            alpha = 0.3  # Smoothing factor
            normalized = alpha * normalized + (1 - alpha) * self._prev_normalized
        
        self._prev_normalized = normalized.copy()
        
        return normalized
    
    def draw_landmarks(self, image: np.ndarray, 
                      pose_result: Optional[PoseResult] = None,
                      draw_connections: bool = True,
                      landmark_color: Tuple[int, int, int] = (245, 117, 66),
                      connection_color: Tuple[int, int, int] = (245, 66, 230)) -> np.ndarray:
        """
        Draw landmarks on image using pre-computed MediaPipe results.
        
        Args:
            image: Original BGR image
            pose_result: Pre-computed pose result from process_frame()
            draw_connections: Whether to draw skeleton connections
            landmark_color: RGB color for landmarks
            connection_color: RGB color for connections
            
        Returns:
            Annotated image with landmarks
        """
        annotated_image = image.copy()
        
        if pose_result and pose_result.mp_results.pose_landmarks:
            # Use pre-computed MediaPipe results
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
            landmarks: Normalized landmarks (33, 3)
            
        Returns:
            Dictionary of body measurements
        """
        idx = self.LANDMARK_INDICES
        
        # Calculate various distances
        shoulder_width = np.linalg.norm(
            landmarks[idx['left_shoulder']] - landmarks[idx['right_shoulder']]
        )
        
        hip_width = np.linalg.norm(
            landmarks[idx['left_hip']] - landmarks[idx['right_hip']]
        )
        
        torso_length = np.linalg.norm(
            (landmarks[idx['left_shoulder']] + landmarks[idx['right_shoulder']]) / 2 -
            (landmarks[idx['left_hip']] + landmarks[idx['right_hip']]) / 2
        )
        
        left_arm_length = (
            np.linalg.norm(landmarks[idx['left_shoulder']] - landmarks[idx['left_elbow']]) +
            np.linalg.norm(landmarks[idx['left_elbow']] - landmarks[idx['left_wrist']])
        )
        
        right_arm_length = (
            np.linalg.norm(landmarks[idx['right_shoulder']] - landmarks[idx['right_elbow']]) +
            np.linalg.norm(landmarks[idx['right_elbow']] - landmarks[idx['right_wrist']])
        )
        
        return {
            'shoulder_width': shoulder_width,
            'hip_width': hip_width,
            'torso_length': torso_length,
            'left_arm_length': left_arm_length,
            'right_arm_length': right_arm_length,
            'arm_length_ratio': left_arm_length / right_arm_length if right_arm_length > 0 else 1.0
        }
    
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


# Example usage and testing
def test_pose_extractor():
    """Test the optimized pose extractor."""
    import time
    
    # Initialize extractor
    extractor = PoseExtractor(use_torso_length=True)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    print("Testing optimized pose extractor. Press 'q' to quit.")
    
    frame_count = 0
    total_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.perf_counter()
        
        # Single inference per frame
        pose_result = extractor.process_frame(frame)
        
        inference_time = time.perf_counter() - start_time
        total_time += inference_time
        frame_count += 1
        
        if pose_result:
            # Use pre-computed results for drawing
            annotated = extractor.draw_landmarks(frame, pose_result)
            
            # Show normalization info
            dimensions = extractor.get_body_dimensions(pose_result.landmarks)
            info_text = f"Torso: {dimensions['torso_length']:.2f}, Shoulders: {dimensions['shoulder_width']:.2f}"
            cv2.putText(annotated, info_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            annotated = frame
        
        # Show performance metrics
        fps = 1.0 / inference_time if inference_time > 0 else 0
        avg_time = total_time / frame_count if frame_count > 0 else 0
        
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated, f"Avg: {avg_time*1000:.1f}ms", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow('Optimized Pose Extractor', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    
    if frame_count > 0:
        print(f"\nPerformance Summary:")
        print(f"  Frames processed: {frame_count}")
        print(f"  Average inference time: {avg_time*1000:.2f}ms")
        print(f"  Average FPS: {1.0/avg_time:.1f}" if avg_time > 0 else "N/A")

if __name__ == "__main__":
    test_pose_extractor()