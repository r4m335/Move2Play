"""
Robust gesture data collection with metadata and quality control.
Redefined "run" as jogging-in-place temporal pattern.

CRITICAL FIX APPLIED:
- Idle sequence now returns shape (33, 4) instead of (33, 3) to match feature engineer expectations
- All sequences maintain 4-channel format (x, y, z, visibility)
- Prevents silent dimension mismatch during feature extraction
"""

import os
import json
import time
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

# Import from config
from config import (
    DATA_DIR, SEQUENCE_LENGTH, MIN_SAMPLES_PER_GESTURE,
    GESTURE_CLASSES, CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT
)
from pose_extractor import PoseExtractor, PoseResult

# ============================================================================
# CONSTANTS FOR LANDMARK FORMAT
# ============================================================================

# MediaPipe Pose 33 landmarks format: [x, y, z, visibility]
NUM_LANDMARKS = 33
LANDMARK_DIM = 4  # x, y, z, visibility


@dataclass
class RecordingMetadata:
    """Metadata for a recorded gesture sequence."""
    # User/Session info
    user_id: str = "anonymous"
    session_id: str = ""
    
    # Recording info
    gesture: str = ""
    timestamp: str = ""
    duration_seconds: float = 0.0
    
    # Camera info
    camera_fps: float = 0.0
    camera_resolution: str = ""
    camera_index: int = 0
    
    # Sequence info
    total_frames: int = 0
    valid_frames: int = 0
    missing_frames: int = 0
    sequence_length: int = SEQUENCE_LENGTH
    
    # Quality metrics
    avg_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0
    frame_gap_indices: List[int] = None
    
    # Processing info
    pose_model: str = "mediapipe_pose_v1"
    normalization_method: str = "torso_length"
    
    # Jogging-specific metadata
    jogging_amplitude: float = 0.0  # Vertical movement range
    jogging_frequency: float = 0.0  # Steps per second
    hip_stability: float = 0.0  # Hip vertical stability (lower = more stable)
    
    def __post_init__(self):
        """Initialize defaults after dataclass creation."""
        if self.frame_gap_indices is None:
            self.frame_gap_indices = []
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.session_id:
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class GestureDataCollector:
    """Records high-quality gesture sequences with metadata and validation."""
    
    def __init__(self, 
                 output_dir: Path = DATA_DIR,
                 max_missing_frames: int = 3,  # Max consecutive missing frames
                 min_valid_frames: int = SEQUENCE_LENGTH - 5,  # Min valid frames per sequence
                 user_id: str = "anonymous"):
        """
        Args:
            output_dir: Directory to save collected data
            max_missing_frames: Maximum allowed consecutive frames without pose
            min_valid_frames: Minimum valid frames required for a sequence
            user_id: Identifier for the person recording gestures
        """
        self.output_dir = Path(output_dir)
        self.pose_extractor = PoseExtractor(use_torso_length=True)
        self.max_missing_frames = max_missing_frames
        self.min_valid_frames = min_valid_frames
        self.user_id = user_id
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        for gesture in GESTURE_CLASSES:
            (self.output_dir / gesture).mkdir(exist_ok=True, parents=True)
        
        # User metadata directory
        self.user_dir = self.output_dir / "users" / user_id
        self.user_dir.mkdir(exist_ok=True, parents=True)
        
        # Recording state
        self.current_sequence: List[np.ndarray] = []
        self.frame_timestamps: List[float] = []
        self.recording_metadata: Optional[RecordingMetadata] = None
        
        # Special instructions for jogging-in-place
        self.jogging_instructions = {
            'run': """🏃 JOGGING-IN-PLACE INSTRUCTIONS:
• Stand in place, feet shoulder-width apart
• Lift knees alternately (jogging motion)
• Swing arms naturally as if running
• Keep torso relatively stable
• Focus on VERTICAL movement, not forward motion
• Target: 2-3 steps per second rhythm""",
        }
        
        print(f"Gesture Data Collector initialized for user: {user_id}")
        print(f"Output directory: {self.output_dir}")
        print(f"Sequence length: {SEQUENCE_LENGTH} frames")
        print(f"Landmark format: {NUM_LANDMARKS} landmarks × {LANDMARK_DIM} channels")
        print(f"Allowed missing frames: {max_missing_frames}")
    
    def start_recording(self, 
                       gesture_name: str, 
                       camera_index: int = CAMERA_INDEX,
                       collect_multiple: bool = False,
                       auto_restart: bool = True):
        """
        Record sequences for a specific gesture.
        
        Args:
            gesture_name: Name of the gesture to record
            camera_index: Camera device index
            collect_multiple: If True, continue recording multiple sequences
            auto_restart: If True, automatically restart after each sequence
        """
        if gesture_name not in GESTURE_CLASSES:
            raise ValueError(
                f"Unknown gesture: '{gesture_name}'. "
                f"Valid gestures: {GESTURE_CLASSES}"
            )
        
        # Initialize camera
        cap = self._initialize_camera(camera_index)
        if cap is None:
            print(f"❌ Failed to initialize camera {camera_index}")
            return
        
        # Get camera info
        camera_fps = cap.get(cv2.CAP_PROP_FPS)
        if camera_fps <= 0:
            camera_fps = 30.0  # Default assumption
        
        camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        camera_resolution = f"{camera_width}x{camera_height}"
        
        print("\n" + "=" * 60)
        print(f"RECORDING GESTURE: {gesture_name.upper()}")
        print("=" * 60)
        
        # Show special instructions for jogging
        if gesture_name == 'run':
            print(self.jogging_instructions['run'])
            print("-" * 60)
        
        print(f"User: {self.user_id}")
        print(f"Camera: {camera_resolution} @ {camera_fps:.1f} FPS")
        print(f"Sequence length: {SEQUENCE_LENGTH} frames")
        print(f"Target: {MIN_SAMPLES_PER_GESTURE} samples per gesture")
        print("\nControls:")
        print("  'r' - Start/restart recording")
        print("  's' - Save current sequence and continue")
        print("  'q' - Quit recording")
        print("=" * 60)
        
        sequence_count = 0
        recording = False
        consecutive_missing = 0
        
        # For jogging analysis
        hip_positions = [] if gesture_name == 'run' else None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame from camera")
                break
            
            # Process frame with pose extractor
            pose_result = self.pose_extractor.process_frame(frame)
            
            # Get frame timestamp
            current_time = time.time()
            
            if recording:
                if pose_result is not None:
                    # Valid frame with pose detected
                    self.current_sequence.append(pose_result.landmarks)
                    self.frame_timestamps.append(current_time)
                    consecutive_missing = 0
                    
                    # Track hip positions for jogging analysis
                    if gesture_name == 'run' and hip_positions is not None:
                        # Extract hip center
                        landmarks = pose_result.landmarks
                        left_hip = landmarks[23]  # left_hip index
                        right_hip = landmarks[24]  # right_hip index
                        hip_center = (left_hip + right_hip) / 2
                        hip_positions.append(hip_center[1])  # y-coordinate (vertical)
                    
                    # Check if sequence is complete
                    if len(self.current_sequence) >= SEQUENCE_LENGTH:
                        # Calculate jogging metrics if this is a run gesture
                        jogging_metrics = None
                        if gesture_name == 'run' and hip_positions:
                            jogging_metrics = self._analyze_jogging_pattern(hip_positions, camera_fps)
                        
                        success = self._save_sequence(
                            gesture_name, 
                            camera_fps, 
                            camera_resolution,
                            camera_index,
                            jogging_metrics
                        )
                        if success:
                            sequence_count += 1
                        
                        if auto_restart and collect_multiple:
                            # Auto-restart for next sequence
                            self._reset_recording_state()
                            hip_positions = [] if gesture_name == 'run' else None
                            recording = False
                            print(f"\n✅ Sequence {sequence_count} saved. Ready for next...")
                        else:
                            recording = False
                else:
                    # No pose detected in frame
                    consecutive_missing += 1
                    self.frame_timestamps.append(current_time)  # Track gap
                    
                    # Abort if too many consecutive missing frames
                    if consecutive_missing > self.max_missing_frames:
                        print(f"❌ Aborting: {consecutive_missing} consecutive frames without pose")
                        self._reset_recording_state()
                        hip_positions = [] if gesture_name == 'run' else None
                        recording = False
                        consecutive_missing = 0
            
            # Draw UI
            annotated_frame = self._draw_ui(
                frame, 
                pose_result, 
                gesture_name, 
                recording,
                len(self.current_sequence),
                sequence_count,
                hip_positions if gesture_name == 'run' else None
            )
            
            cv2.imshow('Gesture Recording', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r') and not recording:
                # Start recording
                recording = True
                self._reset_recording_state()
                hip_positions = [] if gesture_name == 'run' else None
                consecutive_missing = 0
                print(f"🎥 Recording started... ({len(self.current_sequence)}/{SEQUENCE_LENGTH})")
            
            elif key == ord('s') and recording and len(self.current_sequence) > 0:
                # Calculate jogging metrics if this is a run gesture
                jogging_metrics = None
                if gesture_name == 'run' and hip_positions:
                    jogging_metrics = self._analyze_jogging_pattern(hip_positions, camera_fps)
                
                # Save current sequence manually
                success = self._save_sequence(
                    gesture_name, 
                    camera_fps, 
                    camera_resolution,
                    camera_index,
                    jogging_metrics
                )
                if success:
                    sequence_count += 1
                
                if collect_multiple and auto_restart:
                    self._reset_recording_state()
                    hip_positions = [] if gesture_name == 'run' else None
                    recording = False
                    print(f"\n✅ Sequence {sequence_count} saved. Ready for next...")
                else:
                    recording = False
            
            elif key == ord('q'):
                # Quit recording
                if recording and len(self.current_sequence) > 0:
                    response = input("\n⚠️  Recording in progress. Save before quitting? (y/n): ")
                    if response.lower() == 'y':
                        jogging_metrics = None
                        if gesture_name == 'run' and hip_positions:
                            jogging_metrics = self._analyze_jogging_pattern(hip_positions, camera_fps)
                        
                        self._save_sequence(
                            gesture_name, 
                            camera_fps, 
                            camera_resolution,
                            camera_index,
                            jogging_metrics
                        )
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Save session summary
        self._save_session_summary(gesture_name, sequence_count)
        
        print(f"\n✅ Recording complete for '{gesture_name}'")
        print(f"   Sequences collected: {sequence_count}")
        print(f"   User: {self.user_id}")
    
    def collect_idle_data(self, 
                     camera_index: int = CAMERA_INDEX,
                     duration_seconds: int = 60,
                     samples_per_session: int = 30):
        """
        Collect idle/background data (standing still, natural movements).
        
        CRITICAL FIX: Returns shape (33, 4) for landmarks to match feature engineer expectations.
        
        Args:
            camera_index: Camera device index
            duration_seconds: Total duration to collect idle data
            samples_per_session: Number of idle sequences to collect
        """
        print("\n" + "=" * 60)
        print("COLLECTING IDLE/BACKGROUND DATA")
        print("=" * 60)
        print("IMPORTANT: This prevents false positives!")
        print("=" * 60)
        print("INSTRUCTIONS:")
        print("1. Stand naturally in frame")
        print("2. Make small natural movements (breathing, shifting weight)")
        print("3. Don't perform any specific gestures")
        print("4. The system will automatically collect samples")
        print("=" * 60)
        
        # Initialize camera
        cap = self._initialize_camera(camera_index)
        if cap is None:
            print(f"❌ Failed to initialize camera {camera_index}")
            return
        
        # Get camera info
        camera_fps = cap.get(cv2.CAP_PROP_FPS)
        if camera_fps <= 0:
            camera_fps = 30.0
        
        camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        camera_resolution = f"{camera_width}x{camera_height}"
        
        gesture_name = "idle"
        sequence_count = 0
        start_time = time.time()
        
        # Create idle directory if it doesn't exist
        idle_dir = self.output_dir / "idle"
        idle_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Collecting {samples_per_session} idle sequences...")
        print("Press 'q' to stop early")
        
        while sequence_count < samples_per_session and (time.time() - start_time) < duration_seconds:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Process frame
            pose_result = self.pose_extractor.process_frame(frame)
            
            if pose_result is not None:
                # Randomly decide to save a sequence (simulating natural intervals)
                if np.random.random() < 0.02:  # 2% chance per frame
                    # Collect a sequence
                    current_sequence = []
                    frame_timestamps = []
                    
                    for _ in range(SEQUENCE_LENGTH):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        pose_result = self.pose_extractor.process_frame(frame)
                        if pose_result is not None:
                            current_sequence.append(pose_result.landmarks)
                            frame_timestamps.append(time.time())
                        
                        # Show live view
                        display_frame = self._draw_ui(
                            frame, pose_result, "idle", False, 
                            len(current_sequence), sequence_count, None
                        )
                        cv2.imshow('Collecting Idle Data', display_frame)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            cap.release()
                            cv2.destroyAllWindows()
                            return
                    
                    if len(current_sequence) >= self.min_valid_frames:
                        # CRITICAL FIX: Save idle sequence with 4 channels
                        sequence_array = self._prepare_idle_sequence(current_sequence)
                        
                        # Verify shape before saving
                        if sequence_array.shape[1:] != (NUM_LANDMARKS, LANDMARK_DIM):
                            print(f"⚠️  Warning: Unexpected shape {sequence_array.shape}, fixing...")
                            # Fix shape if necessary
                            sequence_array = self._fix_landmark_shape(sequence_array)
                        
                        # Create metadata
                        metadata = RecordingMetadata(
                            user_id=self.user_id,
                            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
                            gesture="idle",
                            timestamp=datetime.now().isoformat(),
                            duration_seconds=frame_timestamps[-1] - frame_timestamps[0] if frame_timestamps else 0.0,
                            camera_fps=camera_fps,
                            camera_resolution=camera_resolution,
                            camera_index=camera_index,
                            total_frames=len(frame_timestamps),
                            valid_frames=len(current_sequence),
                            missing_frames=len(frame_timestamps) - len(current_sequence),
                            sequence_length=SEQUENCE_LENGTH,
                            avg_confidence=0.8,  # Placeholder
                            min_confidence=0.7,
                            max_confidence=0.9,
                            frame_gap_indices=[],
                            pose_model="mediapipe_pose_v1",
                            normalization_method="torso_length"
                        )
                        
                        # Save
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        filename_base = f"idle_{self.user_id}_{timestamp}"
                        sequence_path = idle_dir / f"{filename_base}.npy"
                        
                        np.save(sequence_path, sequence_array)
                        
                        metadata_path = sequence_path.with_suffix('.json')
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata.to_dict(), f, indent=2, default=str)
                        
                        sequence_count += 1
                        print(f"✅ Saved idle sequence {sequence_count}/{samples_per_session}")
            
            # Show live view
            display_frame = self._draw_ui(
                frame, pose_result, "idle", False, 0, sequence_count, None
            )
            cv2.putText(display_frame, f"Idle samples: {sequence_count}/{samples_per_session}", 
                    (10, display_frame.shape[0] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow('Collecting Idle Data', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Collected {sequence_count} idle sequences")
        print(f"   Landmark format: {NUM_LANDMARKS}×{LANDMARK_DIM} (x, y, z, visibility)")
        print("This will significantly reduce false positives!")

    def _prepare_idle_sequence(self, sequence: list) -> np.ndarray:
        """
        Prepare idle sequence array.
        
        CRITICAL FIX: Returns shape (SEQUENCE_LENGTH, 33, 4) to match feature engineer expectations.
        Previously returned (SEQUENCE_LENGTH, 33, 3) which caused dimension mismatch.
        
        Args:
            sequence: List of landmark arrays from pose detector
            
        Returns:
            Array of shape (SEQUENCE_LENGTH, 33, 4)
        """
        if len(sequence) == 0:
            # CRITICAL FIX: Create zeros with 4 channels (x, y, z, visibility)
            zeros_sequence = np.zeros((SEQUENCE_LENGTH, NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
            # Set visibility to 1.0 for all landmarks (fully visible)
            zeros_sequence[..., 3] = 1.0
            return zeros_sequence
        
        # Check the shape of the first element to ensure it has 4 channels
        sample_landmark = sequence[0]
        if sample_landmark.shape[-1] == 3:
            # FIX: Convert 3-channel to 4-channel landmarks
            print("   ⚠️  Converting 3-channel landmarks to 4-channel format...")
            sequence = [self._add_visibility_channel(landmarks) for landmarks in sequence]
        
        # Stack all valid frames
        sequence_array = np.stack(sequence, axis=0)
        
        # Verify we have 4 channels
        if sequence_array.shape[-1] != LANDMARK_DIM:
            raise ValueError(
                f"Expected {LANDMARK_DIM} channels (x,y,z,visibility), "
                f"got {sequence_array.shape[-1]}"
            )
        
        # Pad or truncate to SEQUENCE_LENGTH
        if sequence_array.shape[0] < SEQUENCE_LENGTH:
            pad_amount = SEQUENCE_LENGTH - sequence_array.shape[0]
            # Pad with zeros at the end (edge padding preserves last valid frame)
            padding = np.zeros((pad_amount, NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
            padding[..., 3] = 1.0  # Set visibility to 1.0 for padding
            sequence_array = np.concatenate([sequence_array, padding], axis=0)
        elif sequence_array.shape[0] > SEQUENCE_LENGTH:
            # Take middle portion for better temporal context
            start = (sequence_array.shape[0] - SEQUENCE_LENGTH) // 2
            sequence_array = sequence_array[start:start + SEQUENCE_LENGTH]
        
        return sequence_array.astype(np.float32)
    
    def _add_visibility_channel(self, landmarks_3d: np.ndarray) -> np.ndarray:
        """
        Add visibility channel to 3-channel landmarks.
        
        Args:
            landmarks_3d: Array of shape (33, 3) with x,y,z coordinates
            
        Returns:
            Array of shape (33, 4) with x,y,z,visibility (visibility=1.0)
        """
        if landmarks_3d.shape[-1] != 3:
            return landmarks_3d
        
        visibility = np.ones((landmarks_3d.shape[0], 1), dtype=np.float32)
        return np.concatenate([landmarks_3d, visibility], axis=-1)
    
    def _fix_landmark_shape(self, sequence_array: np.ndarray) -> np.ndarray:
        """
        Fix incorrect landmark shapes by converting to (33, 4) format.
        
        Args:
            sequence_array: Array of any shape
            
        Returns:
            Fixed array of shape (SEQUENCE_LENGTH, 33, 4)
        """
        if len(sequence_array.shape) == 3:
            T, J, D = sequence_array.shape
            
            if D == 3:
                # Add visibility channel
                print(f"   🔧 Adding visibility channel to shape {sequence_array.shape}")
                visibility = np.ones((T, J, 1), dtype=np.float32)
                return np.concatenate([sequence_array, visibility], axis=-1)
            elif D == 4:
                return sequence_array  # Already correct
        
        # Fallback: create zeros with correct shape
        print(f"   🔧 Recreating sequence with correct shape (33,4)")
        zeros_sequence = np.zeros((SEQUENCE_LENGTH, NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
        zeros_sequence[..., 3] = 1.0
        return zeros_sequence
    
    def _analyze_jogging_pattern(self, hip_positions: List[float], fps: float) -> Dict[str, float]:
        """
        Analyze jogging-in-place pattern from hip vertical movements.
        
        Args:
            hip_positions: List of hip y-coordinates over time
            fps: Camera frames per second
            
        Returns:
            Dictionary of jogging metrics
        """
        if len(hip_positions) < 10:  # Need enough frames for analysis
            return {}
        
        hip_array = np.array(hip_positions)
        
        # Calculate amplitude (vertical movement range)
        amplitude = np.max(hip_array) - np.min(hip_array)
        
        # Calculate frequency using peak detection
        from scipy.signal import find_peaks
        
        # Normalize and find peaks (steps)
        normalized = (hip_array - np.mean(hip_array)) / np.std(hip_array)
        peaks, _ = find_peaks(normalized, height=0.5, distance=int(fps/3))  # At least 0.3s between steps
        
        if len(peaks) >= 2:
            # Calculate frequency (steps per second)
            time_between_peaks = (peaks[-1] - peaks[0]) / fps
            frequency = (len(peaks) - 1) / time_between_peaks if time_between_peaks > 0 else 0
        else:
            frequency = 0
        
        # Calculate hip stability (lower = more stable)
        hip_stability = np.std(hip_array)
        
        return {
            'jogging_amplitude': float(amplitude),
            'jogging_frequency': float(frequency),
            'hip_stability': float(hip_stability),
            'step_count': len(peaks),
            'analysis_frames': len(hip_positions)
        }
    
    def _initialize_camera(self, camera_index: int):
        """Initialize camera with optimal settings."""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            return None
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Enable auto-focus if available
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        return cap
    
    def _draw_ui(self, 
                frame: np.ndarray, 
                pose_result: Optional[PoseResult],
                gesture_name: str,
                recording: bool,
                current_frames: int,
                sequence_count: int,
                hip_positions: Optional[List[float]] = None) -> np.ndarray:
        """Draw user interface on frame."""
        annotated = self.pose_extractor.draw_landmarks(
            frame, pose_result, draw_connections=True
        )
        
        # Add status text
        status = "🔴 RECORDING" if recording else "⏸️  READY"
        status_color = (0, 0, 255) if recording else (0, 255, 0)
        
        y_offset = 30
        line_height = 25
        
        texts = [
            f"Gesture: {gesture_name}",
            f"Status: {status}",
            f"Frames: {current_frames}/{SEQUENCE_LENGTH}",
            f"Sequences: {sequence_count}",
            f"User: {self.user_id}",
        ]
        
        # Add jogging feedback if applicable
        if gesture_name == 'run' and hip_positions and len(hip_positions) > 10:
            recent_hips = hip_positions[-10:]  # Last 10 frames
            vertical_range = max(recent_hips) - min(recent_hips)
            
            # Visual feedback for jogging quality
            if vertical_range > 0.05:  # Good vertical movement
                feedback = "✅ Good vertical motion"
                feedback_color = (0, 255, 0)
            elif vertical_range > 0.02:  # Moderate movement
                feedback = "⚠️  More knee lift needed"
                feedback_color = (0, 165, 255)
            else:  # Little movement
                feedback = "❌ Lift knees higher"
                feedback_color = (0, 0, 255)
            
            texts.append(feedback)
            
            # Draw vertical movement indicator
            bar_height = 100
            bar_width = 20
            bar_x = annotated.shape[1] - 40
            bar_y = 50
            
            # Normalize recent movement for visualization
            if len(recent_hips) > 0:
                normalized = (recent_hips[-1] - min(recent_hips)) / max(0.001, max(recent_hips) - min(recent_hips))
                fill_height = int(bar_height * normalized)
                
                # Draw background
                cv2.rectangle(
                    annotated,
                    (bar_x, bar_y),
                    (bar_x + bar_width, bar_y + bar_height),
                    (50, 50, 50), -1
                )
                
                # Draw fill
                cv2.rectangle(
                    annotated,
                    (bar_x, bar_y + bar_height - fill_height),
                    (bar_x + bar_width, bar_y + bar_height),
                    feedback_color, -1
                )
                
                # Label
                cv2.putText(
                    annotated, "V",
                    (bar_x - 5, bar_y + bar_height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
        
        for i, text in enumerate(texts):
            color = status_color if i == 1 else (255, 255, 255)
            if i >= 5:  # Feedback lines
                color = feedback_color if 'feedback_color' in locals() else (255, 255, 255)
            
            cv2.putText(
                annotated, text, (10, y_offset + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
        
        # Add pose detection indicator
        if pose_result is None:
            cv2.putText(
                annotated, "❌ NO POSE DETECTED", (10, y_offset + (len(texts) + 1) * line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
        else:
            # Show confidence
            if hasattr(pose_result.mp_results, 'pose_world_landmarks'):
                landmarks = pose_result.mp_results.pose_world_landmarks.landmark
                avg_visibility = np.mean([lm.visibility for lm in landmarks])
                cv2.putText(
                    annotated, f"Confidence: {avg_visibility:.2f}", 
                    (10, y_offset + (len(texts) + 1) * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
        
        # Add progress bar for recording
        if recording:
            bar_width = 400
            bar_height = 20
            bar_x = (annotated.shape[1] - bar_width) // 2
            bar_y = annotated.shape[0] - 50
            
            # Background
            cv2.rectangle(
                annotated, 
                (bar_x, bar_y), 
                (bar_x + bar_width, bar_y + bar_height), 
                (50, 50, 50), -1
            )
            
            # Progress
            progress = min(current_frames / SEQUENCE_LENGTH, 1.0)
            progress_width = int(bar_width * progress)
            progress_color = (0, int(255 * progress), int(255 * (1 - progress)))
            
            cv2.rectangle(
                annotated,
                (bar_x, bar_y),
                (bar_x + progress_width, bar_y + bar_height),
                progress_color, -1
            )
            
            # Progress text
            progress_text = f"{current_frames}/{SEQUENCE_LENGTH} ({progress:.0%})"
            cv2.putText(
                annotated, progress_text,
                (bar_x + 10, bar_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        
        return annotated
    
    def _save_sequence(self, 
                      gesture_name: str, 
                      camera_fps: float,
                      camera_resolution: str,
                      camera_index: int,
                      jogging_metrics: Optional[Dict[str, float]] = None) -> bool:
        """
        Save the current sequence with metadata.
        
        Returns:
            True if sequence was saved successfully
        """
        if len(self.current_sequence) < self.min_valid_frames:
            print(f"❌ Sequence too short: {len(self.current_sequence)}/{self.min_valid_frames} frames")
            return False
        
        # Pad sequence to required length if needed
        sequence_array = self._prepare_sequence_array()
        
        if sequence_array is None:
            return False
        
        # Verify shape before saving
        if sequence_array.shape[1:] != (NUM_LANDMARKS, LANDMARK_DIM):
            print(f"⚠️  Warning: Unexpected shape {sequence_array.shape}, fixing...")
            sequence_array = self._fix_landmark_shape(sequence_array)
        
        # Calculate timing statistics
        duration = 0.0
        frame_gaps = []
        if len(self.frame_timestamps) > 1:
            timestamps = self.frame_timestamps[:len(self.current_sequence)]
            duration = timestamps[-1] - timestamps[0]
            
            # Find gaps (frames without pose)
            for i in range(1, len(timestamps)):
                gap = timestamps[i] - timestamps[i-1]
                if gap > 1.5 / camera_fps:  # More than 1.5x expected frame interval
                    frame_gaps.append(i)
        
        # Calculate confidence statistics
        confidences = []
        for i in range(len(self.current_sequence)):
            # Extract visibility from landmarks (4th channel)
            if self.current_sequence[i].shape[-1] >= 4:
                visibility = self.current_sequence[i][:, 3]
                confidences.append(np.mean(visibility))
            else:
                confidences.append(0.8)  # Placeholder
        
        # Create metadata
        metadata = RecordingMetadata(
            user_id=self.user_id,
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            gesture=gesture_name,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            camera_fps=camera_fps,
            camera_resolution=camera_resolution,
            camera_index=camera_index,
            total_frames=len(self.frame_timestamps),
            valid_frames=len(self.current_sequence),
            missing_frames=len(self.frame_timestamps) - len(self.current_sequence),
            sequence_length=SEQUENCE_LENGTH,
            avg_confidence=np.mean(confidences) if confidences else 0.0,
            min_confidence=np.min(confidences) if confidences else 0.0,
            max_confidence=np.max(confidences) if confidences else 0.0,
            frame_gap_indices=frame_gaps,
            pose_model="mediapipe_pose_v1",
            normalization_method="torso_length"
        )
        
        # Add jogging metrics if available
        if jogging_metrics:
            metadata.jogging_amplitude = jogging_metrics.get('jogging_amplitude', 0.0)
            metadata.jogging_frequency = jogging_metrics.get('jogging_frequency', 0.0)
            metadata.hip_stability = jogging_metrics.get('hip_stability', 0.0)
            
            print(f"   Jogging analysis:")
            print(f"     Amplitude: {metadata.jogging_amplitude:.3f}")
            print(f"     Frequency: {metadata.jogging_frequency:.1f} steps/sec")
            print(f"     Hip stability: {metadata.hip_stability:.3f}")
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename_base = f"{gesture_name}_{self.user_id}_{timestamp}"
        
        # Save sequence
        sequence_path = self.output_dir / gesture_name / f"{filename_base}.npy"
        np.save(sequence_path, sequence_array)
        
        # Save metadata
        metadata_path = sequence_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)
        
        # Save raw timestamps (optional)
        timestamps_path = sequence_path.with_suffix('.timestamps.npy')
        np.save(timestamps_path, np.array(self.frame_timestamps))
        
        print(f"✅ Saved sequence: {filename_base}")
        print(f"   Frames: {len(self.current_sequence)}/{SEQUENCE_LENGTH}")
        print(f"   Shape: {sequence_array.shape}")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Gaps: {len(frame_gaps)}")
        
        return True
    
    def _prepare_sequence_array(self) -> Optional[np.ndarray]:
        """
        Prepare sequence array, padding if necessary.
        
        Returns:
            Array of shape (SEQUENCE_LENGTH, 33, 4)
        """
        if len(self.current_sequence) == 0:
            return None
        
        # Check if landmarks have visibility channel
        sample_landmark = self.current_sequence[0]
        if sample_landmark.shape[-1] == 3:
            # Convert 3-channel to 4-channel
            print("   ⚠️  Converting 3-channel landmarks to 4-channel format...")
            self.current_sequence = [self._add_visibility_channel(l) for l in self.current_sequence]
        
        # Stack all valid frames
        sequence_array = np.stack(self.current_sequence, axis=0)
        
        # Verify we have 4 channels
        if sequence_array.shape[-1] != LANDMARK_DIM:
            raise ValueError(
                f"Expected {LANDMARK_DIM} channels (x,y,z,visibility), "
                f"got {sequence_array.shape[-1]}"
            )
        
        # Pad or truncate to SEQUENCE_LENGTH
        if sequence_array.shape[0] < SEQUENCE_LENGTH:
            # Pad with zeros at the end
            pad_amount = SEQUENCE_LENGTH - sequence_array.shape[0]
            padding = np.zeros((pad_amount, NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
            padding[..., 3] = 1.0  # Set visibility to 1.0 for padding
            sequence_array = np.concatenate([sequence_array, padding], axis=0)
        elif sequence_array.shape[0] > SEQUENCE_LENGTH:
            # Truncate to SEQUENCE_LENGTH (keep middle portion for better temporal context)
            start = (sequence_array.shape[0] - SEQUENCE_LENGTH) // 2
            sequence_array = sequence_array[start:start + SEQUENCE_LENGTH]
        
        return sequence_array.astype(np.float32)
    
    def _reset_recording_state(self):
        """Reset recording state for a new sequence."""
        self.current_sequence = []
        self.frame_timestamps = []
        self.recording_metadata = None
    
    def _save_session_summary(self, gesture_name: str, sequence_count: int):
        """Save a summary of the recording session."""
        summary = {
            "user_id": self.user_id,
            "gesture": gesture_name,
            "timestamp": datetime.now().isoformat(),
            "sequences_recorded": sequence_count,
            "target_samples": MIN_SAMPLES_PER_GESTURE,
            "camera_used": CAMERA_INDEX,
            "sequence_length": SEQUENCE_LENGTH,
            "landmark_format": f"{NUM_LANDMARKS}×{LANDMARK_DIM}",
            "max_missing_frames": self.max_missing_frames,
            "min_valid_frames": self.min_valid_frames,
        }
        
        summary_path = self.user_dir / f"session_{gesture_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Update user statistics
        self._update_user_statistics(gesture_name, sequence_count)
    
    def _update_user_statistics(self, gesture_name: str, new_sequences: int):
        """Update statistics for the current user."""
        stats_path = self.user_dir / "statistics.json"
        
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
        else:
            stats = {
                "user_id": self.user_id,
                "first_session": datetime.now().isoformat(),
                "total_sessions": 0,
                "total_sequences": 0,
                "gestures": {},
            }
        
        # Update statistics
        stats["last_session"] = datetime.now().isoformat()
        stats["total_sessions"] = stats.get("total_sessions", 0) + 1
        stats["total_sequences"] = stats.get("total_sequences", 0) + new_sequences
        
        # Update gesture-specific stats
        if gesture_name not in stats["gestures"]:
            stats["gestures"][gesture_name] = {
                "first_recorded": datetime.now().isoformat(),
                "total_sequences": 0,
                "last_recorded": datetime.now().isoformat(),
            }
        
        stats["gestures"][gesture_name]["total_sequences"] += new_sequences
        stats["gestures"][gesture_name]["last_recorded"] = datetime.now().isoformat()
        
        # Save updated statistics
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)


# ============================================================================
# DATASET UTILITIES
# ============================================================================

class DatasetManager:
    """Utilities for managing and analyzing the collected dataset."""
    
    def __init__(self, dataset_dir: Path = DATA_DIR):
        self.dataset_dir = Path(dataset_dir)
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the collected dataset."""
        stats = {
            "total_sequences": 0,
            "gestures": {},
            "users": set(),
            "total_duration": 0.0,
            "landmark_format": f"{NUM_LANDMARKS}×{LANDMARK_DIM}",
        }
        
        for gesture in GESTURE_CLASSES:
            gesture_dir = self.dataset_dir / gesture
            if not gesture_dir.exists():
                stats["gestures"][gesture] = {
                    "sequences": 0,
                    "users": set(),
                    "duration": 0.0,
                    "valid_format": True,
                }
                continue
            
            npy_files = list(gesture_dir.glob("*.npy"))
            sequences = len(npy_files)
            
            # Parse metadata for each sequence
            gesture_duration = 0.0
            gesture_users = set()
            valid_format_count = 0
            
            for npy_file in npy_files:
                # Check file format
                try:
                    data = np.load(npy_file)
                    if data.shape[1:] == (NUM_LANDMARKS, LANDMARK_DIM):
                        valid_format_count += 1
                except:
                    pass
                
                json_file = npy_file.with_suffix('.json')
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        metadata = json.load(f)
                    
                    gesture_duration += metadata.get("duration_seconds", 0.0)
                    gesture_users.add(metadata.get("user_id", "unknown"))
                    
                    # Special stats for jogging
                    if gesture == 'run':
                        if 'jogging_amplitude' in metadata:
                            if 'jogging_stats' not in stats["gestures"][gesture]:
                                stats["gestures"][gesture]['jogging_stats'] = {
                                    'amplitudes': [],
                                    'frequencies': [],
                                    'stabilities': []
                                }
                            stats["gestures"][gesture]['jogging_stats']['amplitudes'].append(
                                metadata['jogging_amplitude']
                            )
                            stats["gestures"][gesture]['jogging_stats']['frequencies'].append(
                                metadata['jogging_frequency']
                            )
                            stats["gestures"][gesture]['jogging_stats']['stabilities'].append(
                                metadata['hip_stability']
                            )
            
            stats["gestures"][gesture] = {
                "sequences": sequences,
                "users": list(gesture_users),
                "user_count": len(gesture_users),
                "duration": gesture_duration,
                "valid_format_count": valid_format_count,
                "valid_format_pct": valid_format_count / max(sequences, 1) * 100,
            }
            
            stats["total_sequences"] += sequences
            stats["total_duration"] += gesture_duration
            stats["users"].update(gesture_users)
        
        stats["user_count"] = len(stats["users"])
        stats["users"] = list(stats["users"])
        
        return stats
    
    def print_dataset_summary(self):
        """Print a human-readable summary of the dataset."""
        stats = self.get_dataset_stats()
        
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"Total sequences: {stats['total_sequences']}")
        print(f"Total duration: {stats['total_duration']:.1f} seconds")
        print(f"Unique users: {stats['user_count']}")
        print(f"Users: {', '.join(stats['users'])}")
        print(f"Landmark format: {stats['landmark_format']} (x,y,z,visibility)")
        print("\nPer gesture breakdown:")
        print("-" * 60)
        
        for gesture, data in stats["gestures"].items():
            format_status = "✅" if data.get('valid_format_pct', 0) > 95 else "⚠️"
            print(f"{gesture:15s}: {data['sequences']:4d} sequences")
            print(f"                {data['user_count']:4d} users, {data['duration']:6.1f}s  {format_status} shape valid")
            
            # Special jogging stats
            if gesture == 'run' and 'jogging_stats' in data:
                jogging = data['jogging_stats']
                if jogging['amplitudes']:
                    print(f"                Jogging amplitude: {np.mean(jogging['amplitudes']):.3f} ± {np.std(jogging['amplitudes']):.3f}")
                    print(f"                Step frequency: {np.mean(jogging['frequencies']):.1f} steps/sec")
        
        # Check if we have enough samples
        print("\n" + "=" * 60)
        print("SAMPLING STATUS")
        print("=" * 60)
        
        for gesture, data in stats["gestures"].items():
            status = "✅" if data["sequences"] >= MIN_SAMPLES_PER_GESTURE else "❌"
            format_status = "✓" if data.get('valid_format_pct', 0) > 95 else "!"
            print(f"{gesture:15s}: {status} {data['sequences']:4d}/{MIN_SAMPLES_PER_GESTURE}  [{format_status}]")
        
        print("=" * 60)
    
    def validate_dataset(self) -> bool:
        """Validate dataset integrity and format."""
        print("\nValidating dataset...")
        
        all_valid = True
        
        for gesture in GESTURE_CLASSES:
            gesture_dir = self.dataset_dir / gesture
            if not gesture_dir.exists():
                print(f"❌ Missing directory for gesture: {gesture}")
                all_valid = False
                continue
            
            npy_files = list(gesture_dir.glob("*.npy"))
            
            for npy_file in npy_files:
                # Check that numpy file loads correctly
                try:
                    data = np.load(npy_file)
                    
                    # CRITICAL CHECK: Verify shape is (SEQUENCE_LENGTH, 33, 4)
                    if data.shape[0] != SEQUENCE_LENGTH:
                        print(f"⚠️  {npy_file.name}: Wrong sequence length {data.shape[0]} != {SEQUENCE_LENGTH}")
                        all_valid = False
                    
                    if data.shape[1:] != (NUM_LANDMARKS, LANDMARK_DIM):
                        print(f"❌ {npy_file.name}: Wrong landmark format {data.shape[1:]} != ({NUM_LANDMARKS}, {LANDMARK_DIM})")
                        all_valid = False
                    
                    # Check for NaN or inf values
                    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                        print(f"❌ {npy_file.name}: Contains NaN or inf values")
                        all_valid = False
                    
                except Exception as e:
                    print(f"❌ {npy_file.name}: Failed to load - {e}")
                    all_valid = False
                
                # Check metadata file exists
                json_file = npy_file.with_suffix('.json')
                if not json_file.exists():
                    print(f"⚠️  {npy_file.name}: Missing metadata file")
        
        if all_valid:
            print("✅ Dataset validation passed!")
            print(f"   All sequences are in correct format ({SEQUENCE_LENGTH}, {NUM_LANDMARKS}, {LANDMARK_DIM})")
        else:
            print("❌ Dataset validation failed!")
            print("   Run: python fix_dataset_format.py to repair corrupted files")
        
        return all_valid


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def test_data_collection():
    """Test the data collection system."""
    print("Testing Gesture Data Collection...")
    
    # Initialize collector
    collector = GestureDataCollector(user_id="test_user_01")
    
    # Test with a single gesture (run/jogging)
    collector.start_recording(
        gesture_name="run",
        collect_multiple=True,
        auto_restart=True
    )
    
    # Analyze dataset
    manager = DatasetManager()
    manager.print_dataset_summary()
    manager.validate_dataset()


if __name__ == "__main__":
    test_data_collection()