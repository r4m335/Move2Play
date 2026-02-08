# real_time_inference.py
"""
Robust real-time gesture inference with adaptive smoothing and action cooldowns.
"""

import cv2
import numpy as np
import tensorflow as tf
import time
import json
from collections import deque, defaultdict
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

# Import from config
from config import (
    GESTURE_CLASSES, SEQUENCE_LENGTH,
    INFERENCE_CONFIDENCE_THRESHOLD, PREDICTION_HISTORY_LENGTH,
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, TARGET_FPS,
    LOGS_DIR
)
from pose_extractor import PoseExtractor, PoseResult
from feature_engineer import FeatureEngineer
from gesture_action_contract import GestureActionContract

# ============================================================================
# ACTION COOLDOWN MANAGER
# ============================================================================

class ActionCooldownManager:
    """Manages cooldowns for actions to prevent spam."""
    
    # Default cooldowns in milliseconds
    DEFAULT_COOLDOWNS = {
        'attack': 500,    # 500ms between attacks
        'dodge': 400,     # 400ms between dodges
        'slide': 1000,    # 1 second between slides
        'block': 300,     # 300ms between blocks
        'forward_movement': 100,  # 100ms for movement updates
    }
    
    def __init__(self, cooldowns: Optional[Dict[str, float]] = None):
        """
        Args:
            cooldowns: Custom cooldowns in milliseconds. Uses DEFAULT_COOLDOWNS if None.
        """
        self.cooldowns = cooldowns or self.DEFAULT_COOLDOWNS.copy()
        self.last_action_times: Dict[str, float] = {}
        
    def can_perform_action(self, action_name: str) -> Tuple[bool, float]:
        """
        Check if an action can be performed based on cooldown.
        
        Args:
            action_name: Name of the action to check
            
        Returns:
            Tuple of (can_perform, remaining_cooldown_ms)
        """
        if action_name not in self.cooldowns:
            return True, 0.0
        
        current_time = time.time() * 1000  # Convert to milliseconds
        last_time = self.last_action_times.get(action_name, 0)
        cooldown = self.cooldowns[action_name]
        
        elapsed = current_time - last_time
        remaining = max(0, cooldown - elapsed)
        
        return remaining == 0, remaining
    
    def record_action(self, action_name: str):
        """Record that an action was performed."""
        if action_name in self.cooldowns:
            self.last_action_times[action_name] = time.time() * 1000
    
    def get_cooldown_status(self) -> Dict[str, float]:
        """Get current cooldown status for all actions."""
        status = {}
        current_time = time.time() * 1000
        
        for action_name, cooldown in self.cooldowns.items():
            last_time = self.last_action_times.get(action_name, 0)
            elapsed = current_time - last_time
            remaining = max(0, cooldown - elapsed)
            
            if remaining > 0:
                status[action_name] = remaining / 1000.0  # Convert to seconds
        
        return status
    
    def reset(self):
        """Reset all cooldowns."""
        self.last_action_times.clear()

# ============================================================================
# ADAPTIVE SMOOTHER
# ============================================================================

class AdaptiveSmoother:
    """
    Adaptive prediction smoothing based on timing and confidence.
    Uses time-based windows instead of fixed frame counts.
    """
    
    def __init__(self, 
                 window_duration_ms: float = 500.0,  # 500ms window
                 min_confidence: float = 0.5,
                 max_predictions: int = 20):
        """
        Args:
            window_duration_ms: Time window for smoothing in milliseconds
            min_confidence: Minimum confidence to consider a prediction
            max_predictions: Maximum number of predictions to store
        """
        self.window_duration_ms = window_duration_ms
        self.min_confidence = min_confidence
        self.max_predictions = max_predictions
        
        # Store predictions with timestamps
        self.prediction_history: List[Tuple[float, int, float]] = []  # (timestamp_ms, class_idx, confidence)
        
        # Statistics for adaptive threshold
        self.confidence_stats = {
            'mean': 0.7,
            'std': 0.2,
            'count': 0
        }
        
    def add_prediction(self, class_idx: int, confidence: float):
        """Add a new prediction to the history."""
        current_time = time.time() * 1000
        
        # Update confidence statistics
        self._update_confidence_stats(confidence)
        
        # Add to history
        self.prediction_history.append((current_time, class_idx, confidence))
        
        # Keep only recent predictions
        self._prune_old_predictions()
        
        # Keep within max size
        if len(self.prediction_history) > self.max_predictions:
            self.prediction_history = self.prediction_history[-self.max_predictions:]
    
    def get_smoothed_prediction(self) -> Optional[Tuple[int, float, List[float]]]:
        """
        Get smoothed prediction using time-weighted majority voting.
        
        Returns:
            Tuple of (predicted_class_idx, confidence, all_probabilities) or None
        """
        if not self.prediction_history:
            return None
        
        current_time = time.time() * 1000
        
        # Calculate time-weighted votes
        class_votes = defaultdict(float)
        total_weight = 0.0
        
        for timestamp_ms, class_idx, confidence in self.prediction_history:
            # Calculate time weight (recent predictions have higher weight)
            age_ms = current_time - timestamp_ms
            
            if age_ms > self.window_duration_ms:
                continue  # Too old
            
            # Time weight: linear decay from 1.0 to 0.0 over window duration
            time_weight = 1.0 - (age_ms / self.window_duration_ms)
            
            # Confidence weight
            confidence_weight = confidence
            
            # Combined weight
            weight = time_weight * confidence_weight
            
            class_votes[class_idx] += weight
            total_weight += weight
        
        if total_weight == 0:
            return None
        
        # Find class with highest weighted votes
        best_class = max(class_votes.items(), key=lambda x: x[1])[0]
        
        # Calculate normalized confidence
        best_votes = class_votes[best_class]
        confidence = best_votes / total_weight if total_weight > 0 else 0.0
        
        # Calculate all probabilities (normalized votes)
        all_probs = []
        for i in range(len(GESTURE_CLASSES)):
            prob = class_votes.get(i, 0.0) / total_weight if total_weight > 0 else 0.0
            all_probs.append(prob)
        
        return best_class, confidence, all_probs
    
    def get_adaptive_threshold(self) -> float:
        """Calculate adaptive confidence threshold based on recent statistics."""
        if self.confidence_stats['count'] < 10:
            return self.min_confidence
        
        # Dynamic threshold: mean - 0.5*std, but not below min_confidence
        adaptive_threshold = self.confidence_stats['mean'] - 0.5 * self.confidence_stats['std']
        return max(self.min_confidence, min(adaptive_threshold, 0.9))
    
    def _update_confidence_stats(self, confidence: float):
        """Update running statistics for confidence values."""
        old_mean = self.confidence_stats['mean']
        old_count = self.confidence_stats['count']
        
        # Update mean using Welford's algorithm
        new_count = old_count + 1
        delta = confidence - old_mean
        new_mean = old_mean + delta / new_count
        
        # Update variance
        if old_count > 0:
            old_variance = self.confidence_stats['std'] ** 2
            new_variance = old_variance + delta * (confidence - new_mean)
            new_std = np.sqrt(new_variance / new_count) if new_count > 1 else 0.0
        else:
            new_std = 0.0
        
        self.confidence_stats.update({
            'mean': new_mean,
            'std': new_std,
            'count': new_count
        })
    
    def _prune_old_predictions(self):
        """Remove predictions older than the window duration."""
        current_time = time.time() * 1000
        cutoff_time = current_time - self.window_duration_ms
        
        # Keep only recent predictions
        self.prediction_history = [
            pred for pred in self.prediction_history
            if pred[0] >= cutoff_time
        ]
    
    def reset(self):
        """Reset the smoother state."""
        self.prediction_history.clear()
        self.confidence_stats = {'mean': 0.7, 'std': 0.2, 'count': 0}

# ============================================================================
# INFERENCE LOGGER
# ============================================================================

class InferenceLogger:
    """Logs inference results for debugging and analysis."""
    
    def __init__(self, log_dir: Path = LOGS_DIR):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Create log file with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"inference_{timestamp}.jsonl"
        
        # Buffer for batch writing
        self.log_buffer: List[Dict] = []
        self.buffer_size = 10
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'frames_with_pose': 0,
            'frames_with_prediction': 0,
            'actions_triggered': 0,
            'cooldown_blocked': 0,
        }
    
    def log_frame(self, 
                  frame_data: Dict[str, Any],
                  raw_probabilities: Optional[List[float]] = None):
        """Log frame processing results."""
        log_entry = {
            'timestamp': time.time(),
            'frame_id': self.stats['total_frames'],
            'has_pose': frame_data.get('landmarks') is not None,
            'buffer_fill': len(frame_data.get('sequence_buffer', [])),
            'gesture': frame_data.get('gesture'),
            'confidence': frame_data.get('confidence'),
            'action': frame_data.get('action'),
            'cooldown_blocked': frame_data.get('cooldown_blocked', False),
            'raw_probabilities': raw_probabilities,
        }
        
        self.log_buffer.append(log_entry)
        self.stats['total_frames'] += 1
        
        if log_entry['has_pose']:
            self.stats['frames_with_pose'] += 1
        
        if log_entry['gesture']:
            self.stats['frames_with_prediction'] += 1
        
        if log_entry['action']:
            self.stats['actions_triggered'] += 1
        
        if log_entry['cooldown_blocked']:
            self.stats['cooldown_blocked'] += 1
        
        # Write buffer if full
        if len(self.log_buffer) >= self.buffer_size:
            self._write_buffer()
    
    def _write_buffer(self):
        """Write buffer to log file."""
        if not self.log_buffer:
            return
        
        with open(self.log_file, 'a') as f:
            for entry in self.log_buffer:
                f.write(json.dumps(entry, default=str) + '\n')
        
        self.log_buffer.clear()
    
    def log_special_event(self, event_type: str, data: Dict[str, Any]):
        """Log special events like model loading errors, etc."""
        event_entry = {
            'timestamp': time.time(),
            'type': event_type,
            'data': data,
        }
        
        self.log_buffer.append(event_entry)
        self._write_buffer()  # Always write events immediately
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current inference statistics."""
        stats = self.stats.copy()
        
        if stats['total_frames'] > 0:
            stats['pose_detection_rate'] = stats['frames_with_pose'] / stats['total_frames']
            stats['prediction_rate'] = stats['frames_with_prediction'] / stats['total_frames']
            stats['action_rate'] = stats['actions_triggered'] / stats['total_frames']
        else:
            stats['pose_detection_rate'] = 0.0
            stats['prediction_rate'] = 0.0
            stats['action_rate'] = 0.0
        
        return stats
    
    def save_statistics(self):
        """Save statistics to file."""
        stats = self.get_statistics()
        stats_file = self.log_dir / "inference_stats.json"
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
    
    def close(self):
        """Close logger and write remaining buffer."""
        self._write_buffer()
        self.save_statistics()

# ============================================================================
# MAIN INFERENCE CLASS
# ============================================================================

class RealTimeGestureRecognizer:
    """Handles robust real-time gesture recognition with adaptive smoothing."""
    
    def __init__(self, 
                 tflite_model_path: str,
                 sequence_length: int = SEQUENCE_LENGTH,
                 base_confidence_threshold: float = INFERENCE_CONFIDENCE_THRESHOLD,
                 smoothing_window_ms: float = 500.0,
                 target_fps: int = TARGET_FPS):
        """
        Args:
            tflite_model_path: Path to TFLite model
            sequence_length: Number of frames in input sequence
            base_confidence_threshold: Base confidence threshold
            smoothing_window_ms: Time window for smoothing in milliseconds
            target_fps: Target frames per second for processing
        """
        # Load TFLite model
        try:
            self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"✅ Loaded TFLite model: {tflite_model_path}")
            print(f"   Input shape: {self.input_details[0]['shape']}")
        except Exception as e:
            raise RuntimeError(f"Failed to load TFLite model: {e}")
        
        # Components
        self.pose_extractor = PoseExtractor(use_torso_length=True)
        self.feature_engineer = FeatureEngineer()
        self.action_contract = GestureActionContract()
        
        # Buffers
        self.sequence_length = sequence_length
        self.sequence_buffer = deque(maxlen=sequence_length)
        self.feature_buffer = deque(maxlen=sequence_length)
        
        # Adaptive smoothing
        self.smoother = AdaptiveSmoother(
            window_duration_ms=smoothing_window_ms,
            min_confidence=base_confidence_threshold * 0.8,
            max_predictions=20
        )
        
        # Action cooldown
        self.cooldown_manager = ActionCooldownManager()
        
        # Logging
        self.logger = InferenceLogger()
        
        # Performance tracking
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.last_process_time = 0
        
        # State
        self.current_gesture = None
        self.current_confidence = 0.0
        self.current_action = None
        self.last_action_time = 0
        
        print(f"✅ Inference system initialized")
        print(f"   Gestures: {len(GESTURE_CLASSES)}")
        print(f"   Sequence length: {sequence_length}")
        print(f"   Target FPS: {target_fps}")
        print(f"   Smoothing window: {smoothing_window_ms}ms")
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame and return gesture prediction.
        
        Returns:
            Dictionary with frame processing results
        """
        # Throttle processing to target FPS
        current_time = time.time()
        if current_time - self.last_process_time < self.frame_interval:
            # Skip frame to maintain target FPS
            return {
                'gesture': None,
                'confidence': 0.0,
                'action': None,
                'cooldown_blocked': False,
                'landmarks': None,
                'annotated_frame': frame,
                'skip_reason': 'throttled'
            }
        
        self.last_process_time = current_time
        
        # Extract pose
        pose_result = self.pose_extractor.process_frame(frame)
        
        if pose_result is None:
            # No person detected
            annotated_frame = self.pose_extractor.draw_landmarks(frame, None)
            result = {
                'gesture': None,
                'confidence': 0.0,
                'action': None,
                'cooldown_blocked': False,
                'landmarks': None,
                'annotated_frame': annotated_frame,
                'skip_reason': 'no_pose'
            }
            self.logger.log_frame(result)
            return result
        
        # Add landmarks to buffer
        self.sequence_buffer.append(pose_result.landmarks)
        
        # Extract features
        try:
            features = self.feature_engineer.extract_features(
                np.array([pose_result.landmarks])
            )[0]  # Remove sequence dimension
            self.feature_buffer.append(features)
        except Exception as e:
            print(f"⚠️ Feature extraction error: {e}")
            self.logger.log_special_event('feature_extraction_error', {'error': str(e)})
            annotated_frame = self.pose_extractor.draw_landmarks(frame, pose_result)
            return {
                'gesture': None,
                'confidence': 0.0,
                'action': None,
                'cooldown_blocked': False,
                'landmarks': pose_result.landmarks,
                'annotated_frame': annotated_frame,
                'skip_reason': 'feature_error'
            }
        
        # Check if we have enough frames
        if len(self.sequence_buffer) < self.sequence_buffer.maxlen:
            annotated_frame = self.pose_extractor.draw_landmarks(frame, pose_result)
            result = {
                'gesture': None,
                'confidence': 0.0,
                'action': None,
                'cooldown_blocked': False,
                'landmarks': pose_result.landmarks,
                'annotated_frame': annotated_frame,
                'buffer_fill': len(self.sequence_buffer),
                'skip_reason': 'buffer_filling'
            }
            self.logger.log_frame(result)
            return result
        
        # Prepare input for model
        try:
            feature_sequence = np.array(self.feature_buffer, dtype=np.float32)
            feature_sequence = np.expand_dims(feature_sequence, axis=0)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], feature_sequence)
            self.interpreter.invoke()
            
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
            raw_probabilities = predictions[0].tolist()
            
            # Get best prediction
            pred_idx = np.argmax(predictions[0])
            confidence = predictions[0][pred_idx]
            
        except Exception as e:
            print(f"⚠️ Inference error: {e}")
            self.logger.log_special_event('inference_error', {'error': str(e)})
            annotated_frame = self.pose_extractor.draw_landmarks(frame, pose_result)
            return {
                'gesture': None,
                'confidence': 0.0,
                'action': None,
                'cooldown_blocked': False,
                'landmarks': pose_result.landmarks,
                'annotated_frame': annotated_frame,
                'skip_reason': 'inference_error'
            }
        
        # Add to smoother
        self.smoother.add_prediction(pred_idx, confidence)
        
        # Get smoothed prediction
        smoothed = self.smoother.get_smoothed_prediction()
        
        # Prepare base result
        annotated_frame = self.pose_extractor.draw_landmarks(frame, pose_result)
        result = {
            'gesture': None,
            'confidence': confidence,
            'action': None,
            'cooldown_blocked': False,
            'landmarks': pose_result.landmarks,
            'annotated_frame': annotated_frame,
            'raw_probabilities': raw_probabilities,
            'adaptive_threshold': self.smoother.get_adaptive_threshold(),
        }
        
        if smoothed is None:
            self.logger.log_frame(result, raw_probabilities)
            return result
        
        smoothed_idx, smoothed_confidence, all_probs = smoothed
        
        # Check adaptive confidence threshold
        adaptive_threshold = self.smoother.get_adaptive_threshold()
        if smoothed_confidence < adaptive_threshold:
            result['gesture'] = None
            result['confidence'] = smoothed_confidence
            self.logger.log_frame(result, raw_probabilities)
            return result
        
        # Get gesture name
        gesture_name = GESTURE_CLASSES[smoothed_idx]
        
        # Check action cooldown
        action_type, action_params = self.action_contract.get_action_for_gesture(gesture_name)
        can_perform, remaining = self.cooldown_manager.can_perform_action(action_type)
        
        if not can_perform:
            # Action is on cooldown
            result.update({
                'gesture': gesture_name,
                'confidence': smoothed_confidence,
                'action': None,
                'cooldown_blocked': True,
                'cooldown_remaining': remaining,
            })
            self.logger.log_frame(result, raw_probabilities)
            return result
        
        # Record action and update state
        self.cooldown_manager.record_action(action_type)
        self.current_gesture = gesture_name
        self.current_confidence = smoothed_confidence
        self.current_action = action_type
        
        # Prepare action data
        action_data = {
            'name': action_type,
            'params': action_params,
            'gesture': gesture_name,
            'confidence': smoothed_confidence,
            'timestamp': current_time,
            'frame_id': self.logger.stats['total_frames'],
        }
        
        result.update({
            'gesture': gesture_name,
            'confidence': smoothed_confidence,
            'action': action_data,
            'all_probabilities': all_probs,
        })
        
        self.logger.log_frame(result, raw_probabilities)
        return result
    
    def run_webcam_demo(self, 
                       camera_index: int = CAMERA_INDEX,
                       show_debug: bool = True,
                       record_logs: bool = True):
        """
        Run real-time gesture recognition demo.
        
        Args:
            camera_index: Camera device index
            show_debug: Show debug information on screen
            record_logs: Record inference logs to file
        """
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ Failed to open camera {camera_index}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        print("\n" + "=" * 60)
        print("REAL-TIME GESTURE RECOGNITION DEMO")
        print("=" * 60)
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Reset buffers and cooldowns")
        print("  'l' - Toggle logging")
        print("  'd' - Toggle debug display")
        print("=" * 60)
        
        frame_count = 0
        start_time = time.time()
        logging_enabled = record_logs
        debug_enabled = show_debug
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Process frame
            result = self.process_frame(frame)
            
            # Prepare display frame
            display_frame = result['annotated_frame'].copy()
            
            # Add gesture info if detected
            if result['gesture']:
                # Gesture info
                text_color = (0, 255, 0)  # Green
                if result['cooldown_blocked']:
                    text_color = (0, 165, 255)  # Orange for cooldown
                
                cv2.putText(
                    display_frame,
                    f"{result['gesture']} ({result['confidence']:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2
                )
                
                # Action info
                if result['action']:
                    action_text = f"Action: {result['action']['name']}"
                    cv2.putText(
                        display_frame, action_text,
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )
                elif result['cooldown_blocked']:
                    cv2.putText(
                        display_frame, "⏳ Cooldown",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2
                    )
            
            # Debug information
            if debug_enabled:
                y_offset = 90
                line_height = 20
                
                debug_info = [
                    f"Frame: {frame_count}",
                    f"Buffer: {len(self.sequence_buffer)}/{self.sequence_length}",
                    f"FPS: {self._calculate_fps(frame_count, start_time):.1f}",
                    f"Threshold: {result.get('adaptive_threshold', 0.7):.2f}",
                    f"Logging: {'ON' if logging_enabled else 'OFF'}",
                ]
                
                # Add cooldown status
                cooldown_status = self.cooldown_manager.get_cooldown_status()
                if cooldown_status:
                    debug_info.append("Cooldowns:")
                    for action, remaining in cooldown_status.items():
                        debug_info.append(f"  {action}: {remaining:.1f}s")
                
                for i, text in enumerate(debug_info):
                    cv2.putText(
                        display_frame, text,
                        (10, y_offset + i * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
                    )
            
            # Display
            cv2.imshow('Gesture Recognition Demo', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset()
                print("🔄 Buffers and cooldowns reset")
            elif key == ord('l'):
                logging_enabled = not logging_enabled
                print(f"📝 Logging {'enabled' if logging_enabled else 'disabled'}")
            elif key == ord('d'):
                debug_enabled = not debug_enabled
                print(f"🐛 Debug display {'enabled' if debug_enabled else 'disabled'}")
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Save logs
        if logging_enabled:
            stats = self.logger.get_statistics()
            print("\n" + "=" * 60)
            print("INFERENCE STATISTICS")
            print("=" * 60)
            for key, value in stats.items():
                print(f"{key:25s}: {value}")
            
            self.logger.close()
            print(f"\n📊 Logs saved to: {self.logger.log_file}")
    
    def _calculate_fps(self, frame_count: int, start_time: float) -> float:
        """Calculate current FPS."""
        elapsed = time.time() - start_time
        return frame_count / elapsed if elapsed > 0 else 0.0
    
    def reset(self):
        """Reset all buffers and state."""
        self.sequence_buffer.clear()
        self.feature_buffer.clear()
        self.smoother.reset()
        self.cooldown_manager.reset()
        self.current_gesture = None
        self.current_confidence = 0.0
        self.current_action = None
        
        print("🔄 Inference system reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'buffer_fill': len(self.sequence_buffer),
            'buffer_capacity': self.sequence_length,
            'current_gesture': self.current_gesture,
            'current_confidence': self.current_confidence,
            'current_action': self.current_action,
            'cooldown_status': self.cooldown_manager.get_cooldown_status(),
            'smoother_stats': self.smoother.confidence_stats,
            'logging_stats': self.logger.get_statistics(),
        }
    
    def close(self):
        """Clean up resources."""
        self.logger.close()
        print("✅ Inference system closed")

# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_inference():
    """Test the inference system."""
    from config import TFLITE_MODEL
    
    print("Testing Real-Time Gesture Recognition...")
    
    if not TFLITE_MODEL.exists():
        print(f"❌ TFLite model not found: {TFLITE_MODEL}")
        print("   Train the model first: python main.py train")
        return
    
    # Initialize recognizer
    recognizer = RealTimeGestureRecognizer(
        tflite_model_path=str(TFLITE_MODEL),
        sequence_length=SEQUENCE_LENGTH,
        base_confidence_threshold=0.7,
        smoothing_window_ms=500.0,
        target_fps=20
    )
    
    # Run demo
    recognizer.run_webcam_demo(
        camera_index=CAMERA_INDEX,
        show_debug=True,
        record_logs=True
    )
    
    # Cleanup
    recognizer.close()

if __name__ == "__main__":
    test_inference()