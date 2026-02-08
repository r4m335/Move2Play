# main.py
"""
Main entry point for the gesture recognition system.
Use this to run different components.
"""

import argparse
import sys
import numpy as np
from typing import Optional

# Import configurations
from config import (
    DATA_DIR, MODELS_DIR, EXPORTS_DIR,
    FINAL_MODEL, MODEL_CHECKPOINT, TFLITE_MODEL,
    GESTURE_CLASSES, NUM_GESTURES,
    MIN_SAMPLES_PER_GESTURE, SEQUENCE_LENGTH,
    validate_paths, check_model_exists, get_model_path,
    is_production
)

# Local imports (conditional to avoid circular imports)
def import_data_collector():
    from data_collector import GestureDataCollector
    return GestureDataCollector

def import_train_module():
    from train_gesture_model import main as train_main
    return train_main

def import_inference():
    from real_time_inference import RealTimeGestureRecognizer
    return RealTimeGestureRecognizer

def import_pipeline():
    from data_pipeline import GestureDataPipeline
    return GestureDataPipeline

def import_gesture_model():
    from gesture_model import GestureCNN
    return GestureCNN

# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_data(gesture_list: Optional[list] = None, 
                skip_prompts: bool = False):
    """
    Collect gesture data for training.
    
    Args:
        gesture_list: List of gestures to collect (None = all gestures)
        skip_prompts: If True, collect all gestures without interactive prompts
    """
    # Validate paths first
    validate_paths()
    
    # Import here to avoid circular imports
    GestureDataCollector = import_data_collector()
    collector = GestureDataCollector()
    
    # Use provided list or all gestures
    if gesture_list is None:
        gesture_list = GESTURE_CLASSES
    
    print("=" * 60)
    print("GESTURE DATA COLLECTION")
    print("=" * 60)
    print(f"Project Directory: {DATA_DIR}")
    print(f"Gestures to collect: {len(gesture_list)}")
    print(f"Target samples per gesture: {MIN_SAMPLES_PER_GESTURE}")
    print(f"Sequence length: {SEQUENCE_LENGTH} frames")
    print("\nInstructions:")
    print("1. Position yourself in frame (full body visible)")
    print("2. Press 'r' to start recording")
    print("3. Perform the gesture naturally")
    print("4. Recording stops automatically after 30 frames")
    print("5. Press 'q' to quit current gesture")
    print("=" * 60)
    
    for gesture in gesture_list:
        if skip_prompts:
            # Production mode: collect all without asking
            print(f"\n{'='*40}")
            print(f"Collecting data for: {gesture}")
            print(f"{'='*40}")
            collector.start_recording(gesture)
        else:
            # Interactive mode: ask for confirmation
            print(f"\nGesture: {gesture}")
            response = input(f"Collect data for '{gesture}'? (y/n/skip): ").lower()
            
            if response == 'y':
                collector.start_recording(gesture)
            elif response == 'skip':
                print(f"Skipping {gesture}")
                continue
            elif response == 'n' or response == 'q':
                print("Stopping data collection.")
                break
            else:
                print(f"Unknown response '{response}', skipping {gesture}")
    
    # Print collection summary
    print("\n" + "=" * 60)
    print("DATA COLLECTION SUMMARY")
    print("=" * 60)
    
    from pathlib import Path
    import glob
    
    for gesture in GESTURE_CLASSES:
        gesture_dir = DATA_DIR / gesture
        if gesture_dir.exists():
            npy_files = list(globe.glob(str(gesture_dir / "*.npy")))
            print(f"{gesture:15s}: {len(npy_files):4d} samples")
        else:
            print(f"{gesture:15s}: 0 samples (directory not created)")
    
    print("\nData collection complete!")

# ============================================================================
# TRAINING
# ============================================================================

def train_model(args: Optional[argparse.Namespace] = None):
    """Train the gesture recognition model."""
    validate_paths()
    
    # Check if we have enough data
    from pathlib import Path
    import glob
    
    has_enough_data = True
    for gesture in GESTURE_CLASSES:
        gesture_dir = DATA_DIR / gesture
        if not gesture_dir.exists():
            print(f"❌ Missing data directory for: {gesture}")
            has_enough_data = False
            continue
            
        samples = len(list(globe.glob(str(gesture_dir / "*.npy"))))
        if samples < MIN_SAMPLES_PER_GESTURE:
            print(f"⚠️  Low samples for {gesture}: {samples}/{MIN_SAMPLES_PER_GESTURE}")
    
    if not has_enough_data:
        print(f"\n❌ Insufficient data for training.")
        print(f"   Run data collection first: python main.py collect")
        sys.exit(1)
    
    # Import and run training
    train_main = import_train_module()
    
    # Prepare training arguments
    if args is None:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', default=str(DATA_DIR))
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--augment', action='store_true', default=True)
        parser.add_argument('--sequence_length', type=int, default=SEQUENCE_LENGTH)
        args = parser.parse_args([])
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Model save path: {FINAL_MODEL}")
    print(f"Checkpoint path: {MODEL_CHECKPOINT}")
    print(f"TFLite export: {TFLITE_MODEL}")
    print("=" * 60)
    
    # Run training
    train_main(args)

# ============================================================================
# REAL-TIME DEMO
# ============================================================================

def run_demo(model_type: str = "tflite"):
    """Run real-time gesture recognition demo."""
    validate_paths()
    
    # Check if model exists
    model_path = get_model_path(model_type)
    if not check_model_exists(model_type, raise_error=True):
        return
    
    print("\n" + "=" * 60)
    print("REAL-TIME GESTURE RECOGNITION DEMO")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Gestures: {NUM_GESTURES}")
    print(f"Confidence threshold: 0.7")
    print("\nControls:")
    print("• Press 'q' to quit")
    print("• Make sure you are visible in the camera")
    print("=" * 60)
    
    # Import and run
    RealTimeGestureRecognizer = import_inference()
    recognizer = RealTimeGestureRecognizer(
        tflite_model_path=str(model_path),
        sequence_length=SEQUENCE_LENGTH,
        confidence_threshold=0.7
    )
    
    recognizer.run_webcam_demo()

# ============================================================================
# PERFORMANCE TESTING
# ============================================================================

def test_performance(model_type: str = "final"):
    """Test model performance metrics."""
    validate_paths()
    
    # Check if model exists
    model_path = get_model_path(model_type)
    if not check_model_exists(model_type, raise_error=True):
        return
    
    # Check if test data exists or generate it
    from pathlib import Path
    test_data_file = DATA_DIR / "test_data.npz"
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE TESTING")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Test data: {test_data_file}")
    print("=" * 60)
    
    # Import required modules
    GestureDataPipeline = import_pipeline()
    GestureCNN = import_gesture_model()
    
    # Load or create test data
    if test_data_file.exists():
        print("Loading cached test data...")
        data = np.load(test_data_file)
        X_test = data['X_test']
        y_test = data['y_test']
    else:
        print("Generating test data split...")
        pipeline = GestureDataPipeline(data_dir=str(DATA_DIR))
        (_, _), (_, _), (X_test, y_test) = pipeline.prepare_datasets()
        # Cache for future use
        np.savez(test_data_file, X_test=X_test, y_test=y_test)
    
    # Load model
    model = GestureCNN(input_shape=(X_test.shape[1], X_test.shape[2]))
    
    try:
        if model_type == "tflite":
            print("⚠️  TFLite models require different evaluation method.")
            print("   Use 'final' model for comprehensive testing.")
            return
        else:
            model.load_model(str(model_path))
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print(f"   Try training first: python main.py train")
        return
    
    # Evaluate
    print("\nEvaluating model...")
    test_loss, test_accuracy, test_precision, test_recall = model.model.evaluate(
        X_test, y_test, verbose=1
    )
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   Accuracy:  {test_accuracy:.4f}")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall:    {test_recall:.4f}")
    
    # Calculate F1-Score
    if test_precision + test_recall > 0:
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        print(f"   F1-Score:  {f1_score:.4f}")
    
    # Per-class metrics
    print("\n📈 PER-CLASS METRICS:")
    y_pred = model.model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, 
                               target_names=GESTURE_CLASSES, digits=3))
    
    # Confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=GESTURE_CLASSES,
                yticklabels=GESTURE_CLASSES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'confusion_matrix.png')
    print(f"Confusion matrix saved to: {MODELS_DIR / 'confusion_matrix.png'}")
    
    # Calculate per-class accuracy
    print("\n📋 PER-CLASS ACCURACY:")
    class_correct = np.zeros(NUM_GESTURES)
    class_total = np.zeros(NUM_GESTURES)
    
    for i in range(NUM_GESTURES):
        mask = y_true_classes == i
        if np.any(mask):
            class_correct[i] = np.sum(y_pred_classes[mask] == i)
            class_total[i] = np.sum(mask)
            accuracy = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            print(f"   {GESTURE_CLASSES[i]:15s}: {accuracy:.3f} ({int(class_correct[i])}/{int(class_total[i])})")
    
    print("\n✅ Performance testing complete!")

# ============================================================================
# PIPELINE RUNNER
# ============================================================================

def run_pipeline(skip_data_collection: bool = False):
    """
    Run the complete pipeline from data collection to demo.
    
    Args:
        skip_data_collection: If True, skip data collection step
    """
    print("=" * 60)
    print("COMPLETE PIPELINE EXECUTION")
    print("=" * 60)
    
    steps = [
        ("Data Collection", not skip_data_collection),
        ("Model Training", True),
        ("Performance Testing", True),
        ("Real-time Demo", True),
    ]
    
    for step_name, should_run in steps:
        if not should_run:
            print(f"\n⏭️  Skipping: {step_name}")
            continue
            
        print(f"\n{'='*40}")
        print(f"STEP: {step_name}")
        print(f"{'='*40}")
        
        try:
            if step_name == "Data Collection":
                collect_data(skip_prompts=True)
            elif step_name == "Model Training":
                train_model()
            elif step_name == "Performance Testing":
                test_performance()
            elif step_name == "Real-time Demo":
                run_demo()
                
            print(f"✅ {step_name} completed successfully!")
            
        except KeyboardInterrupt:
            print(f"\n⚠️  {step_name} interrupted by user")
            response = input("Continue with next step? (y/n): ").lower()
            if response != 'y':
                print("Pipeline execution stopped.")
                break
        except Exception as e:
            print(f"❌ Error in {step_name}: {e}")
            response = input("Continue despite error? (y/n): ").lower()
            if response != 'y':
                print("Pipeline execution stopped.")
                break
    
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 60)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point with proper argument handling."""
    parser = argparse.ArgumentParser(
        description='Gesture Recognition System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py collect           # Collect gesture data
  python main.py train             # Train the model
  python main.py demo              # Run real-time demo
  python main.py test              # Test model performance
  python main.py all               # Run complete pipeline
  python main.py collect --gestures run punch_left punch_right  # Collect specific gestures
  python main.py demo --model tflite  # Run demo with TFLite model
        """
    )
    
    parser.add_argument('mode', 
                       choices=['collect', 'train', 'demo', 'test', 'all', 'validate'],
                       help='Mode to run')
    
    # Optional arguments for specific modes
    parser.add_argument('--gestures', nargs='+', choices=GESTURE_CLASSES,
                       help='Specific gestures to collect (collect mode only)')
    parser.add_argument('--model', choices=['best', 'final', 'tflite', 'quantized'],
                       default='tflite' if is_production() else 'final',
                       help='Model type to use (default: tflite in production, final otherwise)')
    parser.add_argument('--skip-prompts', action='store_true',
                       help='Skip interactive prompts in collect mode')
    parser.add_argument('--skip-data', action='store_true',
                       help='Skip data collection in pipeline mode')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print system info
    print("\n" + "=" * 60)
    print("GESTURE RECOGNITION SYSTEM")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Environment: {'Production' if is_production() else 'Development'}")
    print(f"Gestures: {NUM_GESTURES}")
    print(f"Project root: {DATA_DIR.parent}")
    print("=" * 60)
    
    # Execute requested mode
    try:
        if args.mode == 'collect':
            collect_data(args.gestures, args.skip_prompts)
            
        elif args.mode == 'train':
            train_model(args)
            
        elif args.mode == 'demo':
            run_demo(args.model)
            
        elif args.mode == 'test':
            test_performance(args.model)
            
        elif args.mode == 'all':
            run_pipeline(args.skip_data)
            
        elif args.mode == 'validate':
            from gesture_action_contract import GestureActionContract
            GestureActionContract.validate_consistency()
            validate_paths()
            print("\n✅ System validation passed!")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation interrupted by user.")
        sys.exit(0)
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if not is_production():
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Validate imports and configuration
    try:
        validate_paths()
        from gesture_action_contract import GestureActionContract
        GestureActionContract.validate_consistency()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
    
    # Run main if arguments provided, otherwise show help
    if len(sys.argv) > 1:
        main()
    else:
        print("Usage: python main.py [collect|train|demo|test|all|validate]")
        print("\nRun 'python main.py --help' for detailed usage information.")
        sys.exit(1)