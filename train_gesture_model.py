# train_gesture_model.py
"""
Main training script for gesture recognition with idle class handling.
Prevents false positives by ensuring idle has sufficient samples.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple, Dict, Any

# Import from config
from config import (
    GESTURE_CLASSES, NUM_GESTURES, SEQUENCE_LENGTH,
    FINAL_MODEL, TFLITE_MODEL, MIN_SAMPLES_PER_GESTURE,
    get_required_idle_samples, IDLE_SAMPLES_MULTIPLIER
)
from data_pipeline import GestureDataPipeline
from gesture_model import GestureModel

def analyze_dataset_distribution(pipeline: GestureDataPipeline, 
                                X_train: np.ndarray, 
                                y_train: np.ndarray,
                                X_val: np.ndarray, 
                                y_val: np.ndarray,
                                X_test: np.ndarray,
                                y_test: np.ndarray) -> Dict[str, Any]:
    """
    Analyze and print dataset distribution with emphasis on idle class.
    
    Returns:
        Dictionary with distribution analysis
    """
    # Convert one-hot to integer
    y_train_int = np.argmax(y_train, axis=1)
    y_val_int = np.argmax(y_val, axis=1)
    y_test_int = np.argmax(y_test, axis=1)
    
    # Find idle index
    idle_idx = GESTURE_CLASSES.index("idle")
    
    # Count samples per class
    train_counts = np.bincount(y_train_int, minlength=NUM_GESTURES)
    val_counts = np.bincount(y_val_int, minlength=NUM_GESTURES)
    test_counts = np.bincount(y_test_int, minlength=NUM_GESTURES)
    
    # Find max among non-idle classes
    non_idle_indices = [i for i in range(NUM_GESTURES) if i != idle_idx]
    max_non_idle_train = max(train_counts[non_idle_indices]) if non_idle_indices else 0
    required_idle = int(max_non_idle_train * IDLE_SAMPLES_MULTIPLIER)
    
    print("\n" + "=" * 60)
    print("DATASET DISTRIBUTION ANALYSIS")
    print("=" * 60)
    print(f"{'Gesture':<15} {'Train':<8} {'Val':<8} {'Test':<8} {'Status':<10}")
    print("-" * 60)
    
    for i, gesture in enumerate(GESTURE_CLASSES):
        train_count = train_counts[i]
        val_count = val_counts[i]
        test_count = test_counts[i]
        
        # Check if meets minimum requirements
        if gesture == "idle":
            status = "✅ GOOD" if train_count >= required_idle else "⚠️  NEEDS MORE"
            marker = "⭐ "
        else:
            status = "✅ GOOD" if train_count >= MIN_SAMPLES_PER_GESTURE else "⚠️  NEEDS MORE"
            marker = ""
        
        print(f"{marker}{gesture:<13} {train_count:<8} {val_count:<8} {test_count:<8} {status:<10}")
    
    print("-" * 60)
    
    # Idle specific analysis
    idle_train_count = train_counts[idle_idx]
    print(f"\n📊 IDLE CLASS ANALYSIS:")
    print(f"   Idle samples: {idle_train_count}")
    print(f"   Max active gesture: {max_non_idle_train}")
    print(f"   Target ratio: {IDLE_SAMPLES_MULTIPLIER:.1f}x")
    print(f"   Current ratio: {idle_train_count/max_non_idle_train:.2f}x" if max_non_idle_train > 0 else "   No active gestures")
    
    if idle_train_count < required_idle:
        print(f"⚠️  WARNING: Idle needs {required_idle - idle_train_count} more samples!")
        print(f"   Run: python main.py collect_idle")
    
    return {
        'train_counts': train_counts,
        'val_counts': val_counts,
        'test_counts': test_counts,
        'idle_idx': idle_idx,
        'required_idle': required_idle,
        'has_enough_idle': idle_train_count >= required_idle
    }

def compute_balanced_class_weights(y_train_int: np.ndarray, 
                                  idle_idx: int,
                                  analysis: Dict[str, Any]) -> Dict[int, float]:
    """
    Compute class weights with special handling for idle class.
    """
    # Basic balanced weights
    unique_classes = np.unique(y_train_int)
    weights = compute_class_weight(
        'balanced',
        classes=unique_classes,
        y=y_train_int
    )
    
    # Create weight dictionary
    weight_dict = {int(cls): float(w) for cls, w in zip(unique_classes, weights)}
    
    # If idle is under-represented, increase its weight
    if not analysis['has_enough_idle']:
        idle_count = analysis['train_counts'][idle_idx]
        max_active = max([analysis['train_counts'][i] for i in range(NUM_GESTURES) if i != idle_idx])
        
        if idle_count > 0 and max_active > 0:
            # Increase weight proportionally to the shortage
            shortage_ratio = max_active / idle_count
            adjustment = min(shortage_ratio, 3.0)  # Cap at 3x to avoid overfitting
            weight_dict[idle_idx] *= adjustment
            
            print(f"\n⚖️  Adjusted idle class weight by {adjustment:.2f}x")
            print(f"   New idle weight: {weight_dict[idle_idx]:.3f}")
    
    return weight_dict

def select_optimal_architecture(X_train: np.ndarray,
                               y_train: np.ndarray,
                               X_val: np.ndarray,
                               y_val: np.ndarray,
                               dataset_size: int) -> str:
    """
    Select the best model architecture based on dataset characteristics.
    """
    print("\n" + "=" * 60)
    print("ARCHITECTURE SELECTION")
    print("=" * 60)
    
    # Simple rule-based selection
    if dataset_size < 1000:
        print("   Small dataset (<1000 samples) -> Using CNN (simpler)")
        return "cnn"
    elif dataset_size < 5000:
        print("   Medium dataset -> Using CNN+LSTM (good temporal modeling)")
        return "cnn_lstm"
    else:
        print("   Large dataset -> Using Attention (best temporal patterns)")
        return "attention"

def train_model(args: argparse.Namespace) -> Tuple[GestureModel, Dict[str, Any]]:
    """
    Main training function with idle class handling.
    
    Returns:
        Tuple of (trained_model, training_metrics)
    """
    print("=" * 60)
    print("GESTURE RECOGNITION MODEL TRAINING")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 60)
    
    # Step 1: Load and prepare data
    print("\n1️⃣  Loading and preprocessing data...")
    pipeline = GestureDataPipeline(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
        random_seed=42
    )
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = pipeline.prepare_datasets(
        test_size=0.2,
        val_size=0.1,
        stratify=True,
        save_splits=True
    )
    
    print(f"   ✓ Training samples: {len(X_train)}")
    print(f"   ✓ Validation samples: {len(X_val)}")
    print(f"   ✓ Test samples: {len(X_test)}")
    
    # Step 2: Analyze distribution
    print("\n2️⃣  Analyzing dataset distribution...")
    y_train_int = np.argmax(y_train, axis=1)
    analysis = analyze_dataset_distribution(
        pipeline, X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # Step 3: Apply data augmentation if requested
    if args.augment:
        print("\n3️⃣  Applying data augmentation...")
        from data_pipeline import TemporalAugmenter
        
        # Only augment training data
        X_train_aug, y_train_aug = TemporalAugmenter.augment_batch(
            X_train, y_train_int,
            augment_factor=2,
            methods=['noise', 'scaling', 'crop']
        )
        
        # Convert back to one-hot
        from tensorflow.keras.utils import to_categorical
        y_train = to_categorical(y_train_aug, NUM_GESTURES)
        X_train = X_train_aug
        
        print(f"   ✓ Augmented training samples: {len(X_train)}")
    
    # Step 4: Compute class weights
    print("\n4️⃣  Computing class weights...")
    weight_dict = compute_balanced_class_weights(y_train_int, analysis['idle_idx'], analysis)
    
    print("\n   Final class weights:")
    for i, gesture in enumerate(GESTURE_CLASSES):
        weight = weight_dict.get(i, 1.0)
        marker = "⭐ " if gesture == "idle" else ""
        print(f"   {marker}{gesture:<15}: {weight:.3f}")
    
    # Step 5: Select and build model
    print("\n5️⃣  Building model architecture...")
    architecture = select_optimal_architecture(
        X_train, y_train, X_val, y_val, len(X_train)
    )
    
    model = GestureModel(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        num_classes=NUM_GESTURES,
        architecture=architecture,
        dataset_size=len(X_train)
    )
    
    # Compile model
    model.compile(learning_rate=args.learning_rate)
    
    # Print model summary
    print("\n   Model Architecture:")
    model.summary()
    
    # Step 6: Train model
    print(f"\n6️⃣  Training model for {args.epochs} epochs...")
    print("   (Press Ctrl+C to stop early and save best model)")
    
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weights=weight_dict
    )
    
    # Step 7: Evaluate on test set
    print("\n7️⃣  Evaluating on test set...")
    metrics = model.evaluate(X_test, y_test)
    
    # Print results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"📊 Overall Metrics:")
    print(f"   Test Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"   Test Precision: {metrics.get('precision', 0):.4f}")
    print(f"   Test Recall:    {metrics.get('recall', 0):.4f}")
    print(f"   Test F1-Score:  {metrics.get('f1_score', 0):.4f}")
    
    # Per-class accuracy
    per_class_acc = metrics.get('per_class_accuracy', [])
    if per_class_acc:
        print(f"\n🎯 Per-Class Accuracy (Idle is critical!):")
        for i, gesture in enumerate(GESTURE_CLASSES):
            acc = per_class_acc[i] if i < len(per_class_acc) else 0.0
            status = "✅" if acc > 0.8 else "⚠️ " if acc > 0.6 else "❌"
            marker = "⭐ " if gesture == "idle" else ""
            print(f"   {marker}{gesture:<15}: {acc:.3f} {status}")
    
    # Step 8: Save model
    print(f"\n8️⃣  Saving model...")
    model.save_model(FINAL_MODEL)
    
    # Convert to TFLite
    print(f"\n9️⃣  Converting to TFLite...")
    model.convert_to_tflite(
        output_path=TFLITE_MODEL,
        quantize=True,
        optimization_level=2
    )
    
    # Step 9: Plot training history
    print(f"\n🔟  Generating training plots...")
    model.plot_training_history(save_path=FINAL_MODEL.parent / "training_history.png")
    
    # Step 10: Show sample predictions
    print(f"\n📋 Sample predictions from test set:")
    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        sample = X_test[idx:idx+1]
        true_idx = np.argmax(y_test[idx])
        true_class = GESTURE_CLASSES[true_idx]
        
        pred_idx, confidence, probs = model.predict_sequence(sample[0])
        pred_class = GESTURE_CLASSES[pred_idx]
        
        # Format confidence bars
        conf_bar = "█" * int(confidence * 10)
        
        print(f"   Sample {i+1}:")
        print(f"     True: {true_class:<15} Predicted: {pred_class:<15}")
        print(f"     Confidence: {confidence:.3f} [{conf_bar:<10}]")
        
        # Show top 3 predictions if not correct
        if pred_idx != true_idx:
            top_indices = np.argsort(probs)[-3:][::-1]
            print(f"     Top predictions:")
            for j, top_idx in enumerate(top_indices):
                print(f"       {j+1}. {GESTURE_CLASSES[top_idx]:<15} {probs[top_idx]:.3f}")
        print()
    
    return model, metrics

def main():
    parser = argparse.ArgumentParser(
        description='Train gesture recognition model with idle class handling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_gesture_model.py                      # Train with defaults
  python train_gesture_model.py --epochs 50          # Train for 50 epochs
  python train_gesture_model.py --no_augment         # Disable augmentation
  python train_gesture_model.py --data_dir ./my_data # Custom data directory
        """
    )
    
    parser.add_argument('--data_dir', 
                       default='gesture_dataset',
                       help='Directory containing gesture data')
    parser.add_argument('--epochs', 
                       type=int, 
                       default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', 
                       type=int, 
                       default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', 
                       type=float, 
                       default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--augment', 
                       action='store_true', 
                       default=True,
                       help='Apply data augmentation (default: True)')
    parser.add_argument('--no_augment', 
                       action='store_false', 
                       dest='augment',
                       help='Disable data augmentation')
    parser.add_argument('--sequence_length', 
                       type=int, 
                       default=30,
                       help='Sequence length for temporal data')
    
    args = parser.parse_args()
    
    try:
        model, metrics = train_model(args)
        
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Model saved to: {FINAL_MODEL}")
        print(f"TFLite model saved to: {TFLITE_MODEL}")
        print(f"Test accuracy: {metrics.get('accuracy', 0):.4f}")
        
        # Give recommendations based on results
        per_class_acc = metrics.get('per_class_accuracy', [])
        if per_class_acc:
            idle_idx = GESTURE_CLASSES.index("idle")
            idle_acc = per_class_acc[idle_idx] if idle_idx < len(per_class_acc) else 0
            
            if idle_acc < 0.8:
                print(f"\n⚠️  RECOMMENDATION: Idle accuracy is low ({idle_acc:.3f})")
                print(f"   Collect more idle data: python main.py collect_idle")
            
            # Check for any class with very low accuracy
            for i, acc in enumerate(per_class_acc):
                if acc < 0.6 and i != idle_idx:
                    print(f"\n⚠️  RECOMMENDATION: {GESTURE_CLASSES[i]} accuracy is low ({acc:.3f})")
                    print(f"   Collect more data: python main.py collect --gestures {GESTURE_CLASSES[i]}")
        
        print("\nTo run real-time demo:")
        print("  python main.py demo")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        print("The best model so far has been saved.")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()