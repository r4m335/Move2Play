# train_gesture_model.py
"""
Main training script for the gesture recognition model.
"""

import argparse
import numpy as np
from data_pipeline import GestureDataPipeline
from gesture_model import GestureCNN
import matplotlib.pyplot as plt

def plot_training_history(history):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
    axes[1, 0].set_title('Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
    axes[1, 1].set_title('Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train gesture recognition model')
    parser.add_argument('--data_dir', default='gesture_dataset', help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--sequence_length', type=int, default=30, help='Sequence length')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Gesture Recognition Model Training")
    print("=" * 50)
    
    # Step 1: Load and prepare data
    print("\n1. Loading and preprocessing data...")
    pipeline = GestureDataPipeline(args.data_dir, args.sequence_length)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = pipeline.prepare_datasets()
    
    # Get integer labels for class weights
    y_train_int = np.argmax(y_train, axis=1)
    class_weights = pipeline.compute_class_weights(y_train_int)
    
    print(f"Class weights: {class_weights}")
    
    # Step 2: Data augmentation
    if args.augment:
        print("\n2. Applying data augmentation...")
        X_train, y_train = pipeline.augment_sequences(X_train, y_train, augment_factor=2)
        print(f"After augmentation: {len(X_train)} training samples")
    
    # Step 3: Build and compile model
    print("\n3. Building model...")
    input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, features)
    model = GestureCNN(input_shape=input_shape, num_classes=len(pipeline.gesture_classes))
    model.compile_model(learning_rate=args.learning_rate)
    
    model.model.summary()
    
    # Step 4: Train model
    print("\n4. Training model...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Step 5: Evaluate on test set
    print("\n5. Evaluating on test set...")
    test_loss, test_accuracy, test_precision, test_recall = model.model.evaluate(
        X_test, y_test, verbose=0
    )
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    # Step 6: Save model
    print("\n6. Saving model...")
    model.save_model('gesture_model.keras')
    
    # Convert to TFLite
    model.convert_to_tflite('gesture_model.tflite', quantize=True)
    
    # Step 7: Plot training history
    plot_training_history(history)
    
    # Step 8: Show sample predictions
    print("\n7. Sample predictions:")
    for i in range(3):
        sample = X_test[i:i+1]
        true_class = pipeline.gesture_classes[np.argmax(y_test[i])]
        
        pred_idx, confidence, _ = model.predict_sequence(sample)
        pred_class = pipeline.gesture_classes[pred_idx]
        
        print(f"Sample {i+1}: True={true_class}, Predicted={pred_class} (Confidence: {confidence:.2f})")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()