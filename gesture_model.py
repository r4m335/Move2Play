# gesture_model.py
"""
Flexible gesture classification models with runtime validation and adaptive regularization.
Supports CNN, CNN+LSTM, and attention-based architectures.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import json
from pathlib import Path

# Import from config
from config import (
    GESTURE_CLASSES, NUM_GESTURES, SEQUENCE_LENGTH,
    FINAL_MODEL, MODEL_METADATA, CONTRACT_EXPORT
)
from feature_engineer import FeatureEngineer

# ============================================================================
# MODEL ARCHITECTURE OPTIONS
# ============================================================================

class ModelArchitecture:
    """Defines different model architectures with their configurations."""
    
    # Base configurations for different architectures
    CNN_CONFIG = {
        'name': 'cnn',
        'description': '1D CNN with temporal convolutions',
        'conv_layers': [
            {'filters': 64, 'kernel_size': 3, 'dropout': 0.1},
            {'filters': 128, 'kernel_size': 3, 'dropout': 0.2},
            {'filters': 256, 'kernel_size': 3, 'dropout': 0.3},
        ],
        'dense_layers': [128, 64],
        'use_batch_norm': True,
        'use_attention': False,
        'use_lstm': False,
    }
    
    CNN_LSTM_CONFIG = {
        'name': 'cnn_lstm',
        'description': 'CNN for feature extraction + LSTM for temporal modeling',
        'conv_layers': [
            {'filters': 64, 'kernel_size': 3, 'dropout': 0.1},
            {'filters': 128, 'kernel_size': 3, 'dropout': 0.2},
        ],
        'lstm_units': 128,
        'lstm_dropout': 0.3,
        'dense_layers': [64],
        'use_batch_norm': True,
        'use_attention': False,
    }
    
    ATTENTION_CONFIG = {
        'name': 'attention',
        'description': 'CNN with temporal self-attention',
        'conv_layers': [
            {'filters': 64, 'kernel_size': 3, 'dropout': 0.1},
            {'filters': 128, 'kernel_size': 3, 'dropout': 0.2},
        ],
        'attention_heads': 4,
        'attention_dim': 64,
        'dense_layers': [128, 64],
        'use_batch_norm': True,
        'use_attention': True,
    }
    
    @staticmethod
    def get_config(architecture: str = 'cnn') -> Dict[str, Any]:
        """Get configuration for a specific architecture."""
        configs = {
            'cnn': ModelArchitecture.CNN_CONFIG,
            'cnn_lstm': ModelArchitecture.CNN_LSTM_CONFIG,
            'attention': ModelArchitecture.ATTENTION_CONFIG,
        }
        
        if architecture not in configs:
            raise ValueError(
                f"Unknown architecture: {architecture}. "
                f"Available: {list(configs.keys())}"
            )
        
        return configs[architecture].copy()

# ============================================================================
# ATTENTION LAYERS
# ============================================================================

class TemporalAttention(layers.Layer):
    """Self-attention layer for temporal sequences."""
    
    def __init__(self, attention_dim: int = 64, num_heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=attention_dim // num_heads
        )
        self.layer_norm = layers.LayerNormalization()
        self.dropout = layers.Dropout(0.1)
    
    def call(self, inputs, training=False):
        # Add positional encoding if not already present
        seq_len = tf.shape(inputs)[1]
        pos_encoding = self._positional_encoding(seq_len, inputs.shape[-1])
        inputs = inputs + pos_encoding
        
        # Self-attention
        attended = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            training=training
        )
        
        # Residual connection and layer norm
        attended = self.dropout(attended, training=training)
        attended = self.layer_norm(inputs + attended)
        
        return attended
    
    def _positional_encoding(self, seq_len: int, d_model: int):
        """Generate positional encoding for sequences."""
        position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(
            tf.range(0, d_model, 2, dtype=tf.float32) * 
            -(np.log(10000.0) / d_model)
        )
        
        pos_encoding = tf.zeros((seq_len, d_model))
        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding,
            tf.stack([tf.range(seq_len), tf.zeros(seq_len, dtype=tf.int32)], axis=1),
            tf.sin(position * div_term)
        )
        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding,
            tf.stack([tf.range(seq_len), tf.ones(seq_len, dtype=tf.int32)], axis=1),
            tf.cos(position * div_term)
        )
        
        return pos_encoding[tf.newaxis, ...]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'attention_dim': self.attention_dim,
            'num_heads': self.num_heads,
        })
        return config

# ============================================================================
# MAIN MODEL CLASS
# ============================================================================

class GestureModel:
    """
    Flexible gesture classification model with runtime validation and adaptive regularization.
    """
    
    def __init__(self, 
                 input_shape: Optional[Tuple[int, int]] = None,
                 num_classes: int = NUM_GESTURES,
                 architecture: str = 'cnn',
                 dataset_size: int = 0):
        """
        Args:
            input_shape: (time_steps, features). If None, will be determined at build time.
            num_classes: Number of gesture classes
            architecture: Model architecture ('cnn', 'cnn_lstm', or 'attention')
            dataset_size: Number of training samples for adaptive regularization
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.architecture = architecture
        self.dataset_size = dataset_size
        
        # Get architecture configuration
        self.config = ModelArchitecture.get_config(architecture)
        
        # Validate feature dimension at runtime
        self.feature_dim = None
        self.feature_engineer = FeatureEngineer()
        
        # Model will be built when needed
        self.model = None
        self.is_built = False
        
        # Training history
        self.history = None
        
        print(f"✅ GestureModel initialized")
        print(f"   Architecture: {self.config['name']}")
        print(f"   Description: {self.config['description']}")
        print(f"   Dataset size: {dataset_size}")
    
    def _build_model(self, input_shape: Tuple[int, int]):
        """Build the model with the given input shape."""
        print(f"Building model with input shape: {input_shape}")
        
        # Store feature dimension for validation
        self.feature_dim = input_shape[1]
        
        # Adaptive regularization based on dataset size
        dropout_rates = self._get_adaptive_dropout()
        print(f"Adaptive dropout rates: {dropout_rates}")
        
        # Build model based on architecture
        if self.architecture == 'cnn':
            self.model = self._build_cnn_model(input_shape, dropout_rates)
        elif self.architecture == 'cnn_lstm':
            self.model = self._build_cnn_lstm_model(input_shape, dropout_rates)
        elif self.architecture == 'attention':
            self.model = self._build_attention_model(input_shape, dropout_rates)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        self.is_built = True
        self.input_shape = input_shape
        
        return self.model
    
    def _build_cnn_model(self, input_shape: Tuple[int, int], dropout_rates: Dict[str, float]):
        """Build 1D CNN model."""
        inputs = layers.Input(shape=input_shape)
        x = inputs
        
        # Convolutional blocks
        for i, conv_config in enumerate(self.config['conv_layers']):
            x = layers.Conv1D(
                filters=conv_config['filters'],
                kernel_size=conv_config['kernel_size'],
                padding='same',
                name=f'conv_{i+1}'
            )(x)
            
            if self.config['use_batch_norm']:
                x = layers.BatchNormalization(name=f'bn_conv_{i+1}')(x)
            
            x = layers.ReLU(name=f'relu_conv_{i+1}')(x)
            
            if dropout_rates['conv'] > 0:
                x = layers.Dropout(
                    dropout_rates['conv'] * (i + 1) / len(self.config['conv_layers']),
                    name=f'dropout_conv_{i+1}'
                )(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D(name='global_pool')(x)
        
        # Dense layers
        for i, units in enumerate(self.config['dense_layers']):
            x = layers.Dense(units, name=f'dense_{i+1}')(x)
            
            if self.config['use_batch_norm']:
                x = layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)
            
            x = layers.ReLU(name=f'relu_dense_{i+1}')(x)
            
            if dropout_rates['dense'] > 0:
                x = layers.Dropout(
                    dropout_rates['dense'],
                    name=f'dropout_dense_{i+1}'
                )(x)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes, 
            activation='softmax',
            name='output'
        )(x)
        
        return keras.Model(inputs=inputs, outputs=outputs, name='GestureCNN')
    
    def _build_cnn_lstm_model(self, input_shape: Tuple[int, int], dropout_rates: Dict[str, float]):
        """Build CNN + LSTM model."""
        inputs = layers.Input(shape=input_shape)
        x = inputs
        
        # CNN feature extraction
        for i, conv_config in enumerate(self.config['conv_layers']):
            x = layers.Conv1D(
                filters=conv_config['filters'],
                kernel_size=conv_config['kernel_size'],
                padding='same',
                name=f'conv_{i+1}'
            )(x)
            
            if self.config['use_batch_norm']:
                x = layers.BatchNormalization(name=f'bn_conv_{i+1}')(x)
            
            x = layers.ReLU(name=f'relu_conv_{i+1}')(x)
            
            if dropout_rates['conv'] > 0:
                x = layers.Dropout(
                    dropout_rates['conv'] * (i + 1) / len(self.config['conv_layers']),
                    name=f'dropout_conv_{i+1}'
                )(x)
        
        # LSTM for temporal modeling
        x = layers.LSTM(
            units=self.config['lstm_units'],
            return_sequences=False,
            dropout=self.config.get('lstm_dropout', 0.3) if dropout_rates['lstm'] > 0 else 0.0,
            recurrent_dropout=self.config.get('lstm_dropout', 0.3) * 0.5 if dropout_rates['lstm'] > 0 else 0.0,
            name='lstm'
        )(x)
        
        # Dense layers
        for i, units in enumerate(self.config['dense_layers']):
            x = layers.Dense(units, name=f'dense_{i+1}')(x)
            
            if self.config['use_batch_norm']:
                x = layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)
            
            x = layers.ReLU(name=f'relu_dense_{i+1}')(x)
            
            if dropout_rates['dense'] > 0:
                x = layers.Dropout(
                    dropout_rates['dense'],
                    name=f'dropout_dense_{i+1}'
                )(x)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes, 
            activation='softmax',
            name='output'
        )(x)
        
        return keras.Model(inputs=inputs, outputs=outputs, name='GestureCNN_LSTM')
    
    def _build_attention_model(self, input_shape: Tuple[int, int], dropout_rates: Dict[str, float]):
        """Build CNN + Attention model."""
        inputs = layers.Input(shape=input_shape)
        x = inputs
        
        # CNN feature extraction
        for i, conv_config in enumerate(self.config['conv_layers']):
            x = layers.Conv1D(
                filters=conv_config['filters'],
                kernel_size=conv_config['kernel_size'],
                padding='same',
                name=f'conv_{i+1}'
            )(x)
            
            if self.config['use_batch_norm']:
                x = layers.BatchNormalization(name=f'bn_conv_{i+1}')(x)
            
            x = layers.ReLU(name=f'relu_conv_{i+1}')(x)
            
            if dropout_rates['conv'] > 0:
                x = layers.Dropout(
                    dropout_rates['conv'] * (i + 1) / len(self.config['conv_layers']),
                    name=f'dropout_conv_{i+1}'
                )(x)
        
        # Temporal self-attention
        x = TemporalAttention(
            attention_dim=self.config['attention_dim'],
            num_heads=self.config['attention_heads'],
            name='temporal_attention'
        )(x)
        
        # Global pooling (after attention)
        x = layers.GlobalAveragePooling1D(name='global_pool')(x)
        
        # Dense layers
        for i, units in enumerate(self.config['dense_layers']):
            x = layers.Dense(units, name=f'dense_{i+1}')(x)
            
            if self.config['use_batch_norm']:
                x = layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)
            
            x = layers.ReLU(name=f'relu_dense_{i+1}')(x)
            
            if dropout_rates['dense'] > 0:
                x = layers.Dropout(
                    dropout_rates['dense'],
                    name=f'dropout_dense_{i+1}'
                )(x)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes, 
            activation='softmax',
            name='output'
        )(x)
        
        return keras.Model(inputs=inputs, outputs=outputs, name='GestureAttention')
    
    def _get_adaptive_dropout(self) -> Dict[str, float]:
        """
        Calculate adaptive dropout rates based on dataset size.
        Smaller datasets need less regularization to prevent underfitting.
        """
        if self.dataset_size == 0:
            # Default moderate regularization
            return {
                'conv': 0.15,
                'dense': 0.25,
                'lstm': 0.2,
            }
        
        # Adjust based on dataset size
        if self.dataset_size < 1000:
            # Very small dataset - minimal regularization
            return {
                'conv': 0.05,
                'dense': 0.1,
                'lstm': 0.1,
            }
        elif self.dataset_size < 5000:
            # Small dataset - light regularization
            return {
                'conv': 0.1,
                'dense': 0.2,
                'lstm': 0.15,
            }
        elif self.dataset_size < 20000:
            # Medium dataset - moderate regularization
            return {
                'conv': 0.15,
                'dense': 0.3,
                'lstm': 0.2,
            }
        else:
            # Large dataset - strong regularization
            return {
                'conv': 0.2,
                'dense': 0.4,
                'lstm': 0.3,
            }
    
    def validate_input_shape(self, X: np.ndarray) -> Tuple[int, int]:
        """
        Validate input data and extract correct input shape.
        
        Args:
            X: Input data array
            
        Returns:
            Validated input shape (time_steps, features)
        """
        if len(X.shape) != 3:
            raise ValueError(
                f"Input data must be 3D (samples, time_steps, features). "
                f"Got shape: {X.shape}"
            )
        
        time_steps, features = X.shape[1], X.shape[2]
        
        # Validate time steps
        if time_steps != SEQUENCE_LENGTH:
            print(f"⚠️  Warning: Input time_steps ({time_steps}) "
                  f"doesn't match SEQUENCE_LENGTH ({SEQUENCE_LENGTH})")
        
        # Validate feature dimension using FeatureEngineer
        test_landmarks = np.zeros((1, 33, 3))  # Single frame of zeros
        test_features = self.feature_engineer.extract_features(test_landmarks)
        expected_features = test_features.shape[1]
        
        if features != expected_features:
            raise ValueError(
                f"Feature dimension mismatch. "
                f"Expected {expected_features} features from FeatureEngineer, "
                f"but got {features}. "
                f"Check FeatureEngineer implementation."
            )
        
        print(f"✅ Input validation passed:")
        print(f"   Time steps: {time_steps}")
        print(f"   Features: {features} (validated with FeatureEngineer)")
        
        return time_steps, features
    
    def compile(self, 
                learning_rate: float = 0.001,
                optimizer: Optional[keras.optimizers.Optimizer] = None):
        """
        Compile the model with optimizer and metrics.
        
        Args:
            learning_rate: Learning rate for optimizer
            optimizer: Custom optimizer (uses Adam if None)
        """
        if self.model is None:
            raise RuntimeError("Model must be built before compilation. "
                             "Call build() or train() first.")
        
        if optimizer is None:
            optimizer = keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            )
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
            ]
        )
        
        print(f"✅ Model compiled")
        print(f"   Optimizer: {optimizer.__class__.__name__}")
        print(f"   Learning rate: {learning_rate}")
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray, 
              X_val: np.ndarray, 
              y_val: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32,
              class_weights: Optional[Dict[int, float]] = None,
              callbacks: Optional[List] = None) -> keras.callbacks.History:
        """
        Train the model with validation and callbacks.
        
        Args:
            X_train: Training data
            y_train: Training labels (one-hot encoded)
            X_val: Validation data
            y_val: Validation labels (one-hot encoded)
            epochs: Maximum number of epochs
            batch_size: Batch size for training
            class_weights: Optional class weights for imbalanced data
            callbacks: Optional list of Keras callbacks
            
        Returns:
            Training history
        """
        # Validate and build model if needed
        if not self.is_built:
            input_shape = self.validate_input_shape(X_train)
            self._build_model(input_shape)
        
        # Compile if not already compiled
        if not self.model.optimizer:
            self.compile()
        
        # Default callbacks
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        # Print training summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"Architecture: {self.architecture}")
        print(f"Class weights: {class_weights is not None}")
        print("=" * 60)
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        return self.history
    
    def _get_default_callbacks(self) -> List[keras.callbacks.Callback]:
        """Get default callbacks for training."""
        return [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,  # Increased patience for complex architectures
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,  # Reduced patience for LR reduction
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(FINAL_MODEL),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir='logs/tensorboard',
                histogram_freq=1,
                write_graph=True,
                write_images=False,
                update_freq='epoch'
            ),
        ]
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test data
            y_test: Test labels (one-hot encoded)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before evaluation.")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Map metric names to values
        metric_names = [m.name if hasattr(m, 'name') else m for m in self.model.metrics]
        if len(metric_names) != len(results):
            metric_names = ['loss'] + [f'metric_{i}' for i in range(len(results)-1)]
        
        metrics = dict(zip(metric_names, results))
        
        # Calculate additional metrics
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        from sklearn.metrics import f1_score, classification_report
        
        metrics['f1_score'] = f1_score(
            y_true_classes, 
            y_pred_classes, 
            average='weighted'
        )
        
        # Per-class accuracy
        per_class_acc = []
        for i in range(self.num_classes):
            mask = y_true_classes == i
            if np.any(mask):
                acc = np.mean(y_pred_classes[mask] == i)
                per_class_acc.append(acc)
            else:
                per_class_acc.append(0.0)
        
        metrics['per_class_accuracy'] = per_class_acc
        metrics['min_class_accuracy'] = np.min(per_class_acc) if per_class_acc else 0.0
        
        return metrics
    
    def predict_sequence(self, sequence: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Predict gesture for a single sequence.
        
        Args:
            sequence: Input sequence of shape (time_steps, features) or (1, time_steps, features)
            
        Returns:
            Tuple of (predicted_class_index, confidence, all_probabilities)
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction.")
        
        # Add batch dimension if needed
        if len(sequence.shape) == 2:
            sequence = np.expand_dims(sequence, axis=0)
        
        # Validate input shape
        if sequence.shape[1:] != self.input_shape:
            raise ValueError(
                f"Input shape {sequence.shape[1:]} doesn't match "
                f"model input shape {self.input_shape}"
            )
        
        predictions = self.model.predict(sequence, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        return class_idx, confidence, predictions[0]
    
    def save_model(self, filepath: Path = FINAL_MODEL):
        """Save the trained model and metadata."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        
        # Save the model
        self.model.save(filepath)
        
        # Save metadata
        metadata = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'architecture': self.architecture,
            'feature_dim': self.feature_dim,
            'dataset_size': self.dataset_size,
            'gesture_classes': GESTURE_CLASSES,
            'sequence_length': SEQUENCE_LENGTH,
            'config': self.config,
            'training_history': {
                'final_accuracy': float(self.history.history['val_accuracy'][-1]) if self.history else None,
                'final_loss': float(self.history.history['val_loss'][-1]) if self.history else None,
                'epochs_trained': len(self.history.history['loss']) if self.history else 0,
            },
            'timestamp': np.datetime64('now').astype(str),
        }
        
        with open(MODEL_METADATA, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Also save contract for reference
        from gesture_action_contract import GestureActionContract
        GestureActionContract.export_to_json(CONTRACT_EXPORT)
        
        print(f"✅ Model saved to {filepath}")
        print(f"✅ Metadata saved to {MODEL_METADATA}")
    
    def load_model(self, filepath: Path = FINAL_MODEL):
        """Load a trained model and its metadata."""
        # Load the model
        self.model = keras.models.load_model(
            filepath,
            custom_objects={'TemporalAttention': TemporalAttention}
        )
        
        # Load metadata
        if MODEL_METADATA.exists():
            with open(MODEL_METADATA, 'r') as f:
                metadata = json.load(f)
            
            self.input_shape = tuple(metadata['input_shape'])
            self.num_classes = metadata['num_classes']
            self.architecture = metadata['architecture']
            self.feature_dim = metadata['feature_dim']
            self.dataset_size = metadata['dataset_size']
            self.config = metadata['config']
            self.is_built = True
            
            print(f"✅ Model loaded from {filepath}")
            print(f"   Input shape: {self.input_shape}")
            print(f"   Architecture: {self.architecture}")
            print(f"   Feature dimension: {self.feature_dim}")
        else:
            print(f"⚠️  Metadata file not found: {MODEL_METADATA}")
            # Infer from model
            self.input_shape = self.model.input_shape[1:]
            self.is_built = True
    
    def convert_to_tflite(self, 
                         output_path: Path,
                         quantize: bool = True,
                         optimization_level: int = 2):
        """
        Convert model to TensorFlow Lite for mobile deployment.
        
        Args:
            output_path: Path to save TFLite model
            quantize: Whether to apply quantization
            optimization_level: TFLite optimization level (0-3)
        """
        if self.model is None:
            raise RuntimeError("No model to convert.")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Set optimization level
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if quantize:
            # Dynamic range quantization (good balance of size/accuracy)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Optionally: FP16 quantization for GPU acceleration
            # converter.target_spec.supported_types = [tf.float16]
        
        # Set optimization level
        if optimization_level >= 1:
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        if optimization_level >= 2:
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Test conversion with sample input
        self._test_tflite_conversion(tflite_model)
        
        print(f"✅ TFLite model saved to {output_path}")
        print(f"   Quantization: {quantize}")
        print(f"   Optimization level: {optimization_level}")
        
        return tflite_model
    
    def _test_tflite_conversion(self, tflite_model: bytes):
        """Test TFLite model conversion with sample input."""
        try:
            # Create interpreter
            interpreter = tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()
            
            # Get input details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Create sample input
            sample_input = np.random.randn(1, *self.input_shape).astype(np.float32)
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], sample_input)
            interpreter.invoke()
            
            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            print(f"✅ TFLite conversion test passed")
            print(f"   Input shape: {input_details[0]['shape']}")
            print(f"   Output shape: {output_details[0]['shape']}")
            
        except Exception as e:
            print(f"⚠️  TFLite conversion test failed: {e}")
    
    def summary(self):
        """Print model summary."""
        if self.model:
            self.model.summary()
        else:
            print("Model not built yet.")
    
    def plot_training_history(self, save_path: Optional[Path] = None):
        """Plot training history."""
        if self.history is None:
            print("No training history available.")
            return
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Train Precision')
            axes[1, 0].plot(self.history.history['val_precision'], label='Val Precision')
            axes[1, 0].set_title('Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Train Recall')
            axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall')
            axes[1, 1].set_title('Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Training history plot saved to {save_path}")
        
        plt.show()

# ============================================================================
# MODEL SELECTION UTILITIES
# ============================================================================

def select_best_architecture(X_train: np.ndarray, 
                           y_train: np.ndarray,
                           X_val: np.ndarray,
                           y_val: np.ndarray,
                           dataset_size: int) -> str:
    """
    Select the best architecture based on validation performance.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        dataset_size: Size of the training dataset
        
    Returns:
        Name of the best architecture
    """
    architectures = ['cnn', 'cnn_lstm', 'attention']
    results = {}
    
    print("\n" + "=" * 60)
    print("ARCHITECTURE SELECTION")
    print("=" * 60)
    
    for arch in architectures:
        print(f"\nTesting {arch} architecture...")
        
        try:
            # Create and train model
            model = GestureModel(
                input_shape=None,
                num_classes=y_train.shape[1],
                architecture=arch,
                dataset_size=dataset_size
            )
            
            # Quick training with reduced epochs
            history = model.train(
                X_train, y_train,
                X_val, y_val,
                epochs=30,  # Quick test
                batch_size=32,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    )
                ]
            )
            
            # Get best validation accuracy
            best_val_acc = max(history.history['val_accuracy'])
            results[arch] = best_val_acc
            
            print(f"  Best validation accuracy: {best_val_acc:.4f}")
            
        except Exception as e:
            print(f"  ⚠️  Failed to train {arch}: {e}")
            results[arch] = 0.0
    
    # Select best architecture
    best_arch = max(results, key=results.get)
    best_acc = results[best_arch]
    
    print("\n" + "=" * 60)
    print("SELECTION RESULTS")
    print("=" * 60)
    for arch, acc in results.items():
        marker = " ✅" if arch == best_arch else ""
        print(f"{arch:12s}: {acc:.4f}{marker}")
    
    print(f"\nSelected architecture: {best_arch} (accuracy: {best_acc:.4f})")
    
    return best_arch

# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_model():
    """Test the gesture model."""
    print("Testing Gesture Model...")
    
    # Create synthetic data
    time_steps = SEQUENCE_LENGTH
    features = FeatureEngineer().extract_features(np.zeros((1, 33, 3))).shape[1]
    num_samples = 100
    
    X_train = np.random.randn(num_samples, time_steps, features).astype(np.float32)
    y_train = np.eye(NUM_GESTURES)[np.random.randint(0, NUM_GESTURES, num_samples)]
    
    X_val = np.random.randn(20, time_steps, features).astype(np.float32)
    y_val = np.eye(NUM_GESTURES)[np.random.randint(0, NUM_GESTURES, 20)]
    
    # Test different architectures
    for architecture in ['cnn', 'cnn_lstm', 'attention']:
        print(f"\n{'='*60}")
        print(f"Testing {architecture} architecture")
        print(f"{'='*60}")
        
        try:
            model = GestureModel(
                input_shape=None,
                num_classes=NUM_GESTURES,
                architecture=architecture,
                dataset_size=num_samples
            )
            
            # Train
            history = model.train(
                X_train, y_train,
                X_val, y_val,
                epochs=5,  # Quick test
                batch_size=16
            )
            
            # Evaluate
            metrics = model.evaluate(X_val, y_val)
            print(f"Validation accuracy: {metrics['accuracy']:.4f}")
            
            # Test prediction
            test_sequence = X_val[0]
            class_idx, confidence, probs = model.predict_sequence(test_sequence)
            print(f"Sample prediction: class={class_idx}, confidence={confidence:.4f}")
            
            # Save and load
            test_model_path = Path(f"test_model_{architecture}.keras")
            model.save_model(test_model_path)
            
            # Load back
            loaded_model = GestureModel()
            loaded_model.load_model(test_model_path)
            
            print(f"✅ {architecture} architecture test passed")
            
        except Exception as e:
            print(f"❌ {architecture} architecture test failed: {e}")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    test_model()