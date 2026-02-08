# Gesture-Based Zombie Runner Game

A complete implementation of a gesture-controlled mobile game using pose estimation and CNN classification.

## Architecture Overview

```
[Camera Input] → [MediaPipe Pose] → [Feature Engineering] → [CNN Classifier] → [Action Mapping] → [Unity Game]
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd gesture-zombie-runner

# Run setup script
python setup.py
```

### 2. Collect Gesture Data

```bash
# Collect gesture samples
python main.py collect
```

Follow on-screen instructions to record each gesture:
- Run
- Punch (left/right)
- Kick
- Lean left/right
- Squat
- Block

### 3. Train the Model

```bash
# Train the CNN model
python main.py train

# Optional: Monitor training with TensorBoard
tensorboard --logdir=logs
```

### 4. Test the System

```bash
# Test with webcam
python main.py demo
```

## Project Structure

```
gesture-zombie-runner/
├── gesture_dataset/          # Collected gesture sequences
├── models/                   # Trained models
├── exports/                  # Exported TFLite models
├── unity_integration/        # Unity C# scripts
├── pose_extractor.py         # MediaPipe pose extraction
├── feature_engineer.py       # Feature engineering
├── gesture_model.py          # CNN model architecture
├── real_time_inference.py    # Real-time recognition
├── unity_integration.py      # Unity bridge
└── main.py                   # Main entry point
```

## Gesture-Action Mapping

| Gesture | Game Action | Parameters |
|---------|-------------|------------|
| Run | Forward Movement | speed=1.0 |
| Punch Left | Attack | side=left, damage=25 |
| Punch Right | Attack | side=right, damage=25 |
| Kick | Attack | type=kick, damage=35 |
| Lean Left | Dodge | direction=left, distance=2.0 |
| Lean Right | Dodge | direction=right, distance=2.0 |
| Squat | Slide | duration=1.5, height=0.5 |
| Block | Defense | reduction=0.7, duration=2.0 |

## Performance Targets

- **Inference latency:** < 30ms on mobile
- **Classification accuracy:** > 90%
- **End-to-end latency:** < 150ms
- **Frame rate:** 15-20 FPS (pose + inference)

## Mobile Optimization

- **Model Quantization:** FP16/INT8 quantization for TFLite
- **Resolution:** 480p camera input
- **Batch Processing:** Process every 2-3 frames
- **NNAPI:** Use hardware acceleration when available

## Testing Protocol

Run the complete test suite:

```bash
python test_suite.py --all
```

Tests include:
- Gesture accuracy (offline)
- Latency measurements
- False positive rate during movement
- Long-session stability
- Cross-user generalization

## License

MIT License - See LICENSE file for details.

## Citation

If you use this in research, please cite:

```text
@software{gesture_zombie_runner_2023,
  title = {Gesture-Based Zombie Runner: Pose Estimation Game System},
  author = {Your Name},
  year = {2023},
  url = {https://github.com/yourusername/gesture-zombie-runner}
}
```

## Support

For issues and questions:
- Check the troubleshooting guide in docs/
- Open an issue on GitHub
- Contact: your.email@example.com
