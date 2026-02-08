# setup.py
"""
Setup script for the gesture recognition system.
Installs dependencies and sets up the environment.
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is sufficient."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("Python 3.7 or higher is required")
        return False
    return True

def install_dependencies():
    """Install required Python packages."""
    requirements = [
        'tensorflow>=2.10.0',
        'opencv-python>=4.7.0',
        'mediapipe>=0.10.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.5.0',
        'protobuf>=3.20.0'
    ]
    
    print("Installing dependencies...")
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
            return False
    
    return True

def create_directory_structure():
    """Create necessary directories."""
    directories = [
        'gesture_dataset',
        'models',
        'exports',
        'logs',
        'unity_integration'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Create gesture subdirectories
    gesture_classes = [
        'run', 'punch_left', 'punch_right', 'kick',
        'lean_left', 'lean_right', 'squat', 'block'
    ]
    
    for gesture in gesture_classes:
        path = os.path.join('gesture_dataset', gesture)
        os.makedirs(path, exist_ok=True)

def download_mediapipe_models():
    """Download necessary MediaPipe models."""
    models = {
        'pose_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
    }
    
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    print("Downloading MediaPipe models...")
    
    import urllib.request
    
    for filename, url in models.items():
        filepath = os.path.join(models_dir, filename)
        if not os.path.exists(filepath):
            try:
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)
                print(f"✓ Downloaded {filename}")
            except Exception as e:
                print(f"✗ Failed to download {filename}: {e}")

def setup_unity_integration():
    """Create Unity integration files."""
    unity_dir = 'unity_integration'
    os.makedirs(unity_dir, exist_ok=True)
    
    # Create C# scripts
    scripts = {
        'GestureReceiver.cs': UNITY_RECEIVER_SCRIPT,
        'PlayerController.cs': PLAYER_CONTROLLER_SCRIPT,
        'GestureVisualizer.cs': GESTURE_VISUALIZER_SCRIPT
    }
    
    for filename, content in scripts.items():
        filepath = os.path.join(unity_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✓ Created Unity script: {filename}")

def verify_installation():
    """Verify the installation was successful."""
    print("\nVerifying installation...")
    
    tests = [
        ("Python version", lambda: sys.version_info >= (3, 7)),
        ("TensorFlow", lambda: __import__('tensorflow')),
        ("OpenCV", lambda: __import__('cv2')),
        ("MediaPipe", lambda: __import__('mediapipe')),
        ("NumPy", lambda: __import__('numpy')),
    ]
    
    all_passed = True
    
    for name, test_func in tests:
        try:
            test_func()
            print(f"✓ {name}: OK")
        except ImportError:
            print(f"✗ {name}: FAILED")
            all_passed = False
    
    return all_passed

def main():
    print("Gesture Recognition System Setup")
    print("=" * 50)
    
    if not check_python_version():
        sys.exit(1)
    
    steps = [
        ("Creating directory structure", create_directory_structure),
        ("Installing dependencies", install_dependencies),
        ("Downloading MediaPipe models", download_mediapipe_models),
        ("Setting up Unity integration", setup_unity_integration),
        ("Verifying installation", verify_installation)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if not step_func():
            print(f"Failed at step: {step_name}")
            sys.exit(1)
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: python main.py collect    (to collect gesture data)")
    print("2. Run: python main.py train      (to train the model)")
    print("3. Run: python main.py demo       (to test real-time recognition)")
    print("4. Import unity_integration/ scripts into your Unity project")

if __name__ == "__main__":
    # Import the Unity scripts from the integration module
    from unity_integration import UNITY_RECEIVER_SCRIPT, PLAYER_CONTROLLER_SCRIPT, GESTURE_VISUALIZER_SCRIPT
    main()