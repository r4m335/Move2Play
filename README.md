# Move2Play 🏃‍♂️🎮

**Move2Play** is an interactive, markerless motion-controlled exergaming survival action game built in **Unity (URP)** and powered by real-time computer vision using **Google MediaPipe**. 

Players control their character inside an open 3D environment by performing real-world physical gestures—such as running in place, turning, jumping, punching, kicking, dodging, and blocking—captured through a standard RGB webcam with zero extra hardware requirements.

---

## 🌟 Key Features

- **Markerless Full-Body Motion Tracking**: Real-time 33-point 3D pose landmark detection powered by Google MediaPipe. No wearable sensors, specialized controllers, or depth cameras required.
- **Biomechanical Kinematic Engine**: Rule-based gesture recognition pipeline analyzing joint angles, arm abduction, hip displacement impulse, and limb elevation with time-weighted prediction smoothing.
- **Interactive Calibration System**: Guided tutorial ensuring player body proportions, camera distance, and lighting conditions are verified before entering the game.
- **Ergonomic Run-Lock UX**: Automatically engages after 5 seconds of sustained running to reduce arm and shoulder fatigue.
- **Immersive Combat & Survival Loop**:
  - Punch, kick, block (50% damage reduction), and dodge attacks from procedural zombie waves.
  - Multi-tiered spatial awareness: 360° off-screen compass radar and distance-scaled heartbeat audio.
  - Dynamic MedKit navigation arrow that activates under low health conditions ($\le 30\%$).
- **Atmospheric 3D World**: Universal Render Pipeline (URP) environment featuring procedural terrain spawning, water hazards, ambient wildlife, and a cinematic dragon patrol AI.

---

## 🕹️ Motion Controls

| Action | Physical Gesture |
| :--- | :--- |
| **Run** | Raise both arms wide (T-Pose) or sprint in place |
| **Turn Left** | Raise only your left arm sideways |
| **Turn Right** | Raise only your right arm sideways |
| **Jump** | Jump vertically in place |
| **Punch** | Extend wrist forward |
| **Kick** | Raise knee or perform forward kick |
| **Block / Guard** | Bring both hands close to face/head |
| **Dodge / Squat** | Squat down (bend knees $< 145^\circ$) |

*Note: Full keyboard fallbacks (WASD / Arrows, Space, Shift, Mouse Click, G) are also supported.*

---

## 🛠️ Technology Stack

- **Game Engine**: Unity 2022.3 LTS (Universal Render Pipeline - URP)
- **Computer Vision**: Google MediaPipe Pose via [MediaPipeUnityPlugin](https://github.com/homuler/MediaPipeUnityPlugin)
- **Programming Language**: C# (.NET / Mono)
- **Audio Engine**: Custom multi-channel Audio Manager (SFX, Ambient, Music)
- **Physics & AI**: Unity NavMesh Agent, Physics OverlapSphere, Terrain Raycasting

---

## 📁 Repository Structure

```
├── Assets/
│   ├── scripts/
│   │   ├── CV/                          # Computer vision, MediaPipe bridge & recognizer
│   │   │   ├── MediaPipeBridge.cs
│   │   │   ├── MediaPipePoseManager.cs
│   │   │   ├── RuleBasedGestureRecognizer.cs
│   │   │   ├── PoseVisualizer.cs
│   │   │   └── GestureDebugUI.cs
│   │   ├── CalibrationManager.cs        # Pre-game calibration & pose verification
│   │   ├── PlayerController.cs          # Main movement, gesture dispatcher & Run-Lock
│   │   ├── CombatController.cs          # Punch/kick detection, damage radius & block
│   │   ├── PlayerStats.cs               # Health, drowning, damage flash
│   │   ├── EnemyAI.cs                   # 4-state zombie FSM (Idle, Wander, Chase, Attack)
│   │   ├── EnemySpawner.cs              # Terrain-aware procedural wave spawner
│   │   ├── DragonAI.cs                  # Cinematic flying dragon waypoint AI
│   │   ├── EnemyTrackerUI.cs            # Directional radar arrow & heartbeat audio
│   │   ├── MedKitTracker.cs             # Compass to nearest health pickup
│   │   ├── AudioManager.cs              # Centralized audio channel manager
│   │   └── UIController.cs              # Game loop, survival timer, victory/game over
│   └── Scenes/
│       ├── MainMenu.unity
│       ├── Calibration.unity
│       ├── Guide.unity
│       └── Forest.unity
├── Packages/                            # Package manifests and MediaPipe package
├── ProjectSettings/                     # Unity project settings and input configurations
└── .gitignore                           # Excludes Library, Temp, and build artifacts
```

---

## 🚀 Getting Started

### Prerequisites
- **Unity Editor**: `2022.3.62f1` (or compatible 2022.3 LTS release)
- **Webcam**: Standard USB or integrated laptop webcam (720p @ 30 FPS recommended)
- **Hardware**: Windows 10/11 x64

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/r4m335/Move2Play.git
   ```
2. Open **Unity Hub** and click **Add** $\to$ **Add project from disk**.
3. Select the cloned `Move2Play` directory.
4. Allow Unity to resolve packages and generate the local `Library` cache.
5. In the Unity Project window, open `Assets/Scenes/MainMenu.unity`.
6. Press **Play** in the Unity Editor or build an executable via **File > Build Settings**.

---

## 👥 Authors & Academic Credits
Final Year Project (FYP) developed for Software Engineering.
