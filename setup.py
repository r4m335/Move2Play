# setup.py
"""
Setup script for the gesture recognition system.
Installs dependencies and sets up the environment.
"""

import os
import sys
import subprocess
import platform
import urllib.request

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
        'idle', 'run', 'punch_left', 'punch_right', 'kick',
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
    
    # Also create a Python example script for Unity integration
    python_example = '''"""
Example script for connecting to Unity from Python.
Run this after starting your Unity game.
"""

import time
from gesture_to_unity import GestureToUnity

def main():
    # Create the gesture sender
    sender = GestureToUnity(host='127.0.0.1', port=65432)
    
    # Connect to Unity
    if sender.connect():
        print("Connected to Unity!")
        print("Press Ctrl+C to stop")
        
        try:
            # Example: Send some test gestures
            test_gestures = [
                ('idle', 0.95),
                ('punch_left', 0.98),
                ('punch_right', 0.97),
                ('kick', 0.96),
                ('lean_left', 0.94),
                ('lean_right', 0.93),
                ('run', 0.95),
                ('squat', 0.96),
                ('block', 0.97)
            ]
            
            for gesture, confidence in test_gestures:
                print(f"Sending: {gesture}")
                sender.send_gesture(gesture, confidence)
                time.sleep(1.5)
                
        except KeyboardInterrupt:
            print("\\nStopping...")
        finally:
            sender.disconnect()
    else:
        print("Failed to connect to Unity. Make sure:")
        print("1. Unity game is running")
        print("2. GestureReceiver component is attached")
        print("3. Unity is listening on port 65432")

if __name__ == "__main__":
    main()
'''
    
    example_path = os.path.join(unity_dir, 'unity_example.py')
    with open(example_path, 'w') as f:
        f.write(python_example)
    print(f"✓ Created Unity example script: unity_example.py")

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

# Unity C# scripts (defined here to avoid import issues)
UNITY_RECEIVER_SCRIPT = """
using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;
using System.Text;
using System.Collections.Generic;
using System.Collections;

public class GestureReceiver : MonoBehaviour
{
    private TcpListener server;
    private Thread serverThread;
    private bool isRunning = true;
    private readonly object lockObject = new object();
    
    [Header("Connection Settings")]
    public string host = "127.0.0.1";
    public int port = 65432;
    
    [Header("Game Object References")]
    public GameObject playerController;
    public GameObject playerCombat;
    public GameObject playerMovement;
    
    [Header("Debug")]
    public bool logMessages = true;
    public float connectionCheckInterval = 5.0f;
    
    // Message queue for thread-safe communication
    private Queue<string> messageQueue = new Queue<string>();
    
    // Client connection
    private TcpClient connectedClient;
    private NetworkStream clientStream;
    
    void Start()
    {
        StartServer();
        StartCoroutine(CheckConnectionStatus());
    }
    
    void StartServer()
    {
        serverThread = new Thread(new ThreadStart(ListenForConnections));
        serverThread.IsBackground = true;
        serverThread.Start();
        Debug.Log($"Gesture receiver server starting on {host}:{port}");
    }
    
    void ListenForConnections()
    {
        try
        {
            IPAddress ipAddress = IPAddress.Parse(host);
            server = new TcpListener(ipAddress, port);
            server.Start();
            
            Debug.Log($"Server started. Waiting for Python client...");
            
            while (isRunning)
            {
                // Accept a client connection
                connectedClient = server.AcceptTcpClient();
                clientStream = connectedClient.GetStream();
                
                Debug.Log("Python client connected!");
                
                // Handle this client
                HandleClient();
            }
        }
        catch (SocketException e)
        {
            Debug.LogError($"SocketException: {e}");
        }
        finally
        {
            server?.Stop();
        }
    }
    
    void HandleClient()
    {
        byte[] buffer = new byte[4096];
        StringBuilder messageBuilder = new StringBuilder();
        
        try
        {
            while (isRunning && connectedClient.Connected)
            {
                int bytesRead = clientStream.Read(buffer, 0, buffer.Length);
                
                if (bytesRead == 0)
                {
                    // Client disconnected
                    Debug.Log("Python client disconnected.");
                    break;
                }
                
                string data = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                messageBuilder.Append(data);
                
                // Process complete messages (separated by newline)
                string allData = messageBuilder.ToString();
                int newlineIndex;
                
                while ((newlineIndex = allData.IndexOf('\\n')) >= 0)
                {
                    string message = allData.Substring(0, newlineIndex).Trim();
                    allData = allData.Substring(newlineIndex + 1);
                    
                    if (!string.IsNullOrEmpty(message))
                    {
                        lock (lockObject)
                        {
                            messageQueue.Enqueue(message);
                        }
                        
                        // Optional: Send acknowledgment back to Python
                        SendAcknowledgment(message);
                    }
                }
                
                messageBuilder.Clear();
                messageBuilder.Append(allData);
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error handling client: {e}");
        }
        finally
        {
            connectedClient?.Close();
            clientStream?.Close();
            Debug.Log("Client connection closed.");
        }
    }
    
    void SendAcknowledgment(string originalMessage)
    {
        try
        {
            // Simple acknowledgment
            var ack = new
            {
                type = "ack",
                original_message = originalMessage.Substring(0, Math.Min(50, originalMessage.Length)),
                timestamp = DateTime.Now.ToString("HH:mm:ss")
            };
            
            string ackJson = JsonUtility.ToJson(ack);
            byte[] ackBytes = Encoding.UTF8.GetBytes(ackJson + "\\n");
            
            if (clientStream != null && clientStream.CanWrite)
            {
                clientStream.Write(ackBytes, 0, ackBytes.Length);
            }
        }
        catch
        {
            // Ignore acknowledgment errors
        }
    }
    
    void Update()
    {
        // Process messages on main thread
        while (messageQueue.Count > 0)
        {
            string message;
            lock (lockObject)
            {
                message = messageQueue.Dequeue();
            }
            
            ProcessMessage(message);
        }
    }
    
    void ProcessMessage(string jsonMessage)
    {
        try
        {
            if (logMessages)
            {
                Debug.Log($"Received: {jsonMessage}");
            }
            
            // Parse the JSON
            var wrapper = JsonUtility.FromJson<MessageWrapper>(jsonMessage);
            
            switch (wrapper.type)
            {
                case "action":
                    ProcessActionMessage(jsonMessage);
                    break;
                    
                case "gesture":
                    ProcessGestureMessage(jsonMessage);
                    break;
                    
                case "landmarks":
                    ProcessLandmarksMessage(jsonMessage);
                    break;
                    
                case "handshake":
                    Debug.Log("Python client handshake received");
                    break;
                    
                case "ping":
                    // Respond to ping
                    SendPong();
                    break;
                    
                case "disconnect":
                    Debug.Log("Python client requested disconnect");
                    break;
                    
                default:
                    Debug.LogWarning($"Unknown message type: {wrapper.type}");
                    break;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error processing message: {e.Message}\\nMessage: {jsonMessage}");
        }
    }
    
    void ProcessActionMessage(string jsonMessage)
    {
        var action = JsonUtility.FromJson<GestureAction>(jsonMessage);
        
        if (action == null)
        {
            Debug.LogError("Failed to parse action message");
            return;
        }
        
        GameObject targetObject = null;
        
        switch (action.objectName)
        {
            case "PlayerController":
                targetObject = playerController;
                break;
            case "PlayerCombat":
                targetObject = playerCombat;
                break;
            case "PlayerMovement":
                targetObject = playerMovement;
                break;
        }
        
        if (targetObject != null)
        {
            // Convert parameters to appropriate types
            object[] processedParams = new object[action.parameters.Length];
            
            for (int i = 0; i < action.parameters.Length; i++)
            {
                if (float.TryParse(action.parameters[i], out float floatValue))
                {
                    processedParams[i] = floatValue;
                }
                else
                {
                    processedParams[i] = action.parameters[i];
                }
            }
            
            // Send message to GameObject
            targetObject.SendMessage(action.methodName, processedParams.Length > 0 ? processedParams : null, 
                                   SendMessageOptions.DontRequireReceiver);
            
            Debug.Log($"Executed: {action.objectName}.{action.methodName}({string.Join(", ", action.parameters)})");
        }
        else
        {
            Debug.LogWarning($"Target object not found for: {action.objectName}");
        }
    }
    
    void ProcessGestureMessage(string jsonMessage)
    {
        var gesture = JsonUtility.FromJson<GestureData>(jsonMessage);
        Debug.Log($"Gesture detected: {gesture.gesture} (confidence: {gesture.confidence:P0})");
    }
    
    void ProcessLandmarksMessage(string jsonMessage)
    {
        // Optional: Process landmarks for visualization
        var landmarks = JsonUtility.FromJson<LandmarksData>(jsonMessage);
        // Could visualize landmarks in Unity scene
    }
    
    void SendPong()
    {
        try
        {
            var pong = new
            {
                type = "pong",
                timestamp = DateTime.Now.ToString("HH:mm:ss")
            };
            
            string pongJson = JsonUtility.ToJson(pong);
            byte[] pongBytes = Encoding.UTF8.GetBytes(pongJson + "\\n");
            
            if (clientStream != null && clientStream.CanWrite)
            {
                clientStream.Write(pongBytes, 0, pongBytes.Length);
            }
        }
        catch
        {
            // Ignore
        }
    }
    
    IEnumerator CheckConnectionStatus()
    {
        while (true)
        {
            yield return new WaitForSeconds(connectionCheckInterval);
            
            if (connectedClient == null || !connectedClient.Connected)
            {
                Debug.Log("Waiting for Python client connection...");
            }
        }
    }
    
    void OnDestroy()
    {
        isRunning = false;
        server?.Stop();
        serverThread?.Join(1000);
        connectedClient?.Close();
        clientStream?.Close();
        
        Debug.Log("Gesture receiver stopped.");
    }
    
    // Data classes for JSON deserialization
    [System.Serializable]
    private class MessageWrapper
    {
        public string type;
    }
    
    [System.Serializable]
    public class GestureAction
    {
        public string type;
        public string objectName;
        public string methodName;
        public string[] parameters;
        public string gesture;
        public float confidence;
        public double timestamp;
    }
    
    [System.Serializable]
    public class GestureData
    {
        public string type;
        public string gesture;
        public float confidence;
        public double timestamp;
    }
    
    [System.Serializable]
    public class LandmarksData
    {
        public string type;
        public float[][] data;
        public int frame;
        public double timestamp;
    }
}
"""

PLAYER_CONTROLLER_SCRIPT = """
using UnityEngine;

public class PlayerController : MonoBehaviour
{
    [Header("Movement Settings")]
    public float moveSpeed = 5f;
    public float rotationSpeed = 10f;
    
    [Header("Current State")]
    public float currentSpeed = 0f;
    public bool isMoving = false;
    public bool isSliding = false;
    public bool isBlocking = false;
    
    private CharacterController characterController;
    private Animator animator;
    
    void Start()
    {
        characterController = GetComponent<CharacterController>();
        animator = GetComponent<Animator>();
    }
    
    void Update()
    {
        // Handle movement based on current speed
        if (currentSpeed > 0.1f && !isSliding)
        {
            Vector3 moveDirection = transform.forward * currentSpeed * moveSpeed * Time.deltaTime;
            characterController.Move(moveDirection);
            isMoving = true;
        }
        else
        {
            isMoving = false;
        }
        
        // Update animator parameters
        if (animator != null)
        {
            animator.SetFloat("Speed", currentSpeed);
            animator.SetBool("IsMoving", isMoving);
            animator.SetBool("IsSliding", isSliding);
            animator.SetBool("IsBlocking", isBlocking);
        }
    }
    
    // Called from GestureReceiver
    public void SetMovementSpeed(float speed)
    {
        currentSpeed = Mathf.Clamp01(speed);
        Debug.Log($"Movement speed set to: {currentSpeed}");
    }
    
    public void ResetToIdle()
    {
        currentSpeed = 0f;
        isSliding = false;
        isBlocking = false;
        Debug.Log("Reset to idle");
    }
}

// PlayerCombat.cs - Separate script for combat actions
public class PlayerCombat : MonoBehaviour
{
    [Header("Combat Settings")]
    public float punchDamage = 25f;
    public float kickDamage = 35f;
    public float blockDuration = 2f;
    public float blockDamageReduction = 0.7f;
    
    [Header("Current State")]
    public bool isAttacking = false;
    public bool isBlocking = false;
    public float blockTimer = 0f;
    
    private Animator animator;
    private HealthSystem healthSystem;
    
    void Start()
    {
        animator = GetComponent<Animator>();
        healthSystem = GetComponent<HealthSystem>();
    }
    
    void Update()
    {
        // Update block timer
        if (isBlocking)
        {
            blockTimer -= Time.deltaTime;
            if (blockTimer <= 0f)
            {
                EndBlock();
            }
        }
    }
    
    // Called from GestureReceiver
    public void PerformAttack(string attackType)
    {
        if (isAttacking || isBlocking) return;
        
        isAttacking = true;
        
        switch (attackType.ToLower())
        {
            case "left":
                PunchLeft();
                break;
            case "right":
                PunchRight();
                break;
            case "kick":
                Kick();
                break;
            default:
                Debug.LogWarning($"Unknown attack type: {attackType}");
                break;
        }
        
        Debug.Log($"Performed attack: {attackType}");
    }
    
    public void Block(float duration)
    {
        if (isAttacking) return;
        
        isBlocking = true;
        blockTimer = duration;
        
        if (animator != null)
        {
            animator.SetTrigger("Block");
        }
        
        Debug.Log($"Blocking for {duration} seconds");
    }
    
    private void PunchLeft()
    {
        if (animator != null)
        {
            animator.SetTrigger("PunchLeft");
        }
        // In a real game, you would deal damage to enemies here
    }
    
    private void PunchRight()
    {
        if (animator != null)
        {
            animator.SetTrigger("PunchRight");
        }
    }
    
    private void Kick()
    {
        if (animator != null)
        {
            animator.SetTrigger("Kick");
        }
    }
    
    private void EndBlock()
    {
        isBlocking = false;
        blockTimer = 0f;
        
        if (animator != null)
        {
            animator.SetBool("IsBlocking", false);
        }
    }
    
    // Animation event callbacks
    public void OnAttackComplete()
    {
        isAttacking = false;
    }
}

// PlayerMovement.cs - Separate script for movement actions
public class PlayerMovement : MonoBehaviour
{
    [Header("Movement Settings")]
    public float dodgeDistance = 2f;
    public float dodgeSpeed = 10f;
    public float slideDuration = 1.5f;
    public float slideHeightReduction = 0.5f;
    
    [Header("Current State")]
    public bool isDodging = false;
    public bool isSliding = false;
    public float slideTimer = 0f;
    
    private Vector3 originalScale;
    private Vector3 dodgeTarget;
    private CharacterController characterController;
    private Animator animator;
    
    void Start()
    {
        characterController = GetComponent<CharacterController>();
        animator = GetComponent<Animator>();
        originalScale = transform.localScale;
    }
    
    void Update()
    {
        // Handle dodging
        if (isDodging)
        {
            Vector3 moveDirection = (dodgeTarget - transform.position).normalized;
            float distance = Vector3.Distance(transform.position, dodgeTarget);
            
            if (distance > 0.1f)
            {
                characterController.Move(moveDirection * dodgeSpeed * Time.deltaTime);
            }
            else
            {
                isDodging = false;
            }
        }
        
        // Handle sliding
        if (isSliding)
        {
            slideTimer -= Time.deltaTime;
            if (slideTimer <= 0f)
            {
                EndSlide();
            }
        }
    }
    
    // Called from GestureReceiver
    public void Dodge(string direction)
    {
        if (isDodging || isSliding) return;
        
        Vector3 dodgeDirection = Vector3.zero;
        
        switch (direction.ToLower())
        {
            case "left":
                dodgeDirection = -transform.right;
                break;
            case "right":
                dodgeDirection = transform.right;
                break;
            default:
                Debug.LogWarning($"Unknown dodge direction: {direction}");
                return;
        }
        
        dodgeTarget = transform.position + dodgeDirection * dodgeDistance;
        isDodging = true;
        
        if (animator != null)
        {
            animator.SetTrigger($"Dodge{direction}");
        }
        
        Debug.Log($"Dodging {direction}");
    }
    
    public void Slide(float duration)
    {
        if (isDodging || isSliding) return;
        
        isSliding = true;
        slideTimer = duration;
        
        // Reduce height for sliding
        transform.localScale = new Vector3(
            originalScale.x,
            originalScale.y * slideHeightReduction,
            originalScale.z
        );
        
        if (animator != null)
        {
            animator.SetTrigger("Slide");
        }
        
        // Notify PlayerController
        PlayerController playerController = GetComponent<PlayerController>();
        if (playerController != null)
        {
            playerController.isSliding = true;
        }
        
        Debug.Log($"Sliding for {duration} seconds");
    }
    
    private void EndSlide()
    {
        isSliding = false;
        
        // Restore original scale
        transform.localScale = originalScale;
        
        // Notify PlayerController
        PlayerController playerController = GetComponent<PlayerController>();
        if (playerController != null)
        {
            playerController.isSliding = false;
        }
        
        if (animator != null)
        {
            animator.SetBool("IsSliding", false);
        }
    }
}
"""

GESTURE_VISUALIZER_SCRIPT = """
using UnityEngine;
using System.Collections.Generic;

public class GestureVisualizer : MonoBehaviour
{
    [Header("Visualization Settings")]
    public GameObject landmarkPrefab;
    public LineRenderer connectionLinePrefab;
    public float pointScale = 0.1f;
    public Color connectionColor = Color.green;
    
    [Header("Landmark Connections")]
    public List<Vector2Int> poseConnections = new List<Vector2Int>
    {
        // Torso
        new Vector2Int(11, 12),  // Shoulders
        new Vector2Int(11, 23),  // Left shoulder to left hip
        new Vector2Int(12, 24),  // Right shoulder to right hip
        new Vector2Int(23, 24),  // Hips
        
        // Left arm
        new Vector2Int(11, 13),  // Shoulder to elbow
        new Vector2Int(13, 15),  // Elbow to wrist
        
        // Right arm
        new Vector2Int(12, 14),  // Shoulder to elbow
        new Vector2Int(14, 16),  // Elbow to wrist
        
        // Left leg
        new Vector2Int(23, 25),  // Hip to knee
        new Vector2Int(25, 27),  // Knee to ankle
        
        // Right leg
        new Vector2Int(24, 26),  // Hip to knee
        new Vector2Int(26, 28),  // Knee to ankle
    };
    
    private List<GameObject> landmarks = new List<GameObject>();
    private List<LineRenderer> connections = new List<LineRenderer>();
    private Camera mainCamera;
    
    void Start()
    {
        mainCamera = Camera.main;
        
        // Initialize landmarks
        for (int i = 0; i < 33; i++)  // MediaPipe has 33 pose landmarks
        {
            GameObject landmark = Instantiate(landmarkPrefab, transform);
            landmark.SetActive(false);
            landmarks.Add(landmark);
        }
        
        // Initialize connection lines
        foreach (var connection in poseConnections)
        {
            LineRenderer line = Instantiate(connectionLinePrefab, transform);
            line.startColor = connectionColor;
            line.endColor = connectionColor;
            line.positionCount = 2;
            line.gameObject.SetActive(false);
            connections.Add(line);
        }
    }
    
    public void UpdateLandmarks(float[][] landmarkData)
    {
        if (landmarkData == null || landmarkData.Length == 0)
        {
            HideAll();
            return;
        }
        
        // Update landmark positions
        for (int i = 0; i < Mathf.Min(landmarkData.Length, landmarks.Count); i++)
        {
            if (i < landmarkData.Length && landmarkData[i].Length >= 3)
            {
                // Convert normalized coordinates to screen/world space
                float x = landmarkData[i][0];
                float y = landmarkData[i][1];
                float z = landmarkData[i][2];
                
                // Convert to world space (adjust based on your camera setup)
                Vector3 worldPos = mainCamera.ViewportToWorldPoint(new Vector3(x, 1 - y, z * 10));
                
                landmarks[i].transform.position = worldPos;
                landmarks[i].transform.localScale = Vector3.one * pointScale;
                landmarks[i].SetActive(true);
            }
            else
            {
                landmarks[i].SetActive(false);
            }
        }
        
        // Update connection lines
        for (int i = 0; i < Mathf.Min(poseConnections.Count, connections.Count); i++)
        {
            int startIdx = poseConnections[i].x;
            int endIdx = poseConnections[i].y;
            
            if (startIdx < landmarks.Count && endIdx < landmarks.Count &&
                landmarks[startIdx].activeSelf && landmarks[endIdx].activeSelf)
            {
                connections[i].SetPosition(0, landmarks[startIdx].transform.position);
                connections[i].SetPosition(1, landmarks[endIdx].transform.position);
                connections[i].gameObject.SetActive(true);
            }
            else
            {
                connections[i].gameObject.SetActive(false);
            }
        }
    }
    
    void HideAll()
    {
        foreach (var landmark in landmarks)
        {
            landmark.SetActive(false);
        }
        
        foreach (var connection in connections)
        {
            connection.gameObject.SetActive(false);
        }
    }
    
    void OnDestroy()
    {
        // Cleanup
        foreach (var landmark in landmarks)
        {
            Destroy(landmark);
        }
        
        foreach (var connection in connections)
        {
            Destroy(connection);
        }
    }
}

// Simple health system for the player
public class HealthSystem : MonoBehaviour
{
    public float maxHealth = 100f;
    public float currentHealth;
    
    void Start()
    {
        currentHealth = maxHealth;
    }
    
    public void TakeDamage(float damage)
    {
        currentHealth -= damage;
        currentHealth = Mathf.Clamp(currentHealth, 0, maxHealth);
        
        Debug.Log($"Took {damage} damage. Health: {currentHealth}/{maxHealth}");
        
        if (currentHealth <= 0)
        {
            Die();
        }
    }
    
    public void Heal(float amount)
    {
        currentHealth += amount;
        currentHealth = Mathf.Clamp(currentHealth, 0, maxHealth);
        
        Debug.Log($"Healed {amount}. Health: {currentHealth}/{maxHealth}");
    }
    
    void Die()
    {
        Debug.Log("Player died!");
        // Add death logic here
    }
}
"""

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
        try:
            if not step_func():
                print(f"⚠️  Step completed with warnings: {step_name}")
        except Exception as e:
            print(f"✗ Failed at step '{step_name}': {e}")
            print("Try running with administrator/sudo privileges or in a virtual environment.")
            sys.exit(1)
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Collect gesture data:     python main.py collect")
    print("2. Train the model:          python main.py train")
    print("3. Test recognition:         python main.py demo")
    print("4. Connect to Unity:         python unity_integration/unity_example.py")
    print("\nTo use Unity integration:")
    print("1. Import C# scripts from unity_integration/ folder")
    print("2. Attach GestureReceiver to a GameObject")
    print("3. Set up PlayerController, PlayerCombat, and PlayerMovement components")
    print("4. Run the Unity game, then run the Python client")

if __name__ == "__main__":
    main()