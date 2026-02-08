# unity_integration.py (FIXED VERSION)
"""
Tools for integrating the gesture recognition system with Unity.
Python acts as CLIENT, Unity acts as SERVER.
"""

import json
import numpy as np
import socket
import time
from typing import Dict, Any, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnityBridge:
    """
    Bridge between Python gesture recognition and Unity game engine.
    Python acts as CLIENT that connects to Unity SERVER.
    """
    
    def __init__(self, host='127.0.0.1', port=65432, max_reconnect_attempts=5):
        self.host = host
        self.port = port
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = 2  # seconds
        
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.running = False
        
        # Action cooldowns (milliseconds)
        self.cooldowns = {
            'attack': 500,  # 500ms between attacks
            'slide': 1000,  # 1 second between slides
            'block': 300,   # 300ms between blocks
        }
        
        self.last_action_times = {}
        self.reconnect_attempts = 0
        
    def connect(self, auto_reconnect=True):
        """Connect to Unity socket server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            logger.info(f"Attempting to connect to Unity at {self.host}:{self.port}")
            self.socket.connect((self.host, self.port))
            
            # Set a timeout for socket operations
            self.socket.settimeout(5.0)
            
            # Test connection
            test_msg = json.dumps({"type": "handshake", "message": "Python client connected"}).encode('utf-8')
            self.socket.send(test_msg + b'\n')
            
            self.connected = True
            self.running = True
            self.reconnect_attempts = 0
            
            logger.info(f"Connected to Unity server at {self.host}:{self.port}")
            return True
            
        except ConnectionRefusedError:
            logger.warning(f"Connection refused. Make sure Unity server is running on {self.host}:{self.port}")
            
            if auto_reconnect and self.reconnect_attempts < self.max_reconnect_attempts:
                self.reconnect_attempts += 1
                logger.info(f"Reconnect attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} in {self.reconnect_delay}s...")
                time.sleep(self.reconnect_delay)
                return self.connect(auto_reconnect)
                
        except socket.timeout:
            logger.error("Connection timeout")
        except Exception as e:
            logger.error(f"Connection error: {e}")
        
        self.connected = False
        return False
    
    def send_message(self, message_dict: Dict[str, Any]) -> bool:
        """Send JSON message to Unity."""
        if not self.connected or not self.socket:
            # Try to reconnect if not connected
            if not self.connect(auto_reconnect=False):
                return False
        
        # Send JSON message
        try:
            message = json.dumps(message_dict).encode('utf-8')
            self.socket.send(message + b'\n')
            return True
        except BrokenPipeError:
            logger.warning("Connection lost. Attempting to reconnect...")
            self.connected = False
            return False
        except socket.timeout:
            logger.warning("Send timeout")
            return False
        except Exception as e:
            logger.error(f"Send error: {e}")
            return False
    
    def send_action(self, action_data: Dict[str, Any]) -> bool:
        """Send action to Unity."""
        # Check cooldown if action has a name
        action_name = action_data.get('action_name') or action_data.get('gesture')
        current_time = time.time() * 1000  # Convert to milliseconds
        
        if action_name and action_name in self.cooldowns:
            last_time = self.last_action_times.get(action_name, 0)
            if current_time - last_time < self.cooldowns[action_name]:
                return False  # Still in cooldown
        
        # Update last action time
        if action_name:
            self.last_action_times[action_name] = current_time
        
        # Send the message
        return self.send_message(action_data)
    
    def send_gesture_detected(self, gesture_name: str, confidence: float, action_data: Dict[str, Any]):
        """Send gesture detection event to Unity."""
        message = {
            'type': 'gesture',
            'gesture': gesture_name,
            'confidence': confidence,
            'action': action_data,
            'timestamp': time.time()
        }
        return self.send_message(message)
    
    def send_landmarks(self, landmarks: np.ndarray, frame_count: int = 0):
        """Send raw landmarks for debugging/visualization."""
        if not self.connected:
            return False
        
        # Limit landmark sending rate (e.g., every 10 frames)
        if frame_count % 10 != 0:
            return False
        
        try:
            # Convert to list and send
            landmarks_list = landmarks.tolist()
            data = {
                'type': 'landmarks',
                'data': landmarks_list,
                'frame': frame_count,
                'timestamp': time.time()
            }
            
            return self.send_message(data)
        except Exception as e:
            logger.error(f"Error sending landmarks: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Unity."""
        self.running = False
        self.connected = False
        
        try:
            # Send disconnect message
            if self.socket:
                disconnect_msg = json.dumps({"type": "disconnect", "message": "Python client disconnecting"}).encode('utf-8')
                self.socket.send(disconnect_msg + b'\n')
        except:
            pass
        
        # Close socket
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        
        logger.info("Disconnected from Unity server")
    
    def is_connected(self) -> bool:
        """Check if connected to Unity."""
        if not self.connected or not self.socket:
            return False
        
        # Optional: Send a ping to verify connection
        try:
            self.socket.settimeout(1.0)
            ping_msg = json.dumps({"type": "ping"}).encode('utf-8')
            self.socket.send(ping_msg + b'\n')
            # If we can send without error, we're connected
            return True
        except:
            self.connected = False
            return False

class UnityActionMapper:
    """Maps gesture actions to Unity game commands."""
    
    # Unity GameObject names and methods - MATCHING C# EXPECTATIONS
    UNITY_MAPPING = {
        'forward_movement': {
            'objectName': 'PlayerController',  # Changed from 'object' to 'objectName'
            'methodName': 'SetMovementSpeed',  # Changed from 'method' to 'methodName'
            'param_type': 'float',
            'default_param': 1.0
        },
        'attack': {
            'objectName': 'PlayerCombat',
            'methodName': 'PerformAttack',
            'param_type': 'string',  # "left", "right", or "kick"
            'param_map': {
                'punch_left': 'left',
                'punch_right': 'right',
                'kick': 'kick'
            }
        },
        'dodge': {
            'objectName': 'PlayerMovement',
            'methodName': 'Dodge',
            'param_type': 'string',  # "left" or "right"
            'param_map': {
                'lean_left': 'left',
                'lean_right': 'right'
            }
        },
        'slide': {
            'objectName': 'PlayerMovement',
            'methodName': 'Slide',
            'param_type': 'float',  # duration
            'default_param': 1.5
        },
        'defense': {
            'objectName': 'PlayerCombat',
            'methodName': 'Block',
            'param_type': 'float',  # duration
            'default_param': 2.0
        },
        'no_action': {
            'objectName': 'PlayerController',
            'methodName': 'ResetToIdle',
            'param_type': 'none'
        }
    }
    
    @staticmethod
    def format_for_unity(gesture_name: str, action_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Format action for Unity consumption - MATCHING C# STRUCTURE."""
        if action_name not in UnityActionMapper.UNITY_MAPPING:
            logger.error(f"Action {action_name} not found in UNITY_MAPPING")
            return None
        
        mapping = UnityActionMapper.UNITY_MAPPING[action_name]
        
        # Prepare parameters array
        parameters = []
        
        if mapping['param_type'] == 'float':
            # Get float parameter from params or use default
            param_value = next((float(v) for k, v in params.items() 
                              if isinstance(v, (int, float))), mapping.get('default_param', 0.0))
            parameters.append(param_value)
            
        elif mapping['param_type'] == 'string':
            # Map gesture to specific parameter
            if 'param_map' in mapping and gesture_name in mapping['param_map']:
                parameters.append(mapping['param_map'][gesture_name])
            elif 'direction' in params:
                parameters.append(params['direction'])
            elif 'side' in params:
                parameters.append(params['side'])
            elif 'type' in params:
                parameters.append(params['type'])
            else:
                parameters.append(gesture_name)
        
        elif mapping['param_type'] == 'none':
            # No parameters needed
            pass
        
        # CRITICAL: Use field names that match Unity C# expectations
        unity_message = {
            'type': 'action',
            'objectName': mapping['objectName'],  # Must match C# field name
            'methodName': mapping['methodName'],  # Must match C# field name
            'parameters': parameters,
            'gesture': gesture_name,
            'action_name': action_name,  # Added for cooldown tracking
            'timestamp': time.time()
        }
        
        return unity_message
    
    @staticmethod
    def get_mapping_info() -> Dict[str, Any]:
        """Get information about Unity mappings for debugging."""
        return {
            'available_actions': list(UnityActionMapper.UNITY_MAPPING.keys()),
            'mappings': UnityActionMapper.UNITY_MAPPING
        }

class GestureToUnity:
    """Main class for sending gestures to Unity."""
    
    def __init__(self, host='127.0.0.1', port=65432):
        self.bridge = UnityBridge(host, port)
        self.action_mapper = UnityActionMapper()
        self.last_sent_gesture = None
        self.last_sent_time = 0
        self.min_time_between_same_gesture = 0.3  # 300ms
        
    def connect(self):
        """Connect to Unity."""
        return self.bridge.connect()
    
    def send_gesture(self, gesture_name: str, confidence: float = 1.0, 
                    force_send: bool = False) -> bool:
        """Send gesture to Unity."""
        from gesture_action_contract import GestureActionContract
        
        try:
            # Get action from contract
            action_type, params = GestureActionContract.get_action_for_gesture(gesture_name)
            
            # Format for Unity (with corrected field names)
            unity_action = self.action_mapper.format_for_unity(gesture_name, action_type, params)
            
            if not unity_action:
                logger.error(f"Could not format action for Unity: {gesture_name}")
                return False
            
            # Add confidence
            unity_action['confidence'] = confidence
            
            # Optional: Prevent sending same gesture too frequently
            current_time = time.time()
            if not force_send and gesture_name == self.last_sent_gesture:
                time_since_last = current_time - self.last_sent_time
                if time_since_last < self.min_time_between_same_gesture:
                    logger.debug(f"Skipping {gesture_name} - sent {time_since_last:.2f}s ago")
                    return False
            
            # Send to Unity
            success = self.bridge.send_action(unity_action)
            
            if success:
                logger.debug(f"Sent to Unity: {gesture_name} -> {action_type}")
                self.last_sent_gesture = gesture_name
                self.last_sent_time = current_time
            else:
                logger.warning(f"Failed to send to Unity: {gesture_name}")
            
            return success
                
        except Exception as e:
            logger.error(f"Error sending gesture {gesture_name}: {e}")
            return False
    
    def send_gesture_with_landmarks(self, gesture_name: str, landmarks: np.ndarray, 
                                   confidence: float = 1.0, frame_count: int = 0) -> bool:
        """Send gesture and optionally landmarks to Unity."""
        # Send gesture
        gesture_sent = self.send_gesture(gesture_name, confidence)
        
        # Optionally send landmarks for visualization
        if gesture_name != 'idle':  # Don't spam landmarks for idle
            self.bridge.send_landmarks(landmarks, frame_count)
        
        return gesture_sent
    
    def send_test_message(self) -> bool:
        """Send a test message to verify Unity connection."""
        test_message = {
            'type': 'test',
            'message': 'Python test message',
            'timestamp': time.time()
        }
        return self.bridge.send_message(test_message)
    
    def send_direct_message(self, message_dict: Dict[str, Any]) -> bool:
        """Send a custom message directly to Unity."""
        return self.bridge.send_message(message_dict)
    
    def disconnect(self):
        """Disconnect from Unity."""
        self.bridge.disconnect()
    
    def is_connected(self) -> bool:
        """Check connection status."""
        return self.bridge.is_connected()
    
    def get_mapping_info(self) -> Dict[str, Any]:
        """Get information about Unity mappings."""
        return self.action_mapper.get_mapping_info()


# ============================================================================
# ENHANCED UNITY C# RECEIVER TEMPLATE (FOR REFERENCE)
# ============================================================================

UNITY_RECEIVER_SCRIPT_ENHANCED = '''
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
    public bool logParsingErrors = true;
    
    // Message queue for thread-safe communication
    private Queue<string> messageQueue = new Queue<string>();
    
    // Client connection
    private TcpClient connectedClient;
    private NetworkStream clientStream;
    
    void Start()
    {
        StartServer();
    }
    
    void StartServer()
    {
        serverThread = new Thread(new ThreadStart(ListenForConnections));
        serverThread.IsBackground = true;
        serverThread.Start();
        Debug.Log("Gesture receiver server starting on " + host + ":" + port);
    }
    
    void ListenForConnections()
    {
        try
        {
            IPAddress ipAddress = IPAddress.Parse(host);
            server = new TcpListener(ipAddress, port);
            server.Start();
            
            Debug.Log("Server started. Waiting for Python client...");
            
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
            Debug.LogError("SocketException: " + e);
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
                    }
                }
                
                messageBuilder.Clear();
                messageBuilder.Append(allData);
            }
        }
        catch (Exception e)
        {
            Debug.LogError("Error handling client: " + e);
        }
        finally
        {
            connectedClient?.Close();
            clientStream?.Close();
            Debug.Log("Client connection closed.");
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
                Debug.Log("Received: " + jsonMessage);
            }
            
            // Parse as generic message first to get type
            var baseMessage = JsonUtility.FromJson<BaseMessage>(jsonMessage);
            
            switch (baseMessage.type)
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
                    Debug.Log("Python client connected: " + jsonMessage);
                    break;
                    
                case "test":
                    Debug.Log("Test message from Python: " + jsonMessage);
                    break;
                    
                case "ping":
                    // Respond to ping
                    SendPong();
                    break;
                    
                case "disconnect":
                    Debug.Log("Python client requested disconnect");
                    break;
                    
                default:
                    Debug.LogWarning("Unknown message type: " + baseMessage.type);
                    break;
            }
        }
        catch (Exception e)
        {
            if (logParsingErrors)
            {
                Debug.LogError("Error processing message: " + e.Message + "\\nMessage: " + jsonMessage);
            }
        }
    }
    
    void ProcessActionMessage(string jsonMessage)
    {
        // CRITICAL: Use the correct class name that matches Python's JSON
        var action = JsonUtility.FromJson<GestureAction>(jsonMessage);
        
        if (action == null)
        {
            Debug.LogError("Failed to parse action message");
            return;
        }
        
        if (string.IsNullOrEmpty(action.objectName))
        {
            Debug.LogError("Action message missing objectName");
            return;
        }
        
        if (string.IsNullOrEmpty(action.methodName))
        {
            Debug.LogError("Action message missing methodName");
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
            default:
                Debug.LogWarning("Unknown target object: " + action.objectName);
                return;
        }
        
        if (targetObject != null)
        {
            // Convert parameters to appropriate types
            object[] processedParams = new object[action.parameters.Length];
            
            for (int i = 0; i < action.parameters.Length; i++)
            {
                // Try to parse as float first
                if (float.TryParse(action.parameters[i], out float floatValue))
                {
                    processedParams[i] = floatValue;
                }
                else
                {
                    // Keep as string
                    processedParams[i] = action.parameters[i];
                }
            }
            
            // Send message to GameObject
            if (processedParams.Length > 0)
            {
                targetObject.SendMessage(action.methodName, processedParams, 
                                       SendMessageOptions.DontRequireReceiver);
            }
            else
            {
                targetObject.SendMessage(action.methodName, 
                                       SendMessageOptions.DontRequireReceiver);
            }
            
            Debug.Log("Executed: " + action.objectName + "." + action.methodName + 
                     "(" + string.Join(", ", action.parameters) + ")");
        }
        else
        {
            Debug.LogWarning("Target object not found: " + action.objectName);
        }
    }
    
    void ProcessGestureMessage(string jsonMessage)
    {
        var gesture = JsonUtility.FromJson<GestureData>(jsonMessage);
        Debug.Log("Gesture detected: " + gesture.gesture + " (confidence: " + (gesture.confidence * 100).ToString("F0") + "%)");
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
            var pong = new PongMessage
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
    // MUST MATCH PYTHON'S JSON STRUCTURE
    
    [System.Serializable]
    private class BaseMessage
    {
        public string type;
    }
    
    [System.Serializable]
    public class GestureAction
    {
        public string type;
        public string objectName;      // MUST match Python field name
        public string methodName;      // MUST match Python field name
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
    
    [System.Serializable]
    private class PongMessage
    {
        public string type;
        public string timestamp;
    }
}
'''

# ============================================================================
# TEST FUNCTION WITH JSON FIELD VALIDATION
# ============================================================================

def test_unity_integration():
    """Test Unity integration with JSON field validation."""
    print("Testing Unity Integration with JSON Field Fix...")
    
    # Initialize
    gesture_sender = GestureToUnity()
    
    # Test JSON generation
    print("\n🧪 Testing JSON field names...")
    
    from gesture_action_contract import GestureActionContract
    
    test_gestures = ['punch_left', 'run', 'block', 'lean_left']
    
    for gesture in test_gestures:
        try:
            action_type, params = GestureActionContract.get_action_for_gesture(gesture)
            unity_action = gesture_sender.action_mapper.format_for_unity(gesture, action_type, params)
            
            if unity_action:
                print(f"\n{gesture}:")
                print(f"  JSON keys: {list(unity_action.keys())}")
                print(f"  objectName: {unity_action.get('objectName')}")
                print(f"  methodName: {unity_action.get('methodName')}")
                print(f"  parameters: {unity_action.get('parameters')}")
                
                # Validate field names
                required_fields = ['objectName', 'methodName', 'parameters']
                missing_fields = [field for field in required_fields if field not in unity_action]
                
                if missing_fields:
                    print(f"  ❌ Missing fields: {missing_fields}")
                else:
                    print(f"  ✅ All required fields present")
            else:
                print(f"\n{gesture}: ❌ Could not format action")
                
        except Exception as e:
            print(f"\n{gesture}: ❌ Error: {e}")
    
    # Test connection
    print("\n🧪 Testing connection...")
    if gesture_sender.connect():
        print("✅ Connected to Unity server")
        
        # Send test messages
        print("\n🧪 Sending test messages...")
        
        # Send direct test message
        test_msg = {
            'type': 'test',
            'message': 'Python integration test',
            'timestamp': time.time()
        }
        if gesture_sender.send_direct_message(test_msg):
            print("✅ Test message sent")
        else:
            print("❌ Failed to send test message")
        
        # Send a gesture
        if gesture_sender.send_gesture('idle', confidence=0.95):
            print("✅ Gesture message sent")
        else:
            print("❌ Failed to send gesture message")
        
        # Get mapping info
        mapping_info = gesture_sender.get_mapping_info()
        print(f"\n📋 Available Unity mappings: {len(mapping_info['available_actions'])} actions")
        
        # Disconnect
        gesture_sender.disconnect()
        print("✅ Disconnected from Unity")
    else:
        print("❌ Failed to connect to Unity")
        print("  Make sure Unity is running with GestureReceiver component")
    
    print("\n✅ Unity integration test complete!")

def generate_unity_test_script():
    """Generate a Unity C# script with the correct field names."""
    script_content = f'''// GestureReceiver_Fixed.cs
// Generated from Python - USE THIS VERSION
// Fixes JSON field name mismatch: objectName/methodName instead of object/method

{UNITY_RECEIVER_SCRIPT_ENHANCED}
'''
    
    output_path = "GestureReceiver_Fixed.cs"
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Fixed Unity C# script generated: {output_path}")
    print("  Replace your existing GestureReceiver.cs with this file")
    print("  Make sure the C# class names match the JSON structure exactly")

if __name__ == "__main__":
    # Run tests
    test_unity_integration()
    
    # Generate fixed Unity script
    generate_unity_test_script()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("The key fix was changing JSON field names:")
    print("  Python was sending: 'object' and 'method'")
    print("  Unity expects: 'objectName' and 'methodName'")
    print("\nNow Python sends correctly formatted JSON:")
    print("  {{\"objectName\": \"...\", \"methodName\": \"...\", \"parameters\": [...]}}")
    print("\n✅ Fixed JSON field name mismatch!")
    print("=" * 60)