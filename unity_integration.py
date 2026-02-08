# unity_integration.py
"""
Tools for integrating the gesture recognition system with Unity.
"""

import json
import numpy as np
import socket
import threading
import time
from typing import Dict, Any

class UnityBridge:
    """
    Bridge between Python gesture recognition and Unity game engine.
    Sends gesture actions over network socket.
    """
    
    def __init__(self, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port
        self.socket = None
        self.client = None
        self.running = False
        
        # Action cooldowns (milliseconds)
        self.cooldowns = {
            'attack': 500,  # 500ms between attacks
            'slide': 1000,  # 1 second between slides
            'block': 300,   # 300ms between blocks
        }
        
        self.last_action_times = {}
    
    def connect(self):
        """Connect to Unity socket server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            print(f"Unity Bridge listening on {self.host}:{self.port}")
            
            self.client, address = self.socket.accept()
            print(f"Connected to Unity at {address}")
            
            self.running = True
            return True
            
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def send_action(self, action_data: Dict[str, Any]):
        """Send action to Unity."""
        if not self.client or not self.running:
            return False
        
        # Check cooldown
        action_name = action_data.get('name')
        current_time = time.time() * 1000  # Convert to milliseconds
        
        if action_name in self.cooldowns:
            last_time = self.last_action_times.get(action_name, 0)
            if current_time - last_time < self.cooldowns[action_name]:
                return False  # Still in cooldown
        
        # Update last action time
        self.last_action_times[action_name] = current_time
        
        # Send JSON message
        try:
            message = json.dumps(action_data).encode('utf-8')
            self.client.send(message + b'\n')
            return True
        except Exception as e:
            print(f"Send error: {e}")
            return False
    
    def send_landmarks(self, landmarks: np.ndarray):
        """Send raw landmarks for debugging/visualization."""
        if not self.client or not self.running:
            return
        
        # Convert to list and send
        landmarks_list = landmarks.tolist()
        data = {
            'type': 'landmarks',
            'data': landmarks_list,
            'timestamp': time.time()
        }
        
        try:
            message = json.dumps(data).encode('utf-8')
            self.client.send(message + b'\n')
        except:
            pass
    
    def disconnect(self):
        """Disconnect from Unity."""
        self.running = False
        if self.client:
            self.client.close()
        if self.socket:
            self.socket.close()
        print("Disconnected from Unity")

class UnityActionMapper:
    """Maps gesture actions to Unity game commands."""
    
    # Unity GameObject names and methods
    UNITY_MAPPING = {
        'forward_movement': {
            'object': 'PlayerController',
            'method': 'SetMovementSpeed',
            'param_type': 'float'
        },
        'attack': {
            'object': 'PlayerCombat',
            'method': 'PerformAttack',
            'param_type': 'string'  # "left", "right", or "kick"
        },
        'dodge': {
            'object': 'PlayerMovement',
            'method': 'Dodge',
            'param_type': 'string'  # "left" or "right"
        },
        'slide': {
            'object': 'PlayerMovement',
            'method': 'Slide',
            'param_type': 'float'  # duration
        },
        'defense': {
            'object': 'PlayerCombat',
            'method': 'Block',
            'param_type': 'float'  # duration
        }
    }
    
    @staticmethod
    def format_for_unity(action_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Format action for Unity consumption."""
        if action_name not in UnityActionMapper.UNITY_MAPPING:
            return None
        
        mapping = UnityActionMapper.UNITY_MAPPING[action_name]
        
        # Extract parameter based on type
        param_value = None
        if mapping['param_type'] == 'float':
            # Get first float parameter
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    param_value = float(value)
                    break
        elif mapping['param_type'] == 'string':
            # Get direction or type
            if 'direction' in params:
                param_value = params['direction']
            elif 'side' in params:
                param_value = params['side']
            elif 'type' in params:
                param_value = params['type']
        
        unity_message = {
            'object': mapping['object'],
            'method': mapping['method'],
            'parameters': [param_value] if param_value is not None else []
        }
        
        return unity_message

# Unity C# script template for receiving actions
UNITY_RECEIVER_SCRIPT = """
using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;
using System.Text;

public class GestureReceiver : MonoBehaviour
{
    private TcpListener listener;
    private TcpClient client;
    private Thread listenerThread;
    private bool isRunning = true;
    
    public string host = "127.0.0.1";
    public int port = 65432;
    
    // Game objects that handle actions
    public GameObject playerController;
    public GameObject playerCombat;
    public GameObject playerMovement;
    
    private Queue<string> messageQueue = new Queue<string>();
    private object queueLock = new object();
    
    void Start()
    {
        StartServer();
    }
    
    void StartServer()
    {
        listenerThread = new Thread(new ThreadStart(ListenForMessages));
        listenerThread.IsBackground = true;
        listenerThread.Start();
        Debug.Log("Gesture receiver started on port " + port);
    }
    
    void ListenForMessages()
    {
        try
        {
            IPAddress ipAddress = IPAddress.Parse(host);
            listener = new TcpListener(ipAddress, port);
            listener.Start();
            
            byte[] bytes = new byte[1024];
            
            while (isRunning)
            {
                using (client = listener.AcceptTcpClient())
                using (NetworkStream stream = client.GetStream())
                {
                    int length;
                    while ((length = stream.Read(bytes, 0, bytes.Length)) != 0)
                    {
                        string data = Encoding.UTF8.GetString(bytes, 0, length);
                        
                        lock (queueLock)
                        {
                            messageQueue.Enqueue(data);
                        }
                    }
                }
            }
        }
        catch (SocketException e)
        {
            Debug.Log("SocketException: " + e);
        }
    }
    
    void Update()
    {
        // Process messages on main thread
        while (messageQueue.Count > 0)
        {
            string message;
            lock (queueLock)
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
            GestureAction action = JsonUtility.FromJson<GestureAction>(jsonMessage);
            
            switch (action.objectName)
            {
                case "PlayerController":
                    if (playerController != null)
                        playerController.SendMessage(action.methodName, action.parameters);
                    break;
                    
                case "PlayerCombat":
                    if (playerCombat != null)
                        playerCombat.SendMessage(action.methodName, action.parameters);
                    break;
                    
                case "PlayerMovement":
                    if (playerMovement != null)
                        playerMovement.SendMessage(action.methodName, action.parameters);
                    break;
                    
                default:
                    Debug.LogWarning("Unknown object: " + action.objectName);
                    break;
            }
        }
        catch (Exception e)
        {
            Debug.LogError("Error processing message: " + e.Message);
        }
    }
    
    void OnDestroy()
    {
        isRunning = false;
        listener?.Stop();
        listenerThread?.Join();
    }
    
    [System.Serializable]
    public class GestureAction
    {
        public string objectName;
        public string methodName;
        public string[] parameters;
    }
}
"""