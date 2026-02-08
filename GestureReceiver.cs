// GestureReceiver.cs
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
                
                while ((newlineIndex = allData.IndexOf('\n')) >= 0)
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
            byte[] ackBytes = Encoding.UTF8.GetBytes(ackJson + "\n");
            
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
            Debug.LogError($"Error processing message: {e.Message}\nMessage: {jsonMessage}");
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
            byte[] pongBytes = Encoding.UTF8.GetBytes(pongJson + "\n");
            
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
    
    // Helper method to manually send test messages from Unity
    public void SendTestMessage(string message)
    {
        lock (lockObject)
        {
            messageQueue.Enqueue(message);
        }
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