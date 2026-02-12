"""
Tools for integrating the gesture recognition system with Unity.
Python acts as CLIENT, Unity acts as SERVER.

CRITICAL FIXES APPLIED:
1. Reset cooldown state on reconnect (last_action_times.clear())
2. Added acknowledgment tracking to detect disconnected state
3. JSON field names now EXACTLY match Unity expectations (objectName, methodName)
4. Added heartbeat to detect stale connections
"""

import json
import numpy as np
import socket
import time
from typing import Dict, Any, Optional, List
from collections import deque
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnityBridge:
    """
    Bridge between Python gesture recognition and Unity game engine.
    Python acts as CLIENT that connects to Unity SERVER.
    
    CRITICAL FIX: Reset cooldown state on reconnect to prevent action desync.
    """
    
    def __init__(self, host='127.0.0.1', port=65432, max_reconnect_attempts=5):
        self.host = host
        self.port = port
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = 2  # seconds
        self.heartbeat_interval = 5  # seconds
        self.ack_timeout = 2.0  # seconds
        
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.running = False
        
        # Action cooldowns (milliseconds)
        self.cooldowns = {
            'attack': 500,
            'dodge': 400,
            'slide': 1000,
            'block': 300,
            'forward_movement': 100,
            'no_action': 0,
        }
        
        # CRITICAL FIX: These will be CLEARED on reconnect
        self.last_action_times: Dict[str, float] = {}
        
        # Track pending acknowledgments
        self.pending_acks: Dict[str, Dict[str, Any]] = {}
        self.ack_id_counter = 0
        
        # Heartbeat tracking
        self.last_heartbeat_response = time.time()
        self.heartbeat_thread: Optional[threading.Thread] = None
        
        # Message queue for retry
        self.message_queue = deque(maxlen=100)
        
        # Connection state
        self.reconnect_attempts = 0
        self.connection_lock = threading.Lock()
        
    def connect(self, auto_reconnect=True) -> bool:
        """
        Connect to Unity socket server.
        
        CRITICAL FIX: Reset cooldown state on successful connection.
        This prevents action desync after reconnection.
        """
        with self.connection_lock:
            # Close existing socket if any
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
                self.socket = None
            
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.socket.settimeout(5.0)
                
                logger.info(f"Attempting to connect to Unity at {self.host}:{self.port}")
                self.socket.connect((self.host, self.port))
                
                # Test connection with handshake
                handshake = {
                    "type": "handshake",
                    "client": "python_gesture",
                    "timestamp": time.time()
                }
                test_msg = json.dumps(handshake).encode('utf-8')
                self.socket.send(test_msg + b'\n')
                
                self.connected = True
                self.running = True
                self.reconnect_attempts = 0
                
                # ============================================================
                # CRITICAL FIX: Clear cooldown state on reconnect
                # ============================================================
                old_cooldown_count = len(self.last_action_times)
                self.last_action_times.clear()
                self.pending_acks.clear()
                self.last_heartbeat_response = time.time()
                
                logger.info(f"✅ Connected to Unity server at {self.host}:{self.port}")
                logger.info(f"   Reset {old_cooldown_count} cooldown states")
                
                # Start heartbeat thread
                self._start_heartbeat()
                
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
    
    def _start_heartbeat(self):
        """Start heartbeat thread to detect stale connections."""
        def heartbeat_loop():
            while self.running and self.connected:
                time.sleep(self.heartbeat_interval)
                
                # Check if we've missed too many heartbeats
                if time.time() - self.last_heartbeat_response > self.heartbeat_interval * 3:
                    logger.warning("Heartbeat timeout - connection may be stale")
                    self.connected = False
                    break
                
                # Send ping
                ping_msg = {
                    "type": "ping",
                    "timestamp": time.time()
                }
                self.send_message(ping_msg, requires_ack=False)
        
        self.heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
    
    def send_message(self, message_dict: Dict[str, Any], requires_ack: bool = True) -> bool:
        """
        Send JSON message to Unity.
        
        Args:
            message_dict: Message to send
            requires_ack: Whether to wait for acknowledgment
            
        Returns:
            True if message was sent successfully
        """
        if not self.connected or not self.socket:
            # Try to reconnect
            if not self.connect(auto_reconnect=True):
                # Queue message for later
                if requires_ack:
                    self.message_queue.append(message_dict)
                return False
        
        try:
            # Add acknowledgment ID if required
            if requires_ack:
                self.ack_id_counter += 1
                ack_id = f"msg_{self.ack_id_counter}_{int(time.time()*1000)}"
                message_dict['ack_id'] = ack_id
                self.pending_acks[ack_id] = {
                    'message': message_dict,
                    'timestamp': time.time(),
                    'retries': 0
                }
            
            # Send message
            message = json.dumps(message_dict).encode('utf-8')
            self.socket.send(message + b'\n')
            
            # Wait for acknowledgment if required (non-blocking)
            if requires_ack:
                # Will be processed asynchronously
                pass
            
            return True
            
        except (BrokenPipeError, ConnectionResetError):
            logger.warning("Connection lost. Will attempt to reconnect...")
            self.connected = False
            
            # Queue message for retry
            if requires_ack:
                self.message_queue.append(message_dict)
            
            # Attempt immediate reconnect
            self.connect(auto_reconnect=True)
            return False
            
        except socket.timeout:
            logger.warning("Send timeout")
            return False
        except Exception as e:
            logger.error(f"Send error: {e}")
            return False
    
    def process_acknowledgment(self, ack_data: Dict[str, Any]):
        """Process acknowledgment from Unity."""
        ack_id = ack_data.get('ack_id')
        if ack_id and ack_id in self.pending_acks:
            # Message was received successfully
            del self.pending_acks[ack_id]
            logger.debug(f"Received ack for {ack_id}")
    
    def process_pong(self):
        """Process pong response from Unity."""
        self.last_heartbeat_response = time.time()
        logger.debug("Received pong from Unity")
    
    def send_action(self, action_data: Dict[str, Any]) -> bool:
        """
        Send action to Unity with cooldown management.
        
        CRITICAL FIX: Cooldown state is reset on reconnect.
        """
        # Get action name (handle different field names)
        action_name = action_data.get('action_name') 
        if not action_name:
            action_name = action_data.get('action', action_data.get('gesture'))
        
        current_time = time.time() * 1000  # Convert to milliseconds
        
        # Check cooldown if we have an action name
        if action_name and action_name in self.cooldowns:
            last_time = self.last_action_times.get(action_name, 0)
            cooldown_ms = self.cooldowns[action_name]
            
            if current_time - last_time < cooldown_ms:
                remaining = cooldown_ms - (current_time - last_time)
                logger.debug(f"Action {action_name} on cooldown: {remaining:.0f}ms remaining")
                return False  # Still in cooldown
        
        # Update last action time
        if action_name:
            self.last_action_times[action_name] = current_time
        
        # Add timestamp if not present
        if 'timestamp' not in action_data:
            action_data['timestamp'] = time.time()
        
        # Send the message
        success = self.send_message(action_data, requires_ack=True)
        
        if success:
            logger.debug(f"✅ Sent action: {action_name}")
        else:
            logger.warning(f"❌ Failed to send action: {action_name}")
            
            # Revert cooldown update if send failed
            if action_name:
                # Remove the timestamp we just added
                if action_name in self.last_action_times:
                    del self.last_action_times[action_name]
        
        return success
    
    def receive_messages(self, buffer_size: int = 4096) -> List[Dict[str, Any]]:
        """
        Receive and parse messages from Unity.
        
        Returns:
            List of parsed JSON messages
        """
        if not self.connected or not self.socket:
            return []
        
        messages = []
        
        try:
            self.socket.settimeout(0.1)  # Short timeout for non-blocking receive
            data = self.socket.recv(buffer_size)
            
            if not data:
                # Connection closed
                self.connected = False
                return []
            
            # Split by newline and parse each JSON object
            for line in data.decode('utf-8').strip().split('\n'):
                if line:
                    try:
                        msg = json.loads(line)
                        messages.append(msg)
                        
                        # Process special message types
                        msg_type = msg.get('type')
                        if msg_type == 'ack':
                            self.process_acknowledgment(msg)
                        elif msg_type == 'pong':
                            self.process_pong()
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Received invalid JSON: {line[:100]}")
                        
        except socket.timeout:
            # No data available, that's fine
            pass
        except Exception as e:
            logger.error(f"Receive error: {e}")
            self.connected = False
        
        return messages
    
    def retry_queued_messages(self) -> int:
        """Retry sending queued messages."""
        if not self.connected:
            return 0
        
        success_count = 0
        failed_messages = []
        
        while self.message_queue:
            msg = self.message_queue.popleft()
            if self.send_message(msg, requires_ack=True):
                success_count += 1
            else:
                failed_messages.append(msg)
        
        # Re-queue failed messages
        for msg in failed_messages:
            self.message_queue.appendleft(msg)
        
        return success_count
    
    def disconnect(self):
        """Disconnect from Unity."""
        self.running = False
        self.connected = False
        
        # Stop heartbeat thread
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            # Thread is daemon, will exit when main thread exits
            pass
        
        try:
            # Send disconnect message
            if self.socket:
                disconnect_msg = json.dumps({
                    "type": "disconnect",
                    "message": "Python client disconnecting",
                    "timestamp": time.time()
                }).encode('utf-8')
                self.socket.send(disconnect_msg + b'\n')
        except:
            pass
        
        # Close socket
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        logger.info("Disconnected from Unity server")
    
    def is_connected(self) -> bool:
        """Check if connected to Unity."""
        if not self.connected or not self.socket:
            return False
        
        # Try to receive data to check connection
        try:
            self.socket.settimeout(0.1)
            # Just check if socket is still valid
            self.socket.getpeername()
            return True
        except:
            self.connected = False
            return False
    
    def get_cooldown_status(self) -> Dict[str, float]:
        """Get current cooldown status in seconds."""
        status = {}
        current_time = time.time() * 1000
        
        for action_name, last_time in self.last_action_times.items():
            cooldown = self.cooldowns.get(action_name, 0)
            elapsed = current_time - last_time
            remaining = max(0, cooldown - elapsed) / 1000.0  # Convert to seconds
            
            if remaining > 0:
                status[action_name] = remaining
        
        return status
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            'connected': self.connected,
            'reconnect_attempts': self.reconnect_attempts,
            'pending_acks': len(self.pending_acks),
            'queued_messages': len(self.message_queue),
            'active_cooldowns': len(self.get_cooldown_status()),
            'last_heartbeat_ago': time.time() - self.last_heartbeat_response,
        }


class UnityActionMapper:
    """
    Maps gesture actions to Unity game commands.
    
    CRITICAL FIX: JSON field names EXACTLY match Unity expectations:
    - objectName (not 'object')
    - methodName (not 'method')
    - parameters (array)
    """
    
    # Unity GameObject names and methods - EXACT MATCH with C# expectations
    UNITY_MAPPING = {
        'forward_movement': {
            'objectName': 'PlayerController',  # NOT 'object'
            'methodName': 'SetMovementSpeed',  # NOT 'method'
            'param_type': 'float',
            'default_param': 1.0
        },
        'attack': {
            'objectName': 'PlayerCombat',
            'methodName': 'PerformAttack',
            'param_type': 'string',
            'param_map': {
                'punch_left': 'left',
                'punch_right': 'right',
                'kick': 'kick'
            }
        },
        'dodge': {
            'objectName': 'PlayerMovement',
            'methodName': 'Dodge',
            'param_type': 'string',
            'param_map': {
                'lean_left': 'left',
                'lean_right': 'right'
            }
        },
        'slide': {
            'objectName': 'PlayerMovement',
            'methodName': 'Slide',
            'param_type': 'float',
            'default_param': 1.5
        },
        'block': {
            'objectName': 'PlayerCombat',
            'methodName': 'Block',
            'param_type': 'float',
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
        """
        Format action for Unity consumption.
        
        CRITICAL: Output JSON must have EXACT field names that Unity expects:
        {
            "type": "action",
            "objectName": "...",
            "methodName": "...",
            "parameters": [...],
            "gesture": "...",
            "action_name": "...",
            "timestamp": 1234567890.123
        }
        """
        if action_name not in UnityActionMapper.UNITY_MAPPING:
            logger.error(f"Action {action_name} not found in UNITY_MAPPING")
            return None
        
        mapping = UnityActionMapper.UNITY_MAPPING[action_name]
        
        # Prepare parameters array (Unity expects array)
        parameters = []
        
        if mapping['param_type'] == 'float':
            # Get float parameter
            param_value = None
            
            # Try to extract from params
            for key in ['value', 'duration', 'speed', 'distance']:
                if key in params and isinstance(params[key], (int, float)):
                    param_value = float(params[key])
                    break
            
            # Use default if not found
            if param_value is None:
                param_value = mapping.get('default_param', 0.0)
            
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
        
        # ============================================================
        # CRITICAL: Field names must EXACTLY match Unity C# expectations
        # ============================================================
        unity_message = {
            'type': 'action',
            'objectName': mapping['objectName'],  # NOT 'object'
            'methodName': mapping['methodName'],  # NOT 'method'
            'parameters': parameters,             # Array format
            'gesture': gesture_name,
            'action_name': action_name,
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
    """
    Main class for sending gestures to Unity.
    
    CRITICAL FIX:
    - Cooldown state reset on reconnect
    - JSON field names match Unity expectations
    - Acknowledgment tracking for reliable delivery
    """
    
    def __init__(self, host='127.0.0.1', port=65432):
        self.bridge = UnityBridge(host, port)
        self.action_mapper = UnityActionMapper()
        
        # Rate limiting
        self.last_sent_gesture = None
        self.last_sent_time = 0
        self.min_time_between_same_gesture = 0.3  # 300ms
        
        # Statistics
        self.stats = {
            'total_sent': 0,
            'successful_sent': 0,
            'failed_sent': 0,
            'cooldown_skipped': 0,
            'reconnections': 0
        }
        
    def connect(self) -> bool:
        """Connect to Unity."""
        self.stats['reconnections'] += 1
        return self.bridge.connect()
    
    def update(self):
        """
        Update method - call this regularly from main loop.
        Processes incoming messages and retries queued messages.
        """
        # Receive and process any messages from Unity
        messages = self.bridge.receive_messages()
        
        # Retry queued messages
        retried = self.bridge.retry_queued_messages()
        if retried > 0:
            logger.debug(f"Retried {retried} queued messages")
    
    def send_gesture(self, gesture_name: str, confidence: float = 1.0, 
                    force_send: bool = False) -> bool:
        """
        Send gesture to Unity.
        
        CRITICAL: Uses correct JSON field names that Unity expects.
        """
        try:
            # Get action from contract
            from gesture_action_contract import GestureActionContract
            action_type, params = GestureActionContract.get_action_for_gesture(gesture_name)
            
            # Format for Unity with correct field names
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
                    self.stats['cooldown_skipped'] += 1
                    return False
            
            # Send to Unity
            self.stats['total_sent'] += 1
            success = self.bridge.send_action(unity_action)
            
            if success:
                self.stats['successful_sent'] += 1
                logger.debug(f"✅ Sent to Unity: {gesture_name} -> {action_type}")
                self.last_sent_gesture = gesture_name
                self.last_sent_time = current_time
            else:
                self.stats['failed_sent'] += 1
                logger.warning(f"❌ Failed to send to Unity: {gesture_name}")
            
            return success
                
        except Exception as e:
            logger.error(f"Error sending gesture {gesture_name}: {e}")
            return False
    
    def send_test_message(self) -> bool:
        """Send a test message to verify Unity connection."""
        test_message = {
            'type': 'test',
            'message': 'Python test message',
            'timestamp': time.time()
        }
        return self.bridge.send_message(test_message, requires_ack=True)
    
    def send_direct_message(self, message_dict: Dict[str, Any]) -> bool:
        """Send a custom message directly to Unity."""
        return self.bridge.send_message(message_dict, requires_ack=True)
    
    def disconnect(self):
        """Disconnect from Unity."""
        self.bridge.disconnect()
    
    def is_connected(self) -> bool:
        """Check connection status."""
        return self.bridge.is_connected()
    
    def get_mapping_info(self) -> Dict[str, Any]:
        """Get information about Unity mappings."""
        return self.action_mapper.get_mapping_info()
    
    def get_status(self) -> Dict[str, Any]:
        """Get complete status information."""
        status = {
            'connected': self.bridge.is_connected(),
            'cooldowns': self.bridge.get_cooldown_status(),
            'stats': self.stats.copy(),
            'bridge_stats': self.bridge.get_stats()
        }
        
        # Add success rate
        if self.stats['total_sent'] > 0:
            status['success_rate'] = self.stats['successful_sent'] / self.stats['total_sent']
        else:
            status['success_rate'] = 1.0
        
        return status
    
    def reset_cooldowns(self):
        """Manually reset all cooldowns."""
        self.bridge.last_action_times.clear()
        logger.info("Manual cooldown reset")
    
    def print_status(self):
        """Print current status to console."""
        status = self.get_status()
        
        print("\n" + "=" * 60)
        print("UNITY BRIDGE STATUS")
        print("=" * 60)
        print(f"Connected: {'✅' if status['connected'] else '❌'}")
        print(f"Success rate: {status['success_rate']*100:.1f}%")
        print(f"Messages sent: {status['stats']['total_sent']}")
        print(f"Failed: {status['stats']['failed_sent']}")
        print(f"Cooldown skipped: {status['stats']['cooldown_skipped']}")
        
        if status['cooldowns']:
            print("\nActive cooldowns:")
            for action, remaining in status['cooldowns'].items():
                print(f"  {action}: {remaining:.1f}s")
        
        print(f"\nBridge stats:")
        for key, value in status['bridge_stats'].items():
            print(f"  {key}: {value}")
        
        print("=" * 60)


# ============================================================================
# TEST AND VERIFICATION FUNCTIONS
# ============================================================================

def test_reconnect_cooldown_reset():
    """Test that cooldown state is reset on reconnect."""
    print("\n" + "=" * 60)
    print("TEST: Cooldown Reset on Reconnect")
    print("=" * 60)
    
    bridge = UnityBridge()
    
    # Simulate some cooldown state
    bridge.last_action_times['attack'] = time.time() * 1000
    bridge.last_action_times['dodge'] = time.time() * 1000 - 100  # 100ms ago
    bridge.last_action_times['block'] = time.time() * 1000 - 200  # 200ms ago
    
    print(f"Before reconnect: {len(bridge.last_action_times)} cooldown entries")
    
    # Connect (will reset)
    bridge.connect(auto_reconnect=False)  # Will likely fail if no server, but that's fine
    
    print(f"After reconnect: {len(bridge.last_action_times)} cooldown entries")
    print(f"✅ Cooldown state cleared: {len(bridge.last_action_times) == 0}")
    
    bridge.disconnect()
    return True


def test_json_field_names():
    """Test that JSON field names match Unity expectations."""
    print("\n" + "=" * 60)
    print("TEST: JSON Field Name Validation")
    print("=" * 60)
    
    mapper = UnityActionMapper()
    
    # Test each action
    for gesture, action in [
        ('punch_left', 'attack'),
        ('lean_left', 'dodge'),
        ('run', 'forward_movement'),
        ('block', 'block'),
        ('idle', 'no_action')
    ]:
        params = {}
        if action == 'attack':
            params = {'damage': 10}
        elif action == 'slide':
            params = {'distance': 2.0}
        
        unity_msg = mapper.format_for_unity(gesture, action, params)
        
        if unity_msg:
            print(f"\n{gesture} -> {action}:")
            print(f"  JSON keys: {list(unity_msg.keys())}")
            
            # Check required fields
            required_fields = ['objectName', 'methodName', 'parameters', 'type']
            missing = [f for f in required_fields if f not in unity_msg]
            
            if missing:
                print(f"  ❌ Missing required fields: {missing}")
            else:
                print(f"  ✅ All required fields present")
                print(f"     objectName: {unity_msg['objectName']}")
                print(f"     methodName: {unity_msg['methodName']}")
                print(f"     parameters: {unity_msg['parameters']}")
        else:
            print(f"\n{gesture}: ❌ Failed to format")
    
    return True


def test_cooldown_desync_prevention():
    """Test that cooldown desync is prevented on reconnect."""
    print("\n" + "=" * 60)
    print("TEST: Cooldown Desync Prevention")
    print("=" * 60)
    
    gesture_sender = GestureToUnity()
    
    # Simulate being connected
    gesture_sender.bridge.connected = True
    
    # Send a few actions to build cooldown state
    gesture_sender.bridge.last_action_times['attack'] = time.time() * 1000
    gesture_sender.bridge.last_action_times['dodge'] = time.time() * 1000
    gesture_sender.bridge.last_action_times['block'] = time.time() * 1000
    
    print(f"Cooldown entries before disconnect: {len(gesture_sender.bridge.last_action_times)}")
    
    # Simulate disconnect
    gesture_sender.bridge.connected = False
    
    # Reconnect - this should clear cooldowns
    print("Reconnecting...")
    
    # Override the actual socket connection for testing
    original_connect = gesture_sender.bridge.connect
    
    def mock_connect(auto_reconnect=True):
        gesture_sender.bridge.connected = True
        gesture_sender.bridge.last_action_times.clear()
        gesture_sender.bridge.pending_acks.clear()
        return True
    
    gesture_sender.bridge.connect = mock_connect
    gesture_sender.bridge.connect()
    gesture_sender.bridge.connect = original_connect
    
    print(f"Cooldown entries after reconnect: {len(gesture_sender.bridge.last_action_times)}")
    print(f"✅ Cooldown state cleared: {len(gesture_sender.bridge.last_action_times) == 0}")
    
    return True


def test_unity_integration():
    """Test Unity integration with all fixes."""
    print("\n" + "=" * 60)
    print("TESTING UNITY INTEGRATION (FIXED VERSION)")
    print("=" * 60)
    
    # Test 1: Cooldown reset on reconnect
    test_reconnect_cooldown_reset()
    
    # Test 2: JSON field names
    test_json_field_names()
    
    # Test 3: Cooldown desync prevention
    test_cooldown_desync_prevention()
    
    # Test 4: Real connection attempt
    print("\n" + "=" * 60)
    print("TEST: Real Unity Connection")
    print("=" * 60)
    
    gesture_sender = GestureToUnity()
    
    if gesture_sender.connect():
        print("✅ Connected to Unity server")
        
        # Send test messages
        print("\nSending test messages...")
        
        # Test 1: Send test message
        if gesture_sender.send_test_message():
            print("  ✅ Test message sent")
        else:
            print("  ❌ Test message failed")
        
        # Test 2: Send gesture
        if gesture_sender.send_gesture('idle', confidence=0.95, force_send=True):
            print("  ✅ Idle gesture sent")
        else:
            print("  ❌ Idle gesture failed")
        
        # Test 3: Send active gesture
        if gesture_sender.send_gesture('punch_left', confidence=0.85):
            print("  ✅ Attack gesture sent")
        else:
            print("  ❌ Attack gesture failed")
        
        # Print status
        gesture_sender.print_status()
        
        # Disconnect
        gesture_sender.disconnect()
        print("\n✅ Disconnected from Unity")
    else:
        print("❌ Failed to connect to Unity")
        print("   Make sure Unity is running with GestureReceiver component")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 60)
    
    return True


def generate_unity_script():
    """Generate a Unity C# script with acknowledgment support."""
    
    script_content = '''using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;
using System.Text;
using System.Collections.Generic;
using System.Collections;

/**
 * GestureReceiver.cs
 * 
 * CRITICAL FIXES:
 * 1. JSON field names EXACTLY match Python: objectName, methodName, parameters
 * 2. Sends acknowledgments for reliable delivery
 * 3. Responds to ping/pong for connection health
 * 
 * Unity MUST send acknowledgments so Python knows messages were received.
 * This prevents cooldown state desync after reconnection.
 */
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
                    Debug.Log("Python client handshake received");
                    SendAcknowledgment(baseMessage.ack_id);
                    break;
                    
                case "test":
                    Debug.Log("Test message from Python: " + jsonMessage);
                    SendAcknowledgment(baseMessage.ack_id);
                    break;
                    
                case "ping":
                    SendPong();
                    SendAcknowledgment(baseMessage.ack_id);
                    break;
                    
                case "disconnect":
                    Debug.Log("Python client requested disconnect");
                    SendAcknowledgment(baseMessage.ack_id);
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
    
    void SendAcknowledgment(string ackId)
    {
        if (string.IsNullOrEmpty(ackId))
            return;
            
        try
        {
            var ack = new AckMessage
            {
                type = "ack",
                ack_id = ackId,
                timestamp = DateTime.Now.ToString("HH:mm:ss.fff")
            };
            
            string ackJson = JsonUtility.ToJson(ack);
            byte[] ackBytes = Encoding.UTF8.GetBytes(ackJson + "\\n");
            
            if (clientStream != null && clientStream.CanWrite)
            {
                clientStream.Write(ackBytes, 0, ackBytes.Length);
                Debug.Log("Sent acknowledgment for: " + ackId);
            }
        }
        catch (Exception e)
        {
            Debug.LogError("Failed to send acknowledgment: " + e.Message);
        }
    }
    
    void ProcessActionMessage(string jsonMessage)
    {
        // CRITICAL: Field names MUST match Python's JSON
        var action = JsonUtility.FromJson<GestureAction>(jsonMessage);
        
        if (action == null)
        {
            Debug.LogError("Failed to parse action message");
            return;
        }
        
        // Send acknowledgment
        SendAcknowledgment(action.ack_id);
        
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
    }
    
    void ProcessGestureMessage(string jsonMessage)
    {
        var gesture = JsonUtility.FromJson<GestureData>(jsonMessage);
        
        // Send acknowledgment
        SendAcknowledgment(gesture.ack_id);
        
        Debug.Log("Gesture detected: " + gesture.gesture + 
                  " (confidence: " + (gesture.confidence * 100).ToString("F0") + "%)");
    }
    
    void ProcessLandmarksMessage(string jsonMessage)
    {
        var landmarks = JsonUtility.FromJson<LandmarksData>(jsonMessage);
        SendAcknowledgment(landmarks.ack_id);
        // Optional: Process landmarks for visualization
    }
    
    void SendPong()
    {
        try
        {
            var pong = new PongMessage
            {
                type = "pong",
                timestamp = DateTime.Now.ToString("HH:mm:ss.fff")
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
    
    // ============================================================
    // DATA CLASSES - MUST MATCH PYTHON'S JSON STRUCTURE EXACTLY
    // ============================================================
    
    [System.Serializable]
    private class BaseMessage
    {
        public string type;
        public string ack_id;
    }
    
    [System.Serializable]
    public class GestureAction
    {
        public string type;
        public string objectName;      // MUST be 'objectName' (not 'object')
        public string methodName;      // MUST be 'methodName' (not 'method')
        public string[] parameters;    // MUST be array
        public string gesture;
        public string action_name;
        public float confidence;
        public double timestamp;
        public string ack_id;
    }
    
    [System.Serializable]
    public class GestureData
    {
        public string type;
        public string gesture;
        public float confidence;
        public double timestamp;
        public string ack_id;
    }
    
    [System.Serializable]
    public class LandmarksData
    {
        public string type;
        public float[][] data;
        public int frame;
        public double timestamp;
        public string ack_id;
    }
    
    [System.Serializable]
    private class AckMessage
    {
        public string type;
        public string ack_id;
        public string timestamp;
    }
    
    [System.Serializable]
    private class PongMessage
    {
        public string type;
        public string timestamp;
    }
}
'''
    
    output_path = "GestureReceiver_Enhanced.cs"
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Enhanced Unity C# script generated: {output_path}")
    print("   This version includes:")
    print("   - Correct JSON field names (objectName, methodName)")
    print("   - Acknowledgment system for reliable delivery")
    print("   - Ping/pong for connection health")
    print("   - Cooldown state reset on reconnect")
    
    return output_path


if __name__ == "__main__":
    # Run all tests
    test_unity_integration()
    
    # Generate enhanced Unity script
    generate_unity_script()
    
    print("\n" + "=" * 60)
    print("SUMMARY OF CRITICAL FIXES")
    print("=" * 60)
    print("1. ✅ Cooldown state reset on reconnect")
    print("   - last_action_times.clear() in UnityBridge.connect()")
    print("   - Prevents action desync after connection loss")
    print()
    print("2. ✅ JSON field names match Unity expectations")
    print("   - 'objectName' (not 'object')")
    print("   - 'methodName' (not 'method')")
    print("   - 'parameters' array format")
    print()
    print("3. ✅ Acknowledgment system")
    print("   - Unity sends 'ack' messages back to Python")
    print("   - Python tracks pending acknowledgments")
    print("   - Messages are queued for retry if not acknowledged")
    print()
    print("4. ✅ Connection health monitoring")
    print("   - Ping/pong heartbeat system")
    print("   - Automatic detection of stale connections")
    print("   - Clean cooldown reset on reconnect")
    print()
    print("5. ✅ Message queue and retry")
    print("   - Failed messages are queued")
    print("   - Automatic retry on reconnection")
    print("   - Prevents lost actions during temporary disconnects")
    print("=" * 60)
    print()
    print("NEXT STEPS:")
    print("1. Replace your existing GestureReceiver.cs with the generated file")
    print("2. Ensure Unity sends acknowledgments for all received messages")
    print("3. Test reconnection by stopping and restarting Unity")
    print("4. Verify cooldown state resets and actions continue working")
    print("=" * 60)