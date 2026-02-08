# gesture_action_contract.py (MERGED VERSION 1.1)
"""
Defines the immutable mapping between gestures and game actions.
Versioned contract that must remain consistent after training starts.

CONTRACT_VERSION: 1.1
LAST_MODIFIED: 2023-10-20
FROZEN_AFTER_TRAINING: True

IDLE CLASS ADDED: Prevents false positives by teaching model what "not a gesture" looks like.
"""

CONTRACT_VERSION = "1.1"
LAST_MODIFIED = "2023-10-20"
FROZEN_AFTER_TRAINING = True

# ============================================================================
# PRIMARY GESTURE-ACTION MAPPING (WITH IDLE CLASS)
# ============================================================================
# "idle" must be first and represents standing still/natural movements
# CRITICAL: Prevents false positives in real-time inference

GESTURE_ACTION_MAPPING = {
    # IDLE CLASS - Represents standing still, natural movements, breathing, posture shifts
    # This is the most important class for preventing false positives
    "idle": ("no_action", {"description": "standing still, natural movements"}),
    
    # Active gestures (alphabetical order for consistency)
    "block": ("defense", {"damage_reduction": 0.7, "duration": 2.0}),
    "kick": ("attack", {"type": "kick", "damage": 35}),
    "lean_left": ("dodge", {"direction": "left", "distance": 2.0}),
    "lean_right": ("dodge", {"direction": "right", "distance": 2.0}),
    "punch_left": ("attack", {"side": "left", "damage": 25}),
    "punch_right": ("attack", {"side": "right", "damage": 25}),
    "run": ("forward_movement", {"speed": 1.0}),
    "squat": ("slide", {"duration": 1.5, "height_reduction": 0.5}),
}

# ============================================================================
# DERIVED CONSTANTS (DO NOT MODIFY DIRECTLY)
# ============================================================================

# All gesture classes in alphabetical order (idle will be first alphabetically)
GESTURE_CLASSES = sorted(list(GESTURE_ACTION_MAPPING.keys()))
NUM_GESTURES = len(GESTURE_CLASSES)

# Action types for reference
ACTION_TYPES = sorted(list(set(action_type for action_type, _ in GESTURE_ACTION_MAPPING.values())))

# Special constants for idle class
IDLE_CLASS = "idle"
IDLE_ACTION = ("no_action", {"description": "standing still, natural movements"})

# Verify that idle is the first class (alphabetically)
assert GESTURE_CLASSES[0] == "idle", "idle must be the first gesture class (alphabetically)"

# ============================================================================
# VALIDATION AND UTILITY CLASS
# ============================================================================

class GestureActionContract:
    """Validates and enforces the gesture-action mapping."""
    
    @staticmethod
    def get_action_for_gesture(gesture_name: str):
        """
        Returns the corresponding action for a gesture.
        
        Args:
            gesture_name: Name of the detected gesture
            
        Returns:
            Tuple of (action_type, parameters_dict)
            
        Raises:
            ValueError: If gesture_name is not in the contract
        """
        if gesture_name not in GESTURE_ACTION_MAPPING:
            raise ValueError(
                f"Unknown gesture: '{gesture_name}'. "
                f"Valid gestures: {GESTURE_CLASSES}"
            )
        return GESTURE_ACTION_MAPPING[gesture_name]
    
    @staticmethod
    def get_gesture_index(gesture_name: str) -> int:
        """
        Get the index of a gesture in the GESTURE_CLASSES list.
        Used for model training labels.
        
        Args:
            gesture_name: Name of the gesture
            
        Returns:
            Integer index (0-based)
        """
        try:
            return GESTURE_CLASSES.index(gesture_name)
        except ValueError:
            raise ValueError(
                f"Gesture '{gesture_name}' not found in GESTURE_CLASSES. "
                f"Available gestures: {GESTURE_CLASSES}"
            )
    
    @staticmethod
    def validate_consistency() -> bool:
        """
        Validate the contract for consistency.
        
        Checks:
        1. Each gesture maps to exactly one (action_type, parameters) pair
        2. No duplicate gesture names
        3. All parameters are dictionaries
        4. GESTURE_CLASSES matches mapping keys
        5. idle class exists and is properly configured
        
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If any validation fails
        """
        # 1. Check for duplicate gestures (should be impossible with dict, but safe)
        gesture_names = list(GESTURE_ACTION_MAPPING.keys())
        if len(gesture_names) != len(set(gesture_names)):
            duplicates = [g for g in gesture_names if gesture_names.count(g) > 1]
            raise ValueError(f"Duplicate gestures found: {duplicates}")
        
        # 2. Check that all values are tuples of length 2
        for gesture, mapping in GESTURE_ACTION_MAPPING.items():
            if not isinstance(mapping, tuple) or len(mapping) != 2:
                raise ValueError(
                    f"Gesture '{gesture}' must map to (action_type, parameters) tuple. "
                    f"Got: {mapping}"
                )
            
            action_type, parameters = mapping
            
            # 3. Check action_type is string
            if not isinstance(action_type, str):
                raise ValueError(
                    f"Action type for gesture '{gesture}' must be string. "
                    f"Got: {type(action_type)}"
                )
            
            # 4. Check parameters is dict
            if not isinstance(parameters, dict):
                raise ValueError(
                    f"Parameters for gesture '{gesture}' must be dictionary. "
                    f"Got: {type(parameters)}"
                )
        
        # 5. Verify GESTURE_CLASSES matches mapping keys
        contract_gestures = set(GESTURE_ACTION_MAPPING.keys())
        derived_gestures = set(GESTURE_CLASSES)
        
        if contract_gestures != derived_gestures:
            missing_in_derived = contract_gestures - derived_gestures
            extra_in_derived = derived_gestures - contract_gestures
            
            error_msg = "GESTURE_CLASSES mismatch with GESTURE_ACTION_MAPPING keys:\n"
            if missing_in_derived:
                error_msg += f"  Missing in GESTURE_CLASSES: {sorted(missing_in_derived)}\n"
            if extra_in_derived:
                error_msg += f"  Extra in GESTURE_CLASSES: {sorted(extra_in_derived)}"
            
            raise ValueError(error_msg)
        
        # 6. Verify NUM_GESTURES is correct
        if NUM_GESTURES != len(GESTURE_CLASSES):
            raise ValueError(
                f"NUM_GESTURES ({NUM_GESTURES}) doesn't match "
                f"GESTURE_CLASSES length ({len(GESTURE_CLASSES)})"
            )
        
        # 7. Verify idle class exists and is first
        if IDLE_CLASS not in GESTURE_ACTION_MAPPING:
            raise ValueError(f"IDLE_CLASS '{IDLE_CLASS}' not found in mapping")
        
        if GESTURE_CLASSES[0] != IDLE_CLASS:
            raise ValueError(f"IDLE_CLASS '{IDLE_CLASS}' must be first in GESTURE_CLASSES")
        
        # 8. Verify idle action is correct
        if GESTURE_ACTION_MAPPING[IDLE_CLASS] != IDLE_ACTION:
            raise ValueError(f"IDLE_CLASS action mismatch. Expected {IDLE_ACTION}, got {GESTURE_ACTION_MAPPING[IDLE_CLASS]}")
        
        print(f"✅ Contract validation passed (Version {CONTRACT_VERSION})")
        print(f"   Gestures: {NUM_GESTURES}, Actions: {len(ACTION_TYPES)}")
        print(f"   Includes IDLE class for false positive prevention")
        return True
    
    @staticmethod
    def get_contract_info() -> dict:
        """
        Get complete contract information.
        
        Returns:
            Dictionary with contract metadata and mapping
        """
        return {
            "version": CONTRACT_VERSION,
            "last_modified": LAST_MODIFIED,
            "frozen_after_training": FROZEN_AFTER_TRAINING,
            "num_gestures": NUM_GESTURES,
            "num_action_types": len(ACTION_TYPES),
            "gesture_classes": GESTURE_CLASSES,
            "action_types": ACTION_TYPES,
            "mapping": GESTURE_ACTION_MAPPING.copy(),
            "has_idle_class": True,
            "idle_class": IDLE_CLASS,
        }
    
    @staticmethod
    def export_to_json(filepath: str = "gesture_contract.json") -> None:
        """
        Export the contract to a JSON file for reference.
        
        Args:
            filepath: Path to save the JSON file
        """
        import json
        import datetime
        
        export_data = {
            "export_timestamp": datetime.datetime.now().isoformat(),
            "contract": GestureActionContract.get_contract_info(),
            "notes": "DO NOT MODIFY AFTER TRAINING STARTS - Includes IDLE class v1.1",
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"Contract exported to {filepath}")
    
    @staticmethod
    def check_frozen() -> bool:
        """
        Check if the contract is frozen (training has started).
        
        Returns:
            True if contract should be considered frozen
        """
        if FROZEN_AFTER_TRAINING:
            # In a real system, you might check for existence of trained models
            import os
            if os.path.exists("models/trained") or os.path.exists("gesture_model.keras"):
                print("⚠️  Contract is FROZEN (training has started)")
                return True
        return False
    
    @staticmethod
    def get_gestures_by_action(action_type: str) -> list:
        """
        Get all gestures that map to a specific action type.
        
        Args:
            action_type: Type of action to filter by
            
        Returns:
            List of gesture names
        """
        return [
            gesture for gesture, (a_type, _) in GESTURE_ACTION_MAPPING.items()
            if a_type == action_type
        ]
    
    @staticmethod
    def is_idle_gesture(gesture_name: str) -> bool:
        """
        Check if a gesture is the idle/non-action class.
        
        Args:
            gesture_name: Name of the gesture
            
        Returns:
            True if it's the idle gesture
        """
        return gesture_name == IDLE_CLASS
    
    @staticmethod
    def get_active_gestures() -> list:
        """
        Get all non-idle (active) gesture names.
        
        Returns:
            List of active gesture names
        """
        return [g for g in GESTURE_CLASSES if g != IDLE_CLASS]

# ============================================================================
# USAGE EXAMPLES AND TESTING
# ============================================================================

def print_contract_summary():
    """Print a human-readable summary of the contract."""
    print("=" * 60)
    print("GESTURE-ACTION CONTRACT SUMMARY")
    print("=" * 60)
    print(f"Version: {CONTRACT_VERSION}")
    print(f"Gestures: {NUM_GESTURES} (including idle class)")
    print(f"Active Gestures: {NUM_GESTURES - 1}")
    print(f"Action Types: {len(ACTION_TYPES)}")
    print()
    
    # Highlight idle class
    print("IDLE CLASS (False Positive Prevention):")
    _, idle_params = GESTURE_ACTION_MAPPING[IDLE_CLASS]
    print(f"  {IDLE_CLASS}: {idle_params['description']}")
    print()
    
    # Group gestures by action type
    print("ACTIVE GESTURES BY ACTION TYPE:")
    for action_type in ACTION_TYPES:
        if action_type == "no_action":
            continue  # Already shown above
            
        gestures = GestureActionContract.get_gestures_by_action(action_type)
        print(f"  {action_type}:")
        for gesture in gestures:
            if gesture == IDLE_CLASS:
                continue
            _, params = GESTURE_ACTION_MAPPING[gesture]
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"    - {gesture} ({param_str})")
    
    print("\n" + "=" * 60)

def test_contract():
    """Run comprehensive tests on the contract."""
    print("Testing Gesture-Action Contract (v1.1 with IDLE class)...")
    
    try:
        # Validate consistency
        GestureActionContract.validate_consistency()
        
        # Test individual lookups
        test_cases = [
            ("idle", ("no_action", {"description": "standing still, natural movements"})),
            ("run", ("forward_movement", {"speed": 1.0})),
            ("punch_left", ("attack", {"side": "left", "damage": 25})),
            ("block", ("defense", {"damage_reduction": 0.7, "duration": 2.0})),
        ]
        
        for gesture, expected in test_cases:
            result = GestureActionContract.get_action_for_gesture(gesture)
            assert result == expected, f"Mismatch for {gesture}: {result} != {expected}"
            print(f"  ✓ {gesture} -> {result[0]}")
        
        # Test idle class helper
        assert GestureActionContract.is_idle_gesture("idle"), "is_idle_gesture should return True for idle"
        assert not GestureActionContract.is_idle_gesture("run"), "is_idle_gesture should return False for run"
        print("  ✓ Idle class detection works")
        
        # Test active gestures list
        active_gestures = GestureActionContract.get_active_gestures()
        assert "idle" not in active_gestures, "idle should not be in active gestures"
        assert len(active_gestures) == NUM_GESTURES - 1, f"Should have {NUM_GESTURES - 1} active gestures"
        print(f"  ✓ Active gestures: {len(active_gestures)} gestures")
        
        # Test error cases
        try:
            GestureActionContract.get_action_for_gesture("unknown_gesture")
            assert False, "Should have raised ValueError"
        except ValueError:
            print("  ✓ Unknown gesture raises error (as expected)")
        
        # Test gesture indices
        for i, gesture in enumerate(GESTURE_CLASSES):
            idx = GestureActionContract.get_gesture_index(gesture)
            assert idx == i, f"Index mismatch for {gesture}: {idx} != {i}"
        
        print("  ✓ All gesture indices are correct")
        
        # Test grouping by action (should exclude idle)
        attack_gestures = GestureActionContract.get_gestures_by_action("attack")
        assert set(attack_gestures) == {"punch_left", "punch_right", "kick"}
        print(f"  ✓ Attack gestures: {attack_gestures}")
        
        # Verify idle is first in classes list
        assert GESTURE_CLASSES[0] == "idle", "idle must be first in GESTURE_CLASSES"
        print("  ✓ idle is first in gesture classes list")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise

# ============================================================================
# IMMUTABILITY ENFORCEMENT (Development/Production Flag)
# ============================================================================

def assert_contract_unchanged():
    """
    Safety check: Ensure contract hasn't been modified after training.
    Call this at startup in production.
    """
    if GestureActionContract.check_frozen():
        # In production, you might compute a hash of the contract
        # and compare with a stored hash
        import hashlib
        import json
        
        # Create hash of the contract
        contract_str = json.dumps(GESTURE_ACTION_MAPPING, sort_keys=True)
        current_hash = hashlib.sha256(contract_str.encode()).hexdigest()
        
        # In a real system, you'd compare with a stored hash
        # For now, just log a warning
        print(f"⚠️  Contract is frozen (v{CONTRACT_VERSION}). Current hash: {current_hash[:16]}...")
        
        # You could also load a stored hash from file/database
        # and compare:
        # if current_hash != stored_hash:
        #     raise RuntimeError("Gesture-Action contract has been modified!")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # When run directly, validate and print summary
    print_contract_summary()
    test_contract()
    
    # Export for reference
    GestureActionContract.export_to_json()
    
    # Check frozen status
    GestureActionContract.check_frozen()
    
    print("\nContract v1.1 with IDLE class is ready for use!")