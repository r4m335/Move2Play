# gesture_action_contract.py
"""
Defines the immutable mapping between gestures and game actions.
Versioned contract that must remain consistent after training starts.

CONTRACT_VERSION: 1.0
LAST_MODIFIED: 2023-10-15
FROZEN_AFTER_TRAINING: True
"""

CONTRACT_VERSION = "1.0"
LAST_MODIFIED = "2023-10-15"
FROZEN_AFTER_TRAINING = True

# ============================================================================
# PRIMARY GESTURE-ACTION MAPPING
# ============================================================================
# This mapping defines the core relationship between detected gestures and
# game actions. Multiple gestures can map to the same action type with
# different parameters (e.g., punch_left and punch_right both map to 'attack').

GESTURE_ACTION_MAPPING = {
    # Gesture: (Action Type, Action Parameters)
    "run": ("forward_movement", {"speed": 1.0}),
    "punch_left": ("attack", {"side": "left", "damage": 25}),
    "punch_right": ("attack", {"side": "right", "damage": 25}),
    "kick": ("attack", {"type": "kick", "damage": 35}),
    "lean_left": ("dodge", {"direction": "left", "distance": 2.0}),
    "lean_right": ("dodge", {"direction": "right", "distance": 2.0}),
    "squat": ("slide", {"duration": 1.5, "height_reduction": 0.5}),
    "block": ("defense", {"damage_reduction": 0.7, "duration": 2.0}),
}

# ============================================================================
# DERIVED CONSTANTS (DO NOT MODIFY DIRECTLY)
# ============================================================================

# All gesture classes in alphabetical order for consistent model training
GESTURE_CLASSES = sorted(list(GESTURE_ACTION_MAPPING.keys()))
NUM_GESTURES = len(GESTURE_CLASSES)

# Action types for reference
ACTION_TYPES = sorted(list(set(action_type for action_type, _ in GESTURE_ACTION_MAPPING.values())))

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
        
        print(f"✅ Contract validation passed (Version {CONTRACT_VERSION})")
        print(f"   Gestures: {NUM_GESTURES}, Actions: {len(ACTION_TYPES)}")
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
            "notes": "DO NOT MODIFY AFTER TRAINING STARTS",
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

# ============================================================================
# USAGE EXAMPLES AND TESTING
# ============================================================================

def print_contract_summary():
    """Print a human-readable summary of the contract."""
    print("=" * 60)
    print("GESTURE-ACTION CONTRACT SUMMARY")
    print("=" * 60)
    print(f"Version: {CONTRACT_VERSION}")
    print(f"Gestures: {NUM_GESTURES}")
    print(f"Action Types: {len(ACTION_TYPES)}")
    print()
    
    # Group gestures by action type
    print("MAPPING:")
    for action_type in ACTION_TYPES:
        gestures = GestureActionContract.get_gestures_by_action(action_type)
        print(f"  {action_type}:")
        for gesture in gestures:
            _, params = GESTURE_ACTION_MAPPING[gesture]
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"    - {gesture} ({param_str})")
    
    print("\n" + "=" * 60)

def test_contract():
    """Run comprehensive tests on the contract."""
    print("Testing Gesture-Action Contract...")
    
    try:
        # Validate consistency
        GestureActionContract.validate_consistency()
        
        # Test individual lookups
        test_cases = [
            ("run", ("forward_movement", {"speed": 1.0})),
            ("punch_left", ("attack", {"side": "left", "damage": 25})),
            ("block", ("defense", {"damage_reduction": 0.7, "duration": 2.0})),
        ]
        
        for gesture, expected in test_cases:
            result = GestureActionContract.get_action_for_gesture(gesture)
            assert result == expected, f"Mismatch for {gesture}: {result} != {expected}"
            print(f"  ✓ {gesture} -> {result[0]}")
        
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
        
        # Test grouping by action
        attack_gestures = GestureActionContract.get_gestures_by_action("attack")
        assert set(attack_gestures) == {"punch_left", "punch_right", "kick"}
        print(f"  ✓ Attack gestures: {attack_gestures}")
        
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
        print(f"⚠️  Contract is frozen. Current hash: {current_hash[:16]}...")
        
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
    
    print("\nContract is ready for use!")