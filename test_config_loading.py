#!/usr/bin/env python3
"""
Test configuration loading for WAN 2.2 training
"""
import sys
import json
sys.path.append('.')

def test_config_loading():
    """Test loading the training configuration"""
    
    # Load the JSON config directly
    config_path = "training_presets/#wan 2.2 LoRA flexible.json"
    
    print(f"=== Loading config from {config_path} ===")
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        print("Config loaded successfully!")
        print(f"Base model: {config_data.get('base_model_name', 'NOT SET')}")
        print(f"Model type: {config_data.get('model_type', 'NOT SET')}")
        print(f"Resolution: {config_data.get('resolution', 'NOT SET')}")
        print(f"Frames: {config_data.get('frames', 'NOT SET')}")
        print(f"Batch size: {config_data.get('batch_size', 'NOT SET')}")
        print(f"Concept file: {config_data.get('concept_file_name', 'training_concepts/concepts.json')}")
        
        # Check if concept file exists
        concept_file = config_data.get('concept_file_name', 'training_concepts/concepts.json')
        try:
            with open(concept_file, 'r') as f:
                concepts = json.load(f)
            print(f"\nConcept file loaded: {len(concepts)} concepts found")
            for i, concept in enumerate(concepts):
                print(f"  Concept {i}: {concept.get('name', 'UNNAMED')} -> {concept.get('path', 'NO PATH')}")
        except Exception as e:
            print(f"Error loading concept file: {e}")
            
    except Exception as e:
        print(f"Error loading config: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_config_loading()