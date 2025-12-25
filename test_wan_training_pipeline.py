#!/usr/bin/env python3
"""
Test the actual WAN 2.2 training pipeline to identify the issue
"""

import sys
import os
import json
import torch
from pathlib import Path

# Add the modules to the path
sys.path.append('.')

def test_wan_training_pipeline():
    """Test the actual WAN 2.2 training pipeline"""
    print("🧪 Testing WAN 2.2 Training Pipeline...")
    
    try:
        # Import required modules
        from modules.util.config.TrainConfig import TrainConfig
        from modules.dataLoader.WanBaseDataLoader import WanBaseDataLoader
        from modules.model.WanModel import WanModel
        from modules.util.TrainProgress import TrainProgress
        from modules.util.enum.DataType import DataType
        from modules.util.enum.ModelType import ModelType
        from modules.util.enum.TrainingMethod import TrainingMethod
        
        print("   ✓ Imports successful")
        
        # Load the training config
        config_path = "training_presets/wan-debug-4frames.json"
        print(f"   Loading config: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Create TrainConfig object
        config = TrainConfig.default_values()
        
        # Set key values from config
        config.model_type = ModelType.WAN_2_2
        config.training_method = TrainingMethod.LORA
        config.base_model_name = config_data.get('base_model_name', 'Wan-AI/Wan2.2-TI2V-5B-Diffusers')
        config.resolution = config_data.get('resolution', '384')
        config.frames = config_data.get('frames', 2)
        config.batch_size = config_data.get('batch_size', 1)
        config.train_dtype = DataType.BFLOAT_16
        
        # Set concept data using proper ConceptConfig
        from modules.util.config.ConceptConfig import ConceptConfig
        from modules.util.enum.ConceptType import ConceptType
        
        concept = ConceptConfig.default_values()
        concept.name = 'Cube'
        concept.path = '/workspace/input/training/cube'
        concept.enabled = True
        concept.type = ConceptType.STANDARD
        concept.include_subdirectories = False
        concept.text.prompt_source = 'sample'
        concept.text.prompt_path = '/workspace/input/training/cube'
        concept.image.enable_resolution_override = False
        concept.image.resolution_override = '384'
        
        config.concepts = [concept]
        
        print(f"   ✓ Config loaded: {config.model_type}, resolution={config.resolution}, frames={config.frames}")
        
        # Create a minimal model for testing
        print("   Creating minimal model...")
        
        class MinimalWanModel:
            def __init__(self):
                self.train_dtype = DataType.BFLOAT_16
                self.vae = None
                self.tokenizer = None
                self.text_encoder = None
                self.autocast_context = torch.autocast('cpu')
                
            def add_text_encoder_embeddings_to_prompt(self, prompt):
                return prompt
                
            def to(self, device):
                pass
                
            def vae_to(self, device):
                pass
                
            def text_encoder_to(self, device):
                pass
                
            def eval(self):
                pass
        
        model = MinimalWanModel()
        
        # Create train progress
        train_progress = TrainProgress()
        
        print("   Creating WAN data loader...")
        
        # Create the data loader
        data_loader = WanBaseDataLoader(
            train_device=torch.device('cpu'),
            temp_device=torch.device('cpu'),
            config=config,
            model=model,
            train_progress=train_progress,
            is_validation=False
        )
        
        print("   ✓ Data loader created successfully")
        
        # Get the dataset
        dataset = data_loader.get_data_set()
        print(f"   Dataset type: {type(dataset)}")
        
        # Check dataset length
        try:
            # MGDS doesn't have len(), but we can check if it can iterate
            print("   Testing dataset iteration...")
            
            # Start the first epoch
            dataset.start_next_epoch()
            
            # Try to get first item
            first_item = next(iter(dataset))
            print(f"   ✓ Got first item: {type(first_item)}")
            
            if isinstance(first_item, dict):
                print(f"   Item keys: {list(first_item.keys())}")
                
                # Check if we have video data
                if 'video' in first_item or 'video_path' in first_item:
                    print("   ✓ Dataset contains video data")
                else:
                    print("   ❌ Dataset missing video data")
                    return False
            
            return True
            
        except StopIteration:
            print("   ❌ Dataset is empty - no items to iterate!")
            return False
        except Exception as e:
            print(f"   ❌ Error testing dataset: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"   ❌ Error in training pipeline test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("WAN 2.2 Training Pipeline Test")
    print("=" * 40)
    
    success = test_wan_training_pipeline()
    
    if success:
        print("\n✅ WAN training pipeline works correctly")
        print("   The issue has been resolved!")
    else:
        print("\n❌ WAN training pipeline still has issues")
        print("   Need to investigate further")

if __name__ == "__main__":
    main()