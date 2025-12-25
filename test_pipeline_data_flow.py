#!/usr/bin/env python3
"""
Test the WAN 2.2 pipeline data flow step by step to identify where it breaks
"""

import sys
import os
import json
import torch
from pathlib import Path

# Add the modules to the path
sys.path.append('.')

def test_pipeline_data_flow():
    """Test each step of the pipeline data flow"""
    print("🔍 Testing WAN 2.2 Pipeline Data Flow...")
    
    try:
        # Import required modules
        from modules.util.config.TrainConfig import TrainConfig
        from modules.dataLoader.WanBaseDataLoader import WanBaseDataLoader
        from modules.util.config.ConceptConfig import ConceptConfig
        from modules.util.enum.ConceptType import ConceptType
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
        
        # Start the first epoch to initialize everything
        print("   Starting first epoch...")
        dataset.start_next_epoch()
        
        # Now let's manually test each module in the pipeline
        print("   Testing individual modules...")
        
        # Get the pipeline
        pipeline = dataset.loading_pipeline
        print(f"   Pipeline has {len(pipeline.modules)} modules")
        
        # Test the first few modules manually
        for i in range(min(5, len(pipeline.modules))):
            module = pipeline.modules[i]
            module_name = type(module).__name__
            print(f"   Testing module {i}: {module_name}")
            
            try:
                # Check if module has length
                if hasattr(module, 'length'):
                    length = module.length()
                    print(f"     Length: {length}")
                
                # Try to get first item if it's a data module
                if hasattr(module, 'get_item') and i >= 2:  # Skip ConceptPipelineModule and SettingsPipelineModule
                    print(f"     Testing get_item(0, 0)...")
                    item = module.get_item(0, 0)
                    print(f"     Result: {type(item)}")
                    if isinstance(item, dict):
                        print(f"     Keys: {list(item.keys())}")
                    elif item is None:
                        print(f"     ❌ Module returned None!")
                        return False
                    else:
                        print(f"     Value: {item}")
                
            except Exception as e:
                print(f"     ❌ Error testing module {i}: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        print("   ✓ Individual module tests passed")
        
        # Now test the full pipeline iteration
        print("   Testing full pipeline iteration...")
        try:
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
            
        except Exception as e:
            print(f"   ❌ Error in full pipeline iteration: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"   ❌ Error in pipeline data flow test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("WAN 2.2 Pipeline Data Flow Test")
    print("=" * 40)
    
    success = test_pipeline_data_flow()
    
    if success:
        print("\n✅ WAN pipeline data flow works correctly")
        print("   The issue has been resolved!")
    else:
        print("\n❌ WAN pipeline data flow still has issues")
        print("   Need to investigate the specific module that's failing")

if __name__ == "__main__":
    main()