#!/usr/bin/env python3
"""
Test script to debug WAN 2.2 video pipeline and see where videos get filtered out.
"""

import sys
import os
sys.path.append('.')

import json
import torch
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def test_wan_pipeline():
    """Test the WAN 2.2 pipeline step by step"""
    
    try:
        from modules.util.config.TrainConfig import TrainConfig
        from modules.util.TrainProgress import TrainProgress
        from modules.util.enum.ModelType import ModelType
        from modules.util.enum.DataType import DataType
        
        print("🎯 Testing WAN 2.2 Video Pipeline")
        print("=" * 40)
        
        # Create config
        config = TrainConfig.default_values()
        config.model_type = ModelType.WAN_2_2
        config.base_model_name = 'Wan-AI/Wan2.2-TI2V-5B-Diffusers'
        config.concept_file_name = 'training_concepts/concepts.json'
        config.resolution = '256'
        config.frames = 4  # Reduced for testing
        config.batch_size = 1
        config.training_method = 'LORA'
        config.train_dtype = DataType.BFLOAT_16
        config.dataloader_threads = 1
        
        print(f"✓ Config: {config.model_type}, resolution={config.resolution}, frames={config.frames}")
        
        # Check concepts
        with open(config.concept_file_name, 'r') as f:
            concepts = json.load(f)
        
        concept = concepts[0]
        print(f"✓ Concept: {concept['name']} -> {concept['path']}")
        print(f"   Video count from stats: {concept['concept_stats']['video_count']}")
        
        # Check files directly
        from pathlib import Path
        cube_path = Path(concept['path'])
        video_files = list(cube_path.glob('*.mp4'))
        print(f"✓ Direct file check: {len(video_files)} MP4 files")
        
        # Test with minimal model setup
        print("\n🔧 Testing data loader creation...")
        
        try:
            from modules.dataLoader.WanBaseDataLoader import WanBaseDataLoader
            from modules.util.enum.DataType import DataType
            
            # Create train progress
            train_progress = TrainProgress()
            
            # Create devices
            train_device = torch.device('cpu')
            temp_device = torch.device('cpu')
            
            # Create a minimal mock model with required attributes
            class MockModel:
                def __init__(self):
                    self.train_dtype = DataType.BFLOAT_16
                    self.vae = None
                    self.text_encoder = None
                    self.transformer = None
                    self.autocast_context = None
                    
            mock_model = MockModel()
            
            print("Creating WanBaseDataLoader...")
            data_loader = WanBaseDataLoader(
                train_device=train_device,
                temp_device=temp_device,
                config=config,
                model=mock_model,
                train_progress=train_progress,
                is_validation=False
            )
            
            print("✓ Created WanBaseDataLoader successfully")
            
            # Test dataset length
            try:
                dataset_length = len(data_loader)
                print(f"✓ Dataset length: {dataset_length}")
                
                if dataset_length > 0:
                    print("🎉 SUCCESS: Videos are being loaded!")
                    
                    # Try to get a sample
                    try:
                        sample = data_loader[0]
                        print(f"✓ Got sample: {type(sample)}")
                        if isinstance(sample, dict):
                            print(f"   Sample keys: {list(sample.keys())}")
                    except Exception as e:
                        print(f"⚠️  Could not get sample: {e}")
                        
                else:
                    print("❌ PROBLEM: Dataset is empty - videos are being filtered out")
                    
            except Exception as e:
                print(f"❌ Error getting dataset length: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"❌ Error creating data loader: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Error in test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_wan_pipeline()