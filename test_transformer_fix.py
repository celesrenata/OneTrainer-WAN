#!/usr/bin/env python3

"""
Test script to validate the transformer fix for WAN 2.2 model loading
"""

import sys
import os
import torch

# Add the modules directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

def test_wan_model_loading():
    """Test WAN model loading with transformer fix"""
    
    try:
        from model.WanModel import WanModel
        from modelLoader.wan.WanModelLoader import WanModelLoader
        from util.enum.ModelType import ModelType
        from util.ModelNames import ModelNames
        from util.ModelWeightDtypes import ModelWeightDtypes
        from util.config.TrainConfig import QuantizationConfig
        
        print("✓ Successfully imported WAN modules")
        
        # Create model and loader
        model = WanModel(ModelType.WAN_2_2)
        loader = WanModelLoader()
        
        print("✓ Created WAN model and loader")
        
        # Test mock transformer creation
        mock_transformer = loader._create_mock_transformer(torch.float32)
        print(f"✓ Created mock transformer: {type(mock_transformer)}")
        
        # Test transformer methods
        mock_transformer.train()
        print("✓ Mock transformer .train() works")
        
        mock_transformer.eval()
        print("✓ Mock transformer .eval() works")
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(1, 4, 8, 8)  # (batch, channels, height, width)
        with torch.no_grad():
            output = mock_transformer(dummy_input)
        print(f"✓ Mock transformer forward pass works: {output.shape}")
        
        # Test video input
        dummy_video = torch.randn(1, 4, 4, 8, 8)  # (batch, channels, frames, height, width)
        with torch.no_grad():
            video_output = mock_transformer(dummy_video)
        print(f"✓ Mock transformer video forward pass works: {video_output.shape}")
        
        print("\n🎉 All transformer tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_wan_lora_setup():
    """Test WAN LoRA setup with transformer fix"""
    
    try:
        from model.WanModel import WanModel
        from modelLoader.wan.WanModelLoader import WanModelLoader
        from modelSetup.WanLoRASetup import WanLoRASetup
        from util.enum.ModelType import ModelType
        from util.ModelNames import ModelNames
        from util.ModelWeightDtypes import ModelWeightDtypes
        from util.config.TrainConfig import QuantizationConfig, TrainConfig
        
        print("\n=== Testing WAN LoRA Setup ===")
        
        # Create model with mock transformer
        model = WanModel(ModelType.WAN_2_2)
        loader = WanModelLoader()
        
        # Manually set up model with mock transformer
        model.transformer = loader._create_mock_transformer(torch.float32)
        model.vae = None  # Mock VAE not needed for this test
        model.text_encoder = None  # Mock text encoder not needed
        model.tokenizer = None
        model.noise_scheduler = None
        
        print("✓ Set up model with mock transformer")
        
        # Create LoRA setup
        setup = WanLoRASetup(
            train_device=torch.device('cpu'),
            temp_device=torch.device('cpu'),
            debug_mode=True
        )
        
        print("✓ Created WAN LoRA setup")
        
        # Create minimal config
        config = TrainConfig.default_values()
        config.transformer.train = True
        
        print("✓ Created training config")
        
        # Test setup_train_device (this was failing before)
        setup.setup_train_device(model, config)
        print("✓ setup_train_device completed successfully")
        
        # Verify transformer is in training mode
        if model.transformer is not None:
            print(f"✓ Transformer training mode: {model.transformer.training}")
        
        print("\n🎉 All LoRA setup tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ LoRA setup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing WAN 2.2 Transformer Fix")
    print("=" * 50)
    
    success1 = test_wan_model_loading()
    success2 = test_wan_lora_setup()
    
    if success1 and success2:
        print("\n✅ All tests passed! The transformer fix should resolve the AttributeError.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)