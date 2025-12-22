#!/usr/bin/env python3
"""
Test WAN 2.2 factory function integration fix.
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

def test_wan_factory_functions():
    """Test that WAN 2.2 factory functions work correctly."""
    print("🔧 Testing WAN 2.2 Factory Function Fix")
    print("=" * 50)
    
    try:
        from modules.util.enum.ModelType import ModelType
        from modules.util.enum.DataType import DataType
        from modules.util.create import create_model_loader, create_model_saver, create_model_setup
        
        print("✅ Imports successful")
        
        # Test model loader factory
        print("\n1. Testing Model Loader Factory...")
        loader = create_model_loader(ModelType.WAN_2_2, train_dtype=DataType.FLOAT_32)
        if loader is not None:
            print(f"✅ Fine-tune loader: {type(loader).__name__}")
        else:
            print("❌ Fine-tune loader returned None")
            return False
        
        # Test model saver factory
        print("\n2. Testing Model Saver Factory...")
        saver = create_model_saver(ModelType.WAN_2_2)
        if saver is not None:
            print(f"✅ Fine-tune saver: {type(saver).__name__}")
        else:
            print("❌ Fine-tune saver returned None")
            return False
        
        # Test model setup factory (requires torch device)
        print("\n3. Testing Model Setup Factory...")
        try:
            import torch
            device = torch.device('cpu')
            setup = create_model_setup(ModelType.WAN_2_2, device, device)
            if setup is not None:
                print(f"✅ Fine-tune setup: {type(setup).__name__}")
            else:
                print("❌ Fine-tune setup returned None")
                return False
        except ImportError:
            print("⚠ PyTorch not available, skipping setup test")
        
        print("\n🎉 All factory functions working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Factory function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_wan_factory_functions()
    if success:
        print("\n✅ WAN 2.2 factory function fix is working!")
        print("The 'NoneType' object has no attribute 'load' error should be resolved.")
    else:
        print("\n❌ Factory function fix needs more work.")
    
    sys.exit(0 if success else 1)