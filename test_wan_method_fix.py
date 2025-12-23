#!/usr/bin/env python3

"""
Test script to validate the WAN method name and attribute fixes
"""

import sys
import os

# Add the modules directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

def test_wan_model_attributes():
    """Test that WanModel has all required attributes"""
    
    try:
        from model.WanModel import WanModel
        from util.enum.ModelType import ModelType
        
        # Create model
        model = WanModel(ModelType.WAN_2_2)
        print("✓ Created WAN model successfully")
        
        # Check for required attributes
        required_attrs = [
            'add_text_encoder_embeddings_to_prompt',
            'autocast_context',
            'train_dtype',
            'text_encoder_autocast_context',
            'transformer_autocast_context',
            'text_encoder_train_dtype',
            'transformer_train_dtype'
        ]
        
        for attr in required_attrs:
            if hasattr(model, attr):
                print(f"✓ {attr} attribute exists")
            else:
                print(f"❌ {attr} attribute missing")
                return False
        
        # Test the method call
        try:
            result = model.add_text_encoder_embeddings_to_prompt("test prompt")
            print(f"✓ add_text_encoder_embeddings_to_prompt method works: '{result}'")
        except Exception as e:
            print(f"❌ add_text_encoder_embeddings_to_prompt method failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ WAN model attributes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loader_compatibility():
    """Test that the data loader can access model attributes"""
    
    try:
        from model.WanModel import WanModel
        from util.enum.ModelType import ModelType
        
        # Create model
        model = WanModel(ModelType.WAN_2_2)
        print("✓ Created WAN model for data loader test")
        
        # Test accessing attributes that data loader needs
        try:
            # Test method access
            method = model.add_text_encoder_embeddings_to_prompt
            print("✓ Data loader can access add_text_encoder_embeddings_to_prompt method")
            
            # Test attribute access
            autocast_ctx = model.autocast_context
            print("✓ Data loader can access autocast_context attribute")
            
            train_dtype = model.train_dtype
            print("✓ Data loader can access train_dtype attribute")
            
            # Test dtype method call
            if hasattr(train_dtype, 'torch_dtype'):
                dtype_result = train_dtype.torch_dtype()
                print(f"✓ train_dtype.torch_dtype() works: {dtype_result}")
            
            return True
            
        except AttributeError as e:
            print(f"❌ Data loader compatibility test failed: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Data loader compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fix_summary():
    """Summarize the fixes applied"""
    
    print("\nFix Summary:")
    print("=" * 40)
    
    fixes = [
        "Fixed method name: add_embeddings_to_prompt → add_text_encoder_embeddings_to_prompt",
        "Added missing autocast_context attribute to WanModel.__init__",
        "Added missing train_dtype attribute to WanModel.__init__",
        "Ensured data loader compatibility with model attributes"
    ]
    
    for fix in fixes:
        print(f"✓ {fix}")
    
    return True

if __name__ == "__main__":
    print("Testing WAN Method Name and Attribute Fixes")
    print("=" * 50)
    
    success1 = test_wan_model_attributes()
    success2 = test_data_loader_compatibility()
    success3 = test_fix_summary()
    
    if success1 and success2 and success3:
        print("\n🎉 All method and attribute fix tests passed!")
        print("\nThe AttributeError should now be resolved:")
        print("❌ Before: 'WanModel' object has no attribute 'add_embeddings_to_prompt'")
        print("✅ After: Method name corrected and all attributes available")
        
        sys.exit(0)
    else:
        print("\n❌ Some method and attribute fix tests failed.")
        sys.exit(1)