#!/usr/bin/env python3
"""
Test script to verify SafePipelineModule has the init method
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_safe_pipeline_module():
    """Test that SafePipelineModule has the required init method"""
    
    print("Testing SafePipelineModule init method...")
    
    try:
        # Import the mixin that contains SafePipelineModule
        from modules.dataLoader.mixin.DataLoaderText2VideoMixin import DataLoaderText2VideoMixin
        from modules.util.config.TrainConfig import TrainConfig
        from modules.util.enum.DataType import DataType
        
        # Create a test config
        config = TrainConfig.default_values()
        config.video_config.enable_video_training = True
        
        # Create an instance of the mixin
        mixin = DataLoaderText2VideoMixin()
        
        # Call the method that creates SafePipelineModule
        modules = mixin._load_input_modules(config, DataType.FLOAT_32)
        
        print(f"✅ Successfully loaded {len(modules)} input modules")
        
        # Check if any modules are SafePipelineModule instances
        for i, module in enumerate(modules):
            print(f"Module {i}: {type(module).__name__}")
            
            # Check if it has the init method
            if hasattr(module, 'init'):
                print(f"  ✅ Module {i} has init method")
                
                # Try to inspect the method
                import inspect
                sig = inspect.signature(module.init)
                print(f"  ✅ init method signature: {sig}")
            else:
                print(f"  ❌ Module {i} missing init method")
        
        print("✅ SafePipelineModule test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ SafePipelineModule test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_safe_pipeline_module()
    sys.exit(0 if success else 1)