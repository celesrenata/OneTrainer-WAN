
import sys
import os
sys.path.append('.')

try:
    from modules.dataLoader.mixin.DataLoaderText2VideoMixin import DataLoaderText2VideoMixin
    print("✓ Successfully imported DataLoaderText2VideoMixin")
    
    # Try to create an instance (this will test the debug logging)
    print("Testing video validation modules...")
    
    # Create a mock config
    class MockConfig:
        def __init__(self):
            self.masked_training = False
            self.model_type = MockModelType()
            self.custom_conditioning_image = False
    
    class MockModelType:
        def has_mask_input(self):
            return False
        def has_depth_input(self):
            return False
    
    # Create mixin instance
    mixin = DataLoaderText2VideoMixin()
    config = MockConfig()
    
    # Test video validation modules
    validation_modules = mixin._video_validation_modules(config)
    print(f"✓ Video validation modules: {len(validation_modules)}")
    
    print("✅ Debug logging test completed successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
