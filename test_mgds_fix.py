#!/usr/bin/env python3

"""
Test script to validate the MGDS FilterByFunction fix
"""

import sys
import os

# Add the modules directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

def test_mgds_import_fix():
    """Test that the MGDS import issue is fixed"""
    
    try:
        # Test the import that was failing
        from dataLoader.mixin.DataLoaderText2VideoMixin import DataLoaderText2VideoMixin
        print("✓ DataLoaderText2VideoMixin import successful")
        
        # Test creating an instance
        mixin = DataLoaderText2VideoMixin()
        print("✓ DataLoaderText2VideoMixin instance created")
        
        # Test the video validation modules method
        class MockConfig:
            validate_video_files = False
        
        config = MockConfig()
        modules = mixin._video_validation_modules(config)
        print(f"✓ _video_validation_modules returned {len(modules)} modules")
        
        # Test with validation enabled
        config.validate_video_files = True
        modules = mixin._video_validation_modules(config)
        print(f"✓ _video_validation_modules with validation returned {len(modules)} modules")
        
        return True
        
    except Exception as e:
        print(f"❌ MGDS import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_video_util_import():
    """Test that video utility functions are available"""
    
    try:
        from util.video_util import validate_video_file
        print("✓ validate_video_file import successful")
        
        # Test with a non-existent file (should return False, error message)
        is_valid, error_msg = validate_video_file("/nonexistent/video.mp4")
        print(f"✓ validate_video_file works: valid={is_valid}, error='{error_msg}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Video util import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing MGDS FilterByFunction Fix")
    print("=" * 40)
    
    success1 = test_mgds_import_fix()
    success2 = test_video_util_import()
    
    if success1 and success2:
        print("\n🎉 All MGDS fix tests passed!")
        print("\nSummary of fix:")
        print("- Replaced missing FilterByFunction with custom VideoValidationModule")
        print("- Custom module extends PipelineModule and validates video files")
        print("- Only applies validation when config.validate_video_files is True")
        print("- Maintains compatibility with existing MGDS pipeline")
        
        sys.exit(0)
    else:
        print("\n❌ Some MGDS fix tests failed.")
        sys.exit(1)