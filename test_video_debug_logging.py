#!/usr/bin/env python3
"""
Test script to verify video debug logging is working.

This script tests the video loading pipeline with debug logging enabled.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_video_debug_logging():
    """Test that video debug logging is working"""
    print("🧪 Testing Video Debug Logging")
    print("=" * 35)
    
    # Check if debug logging was added
    debug_file = "modules/dataLoader/mixin/DataLoaderText2VideoMixin.py"
    
    if not os.path.exists(debug_file):
        print(f"❌ File not found: {debug_file}")
        return False
    
    try:
        with open(debug_file, 'r') as f:
            content = f.read()
        
        # Check for debug logging
        debug_indicators = [
            "DEBUG SAFE_LOAD_VIDEO:",
            "DEBUG VIDEO VALIDATION:",
            "Video validation enabled"
        ]
        
        found_debug = []
        for indicator in debug_indicators:
            if indicator in content:
                found_debug.append(indicator)
                print(f"✓ Found: {indicator}")
            else:
                print(f"❌ Missing: {indicator}")
        
        if len(found_debug) >= 2:
            print(f"\n✅ Debug logging appears to be enabled!")
            print(f"   Found {len(found_debug)}/{len(debug_indicators)} debug indicators")
        else:
            print(f"\n⚠️  Debug logging may not be fully enabled")
            print(f"   Found only {len(found_debug)}/{len(debug_indicators)} debug indicators")
        
        # Check if validation is no longer disabled
        if "Video validation temporarily disabled" in content:
            print(f"⚠️  Video validation still shows as disabled")
        else:
            print(f"✓ Video validation disable message removed")
        
        return len(found_debug) >= 2
        
    except Exception as e:
        print(f"❌ Error checking debug logging: {e}")
        return False

def create_minimal_test_config():
    """Create a minimal test configuration"""
    print(f"\n📝 Creating minimal test configuration...")
    
    # Create a simple test that just imports the module
    test_code = '''
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
'''
    
    with open("test_debug_import.py", 'w') as f:
        f.write(test_code)
    
    print(f"✓ Created test_debug_import.py")

def main():
    """Main test function"""
    success = test_video_debug_logging()
    create_minimal_test_config()
    
    print(f"\n🎯 Summary:")
    if success:
        print(f"✅ Video debug logging is enabled")
        print(f"📋 Next steps:")
        print(f"1. Run training again to see detailed debug output")
        print(f"2. Look for 'DEBUG SAFE_LOAD_VIDEO:' and 'DEBUG VIDEO VALIDATION:' messages")
        print(f"3. The debug output will show exactly why videos are being rejected")
    else:
        print(f"❌ Video debug logging may not be working")
        print(f"📋 Manual steps needed:")
        print(f"1. Check modules/dataLoader/mixin/DataLoaderText2VideoMixin.py")
        print(f"2. Ensure debug logging code was added correctly")
    
    print(f"\n🧪 Test import:")
    print(f"python test_debug_import.py")

if __name__ == "__main__":
    main()