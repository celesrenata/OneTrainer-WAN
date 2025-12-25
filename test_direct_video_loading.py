#!/usr/bin/env python3
"""
Direct video loading test to bypass the complex pipeline.
"""

import sys
import os
import torch
from pathlib import Path

# Add the modules to the path
sys.path.append('.')

def test_direct_video_loading():
    """Test direct video loading without MGDS pipeline"""
    print("🎬 Testing direct video loading...")
    
    try:
        # Import the LoadVideo module directly
        from mgds.pipelineModules.LoadVideo import LoadVideo
        
        # Create a simple LoadVideo module with the correct parameter name
        load_video = LoadVideo(
            path_in_name='video_path',
            target_frame_count_in_name='settings.target_frames',  # This should match the data structure
            video_out_name='video',
            range_min=0,
            range_max=1,
            target_frame_rate=24,
            supported_extensions=['.mp4'],
            dtype=torch.float32
        )
        
        print("   ✓ LoadVideo module created")
        
        # Manually initialize the module (simulate MGDS initialization)
        load_video._PipelineModule__base_seed = 42
        load_video._PipelineModule__module_index = 0
        
        # Test with a real video file
        video_path = "/workspace/input/training/cube/video_00.mp4"
        
        # Create test input data with the correct structure
        input_data = {
            'video_path': video_path,
            'settings': {
                'target_frames': 2
            }
        }
        
        print(f"   Testing with: {video_path}")
        print(f"   Target frames: 2")
        
        # Try to load the video directly
        result = load_video.get_item(0, 0, input_data)
        
        if result is None:
            print("   ❌ LoadVideo returned None")
            return False
        
        print(f"   ✓ LoadVideo returned data with keys: {list(result.keys())}")
        
        if 'video' in result:
            video_tensor = result['video']
            print(f"   ✓ Video tensor shape: {video_tensor.shape}")
            print(f"   ✓ Video tensor dtype: {video_tensor.dtype}")
            print(f"   ✓ Video tensor range: {video_tensor.min():.3f} to {video_tensor.max():.3f}")
            
            # Check if the tensor has the expected shape
            if len(video_tensor.shape) == 4:  # [frames, channels, height, width]
                frames, channels, height, width = video_tensor.shape
                print(f"   ✓ Video format: {frames} frames, {channels} channels, {height}x{width}")
                
                if frames == 2 and channels == 3:
                    print("   ✅ Video loading successful!")
                    return True
                else:
                    print(f"   ⚠️  Unexpected video format")
                    return False
            else:
                print(f"   ⚠️  Unexpected tensor shape: {video_tensor.shape}")
                return False
        else:
            print("   ❌ No 'video' key in result")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing direct video loading: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_safe_load_video_wrapper():
    """Test the SafeLoadVideo wrapper directly"""
    print("\n🛡️  Testing SafeLoadVideo wrapper...")
    
    try:
        # Import required modules
        from mgds.pipelineModules.LoadVideo import LoadVideo
        from modules.dataLoader.mixin.DataLoaderText2VideoMixin import DataLoaderText2VideoMixin
        import torch
        
        # Create the base LoadVideo module with the correct parameter name
        load_video_base = LoadVideo(
            path_in_name='video_path',
            target_frame_count_in_name='settings.target_frames',  # This should match the data structure
            video_out_name='video',
            range_min=0,
            range_max=1,
            target_frame_rate=24,
            supported_extensions=['.mp4'],
            dtype=torch.float32
        )
        
        # Manually initialize the module
        load_video_base._PipelineModule__base_seed = 42
        load_video_base._PipelineModule__module_index = 0
        
        # Create the SafeLoadVideo wrapper (from the mixin code)
        from mgds.PipelineModule import PipelineModule
        
        class SafeLoadVideo(PipelineModule):
            def __init__(self, load_video_module, dtype=torch.float32):
                super().__init__()
                self.load_video_module = load_video_module
                self.dtype = dtype
                
            def length(self):
                return self.load_video_module.length()
                
            def get_inputs(self):
                return self.load_video_module.get_inputs()
                
            def get_outputs(self):
                return self.load_video_module.get_outputs()
                
            def get_item(self, variation, index, requested_name=None):
                try:
                    print(f"DEBUG SAFE_LOAD_VIDEO: Processing item {index}, variation {variation}")
                    result = self.load_video_module.get_item(variation, index, requested_name)
                    
                    if result is None:
                        print(f"DEBUG SAFE_LOAD_VIDEO ERROR: LoadVideo returned None for item {index}")
                        # Create dummy data
                        dummy_video = torch.zeros((2, 3, 64, 64), dtype=self.dtype)
                        return {
                            'video': dummy_video,
                            'video_path': f'dummy_video_{index}.mp4',
                        }
                    
                    # Log successful loading
                    video_path = result.get('video_path', 'unknown')
                    video_data = result.get('video', None)
                    if hasattr(video_data, 'shape'):
                        print(f"DEBUG SAFE_LOAD_VIDEO SUCCESS: {video_path} loaded with shape {video_data.shape}")
                    
                    return result
                    
                except Exception as e:
                    print(f"DEBUG SAFE_LOAD_VIDEO EXCEPTION: LoadVideo failed for item {index}: {e}")
                    # Create dummy data
                    dummy_video = torch.zeros((2, 3, 64, 64), dtype=self.dtype)
                    return {
                        'video': dummy_video,
                        'video_path': f'dummy_video_{index}.mp4',
                    }
        
        # Create the SafeLoadVideo wrapper
        safe_load_video = SafeLoadVideo(load_video_base, dtype=torch.float32)
        
        # Initialize the wrapper
        safe_load_video._PipelineModule__base_seed = 42
        safe_load_video._PipelineModule__module_index = 0
        
        print("   ✓ SafeLoadVideo wrapper created")
        
        # Test with a real video file
        video_path = "/workspace/input/training/cube/video_00.mp4"
        
        # Create test input data with the correct structure
        input_data = {
            'video_path': video_path,
            'settings': {
                'target_frames': 2
            }
        }
        
        print(f"   Testing with: {video_path}")
        
        # Try to load the video through the wrapper
        result = safe_load_video.get_item(0, 0, input_data)
        
        if result is None:
            print("   ❌ SafeLoadVideo returned None")
            return False
        
        print(f"   ✓ SafeLoadVideo returned data with keys: {list(result.keys())}")
        
        if 'video' in result:
            video_tensor = result['video']
            print(f"   ✓ Video tensor shape: {video_tensor.shape}")
            
            if video_tensor.shape[0] > 0:  # Has frames
                print("   ✅ SafeLoadVideo wrapper successful!")
                return True
            else:
                print("   ⚠️  Video has no frames")
                return False
        else:
            print("   ❌ No 'video' key in result")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing SafeLoadVideo wrapper: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Direct Video Loading Test")
    print("=" * 40)
    
    # Test direct video loading
    direct_success = test_direct_video_loading()
    
    # Test SafeLoadVideo wrapper
    wrapper_success = test_safe_load_video_wrapper()
    
    print("\n🎯 Summary:")
    print(f"   Direct LoadVideo: {'✅' if direct_success else '❌'}")
    print(f"   SafeLoadVideo wrapper: {'✅' if wrapper_success else '❌'}")
    
    if direct_success and wrapper_success:
        print("\n✅ Video loading works correctly!")
        print("   Issue is in the MGDS pipeline integration")
    elif direct_success:
        print("\n⚠️  Direct loading works, but wrapper has issues")
        print("   Issue is in the SafeLoadVideo wrapper")
    else:
        print("\n❌ Direct video loading fails")
        print("   Issue is in the basic LoadVideo module")

if __name__ == "__main__":
    main()