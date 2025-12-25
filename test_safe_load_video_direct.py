#!/usr/bin/env python3
"""
Direct test of SafeLoadVideo to see if it's working
"""

import sys
import os
import torch
from pathlib import Path

# Add the modules to the path
sys.path.append('.')

def test_safe_load_video_direct():
    """Test SafeLoadVideo directly with a real video file"""
    print("🎬 Testing SafeLoadVideo directly...")
    
    try:
        # Import the LoadVideo module
        from mgds.pipelineModules.LoadVideo import LoadVideo
        from modules.util.enum.DataType import DataType
        
        # Create LoadVideo module
        load_video = LoadVideo(
            path_in_name='video_path',
            target_frame_count_in_name='target_frames',
            video_out_name='video',
            range_min=0,
            range_max=1,
            target_frame_rate=24,
            supported_extensions=['.mp4', '.avi', '.mov', '.webm', '.mkv'],
            dtype=torch.float32
        )
        
        print("   ✓ LoadVideo module created")
        
        # Test with a real video file
        video_path = "/workspace/input/training/cube/video_00.mp4"
        
        # Create test data
        test_data = {
            'video_path': video_path,
            'target_frames': 16
        }
        
        print(f"   Testing with: {video_path}")
        print(f"   Target frames: 16")
        
        # Try to load the video
        result = load_video.get_item(0, 0, test_data)
        
        if result is None:
            print("   ❌ LoadVideo returned None")
            return False
        
        print(f"   ✓ LoadVideo returned data with keys: {list(result.keys())}")
        
        if 'video' in result:
            video_tensor = result['video']
            print(f"   ✓ Video tensor shape: {video_tensor.shape}")
            print(f"   ✓ Video tensor dtype: {video_tensor.dtype}")
            print(f"   ✓ Video tensor range: {video_tensor.min():.3f} to {video_tensor.max():.3f}")
        else:
            print("   ❌ No 'video' key in result")
            return False
            
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing LoadVideo: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Direct SafeLoadVideo Test")
    print("=" * 30)
    
    success = test_safe_load_video_direct()
    
    if success:
        print("\n✅ SafeLoadVideo works correctly")
        print("   Issue is likely in pipeline integration or data flow")
    else:
        print("\n❌ SafeLoadVideo has issues")
        print("   Need to fix video loading first")

if __name__ == "__main__":
    main()