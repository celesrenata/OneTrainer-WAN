#!/usr/bin/env python3
"""
Minimal pipeline test to isolate the video loading issue.
"""

import sys
import os
import json
from pathlib import Path

# Add the modules to the path
sys.path.append('.')

def test_minimal_video_pipeline():
    """Test a minimal video pipeline to isolate the issue"""
    print("🧪 Testing minimal video pipeline...")
    
    try:
        # Import required modules
        from mgds.pipelineModules.CollectPaths import CollectPaths
        from mgds.pipelineModules.LoadVideo import LoadVideo
        from mgds.OutputPipelineModule import OutputPipelineModule
        from mgds.PipelineModule import PipelineModule
        import torch
        
        print("   ✓ Imports successful")
        
        # Create a simple pipeline with just CollectPaths -> LoadVideo
        print("\n   Creating CollectPaths module...")
        collect_paths = CollectPaths(
            concept_in_name='concept',
            path_in_name='path',
            include_subdirectories_in_name='include_subdirectories',
            enabled_in_name='enabled',
            extensions=['.mp4'],
            concept_out_name='concept',
            path_out_name='video_path',
            include_postfix=None,
            exclude_postfix=[]
        )
        
        print("   Creating LoadVideo module...")
        load_video = LoadVideo(
            path_in_name='video_path',
            target_frame_count_in_name='target_frames',
            video_out_name='video',
            range_min=0,
            range_max=1,
            target_frame_rate=24,
            supported_extensions=['.mp4'],
            dtype=torch.float32
        )
        
        print("   Creating output module...")
        output_module = OutputPipelineModule(['video', 'video_path', 'concept'])
        
        # Create the pipeline
        print("\n   Creating MGDS pipeline...")
        from mgds.MGDS import MGDS
        from mgds.PipelineModule import PipelineState
        import torch
        
        # Create concepts data
        concepts = [{
            'name': 'Cube',
            'path': '/workspace/input/training/cube',
            'enabled': True,
            'include_subdirectories': False
        }]
        
        # Create settings
        settings = {
            "target_resolution": 384,
            "target_frames": 4,
        }
        
        # Create module definition (list of module groups)
        definition = [
            [collect_paths],  # Group 0: Find files
            [load_video],     # Group 1: Load videos
            [output_module]   # Group 2: Output
        ]
        
        # Create the pipeline
        mgds = MGDS(
            torch.device('cpu'),  # Use CPU for testing
            concepts,
            settings,
            definition,
            batch_size=1,
            state=PipelineState(1),  # 1 thread
            initial_epoch=0,
            initial_epoch_sample=0,
        )
        
        print(f"   ✓ MGDS pipeline created")
        
        # Try to get first item directly
        print("\n   Testing first item...")
        try:
            item = mgds[0]
            print(f"   ✓ Got item: {type(item)}")
            if isinstance(item, dict):
                print(f"   Item keys: {list(item.keys())}")
                if 'video' in item:
                    video = item['video']
                    if hasattr(video, 'shape'):
                        print(f"   Video shape: {video.shape}")
                    else:
                        print(f"   Video type: {type(video)}")
            else:
                print(f"   Item is not a dict: {item}")
            return True
        except Exception as e:
            print(f"   ❌ Error getting first item: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"   ❌ Error creating minimal pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Minimal Pipeline Test")
    print("=" * 30)
    
    success = test_minimal_video_pipeline()
    
    if success:
        print("\n✅ Minimal pipeline works")
        print("   Issue is in the full training pipeline complexity")
    else:
        print("\n❌ Minimal pipeline fails")
        print("   Issue is in basic video loading")

if __name__ == "__main__":
    main()