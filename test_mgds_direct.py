#!/usr/bin/env python3
"""
Direct test of MGDS pipeline to see what's failing
"""

import sys
import os
import json
import torch
from pathlib import Path

# Add the modules to the path
sys.path.append('.')

def test_mgds_pipeline_direct():
    """Test MGDS pipeline directly to isolate the issue"""
    print("🧪 Testing MGDS pipeline directly...")
    
    try:
        from mgds.MGDS import MGDS
        from mgds.PipelineModule import PipelineState
        from mgds.pipelineModules.CollectPaths import CollectPaths
        from mgds.OutputPipelineModule import OutputPipelineModule
        
        print("   ✓ MGDS imports successful")
        
        # Debug: Check if the directory exists and has files
        cube_path = "/workspace/input/training/cube"
        print(f"   Checking directory: {cube_path}")
        if os.path.exists(cube_path):
            files = list(Path(cube_path).glob("*"))
            mp4_files = [f for f in files if f.suffix.lower() == '.mp4']
            print(f"   Directory exists: {len(files)} total files, {len(mp4_files)} MP4 files")
            if mp4_files:
                print(f"   MP4 files: {[f.name for f in mp4_files[:3]]}")
        else:
            print(f"   ❌ Directory does not exist!")
        
        # Create the pipeline modules in the correct order
        print("   Creating pipeline modules...")
        
        from mgds.pipelineModules.DownloadHuggingfaceDatasets import DownloadHuggingfaceDatasets
        
        download_datasets = DownloadHuggingfaceDatasets(
            concept_in_name='concept', 
            path_in_name='path', 
            enabled_in_name='enabled',
            concept_out_name='concept',
        )
        
        collect_paths = CollectPaths(
            concept_in_name='concept', 
            path_in_name='path', 
            include_subdirectories_in_name='include_subdirectories', 
            enabled_in_name='enabled',
            path_out_name='video_path', 
            concept_out_name='concept',
            extensions=['.mp4'], 
            include_postfix=None, 
            exclude_postfix=[]
        )
        
        print(f"   Created DownloadHuggingfaceDatasets and CollectPaths")
        
        output_module = OutputPipelineModule(['video_path', 'concept'])
        
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
            "target_frames": 2,
        }
        
        # Create module definition with the correct module order
        definition = [
            [download_datasets, collect_paths, output_module]  # Single group with all modules
        ]
        
        print("   Creating MGDS with minimal pipeline...")
        
        # Debug: Print concepts data
        print(f"   Concepts data: {concepts}")
        
        # Create the pipeline
        mgds = MGDS(
            torch.device('cpu'),
            concepts,
            settings,
            definition,
            batch_size=1,
            state=PipelineState(1),
            initial_epoch=0,
            initial_epoch_sample=0,
        )
        
        print("   ✓ MGDS pipeline created")
        
        # Debug: Check if CollectPaths has any paths after MGDS initialization
        try:
            collect_paths_module = definition[0][1]  # Second module in first group (CollectPaths)
            
            # Don't manually initialize - let MGDS handle it properly
            print("   Checking CollectPaths after MGDS initialization...")
            
            # Check paths
            if hasattr(collect_paths_module, 'paths'):
                print(f"   CollectPaths.paths length after MGDS init: {len(collect_paths_module.paths)}")
                if len(collect_paths_module.paths) > 0:
                    print(f"   First few paths: {collect_paths_module.paths[:3]}")
                else:
                    print("   ❌ CollectPaths.paths still empty after MGDS init!")
                    
                    # Debug CollectPaths internals
                    print("   Debugging CollectPaths internals...")
                    if hasattr(collect_paths_module, 'concept_in_name'):
                        print(f"   concept_in_name: {collect_paths_module.concept_in_name}")
                    if hasattr(collect_paths_module, 'path_in_name'):
                        print(f"   path_in_name: {collect_paths_module.path_in_name}")
                    if hasattr(collect_paths_module, 'extensions'):
                        print(f"   extensions: {collect_paths_module.extensions}")
            else:
                print("   CollectPaths doesn't have 'paths' attribute yet")
        except Exception as e:
            print(f"   Error checking CollectPaths: {e}")
            import traceback
            traceback.print_exc()
        
        # Try to get first item
        print("   Testing first item...")
        try:
            # Start the first epoch explicitly
            print("   Starting first epoch...")
            mgds.start_next_epoch()
            
            # Try to iterate through the dataset
            for i, item in enumerate(mgds):
                print(f"   ✓ Got item {i}: {type(item)}")
                if isinstance(item, dict):
                    print(f"   Item keys: {list(item.keys())}")
                else:
                    print(f"   Item: {item}")
                
                if i >= 2:  # Only test first 3 items
                    break
                    
            return True
            
        except Exception as e:
            print(f"   ❌ Error getting items: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"   ❌ Error creating MGDS pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Direct MGDS Pipeline Test")
    print("=" * 40)
    
    success = test_mgds_pipeline_direct()
    
    if success:
        print("\n✅ MGDS pipeline works with minimal setup")
        print("   Issue is in the complex training pipeline")
    else:
        print("\n❌ MGDS pipeline fails even with minimal setup")
        print("   Issue is in basic MGDS functionality")

if __name__ == "__main__":
    main()