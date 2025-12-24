#!/usr/bin/env python3
"""
Test WAN 2.2 data loading to see if the pipeline can find and process the training data
"""

import sys
import os
import json
from pathlib import Path

# Add the modules directory to the path
sys.path.insert(0, 'modules')

def test_data_loading():
    """Test if the data loading pipeline can find and process training data"""
    
    print("=== WAN 2.2 Data Loading Test ===\n")
    
    # Load concepts
    concepts_file = "training_concepts/concepts.json"
    with open(concepts_file, 'r') as f:
        concepts = json.load(f)
    
    print(f"Loaded {len(concepts)} concepts")
    
    for concept in concepts:
        name = concept.get('name')
        path = concept.get('path')
        enabled = concept.get('enabled', False)
        
        print(f"\nConcept: {name}")
        print(f"Path: {path}")
        print(f"Enabled: {enabled}")
        
        if not enabled:
            print("⚠️  Concept disabled, skipping")
            continue
            
        if not os.path.exists(path):
            print("❌ Path does not exist")
            continue
            
        # Count files
        path_obj = Path(path)
        all_files = list(path_obj.iterdir())
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v', '.wmv', '.mpeg'}
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.avif'}
        
        video_files = [f for f in all_files if f.is_file() and f.suffix.lower() in video_extensions]
        image_files = [f for f in all_files if f.is_file() and f.suffix.lower() in image_extensions]
        text_files = [f for f in all_files if f.is_file() and f.suffix.lower() == '.txt']
        
        print(f"📁 Total files: {len(all_files)}")
        print(f"🎥 Video files: {len(video_files)}")
        print(f"🖼️  Image files: {len(image_files)}")
        print(f"📝 Text files: {len(text_files)}")
        
        # Show some examples
        if video_files:
            print("   Video examples:")
            for vf in video_files[:3]:
                size = vf.stat().st_size
                print(f"     - {vf.name} ({size:,} bytes)")
                
        if image_files:
            print("   Image examples:")
            for img in image_files[:3]:
                size = img.stat().st_size
                print(f"     - {img.name} ({size:,} bytes)")
        
        total_media = len(video_files) + len(image_files)
        if total_media > 0:
            print(f"✅ Found {total_media} media files for training")
        else:
            print("❌ No media files found")

def test_mgds_import():
    """Test if MGDS modules can be imported"""
    print(f"\n=== Testing MGDS Imports ===")
    
    try:
        from mgds.MGDS import MGDS
        print("✅ MGDS imported successfully")
    except Exception as e:
        print(f"❌ MGDS import failed: {e}")
        return False
        
    try:
        from mgds.pipelineModules.CollectPaths import CollectPaths
        print("✅ CollectPaths imported successfully")
    except Exception as e:
        print(f"❌ CollectPaths import failed: {e}")
        return False
        
    return True

def test_wan_imports():
    """Test if WAN modules can be imported"""
    print(f"\n=== Testing WAN Module Imports ===")
    
    try:
        from modules.dataLoader.WanBaseDataLoader import WanBaseDataLoader
        print("✅ WanBaseDataLoader imported successfully")
    except Exception as e:
        print(f"❌ WanBaseDataLoader import failed: {e}")
        return False
        
    try:
        from modules.util.config.TrainConfig import TrainConfig
        print("✅ TrainConfig imported successfully")
    except Exception as e:
        print(f"❌ TrainConfig import failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    print("Testing WAN 2.2 data loading pipeline...\n")
    
    try:
        # Test data availability
        test_data_loading()
        
        # Test imports
        mgds_ok = test_mgds_import()
        wan_ok = test_wan_imports()
        
        print(f"\n=== Summary ===")
        print(f"MGDS imports: {'✅ OK' if mgds_ok else '❌ Failed'}")
        print(f"WAN imports: {'✅ OK' if wan_ok else '❌ Failed'}")
        
        if mgds_ok and wan_ok:
            print(f"🎉 All systems ready for training!")
        else:
            print(f"⚠️  Some issues detected")
            
    except Exception as e:
        print(f"💥 Error during testing: {e}")
        import traceback
        traceback.print_exc()