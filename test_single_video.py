#!/usr/bin/env python3

import os
import sys
from pathlib import Path

def test_single_video():
    """Test loading a single video file to see where it fails"""
    print("=== Single Video Loading Test ===")
    
    # Test the first video file
    video_path = Path("/workspace/input/training/cube/video_00.mp4")
    text_path = Path("/workspace/input/training/cube/video_00.txt")
    
    print(f"Testing video: {video_path}")
    print(f"Testing text: {text_path}")
    
    # Check files exist
    if not video_path.exists():
        print(f"❌ Video file does not exist: {video_path}")
        return
    
    if not text_path.exists():
        print(f"❌ Text file does not exist: {text_path}")
        return
    
    print("✅ Both files exist")
    
    # Check file sizes
    video_size = video_path.stat().st_size
    text_size = text_path.stat().st_size
    
    print(f"Video size: {video_size:,} bytes")
    print(f"Text size: {text_size} bytes")
    
    # Read text content
    with open(text_path, 'r') as f:
        text_content = f.read().strip()
    
    print(f"Text content: '{text_content}'")
    
    # Try to check video properties without importing torch
    print("\n=== Basic Video Check ===")
    
    # Check if it's a valid MP4 file by reading the header
    try:
        with open(video_path, 'rb') as f:
            header = f.read(12)
            
        # MP4 files should have 'ftyp' at bytes 4-8
        if len(header) >= 8 and header[4:8] == b'ftyp':
            print("✅ Video file has valid MP4 header")
        else:
            print("❌ Video file does not have valid MP4 header")
            print(f"Header bytes: {header}")
            
    except Exception as e:
        print(f"❌ Error reading video file: {e}")
    
    print("\n=== File Permissions Check ===")
    
    # Check file permissions
    video_readable = os.access(video_path, os.R_OK)
    text_readable = os.access(text_path, os.R_OK)
    
    print(f"Video readable: {video_readable}")
    print(f"Text readable: {text_readable}")
    
    print("\n=== Directory Structure Check ===")
    
    # Check the directory structure
    cube_dir = Path("/workspace/input/training/cube")
    print(f"Cube directory: {cube_dir}")
    print(f"Directory exists: {cube_dir.exists()}")
    print(f"Directory readable: {os.access(cube_dir, os.R_OK)}")
    
    # List all files in the directory
    try:
        all_files = list(cube_dir.iterdir())
        print(f"Total files in directory: {len(all_files)}")
        
        # Show video-text pairs
        video_files = [f for f in all_files if f.suffix.lower() == '.mp4']
        print(f"MP4 files: {len(video_files)}")
        
        valid_pairs = 0
        for vf in video_files:
            tf = vf.with_suffix('.txt')
            if tf.exists():
                valid_pairs += 1
                print(f"  ✅ {vf.name} + {tf.name}")
            else:
                print(f"  ❌ {vf.name} (no text file)")
        
        print(f"Valid video-text pairs: {valid_pairs}")
        
    except Exception as e:
        print(f"❌ Error listing directory: {e}")
    
    print("\n=== Summary ===")
    print(f"- Video file exists and is readable: {video_path.exists() and video_readable}")
    print(f"- Text file exists and is readable: {text_path.exists() and text_readable}")
    print(f"- Video has valid MP4 header: Check above")
    print(f"- Text content: '{text_content}'")
    print(f"- Should be processable by pipeline: Likely yes")

if __name__ == "__main__":
    test_single_video()