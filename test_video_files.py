#!/usr/bin/env python3

import os
from pathlib import Path

def test_video_files():
    """Test the video files to see if they're valid"""
    print("=== Video File Validation Test ===")
    
    data_path = Path("/workspace/input/training/cube")
    
    if not data_path.exists():
        print(f"❌ Data path does not exist: {data_path}")
        return
    
    print(f"✅ Data path exists: {data_path}")
    
    # Get all files
    all_files = list(data_path.iterdir())
    video_files = [f for f in all_files if f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']]
    text_files = [f for f in all_files if f.suffix.lower() == '.txt']
    
    print(f"Found {len(all_files)} total files:")
    print(f"  - {len(video_files)} video files")
    print(f"  - {len(text_files)} text files")
    print(f"  - {len(all_files) - len(video_files) - len(text_files)} other files")
    
    print("\nVideo files:")
    for i, vf in enumerate(video_files):
        size = vf.stat().st_size
        print(f"  {i+1}. {vf.name} ({size:,} bytes)")
        
        # Check if there's a corresponding text file
        text_file = vf.with_suffix('.txt')
        if text_file.exists():
            text_size = text_file.stat().st_size
            print(f"      -> {text_file.name} ({text_size} bytes)")
            
            # Read text content
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    print(f"      -> Text: '{content[:50]}{'...' if len(content) > 50 else ''}'")
            except Exception as e:
                print(f"      -> Error reading text: {e}")
        else:
            print(f"      -> ❌ No corresponding text file: {text_file.name}")
    
    print("\nText files:")
    for i, tf in enumerate(text_files):
        size = tf.stat().st_size
        print(f"  {i+1}. {tf.name} ({size} bytes)")
        
        try:
            with open(tf, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                print(f"      Content: '{content}'")
        except Exception as e:
            print(f"      Error reading: {e}")
    
    # Check for potential issues
    print("\n=== Potential Issues ===")
    
    issues_found = False
    
    # Check for missing text files
    for vf in video_files:
        text_file = vf.with_suffix('.txt')
        if not text_file.exists():
            print(f"❌ Missing text file for {vf.name}")
            issues_found = True
    
    # Check for empty text files
    for tf in text_files:
        if tf.stat().st_size == 0:
            print(f"❌ Empty text file: {tf.name}")
            issues_found = True
    
    # Check for very small video files (likely corrupted)
    for vf in video_files:
        if vf.stat().st_size < 1000:  # Less than 1KB
            print(f"❌ Very small video file (possibly corrupted): {vf.name}")
            issues_found = True
    
    if not issues_found:
        print("✅ No obvious issues found with the files")
    
    print(f"\n=== Summary ===")
    print(f"Total video files: {len(video_files)}")
    print(f"Total text files: {len(text_files)}")
    print(f"Files look valid: {'Yes' if not issues_found else 'No'}")

if __name__ == "__main__":
    test_video_files()