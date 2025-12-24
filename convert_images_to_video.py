#!/usr/bin/env python3
"""
Convert images to video for WAN 2.2 training.

This script converts the existing JPG images in the training directory
to video files that WAN 2.2 can use for training.
"""

import os
import sys
import json
import subprocess

def convert_images_to_video():
    """Convert images in training directory to video files"""
    
    print("🎥 Converting images to video for WAN 2.2 training...")
    
    try:
        # Load concepts to get the path
        with open("training_concepts/concepts.json", 'r') as f:
            concepts = json.load(f)
        
        for concept in concepts:
            path = concept.get('path', '')
            if not os.path.exists(path):
                print(f"   ❌ Path does not exist: {path}")
                continue
                
            print(f"   📁 Processing path: {path}")
            
            # Get all image files
            files = os.listdir(path)
            image_files = []
            
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    image_files.append(file)
            
            print(f"   Found {len(image_files)} image files")
            
            if len(image_files) == 0:
                print("   ❌ No image files to convert")
                continue
            
            # Sort files to ensure consistent ordering
            image_files.sort()
            
            # Create video from images using ffmpeg
            # This creates a simple slideshow video where each image is shown for 1 second
            output_video = os.path.join(path, f"{concept.get('name', 'video')}_training.mp4")
            
            print(f"   🎬 Creating video: {output_video}")
            
            # Create a temporary file list for ffmpeg
            filelist_path = os.path.join(path, "temp_filelist.txt")
            
            with open(filelist_path, 'w') as f:
                for img_file in image_files:
                    # Each image shown for 1 second (adjust duration as needed)
                    f.write(f"file '{img_file}'\n")
                    f.write("duration 1\n")
                # Add the last image again to ensure proper ending
                if image_files:
                    f.write(f"file '{image_files[-1]}'\n")
            
            # FFmpeg command to create video from images
            cmd = [
                'ffmpeg', '-y',  # -y to overwrite existing files
                '-f', 'concat',
                '-safe', '0',
                '-i', filelist_path,
                '-vf', 'scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-r', '8',  # 8 fps for WAN 2.2
                output_video
            ]
            
            print(f"   🔧 Running ffmpeg command...")
            print(f"   Command: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(cmd, cwd=path, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"   ✅ Video created successfully: {output_video}")
                    
                    # Check video properties
                    if os.path.exists(output_video):
                        size = os.path.getsize(output_video)
                        print(f"   📊 Video size: {size} bytes")
                else:
                    print(f"   ❌ FFmpeg failed: {result.stderr}")
                    
            except FileNotFoundError:
                print("   ❌ FFmpeg not found. Please install ffmpeg on the remote server.")
                print("   💡 Alternative: Upload video files directly to the training directory")
                
            finally:
                # Clean up temporary file
                if os.path.exists(filelist_path):
                    os.remove(filelist_path)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error converting images to video: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_remote_video_files():
    """Check what video files exist on the remote server"""
    
    print("\n📡 Checking remote video files...")
    
    try:
        # Load concepts to get the path
        with open("training_concepts/concepts.json", 'r') as f:
            concepts = json.load(f)
        
        for concept in concepts:
            path = concept.get('path', '')
            print(f"   📁 Remote path: {path}")
            
            if os.path.exists(path):
                files = os.listdir(path)
                video_files = [f for f in files if f.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.mkv', '.flv', '.m4v', '.mpeg', '.wmv'))]
                
                print(f"   📹 Video files: {len(video_files)}")
                for vf in video_files:
                    file_path = os.path.join(path, vf)
                    size = os.path.getsize(file_path)
                    print(f"     - {vf} ({size} bytes)")
                    
                if len(video_files) == 0:
                    print("   ❌ No video files found on remote server")
                    print("   💡 Need to add video files for WAN 2.2 training")
            else:
                print(f"   ❌ Remote path does not exist: {path}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error checking remote files: {e}")
        return False

if __name__ == "__main__":
    print("WAN 2.2 Video Training Data Fix")
    print("=" * 35)
    
    print("🔍 Issue: Dataset length is 0 because no video files exist")
    print("🎯 Solution: Convert images to video or add video files")
    
    success1 = check_remote_video_files()
    success2 = convert_images_to_video()
    
    if success1:
        print("\n💡 RECOMMENDATIONS:")
        print("1. If ffmpeg worked: Video file created from images")
        print("2. If no ffmpeg: Upload video files to remote training directory")
        print("3. Restart training after adding video files")
        print("\n📡 Remote training directory: /workspace/input/training/clawdia-qwen/")
        print("🔗 Remote server: ssh -p 3701 10.1.1.12")
    else:
        print("\n❌ Could not fix video data issue")
        
    sys.exit(0 if success1 else 1)