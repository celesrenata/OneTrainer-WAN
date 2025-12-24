#!/usr/bin/env python3
"""
Create simple sample training data for WAN 2.2 testing
"""

import os
from PIL import Image

def create_sample_data():
    """Create sample training images and text files"""
    
    data_dir = "/workspace/input/training/cube"
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Creating sample training data in: {data_dir}")
    
    # Create 3 sample images with different solid colors
    colors = [
        (255, 100, 100),  # Red
        (100, 255, 100),  # Green  
        (100, 100, 255),  # Blue
    ]
    
    for i, color in enumerate(colors):
        # Create a simple solid color image
        img = Image.new('RGB', (512, 512), color)
        
        # Save the image
        img_path = os.path.join(data_dir, f"cube_{i+1:03d}.jpg")
        img.save(img_path, 'JPEG', quality=95)
        print(f"Created: {img_path}")
        
        # Create corresponding text file with prompt
        txt_path = os.path.join(data_dir, f"cube_{i+1:03d}.txt")
        with open(txt_path, 'w') as f:
            prompts = [
                "a red cube",
                "a green cube", 
                "a blue cube"
            ]
            f.write(prompts[i])
        print(f"Created: {txt_path}")
    
    # List the created files
    print(f"\nFiles created in {data_dir}:")
    files = sorted(os.listdir(data_dir))
    for file in files:
        file_path = os.path.join(data_dir, file)
        size = os.path.getsize(file_path)
        print(f"  {file} ({size:,} bytes)")
    
    print(f"\n✅ Sample training data created successfully!")
    print(f"   Total files: {len(files)}")
    print(f"   Images: {len([f for f in files if f.endswith('.jpg')])}")
    print(f"   Text files: {len([f for f in files if f.endswith('.txt')])}")

if __name__ == "__main__":
    create_sample_data()