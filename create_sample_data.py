#!/usr/bin/env python3
"""
Create sample training data for WAN 2.2 testing
"""

import os
import numpy as np
from PIL import Image

def create_sample_data():
    """Create sample training images and text files"""
    
    data_dir = "/workspace/input/training/cube"
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Creating sample training data in: {data_dir}")
    
    # Create 3 sample images with different patterns
    for i in range(3):
        # Create a simple test image
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        
        if i == 0:
            # Cube-like gradient pattern
            for y in range(512):
                for x in range(512):
                    r = min(255, (x + y) // 4)
                    g = min(255, abs(x - y) // 2)
                    b = min(255, (x * y) // 1000)
                    img[y, x] = [r, g, b]
        elif i == 1:
            # Checkerboard pattern
            for y in range(512):
                for x in range(512):
                    if (x // 64 + y // 64) % 2:
                        img[y, x] = [200, 100, 50]  # Orange
                    else:
                        img[y, x] = [50, 100, 200]  # Blue
        else:
            # Circular pattern
            center_x, center_y = 256, 256
            for y in range(512):
                for x in range(512):
                    dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    r = min(255, int(dist // 2))
                    g = min(255, int((256 - dist) // 2))
                    b = min(255, int(abs(x - y) // 4))
                    img[y, x] = [r, g, b]
        
        # Save the image
        img_path = os.path.join(data_dir, f"cube_{i+1:03d}.jpg")
        Image.fromarray(img).save(img_path, 'JPEG', quality=95)
        print(f"Created: {img_path}")
        
        # Create corresponding text file with prompt
        txt_path = os.path.join(data_dir, f"cube_{i+1:03d}.txt")
        with open(txt_path, 'w') as f:
            prompts = [
                "a colorful geometric cube with gradient patterns",
                "a cube with checkerboard pattern in orange and blue colors",
                "a circular cube pattern with radial color gradients"
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