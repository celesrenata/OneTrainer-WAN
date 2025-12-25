#!/usr/bin/env python3
"""
Create minimal test data for WAN 2.2 training
"""
import os
from PIL import Image

def create_test_data():
    """Create minimal test data structure"""
    
    # Create the directory structure
    data_dir = "/workspace/input/training/cube"
    os.makedirs(data_dir, exist_ok=True)
    
    # Create a simple test image
    img = Image.new('RGB', (256, 256), color='red')
    img.save(os.path.join(data_dir, 'test_image.jpg'))
    
    # Create a simple text file
    with open(os.path.join(data_dir, 'test_image.txt'), 'w') as f:
        f.write('a red cube')
    
    print(f"Created test data in {data_dir}")
    print(f"Files created:")
    for file in os.listdir(data_dir):
        print(f"  - {file}")

if __name__ == "__main__":
    create_test_data()