#!/usr/bin/env python3
"""
Create minimal test data for WAN 2.2 training on remote system
"""
import os

def create_test_data():
    """Create minimal test data structure"""
    
    # Create the directory structure
    data_dir = "/workspace/input/training/cube"
    os.makedirs(data_dir, exist_ok=True)
    
    # Create a simple text file (no image dependencies)
    with open(os.path.join(data_dir, 'test_prompt.txt'), 'w') as f:
        f.write('a red cube rotating slowly')
    
    # Create another text file
    with open(os.path.join(data_dir, 'test_prompt2.txt'), 'w') as f:
        f.write('a blue cube spinning')
    
    # Create a third text file
    with open(os.path.join(data_dir, 'test_prompt3.txt'), 'w') as f:
        f.write('a green cube moving')
    
    print(f"Created test data in {data_dir}")
    print(f"Files created:")
    for file in os.listdir(data_dir):
        print(f"  - {file}")
    
    # Also create a simple concepts file that points to a local directory
    local_concepts = {
        "name": "Cube",
        "path": "./test_data/cube",
        "enabled": True,
        "type": "STANDARD"
    }
    
    # Create local test directory
    local_dir = "./test_data/cube"
    os.makedirs(local_dir, exist_ok=True)
    
    # Create local test files
    with open(os.path.join(local_dir, 'cube1.txt'), 'w') as f:
        f.write('a red cube')
    
    with open(os.path.join(local_dir, 'cube2.txt'), 'w') as f:
        f.write('a blue cube')
    
    print(f"\nAlso created local test data in {local_dir}")
    print(f"Local files created:")
    for file in os.listdir(local_dir):
        print(f"  - {file}")

if __name__ == "__main__":
    create_test_data()