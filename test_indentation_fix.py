#!/usr/bin/env python3
"""
Test script to verify the indentation fix for DataLoaderText2VideoMixin.py
"""

def test_import():
    """Test if the module can be imported without syntax errors"""
    try:
        print("Testing import of DataLoaderText2VideoMixin...")
        from modules.dataLoader.mixin.DataLoaderText2VideoMixin import DataLoaderText2VideoMixin
        print("✓ DataLoaderText2VideoMixin imported successfully")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error in DataLoaderText2VideoMixin: {e}")
        return False
    except ImportError as e:
        print(f"✗ Import error (dependencies missing): {e}")
        return False
    except Exception as e:
        print(f"✗ Other error: {e}")
        return False

def test_create_module():
    """Test if the create module can be imported"""
    try:
        print("Testing import of create module...")
        from modules.util import create
        print("✓ create module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Error importing create module: {e}")
        return False

def test_training_script_syntax():
    """Test if the training script can at least parse without syntax errors"""
    try:
        print("Testing training script syntax...")
        import ast
        with open('scripts/train.py', 'r') as f:
            content = f.read()
        ast.parse(content)
        print("✓ Training script syntax is valid")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error in training script: {e}")
        return False
    except Exception as e:
        print(f"✗ Error reading training script: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Indentation Fix ===")
    
    success = True
    success &= test_training_script_syntax()
    success &= test_import()
    success &= test_create_module()
    
    if success:
        print("\n✓ All tests passed! The indentation fix appears to be working.")
        print("You can now try running the training script.")
    else:
        print("\n✗ Some tests failed. There may still be issues to fix.")