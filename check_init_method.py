#!/usr/bin/env python3
"""
Simple check for SafePipelineModule init method
"""

import ast
import sys

def check_init_method():
    """Check if SafePipelineModule has init method by parsing the AST"""
    
    print("Checking SafePipelineModule init method via AST parsing...")
    
    try:
        # Read the file
        with open('modules/dataLoader/mixin/DataLoaderText2VideoMixin.py', 'r') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        
        # Find SafePipelineModule class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'SafePipelineModule':
                print(f"✅ Found SafePipelineModule class at line {node.lineno}")
                
                # Check for init method
                init_found = False
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == 'init':
                        print(f"✅ Found init method at line {item.lineno}")
                        
                        # Check method signature
                        args = [arg.arg for arg in item.args.args]
                        print(f"✅ init method args: {args}")
                        init_found = True
                        break
                
                if not init_found:
                    print("❌ init method not found in SafePipelineModule")
                    return False
                
                return True
        
        print("❌ SafePipelineModule class not found")
        return False
        
    except Exception as e:
        print(f"❌ Error checking init method: {e}")
        return False

if __name__ == "__main__":
    success = check_init_method()
    print(f"Result: {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)