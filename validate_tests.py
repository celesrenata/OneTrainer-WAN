#!/usr/bin/env python3
"""
Simple validation script for WAN 2.2 test structure.
Validates test files without requiring external dependencies.
"""
import os
import sys
import ast
import importlib.util
from pathlib import Path


def validate_python_syntax(file_path):
    """Validate Python syntax of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"


def check_test_structure():
    """Check test directory structure."""
    required_dirs = [
        'tests',
        'tests/unit',
        'tests/integration', 
        'tests/nix'
    ]
    
    required_files = [
        'tests/__init__.py',
        'tests/conftest.py',
        'tests/unit/__init__.py',
        'tests/unit/test_wan_model.py',
        'tests/unit/test_wan_data_loader.py',
        'tests/unit/test_wan_lora.py',
        'tests/integration/__init__.py',
        'tests/integration/test_wan_training_workflow.py',
        'tests/integration/test_wan_sampling.py',
        'tests/nix/__init__.py',
        'tests/nix/test_nix_environment.py',
        'tests/nix/test_gpu_compatibility.py',
        'tests/nix/test_virtual_environment.py',
        'tests/nix/test_cpu_fallback.py',
        'pytest.ini',
        'run_tests.py'
    ]
    
    print("Checking test directory structure...")
    
    # Check directories
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.isdir(dir_path):
            missing_dirs.append(dir_path)
        else:
            print(f"✓ {dir_path}")
    
    if missing_dirs:
        print(f"✗ Missing directories: {missing_dirs}")
        return False
    
    # Check files
    missing_files = []
    for file_path in required_files:
        if not os.path.isfile(file_path):
            missing_files.append(file_path)
        else:
            print(f"✓ {file_path}")
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    
    return True


def validate_test_files():
    """Validate syntax of all test files."""
    print("\nValidating test file syntax...")
    
    test_files = []
    for root, dirs, files in os.walk('tests'):
        for file in files:
            if file.endswith('.py'):
                test_files.append(os.path.join(root, file))
    
    # Also check main files
    test_files.extend(['run_tests.py', 'validate_tests.py'])
    
    syntax_errors = []
    for file_path in test_files:
        if os.path.exists(file_path):
            valid, error = validate_python_syntax(file_path)
            if valid:
                print(f"✓ {file_path}")
            else:
                print(f"✗ {file_path}: {error}")
                syntax_errors.append((file_path, error))
    
    return len(syntax_errors) == 0, syntax_errors


def check_test_content():
    """Check test content for basic structure."""
    print("\nChecking test content structure...")
    
    test_files = [
        'tests/unit/test_wan_model.py',
        'tests/unit/test_wan_data_loader.py', 
        'tests/unit/test_wan_lora.py',
        'tests/integration/test_wan_training_workflow.py',
        'tests/integration/test_wan_sampling.py'
    ]
    
    content_issues = []
    
    for file_path in test_files:
        if not os.path.exists(file_path):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for basic test structure
            checks = [
                ('import pytest', 'pytest import'),
                ('class Test', 'test class definition'),
                ('def test_', 'test method definition'),
                ('assert ', 'assertion statements')
            ]
            
            file_issues = []
            for check_str, description in checks:
                if check_str not in content:
                    file_issues.append(f"Missing {description}")
            
            if file_issues:
                content_issues.append((file_path, file_issues))
                print(f"✗ {file_path}: {', '.join(file_issues)}")
            else:
                print(f"✓ {file_path}")
                
        except Exception as e:
            content_issues.append((file_path, [f"Error reading file: {e}"]))
            print(f"✗ {file_path}: Error reading file: {e}")
    
    return len(content_issues) == 0, content_issues


def check_pytest_config():
    """Check pytest configuration."""
    print("\nChecking pytest configuration...")
    
    if not os.path.exists('pytest.ini'):
        print("✗ pytest.ini not found")
        return False
    
    try:
        with open('pytest.ini', 'r') as f:
            content = f.read()
        
        required_sections = ['testpaths', 'python_files', 'markers']
        missing_sections = []
        
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"✗ pytest.ini missing sections: {missing_sections}")
            return False
        else:
            print("✓ pytest.ini configuration looks good")
            return True
            
    except Exception as e:
        print(f"✗ Error reading pytest.ini: {e}")
        return False


def main():
    """Main validation function."""
    print("OneTrainer WAN 2.2 Test Structure Validation")
    print("=" * 50)
    
    all_good = True
    
    # Check directory structure
    if not check_test_structure():
        all_good = False
    
    # Validate file syntax
    syntax_valid, syntax_errors = validate_test_files()
    if not syntax_valid:
        all_good = False
        print(f"\nSyntax errors found in {len(syntax_errors)} files:")
        for file_path, error in syntax_errors:
            print(f"  {file_path}: {error}")
    
    # Check test content
    content_valid, content_issues = check_test_content()
    if not content_valid:
        all_good = False
        print(f"\nContent issues found in {len(content_issues)} files:")
        for file_path, issues in content_issues:
            print(f"  {file_path}: {', '.join(issues)}")
    
    # Check pytest config
    if not check_pytest_config():
        all_good = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_good:
        print("✓ All validation checks passed!")
        print("\nTest structure is ready. To run tests:")
        print("1. Install dependencies: pip install pytest torch torchvision numpy pillow")
        print("2. Run tests: python run_tests.py quick")
        sys.exit(0)
    else:
        print("✗ Some validation checks failed.")
        print("Please fix the issues above before running tests.")
        sys.exit(1)


if __name__ == "__main__":
    main()