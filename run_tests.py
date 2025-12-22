#!/usr/bin/env python3
"""
Test runner for OneTrainer WAN 2.2 comprehensive test suite.
Supports different test categories and environments.
"""
import sys
import subprocess
import argparse
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    if description:
        print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=False, capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def check_dependencies():
    """Check if required dependencies are available."""
    print("Checking dependencies...")
    
    try:
        import pytest
        print(f"✓ pytest {pytest.__version__}")
    except ImportError:
        print("✗ pytest not found. Install with: pip install pytest")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not found. Install PyTorch first.")
        return False
    
    return True


def run_unit_tests():
    """Run unit tests."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/unit/", 
        "-v", 
        "--tb=short",
        "-m", "not slow"
    ]
    return run_command(cmd, "Unit Tests")


def run_integration_tests():
    """Run integration tests."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/integration/", 
        "-v", 
        "--tb=short",
        "-m", "not slow"
    ]
    return run_command(cmd, "Integration Tests")


def run_nix_tests():
    """Run Nix environment tests."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/nix/", 
        "-v", 
        "--tb=short"
    ]
    return run_command(cmd, "Nix Environment Tests")


def run_gpu_tests():
    """Run GPU compatibility tests."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/nix/test_gpu_compatibility.py", 
        "-v", 
        "--tb=short"
    ]
    return run_command(cmd, "GPU Compatibility Tests")


def run_cpu_tests():
    """Run CPU fallback tests."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/nix/test_cpu_fallback.py", 
        "-v", 
        "--tb=short"
    ]
    return run_command(cmd, "CPU Fallback Tests")


def run_all_tests():
    """Run all tests."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/", 
        "-v", 
        "--tb=short",
        "-m", "not slow"
    ]
    return run_command(cmd, "All Tests")


def run_quick_tests():
    """Run quick smoke tests."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/unit/test_wan_model.py::TestWanModel::test_wan_model_initialization",
        "tests/nix/test_nix_environment.py::TestNixEnvironment::test_python_version_compatibility",
        "tests/nix/test_cpu_fallback.py::TestCPUFallback::test_cpu_device_availability",
        "-v"
    ]
    return run_command(cmd, "Quick Smoke Tests")


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="OneTrainer WAN 2.2 Test Runner")
    parser.add_argument(
        "test_type", 
        choices=["unit", "integration", "nix", "gpu", "cpu", "all", "quick"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--check-deps", 
        action="store_true",
        help="Check dependencies before running tests"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not os.path.exists("tests"):
        print("Error: tests directory not found. Run from OneTrainer root directory.")
        sys.exit(1)
    
    # Check dependencies if requested
    if args.check_deps or args.test_type != "quick":
        if not check_dependencies():
            print("\nDependency check failed. Please install missing dependencies.")
            sys.exit(1)
    
    # Run the requested tests
    success = False
    
    if args.test_type == "unit":
        success = run_unit_tests()
    elif args.test_type == "integration":
        success = run_integration_tests()
    elif args.test_type == "nix":
        success = run_nix_tests()
    elif args.test_type == "gpu":
        success = run_gpu_tests()
    elif args.test_type == "cpu":
        success = run_cpu_tests()
    elif args.test_type == "all":
        success = run_all_tests()
    elif args.test_type == "quick":
        success = run_quick_tests()
    
    # Print summary
    print(f"\n{'='*60}")
    if success:
        print("✓ Tests completed successfully!")
        sys.exit(0)
    else:
        print("✗ Some tests failed or encountered errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()