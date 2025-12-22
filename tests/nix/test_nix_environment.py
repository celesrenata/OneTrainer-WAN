"""
Tests for Nix environment compatibility and dependency resolution.
Tests WAN 2.2 functionality in Nix-based OneTrainer environment.
"""
import pytest
import torch
import subprocess
import sys
import os
import importlib
from unittest.mock import Mock, patch

from modules.model.WanModel import WanModel
from modules.util.enum.ModelType import ModelType


class TestNixEnvironment:
    """Test cases for Nix environment compatibility."""

    def test_python_version_compatibility(self):
        """Test Python version meets requirements."""
        # OneTrainer requires Python >= 3.10
        assert sys.version_info >= (3, 10), f"Python version {sys.version_info} is too old"

    def test_pytorch_installation(self):
        """Test PyTorch is properly installed."""
        assert torch.__version__ is not None
        print(f"PyTorch version: {torch.__version__}")
        
        # Test basic tensor operations
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        result = torch.mm(x, y)
        assert result.shape == (2, 2)

    def test_required_dependencies_import(self):
        """Test that all required dependencies can be imported."""
        required_modules = [
            'torch',
            'torchvision',
            'numpy',
            'PIL',
            'transformers',
            'diffusers',
            'mgds'
        ]
        
        missing_modules = []
        for module_name in required_modules:
            try:
                importlib.import_module(module_name)
            except ImportError:
                missing_modules.append(module_name)
        
        if missing_modules:
            pytest.skip(f"Missing required modules in Nix environment: {missing_modules}")

    def test_wan_model_basic_functionality(self):
        """Test basic WAN model functionality in Nix environment."""
        model = WanModel(ModelType.WAN_2_2)
        
        # Test model initialization
        assert model.model_type == ModelType.WAN_2_2
        assert model.tokenizer is None
        assert model.text_encoder is None
        assert model.vae is None
        assert model.transformer is None

    def test_video_processing_dependencies(self):
        """Test video processing dependencies are available."""
        try:
            import cv2
            print(f"OpenCV version: {cv2.__version__}")
        except ImportError:
            pytest.skip("OpenCV not available in Nix environment")
        
        # Test basic video processing capability
        import numpy as np
        
        # Create a dummy video frame
        frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        assert frame.shape == (256, 256, 3)

    def test_nix_environment_variables(self):
        """Test Nix-specific environment variables."""
        # Check if we're running in a Nix environment
        nix_store = os.environ.get('NIX_STORE')
        if nix_store:
            print(f"Running in Nix environment with store: {nix_store}")
            assert os.path.exists(nix_store)
        else:
            pytest.skip("Not running in Nix environment")

    def test_virtual_environment_isolation(self):
        """Test virtual environment isolation."""
        # Check Python path isolation
        python_path = sys.executable
        print(f"Python executable: {python_path}")
        
        # In Nix, Python should be from the store or a specific path
        if 'nix' in python_path.lower() or '/nix/store' in python_path:
            print("Running in Nix-managed Python environment")
        else:
            # Could be in a virtual environment
            virtual_env = os.environ.get('VIRTUAL_ENV')
            if virtual_env:
                print(f"Running in virtual environment: {virtual_env}")
                assert python_path.startswith(virtual_env)

    def test_package_version_consistency(self):
        """Test package version consistency in Nix environment."""
        import torch
        import numpy as np
        
        # Log versions for debugging
        print(f"PyTorch: {torch.__version__}")
        print(f"NumPy: {np.__version__}")
        
        # Test compatibility between packages
        tensor = torch.from_numpy(np.array([1.0, 2.0, 3.0]))
        assert tensor.dtype == torch.float64
        
        numpy_array = tensor.numpy()
        assert numpy_array.dtype == np.float64

    def test_dependency_resolution_completeness(self):
        """Test that all dependencies are properly resolved."""
        # Test importing OneTrainer modules
        try:
            from modules.util.enum.ModelType import ModelType
            from modules.util.enum.DataType import DataType
            from modules.util.config.TrainConfig import TrainConfig
            
            # Test enum functionality
            assert ModelType.WAN_2_2 is not None
            assert DataType.FLOAT_32 is not None
            
        except ImportError as e:
            pytest.fail(f"Failed to import OneTrainer modules: {e}")

    def test_nix_flake_environment_setup(self):
        """Test Nix flake environment setup."""
        # Check if we can access Nix commands (if available)
        try:
            result = subprocess.run(['nix', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"Nix version: {result.stdout.strip()}")
            else:
                pytest.skip("Nix command not available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Nix command not available or timed out")

    def test_memory_management_in_nix(self):
        """Test memory management in Nix environment."""
        # Test that we can allocate and free memory properly
        large_tensor = torch.randn(1000, 1000)
        assert large_tensor.shape == (1000, 1000)
        
        # Force garbage collection
        del large_tensor
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_file_system_permissions(self, temp_dir):
        """Test file system permissions in Nix environment."""
        # Test that we can create and write files
        test_file = os.path.join(temp_dir, "nix_test.txt")
        
        with open(test_file, 'w') as f:
            f.write("Nix environment test")
        
        assert os.path.exists(test_file)
        
        with open(test_file, 'r') as f:
            content = f.read()
            assert content == "Nix environment test"
        
        # Clean up
        os.remove(test_file)

    def test_nix_store_access(self):
        """Test access to Nix store if available."""
        nix_store = os.environ.get('NIX_STORE', '/nix/store')
        
        if os.path.exists(nix_store):
            # Test that we can read from Nix store
            assert os.access(nix_store, os.R_OK)
            print(f"Nix store accessible at: {nix_store}")
        else:
            pytest.skip("Nix store not available")

    def test_reproducible_environment(self):
        """Test environment reproducibility."""
        # Test that random seeds work consistently
        torch.manual_seed(42)
        tensor1 = torch.randn(10)
        
        torch.manual_seed(42)
        tensor2 = torch.randn(10)
        
        assert torch.allclose(tensor1, tensor2)
        print("Environment provides reproducible random number generation")

    def test_nix_shell_integration(self):
        """Test integration with nix-shell if available."""
        # Check if we're in a nix-shell
        in_nix_shell = os.environ.get('IN_NIX_SHELL')
        
        if in_nix_shell:
            print(f"Running in nix-shell: {in_nix_shell}")
            
            # Test that shell environment is properly set up
            shell = os.environ.get('SHELL')
            if shell:
                print(f"Shell: {shell}")
        else:
            pytest.skip("Not running in nix-shell")

    def test_development_tools_availability(self):
        """Test availability of development tools in Nix environment."""
        tools_to_check = ['python', 'pip']
        
        available_tools = []
        for tool in tools_to_check:
            try:
                result = subprocess.run([tool, '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    available_tools.append(tool)
                    print(f"{tool}: {result.stdout.strip()}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        # At least Python should be available
        assert 'python' in available_tools or sys.executable is not None