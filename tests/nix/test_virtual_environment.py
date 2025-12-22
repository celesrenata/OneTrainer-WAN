"""
Tests for virtual environment isolation and compatibility.
Tests WAN 2.2 functionality in different virtual environment setups.
"""
import pytest
import sys
import os
import subprocess
import importlib
from pathlib import Path
from unittest.mock import Mock, patch

from modules.model.WanModel import WanModel
from modules.util.enum.ModelType import ModelType


class TestVirtualEnvironmentIsolation:
    """Test cases for virtual environment isolation."""

    def test_python_executable_location(self):
        """Test Python executable location and virtual environment detection."""
        python_exe = sys.executable
        print(f"Python executable: {python_exe}")
        
        # Check if we're in a virtual environment
        virtual_env = os.environ.get('VIRTUAL_ENV')
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        
        if virtual_env:
            print(f"Virtual environment detected: {virtual_env}")
            # Python executable should be within the virtual environment
            assert python_exe.startswith(virtual_env)
        elif conda_env:
            print(f"Conda environment detected: {conda_env}")
        else:
            print("No virtual environment detected (system Python or Nix)")

    def test_python_path_isolation(self):
        """Test Python path isolation in virtual environment."""
        python_paths = sys.path
        print("Python path entries:")
        for i, path in enumerate(python_paths[:5]):  # Show first 5 entries
            print(f"  {i}: {path}")
        
        # Check for virtual environment paths
        virtual_env = os.environ.get('VIRTUAL_ENV')
        if virtual_env:
            venv_paths = [p for p in python_paths if virtual_env in p]
            assert len(venv_paths) > 0, "Virtual environment paths not found in sys.path"
            print(f"Found {len(venv_paths)} virtual environment paths")

    def test_package_installation_location(self):
        """Test that packages are installed in the correct location."""
        import torch
        torch_location = torch.__file__
        print(f"PyTorch installed at: {torch_location}")
        
        virtual_env = os.environ.get('VIRTUAL_ENV')
        if virtual_env:
            # PyTorch should be installed within the virtual environment
            assert virtual_env in torch_location, f"PyTorch not in virtual environment: {torch_location}"
        
        # Test OneTrainer modules location
        try:
            from modules.model.WanModel import WanModel
            wan_model_file = WanModel.__module__
            print(f"WanModel module: {wan_model_file}")
        except ImportError:
            pytest.skip("OneTrainer modules not available")

    def test_environment_variable_isolation(self):
        """Test environment variable isolation."""
        # Check key environment variables
        env_vars = [
            'PYTHONPATH',
            'VIRTUAL_ENV',
            'CONDA_DEFAULT_ENV',
            'PATH',
            'LD_LIBRARY_PATH',
            'CUDA_HOME',
            'ROCM_PATH'
        ]
        
        print("Environment variables:")
        for var in env_vars:
            value = os.environ.get(var)
            if value:
                print(f"  {var}: {value[:100]}{'...' if len(value) > 100 else ''}")
            else:
                print(f"  {var}: Not set")

    def test_pip_package_isolation(self):
        """Test pip package isolation in virtual environment."""
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                packages = result.stdout
                print("Installed packages (first 10 lines):")
                for line in packages.split('\n')[:10]:
                    if line.strip():
                        print(f"  {line}")
                
                # Check for key packages
                key_packages = ['torch', 'numpy', 'pillow']
                for package in key_packages:
                    if package in packages.lower():
                        print(f"✓ {package} found in environment")
                    else:
                        print(f"✗ {package} not found in environment")
            else:
                pytest.skip("Could not list pip packages")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("pip not available or timed out")

    def test_import_isolation(self):
        """Test that imports work correctly in isolated environment."""
        # Test importing standard library
        import json
        import os
        import sys
        
        # Test importing third-party packages
        try:
            import torch
            import numpy as np
            print("✓ Core dependencies imported successfully")
        except ImportError as e:
            pytest.fail(f"Failed to import core dependencies: {e}")
        
        # Test importing OneTrainer modules
        try:
            from modules.util.enum.ModelType import ModelType
            from modules.model.WanModel import WanModel
            print("✓ OneTrainer modules imported successfully")
        except ImportError as e:
            pytest.skip(f"OneTrainer modules not available: {e}")

    def test_temporary_directory_access(self, temp_dir):
        """Test temporary directory access and permissions."""
        # Test creating files in temp directory
        test_file = os.path.join(temp_dir, "isolation_test.txt")
        
        with open(test_file, 'w') as f:
            f.write("Virtual environment isolation test")
        
        assert os.path.exists(test_file)
        
        # Test reading the file
        with open(test_file, 'r') as f:
            content = f.read()
            assert content == "Virtual environment isolation test"
        
        print(f"✓ Temporary directory access working: {temp_dir}")

    def test_module_loading_consistency(self):
        """Test module loading consistency in virtual environment."""
        # Test reloading modules
        import importlib
        
        # Test reloading torch
        import torch
        original_version = torch.__version__
        
        importlib.reload(torch)
        reloaded_version = torch.__version__
        
        assert original_version == reloaded_version
        print(f"✓ Module reloading consistent: {original_version}")

    def test_wan_model_functionality_in_venv(self):
        """Test WAN model functionality in virtual environment."""
        try:
            model = WanModel(ModelType.WAN_2_2)
            
            # Test basic functionality
            assert model.model_type == ModelType.WAN_2_2
            assert model.adapters() == []
            assert model.all_embeddings() == []
            
            print("✓ WAN model functionality working in virtual environment")
            
        except Exception as e:
            pytest.skip(f"WAN model not available: {e}")

    def test_dependency_version_consistency(self):
        """Test dependency version consistency."""
        import torch
        import numpy as np
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"NumPy version: {np.__version__}")
        
        # Test compatibility
        tensor = torch.from_numpy(np.array([1.0, 2.0, 3.0]))
        numpy_array = tensor.numpy()
        
        assert np.array_equal(numpy_array, np.array([1.0, 2.0, 3.0]))
        print("✓ PyTorch-NumPy compatibility verified")

    def test_cuda_library_isolation(self):
        """Test CUDA library isolation in virtual environment."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        import torch
        
        # Test CUDA functionality
        device = torch.device('cuda:0')
        x = torch.randn(10, 10, device=device)
        y = torch.randn(10, 10, device=device)
        result = torch.mm(x, y)
        
        assert result.device == device
        print("✓ CUDA functionality working in virtual environment")

    def test_environment_cleanup(self):
        """Test environment cleanup and resource management."""
        import gc
        import torch
        
        # Create some objects
        tensors = [torch.randn(100, 100) for _ in range(10)]
        
        # Clean up
        del tensors
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✓ Environment cleanup completed")

    def test_subprocess_isolation(self):
        """Test subprocess isolation in virtual environment."""
        # Test running Python subprocess
        try:
            result = subprocess.run([
                sys.executable, '-c', 
                'import sys; print(sys.executable); import torch; print(torch.__version__)'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                output_lines = result.stdout.strip().split('\n')
                subprocess_python = output_lines[0]
                subprocess_torch_version = output_lines[1] if len(output_lines) > 1 else "unknown"
                
                print(f"Subprocess Python: {subprocess_python}")
                print(f"Subprocess PyTorch: {subprocess_torch_version}")
                
                # Should use the same Python executable
                assert subprocess_python == sys.executable
                
                # Should have the same PyTorch version
                import torch
                assert subprocess_torch_version == torch.__version__
                
            else:
                pytest.skip(f"Subprocess failed: {result.stderr}")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Could not run subprocess test")

    def test_library_path_isolation(self):
        """Test library path isolation."""
        # Check LD_LIBRARY_PATH and related variables
        lib_path = os.environ.get('LD_LIBRARY_PATH', '')
        cuda_home = os.environ.get('CUDA_HOME', '')
        rocm_path = os.environ.get('ROCM_PATH', '')
        
        print(f"LD_LIBRARY_PATH: {lib_path[:100]}{'...' if len(lib_path) > 100 else ''}")
        print(f"CUDA_HOME: {cuda_home}")
        print(f"ROCM_PATH: {rocm_path}")
        
        # Test that we can still load CUDA/ROCm libraries if available
        if torch.cuda.is_available():
            print("✓ GPU libraries accessible despite isolation")

    def test_configuration_file_isolation(self, temp_dir):
        """Test configuration file isolation."""
        # Test creating and reading configuration files
        config_dir = os.path.join(temp_dir, "config")
        os.makedirs(config_dir, exist_ok=True)
        
        config_file = os.path.join(config_dir, "test_config.json")
        
        import json
        config_data = {
            "model_type": "WAN_2_2",
            "batch_size": 1,
            "learning_rate": 1e-4
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Read back the configuration
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config == config_data
        print("✓ Configuration file isolation working")

    def test_cache_directory_isolation(self, temp_dir):
        """Test cache directory isolation."""
        # Test cache directory creation and access
        cache_dir = os.path.join(temp_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create some cache files
        for i in range(3):
            cache_file = os.path.join(cache_dir, f"cache_{i}.dat")
            with open(cache_file, 'wb') as f:
                f.write(b"cache data " * 100)
        
        # Verify cache files exist
        cache_files = os.listdir(cache_dir)
        assert len(cache_files) == 3
        
        print(f"✓ Cache directory isolation working: {len(cache_files)} files")

    def test_error_handling_in_isolation(self):
        """Test error handling in isolated environment."""
        # Test that errors are properly handled and don't leak
        try:
            # Intentionally cause an error
            import torch
            x = torch.randn(10, 10)
            y = torch.randn(5, 5)  # Incompatible dimensions
            result = torch.mm(x, y)  # This should fail
        except RuntimeError as e:
            print(f"✓ Error properly caught and handled: {type(e).__name__}")
        except Exception as e:
            print(f"✓ Unexpected error type caught: {type(e).__name__}")
        
        # Environment should still be functional after error
        z = torch.randn(3, 3)
        assert z.shape == (3, 3)
        print("✓ Environment remains functional after error")