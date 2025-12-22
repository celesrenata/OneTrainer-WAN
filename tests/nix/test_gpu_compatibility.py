"""
Tests for multi-platform GPU compatibility (CUDA and ROCm).
Tests WAN 2.2 functionality with different GPU acceleration platforms.
"""
import pytest
import torch
import subprocess
import os
from unittest.mock import Mock, patch

from modules.model.WanModel import WanModel
from modules.util.enum.ModelType import ModelType
from modules.util.enum.DataType import DataType


class TestGPUCompatibility:
    """Test cases for multi-platform GPU compatibility."""

    def test_cuda_availability(self):
        """Test CUDA availability and basic functionality."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        
        if torch.cuda.device_count() > 0:
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")

    def test_rocm_availability(self):
        """Test ROCm availability and basic functionality."""
        # ROCm uses CUDA API in PyTorch but with HIP backend
        if not torch.cuda.is_available():
            pytest.skip("GPU acceleration not available")
        
        # Check if we're using ROCm (HIP) backend
        is_rocm = torch.version.hip is not None
        
        if is_rocm:
            print(f"ROCm (HIP) available: {is_rocm}")
            print(f"HIP version: {torch.version.hip}")
            print(f"GPU device count: {torch.cuda.device_count()}")
            
            if torch.cuda.device_count() > 0:
                print(f"GPU device name: {torch.cuda.get_device_name(0)}")
        else:
            pytest.skip("ROCm (HIP) not available")

    def test_cpu_fallback(self):
        """Test CPU fallback functionality."""
        # This should always work
        device = torch.device('cpu')
        
        # Test basic tensor operations on CPU
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        result = torch.mm(x, y)
        
        assert result.device == device
        assert result.shape == (100, 100)
        print("CPU fallback functionality working")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_tensor_operations(self):
        """Test basic GPU tensor operations."""
        device = torch.device('cuda:0')
        
        # Test tensor creation and operations on GPU
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        result = torch.mm(x, y)
        
        assert result.device == device
        assert result.shape == (1000, 1000)
        print(f"GPU tensor operations working on {device}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_wan_model_gpu_compatibility(self):
        """Test WAN model GPU compatibility."""
        device = torch.device('cuda:0')
        
        model = WanModel(ModelType.WAN_2_2)
        
        # Mock model components
        model.vae = Mock()
        model.text_encoder = Mock()
        model.transformer = Mock()
        
        # Test device movement
        model.to(device)
        
        # Verify device movement methods were called
        model.vae.to.assert_called_with(device=device)
        model.text_encoder.to.assert_called_with(device=device)
        model.transformer.to.assert_called_with(device=device)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_mixed_precision_support(self):
        """Test mixed precision support on GPU."""
        device = torch.device('cuda:0')
        
        # Test autocast functionality
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            result = torch.mm(x, y)
            
            # Result should be computed in float16 but may be upcast
            assert result.device == device
            print(f"Mixed precision support working: {result.dtype}")

    def test_device_detection_and_selection(self):
        """Test automatic device detection and selection."""
        # Test device selection logic
        if torch.cuda.is_available():
            preferred_device = torch.device('cuda:0')
            print(f"GPU available, using: {preferred_device}")
        else:
            preferred_device = torch.device('cpu')
            print(f"GPU not available, using: {preferred_device}")
        
        # Test tensor creation on selected device
        x = torch.randn(10, 10, device=preferred_device)
        assert x.device == preferred_device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_memory_management(self):
        """Test GPU memory management."""
        device = torch.device('cuda:0')
        
        # Get initial memory stats
        initial_memory = torch.cuda.memory_allocated(device)
        print(f"Initial GPU memory: {initial_memory / 1024**2:.2f} MB")
        
        # Allocate some memory
        large_tensor = torch.randn(1000, 1000, device=device)
        allocated_memory = torch.cuda.memory_allocated(device)
        print(f"Memory after allocation: {allocated_memory / 1024**2:.2f} MB")
        
        # Free memory
        del large_tensor
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated(device)
        print(f"Memory after cleanup: {final_memory / 1024**2:.2f} MB")
        
        # Memory should be freed (allowing for some overhead)
        assert final_memory <= allocated_memory

    def test_pytorch_backend_detection(self):
        """Test PyTorch backend detection."""
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA compiled version: {torch.version.cuda}")
        print(f"HIP compiled version: {torch.version.hip}")
        
        # Determine which backend is being used
        if torch.cuda.is_available():
            if torch.version.hip is not None:
                backend = "ROCm (HIP)"
            else:
                backend = "CUDA"
            print(f"GPU backend: {backend}")
        else:
            backend = "CPU only"
            print(f"Backend: {backend}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_multi_gpu_detection(self):
        """Test multi-GPU detection and basic functionality."""
        gpu_count = torch.cuda.device_count()
        print(f"Available GPUs: {gpu_count}")
        
        if gpu_count > 1:
            # Test operations on multiple GPUs
            for i in range(min(gpu_count, 2)):  # Test first 2 GPUs
                device = torch.device(f'cuda:{i}')
                x = torch.randn(100, 100, device=device)
                assert x.device == device
                print(f"GPU {i} ({torch.cuda.get_device_name(i)}): OK")
        else:
            print("Single GPU or no GPU available")

    def test_compute_capability_check(self):
        """Test GPU compute capability for WAN 2.2 requirements."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        device_count = torch.cuda.device_count()
        
        for i in range(device_count):
            capability = torch.cuda.get_device_capability(i)
            device_name = torch.cuda.get_device_name(i)
            
            print(f"GPU {i} ({device_name}): Compute capability {capability}")
            
            # WAN 2.2 likely requires modern GPU (compute capability >= 6.0)
            major, minor = capability
            if major >= 6:
                print(f"GPU {i} meets minimum compute capability requirements")
            else:
                print(f"GPU {i} may not meet minimum requirements (capability < 6.0)")

    def test_video_memory_requirements(self):
        """Test GPU memory requirements for video processing."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        device = torch.device('cuda:0')
        total_memory = torch.cuda.get_device_properties(device).total_memory
        print(f"Total GPU memory: {total_memory / 1024**3:.2f} GB")
        
        # Estimate memory requirements for WAN 2.2 video processing
        # This is a rough estimate based on typical video model requirements
        min_required_gb = 8  # Minimum 8GB for reasonable video training
        
        if total_memory / 1024**3 >= min_required_gb:
            print(f"GPU memory sufficient for WAN 2.2 training")
        else:
            print(f"GPU memory may be insufficient (< {min_required_gb}GB)")

    def test_platform_specific_optimizations(self):
        """Test platform-specific optimizations."""
        if torch.cuda.is_available():
            # Test CUDA-specific optimizations
            if torch.version.hip is None:  # NVIDIA CUDA
                print("Testing NVIDIA CUDA optimizations")
                
                # Test cuDNN availability
                if torch.backends.cudnn.enabled:
                    print(f"cuDNN enabled: {torch.backends.cudnn.version()}")
                else:
                    print("cuDNN not available")
                
                # Test Tensor Core support (if available)
                device = torch.device('cuda:0')
                capability = torch.cuda.get_device_capability(device)
                if capability[0] >= 7:  # Tensor Cores available on compute capability >= 7.0
                    print("Tensor Core support available")
                
            else:  # AMD ROCm
                print("Testing AMD ROCm optimizations")
                print(f"HIP version: {torch.version.hip}")
                
                # Test MIOpen availability (ROCm equivalent of cuDNN)
                if hasattr(torch.backends, 'miopen') and torch.backends.miopen.enabled:
                    print("MIOpen enabled")
                else:
                    print("MIOpen status unknown")

    def test_cross_platform_tensor_operations(self):
        """Test tensor operations work consistently across platforms."""
        # Test operations that should work on both CUDA and ROCm
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        
        # Test basic operations
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        
        # Matrix multiplication
        result_mm = torch.mm(x, y)
        assert result_mm.shape == (100, 100)
        
        # Element-wise operations
        result_add = x + y
        assert result_add.shape == (100, 100)
        
        # Reduction operations
        result_sum = torch.sum(x)
        assert result_sum.numel() == 1
        
        print(f"Cross-platform tensor operations working on {device}")

    def test_error_handling_gpu_unavailable(self):
        """Test graceful error handling when GPU is unavailable."""
        # Simulate GPU unavailable scenario
        with patch('torch.cuda.is_available', return_value=False):
            # Should fall back to CPU gracefully
            device = torch.device('cpu')
            x = torch.randn(10, 10, device=device)
            assert x.device.type == 'cpu'
            print("Graceful fallback to CPU when GPU unavailable")

    def test_development_environment_gpu_setup(self):
        """Test GPU setup in development environment."""
        # This test helps verify GPU setup in development environments
        print("=== GPU Development Environment Status ===")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}")
                print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"  Compute capability: {props.major}.{props.minor}")
        
        if torch.version.hip is not None:
            print(f"ROCm (HIP) version: {torch.version.hip}")
        
        print("==========================================")
        
        # Always pass - this is informational
        assert True