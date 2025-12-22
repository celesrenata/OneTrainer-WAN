"""
Tests for CPU fallback functionality in development environments.
Tests WAN 2.2 functionality when GPU is not available.
"""
import pytest
import torch
import os
from unittest.mock import Mock, patch

from modules.model.WanModel import WanModel
from modules.dataLoader.WanBaseDataLoader import WanBaseDataLoader
from modules.util.enum.ModelType import ModelType
from modules.util.enum.DataType import DataType


class TestCPUFallback:
    """Test cases for CPU fallback functionality."""

    def test_cpu_device_availability(self):
        """Test CPU device is always available."""
        device = torch.device('cpu')
        assert device.type == 'cpu'
        
        # Test basic tensor operations on CPU
        x = torch.randn(10, 10, device=device)
        y = torch.randn(10, 10, device=device)
        result = torch.mm(x, y)
        
        assert result.device == device
        assert result.shape == (10, 10)
        print("✓ CPU device functionality verified")

    def test_wan_model_cpu_initialization(self):
        """Test WAN model initialization on CPU."""
        model = WanModel(ModelType.WAN_2_2)
        device = torch.device('cpu')
        
        # Mock model components
        model.vae = Mock()
        model.text_encoder = Mock()
        model.transformer = Mock()
        
        # Test device movement to CPU
        model.to(device)
        
        # Verify components were moved to CPU
        model.vae.to.assert_called_with(device=device)
        model.text_encoder.to.assert_called_with(device=device)
        model.transformer.to.assert_called_with(device=device)
        
        print("✓ WAN model CPU initialization working")

    def test_cpu_tensor_operations_performance(self):
        """Test CPU tensor operations performance and correctness."""
        device = torch.device('cpu')
        
        # Test various tensor operations
        size = 100
        x = torch.randn(size, size, device=device)
        y = torch.randn(size, size, device=device)
        
        # Matrix multiplication
        result_mm = torch.mm(x, y)
        assert result_mm.shape == (size, size)
        assert result_mm.device == device
        
        # Element-wise operations
        result_add = x + y
        result_mul = x * y
        assert result_add.shape == (size, size)
        assert result_mul.shape == (size, size)
        
        # Reduction operations
        result_sum = torch.sum(x)
        result_mean = torch.mean(x)
        assert result_sum.device == device
        assert result_mean.device == device
        
        print("✓ CPU tensor operations working correctly")

    def test_cpu_memory_management(self):
        """Test CPU memory management."""
        device = torch.device('cpu')
        
        # Allocate large tensors
        large_tensors = []
        for i in range(5):
            tensor = torch.randn(1000, 1000, device=device)
            large_tensors.append(tensor)
        
        # Verify tensors are on CPU
        for tensor in large_tensors:
            assert tensor.device == device
        
        # Clean up
        del large_tensors
        
        # Test that we can still allocate after cleanup
        new_tensor = torch.randn(500, 500, device=device)
        assert new_tensor.device == device
        
        print("✓ CPU memory management working")

    def test_wan_model_text_encoding_cpu(self):
        """Test WAN model text encoding on CPU."""
        model = WanModel(ModelType.WAN_2_2)
        device = torch.device('cpu')
        
        # Mock components for CPU
        model.tokenizer = Mock()
        model.text_encoder = Mock()
        model.train_dtype = DataType.FLOAT_32
        
        # Configure mocks
        model.tokenizer.return_value = Mock()
        model.tokenizer.return_value.input_ids = torch.randint(0, 1000, (1, 77), device=device)
        
        model.text_encoder.return_value = [torch.randn(1, 77, 768, device=device)]
        model.text_encoder.device = device
        
        model.add_text_encoder_embeddings_to_prompt = Mock(side_effect=lambda x: x)
        model._apply_output_embeddings = Mock(side_effect=lambda *args: args[2])
        
        # Test text encoding
        result = model.encode_text(
            train_device=device,
            batch_size=1,
            text="test prompt"
        )
        
        assert result is not None
        assert result.device == device
        assert result.shape == (1, 77, 768)
        
        print("✓ WAN model text encoding working on CPU")

    def test_video_latent_processing_cpu(self):
        """Test video latent processing on CPU."""
        model = WanModel(ModelType.WAN_2_2)
        device = torch.device('cpu')
        
        # Create sample video latents on CPU
        batch_size, channels, frames, height, width = 1, 4, 16, 32, 32
        latents = torch.randn(batch_size, channels, frames, height, width, device=device)
        
        # Test packing
        packed = model.pack_latents(latents)
        expected_shape = (batch_size, channels * frames, height, width)
        assert packed.shape == expected_shape
        assert packed.device == device
        
        # Test unpacking
        unpacked = model.unpack_latents(packed, frames, height, width)
        assert unpacked.shape == latents.shape
        assert unpacked.device == device
        
        print("✓ Video latent processing working on CPU")

    def test_cpu_data_loading_workflow(self, temp_dir):
        """Test data loading workflow on CPU."""
        device = torch.device('cpu')
        
        # Create mock model
        model = Mock(spec=WanModel)
        model.vae = Mock()
        model.tokenizer = Mock()
        model.text_encoder = Mock()
        model.autocast_context = Mock()
        model.train_dtype = Mock()
        model.train_dtype.torch_dtype.return_value = torch.float32
        model.add_embeddings_to_prompt = Mock(side_effect=lambda x: x)
        
        # Create mock config
        config = Mock()
        config.batch_size = 1
        config.multi_gpu = False
        config.masked_training = False
        config.latent_caching = False
        config.debug_mode = False
        config.cache_dir = os.path.join(temp_dir, "cache")
        config.debug_dir = os.path.join(temp_dir, "debug")
        config.target_frames = 16
        config.frame_sample_strategy = "uniform"
        config.temporal_consistency_weight = 1.0
        config.seed = 42
        config.train_text_encoder_or_embedding.return_value = False
        config.model_type = ModelType.WAN_2_2
        
        train_progress = Mock()
        
        # Test data loader creation on CPU
        with patch('modules.dataLoader.WanBaseDataLoader.MGDS') as mock_mgds:
            with patch('modules.dataLoader.WanBaseDataLoader.TrainDataLoader') as mock_train_dl:
                mock_mgds.return_value = Mock()
                mock_train_dl.return_value = Mock()
                
                data_loader = WanBaseDataLoader(
                    train_device=device,
                    temp_device=device,
                    config=config,
                    model=model,
                    train_progress=train_progress
                )
                
                assert data_loader is not None
                print("✓ Data loading workflow working on CPU")

    def test_cpu_mixed_precision_fallback(self):
        """Test mixed precision fallback on CPU."""
        device = torch.device('cpu')
        
        # CPU doesn't support autocast, but should handle gracefully
        with torch.autocast(device_type='cpu', enabled=False):
            x = torch.randn(100, 100, device=device, dtype=torch.float32)
            y = torch.randn(100, 100, device=device, dtype=torch.float32)
            result = torch.mm(x, y)
            
            assert result.device == device
            assert result.dtype == torch.float32
        
        print("✓ Mixed precision fallback working on CPU")

    def test_cpu_gradient_computation(self):
        """Test gradient computation on CPU."""
        device = torch.device('cpu')
        
        # Create tensors requiring gradients
        x = torch.randn(10, 10, device=device, requires_grad=True)
        y = torch.randn(10, 10, device=device, requires_grad=True)
        
        # Forward pass
        z = torch.mm(x, y)
        loss = torch.sum(z)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert x.grad is not None
        assert y.grad is not None
        assert x.grad.device == device
        assert y.grad.device == device
        
        print("✓ Gradient computation working on CPU")

    def test_cpu_model_evaluation_mode(self):
        """Test model evaluation mode on CPU."""
        model = WanModel(ModelType.WAN_2_2)
        device = torch.device('cpu')
        
        # Mock components
        model.vae = Mock()
        model.text_encoder = Mock()
        model.transformer = Mock()
        
        # Test eval mode
        model.eval()
        
        # Verify eval was called on all components
        model.vae.eval.assert_called_once()
        model.text_encoder.eval.assert_called_once()
        model.transformer.eval.assert_called_once()
        
        print("✓ Model evaluation mode working on CPU")

    def test_cpu_batch_processing(self):
        """Test batch processing on CPU."""
        device = torch.device('cpu')
        batch_size = 4
        
        # Create batch of data
        batch_data = torch.randn(batch_size, 3, 256, 256, device=device)
        
        # Process batch (simulate model forward pass)
        processed_batch = torch.nn.functional.avg_pool2d(batch_data, kernel_size=2)
        
        assert processed_batch.shape == (batch_size, 3, 128, 128)
        assert processed_batch.device == device
        
        print("✓ Batch processing working on CPU")

    def test_cpu_error_handling(self):
        """Test error handling on CPU."""
        device = torch.device('cpu')
        
        # Test dimension mismatch error
        try:
            x = torch.randn(10, 5, device=device)
            y = torch.randn(3, 10, device=device)
            result = torch.mm(x, y)  # Should fail
        except RuntimeError as e:
            print(f"✓ Dimension mismatch error properly caught: {type(e).__name__}")
        
        # Test that we can continue after error
        x = torch.randn(5, 5, device=device)
        y = torch.randn(5, 5, device=device)
        result = torch.mm(x, y)
        assert result.shape == (5, 5)
        
        print("✓ Error handling working on CPU")

    def test_cpu_deterministic_behavior(self):
        """Test deterministic behavior on CPU."""
        device = torch.device('cpu')
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        x1 = torch.randn(10, 10, device=device)
        
        torch.manual_seed(42)
        x2 = torch.randn(10, 10, device=device)
        
        # Should be identical
        assert torch.allclose(x1, x2)
        
        print("✓ Deterministic behavior working on CPU")

    def test_cpu_performance_monitoring(self):
        """Test performance monitoring on CPU."""
        device = torch.device('cpu')
        
        import time
        
        # Time a computation
        start_time = time.time()
        
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        result = torch.mm(x, y)
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        print(f"✓ CPU computation time: {computation_time:.4f} seconds")
        
        # Should complete in reasonable time (less than 10 seconds)
        assert computation_time < 10.0

    def test_cpu_memory_efficiency(self):
        """Test memory efficiency on CPU."""
        device = torch.device('cpu')
        
        # Test in-place operations for memory efficiency
        x = torch.randn(1000, 1000, device=device)
        original_data_ptr = x.data_ptr()
        
        # In-place operation
        x.add_(1.0)
        
        # Should use same memory
        assert x.data_ptr() == original_data_ptr
        
        print("✓ CPU memory efficiency (in-place operations) working")

    def test_cpu_multithreading_support(self):
        """Test multithreading support on CPU."""
        device = torch.device('cpu')
        
        # Check number of threads
        num_threads = torch.get_num_threads()
        print(f"CPU threads available: {num_threads}")
        
        # Test parallel computation
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # This should use multiple threads automatically
        result = torch.mm(x, y)
        
        assert result.shape == (1000, 1000)
        assert result.device == device
        
        print("✓ CPU multithreading support working")

    def test_cpu_development_workflow(self, temp_dir):
        """Test complete development workflow on CPU."""
        device = torch.device('cpu')
        
        # 1. Model initialization
        model = WanModel(ModelType.WAN_2_2)
        
        # 2. Mock components for CPU development
        model.tokenizer = Mock()
        model.text_encoder = Mock()
        model.vae = Mock()
        model.transformer = Mock()
        
        # 3. Configure for CPU
        model.to(device)
        model.eval()
        
        # 4. Test basic functionality
        sample_latents = torch.randn(1, 4, 16, 32, 32, device=device)
        packed = model.pack_latents(sample_latents)
        unpacked = model.unpack_latents(packed, 16, 32, 32)
        
        assert torch.allclose(sample_latents, unpacked)
        
        # 5. Test configuration saving
        config_file = os.path.join(temp_dir, "cpu_config.json")
        import json
        
        config = {
            "device": "cpu",
            "model_type": "WAN_2_2",
            "batch_size": 1
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        assert os.path.exists(config_file)
        
        print("✓ Complete CPU development workflow working")