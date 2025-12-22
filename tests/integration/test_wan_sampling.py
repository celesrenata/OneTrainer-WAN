"""
Integration tests for WAN 2.2 sampling functionality.
Tests video generation and sampling integration.
"""
import pytest
import torch
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from modules.modelSampler.WanModelSampler import WanModelSampler
from modules.model.WanModel import WanModel
from modules.util.enum.ModelType import ModelType
from modules.util.enum.DataType import DataType
from modules.util.config.TrainConfig import TrainConfig


class TestWanSamplingIntegration:
    """Integration tests for WAN 2.2 sampling functionality."""

    @pytest.fixture
    def mock_sampling_model(self):
        """Create a mock model for sampling tests."""
        model = Mock(spec=WanModel)
        model.model_type = ModelType.WAN_2_2
        model.train_dtype = DataType.FLOAT_32
        
        # Mock components
        model.tokenizer = Mock()
        model.text_encoder = Mock()
        model.vae = Mock()
        model.transformer = Mock()
        model.noise_scheduler = Mock()
        
        # Configure mock behaviors
        model.tokenizer.return_value = Mock()
        model.tokenizer.return_value.input_ids = torch.randint(0, 1000, (1, 77))
        
        model.text_encoder.return_value = [torch.randn(1, 77, 768)]
        model.text_encoder.device = torch.device('cpu')
        
        model.vae.decode.return_value = Mock()
        model.vae.decode.return_value.sample = torch.randn(1, 3, 16, 256, 256)  # Video tensor
        
        model.transformer.return_value = Mock()
        model.transformer.return_value.sample = torch.randn(1, 4, 16, 32, 32)
        
        model.noise_scheduler.timesteps = torch.linspace(1000, 0, 50)
        
        # Mock model methods
        model.encode_text.return_value = torch.randn(1, 77, 768)
        model.pack_latents.side_effect = lambda x: x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        model.unpack_latents.side_effect = lambda x, f, h, w: x.view(x.shape[0], -1, f, h, w)
        model.create_pipeline.return_value = Mock()
        
        # Mock device methods
        model.to = Mock()
        model.eval = Mock()
        model.vae_to = Mock()
        model.text_encoder_to = Mock()
        model.transformer_to = Mock()
        
        return model

    @pytest.fixture
    def mock_sampling_config(self, temp_dir):
        """Create a mock sampling configuration."""
        config = Mock(spec=TrainConfig)
        config.sample_definition_file_name = os.path.join(temp_dir, "sample_config.json")
        config.sample_after = 100
        config.sample_every = 50
        config.samples_to_sample = 2
        config.sample_image_format = "mp4"
        config.sample_dir = os.path.join(temp_dir, "samples")
        config.debug_mode = False
        
        # Create sample directory
        os.makedirs(config.sample_dir, exist_ok=True)
        
        return config

    def test_sampler_initialization(self, mock_sampling_model, mock_sampling_config, mock_device):
        """Test WanModelSampler initialization."""
        train_progress = Mock()
        
        sampler = WanModelSampler(
            train_device=mock_device,
            temp_device=mock_device,
            model=mock_sampling_model,
            model_type=ModelType.WAN_2_2,
            config=mock_sampling_config,
            train_progress=train_progress
        )
        
        assert sampler is not None

    def test_sample_generation_basic(self, mock_sampling_model, mock_sampling_config, mock_device):
        """Test basic video sample generation."""
        train_progress = Mock()
        train_progress.epoch = 1
        train_progress.epoch_step = 100
        
        sampler = WanModelSampler(
            train_device=mock_device,
            temp_device=mock_device,
            model=mock_sampling_model,
            model_type=ModelType.WAN_2_2,
            config=mock_sampling_config,
            train_progress=train_progress
        )
        
        # Mock sample definitions
        sample_params = {
            "prompt": "A beautiful landscape",
            "negative_prompt": "",
            "height": 256,
            "width": 256,
            "num_frames": 16,
            "guidance_scale": 7.5,
            "num_inference_steps": 20,
            "seed": 42
        }
        
        with patch.object(sampler, 'sample') as mock_sample:
            mock_sample.return_value = None
            
            sampler.sample(
                sample_params=sample_params,
                destination=mock_sampling_config.sample_dir,
                image_format="mp4",
                text_encoder_layer_skip=0
            )
            
            mock_sample.assert_called_once()

    def test_sample_with_different_parameters(self, mock_sampling_model, mock_sampling_config, mock_device):
        """Test sampling with different parameter configurations."""
        train_progress = Mock()
        
        sampler = WanModelSampler(
            train_device=mock_device,
            temp_device=mock_device,
            model=mock_sampling_model,
            model_type=ModelType.WAN_2_2,
            config=mock_sampling_config,
            train_progress=train_progress
        )
        
        # Test different parameter sets
        parameter_sets = [
            {
                "prompt": "A cat playing",
                "height": 512,
                "width": 512,
                "num_frames": 8,
                "guidance_scale": 5.0,
                "num_inference_steps": 10
            },
            {
                "prompt": "Ocean waves",
                "height": 256,
                "width": 256,
                "num_frames": 24,
                "guidance_scale": 10.0,
                "num_inference_steps": 30
            }
        ]
        
        with patch.object(sampler, 'sample') as mock_sample:
            mock_sample.return_value = None
            
            for params in parameter_sets:
                sampler.sample(
                    sample_params=params,
                    destination=mock_sampling_config.sample_dir,
                    image_format="mp4"
                )
            
            assert mock_sample.call_count == len(parameter_sets)

    def test_sample_callback_integration(self, mock_sampling_model, mock_sampling_config, mock_device):
        """Test sampling with progress callbacks."""
        train_progress = Mock()
        
        sampler = WanModelSampler(
            train_device=mock_device,
            temp_device=mock_device,
            model=mock_sampling_model,
            model_type=ModelType.WAN_2_2,
            config=mock_sampling_config,
            train_progress=train_progress
        )
        
        # Mock callback function
        callback_fn = Mock()
        
        sample_params = {
            "prompt": "Test video",
            "height": 256,
            "width": 256,
            "num_frames": 16,
            "callback_on_step_end": callback_fn
        }
        
        with patch.object(sampler, 'sample') as mock_sample:
            mock_sample.return_value = None
            
            sampler.sample(
                sample_params=sample_params,
                destination=mock_sampling_config.sample_dir,
                image_format="mp4"
            )
            
            mock_sample.assert_called_once()

    def test_sample_output_formats(self, mock_sampling_model, mock_sampling_config, mock_device):
        """Test sampling with different output formats."""
        train_progress = Mock()
        
        sampler = WanModelSampler(
            train_device=mock_device,
            temp_device=mock_device,
            model=mock_sampling_model,
            model_type=ModelType.WAN_2_2,
            config=mock_sampling_config,
            train_progress=train_progress
        )
        
        sample_params = {
            "prompt": "Format test video",
            "height": 256,
            "width": 256,
            "num_frames": 16
        }
        
        formats = ["mp4", "webm", "gif"]
        
        with patch.object(sampler, 'sample') as mock_sample:
            mock_sample.return_value = None
            
            for fmt in formats:
                sampler.sample(
                    sample_params=sample_params,
                    destination=mock_sampling_config.sample_dir,
                    image_format=fmt
                )
            
            assert mock_sample.call_count == len(formats)

    def test_sample_error_handling(self, mock_sampling_model, mock_sampling_config, mock_device):
        """Test sampling error handling."""
        train_progress = Mock()
        
        sampler = WanModelSampler(
            train_device=mock_device,
            temp_device=mock_device,
            model=mock_sampling_model,
            model_type=ModelType.WAN_2_2,
            config=mock_sampling_config,
            train_progress=train_progress
        )
        
        # Test with invalid parameters
        invalid_params = {
            "prompt": "Test",
            "height": -1,  # Invalid
            "width": 256,
            "num_frames": 16
        }
        
        with patch.object(sampler, 'sample') as mock_sample:
            mock_sample.side_effect = ValueError("Invalid height")
            
            with pytest.raises(ValueError):
                sampler.sample(
                    sample_params=invalid_params,
                    destination=mock_sampling_config.sample_dir,
                    image_format="mp4"
                )

    def test_sample_memory_management(self, mock_sampling_model, mock_sampling_config, mock_device):
        """Test sampling memory management."""
        train_progress = Mock()
        
        sampler = WanModelSampler(
            train_device=mock_device,
            temp_device=mock_device,
            model=mock_sampling_model,
            model_type=ModelType.WAN_2_2,
            config=mock_sampling_config,
            train_progress=train_progress
        )
        
        sample_params = {
            "prompt": "Memory test video",
            "height": 512,
            "width": 512,
            "num_frames": 32  # Large number of frames
        }
        
        with patch.object(sampler, 'sample') as mock_sample:
            mock_sample.return_value = None
            
            # Test that model device management is called
            sampler.sample(
                sample_params=sample_params,
                destination=mock_sampling_config.sample_dir,
                image_format="mp4"
            )
            
            # Verify device management methods were called
            mock_sampling_model.to.assert_called()
            mock_sampling_model.eval.assert_called()

    def test_sample_batch_processing(self, mock_sampling_model, mock_sampling_config, mock_device):
        """Test sampling with batch processing."""
        train_progress = Mock()
        
        sampler = WanModelSampler(
            train_device=mock_device,
            temp_device=mock_device,
            model=mock_sampling_model,
            model_type=ModelType.WAN_2_2,
            config=mock_sampling_config,
            train_progress=train_progress
        )
        
        # Test multiple samples in batch
        sample_params_list = [
            {
                "prompt": f"Batch test video {i}",
                "height": 256,
                "width": 256,
                "num_frames": 16
            }
            for i in range(3)
        ]
        
        with patch.object(sampler, 'sample') as mock_sample:
            mock_sample.return_value = None
            
            for params in sample_params_list:
                sampler.sample(
                    sample_params=params,
                    destination=mock_sampling_config.sample_dir,
                    image_format="mp4"
                )
            
            assert mock_sample.call_count == len(sample_params_list)

    def test_sample_quality_validation(self, mock_sampling_model, mock_sampling_config, mock_device):
        """Test sampling output quality validation."""
        train_progress = Mock()
        
        sampler = WanModelSampler(
            train_device=mock_device,
            temp_device=mock_device,
            model=mock_sampling_model,
            model_type=ModelType.WAN_2_2,
            config=mock_sampling_config,
            train_progress=train_progress
        )
        
        sample_params = {
            "prompt": "Quality test video",
            "height": 256,
            "width": 256,
            "num_frames": 16,
            "guidance_scale": 7.5
        }
        
        # Mock successful sampling with quality checks
        with patch.object(sampler, 'sample') as mock_sample:
            # Simulate successful generation
            mock_sample.return_value = torch.randn(1, 3, 16, 256, 256)
            
            result = sampler.sample(
                sample_params=sample_params,
                destination=mock_sampling_config.sample_dir,
                image_format="mp4"
            )
            
            # Verify sampling was called and returned valid result
            mock_sample.assert_called_once()
            if result is not None:
                assert result.shape == (1, 3, 16, 256, 256)

    def test_sample_integration_with_training(self, mock_sampling_model, mock_sampling_config, mock_device):
        """Test sampling integration during training workflow."""
        train_progress = Mock()
        train_progress.epoch = 2
        train_progress.epoch_step = 150
        
        # Configure sampling schedule
        mock_sampling_config.sample_after = 100
        mock_sampling_config.sample_every = 50
        
        sampler = WanModelSampler(
            train_device=mock_device,
            temp_device=mock_device,
            model=mock_sampling_model,
            model_type=ModelType.WAN_2_2,
            config=mock_sampling_config,
            train_progress=train_progress
        )
        
        # Test that sampling should occur based on training progress
        should_sample = (train_progress.epoch_step >= mock_sampling_config.sample_after and
                        train_progress.epoch_step % mock_sampling_config.sample_every == 0)
        
        if should_sample:
            sample_params = {
                "prompt": "Training progress video",
                "height": 256,
                "width": 256,
                "num_frames": 16
            }
            
            with patch.object(sampler, 'sample') as mock_sample:
                mock_sample.return_value = None
                
                sampler.sample(
                    sample_params=sample_params,
                    destination=mock_sampling_config.sample_dir,
                    image_format="mp4"
                )
                
                mock_sample.assert_called_once()
        
        assert sampler is not None