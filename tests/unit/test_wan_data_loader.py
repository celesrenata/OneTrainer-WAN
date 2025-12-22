"""
Unit tests for WanBaseDataLoader class.
Tests video data processing and validation functionality.
"""
import pytest
import torch
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from modules.dataLoader.WanBaseDataLoader import WanBaseDataLoader
from modules.model.WanModel import WanModel
from modules.util.enum.ModelType import ModelType
from modules.util.video_util import FrameSamplingStrategy, VideoValidationError


class TestWanBaseDataLoader:
    """Test cases for WanBaseDataLoader class."""

    def test_video_config_validation_valid(self, mock_train_config, mock_device):
        """Test video configuration validation with valid parameters."""
        mock_train_config.target_frames = 16
        mock_train_config.min_video_resolution = (256, 256)
        mock_train_config.max_video_resolution = (1024, 1024)
        mock_train_config.max_video_duration = 10.0
        
        model = Mock(spec=WanModel)
        train_progress = Mock()
        
        with patch.object(WanBaseDataLoader, 'create_dataset'):
            loader = WanBaseDataLoader(
                train_device=mock_device,
                temp_device=mock_device,
                config=mock_train_config,
                model=model,
                train_progress=train_progress
            )
            
            # Check that validation constraints are set
            assert loader._video_constraints['target_frames'] == 16
            assert loader._video_constraints['min_resolution'] == (256, 256)
            assert loader._video_constraints['max_resolution'] == (1024, 1024)
            assert loader._video_constraints['max_duration'] == 10.0

    def test_video_config_validation_invalid_frames(self, mock_train_config, mock_device):
        """Test video configuration validation with invalid frame count."""
        mock_train_config.target_frames = 0  # Invalid
        
        model = Mock(spec=WanModel)
        train_progress = Mock()
        
        with patch.object(WanBaseDataLoader, 'create_dataset'):
            with pytest.raises(ValueError, match="target_frames must be at least 1"):
                WanBaseDataLoader(
                    train_device=mock_device,
                    temp_device=mock_device,
                    config=mock_train_config,
                    model=model,
                    train_progress=train_progress
                )

    def test_video_config_validation_high_frames_warning(self, mock_train_config, mock_device, capfd):
        """Test warning for high frame count."""
        mock_train_config.target_frames = 100  # Very high
        
        model = Mock(spec=WanModel)
        train_progress = Mock()
        
        with patch.object(WanBaseDataLoader, 'create_dataset'):
            WanBaseDataLoader(
                train_device=mock_device,
                temp_device=mock_device,
                config=mock_train_config,
                model=model,
                train_progress=train_progress
            )
            
            captured = capfd.readouterr()
            assert "Warning: target_frames=100 is quite high" in captured.out

    def test_frame_sampling_strategy_valid(self, mock_train_config, mock_device):
        """Test frame sampling strategy parsing with valid values."""
        mock_train_config.frame_sample_strategy = "uniform"
        
        model = Mock(spec=WanModel)
        train_progress = Mock()
        
        with patch.object(WanBaseDataLoader, 'create_dataset'):
            loader = WanBaseDataLoader(
                train_device=mock_device,
                temp_device=mock_device,
                config=mock_train_config,
                model=model,
                train_progress=train_progress
            )
            
            strategy = loader._get_frame_sampling_strategy(mock_train_config)
            assert strategy == FrameSamplingStrategy.UNIFORM

    def test_frame_sampling_strategy_invalid(self, mock_train_config, mock_device, capfd):
        """Test frame sampling strategy with invalid value falls back to uniform."""
        mock_train_config.frame_sample_strategy = "invalid_strategy"
        
        model = Mock(spec=WanModel)
        train_progress = Mock()
        
        with patch.object(WanBaseDataLoader, 'create_dataset'):
            loader = WanBaseDataLoader(
                train_device=mock_device,
                temp_device=mock_device,
                config=mock_train_config,
                model=model,
                train_progress=train_progress
            )
            
            strategy = loader._get_frame_sampling_strategy(mock_train_config)
            assert strategy == FrameSamplingStrategy.UNIFORM
            
            captured = capfd.readouterr()
            assert "Warning: Unknown frame sampling strategy 'invalid_strategy'" in captured.out

    @patch('modules.dataLoader.WanBaseDataLoader.VideoFrameSampler')
    @patch('modules.dataLoader.WanBaseDataLoader.TemporalConsistencyVAE')
    @patch('modules.dataLoader.WanBaseDataLoader.WanVideoTextEncoder')
    def test_preparation_modules_basic(self, mock_text_encoder, mock_vae, mock_sampler, 
                                     mock_train_config, mock_device):
        """Test preparation modules creation with basic configuration."""
        model = Mock(spec=WanModel)
        model.vae = Mock()
        model.tokenizer = Mock()
        model.text_encoder = Mock()
        model.autocast_context = Mock()
        model.train_dtype = Mock()
        model.train_dtype.torch_dtype.return_value = torch.float32
        model.add_embeddings_to_prompt = Mock()
        
        mock_train_config.temporal_consistency_weight = 1.0
        mock_train_config.seed = 42
        mock_train_config.masked_training = False
        
        # Mock the train_text_encoder_or_embedding method
        mock_train_config.train_text_encoder_or_embedding.return_value = False
        
        train_progress = Mock()
        
        with patch.object(WanBaseDataLoader, 'create_dataset'):
            loader = WanBaseDataLoader(
                train_device=mock_device,
                temp_device=mock_device,
                config=mock_train_config,
                model=model,
                train_progress=train_progress
            )
            
            modules = loader._preparation_modules(mock_train_config, model)
            
            # Should have video sampler, rescale, VAE encoding, and text encoding
            assert len(modules) >= 3  # At least sampler, rescale, and VAE
            
            # Verify VideoFrameSampler was created
            mock_sampler.assert_called_once()
            
            # Verify TemporalConsistencyVAE was created
            mock_vae.assert_called_once()

    @patch('modules.dataLoader.WanBaseDataLoader.DiskCache')
    def test_cache_modules_with_caching(self, mock_disk_cache, mock_train_config, mock_device):
        """Test cache modules creation when caching is enabled."""
        model = Mock(spec=WanModel)
        model.vae = Mock()
        model.tokenizer = Mock()
        model.text_encoder = Mock()
        
        mock_train_config.latent_caching = True
        mock_train_config.cache_dir = "/tmp/cache"
        mock_train_config.masked_training = False
        mock_train_config.train_text_encoder_or_embedding.return_value = False
        
        train_progress = Mock()
        
        with patch.object(WanBaseDataLoader, 'create_dataset'):
            loader = WanBaseDataLoader(
                train_device=mock_device,
                temp_device=mock_device,
                config=mock_train_config,
                model=model,
                train_progress=train_progress
            )
            
            modules = loader._cache_modules(mock_train_config, model)
            
            # Should have at least one cache module when caching is enabled
            assert len(modules) >= 1
            
            # Verify DiskCache was called for video caching
            assert mock_disk_cache.call_count >= 1

    def test_cache_modules_without_caching(self, mock_train_config, mock_device):
        """Test cache modules when caching is disabled."""
        model = Mock(spec=WanModel)
        model.vae = Mock()
        
        mock_train_config.latent_caching = False
        mock_train_config.train_text_encoder_or_embedding.return_value = False
        
        train_progress = Mock()
        
        with patch.object(WanBaseDataLoader, 'create_dataset'):
            loader = WanBaseDataLoader(
                train_device=mock_device,
                temp_device=mock_device,
                config=mock_train_config,
                model=model,
                train_progress=train_progress
            )
            
            modules = loader._cache_modules(mock_train_config, model)
            
            # Should have variation sorting module even without caching
            assert len(modules) >= 1

    def test_output_modules_basic(self, mock_train_config, mock_device):
        """Test output modules creation."""
        model = Mock(spec=WanModel)
        model.vae = Mock()
        model.autocast_context = Mock()
        model.train_dtype = Mock()
        
        mock_train_config.masked_training = False
        mock_train_config.train_text_encoder_or_embedding.return_value = False
        
        train_progress = Mock()
        
        with patch.object(WanBaseDataLoader, 'create_dataset'):
            with patch.object(WanBaseDataLoader, '_video_output_modules_from_out_names') as mock_output:
                mock_output.return_value = []
                
                loader = WanBaseDataLoader(
                    train_device=mock_device,
                    temp_device=mock_device,
                    config=mock_train_config,
                    model=model,
                    train_progress=train_progress
                )
                
                modules = loader._output_modules(mock_train_config, model)
                
                # Verify output modules method was called
                mock_output.assert_called_once()
                
                # Check that required output names are included
                call_args = mock_output.call_args
                output_names = call_args[1]['output_names']
                
                expected_names = [
                    'video_path', 'latent_video', 'prompt_with_embeddings',
                    'tokens', 'original_resolution', 'crop_resolution', 'crop_offset'
                ]
                
                for name in expected_names:
                    assert name in output_names

    @patch('modules.dataLoader.WanBaseDataLoader.DecodeVAE')
    @patch('modules.dataLoader.WanBaseDataLoader.SaveImage')
    def test_debug_modules(self, mock_save_image, mock_decode_vae, mock_train_config, mock_device):
        """Test debug modules creation."""
        model = Mock(spec=WanModel)
        model.vae = Mock()
        model.tokenizer = Mock()
        model.autocast_context = Mock()
        model.train_dtype = Mock()
        model.train_dtype.torch_dtype.return_value = torch.float32
        
        mock_train_config.debug_dir = "/tmp/debug"
        mock_train_config.masked_training = False
        
        train_progress = Mock()
        
        with patch.object(WanBaseDataLoader, 'create_dataset'):
            loader = WanBaseDataLoader(
                train_device=mock_device,
                temp_device=mock_device,
                config=mock_train_config,
                model=model,
                train_progress=train_progress
            )
            
            modules = loader._debug_modules(mock_train_config, model)
            
            # Should have decode and save modules
            assert len(modules) >= 3  # decode_video, save_video_frame, decode_prompt, save_prompt
            
            # Verify DecodeVAE was created
            mock_decode_vae.assert_called_once()
            
            # Verify SaveImage was called for video frames
            assert mock_save_image.call_count >= 1

    def test_validation_mode_config_copy(self, mock_train_config, mock_device):
        """Test that validation mode creates a copy of config with batch_size=1."""
        model = Mock(spec=WanModel)
        train_progress = Mock()
        
        # Set original batch size
        mock_train_config.batch_size = 4
        mock_train_config.multi_gpu = True
        
        with patch.object(WanBaseDataLoader, 'create_dataset') as mock_create:
            WanBaseDataLoader(
                train_device=mock_device,
                temp_device=mock_device,
                config=mock_train_config,
                model=model,
                train_progress=train_progress,
                is_validation=True
            )
            
            # Verify create_dataset was called with modified config
            mock_create.assert_called_once()
            call_args = mock_create.call_args[1]
            config_used = call_args['config']
            
            # Config should be modified for validation
            assert config_used.batch_size == 1
            assert config_used.multi_gpu is False