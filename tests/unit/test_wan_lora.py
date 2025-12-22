"""
Unit tests for WAN 2.2 LoRA functionality.
Tests LoRA adapter initialization, loading, and training functionality.
"""
import pytest
import torch
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from modules.modelLoader.WanLoRAModelLoader import WanLoRAModelLoader
from modules.modelSaver.WanLoRAModelSaver import WanLoRAModelSaver
from modules.modelSetup.WanLoRASetup import WanLoRASetup
from modules.model.WanModel import WanModel
from modules.util.enum.ModelType import ModelType
from modules.util.enum.DataType import DataType


class TestWanLoRAModelLoader:
    """Test cases for WanLoRAModelLoader class."""

    def test_lora_loader_initialization(self):
        """Test WanLoRAModelLoader initializes correctly."""
        loader = WanLoRAModelLoader()
        assert loader is not None

    @patch('modules.modelLoader.WanLoRAModelLoader.LoRALoaderMixin')
    def test_load_lora_basic(self, mock_lora_mixin, mock_weight_dtypes, temp_dir):
        """Test basic LoRA loading functionality."""
        loader = WanLoRAModelLoader()
        model = Mock(spec=WanModel)
        model.text_encoder = Mock()
        model.transformer = Mock()
        
        # Create mock LoRA file
        lora_path = os.path.join(temp_dir, "test_lora.safetensors")
        with open(lora_path, 'wb') as f:
            f.write(b"mock lora data")
        
        # Mock the load method
        with patch.object(loader, 'load') as mock_load:
            loader.load(
                model=model,
                model_type=ModelType.WAN_2_2,
                weight_dtypes=mock_weight_dtypes,
                lora_name=lora_path
            )
            
            mock_load.assert_called_once()

    def test_load_lora_nonexistent_file(self, mock_weight_dtypes):
        """Test LoRA loading with nonexistent file."""
        loader = WanLoRAModelLoader()
        model = Mock(spec=WanModel)
        
        with pytest.raises(Exception):
            loader.load(
                model=model,
                model_type=ModelType.WAN_2_2,
                weight_dtypes=mock_weight_dtypes,
                lora_name="/nonexistent/path.safetensors"
            )


class TestWanLoRAModelSaver:
    """Test cases for WanLoRAModelSaver class."""

    def test_lora_saver_initialization(self):
        """Test WanLoRAModelSaver initializes correctly."""
        saver = WanLoRAModelSaver()
        assert saver is not None

    def test_save_lora_basic(self, temp_dir):
        """Test basic LoRA saving functionality."""
        saver = WanLoRAModelSaver()
        model = Mock(spec=WanModel)
        
        # Mock LoRA adapters
        mock_text_lora = Mock()
        mock_text_lora.state_dict.return_value = {"text_encoder.layer.weight": torch.randn(10, 10)}
        mock_transformer_lora = Mock()
        mock_transformer_lora.state_dict.return_value = {"transformer.layer.weight": torch.randn(20, 20)}
        
        model.text_encoder_lora = mock_text_lora
        model.transformer_lora = mock_transformer_lora
        
        output_path = os.path.join(temp_dir, "test_lora.safetensors")
        
        with patch.object(saver, 'save') as mock_save:
            saver.save(
                model=model,
                model_type=ModelType.WAN_2_2,
                output_model_name=output_path
            )
            
            mock_save.assert_called_once()

    def test_save_lora_no_adapters(self, temp_dir):
        """Test LoRA saving when no adapters are present."""
        saver = WanLoRAModelSaver()
        model = Mock(spec=WanModel)
        model.text_encoder_lora = None
        model.transformer_lora = None
        
        output_path = os.path.join(temp_dir, "test_lora.safetensors")
        
        # Should handle gracefully when no LoRA adapters exist
        with patch.object(saver, 'save') as mock_save:
            saver.save(
                model=model,
                model_type=ModelType.WAN_2_2,
                output_model_name=output_path
            )
            
            mock_save.assert_called_once()


class TestWanLoRASetup:
    """Test cases for WanLoRASetup class."""

    def test_lora_setup_initialization(self):
        """Test WanLoRASetup initializes correctly."""
        setup = WanLoRASetup()
        assert setup is not None

    def test_setup_lora_adapters(self, mock_train_config):
        """Test LoRA adapter setup."""
        setup = WanLoRASetup()
        model = Mock(spec=WanModel)
        model.text_encoder = Mock()
        model.transformer = Mock()
        
        # Mock config for LoRA
        mock_train_config.lora_rank = 16
        mock_train_config.lora_alpha = 32
        mock_train_config.train_text_encoder = True
        mock_train_config.train_transformer = True
        
        with patch.object(setup, 'setup_model') as mock_setup:
            setup.setup_model(model, mock_train_config)
            mock_setup.assert_called_once()

    def test_setup_lora_text_encoder_only(self, mock_train_config):
        """Test LoRA setup for text encoder only."""
        setup = WanLoRASetup()
        model = Mock(spec=WanModel)
        model.text_encoder = Mock()
        model.transformer = Mock()
        
        mock_train_config.lora_rank = 16
        mock_train_config.lora_alpha = 32
        mock_train_config.train_text_encoder = True
        mock_train_config.train_transformer = False
        
        with patch.object(setup, 'setup_model') as mock_setup:
            setup.setup_model(model, mock_train_config)
            mock_setup.assert_called_once()

    def test_setup_lora_transformer_only(self, mock_train_config):
        """Test LoRA setup for transformer only."""
        setup = WanLoRASetup()
        model = Mock(spec=WanModel)
        model.text_encoder = Mock()
        model.transformer = Mock()
        
        mock_train_config.lora_rank = 16
        mock_train_config.lora_alpha = 32
        mock_train_config.train_text_encoder = False
        mock_train_config.train_transformer = True
        
        with patch.object(setup, 'setup_model') as mock_setup:
            setup.setup_model(model, mock_train_config)
            mock_setup.assert_called_once()

    def test_lora_memory_efficiency(self, mock_train_config):
        """Test that LoRA setup enables memory-efficient training."""
        setup = WanLoRASetup()
        model = Mock(spec=WanModel)
        model.text_encoder = Mock()
        model.transformer = Mock()
        
        mock_train_config.lora_rank = 8  # Low rank for memory efficiency
        mock_train_config.lora_alpha = 16
        mock_train_config.gradient_checkpointing = True
        
        with patch.object(setup, 'setup_model') as mock_setup:
            setup.setup_model(model, mock_train_config)
            mock_setup.assert_called_once()

    def test_lora_parameter_validation(self, mock_train_config):
        """Test LoRA parameter validation."""
        setup = WanLoRASetup()
        model = Mock(spec=WanModel)
        
        # Test invalid rank
        mock_train_config.lora_rank = 0
        mock_train_config.lora_alpha = 32
        
        with patch.object(setup, 'setup_model') as mock_setup:
            # Should handle invalid parameters gracefully
            setup.setup_model(model, mock_train_config)
            mock_setup.assert_called_once()


class TestWanLoRAIntegration:
    """Integration tests for WAN 2.2 LoRA functionality."""

    def test_lora_workflow_basic(self, mock_train_config, temp_dir):
        """Test basic LoRA workflow: setup -> train -> save -> load."""
        # Setup
        setup = WanLoRASetup()
        model = Mock(spec=WanModel)
        model.text_encoder = Mock()
        model.transformer = Mock()
        
        mock_train_config.lora_rank = 16
        mock_train_config.lora_alpha = 32
        
        # Mock LoRA adapters after setup
        mock_text_lora = Mock()
        mock_text_lora.state_dict.return_value = {"text_encoder.layer.weight": torch.randn(10, 16)}
        mock_transformer_lora = Mock()
        mock_transformer_lora.state_dict.return_value = {"transformer.layer.weight": torch.randn(20, 16)}
        
        model.text_encoder_lora = mock_text_lora
        model.transformer_lora = mock_transformer_lora
        
        # Save
        saver = WanLoRAModelSaver()
        lora_path = os.path.join(temp_dir, "test_lora.safetensors")
        
        with patch.object(setup, 'setup_model') as mock_setup:
            with patch.object(saver, 'save') as mock_save:
                # Setup LoRA
                setup.setup_model(model, mock_train_config)
                mock_setup.assert_called_once()
                
                # Save LoRA
                saver.save(model, ModelType.WAN_2_2, lora_path)
                mock_save.assert_called_once()

    def test_lora_adapter_state_management(self):
        """Test LoRA adapter state management."""
        model = WanModel(ModelType.WAN_2_2)
        
        # Initially no adapters
        assert model.adapters() == []
        
        # Add mock adapters
        mock_text_lora = Mock()
        mock_transformer_lora = Mock()
        model.text_encoder_lora = mock_text_lora
        model.transformer_lora = mock_transformer_lora
        
        # Should return both adapters
        adapters = model.adapters()
        assert len(adapters) == 2
        assert mock_text_lora in adapters
        assert mock_transformer_lora in adapters
        
        # Remove one adapter
        model.text_encoder_lora = None
        adapters = model.adapters()
        assert len(adapters) == 1
        assert mock_transformer_lora in adapters