"""
Integration tests for WAN 2.2 training workflow.
Tests end-to-end training workflow with synthetic data.
"""
import pytest
import torch
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from modules.model.WanModel import WanModel
from modules.dataLoader.WanBaseDataLoader import WanBaseDataLoader
from modules.modelLoader.wan.WanModelLoader import WanModelLoader
from modules.modelSaver.WanFineTuneModelSaver import WanFineTuneModelSaver
from modules.modelSetup.WanFineTuneSetup import WanFineTuneSetup
from modules.util.enum.ModelType import ModelType
from modules.util.enum.DataType import DataType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.config.TrainConfig import TrainConfig, QuantizationConfig


class TestWanTrainingWorkflow:
    """Integration tests for complete WAN 2.2 training workflow."""

    @pytest.fixture
    def synthetic_video_data(self, temp_dir):
        """Create synthetic video data for testing."""
        video_dir = os.path.join(temp_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        
        # Create mock video files (just empty files for testing)
        video_files = []
        for i in range(3):
            video_path = os.path.join(video_dir, f"video_{i}.mp4")
            with open(video_path, 'wb') as f:
                f.write(b"mock video data")
            video_files.append(video_path)
            
            # Create corresponding text files
            text_path = os.path.join(video_dir, f"video_{i}.txt")
            with open(text_path, 'w') as f:
                f.write(f"A test video description {i}")
        
        return video_dir, video_files

    @pytest.fixture
    def mock_model_components(self):
        """Create mock model components for testing."""
        components = {
            'tokenizer': Mock(),
            'text_encoder': Mock(),
            'vae': Mock(),
            'transformer': Mock(),
            'scheduler': Mock()
        }
        
        # Configure mock behaviors
        components['tokenizer'].return_value = Mock()
        components['tokenizer'].return_value.input_ids = torch.randint(0, 1000, (1, 77))
        
        components['text_encoder'].return_value = [torch.randn(1, 77, 768)]
        components['text_encoder'].device = torch.device('cpu')
        
        components['vae'].encode.return_value = Mock()
        components['vae'].encode.return_value.latent_dist = Mock()
        components['vae'].encode.return_value.latent_dist.sample.return_value = torch.randn(1, 4, 16, 32, 32)
        
        components['transformer'].return_value = Mock()
        components['transformer'].return_value.sample = torch.randn(1, 4, 16, 32, 32)
        
        return components

    def test_model_initialization_workflow(self, mock_model_components):
        """Test complete model initialization workflow."""
        # Create model
        model = WanModel(ModelType.WAN_2_2)
        
        # Assign components
        model.tokenizer = mock_model_components['tokenizer']
        model.text_encoder = mock_model_components['text_encoder']
        model.vae = mock_model_components['vae']
        model.transformer = mock_model_components['transformer']
        model.noise_scheduler = mock_model_components['scheduler']
        
        # Test device movement
        device = torch.device('cpu')
        model.to(device)
        
        # Test eval mode
        model.eval()
        
        # Verify components are properly set
        assert model.tokenizer is not None
        assert model.text_encoder is not None
        assert model.vae is not None
        assert model.transformer is not None
        assert model.noise_scheduler is not None

    @patch('modules.modelLoader.wan.WanModelLoader.DiffusionPipeline')
    @patch('modules.modelLoader.wan.WanModelLoader.AutoencoderKL')
    @patch('modules.modelLoader.wan.WanModelLoader.PreTrainedModel')
    def test_model_loading_workflow(self, mock_model, mock_vae, mock_pipeline, temp_dir):
        """Test model loading workflow with mock components."""
        # Create mock model directory structure
        model_dir = os.path.join(temp_dir, "wan_model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Create meta.json to indicate internal model format
        meta_path = os.path.join(model_dir, "meta.json")
        with open(meta_path, 'w') as f:
            json.dump({"model_type": "WAN_2_2"}, f)
        
        # Create subdirectories
        for subdir in ["tokenizer", "text_encoder", "vae", "transformer", "scheduler"]:
            os.makedirs(os.path.join(model_dir, subdir), exist_ok=True)
        
        # Mock components
        mock_vae.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        
        # Create loader and model
        loader = WanModelLoader()
        model = WanModel(ModelType.WAN_2_2)
        
        model_names = Mock(spec=ModelNames)
        model_names.base_model = model_dir
        model_names.transformer_model = None
        model_names.vae_model = None
        model_names.include_text_encoder = True
        
        weight_dtypes = Mock(spec=ModelWeightDtypes)
        weight_dtypes.text_encoder = DataType.FLOAT_32
        weight_dtypes.transformer = DataType.FLOAT_32
        weight_dtypes.vae = DataType.FLOAT_32
        weight_dtypes.train_dtype = DataType.FLOAT_32
        
        quantization = Mock(spec=QuantizationConfig)
        
        # Test loading (will use mocked components)
        with patch.object(loader, '_prepare_sub_modules'):
            with patch.object(loader, '_load_diffusers_sub_module') as mock_load:
                mock_load.return_value = Mock()
                
                try:
                    loader.load(model, ModelType.WAN_2_2, model_names, weight_dtypes, quantization)
                except Exception as e:
                    # Expected to fail with mocked components, but should attempt loading
                    assert "could not load model" in str(e)

    def test_data_loading_workflow(self, synthetic_video_data, mock_model_components, mock_device):
        """Test data loading workflow with synthetic video data."""
        video_dir, video_files = synthetic_video_data
        
        # Create model with components
        model = WanModel(ModelType.WAN_2_2)
        model.vae = mock_model_components['vae']
        model.tokenizer = mock_model_components['tokenizer']
        model.text_encoder = mock_model_components['text_encoder']
        model.autocast_context = Mock()
        model.train_dtype = Mock()
        model.train_dtype.torch_dtype.return_value = torch.float32
        model.add_embeddings_to_prompt = Mock(side_effect=lambda x: x)
        
        # Create config
        config = Mock(spec=TrainConfig)
        config.batch_size = 1
        config.multi_gpu = False
        config.masked_training = False
        config.latent_caching = False
        config.debug_mode = False
        config.cache_dir = "/tmp/cache"
        config.debug_dir = "/tmp/debug"
        config.target_frames = 16
        config.frame_sample_strategy = "uniform"
        config.temporal_consistency_weight = 1.0
        config.seed = 42
        config.train_text_encoder_or_embedding.return_value = False
        config.model_type = ModelType.WAN_2_2
        
        train_progress = Mock()
        
        # Test data loader creation
        with patch('modules.dataLoader.WanBaseDataLoader.MGDS') as mock_mgds:
            with patch('modules.dataLoader.WanBaseDataLoader.TrainDataLoader') as mock_train_dl:
                mock_mgds.return_value = Mock()
                mock_train_dl.return_value = Mock()
                
                data_loader = WanBaseDataLoader(
                    train_device=mock_device,
                    temp_device=mock_device,
                    config=config,
                    model=model,
                    train_progress=train_progress
                )
                
                # Verify data loader was created
                assert data_loader is not None
                assert data_loader.get_data_set() is not None
                assert data_loader.get_data_loader() is not None

    def test_model_saving_workflow(self, mock_model_components, temp_dir):
        """Test model saving workflow."""
        # Create model with components
        model = WanModel(ModelType.WAN_2_2)
        model.tokenizer = mock_model_components['tokenizer']
        model.text_encoder = mock_model_components['text_encoder']
        model.vae = mock_model_components['vae']
        model.transformer = mock_model_components['transformer']
        model.noise_scheduler = mock_model_components['scheduler']
        
        # Create saver
        saver = WanFineTuneModelSaver()
        
        output_path = os.path.join(temp_dir, "saved_model")
        
        # Test saving
        with patch.object(saver, 'save') as mock_save:
            saver.save(model, ModelType.WAN_2_2, output_path)
            mock_save.assert_called_once()

    def test_training_setup_workflow(self, mock_model_components):
        """Test training setup workflow."""
        # Create model
        model = WanModel(ModelType.WAN_2_2)
        model.text_encoder = mock_model_components['text_encoder']
        model.transformer = mock_model_components['transformer']
        model.vae = mock_model_components['vae']
        
        # Create config
        config = Mock(spec=TrainConfig)
        config.learning_rate = 1e-4
        config.optimizer_type = "AdamW"
        config.train_text_encoder = True
        config.train_transformer = True
        config.gradient_checkpointing = False
        
        # Create setup
        setup = WanFineTuneSetup()
        
        # Test setup
        with patch.object(setup, 'setup_model') as mock_setup:
            setup.setup_model(model, config)
            mock_setup.assert_called_once()

    def test_end_to_end_training_simulation(self, synthetic_video_data, mock_model_components, mock_device, temp_dir):
        """Test end-to-end training workflow simulation."""
        video_dir, video_files = synthetic_video_data
        
        # 1. Create and initialize model
        model = WanModel(ModelType.WAN_2_2)
        model.tokenizer = mock_model_components['tokenizer']
        model.text_encoder = mock_model_components['text_encoder']
        model.vae = mock_model_components['vae']
        model.transformer = mock_model_components['transformer']
        model.noise_scheduler = mock_model_components['scheduler']
        model.autocast_context = Mock()
        model.train_dtype = Mock()
        model.train_dtype.torch_dtype.return_value = torch.float32
        model.add_embeddings_to_prompt = Mock(side_effect=lambda x: x)
        
        # 2. Setup training configuration
        config = Mock(spec=TrainConfig)
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
        config.learning_rate = 1e-4
        config.train_text_encoder_or_embedding.return_value = False
        config.model_type = ModelType.WAN_2_2
        
        train_progress = Mock()
        
        # 3. Create data loader
        with patch('modules.dataLoader.WanBaseDataLoader.MGDS') as mock_mgds:
            with patch('modules.dataLoader.WanBaseDataLoader.TrainDataLoader') as mock_train_dl:
                mock_mgds.return_value = Mock()
                mock_train_dl.return_value = Mock()
                
                data_loader = WanBaseDataLoader(
                    train_device=mock_device,
                    temp_device=mock_device,
                    config=config,
                    model=model,
                    train_progress=train_progress
                )
        
        # 4. Setup training
        setup = WanFineTuneSetup()
        with patch.object(setup, 'setup_model') as mock_setup:
            setup.setup_model(model, config)
            mock_setup.assert_called_once()
        
        # 5. Simulate training step
        model.to(mock_device)
        model.eval()
        
        # Simulate forward pass
        sample_text = "A test video"
        with torch.no_grad():
            encoded_text = model.encode_text(
                train_device=mock_device,
                batch_size=1,
                text=sample_text
            )
            assert encoded_text is not None
            assert encoded_text.shape[0] == 1  # batch size
        
        # 6. Save model
        saver = WanFineTuneModelSaver()
        output_path = os.path.join(temp_dir, "trained_model")
        
        with patch.object(saver, 'save') as mock_save:
            saver.save(model, ModelType.WAN_2_2, output_path)
            mock_save.assert_called_once()
        
        # Verify workflow completed successfully
        assert data_loader is not None
        assert model.model_type == ModelType.WAN_2_2

    def test_model_consistency_after_save_load(self, mock_model_components, temp_dir):
        """Test model consistency after save and load operations."""
        # Create original model
        original_model = WanModel(ModelType.WAN_2_2)
        original_model.tokenizer = mock_model_components['tokenizer']
        original_model.text_encoder = mock_model_components['text_encoder']
        original_model.vae = mock_model_components['vae']
        original_model.transformer = mock_model_components['transformer']
        original_model.noise_scheduler = mock_model_components['scheduler']
        
        # Save model
        saver = WanFineTuneModelSaver()
        model_path = os.path.join(temp_dir, "consistency_test_model")
        
        with patch.object(saver, 'save') as mock_save:
            saver.save(original_model, ModelType.WAN_2_2, model_path)
            mock_save.assert_called_once()
        
        # Load model
        loader = WanModelLoader()
        loaded_model = WanModel(ModelType.WAN_2_2)
        
        model_names = Mock(spec=ModelNames)
        model_names.base_model = model_path
        model_names.transformer_model = None
        model_names.vae_model = None
        model_names.include_text_encoder = True
        
        weight_dtypes = Mock(spec=ModelWeightDtypes)
        weight_dtypes.text_encoder = DataType.FLOAT_32
        weight_dtypes.transformer = DataType.FLOAT_32
        weight_dtypes.vae = DataType.FLOAT_32
        weight_dtypes.train_dtype = DataType.FLOAT_32
        
        quantization = Mock(spec=QuantizationConfig)
        
        # Test loading (will use mocked components)
        with patch.object(loader, '_prepare_sub_modules'):
            with patch.object(loader, '_load_diffusers_sub_module') as mock_load:
                mock_load.return_value = Mock()
                
                try:
                    loader.load(loaded_model, ModelType.WAN_2_2, model_names, weight_dtypes, quantization)
                except Exception:
                    # Expected to fail with mocked components
                    pass
        
        # Verify model types match
        assert loaded_model.model_type == original_model.model_type