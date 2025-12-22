"""
Unit tests for WanModel class.
Tests WanModel initialization, device movement, and core functionality.
"""
import pytest
import torch
from unittest.mock import Mock, patch

from modules.model.WanModel import WanModel, WanModelEmbedding
from modules.util.enum.ModelType import ModelType
from modules.util.enum.DataType import DataType


class TestWanModel:
    """Test cases for WanModel class."""

    def test_wan_model_initialization(self):
        """Test WanModel initializes with correct default values."""
        model = WanModel(ModelType.WAN_2_2)
        
        assert model.model_type == ModelType.WAN_2_2
        assert model.tokenizer is None
        assert model.text_encoder is None
        assert model.vae is None
        assert model.transformer is None
        assert model.noise_scheduler is None
        assert model.embedding is None
        assert model.additional_embeddings == []
        assert model.text_encoder_lora is None
        assert model.transformer_lora is None
        assert model.text_encoder_train_dtype == DataType.FLOAT_32
        assert model.transformer_train_dtype == DataType.FLOAT_32

    def test_wan_model_device_movement(self, mock_device, mock_vae, mock_text_encoder, mock_transformer):
        """Test device movement methods work correctly."""
        model = WanModel(ModelType.WAN_2_2)
        model.vae = mock_vae
        model.text_encoder = mock_text_encoder
        model.transformer = mock_transformer
        
        # Test individual component movement
        model.vae_to(mock_device)
        mock_vae.to.assert_called_with(device=mock_device)
        
        model.text_encoder_to(mock_device)
        mock_text_encoder.to.assert_called_with(device=mock_device)
        
        model.transformer_to(mock_device)
        mock_transformer.to.assert_called_with(device=mock_device)
        
        # Test full model movement
        model.to(mock_device)
        assert mock_vae.to.call_count >= 2  # Called in vae_to and to methods
        assert mock_text_encoder.to.call_count >= 2
        assert mock_transformer.to.call_count >= 2

    def test_wan_model_eval_mode(self, mock_vae, mock_text_encoder, mock_transformer):
        """Test eval mode is set correctly on all components."""
        model = WanModel(ModelType.WAN_2_2)
        model.vae = mock_vae
        model.text_encoder = mock_text_encoder
        model.transformer = mock_transformer
        
        model.eval()
        
        mock_vae.eval.assert_called_once()
        mock_text_encoder.eval.assert_called_once()
        mock_transformer.eval.assert_called_once()

    def test_wan_model_adapters(self):
        """Test adapters method returns correct LoRA adapters."""
        model = WanModel(ModelType.WAN_2_2)
        
        # No adapters initially
        assert model.adapters() == []
        
        # Add mock adapters
        mock_text_lora = Mock()
        mock_transformer_lora = Mock()
        model.text_encoder_lora = mock_text_lora
        model.transformer_lora = mock_transformer_lora
        
        adapters = model.adapters()
        assert len(adapters) == 2
        assert mock_text_lora in adapters
        assert mock_transformer_lora in adapters

    def test_wan_model_embeddings(self):
        """Test embedding management methods."""
        model = WanModel(ModelType.WAN_2_2)
        
        # Create mock embeddings
        embedding1 = Mock(spec=WanModelEmbedding)
        embedding1.text_encoder_embedding = Mock()
        embedding2 = Mock(spec=WanModelEmbedding)
        embedding2.text_encoder_embedding = Mock()
        
        model.embedding = embedding1
        model.additional_embeddings = [embedding2]
        
        # Test all_embeddings
        all_embeddings = model.all_embeddings()
        assert len(all_embeddings) == 2
        assert embedding1 in all_embeddings
        assert embedding2 in all_embeddings
        
        # Test all_text_encoder_embeddings
        text_embeddings = model.all_text_encoder_embeddings()
        assert len(text_embeddings) == 2
        assert embedding1.text_encoder_embedding in text_embeddings
        assert embedding2.text_encoder_embedding in text_embeddings

    def test_encode_text_basic(self, mock_tokenizer, mock_text_encoder, mock_device):
        """Test basic text encoding functionality."""
        model = WanModel(ModelType.WAN_2_2)
        model.tokenizer = mock_tokenizer
        model.text_encoder = mock_text_encoder
        model.train_dtype = DataType.FLOAT_32
        
        # Mock the tokenizer output
        mock_tokenizer.return_value = Mock()
        mock_tokenizer.return_value.input_ids = torch.randint(0, 1000, (1, 77))
        
        # Mock text encoder output
        mock_text_encoder.return_value = [torch.randn(1, 77, 768)]
        mock_text_encoder.device = mock_device
        
        result = model.encode_text(
            train_device=mock_device,
            batch_size=1,
            text="test prompt"
        )
        
        assert result is not None
        assert result.shape == (1, 77, 768)
        mock_tokenizer.assert_called_once()
        mock_text_encoder.assert_called_once()

    def test_pack_unpack_latents(self):
        """Test video latent packing and unpacking."""
        model = WanModel(ModelType.WAN_2_2)
        
        # Create sample video latents (batch, channels, frames, height, width)
        latents = torch.randn(2, 4, 16, 32, 32)
        
        # Test packing
        packed = model.pack_latents(latents)
        expected_shape = (2, 4 * 16, 32, 32)  # channels * frames
        assert packed.shape == expected_shape
        
        # Test unpacking
        unpacked = model.unpack_latents(packed, frames=16, height=32, width=32)
        assert unpacked.shape == latents.shape
        
        # Verify data integrity (should be approximately equal due to reshape operations)
        assert torch.allclose(latents, unpacked, atol=1e-6)

    @patch('modules.model.WanModel.DiffusionPipeline')
    def test_create_pipeline(self, mock_pipeline_class, mock_vae, mock_text_encoder, mock_transformer, mock_scheduler, mock_tokenizer):
        """Test pipeline creation."""
        model = WanModel(ModelType.WAN_2_2)
        model.vae = mock_vae
        model.text_encoder = mock_text_encoder
        model.transformer = mock_transformer
        model.noise_scheduler = mock_scheduler
        model.tokenizer = mock_tokenizer
        
        mock_pipeline_instance = Mock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance
        
        pipeline = model.create_pipeline()
        
        assert pipeline == mock_pipeline_instance
        mock_pipeline_class.from_pretrained.assert_called_once()


class TestWanModelEmbedding:
    """Test cases for WanModelEmbedding class."""

    def test_wan_model_embedding_initialization(self):
        """Test WanModelEmbedding initializes correctly."""
        uuid = "test-uuid"
        vector = torch.randn(768)
        placeholder = "test_token"
        
        embedding = WanModelEmbedding(
            uuid=uuid,
            text_encoder_vector=vector,
            placeholder=placeholder,
            is_output_embedding=True
        )
        
        assert embedding.text_encoder_embedding is not None
        assert embedding.text_encoder_embedding.uuid == uuid
        assert embedding.text_encoder_embedding.placeholder == placeholder
        assert embedding.text_encoder_embedding.is_output_embedding is True
        assert torch.equal(embedding.text_encoder_embedding.vector, vector)

    def test_wan_model_embedding_none_vector(self):
        """Test WanModelEmbedding handles None vector correctly."""
        embedding = WanModelEmbedding(
            uuid="test-uuid",
            text_encoder_vector=None,
            placeholder="test_token",
            is_output_embedding=False
        )
        
        assert embedding.text_encoder_embedding is not None
        assert embedding.text_encoder_embedding.vector is None
        assert embedding.text_encoder_embedding.is_output_embedding is False