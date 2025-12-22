"""
Pytest configuration and fixtures for OneTrainer tests.
"""
import pytest
import torch
import tempfile
import os
from unittest.mock import Mock, MagicMock

from modules.util.enum.ModelType import ModelType
from modules.util.enum.DataType import DataType
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.config.TrainConfig import TrainConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_device():
    """Mock torch device for testing."""
    return torch.device('cpu')


@pytest.fixture
def mock_train_config():
    """Create a mock training configuration for testing."""
    config = Mock(spec=TrainConfig)
    config.batch_size = 1
    config.multi_gpu = False
    config.masked_training = False
    config.latent_caching = False
    config.debug_mode = False
    config.cache_dir = "/tmp/cache"
    config.debug_dir = "/tmp/debug"
    config.seed = 42
    config.target_frames = 16
    config.frame_sample_strategy = "uniform"
    config.temporal_consistency_weight = 1.0
    config.min_video_resolution = (256, 256)
    config.max_video_resolution = (1024, 1024)
    config.max_video_duration = 10.0
    config.model_type = ModelType.WAN_2_2
    
    # Mock methods
    config.train_text_encoder_or_embedding.return_value = False
    
    return config


@pytest.fixture
def mock_weight_dtypes():
    """Create mock weight dtypes for testing."""
    dtypes = Mock(spec=ModelWeightDtypes)
    dtypes.text_encoder = DataType.FLOAT_32
    dtypes.transformer = DataType.FLOAT_32
    dtypes.vae = DataType.FLOAT_32
    dtypes.train_dtype = DataType.FLOAT_32
    return dtypes


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.return_value = Mock()
    tokenizer.return_value.input_ids = torch.randint(0, 1000, (1, 77))
    return tokenizer


@pytest.fixture
def mock_text_encoder():
    """Create a mock text encoder for testing."""
    text_encoder = Mock()
    text_encoder.return_value = [torch.randn(1, 77, 768)]
    text_encoder.device = torch.device('cpu')
    return text_encoder


@pytest.fixture
def mock_vae():
    """Create a mock VAE for testing."""
    vae = Mock()
    vae.encode.return_value = Mock()
    vae.encode.return_value.latent_dist = Mock()
    vae.encode.return_value.latent_dist.sample.return_value = torch.randn(1, 4, 32, 32)
    vae.decode.return_value = Mock()
    vae.decode.return_value.sample = torch.randn(1, 3, 256, 256)
    return vae


@pytest.fixture
def mock_transformer():
    """Create a mock transformer for testing."""
    transformer = Mock()
    transformer.return_value = Mock()
    transformer.return_value.sample = torch.randn(1, 4, 32, 32)
    return transformer


@pytest.fixture
def mock_scheduler():
    """Create a mock noise scheduler for testing."""
    scheduler = Mock()
    scheduler.timesteps = torch.linspace(1000, 0, 50)
    return scheduler


@pytest.fixture
def sample_video_tensor():
    """Create a sample video tensor for testing."""
    # Shape: (batch, channels, frames, height, width)
    return torch.randn(1, 3, 16, 256, 256)


@pytest.fixture
def sample_text_prompt():
    """Sample text prompt for testing."""
    return "A beautiful landscape with mountains and trees"


@pytest.fixture
def skip_if_no_gpu():
    """Skip test if no GPU is available."""
    def _skip_if_no_gpu():
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
    return _skip_if_no_gpu


@pytest.fixture
def skip_if_no_rocm():
    """Skip test if ROCm is not available."""
    def _skip_if_no_rocm():
        if not (torch.cuda.is_available() and torch.version.hip is not None):
            pytest.skip("ROCm not available")
    return _skip_if_no_rocm