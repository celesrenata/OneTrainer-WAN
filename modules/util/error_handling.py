"""
Error handling utilities for WAN 2.2 implementation.
Provides consistent error handling and logging patterns.
"""
import logging
import traceback
from typing import Optional, Any, Callable
from functools import wraps


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging for WAN 2.2 components.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def safe_execute(func: Callable, *args, default_return: Any = None, **kwargs) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        default_return: Default return value on error
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error executing {func.__name__}: {e}")
        logger.debug(traceback.format_exc())
        return default_return


def validate_model_components(model) -> bool:
    """
    Validate that model components are properly initialized.
    
    Args:
        model: Model to validate
        
    Returns:
        True if model is valid
    """
    required_components = ['tokenizer', 'text_encoder', 'vae', 'transformer']
    
    for component in required_components:
        if not hasattr(model, component) or getattr(model, component) is None:
            raise ValueError(f"Model missing required component: {component}")
    
    return True


def validate_training_config(config) -> bool:
    """
    Validate training configuration parameters.
    
    Args:
        config: Training configuration
        
    Returns:
        True if config is valid
    """
    required_fields = ['batch_size', 'learning_rate', 'model_type']
    
    for field in required_fields:
        if not hasattr(config, field):
            raise ValueError(f"Config missing required field: {field}")
    
    # Validate ranges
    if config.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    
    if config.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    
    return True


def handle_cuda_errors(func: Callable) -> Callable:
    """
    Decorator to handle CUDA-related errors gracefully.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "CUDA" in str(e) or "out of memory" in str(e):
                logger = logging.getLogger(__name__)
                logger.warning(f"CUDA error in {func.__name__}: {e}")
                logger.info("Falling back to CPU computation")
                
                # Try to clear CUDA cache
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                
                # Re-raise the error for caller to handle
                raise
            else:
                raise
    
    return wrapper


def validate_video_data(video_path: str) -> bool:
    """
    Validate video data file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if video is valid
    """
    import os
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Check file extension
    valid_extensions = ['.mp4', '.avi', '.mov', '.webm']
    file_ext = os.path.splitext(video_path)[1].lower()
    
    if file_ext not in valid_extensions:
        raise ValueError(f"Unsupported video format: {file_ext}")
    
    # Check file size
    file_size = os.path.getsize(video_path)
    if file_size == 0:
        raise ValueError(f"Video file is empty: {video_path}")
    
    return True


class WanError(Exception):
    """Base exception class for WAN 2.2 errors."""
    pass


class ModelLoadError(WanError):
    """Error loading WAN 2.2 model."""
    pass


class DataLoadError(WanError):
    """Error loading training data."""
    pass


class TrainingError(WanError):
    """Error during training process."""
    pass


class ValidationError(WanError):
    """Error during validation."""
    pass
