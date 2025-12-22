"""
Configuration validation utilities for WAN 2.2.
Provides validation functions for training configurations.
"""
from typing import Dict, Any, List, Tuple
from modules.util.enum.ModelType import ModelType


def validate_wan_config(config) -> List[str]:
    """
    Validate WAN 2.2 specific configuration parameters.
    
    Args:
        config: Training configuration object
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check model type
    if not hasattr(config, 'model_type') or config.model_type != ModelType.WAN_2_2:
        errors.append("Model type must be WAN_2_2")
    
    # Validate video parameters
    if hasattr(config, 'target_frames'):
        if config.target_frames <= 0 or config.target_frames > 64:
            errors.append("target_frames must be between 1 and 64")
    
    if hasattr(config, 'frame_sample_strategy'):
        valid_strategies = ['uniform', 'random', 'keyframe']
        if config.frame_sample_strategy not in valid_strategies:
            errors.append(f"frame_sample_strategy must be one of {valid_strategies}")
    
    if hasattr(config, 'temporal_consistency_weight'):
        if config.temporal_consistency_weight < 0 or config.temporal_consistency_weight > 10:
            errors.append("temporal_consistency_weight must be between 0 and 10")
    
    # Validate resolution parameters
    if hasattr(config, 'min_video_resolution'):
        min_res = config.min_video_resolution
        if not isinstance(min_res, (tuple, list)) or len(min_res) != 2:
            errors.append("min_video_resolution must be a tuple of (width, height)")
        elif min_res[0] < 64 or min_res[1] < 64:
            errors.append("min_video_resolution must be at least (64, 64)")
    
    if hasattr(config, 'max_video_resolution'):
        max_res = config.max_video_resolution
        if not isinstance(max_res, (tuple, list)) or len(max_res) != 2:
            errors.append("max_video_resolution must be a tuple of (width, height)")
        elif max_res[0] > 2048 or max_res[1] > 2048:
            errors.append("max_video_resolution should not exceed (2048, 2048)")
    
    # Validate duration
    if hasattr(config, 'max_video_duration'):
        if config.max_video_duration <= 0 or config.max_video_duration > 60:
            errors.append("max_video_duration must be between 0 and 60 seconds")
    
    return errors


def validate_training_parameters(config) -> List[str]:
    """
    Validate general training parameters.
    
    Args:
        config: Training configuration object
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Batch size
    if hasattr(config, 'batch_size'):
        if config.batch_size <= 0 or config.batch_size > 32:
            errors.append("batch_size must be between 1 and 32")
    
    # Learning rate
    if hasattr(config, 'learning_rate'):
        if config.learning_rate <= 0 or config.learning_rate > 1:
            errors.append("learning_rate must be between 0 and 1")
    
    # Gradient accumulation
    if hasattr(config, 'gradient_accumulation_steps'):
        if config.gradient_accumulation_steps <= 0:
            errors.append("gradient_accumulation_steps must be positive")
    
    # Epochs
    if hasattr(config, 'max_epochs'):
        if config.max_epochs <= 0 or config.max_epochs > 1000:
            errors.append("max_epochs must be between 1 and 1000")
    
    return errors


def apply_wan_defaults(config) -> None:
    """
    Apply WAN 2.2 default values to configuration.
    
    Args:
        config: Configuration object to update
    """
    defaults = {
        'target_frames': 16,
        'frame_sample_strategy': 'uniform',
        'temporal_consistency_weight': 1.0,
        'min_video_resolution': (256, 256),
        'max_video_resolution': (1024, 1024),
        'max_video_duration': 10.0,
        'video_fps': 24,
        'batch_size': 1,
        'learning_rate': 1e-4,
        'gradient_accumulation_steps': 4,
        'max_epochs': 10
    }
    
    for key, value in defaults.items():
        if not hasattr(config, key):
            setattr(config, key, value)


def get_recommended_config(gpu_memory_gb: int = 8) -> Dict[str, Any]:
    """
    Get recommended configuration based on available GPU memory.
    
    Args:
        gpu_memory_gb: Available GPU memory in GB
        
    Returns:
        Recommended configuration dictionary
    """
    if gpu_memory_gb >= 24:
        return {
            'batch_size': 4,
            'target_frames': 24,
            'max_video_resolution': (1024, 1024),
            'gradient_accumulation_steps': 2,
            'mixed_precision': True
        }
    elif gpu_memory_gb >= 16:
        return {
            'batch_size': 2,
            'target_frames': 16,
            'max_video_resolution': (768, 768),
            'gradient_accumulation_steps': 4,
            'mixed_precision': True
        }
    elif gpu_memory_gb >= 8:
        return {
            'batch_size': 1,
            'target_frames': 16,
            'max_video_resolution': (512, 512),
            'gradient_accumulation_steps': 8,
            'mixed_precision': True
        }
    else:
        return {
            'batch_size': 1,
            'target_frames': 8,
            'max_video_resolution': (256, 256),
            'gradient_accumulation_steps': 16,
            'mixed_precision': False
        }
