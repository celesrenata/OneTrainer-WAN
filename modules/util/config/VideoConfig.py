from typing import Any

from modules.util.config.BaseConfig import BaseConfig


class VideoConfig(BaseConfig):
    # Video data processing parameters
    max_frames: int
    frame_sample_strategy: str  # "uniform", "random", "keyframe"
    target_fps: float
    max_duration: float  # seconds
    min_resolution: list[int]  # [width, height]
    max_resolution: list[int]  # [width, height]
    
    # Temporal consistency parameters
    temporal_consistency_weight: float
    use_temporal_attention: bool
    
    # Memory management for video training
    spatial_compression_ratio: int
    temporal_compression_ratio: int
    video_batch_size_multiplier: float  # Multiplier for batch size when processing video
    
    # Video-specific training parameters
    frame_dropout_probability: float  # Probability of dropping frames during training
    temporal_augmentation: bool  # Enable temporal augmentations
    
    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def default_values():
        data = []
        
        # Video data processing parameters
        data.append(("max_frames", 16, int, False))
        data.append(("frame_sample_strategy", "uniform", str, False))
        data.append(("target_fps", 24.0, float, False))
        data.append(("max_duration", 10.0, float, False))
        data.append(("min_resolution", [256, 256], list, False))
        data.append(("max_resolution", [1024, 1024], list, False))
        
        # Temporal consistency parameters
        data.append(("temporal_consistency_weight", 1.0, float, False))
        data.append(("use_temporal_attention", True, bool, False))
        
        # Memory management for video training
        data.append(("spatial_compression_ratio", 8, int, False))
        data.append(("temporal_compression_ratio", 4, int, False))
        data.append(("video_batch_size_multiplier", 0.5, float, False))
        
        # Video-specific training parameters
        data.append(("frame_dropout_probability", 0.0, float, False))
        data.append(("temporal_augmentation", False, bool, False))
        
        return VideoConfig(data)