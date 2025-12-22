import torch
from typing import Optional, Union
from enum import Enum

from mgds.PipelineModule import PipelineModule
from modules.util.video_util import FrameSamplingStrategy, sample_video_frames


class VideoFrameSampler(PipelineModule):
    """
    MGDS pipeline module for sampling frames from video data.
    
    This module handles different frame sampling strategies (uniform, random, keyframe)
    and ensures consistent frame counts across the dataset.
    """
    
    def __init__(
        self,
        video_in_name: str,
        video_path_in_name: str,
        video_out_name: str,
        target_frames_in_name: str,
        sampling_strategy: Union[str, FrameSamplingStrategy] = FrameSamplingStrategy.UNIFORM,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.video_in_name = video_in_name
        self.video_path_in_name = video_path_in_name
        self.video_out_name = video_out_name
        self.target_frames_in_name = target_frames_in_name
        
        # Convert string to enum if needed
        if isinstance(sampling_strategy, str):
            self.sampling_strategy = FrameSamplingStrategy(sampling_strategy.lower())
        else:
            self.sampling_strategy = sampling_strategy
            
        self.seed = seed
    
    def length(self) -> int:
        return self._get_previous_length(self.video_in_name)
    
    def get_inputs(self) -> list[str]:
        return [self.video_in_name, self.video_path_in_name, self.target_frames_in_name]
    
    def get_outputs(self) -> list[str]:
        return [self.video_out_name]
    
    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        video = self._get_previous_item(variation, index, self.video_in_name)
        video_path = self._get_previous_item(variation, index, self.video_path_in_name)
        target_frames = self._get_previous_item(variation, index, self.target_frames_in_name)
        
        # Ensure video is in the correct format
        if video.dim() == 5:  # [batch, frames, channels, height, width]
            video = video.squeeze(0)  # Remove batch dimension
        
        num_frames, channels, height, width = video.shape
        
        # If we already have the target number of frames, return as-is
        if num_frames == target_frames:
            return {self.video_out_name: video}
        
        # Sample frame indices based on strategy
        try:
            frame_indices = sample_video_frames(
                video_path, 
                target_frames, 
                self.sampling_strategy, 
                self.seed
            )
        except Exception as e:
            # Fallback to uniform sampling if video path sampling fails
            print(f"Warning: Frame sampling failed for {video_path}, using uniform fallback: {e}")
            frame_indices = self._uniform_sample_indices(num_frames, target_frames)
        
        # Ensure indices are within bounds
        frame_indices = [min(idx, num_frames - 1) for idx in frame_indices]
        
        # Sample the frames
        sampled_video = video[frame_indices]
        
        # Pad with last frame if we don't have enough frames
        if len(frame_indices) < target_frames:
            last_frame = sampled_video[-1:].repeat(target_frames - len(frame_indices), 1, 1, 1)
            sampled_video = torch.cat([sampled_video, last_frame], dim=0)
        
        return {self.video_out_name: sampled_video}
    
    def _uniform_sample_indices(self, total_frames: int, target_frames: int) -> list[int]:
        """Fallback uniform sampling when video path is not available."""
        if target_frames >= total_frames:
            return list(range(total_frames))
        
        step = total_frames / target_frames
        indices = [int(i * step) for i in range(target_frames)]
        return [min(idx, total_frames - 1) for idx in indices]