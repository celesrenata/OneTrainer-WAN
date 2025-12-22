import torch
from typing import Optional, List

from mgds.PipelineModule import PipelineModule


class TemporalConsistencyVAE(PipelineModule):
    """
    MGDS pipeline module for encoding video frames with temporal consistency.
    
    This module ensures that VAE encoding maintains temporal relationships
    between frames by processing them in sequence and applying consistency
    constraints.
    """
    
    def __init__(
        self,
        video_in_name: str,
        latent_out_name: str,
        vae,
        temporal_consistency_weight: float = 1.0,
        autocast_contexts: Optional[List[torch.autocast]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.video_in_name = video_in_name
        self.latent_out_name = latent_out_name
        self.vae = vae
        self.temporal_consistency_weight = temporal_consistency_weight
        self.autocast_contexts = autocast_contexts or []
        self.dtype = dtype
    
    def length(self) -> int:
        return self._get_previous_length(self.video_in_name)
    
    def get_inputs(self) -> list[str]:
        return [self.video_in_name]
    
    def get_outputs(self) -> list[str]:
        return [self.latent_out_name]
    
    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        video = self._get_previous_item(variation, index, self.video_in_name)
        
        # Ensure video is in the correct format [batch, frames, channels, height, width]
        if video.dim() == 4:  # [frames, channels, height, width]
            video = video.unsqueeze(0)  # Add batch dimension
        
        batch_size, num_frames, channels, height, width = video.shape
        
        # Encode frames with temporal consistency
        with torch.no_grad():
            # Apply autocast contexts if provided
            autocast_context = torch.autocast('cuda', enabled=False)
            if self.autocast_contexts:
                autocast_context = self.autocast_contexts[0]
            
            with autocast_context:
                # Reshape for VAE processing: [batch * frames, channels, height, width]
                video_reshaped = video.reshape(batch_size * num_frames, channels, height, width)
                
                # Convert to target dtype
                video_reshaped = video_reshaped.to(dtype=self.dtype)
                
                # Encode all frames
                latent_dist = self.vae.encode(video_reshaped)
                
                # Get latent dimensions
                latent_sample = latent_dist.latent_dist.sample()
                latent_channels, latent_height, latent_width = latent_sample.shape[1:]
                
                # Reshape back to video format: [batch, frames, latent_channels, latent_height, latent_width]
                latent_video = latent_sample.reshape(
                    batch_size, num_frames, latent_channels, latent_height, latent_width
                )
                
                # Apply temporal consistency if weight > 0
                if self.temporal_consistency_weight > 0 and num_frames > 1:
                    latent_video = self._apply_temporal_consistency(latent_video)
        
        # Remove batch dimension if it was added
        if batch_size == 1:
            latent_video = latent_video.squeeze(0)
        
        return {self.latent_out_name: latent_video}
    
    def _apply_temporal_consistency(self, latent_video: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal consistency to latent video frames.
        
        This is a simplified implementation that applies smoothing between
        consecutive frames to maintain temporal coherence.
        """
        batch_size, num_frames, channels, height, width = latent_video.shape
        
        if num_frames <= 1:
            return latent_video
        
        # Apply temporal smoothing
        smoothed_latents = latent_video.clone()
        
        for i in range(1, num_frames):
            # Calculate difference between consecutive frames
            frame_diff = latent_video[:, i] - latent_video[:, i-1]
            
            # Apply consistency weight to reduce temporal discontinuities
            consistency_factor = 1.0 - self.temporal_consistency_weight * 0.1
            smoothed_latents[:, i] = (
                latent_video[:, i-1] + frame_diff * consistency_factor
            )
        
        return smoothed_latents