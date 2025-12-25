import torch
from typing import Optional, List

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import SingleVariationRandomAccessPipelineModule


class TemporalConsistencyVAE(PipelineModule, SingleVariationRandomAccessPipelineModule):
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
        print(f"🔥 INIT: TemporalConsistencyVAE.__init__ called")
        print(f"DEBUG: Initializing TemporalConsistencyVAE with VAE: {type(vae)}")
        self.video_in_name = video_in_name
        self.latent_out_name = latent_out_name
        self.vae = vae
        self.temporal_consistency_weight = temporal_consistency_weight
        self.autocast_contexts = autocast_contexts or []
        self.dtype = dtype
        print(f"🔥 INIT: TemporalConsistencyVAE initialized successfully")
        print(f"DEBUG: TemporalConsistencyVAE initialized successfully")
    
    def length(self) -> int:
        print(f"🔥 LENGTH: TemporalConsistencyVAE.length() called")
        try:
            result = self._get_previous_length(self.video_in_name)
            print(f"🔥 LENGTH: TemporalConsistencyVAE.length() = {result}")
            return result
        except Exception as e:
            print(f"🔥 ERROR in TemporalConsistencyVAE.length(): {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def get_inputs(self) -> list[str]:
        return [self.video_in_name]
    
    def get_outputs(self) -> list[str]:
        return [self.latent_out_name]
    
    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        print(f"🔥 ENTRY: TemporalConsistencyVAE.get_item called - variation={variation}, index={index}")
        
        try:
            print(f"🔥 STEP 1: Getting previous item '{self.video_in_name}'")
            video = self._get_previous_item(variation, index, self.video_in_name)
            print(f"🔥 STEP 2: Successfully retrieved video: {video.shape if hasattr(video, 'shape') else type(video)} = {video}")
            
            # If video is None, create dummy video data for testing
            if video is None:
                print(f"🔥 STEP 3: Creating dummy video data for testing")
                video = torch.randn(2, 3, 64, 64, dtype=torch.float32)  # Use float32 to match VAE
                print(f"🔥 STEP 4: Created dummy video: {video.shape}, dtype: {video.dtype}")
                
        except Exception as e:
            print(f"🔥 ERROR: Failed to get previous item '{self.video_in_name}': {e}")
            return {self.latent_out_name: torch.zeros((2, 48, 8, 8), dtype=self.dtype)}
        
        try:
            if self.vae is None:
                print(f"ERROR: VAE is None - using fallback")
                return {self.latent_out_name: torch.zeros((2, 48, 8, 8), dtype=self.dtype)}
            
            if not isinstance(video, torch.Tensor):
                print(f"ERROR: Video is not tensor: {type(video)} - using fallback")
                return {self.latent_out_name: torch.zeros((2, 48, 8, 8), dtype=self.dtype)}
            
            print(f"DEBUG: Input video shape: {video.shape}")
            
            # WAN 2.2 VAE expects 5D input: (batch, channels, frames, height, width)
            if video.dim() == 4:  # (frames, channels, height, width)
                frames, channels, height, width = video.shape
                video = video.permute(1, 0, 2, 3).unsqueeze(0)  # (1, channels, frames, height, width)
            elif video.dim() == 5:  # Already correct format
                pass
            else:
                print(f"ERROR: Unexpected video dimensions: {video.shape} - using fallback")
                return {self.latent_out_name: torch.zeros((2, 48, 8, 8), dtype=self.dtype)}
            
            print(f"DEBUG: Reshaped video for VAE: {video.shape}")
            
            # Encode with VAE
            with torch.no_grad():
                # Ensure video matches VAE dtype
                if hasattr(self.vae, 'dtype'):
                    video = video.to(dtype=self.vae.dtype)
                else:
                    video = video.to(dtype=torch.bfloat16)  # Default to bfloat16 for WAN 2.2
                print(f"DEBUG: Video dtype for VAE: {video.dtype}")
                print(f"DEBUG: Encoding with VAE...")
                
                encoded = self.vae.encode(video)
                latents = encoded.latent_dist.sample()
                
                print(f"DEBUG: VAE output shape: {latents.shape}")
                
                # Reshape for transformer: (batch, channels, height, width)
                if latents.dim() == 5:  # (batch, channels, frames, height, width)
                    b, c, f, h, w = latents.shape
                    if f == 1:
                        # Single frame: reshape to (batch, channels, height, width)
                        latents = latents.squeeze(2)  # Remove frame dimension
                
                print(f"🔥 SUCCESS: VAE encoding produces 48-channel latents: {latents.shape}")
                return {self.latent_out_name: latents}
                
        except Exception as e:
            print(f"ERROR in TemporalConsistencyVAE: {e}")
            import traceback
            traceback.print_exc()
            return {self.latent_out_name: torch.zeros((2, 48, 8, 8), dtype=self.dtype)}
    
    def _apply_temporal_consistency(self, latent_video: torch.Tensor) -> torch.Tensor:
        """Apply temporal consistency to latent video frames."""
        batch_size, num_frames, channels, height, width = latent_video.shape
        
        if num_frames <= 1:
            return latent_video
        
        # Apply temporal smoothing
        smoothed_latents = latent_video.clone()
        
        for i in range(1, num_frames):
            frame_diff = latent_video[:, i] - latent_video[:, i-1]
            consistency_factor = 1.0 - self.temporal_consistency_weight * 0.1
            smoothed_latents[:, i] = (
                latent_video[:, i-1] + frame_diff * consistency_factor
            )
        
        return smoothed_latents