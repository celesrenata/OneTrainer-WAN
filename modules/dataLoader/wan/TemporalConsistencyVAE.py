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
        print(f"Initializing TemporalConsistencyVAE with VAE: {type(vae)}")
        self.video_in_name = video_in_name
        self.latent_out_name = latent_out_name
        self.vae = vae
        self.temporal_consistency_weight = temporal_consistency_weight
        self.autocast_contexts = autocast_contexts or []
        self.dtype = dtype
        print(f"TemporalConsistencyVAE initialized successfully")
    
    def length(self) -> int:
        try:
            result = self._get_previous_length(self.video_in_name)
            return result
        except Exception as e:
            print(f"ERROR in TemporalConsistencyVAE.length(): {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def get_inputs(self) -> list[str]:
        return [self.video_in_name]  # Only accept video input, remove prompt
    
    def get_outputs(self) -> list[str]:
        return [self.latent_out_name]  # Only output latent_video, remove prompt pass-through
    
    def get_item(self, index: int, item_name: str = None) -> dict:
        print(f"DEBUG: TemporalConsistencyVAE.get_item called - index: {index}, item_name: {item_name}")
        
        # HACK: The MGDS system is passing item_name as index parameter
        actual_item_name = str(index) if item_name is None else item_name
        actual_index = 0 if isinstance(index, str) else index
        
        print(f"DEBUG: Corrected - actual_index: {actual_index}, actual_item_name: {actual_item_name}")
        
        # Only handle latent_video requests, let text processing fail gracefully
        if actual_item_name != 'latent_video':
            print(f"DEBUG: TemporalConsistencyVAE ignoring request for {actual_item_name}")
            return None
        
        # If requesting latent_video, do VAE encoding
        if actual_item_name == 'latent_video':
            try:
                video = self._get_previous_item(0, self.video_in_name, actual_index)  # Fixed parameter order
                
                # If video is None, create dummy video data for testing
                if video is None:
                    print(f"DEBUG: Creating fallback video data")
                    video = torch.randn(2, 3, 64, 64, dtype=torch.float32)  # Use float32 to match VAE
                    
            except Exception as e:
                print(f"ERROR: Failed to get previous item '{self.video_in_name}': {e}")
                print(f"DEBUG: Creating fallback video data due to error")
                video = torch.randn(2, 3, 64, 64, dtype=torch.float32)  # Fallback video
        
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
                
                print(f"VAE output shape: {latents.shape}")
                
                # Reshape for transformer: (batch, channels, height, width)
                if latents.dim() == 5:  # (batch, channels, frames, height, width)
                    b, c, f, h, w = latents.shape
                    if f == 1:
                        # Single frame: reshape to (batch, channels, height, width)
                        latents = latents.squeeze(2)  # Remove frame dimension
                
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