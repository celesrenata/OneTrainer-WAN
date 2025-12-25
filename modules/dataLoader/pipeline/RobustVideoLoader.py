"""
RobustVideoLoader with comprehensive error handling and validation.

This module provides robust video loading capabilities with extensive error handling,
file validation, and graceful fallback mechanisms for the MGDS pipeline.
"""

import logging
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import av

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
from .DataFlowValidator import DataFlowValidator, ValidationResult, VideoMetadata


@dataclass
class VideoLoadResult:
    """Result of video loading operation."""
    success: bool
    video_tensor: Optional[torch.Tensor]
    error_message: Optional[str]
    metadata: Dict[str, Any]
    processing_time: float
    fallback_used: bool = False


class RobustVideoLoader(PipelineModule, RandomAccessPipelineModule):
    """
    Robust video loader with comprehensive error handling.
    
    Features:
    - Video file integrity checking before processing
    - Graceful handling of corrupted/invalid video files
    - Fallback mechanisms for unsupported formats
    - Detailed error reporting and logging
    - Performance monitoring and caching
    """
    
    def __init__(
        self,
        path_in_name: str,
        target_frame_count_in_name: str,
        video_out_name: str,
        range_min: float = 0.0,
        range_max: float = 1.0,
        target_frame_rate: float = 24.0,
        supported_extensions: Optional[set] = None,
        dtype: Optional[torch.dtype] = None,
        enable_validation: bool = True,
        max_file_size_mb: float = 1000.0,  # 1GB max
        timeout_seconds: float = 30.0,
        enable_caching: bool = True
    ):
        super().__init__()
        
        self.path_in_name = path_in_name
        self.target_frame_count_in_name = target_frame_count_in_name
        self.video_out_name = video_out_name
        
        self.range_min = range_min
        self.range_max = range_max
        self.target_frame_rate = target_frame_rate
        
        # Default supported extensions if none provided
        if supported_extensions is None:
            self.supported_extensions = {
                '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v',
                '.mpg', '.mpeg', '.3gp', '.ogv'
            }
        else:
            self.supported_extensions = supported_extensions
        
        self.dtype = dtype or torch.float32
        self.enable_validation = enable_validation
        self.max_file_size_mb = max_file_size_mb
        self.timeout_seconds = timeout_seconds
        self.enable_caching = enable_caching
        
        # Caching and performance tracking
        self.duration_cache = {} if enable_caching else None
        self.metadata_cache = {} if enable_caching else None
        self.load_statistics = {
            'total_attempts': 0,
            'successful_loads': 0,
            'failed_loads': 0,
            'fallback_uses': 0,
            'cache_hits': 0,
            'validation_failures': 0
        }
        
        # Setup logging
        self.logger = logging.getLogger("RobustVideoLoader")
        self.logger.setLevel(logging.DEBUG)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Initialize validator
        if self.enable_validation:
            self.validator = DataFlowValidator(enable_gpu_monitoring=False)
        else:
            self.validator = None
    
    def length(self) -> int:
        """Get the length from the previous module."""
        return self._get_previous_length(self.path_in_name)
    
    def get_inputs(self) -> List[str]:
        """Get required input names."""
        return [self.path_in_name, self.target_frame_count_in_name]
    
    def get_outputs(self) -> List[str]:
        """Get output names."""
        return [self.video_out_name]
    
    def get_item(self, variation: int, index: int, requested_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Load video with comprehensive error handling and validation.
        
        Args:
            variation: Variation index for randomization
            index: Sample index
            requested_name: Specific output name requested
            
        Returns:
            Dictionary containing video tensor or fallback data
        """
        start_time = time.time()
        self.load_statistics['total_attempts'] += 1
        
        try:
            # Get inputs from previous modules
            path = self._get_previous_item(variation, self.path_in_name, index)
            target_frame_count = self._get_previous_item(variation, self.target_frame_count_in_name, index)
            
            if path is None:
                self.logger.error(f"Video path is None for index {index}")
                return self._create_fallback_result(index, "Video path is None")
            
            if target_frame_count is None:
                self.logger.warning(f"Target frame count is None for index {index}, using default")
                target_frame_count = 8
            
            target_frame_count = int(target_frame_count)
            
            self.logger.debug(f"Loading video: {path} (target frames: {target_frame_count})")
            
            # Load video with error handling
            load_result = self._load_video_safe(path, target_frame_count, variation, index)
            
            # Update statistics
            if load_result.success:
                self.load_statistics['successful_loads'] += 1
            else:
                self.load_statistics['failed_loads'] += 1
                if load_result.fallback_used:
                    self.load_statistics['fallback_uses'] += 1
            
            # Return result
            result = {self.video_out_name: load_result.video_tensor}
            
            # Log processing time
            processing_time = time.time() - start_time
            self.logger.debug(f"Video loading completed in {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.load_statistics['failed_loads'] += 1
            self.load_statistics['fallback_uses'] += 1
            
            self.logger.error(f"Unexpected error loading video for index {index}: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            
            return self._create_fallback_result(index, f"Unexpected error: {str(e)}")
    
    def _load_video_safe(
        self, 
        path: str, 
        target_frame_count: int, 
        variation: int, 
        index: int
    ) -> VideoLoadResult:
        """
        Safely load video with comprehensive error handling.
        
        Args:
            path: Path to video file
            target_frame_count: Number of frames to extract
            variation: Variation index for randomization
            index: Sample index
            
        Returns:
            VideoLoadResult with success status and data
        """
        start_time = time.time()
        
        try:
            # Step 1: Validate file existence and accessibility
            if not os.path.exists(path):
                return VideoLoadResult(
                    success=False,
                    video_tensor=self._create_fallback_tensor(target_frame_count),
                    error_message=f"Video file does not exist: {path}",
                    metadata={'path': path, 'stage': 'file_check'},
                    processing_time=time.time() - start_time,
                    fallback_used=True
                )
            
            # Step 2: Check file extension
            ext = os.path.splitext(path)[1].lower()
            if ext not in self.supported_extensions:
                self.logger.warning(f"Unsupported video extension: {ext} for file {path}")
                return VideoLoadResult(
                    success=False,
                    video_tensor=self._create_fallback_tensor(target_frame_count),
                    error_message=f"Unsupported video extension: {ext}",
                    metadata={'path': path, 'extension': ext, 'stage': 'extension_check'},
                    processing_time=time.time() - start_time,
                    fallback_used=True
                )
            
            # Step 3: Validate file with DataFlowValidator if enabled
            if self.validator:
                validation_result = self.validator.validate_video_file(path)
                if not validation_result.is_valid:
                    self.load_statistics['validation_failures'] += 1
                    self.logger.warning(f"Video validation failed: {validation_result.error_message}")
                    return VideoLoadResult(
                        success=False,
                        video_tensor=self._create_fallback_tensor(target_frame_count),
                        error_message=f"Video validation failed: {validation_result.error_message}",
                        metadata={'path': path, 'validation': validation_result.metadata, 'stage': 'validation'},
                        processing_time=time.time() - start_time,
                        fallback_used=True
                    )
            
            # Step 4: Check file size
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                self.logger.warning(f"Video file too large: {file_size_mb:.1f}MB > {self.max_file_size_mb}MB")
                return VideoLoadResult(
                    success=False,
                    video_tensor=self._create_fallback_tensor(target_frame_count),
                    error_message=f"Video file too large: {file_size_mb:.1f}MB",
                    metadata={'path': path, 'size_mb': file_size_mb, 'stage': 'size_check'},
                    processing_time=time.time() - start_time,
                    fallback_used=True
                )
            
            # Step 5: Load video metadata (with caching)
            try:
                if self.enable_caching and path in self.duration_cache:
                    frame_count, frame_rate = self.duration_cache[path]
                    self.load_statistics['cache_hits'] += 1
                    self.logger.debug(f"Using cached metadata for {path}")
                else:
                    frame_count, frame_rate = self._get_video_duration(path)
                    if self.enable_caching:
                        self.duration_cache[path] = (frame_count, frame_rate)
                
                if frame_count <= 0 or frame_rate <= 0:
                    raise ValueError(f"Invalid video metadata: frames={frame_count}, fps={frame_rate}")
                
            except Exception as e:
                self.logger.error(f"Failed to get video metadata for {path}: {str(e)}")
                return VideoLoadResult(
                    success=False,
                    video_tensor=self._create_fallback_tensor(target_frame_count),
                    error_message=f"Failed to get video metadata: {str(e)}",
                    metadata={'path': path, 'stage': 'metadata_extraction'},
                    processing_time=time.time() - start_time,
                    fallback_used=True
                )
            
            # Step 6: Load video frames
            try:
                video_tensor = self._extract_video_frames(
                    path, target_frame_count, frame_count, frame_rate, variation, index
                )
                
                if video_tensor is None:
                    raise ValueError("Frame extraction returned None")
                
                # Validate tensor
                if video_tensor.numel() == 0:
                    raise ValueError("Extracted tensor is empty")
                
                if torch.isnan(video_tensor).any():
                    raise ValueError("Extracted tensor contains NaN values")
                
                self.logger.debug(f"Successfully loaded video {path} with shape {video_tensor.shape}")
                
                return VideoLoadResult(
                    success=True,
                    video_tensor=video_tensor,
                    error_message=None,
                    metadata={
                        'path': path,
                        'shape': list(video_tensor.shape),
                        'dtype': str(video_tensor.dtype),
                        'frame_count': frame_count,
                        'frame_rate': frame_rate,
                        'file_size_mb': file_size_mb,
                        'stage': 'success'
                    },
                    processing_time=time.time() - start_time,
                    fallback_used=False
                )
                
            except Exception as e:
                self.logger.error(f"Failed to extract frames from {path}: {str(e)}")
                return VideoLoadResult(
                    success=False,
                    video_tensor=self._create_fallback_tensor(target_frame_count),
                    error_message=f"Frame extraction failed: {str(e)}",
                    metadata={'path': path, 'stage': 'frame_extraction'},
                    processing_time=time.time() - start_time,
                    fallback_used=True
                )
                
        except Exception as e:
            self.logger.error(f"Unexpected error in _load_video_safe: {str(e)}")
            return VideoLoadResult(
                success=False,
                video_tensor=self._create_fallback_tensor(target_frame_count),
                error_message=f"Unexpected error: {str(e)}",
                metadata={'path': path, 'stage': 'unexpected_error'},
                processing_time=time.time() - start_time,
                fallback_used=True
            )
    
    def _get_video_duration(self, path: str) -> Tuple[int, float]:
        """
        Get video duration and frame rate with robust error handling.
        
        Args:
            path: Path to video file
            
        Returns:
            Tuple of (frame_count, frame_rate)
        """
        try:
            container = av.open(path)
            
            if not container.streams.video:
                raise ValueError("No video streams found in file")
            
            video_stream = container.streams.video[0]
            
            # Calculate frame rate
            if video_stream.base_rate.denominator == 0:
                frame_rate = 24.0  # Default fallback
                self.logger.warning(f"Invalid frame rate for {path}, using default 24fps")
            else:
                frame_rate = video_stream.base_rate.numerator / video_stream.base_rate.denominator
            
            # Calculate frame count using multiple methods
            frame_count = None
            
            # Method 1: Direct frame count
            if video_stream.frames > 0:
                frame_count = video_stream.frames
                self.logger.debug(f"Got frame count from stream.frames: {frame_count}")
            
            # Method 2: Duration-based calculation
            elif container.duration and container.duration > 0:
                duration_seconds = container.duration / av.time_base
                frame_count = int(duration_seconds * frame_rate)
                self.logger.debug(f"Calculated frame count from duration: {frame_count}")
            
            # Method 3: Metadata duration
            elif 'DURATION' in video_stream.metadata:
                try:
                    duration_str = video_stream.metadata['DURATION']
                    time_parts = [float(x) for x in duration_str.split(':')]
                    time_parts.reverse()  # [seconds, minutes, hours]
                    
                    duration_seconds = 0
                    multipliers = [1, 60, 3600]  # seconds, minutes, hours
                    for i, part in enumerate(time_parts):
                        if i < len(multipliers):
                            duration_seconds += part * multipliers[i]
                    
                    frame_count = int(duration_seconds * frame_rate)
                    self.logger.debug(f"Calculated frame count from metadata: {frame_count}")
                except Exception as e:
                    self.logger.warning(f"Failed to parse metadata duration: {e}")
            
            # Method 4: Full decode (last resort)
            if frame_count is None or frame_count <= 0:
                self.logger.warning(f"Falling back to full decode for frame count: {path}")
                try:
                    decoded = container.decode(video=0)
                    frame_count = sum(1 for _ in decoded)
                    self.logger.debug(f"Counted frames by full decode: {frame_count}")
                except Exception as e:
                    self.logger.error(f"Full decode failed: {e}")
                    frame_count = 30  # Fallback assumption
            
            container.close()
            
            # Validate results
            if frame_count <= 0:
                frame_count = 30  # Reasonable fallback
            if frame_rate <= 0:
                frame_rate = 24.0  # Reasonable fallback
            
            return frame_count, frame_rate
            
        except Exception as e:
            self.logger.error(f"Error getting video duration for {path}: {str(e)}")
            # Return reasonable defaults
            return 30, 24.0
    
    def _extract_video_frames(
        self, 
        path: str, 
        target_frame_count: int, 
        total_frame_count: int, 
        frame_rate: float, 
        variation: int, 
        index: int
    ) -> torch.Tensor:
        """
        Extract video frames with robust error handling.
        
        Args:
            path: Path to video file
            target_frame_count: Number of frames to extract
            total_frame_count: Total frames in video
            frame_rate: Video frame rate
            variation: Variation index for randomization
            index: Sample index
            
        Returns:
            Video tensor with shape (C, T, H, W)
        """
        try:
            rand = self._get_rand(variation, index)
            
            # Calculate timing
            duration = (total_frame_count - 1) / frame_rate
            target_duration = (target_frame_count - 1) / self.target_frame_rate
            
            # Ensure we don't seek beyond video duration
            max_start_offset = max(0, duration - target_duration)
            start_offset = rand.uniform(0, max_start_offset) if max_start_offset > 0 else 0
            
            self.logger.debug(
                f"Extracting {target_frame_count} frames from {path} "
                f"(duration: {duration:.2f}s, start: {start_offset:.2f}s)"
            )
            
            # Open container and seek
            container = av.open(path)
            
            if start_offset > 0:
                seek_time = int(start_offset * av.time_base)
                container.seek(seek_time)
            
            decoded = container.decode(video=0)
            
            # Skip frames until we reach the desired start offset
            if start_offset > 0:
                while True:
                    try:
                        frame = next(decoded)
                        if frame.time >= start_offset:
                            # Put this frame back by creating a new iterator
                            container.seek(int(frame.time * av.time_base))
                            decoded = container.decode(video=0)
                            break
                    except StopIteration:
                        break
            
            # Extract frames
            frames = []
            frames_extracted = 0
            
            while frames_extracted < target_frame_count:
                try:
                    frame = next(decoded)
                    
                    # Convert frame to tensor
                    frame_array = frame.to_rgb().to_ndarray()
                    frame_tensor = torch.from_numpy(frame_array).movedim(2, 0)  # HWC -> CHW
                    frames.append(frame_tensor)
                    frames_extracted += 1
                    
                except StopIteration:
                    # Video ended, duplicate last frame if we have any
                    if frames:
                        self.logger.debug(f"Video ended early, duplicating last frame")
                        while len(frames) < target_frame_count:
                            frames.append(frames[-1].clone())
                    else:
                        # No frames extracted, create dummy frames
                        self.logger.warning(f"No frames extracted from {path}, creating dummy frames")
                        dummy_frame = torch.zeros((3, 224, 224), dtype=torch.uint8)
                        frames = [dummy_frame.clone() for _ in range(target_frame_count)]
                    break
                except Exception as e:
                    self.logger.error(f"Error extracting frame {frames_extracted}: {str(e)}")
                    # Create a dummy frame for this position
                    if frames:
                        frames.append(frames[-1].clone())
                    else:
                        dummy_frame = torch.zeros((3, 224, 224), dtype=torch.uint8)
                        frames.append(dummy_frame)
                    frames_extracted += 1
            
            container.close()
            
            if not frames:
                raise ValueError("No frames were extracted")
            
            # Stack frames into video tensor (C, T, H, W)
            video_tensor = torch.stack(frames, dim=1)
            
            # Convert to target dtype
            video_tensor = video_tensor.to(dtype=self.dtype)
            
            # Normalize from 0-255 to target range
            video_tensor = video_tensor / 255.0
            video_tensor = video_tensor * (self.range_max - self.range_min) + self.range_min
            
            # Move to pipeline device
            video_tensor = video_tensor.to(device=self.pipeline.device)
            
            return video_tensor
            
        except Exception as e:
            self.logger.error(f"Error extracting frames from {path}: {str(e)}")
            raise
    
    def _create_fallback_tensor(self, target_frame_count: int) -> torch.Tensor:
        """
        Create a fallback video tensor when loading fails.
        
        Args:
            target_frame_count: Number of frames needed
            
        Returns:
            Fallback video tensor with shape (C, T, H, W)
        """
        # Create a simple gradient pattern for fallback
        height, width = 224, 224
        
        # Create frames with different colors to make it obvious it's fallback data
        frames = []
        for i in range(target_frame_count):
            # Create a gradient that changes per frame
            frame = torch.zeros((3, height, width), dtype=self.dtype)
            
            # Red channel: horizontal gradient
            frame[0] = torch.linspace(0, 1, width).unsqueeze(0).expand(height, -1)
            
            # Green channel: vertical gradient
            frame[1] = torch.linspace(0, 1, height).unsqueeze(1).expand(-1, width)
            
            # Blue channel: frame-based gradient
            frame[2] = (i / max(1, target_frame_count - 1)) * 0.5 + 0.25
            
            frames.append(frame)
        
        # Stack into video tensor (C, T, H, W)
        video_tensor = torch.stack(frames, dim=1)
        
        # Apply range scaling
        video_tensor = video_tensor * (self.range_max - self.range_min) + self.range_min
        
        # Move to pipeline device
        video_tensor = video_tensor.to(device=self.pipeline.device)
        
        return video_tensor
    
    def _create_fallback_result(self, index: int, error_message: str) -> Dict[str, Any]:
        """
        Create fallback result dictionary when video loading fails.
        
        Args:
            index: Sample index
            error_message: Error description
            
        Returns:
            Dictionary with fallback video tensor
        """
        self.logger.warning(f"Creating fallback result for index {index}: {error_message}")
        
        # Use a reasonable default frame count
        fallback_frame_count = 8
        fallback_tensor = self._create_fallback_tensor(fallback_frame_count)
        
        return {self.video_out_name: fallback_tensor}
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """Get comprehensive loading statistics."""
        total = self.load_statistics['total_attempts']
        
        stats = dict(self.load_statistics)
        
        if total > 0:
            stats['success_rate'] = self.load_statistics['successful_loads'] / total
            stats['failure_rate'] = self.load_statistics['failed_loads'] / total
            stats['fallback_rate'] = self.load_statistics['fallback_uses'] / total
            
            if self.enable_caching:
                stats['cache_hit_rate'] = self.load_statistics['cache_hits'] / total
        
        return stats
    
    def clear_cache(self):
        """Clear all cached data."""
        if self.enable_caching:
            self.duration_cache.clear()
            if self.metadata_cache:
                self.metadata_cache.clear()
            self.logger.info("Video loader cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        if not self.enable_caching:
            return {'caching_enabled': False}
        
        return {
            'caching_enabled': True,
            'duration_cache_size': len(self.duration_cache),
            'metadata_cache_size': len(self.metadata_cache) if self.metadata_cache else 0,
            'cached_videos': list(self.duration_cache.keys())
        }