import cv2
import os
import pathlib
from typing import List, Tuple, Optional, Union
from enum import Enum

from modules.util import path_util


class FrameSamplingStrategy(Enum):
    """Frame sampling strategies for video processing."""
    UNIFORM = "uniform"
    RANDOM = "random"
    KEYFRAME = "keyframe"


class VideoValidationError(Exception):
    """Exception raised for video validation errors."""
    pass


def validate_video_format(video_path: Union[str, pathlib.Path]) -> bool:
    """
    Validate if the video file has a supported format.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        True if format is supported, False otherwise
    """
    path = pathlib.Path(video_path)
    return path.suffix.lower() in path_util.SUPPORTED_VIDEO_EXTENSIONS


def validate_video_file(video_path: Union[str, pathlib.Path]) -> Tuple[bool, Optional[str]]:
    """
    Validate if a video file can be opened and read.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    path = pathlib.Path(video_path)
    
    # Check if file exists
    if not path.is_file():
        return False, f"Video file does not exist: {path}"
    
    # Check format
    if not validate_video_format(path):
        return False, f"Unsupported video format: {path.suffix}. Supported formats: {path_util.SUPPORTED_VIDEO_EXTENSIONS}"
    
    # Try to open and read the video
    vid = cv2.VideoCapture(str(path))
    try:
        if not vid.isOpened():
            return False, f"Cannot open video file: {path}"
        
        # Try to read first frame
        ret, frame = vid.read()
        if not ret or frame is None:
            return False, f"Cannot read frames from video file: {path}"
        
        return True, None
    finally:
        vid.release()


def get_video_info(video_path: Union[str, pathlib.Path]) -> dict:
    """
    Get video information including resolution, frame count, duration, and FPS.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary containing video information
        
    Raises:
        VideoValidationError: If video cannot be processed
    """
    path = pathlib.Path(video_path)
    
    # Validate video first
    is_valid, error_msg = validate_video_file(path)
    if not is_valid:
        raise VideoValidationError(error_msg)
    
    vid = cv2.VideoCapture(str(path))
    try:
        if not vid.isOpened():
            raise VideoValidationError(f"Cannot open video file: {path}")
        
        # Get video properties
        frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vid.get(cv2.CAP_PROP_FPS)
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate duration
        duration = frame_count / fps if fps > 0 else 0
        
        return {
            'path': str(path),
            'frame_count': frame_count,
            'fps': fps,
            'width': width,
            'height': height,
            'resolution': (width, height),
            'duration': duration,
            'aspect_ratio': width / height if height > 0 else 0
        }
    finally:
        vid.release()


def validate_video_constraints(
    video_path: Union[str, pathlib.Path],
    min_resolution: Optional[Tuple[int, int]] = None,
    max_resolution: Optional[Tuple[int, int]] = None,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    min_frame_count: Optional[int] = None,
    max_frame_count: Optional[int] = None,
    required_fps: Optional[float] = None,
    fps_tolerance: float = 1.0
) -> Tuple[bool, List[str]]:
    """
    Validate video against specified constraints.
    
    Args:
        video_path: Path to the video file
        min_resolution: Minimum (width, height) resolution
        max_resolution: Maximum (width, height) resolution
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        min_frame_count: Minimum number of frames
        max_frame_count: Maximum number of frames
        required_fps: Required FPS (with tolerance)
        fps_tolerance: Tolerance for FPS validation
        
    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    try:
        video_info = get_video_info(video_path)
    except VideoValidationError as e:
        return False, [str(e)]
    
    errors = []
    
    # Resolution validation
    width, height = video_info['resolution']
    if min_resolution:
        min_w, min_h = min_resolution
        if width < min_w or height < min_h:
            errors.append(f"Resolution {width}x{height} is below minimum {min_w}x{min_h}")
    
    if max_resolution:
        max_w, max_h = max_resolution
        if width > max_w or height > max_h:
            errors.append(f"Resolution {width}x{height} exceeds maximum {max_w}x{max_h}")
    
    # Duration validation
    duration = video_info['duration']
    if min_duration is not None and duration < min_duration:
        errors.append(f"Duration {duration:.2f}s is below minimum {min_duration}s")
    
    if max_duration is not None and duration > max_duration:
        errors.append(f"Duration {duration:.2f}s exceeds maximum {max_duration}s")
    
    # Frame count validation
    frame_count = video_info['frame_count']
    if min_frame_count is not None and frame_count < min_frame_count:
        errors.append(f"Frame count {frame_count} is below minimum {min_frame_count}")
    
    if max_frame_count is not None and frame_count > max_frame_count:
        errors.append(f"Frame count {frame_count} exceeds maximum {max_frame_count}")
    
    # FPS validation
    fps = video_info['fps']
    if required_fps is not None:
        if abs(fps - required_fps) > fps_tolerance:
            errors.append(f"FPS {fps:.2f} does not match required {required_fps:.2f} (tolerance: {fps_tolerance})")
    
    return len(errors) == 0, errors


def get_frame_indices_uniform(total_frames: int, target_frames: int) -> List[int]:
    """
    Get uniformly distributed frame indices.
    
    Args:
        total_frames: Total number of frames in video
        target_frames: Number of frames to sample
        
    Returns:
        List of frame indices
    """
    if target_frames >= total_frames:
        return list(range(total_frames))
    
    # Calculate step size for uniform sampling
    step = total_frames / target_frames
    indices = [int(i * step) for i in range(target_frames)]
    
    # Ensure indices are within bounds
    indices = [min(idx, total_frames - 1) for idx in indices]
    
    return indices


def get_frame_indices_random(total_frames: int, target_frames: int, seed: Optional[int] = None) -> List[int]:
    """
    Get randomly sampled frame indices.
    
    Args:
        total_frames: Total number of frames in video
        target_frames: Number of frames to sample
        seed: Random seed for reproducibility
        
    Returns:
        List of frame indices
    """
    import random
    
    if seed is not None:
        random.seed(seed)
    
    if target_frames >= total_frames:
        return list(range(total_frames))
    
    # Sample random indices without replacement
    indices = random.sample(range(total_frames), target_frames)
    indices.sort()  # Sort to maintain temporal order
    
    return indices


def get_frame_indices_keyframe(video_path: Union[str, pathlib.Path], target_frames: int) -> List[int]:
    """
    Get frame indices based on keyframe detection.
    
    Args:
        video_path: Path to the video file
        target_frames: Number of frames to sample
        
    Returns:
        List of frame indices
        
    Note:
        This is a simplified keyframe detection. For more advanced detection,
        consider using ffmpeg or other specialized libraries.
    """
    try:
        video_info = get_video_info(video_path)
        total_frames = video_info['frame_count']
        
        # For now, fall back to uniform sampling
        # TODO: Implement proper keyframe detection using ffmpeg or similar
        return get_frame_indices_uniform(total_frames, target_frames)
    except VideoValidationError:
        # If video info cannot be obtained, return empty list
        return []


def sample_video_frames(
    video_path: Union[str, pathlib.Path],
    target_frames: int,
    strategy: FrameSamplingStrategy = FrameSamplingStrategy.UNIFORM,
    seed: Optional[int] = None
) -> List[int]:
    """
    Sample frame indices from a video using the specified strategy.
    
    Args:
        video_path: Path to the video file
        target_frames: Number of frames to sample
        strategy: Sampling strategy to use
        seed: Random seed for reproducible sampling
        
    Returns:
        List of frame indices
        
    Raises:
        VideoValidationError: If video cannot be processed
        ValueError: If strategy is not supported
    """
    video_info = get_video_info(video_path)
    total_frames = video_info['frame_count']
    
    if strategy == FrameSamplingStrategy.UNIFORM:
        return get_frame_indices_uniform(total_frames, target_frames)
    elif strategy == FrameSamplingStrategy.RANDOM:
        return get_frame_indices_random(total_frames, target_frames, seed)
    elif strategy == FrameSamplingStrategy.KEYFRAME:
        return get_frame_indices_keyframe(video_path, target_frames)
    else:
        raise ValueError(f"Unsupported sampling strategy: {strategy}")


def validate_video_dataset(
    video_paths: List[Union[str, pathlib.Path]],
    min_resolution: Optional[Tuple[int, int]] = None,
    max_resolution: Optional[Tuple[int, int]] = None,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    min_frame_count: Optional[int] = None,
    max_frame_count: Optional[int] = None,
    required_fps: Optional[float] = None,
    fps_tolerance: float = 1.0,
    verbose: bool = True
) -> Tuple[List[str], List[Tuple[str, List[str]]]]:
    """
    Validate a list of video files against constraints.
    
    Args:
        video_paths: List of video file paths
        min_resolution: Minimum (width, height) resolution
        max_resolution: Maximum (width, height) resolution
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        min_frame_count: Minimum number of frames
        max_frame_count: Maximum number of frames
        required_fps: Required FPS (with tolerance)
        fps_tolerance: Tolerance for FPS validation
        verbose: Whether to print validation progress
        
    Returns:
        Tuple of (valid_paths, invalid_paths_with_errors)
    """
    valid_paths = []
    invalid_paths = []
    
    for i, video_path in enumerate(video_paths):
        if verbose and i % 100 == 0:
            print(f"Validating video {i+1}/{len(video_paths)}: {video_path}")
        
        is_valid, errors = validate_video_constraints(
            video_path,
            min_resolution=min_resolution,
            max_resolution=max_resolution,
            min_duration=min_duration,
            max_duration=max_duration,
            min_frame_count=min_frame_count,
            max_frame_count=max_frame_count,
            required_fps=required_fps,
            fps_tolerance=fps_tolerance
        )
        
        if is_valid:
            valid_paths.append(str(video_path))
        else:
            invalid_paths.append((str(video_path), errors))
    
    if verbose:
        print(f"Validation complete: {len(valid_paths)} valid, {len(invalid_paths)} invalid videos")
    
    return valid_paths, invalid_paths


def preprocess_video_for_training(
    video_path: Union[str, pathlib.Path],
    target_frames: int = 16,
    target_resolution: Optional[Tuple[int, int]] = None,
    sampling_strategy: FrameSamplingStrategy = FrameSamplingStrategy.UNIFORM,
    seed: Optional[int] = None
) -> dict:
    """
    Preprocess a video for training by extracting frame information and sampling strategy.
    
    Args:
        video_path: Path to the video file
        target_frames: Number of frames to extract
        target_resolution: Target resolution for frames (width, height)
        sampling_strategy: Strategy for frame sampling
        seed: Random seed for reproducible sampling
        
    Returns:
        Dictionary containing preprocessing information
        
    Raises:
        VideoValidationError: If video cannot be processed
    """
    # Get video information
    video_info = get_video_info(video_path)
    
    # Sample frame indices
    frame_indices = sample_video_frames(
        video_path, 
        target_frames, 
        sampling_strategy, 
        seed
    )
    
    # Calculate scaling if target resolution is specified
    scale_factor = None
    if target_resolution:
        target_w, target_h = target_resolution
        current_w, current_h = video_info['resolution']
        scale_w = target_w / current_w
        scale_h = target_h / current_h
        scale_factor = min(scale_w, scale_h)  # Maintain aspect ratio
    
    return {
        'video_info': video_info,
        'frame_indices': frame_indices,
        'target_frames': len(frame_indices),
        'sampling_strategy': sampling_strategy.value,
        'scale_factor': scale_factor,
        'target_resolution': target_resolution,
        'preprocessing_seed': seed
    }

def calculate_video_quality_metrics(video_tensor: 'torch.Tensor') -> dict:
    """
    Calculate basic quality metrics for a video tensor.
    
    Args:
        video_tensor: Video tensor of shape (frames, height, width, channels)
        
    Returns:
        Dictionary containing quality metrics
    """
    import torch
    
    if video_tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor (frames, height, width, channels), got {video_tensor.dim()}D")
    
    frames, height, width, channels = video_tensor.shape
    
    # Convert to float for calculations
    video_float = video_tensor.float() / 255.0 if video_tensor.dtype == torch.uint8 else video_tensor.float()
    
    # Basic statistics
    mean_brightness = video_float.mean().item()
    std_brightness = video_float.std().item()
    
    # Frame-to-frame consistency (temporal stability)
    if frames > 1:
        frame_diffs = torch.diff(video_float, dim=0)
        temporal_consistency = 1.0 - frame_diffs.abs().mean().item()
    else:
        temporal_consistency = 1.0
    
    # Color distribution (for RGB videos)
    if channels == 3:
        r_mean = video_float[:, :, :, 0].mean().item()
        g_mean = video_float[:, :, :, 1].mean().item()
        b_mean = video_float[:, :, :, 2].mean().item()
        color_balance = {
            'r_mean': r_mean,
            'g_mean': g_mean,
            'b_mean': b_mean,
            'color_variance': torch.var(torch.stack([
                video_float[:, :, :, 0].mean(),
                video_float[:, :, :, 1].mean(),
                video_float[:, :, :, 2].mean()
            ])).item()
        }
    else:
        color_balance = None
    
    # Sharpness estimation (using Laplacian variance)
    if video_tensor.dtype == torch.uint8:
        # Convert to grayscale for sharpness calculation
        if channels == 3:
            gray_video = (0.299 * video_float[:, :, :, 0] + 
                         0.587 * video_float[:, :, :, 1] + 
                         0.114 * video_float[:, :, :, 2])
        else:
            gray_video = video_float[:, :, :, 0]
        
        # Simple sharpness metric using gradient magnitude
        grad_x = torch.diff(gray_video, dim=2)
        grad_y = torch.diff(gray_video, dim=1)
        sharpness = (grad_x.abs().mean() + grad_y.abs().mean()).item()
    else:
        sharpness = None
    
    return {
        'frame_count': frames,
        'resolution': (width, height),
        'channels': channels,
        'mean_brightness': mean_brightness,
        'brightness_std': std_brightness,
        'temporal_consistency': temporal_consistency,
        'sharpness': sharpness,
        'color_balance': color_balance,
        'data_range': (video_float.min().item(), video_float.max().item())
    }


def save_video_with_quality_options(
    video_tensor: 'torch.Tensor',
    output_path: Union[str, pathlib.Path],
    fps: float = 24.0,
    quality_preset: str = "medium",
    codec_options: Optional[dict] = None
) -> bool:
    """
    Save video tensor with quality options and validation.
    
    Args:
        video_tensor: Video tensor of shape (frames, height, width, channels)
        output_path: Output file path
        fps: Frames per second
        quality_preset: Quality preset ("low", "medium", "high")
        codec_options: Custom codec options
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import torch
        from torchvision.io import write_video
        
        path = pathlib.Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate tensor
        if video_tensor.dim() != 4:
            raise ValueError(f"Expected 4D tensor (frames, height, width, channels), got {video_tensor.dim()}D")
        
        # Ensure tensor is in correct format
        if video_tensor.dtype != torch.uint8:
            video_tensor = (video_tensor.clamp(0, 1) * 255).to(torch.uint8)
        
        # Default codec options based on quality preset
        default_options = {
            "low": {"crf": "28", "preset": "fast"},
            "medium": {"crf": "23", "preset": "medium"},
            "high": {"crf": "18", "preset": "slow"}
        }
        
        options = default_options.get(quality_preset, default_options["medium"])
        if codec_options:
            options.update(codec_options)
        
        # Write video
        write_video(str(path), video_tensor, fps=fps, options=options)
        
        # Validate output file
        if path.exists() and path.stat().st_size > 0:
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error saving video: {e}")
        return False


def validate_generated_video(
    video_path: Union[str, pathlib.Path],
    expected_frames: Optional[int] = None,
    expected_resolution: Optional[Tuple[int, int]] = None,
    min_quality_threshold: float = 0.1
) -> Tuple[bool, List[str]]:
    """
    Validate a generated video file for quality and correctness.
    
    Args:
        video_path: Path to the video file
        expected_frames: Expected number of frames
        expected_resolution: Expected resolution (width, height)
        min_quality_threshold: Minimum quality threshold for validation
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    try:
        # Basic file validation
        is_valid, error_msg = validate_video_file(video_path)
        if not is_valid:
            issues.append(error_msg)
            return False, issues
        
        # Get video info
        video_info = get_video_info(video_path)
        
        # Check frame count
        if expected_frames is not None:
            actual_frames = video_info['frame_count']
            if actual_frames != expected_frames:
                issues.append(f"Frame count mismatch: expected {expected_frames}, got {actual_frames}")
        
        # Check resolution
        if expected_resolution is not None:
            actual_resolution = video_info['resolution']
            if actual_resolution != expected_resolution:
                issues.append(f"Resolution mismatch: expected {expected_resolution}, got {actual_resolution}")
        
        # Check for reasonable duration
        duration = video_info['duration']
        if duration <= 0:
            issues.append(f"Invalid duration: {duration}")
        
        # Check for reasonable FPS
        fps = video_info['fps']
        if fps <= 0 or fps > 120:  # Reasonable FPS range
            issues.append(f"Unusual FPS: {fps}")
        
        # Check file size (should not be too small)
        file_size = pathlib.Path(video_path).stat().st_size
        min_expected_size = video_info['frame_count'] * 1000  # Very rough estimate
        if file_size < min_expected_size:
            issues.append(f"File size seems too small: {file_size} bytes")
        
        return len(issues) == 0, issues
        
    except Exception as e:
        issues.append(f"Validation error: {str(e)}")
        return False, issues


def create_video_sampling_config(
    frames: int = 16,
    fps: float = 24.0,
    resolution: Tuple[int, int] = (512, 512),
    quality_preset: str = "medium",
    format_preference: str = "mp4"
) -> dict:
    """
    Create a video sampling configuration dictionary.
    
    Args:
        frames: Number of frames to generate
        fps: Target frames per second
        resolution: Target resolution (width, height)
        quality_preset: Quality preset for encoding
        format_preference: Preferred output format
        
    Returns:
        Configuration dictionary
    """
    return {
        'frames': frames,
        'fps': fps,
        'resolution': resolution,
        'quality_preset': quality_preset,
        'format_preference': format_preference,
        'duration': frames / fps,
        'aspect_ratio': resolution[0] / resolution[1],
        'total_pixels': resolution[0] * resolution[1] * frames
    }


def estimate_video_memory_requirements(
    frames: int,
    resolution: Tuple[int, int],
    channels: int = 3,
    dtype_size: int = 4  # float32
) -> dict:
    """
    Estimate memory requirements for video processing.
    
    Args:
        frames: Number of frames
        resolution: Resolution (width, height)
        channels: Number of channels
        dtype_size: Size of data type in bytes
        
    Returns:
        Dictionary with memory estimates in bytes and MB
    """
    width, height = resolution
    
    # Raw video tensor size
    raw_size = frames * height * width * channels * dtype_size
    
    # Latent size (assuming 8x spatial compression, 4x temporal compression)
    latent_frames = (frames - 1) // 4 + 1
    latent_height = height // 8
    latent_width = width // 8
    latent_channels = 16  # Typical for video models
    latent_size = latent_frames * latent_height * latent_width * latent_channels * dtype_size
    
    # Estimated peak memory (including gradients, activations, etc.)
    peak_multiplier = 3.0  # Conservative estimate
    peak_memory = (raw_size + latent_size) * peak_multiplier
    
    return {
        'raw_video_bytes': raw_size,
        'raw_video_mb': raw_size / (1024 * 1024),
        'latent_bytes': latent_size,
        'latent_mb': latent_size / (1024 * 1024),
        'peak_memory_bytes': peak_memory,
        'peak_memory_mb': peak_memory / (1024 * 1024),
        'peak_memory_gb': peak_memory / (1024 * 1024 * 1024)
    }