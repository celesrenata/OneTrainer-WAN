"""
DataFlowValidator for pipeline data integrity checking.

This module provides comprehensive validation of data flowing through the MGDS pipeline,
ensuring data integrity and providing detailed diagnostics for debugging.
"""

import logging
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import psutil


@dataclass
class ValidationResult:
    """Result of data validation operation."""
    is_valid: bool
    error_message: Optional[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    processing_time: float


@dataclass
class VideoMetadata:
    """Metadata for video files."""
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int
    format: str
    codec: str
    is_corrupted: bool
    error_details: Optional[str]


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    current_usage_mb: float
    peak_usage_mb: float
    available_mb: float
    gpu_usage_mb: Optional[float]
    swap_usage_mb: float


class DataFlowValidator:
    """
    Comprehensive data validation for MGDS pipeline.
    
    Provides validation for:
    - Video file integrity and metadata
    - Pipeline data structures and types
    - Memory usage monitoring
    - Performance metrics collection
    """
    
    def __init__(self, enable_gpu_monitoring: bool = True):
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.logger = logging.getLogger("DataFlowValidator")
        self.logger.setLevel(logging.DEBUG)
        
        # Create console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Performance tracking
        self.validation_metrics: Dict[str, List[float]] = {}
        self.memory_snapshots: List[MemoryStats] = []
    
    def validate_video_file(self, video_path: str) -> ValidationResult:
        """
        Validate video file integrity and extract metadata.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            ValidationResult with validation status and metadata
        """
        start_time = time.time()
        
        try:
            # Check if file exists
            if not os.path.exists(video_path):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Video file does not exist: {video_path}",
                    warnings=[],
                    metadata={'path': video_path},
                    processing_time=time.time() - start_time
                )
            
            # Check if file is readable
            if not os.access(video_path, os.R_OK):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Video file is not readable: {video_path}",
                    warnings=[],
                    metadata={'path': video_path},
                    processing_time=time.time() - start_time
                )
            
            # Check file size
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Video file is empty: {video_path}",
                    warnings=[],
                    metadata={'path': video_path, 'size': file_size},
                    processing_time=time.time() - start_time
                )
            
            # Try to extract video metadata using basic file operations
            # Note: For full video validation, we'd need opencv or ffmpeg
            # This is a basic validation that checks file accessibility
            
            warnings = []
            metadata = {
                'path': video_path,
                'size': file_size,
                'exists': True,
                'readable': True
            }
            
            # Check file extension
            _, ext = os.path.splitext(video_path)
            supported_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
            if ext.lower() not in supported_extensions:
                warnings.append(f"Unusual video file extension: {ext}")
            
            # Basic size checks
            if file_size < 1024:  # Less than 1KB
                warnings.append("Video file is very small, may be corrupted")
            elif file_size > 1024 * 1024 * 1024:  # Larger than 1GB
                warnings.append("Video file is very large, may cause memory issues")
            
            processing_time = time.time() - start_time
            
            return ValidationResult(
                is_valid=True,
                error_message=None,
                warnings=warnings,
                metadata=metadata,
                processing_time=processing_time
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Error validating video file: {str(e)}",
                warnings=[],
                metadata={'path': video_path, 'exception': str(e)},
                processing_time=time.time() - start_time
            )
    
    def validate_pipeline_item(self, item: Any, expected_schema: Dict[str, Any]) -> ValidationResult:
        """
        Validate pipeline data item against expected schema.
        
        Args:
            item: The data item to validate
            expected_schema: Expected structure and types
            
        Returns:
            ValidationResult with validation status
        """
        start_time = time.time()
        warnings = []
        metadata = {}
        
        try:
            # Check if item is None
            if item is None:
                return ValidationResult(
                    is_valid=False,
                    error_message="Pipeline item is None",
                    warnings=[],
                    metadata={},
                    processing_time=time.time() - start_time
                )
            
            # Check if item is a dictionary
            if not isinstance(item, dict):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Pipeline item must be a dictionary, got {type(item)}",
                    warnings=[],
                    metadata={'actual_type': str(type(item))},
                    processing_time=time.time() - start_time
                )
            
            # Check for empty dictionary
            if not item:
                return ValidationResult(
                    is_valid=False,
                    error_message="Pipeline item is an empty dictionary",
                    warnings=[],
                    metadata={},
                    processing_time=time.time() - start_time
                )
            
            # Validate against expected schema
            missing_keys = []
            unexpected_keys = []
            type_mismatches = []
            
            # Check for required keys
            for key, expected_type in expected_schema.items():
                if key not in item:
                    missing_keys.append(key)
                else:
                    # Check type if specified
                    if expected_type is not None:
                        actual_value = item[key]
                        if not self._check_type_compatibility(actual_value, expected_type):
                            type_mismatches.append({
                                'key': key,
                                'expected': str(expected_type),
                                'actual': str(type(actual_value))
                            })
            
            # Check for unexpected keys (optional warning)
            for key in item.keys():
                if key not in expected_schema:
                    unexpected_keys.append(key)
            
            # Validate tensor data if present
            tensor_warnings = self._validate_tensors(item)
            warnings.extend(tensor_warnings)
            
            # Build metadata
            metadata = {
                'item_keys': list(item.keys()),
                'expected_keys': list(expected_schema.keys()),
                'missing_keys': missing_keys,
                'unexpected_keys': unexpected_keys,
                'type_mismatches': type_mismatches
            }
            
            # Add warnings for unexpected keys
            if unexpected_keys:
                warnings.append(f"Unexpected keys found: {unexpected_keys}")
            
            # Determine if validation passed
            is_valid = len(missing_keys) == 0 and len(type_mismatches) == 0
            error_message = None
            
            if not is_valid:
                error_parts = []
                if missing_keys:
                    error_parts.append(f"Missing required keys: {missing_keys}")
                if type_mismatches:
                    error_parts.append(f"Type mismatches: {type_mismatches}")
                error_message = "; ".join(error_parts)
            
            return ValidationResult(
                is_valid=is_valid,
                error_message=error_message,
                warnings=warnings,
                metadata=metadata,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Error during validation: {str(e)}",
                warnings=[],
                metadata={'exception': str(e), 'traceback': traceback.format_exc()},
                processing_time=time.time() - start_time
            )
    
    def _check_type_compatibility(self, value: Any, expected_type: Any) -> bool:
        """Check if value is compatible with expected type."""
        if expected_type is None:
            return True
        
        # Handle torch.Tensor specifically
        if expected_type == torch.Tensor:
            return isinstance(value, torch.Tensor)
        
        # Handle basic types
        if isinstance(expected_type, type):
            return isinstance(value, expected_type)
        
        # Handle string type specifications
        if isinstance(expected_type, str):
            if expected_type == 'tensor':
                return isinstance(value, torch.Tensor)
            elif expected_type == 'string':
                return isinstance(value, str)
            elif expected_type == 'number':
                return isinstance(value, (int, float))
            elif expected_type == 'dict':
                return isinstance(value, dict)
            elif expected_type == 'list':
                return isinstance(value, list)
        
        return True  # Default to compatible if we can't determine
    
    def _validate_tensors(self, item: Dict[str, Any]) -> List[str]:
        """Validate tensor data in pipeline item."""
        warnings = []
        
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                # Check for empty tensors
                if value.numel() == 0:
                    warnings.append(f"Tensor '{key}' is empty")
                
                # Check for NaN values
                if torch.isnan(value).any():
                    warnings.append(f"Tensor '{key}' contains NaN values")
                
                # Check for infinite values
                if torch.isinf(value).any():
                    warnings.append(f"Tensor '{key}' contains infinite values")
                
                # Check for reasonable value ranges
                if value.dtype.is_floating_point:
                    min_val = value.min().item()
                    max_val = value.max().item()
                    
                    if min_val < -1000 or max_val > 1000:
                        warnings.append(f"Tensor '{key}' has extreme values: min={min_val:.2f}, max={max_val:.2f}")
                
                # Check tensor shape reasonableness
                if len(value.shape) > 5:
                    warnings.append(f"Tensor '{key}' has unusual number of dimensions: {len(value.shape)}")
                
                # Check for very large tensors
                tensor_size_mb = value.numel() * value.element_size() / (1024 * 1024)
                if tensor_size_mb > 100:  # More than 100MB
                    warnings.append(f"Tensor '{key}' is very large: {tensor_size_mb:.1f}MB")
        
        return warnings
    
    def check_memory_usage(self) -> MemoryStats:
        """Check current memory usage statistics."""
        try:
            # Get system memory info
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Get GPU memory if available and enabled
            gpu_usage_mb = None
            if self.enable_gpu_monitoring and torch.cuda.is_available():
                try:
                    gpu_usage_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                except Exception as e:
                    self.logger.warning(f"Could not get GPU memory usage: {e}")
            
            stats = MemoryStats(
                current_usage_mb=memory.used / (1024 * 1024),
                peak_usage_mb=memory.used / (1024 * 1024),  # psutil doesn't track peak
                available_mb=memory.available / (1024 * 1024),
                gpu_usage_mb=gpu_usage_mb,
                swap_usage_mb=swap.used / (1024 * 1024)
            )
            
            # Store snapshot for trend analysis
            self.memory_snapshots.append(stats)
            if len(self.memory_snapshots) > 100:  # Keep last 100 snapshots
                self.memory_snapshots.pop(0)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error checking memory usage: {e}")
            return MemoryStats(
                current_usage_mb=0.0,
                peak_usage_mb=0.0,
                available_mb=0.0,
                gpu_usage_mb=None,
                swap_usage_mb=0.0
            )
    
    def log_pipeline_metrics(self, module_name: str, processing_time: float):
        """Log processing metrics for pipeline module."""
        if module_name not in self.validation_metrics:
            self.validation_metrics[module_name] = []
        
        self.validation_metrics[module_name].append(processing_time)
        
        # Keep only recent metrics
        if len(self.validation_metrics[module_name]) > 1000:
            self.validation_metrics[module_name] = self.validation_metrics[module_name][-500:]
        
        self.logger.debug(f"Pipeline metrics - {module_name}: {processing_time:.3f}s")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {}
        
        for module_name, times in self.validation_metrics.items():
            if times:
                summary[module_name] = {
                    'count': len(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'total_time': sum(times)
                }
        
        # Add memory trend analysis
        if self.memory_snapshots:
            memory_usage = [snap.current_usage_mb for snap in self.memory_snapshots]
            summary['memory_trends'] = {
                'current_mb': memory_usage[-1] if memory_usage else 0,
                'avg_mb': sum(memory_usage) / len(memory_usage),
                'peak_mb': max(memory_usage),
                'min_mb': min(memory_usage)
            }
            
            if self.memory_snapshots[-1].gpu_usage_mb is not None:
                gpu_usage = [snap.gpu_usage_mb for snap in self.memory_snapshots if snap.gpu_usage_mb is not None]
                if gpu_usage:
                    summary['gpu_memory_trends'] = {
                        'current_mb': gpu_usage[-1],
                        'avg_mb': sum(gpu_usage) / len(gpu_usage),
                        'peak_mb': max(gpu_usage),
                        'min_mb': min(gpu_usage)
                    }
        
        return summary
    
    def validate_data_shapes_and_types(self, item: Dict[str, Any], stage_name: str) -> ValidationResult:
        """
        Validate data shapes and types at pipeline boundaries.
        
        Args:
            item: Pipeline data item
            stage_name: Name of the pipeline stage for context
            
        Returns:
            ValidationResult with boundary validation status
        """
        start_time = time.time()
        warnings = []
        metadata = {'stage': stage_name}
        
        try:
            # Basic validation
            if item is None:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Data is None at stage '{stage_name}'",
                    warnings=[],
                    metadata=metadata,
                    processing_time=time.time() - start_time
                )
            
            if not isinstance(item, dict):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Data must be dict at stage '{stage_name}', got {type(item)}",
                    warnings=[],
                    metadata=metadata,
                    processing_time=time.time() - start_time
                )
            
            # Validate tensor shapes and types
            tensor_info = {}
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    tensor_info[key] = {
                        'shape': list(value.shape),
                        'dtype': str(value.dtype),
                        'device': str(value.device),
                        'requires_grad': value.requires_grad
                    }
                    
                    # Check for common issues
                    if value.numel() == 0:
                        warnings.append(f"Empty tensor '{key}' at stage '{stage_name}'")
                    
                    if torch.isnan(value).any():
                        warnings.append(f"NaN values in tensor '{key}' at stage '{stage_name}'")
                    
                    if torch.isinf(value).any():
                        warnings.append(f"Infinite values in tensor '{key}' at stage '{stage_name}'")
            
            metadata['tensor_info'] = tensor_info
            metadata['data_keys'] = list(item.keys())
            
            return ValidationResult(
                is_valid=True,
                error_message=None,
                warnings=warnings,
                metadata=metadata,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Error validating data at stage '{stage_name}': {str(e)}",
                warnings=[],
                metadata=metadata,
                processing_time=time.time() - start_time
            )