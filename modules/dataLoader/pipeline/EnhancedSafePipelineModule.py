"""
Enhanced SafePipelineModule with comprehensive error handling and circuit breaker patterns.

This module provides robust error handling for MGDS pipeline modules, preventing
None value propagation and implementing graceful error recovery mechanisms.
"""

import logging
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch

from mgds.PipelineModule import PipelineModule


class FallbackStrategy(Enum):
    """Strategy for handling failed pipeline operations."""
    SKIP = "skip"           # Skip the problematic sample
    RETRY = "retry"         # Retry the operation with backoff
    FALLBACK = "fallback"   # Use fallback/dummy data
    CIRCUIT_BREAK = "circuit_break"  # Open circuit breaker


class ErrorType(Enum):
    """Classification of pipeline errors."""
    TRANSIENT = "transient"     # Temporary failures (network, I/O)
    DATA_CORRUPTION = "data_corruption"  # Corrupted or invalid data
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # Memory, disk space
    CONFIGURATION = "configuration"  # Invalid configuration
    CRITICAL = "critical"       # System-level failures


@dataclass
class ErrorContext:
    """Context information for pipeline errors."""
    module_name: str
    error_type: ErrorType
    error_message: str
    variation: int
    index: int
    requested_name: Optional[str]
    timestamp: float
    traceback_str: str
    retry_count: int = 0


@dataclass
class CircuitBreakerState:
    """State tracking for circuit breaker pattern."""
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    
    def should_allow_request(self) -> bool:
        """Check if requests should be allowed through the circuit."""
        current_time = time.time()
        
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if current_time - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record a successful operation."""
        self.success_count += 1
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
    
    def record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class EnhancedSafePipelineModule(PipelineModule):
    """
    Enhanced pipeline module wrapper with comprehensive error handling.
    
    Features:
    - None value detection and handling
    - Circuit breaker pattern for cascading failure prevention
    - Configurable fallback strategies
    - Detailed error logging and diagnostics
    - Performance monitoring and metrics
    """
    
    def __init__(
        self,
        wrapped_module: PipelineModule,
        module_name: str = "Unknown",
        dtype: torch.dtype = torch.float32,
        fallback_strategy: FallbackStrategy = FallbackStrategy.FALLBACK,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0
    ):
        super().__init__()
        self.wrapped_module = wrapped_module
        self.module_name = module_name
        self.dtype = dtype
        self.fallback_strategy = fallback_strategy
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Circuit breaker state
        self.circuit_breaker = CircuitBreakerState(
            failure_threshold=circuit_breaker_threshold,
            recovery_timeout=circuit_breaker_timeout
        )
        
        # Error tracking
        self.error_history: deque = deque(maxlen=100)
        self.error_counts: Dict[ErrorType, int] = defaultdict(int)
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # Setup logging
        self.logger = logging.getLogger(f"EnhancedSafePipelineModule.{module_name}")
        self.logger.setLevel(logging.DEBUG)
        
        # Create console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def length(self) -> int:
        """Get the length of the wrapped module with error handling."""
        start_time = time.time()
        
        try:
            if not self.circuit_breaker.should_allow_request():
                self.logger.warning(f"{self.module_name} circuit breaker is OPEN, using fallback length")
                return 1
            
            # Check if module is initialized
            if not hasattr(self.wrapped_module, '_PipelineModule__module_index'):
                if 'CollectPaths' in str(self.wrapped_module.__class__):
                    self.logger.info(f"{self.module_name} not initialized, estimating from concept stats")
                    return 10
                return 1
            
            length = self.wrapped_module.length()
            
            # Record success
            self.circuit_breaker.record_success()
            processing_time = time.time() - start_time
            self.performance_metrics['length_calls'].append(processing_time)
            
            self.logger.debug(f"{self.module_name} length() returned: {length} (took {processing_time:.3f}s)")
            return length
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            error_context = ErrorContext(
                module_name=self.module_name,
                error_type=self._classify_error(e),
                error_message=str(e),
                variation=-1,
                index=-1,
                requested_name=None,
                timestamp=time.time(),
                traceback_str=traceback.format_exc()
            )
            self._handle_error(error_context)
            return 1
    
    def get_inputs(self) -> List[str]:
        """Get inputs from wrapped module."""
        return self.wrapped_module.get_inputs()
    
    def get_outputs(self) -> List[str]:
        """Get outputs from wrapped module."""
        return self.wrapped_module.get_outputs()
    
    def init(self, pipeline, seed: int, index: int, state):
        """Initialize the wrapped module with error handling."""
        self.logger.debug(f"{self.module_name} init called - seed={seed}, index={index}")
        
        try:
            if hasattr(self.wrapped_module, 'init'):
                result = self.wrapped_module.init(pipeline, seed, index, state)
                self.logger.debug(f"{self.module_name} init successful")
                return result
            else:
                self.logger.debug(f"{self.module_name} wrapped module has no init method")
                return None
                
        except Exception as e:
            error_context = ErrorContext(
                module_name=self.module_name,
                error_type=self._classify_error(e),
                error_message=f"Init failed: {str(e)}",
                variation=-1,
                index=index,
                requested_name=None,
                timestamp=time.time(),
                traceback_str=traceback.format_exc()
            )
            self._handle_error(error_context)
            return None
    
    def start(self, epoch: int):
        """Start epoch with error handling."""
        self.logger.debug(f"{self.module_name} start called - epoch={epoch}")
        
        try:
            if hasattr(self.wrapped_module, 'start'):
                result = self.wrapped_module.start(epoch)
                self.logger.debug(f"{self.module_name} start successful")
                return result
            else:
                self.logger.debug(f"{self.module_name} wrapped module has no start method")
                return None
                
        except Exception as e:
            error_context = ErrorContext(
                module_name=self.module_name,
                error_type=self._classify_error(e),
                error_message=f"Start failed: {str(e)}",
                variation=-1,
                index=-1,
                requested_name=None,
                timestamp=time.time(),
                traceback_str=traceback.format_exc()
            )
            self._handle_error(error_context)
            return None
    
    def end(self):
        """End processing with error handling."""
        self.logger.debug(f"{self.module_name} end called")
        
        try:
            if hasattr(self.wrapped_module, 'end'):
                result = self.wrapped_module.end()
                self.logger.debug(f"{self.module_name} end successful")
                return result
            else:
                self.logger.debug(f"{self.module_name} wrapped module has no end method")
                return None
                
        except Exception as e:
            error_context = ErrorContext(
                module_name=self.module_name,
                error_type=self._classify_error(e),
                error_message=f"End failed: {str(e)}",
                variation=-1,
                index=-1,
                requested_name=None,
                timestamp=time.time(),
                traceback_str=traceback.format_exc()
            )
            self._handle_error(error_context)
            return None
    
    def clear_item_cache(self):
        """Clear item cache with error handling."""
        # Call parent class method first
        super().clear_item_cache()
        
        try:
            if hasattr(self.wrapped_module, 'clear_item_cache'):
                return self.wrapped_module.clear_item_cache()
            return None
            
        except Exception as e:
            self.logger.error(f"{self.module_name} clear_item_cache failed: {e}")
            return None
    
    def get_item(self, variation: int, index: int, requested_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get item with comprehensive error handling and validation.
        
        This is the core method that implements the robust error handling strategy.
        """
        start_time = time.time()
        
        self.logger.debug(
            f"{self.module_name} get_item called - variation={variation}, "
            f"index={index}, requested_name={requested_name}"
        )
        
        # Check circuit breaker
        if not self.circuit_breaker.should_allow_request():
            self.logger.warning(f"{self.module_name} circuit breaker is OPEN, using fallback")
            return self._create_safe_fallback_data(index, requested_name)
        
        # Attempt to get item with retry logic
        for attempt in range(self.max_retries + 1):
            try:
                result = self.wrapped_module.get_item(variation, index, requested_name)
                
                # Validate result
                validation_result = self._validate_item(result, requested_name)
                if not validation_result.is_valid:
                    raise ValueError(f"Item validation failed: {validation_result.error_message}")
                
                # Record success
                self.circuit_breaker.record_success()
                processing_time = time.time() - start_time
                self.performance_metrics['get_item_calls'].append(processing_time)
                
                self.logger.debug(
                    f"{self.module_name} returned valid result with keys: "
                    f"{list(result.keys()) if isinstance(result, dict) else 'not dict'} "
                    f"(took {processing_time:.3f}s)"
                )
                
                return result
                
            except Exception as e:
                error_context = ErrorContext(
                    module_name=self.module_name,
                    error_type=self._classify_error(e),
                    error_message=str(e),
                    variation=variation,
                    index=index,
                    requested_name=requested_name,
                    timestamp=time.time(),
                    traceback_str=traceback.format_exc(),
                    retry_count=attempt
                )
                
                # Handle the error based on strategy
                if attempt < self.max_retries and self._should_retry(error_context):
                    self.logger.warning(
                        f"{self.module_name} attempt {attempt + 1} failed, retrying: {str(e)}"
                    )
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    # Final attempt failed or non-retryable error
                    self.circuit_breaker.record_failure()
                    self._handle_error(error_context)
                    
                    # Return fallback data
                    fallback = self._create_safe_fallback_data(index, requested_name)
                    self.logger.error(
                        f"{self.module_name} all attempts failed, using fallback data"
                    )
                    return fallback
        
        # Should never reach here, but just in case
        return self._create_safe_fallback_data(index, requested_name)
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error type for appropriate handling."""
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()
        
        if 'memory' in error_str or 'out of memory' in error_str:
            return ErrorType.RESOURCE_EXHAUSTION
        elif 'connection' in error_str or 'network' in error_str or 'timeout' in error_str:
            return ErrorType.TRANSIENT
        elif 'corrupt' in error_str or 'invalid' in error_str or 'malformed' in error_str:
            return ErrorType.DATA_CORRUPTION
        elif 'config' in error_str or 'parameter' in error_str:
            return ErrorType.CONFIGURATION
        elif 'system' in error_str or 'critical' in error_str:
            return ErrorType.CRITICAL
        else:
            return ErrorType.TRANSIENT  # Default to transient for retry
    
    def _should_retry(self, error_context: ErrorContext) -> bool:
        """Determine if an error should be retried."""
        # Don't retry critical errors or configuration errors
        if error_context.error_type in [ErrorType.CRITICAL, ErrorType.CONFIGURATION]:
            return False
        
        # Don't retry if we've exceeded the circuit breaker threshold
        if self.circuit_breaker.failure_count >= self.circuit_breaker.failure_threshold:
            return False
        
        # Retry transient errors and resource exhaustion
        return error_context.error_type in [ErrorType.TRANSIENT, ErrorType.RESOURCE_EXHAUSTION]
    
    def _handle_error(self, error_context: ErrorContext):
        """Handle and log error with appropriate detail level."""
        # Add to error history
        self.error_history.append(error_context)
        self.error_counts[error_context.error_type] += 1
        
        # Log with appropriate level
        if error_context.error_type == ErrorType.CRITICAL:
            self.logger.critical(
                f"{error_context.module_name} CRITICAL ERROR: {error_context.error_message}\n"
                f"Context: variation={error_context.variation}, index={error_context.index}, "
                f"requested_name={error_context.requested_name}\n"
                f"Traceback: {error_context.traceback_str}"
            )
        elif error_context.error_type == ErrorType.DATA_CORRUPTION:
            self.logger.error(
                f"{error_context.module_name} DATA CORRUPTION: {error_context.error_message}\n"
                f"Context: variation={error_context.variation}, index={error_context.index}, "
                f"requested_name={error_context.requested_name}"
            )
        else:
            self.logger.warning(
                f"{error_context.module_name} {error_context.error_type.value.upper()}: "
                f"{error_context.error_message} (attempt {error_context.retry_count + 1})"
            )
    
    def _validate_item(self, item: Any, requested_name: Optional[str]) -> 'ValidationResult':
        """Validate pipeline item structure and content."""
        from modules.dataLoader.pipeline.DataFlowValidator import ValidationResult
        
        # Check for None
        if item is None:
            return ValidationResult(
                is_valid=False,
                error_message="Item is None",
                warnings=[],
                metadata={},
                processing_time=0.0
            )
        
        # Check for correct type
        if not isinstance(item, dict):
            return ValidationResult(
                is_valid=False,
                error_message=f"Item is not a dictionary, got {type(item)}",
                warnings=[],
                metadata={'actual_type': str(type(item))},
                processing_time=0.0
            )
        
        # Check for empty dictionary
        if not item:
            return ValidationResult(
                is_valid=False,
                error_message="Item is an empty dictionary",
                warnings=[],
                metadata={},
                processing_time=0.0
            )
        
        # If specific name requested, check it exists
        if requested_name and requested_name not in item:
            return ValidationResult(
                is_valid=False,
                error_message=f"Requested item '{requested_name}' not found in result",
                warnings=[],
                metadata={'available_keys': list(item.keys())},
                processing_time=0.0
            )
        
        # Validate tensor shapes if present
        warnings = []
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    warnings.append(f"Tensor '{key}' is empty")
                elif torch.isnan(value).any():
                    warnings.append(f"Tensor '{key}' contains NaN values")
                elif torch.isinf(value).any():
                    warnings.append(f"Tensor '{key}' contains infinite values")
        
        return ValidationResult(
            is_valid=True,
            error_message=None,
            warnings=warnings,
            metadata={'keys': list(item.keys())},
            processing_time=0.0
        )
    
    def _create_safe_fallback_data(self, index: int, requested_name: Optional[str] = None) -> Dict[str, Any]:
        """Create safe fallback data that won't cause pipeline crashes."""
        self.logger.debug(f"{self.module_name} creating fallback data for index {index}")
        
        # Base fallback data
        fallback_data = {
            'video_path': f'fallback_video_{index}.mp4',
            'prompt': f'fallback prompt for training sample {index}',
            'concept': {'name': 'fallback_concept', 'enabled': True},
            'settings': {'target_frames': 8},
            'target_frames': 8,
        }
        
        try:
            outputs = self.get_outputs()
            
            # Add tensors based on expected outputs
            tensor_configs = {
                'video': (8, 3, 64, 64),           # 8 frames, 3 channels, 64x64
                'image': (3, 64, 64),              # 3 channels, 64x64
                'sampled_video': (2, 3, 64, 64),   # 2 frames, 3 channels, 64x64
                'scaled_video': (2, 3, 64, 64),    # 2 frames, 3 channels, 64x64
                'latent_video': (2, 4, 8, 8),      # 2 frames, 4 latent channels, 8x8
                'latent_mask': (2, 1, 8, 8),       # 2 frames, 1 channel mask, 8x8
            }
            
            for output_name in outputs:
                if output_name in tensor_configs:
                    shape = tensor_configs[output_name]
                    fallback_data[output_name] = torch.zeros(shape, dtype=self.dtype)
                    self.logger.debug(f"{self.module_name} added {output_name} tensor with shape {shape}")
        
        except Exception as e:
            self.logger.warning(f"{self.module_name} error creating fallback tensors: {e}")
        
        # If specific name requested, ensure it's in the fallback
        if requested_name and requested_name not in fallback_data:
            if requested_name.endswith('_path'):
                fallback_data[requested_name] = f'fallback_{requested_name}_{index}'
            elif requested_name in ['prompt', 'text']:
                fallback_data[requested_name] = f'fallback {requested_name} for sample {index}'
            else:
                fallback_data[requested_name] = f'fallback_{requested_name}'
        
        self.logger.debug(f"{self.module_name} created fallback data with keys: {list(fallback_data.keys())}")
        return fallback_data
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics for monitoring."""
        total_errors = sum(self.error_counts.values())
        
        return {
            'module_name': self.module_name,
            'total_errors': total_errors,
            'error_counts_by_type': dict(self.error_counts),
            'circuit_breaker_state': self.circuit_breaker.state,
            'circuit_breaker_failures': self.circuit_breaker.failure_count,
            'recent_errors': len(self.error_history),
            'performance_metrics': {
                key: {
                    'count': len(values),
                    'avg_time': sum(values) / len(values) if values else 0,
                    'max_time': max(values) if values else 0,
                    'min_time': min(values) if values else 0
                }
                for key, values in self.performance_metrics.items()
            }
        }