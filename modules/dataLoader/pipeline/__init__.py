"""
Enhanced pipeline modules for robust MGDS data processing.

This package provides enhanced error handling, validation, and recovery
mechanisms for the MGDS video processing pipeline.
"""

from .DataFlowValidator import DataFlowValidator, ValidationResult, VideoMetadata, MemoryStats
from .EnhancedSafePipelineModule import (
    EnhancedSafePipelineModule,
    FallbackStrategy,
    ErrorType,
    ErrorContext,
    CircuitBreakerState
)
from .RobustVideoLoader import RobustVideoLoader, VideoLoadResult
from .ValidatedDiskCache import ValidatedDiskCache, CacheStats, CacheValidationResult
from .PipelineLogger import (
    PipelineLogger,
    LogEntry,
    DataFlowTrace,
    ErrorDiagnostic,
    PerformanceMetric,
    get_pipeline_logger,
    initialize_pipeline_logger
)
from .ErrorRecoveryManager import (
    ErrorRecoveryManager,
    ErrorSeverity,
    RecoveryStrategy,
    FailurePattern,
    RecoveryAction
)
from .MetricsCollector import (
    MetricsCollector,
    ProcessingStats,
    MemoryUsageSnapshot,
    CachePerformanceMetrics,
    TrainingProgressMetrics,
    get_metrics_collector,
    initialize_metrics_collector
)

__all__ = [
    'DataFlowValidator',
    'ValidationResult',
    'VideoMetadata',
    'MemoryStats',
    'EnhancedSafePipelineModule',
    'FallbackStrategy',
    'ErrorType',
    'ErrorContext',
    'CircuitBreakerState',
    'RobustVideoLoader',
    'VideoLoadResult',
    'ValidatedDiskCache',
    'CacheStats',
    'CacheValidationResult',
    'PipelineLogger',
    'LogEntry',
    'DataFlowTrace',
    'ErrorDiagnostic',
    'PerformanceMetric',
    'get_pipeline_logger',
    'initialize_pipeline_logger',
    'ErrorRecoveryManager',
    'ErrorSeverity',
    'RecoveryStrategy',
    'FailurePattern',
    'RecoveryAction',
    'MetricsCollector',
    'ProcessingStats',
    'MemoryUsageSnapshot',
    'CachePerformanceMetrics',
    'TrainingProgressMetrics',
    'get_metrics_collector',
    'initialize_metrics_collector'
]