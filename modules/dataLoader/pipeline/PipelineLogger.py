"""
Comprehensive logging and diagnostics system for MGDS pipeline.

This module provides structured logging, data flow tracing, error diagnostics,
and performance monitoring capabilities for the video processing pipeline.
"""

import json
import logging
import os
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import torch


@dataclass
class LogEntry:
    """Structured log entry for pipeline operations."""
    timestamp: float
    level: str
    module_name: str
    operation: str
    message: str
    metadata: Dict[str, Any]
    duration_ms: Optional[float] = None
    error_details: Optional[str] = None


@dataclass
class DataFlowTrace:
    """Data flow trace entry."""
    timestamp: float
    module_name: str
    module_index: int
    operation: str  # 'input', 'output', 'transform'
    item_name: str
    item_type: str
    item_shape: Optional[str]
    variation: int
    index: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class ErrorDiagnostic:
    """Comprehensive error diagnostic information."""
    timestamp: float
    module_name: str
    error_type: str
    error_message: str
    context: Dict[str, Any]
    traceback_str: str
    recovery_action: Optional[str] = None
    impact_level: str = "medium"  # low, medium, high, critical


@dataclass
class PerformanceMetric:
    """Performance metric entry."""
    timestamp: float
    module_name: str
    operation: str
    duration_ms: float
    memory_usage_mb: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    metadata: Dict[str, Any] = None


class PipelineLogger:
    """
    Comprehensive logging and diagnostics system for MGDS pipeline.
    
    Features:
    - Structured logging with consistent format
    - Data flow tracing capabilities
    - Error diagnostic reporting
    - Performance metrics collection
    - Video processing detail logging
    - Export capabilities for analysis
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        enable_file_logging: bool = True,
        enable_console_logging: bool = True,
        log_level: str = "DEBUG",
        max_trace_entries: int = 10000,
        max_error_entries: int = 1000,
        max_performance_entries: int = 5000,
        enable_data_flow_tracing: bool = True,
        enable_performance_monitoring: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        self.log_level = getattr(logging, log_level.upper())
        self.max_trace_entries = max_trace_entries
        self.max_error_entries = max_error_entries
        self.max_performance_entries = max_performance_entries
        self.enable_data_flow_tracing = enable_data_flow_tracing
        self.enable_performance_monitoring = enable_performance_monitoring
        
        # Create log directory
        if self.enable_file_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self.log_entries: deque = deque(maxlen=max_trace_entries)
        self.data_flow_traces: deque = deque(maxlen=max_trace_entries)
        self.error_diagnostics: deque = deque(maxlen=max_error_entries)
        self.performance_metrics: deque = deque(maxlen=max_performance_entries)
        
        # Module-specific loggers
        self.module_loggers: Dict[str, logging.Logger] = {}
        
        # Statistics
        self.stats = {
            'total_logs': 0,
            'error_count': 0,
            'warning_count': 0,
            'trace_count': 0,
            'performance_count': 0
        }
        
        # Setup main logger
        self.main_logger = self._setup_logger("PipelineLogger")
        self.main_logger.info("PipelineLogger initialized")
    
    def _setup_logger(self, name: str) -> logging.Logger:
        """Setup a logger with consistent formatting."""
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if self.enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if self.enable_file_logging:
            log_file = self.log_dir / f"{name.lower()}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def get_module_logger(self, module_name: str) -> logging.Logger:
        """Get or create a logger for a specific module."""
        if module_name not in self.module_loggers:
            self.module_loggers[module_name] = self._setup_logger(f"Pipeline.{module_name}")
        return self.module_loggers[module_name]
    
    def log_operation(
        self,
        module_name: str,
        operation: str,
        message: str,
        level: str = "INFO",
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        error_details: Optional[str] = None
    ):
        """Log a pipeline operation with structured data."""
        timestamp = time.time()
        
        # Create log entry
        entry = LogEntry(
            timestamp=timestamp,
            level=level,
            module_name=module_name,
            operation=operation,
            message=message,
            metadata=metadata or {},
            duration_ms=duration_ms,
            error_details=error_details
        )
        
        # Store entry
        self.log_entries.append(entry)
        self.stats['total_logs'] += 1
        
        if level == "ERROR":
            self.stats['error_count'] += 1
        elif level == "WARNING":
            self.stats['warning_count'] += 1
        
        # Log to appropriate logger
        logger = self.get_module_logger(module_name)
        log_message = f"[{operation}] {message}"
        
        if metadata:
            log_message += f" | Metadata: {json.dumps(metadata, default=str)}"
        
        if duration_ms is not None:
            log_message += f" | Duration: {duration_ms:.2f}ms"
        
        if error_details:
            log_message += f" | Error: {error_details}"
        
        getattr(logger, level.lower())(log_message)
    
    def trace_data_flow(
        self,
        module_name: str,
        module_index: int,
        operation: str,
        item_name: str,
        item_data: Any,
        variation: int,
        index: int,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Trace data flow through the pipeline."""
        if not self.enable_data_flow_tracing:
            return
        
        timestamp = time.time()
        
        # Analyze item data
        item_type = type(item_data).__name__
        item_shape = None
        
        if isinstance(item_data, torch.Tensor):
            item_shape = str(list(item_data.shape))
        elif isinstance(item_data, dict):
            item_type = f"dict[{len(item_data)}]"
            # Get shapes of tensor values
            tensor_shapes = {}
            for key, value in item_data.items():
                if isinstance(value, torch.Tensor):
                    tensor_shapes[key] = list(value.shape)
            if tensor_shapes:
                item_shape = json.dumps(tensor_shapes, default=str)
        elif isinstance(item_data, (list, tuple)):
            item_type = f"{item_type}[{len(item_data)}]"
        elif isinstance(item_data, str):
            item_shape = f"len={len(item_data)}"
        
        # Create trace entry
        trace = DataFlowTrace(
            timestamp=timestamp,
            module_name=module_name,
            module_index=module_index,
            operation=operation,
            item_name=item_name,
            item_type=item_type,
            item_shape=item_shape,
            variation=variation,
            index=index,
            success=success,
            error_message=error_message
        )
        
        # Store trace
        self.data_flow_traces.append(trace)
        self.stats['trace_count'] += 1
        
        # Log trace
        logger = self.get_module_logger(module_name)
        trace_message = (
            f"[TRACE] {operation} | {item_name} | {item_type} | "
            f"var={variation} idx={index} | {'✓' if success else '✗'}"
        )
        
        if item_shape:
            trace_message += f" | shape={item_shape}"
        
        if error_message:
            trace_message += f" | error={error_message}"
        
        logger.debug(trace_message)
    
    def log_error_diagnostic(
        self,
        module_name: str,
        error_type: str,
        error_message: str,
        context: Dict[str, Any],
        traceback_str: Optional[str] = None,
        recovery_action: Optional[str] = None,
        impact_level: str = "medium"
    ):
        """Log comprehensive error diagnostic information."""
        timestamp = time.time()
        
        # Create diagnostic entry
        diagnostic = ErrorDiagnostic(
            timestamp=timestamp,
            module_name=module_name,
            error_type=error_type,
            error_message=error_message,
            context=context,
            traceback_str=traceback_str or traceback.format_exc(),
            recovery_action=recovery_action,
            impact_level=impact_level
        )
        
        # Store diagnostic
        self.error_diagnostics.append(diagnostic)
        
        # Log diagnostic
        logger = self.get_module_logger(module_name)
        diagnostic_message = (
            f"[ERROR_DIAGNOSTIC] {error_type} | {error_message} | "
            f"impact={impact_level}"
        )
        
        if context:
            diagnostic_message += f" | context={json.dumps(context, default=str)}"
        
        if recovery_action:
            diagnostic_message += f" | recovery={recovery_action}"
        
        logger.error(diagnostic_message)
        
        # Also log to main logger for critical errors
        if impact_level in ["high", "critical"]:
            self.main_logger.critical(f"CRITICAL ERROR in {module_name}: {error_message}")
    
    def log_performance_metric(
        self,
        module_name: str,
        operation: str,
        duration_ms: float,
        memory_usage_mb: Optional[float] = None,
        gpu_memory_mb: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log performance metrics."""
        if not self.enable_performance_monitoring:
            return
        
        timestamp = time.time()
        
        # Create metric entry
        metric = PerformanceMetric(
            timestamp=timestamp,
            module_name=module_name,
            operation=operation,
            duration_ms=duration_ms,
            memory_usage_mb=memory_usage_mb,
            gpu_memory_mb=gpu_memory_mb,
            metadata=metadata or {}
        )
        
        # Store metric
        self.performance_metrics.append(metric)
        self.stats['performance_count'] += 1
        
        # Log metric
        logger = self.get_module_logger(module_name)
        metric_message = f"[PERF] {operation} | {duration_ms:.2f}ms"
        
        if memory_usage_mb is not None:
            metric_message += f" | mem={memory_usage_mb:.1f}MB"
        
        if gpu_memory_mb is not None:
            metric_message += f" | gpu={gpu_memory_mb:.1f}MB"
        
        if metadata:
            metric_message += f" | {json.dumps(metadata, default=str)}"
        
        logger.debug(metric_message)
    
    def log_video_processing_details(
        self,
        module_name: str,
        video_path: str,
        operation: str,
        success: bool,
        details: Dict[str, Any],
        duration_ms: Optional[float] = None,
        error_message: Optional[str] = None
    ):
        """Log detailed video processing information."""
        metadata = {
            'video_path': video_path,
            'success': success,
            **details
        }
        
        level = "INFO" if success else "ERROR"
        message = f"Video {operation}: {os.path.basename(video_path)}"
        
        if not success and error_message:
            message += f" - {error_message}"
        
        self.log_operation(
            module_name=module_name,
            operation=f"video_{operation}",
            message=message,
            level=level,
            metadata=metadata,
            duration_ms=duration_ms,
            error_details=error_message if not success else None
        )
        
        # Also trace the data flow
        if self.enable_data_flow_tracing:
            self.trace_data_flow(
                module_name=module_name,
                module_index=-1,  # Unknown for video processing
                operation=operation,
                item_name="video",
                item_data=details.get('tensor_shape', 'unknown'),
                variation=details.get('variation', -1),
                index=details.get('index', -1),
                success=success,
                error_message=error_message
            )
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline execution summary."""
        current_time = time.time()
        
        # Calculate time ranges
        if self.log_entries:
            start_time = min(entry.timestamp for entry in self.log_entries)
            end_time = max(entry.timestamp for entry in self.log_entries)
            duration_seconds = end_time - start_time
        else:
            start_time = current_time
            end_time = current_time
            duration_seconds = 0
        
        # Module statistics
        module_stats = defaultdict(lambda: {
            'operations': 0,
            'errors': 0,
            'warnings': 0,
            'avg_duration_ms': 0,
            'total_duration_ms': 0
        })
        
        for entry in self.log_entries:
            stats = module_stats[entry.module_name]
            stats['operations'] += 1
            
            if entry.level == "ERROR":
                stats['errors'] += 1
            elif entry.level == "WARNING":
                stats['warnings'] += 1
            
            if entry.duration_ms is not None:
                stats['total_duration_ms'] += entry.duration_ms
        
        # Calculate averages
        for module_name, stats in module_stats.items():
            if stats['operations'] > 0:
                stats['avg_duration_ms'] = stats['total_duration_ms'] / stats['operations']
        
        # Performance summary
        performance_summary = {}
        if self.performance_metrics:
            operations = defaultdict(list)
            for metric in self.performance_metrics:
                operations[metric.operation].append(metric.duration_ms)
            
            for operation, durations in operations.items():
                performance_summary[operation] = {
                    'count': len(durations),
                    'avg_ms': sum(durations) / len(durations),
                    'min_ms': min(durations),
                    'max_ms': max(durations),
                    'total_ms': sum(durations)
                }
        
        # Error summary
        error_summary = defaultdict(int)
        for diagnostic in self.error_diagnostics:
            error_summary[diagnostic.error_type] += 1
        
        return {
            'execution_summary': {
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.fromtimestamp(end_time).isoformat(),
                'duration_seconds': duration_seconds,
                'total_operations': len(self.log_entries),
                'total_errors': self.stats['error_count'],
                'total_warnings': self.stats['warning_count']
            },
            'module_statistics': dict(module_stats),
            'performance_summary': performance_summary,
            'error_summary': dict(error_summary),
            'data_flow_traces': len(self.data_flow_traces),
            'system_stats': self.stats
        }
    
    def export_logs(self, export_dir: Optional[str] = None) -> Dict[str, str]:
        """Export all logs to files for analysis."""
        if export_dir is None:
            export_dir = self.log_dir / "exports" / datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            export_dir = Path(export_dir)
        
        export_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # Export log entries
        if self.log_entries:
            log_file = export_dir / "log_entries.json"
            with open(log_file, 'w') as f:
                json.dump([asdict(entry) for entry in self.log_entries], f, indent=2, default=str)
            exported_files['log_entries'] = str(log_file)
        
        # Export data flow traces
        if self.data_flow_traces:
            trace_file = export_dir / "data_flow_traces.json"
            with open(trace_file, 'w') as f:
                json.dump([asdict(trace) for trace in self.data_flow_traces], f, indent=2, default=str)
            exported_files['data_flow_traces'] = str(trace_file)
        
        # Export error diagnostics
        if self.error_diagnostics:
            error_file = export_dir / "error_diagnostics.json"
            with open(error_file, 'w') as f:
                json.dump([asdict(diagnostic) for diagnostic in self.error_diagnostics], f, indent=2, default=str)
            exported_files['error_diagnostics'] = str(error_file)
        
        # Export performance metrics
        if self.performance_metrics:
            perf_file = export_dir / "performance_metrics.json"
            with open(perf_file, 'w') as f:
                json.dump([asdict(metric) for metric in self.performance_metrics], f, indent=2, default=str)
            exported_files['performance_metrics'] = str(perf_file)
        
        # Export summary
        summary_file = export_dir / "pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.get_pipeline_summary(), f, indent=2, default=str)
        exported_files['summary'] = str(summary_file)
        
        self.main_logger.info(f"Logs exported to {export_dir}")
        return exported_files
    
    def clear_logs(self):
        """Clear all stored logs and reset statistics."""
        self.log_entries.clear()
        self.data_flow_traces.clear()
        self.error_diagnostics.clear()
        self.performance_metrics.clear()
        
        self.stats = {
            'total_logs': 0,
            'error_count': 0,
            'warning_count': 0,
            'trace_count': 0,
            'performance_count': 0
        }
        
        self.main_logger.info("All logs cleared")


# Global pipeline logger instance
_global_pipeline_logger: Optional[PipelineLogger] = None


def get_pipeline_logger() -> PipelineLogger:
    """Get the global pipeline logger instance."""
    global _global_pipeline_logger
    if _global_pipeline_logger is None:
        _global_pipeline_logger = PipelineLogger()
    return _global_pipeline_logger


def initialize_pipeline_logger(
    log_dir: str = "logs",
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
    log_level: str = "DEBUG"
) -> PipelineLogger:
    """Initialize the global pipeline logger with custom settings."""
    global _global_pipeline_logger
    _global_pipeline_logger = PipelineLogger(
        log_dir=log_dir,
        enable_file_logging=enable_file_logging,
        enable_console_logging=enable_console_logging,
        log_level=log_level
    )
    return _global_pipeline_logger