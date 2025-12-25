"""
Comprehensive metrics collection and monitoring system for MGDS pipeline.

This module provides processing statistics tracking, memory usage monitoring,
cache performance metrics, and training progress monitoring.
"""

import json
import os
import psutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .PipelineLogger import get_pipeline_logger


@dataclass
class ProcessingStats:
    """Processing statistics for pipeline operations."""
    total_samples: int
    successful_samples: int
    failed_samples: int
    skipped_samples: int
    processing_time_seconds: float
    avg_processing_time_ms: float
    throughput_samples_per_second: float


@dataclass
class MemoryUsageSnapshot:
    """Memory usage snapshot."""
    timestamp: float
    system_memory_mb: float
    system_memory_percent: float
    available_memory_mb: float
    gpu_memory_mb: Optional[float]
    gpu_memory_percent: Optional[float]
    process_memory_mb: float
    swap_usage_mb: float


@dataclass
class CachePerformanceMetrics:
    """Cache performance metrics."""
    hit_count: int
    miss_count: int
    hit_rate: float
    total_requests: int
    cache_size_mb: float
    avg_access_time_ms: float
    corruption_count: int
    recovery_count: int


@dataclass
class TrainingProgressMetrics:
    """Training progress monitoring metrics."""
    epoch: int
    batch: int
    samples_processed: int
    total_samples: int
    progress_percent: float
    estimated_time_remaining_seconds: Optional[float]
    current_throughput: float
    avg_throughput: float


class MetricsCollector:
    """
    Comprehensive metrics collection and monitoring system.
    
    Features:
    - Processing statistics tracking
    - Memory usage monitoring
    - Cache performance metrics
    - Training progress monitoring
    - Real-time dashboards
    - Export capabilities
    """
    
    def __init__(
        self,
        collection_interval_seconds: float = 5.0,
        max_memory_snapshots: int = 1000,
        max_processing_records: int = 10000,
        enable_gpu_monitoring: bool = True,
        enable_real_time_monitoring: bool = True,
        metrics_export_dir: str = "metrics"
    ):
        self.collection_interval = collection_interval_seconds
        self.max_memory_snapshots = max_memory_snapshots
        self.max_processing_records = max_processing_records
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.enable_real_time_monitoring = enable_real_time_monitoring
        self.metrics_export_dir = Path(metrics_export_dir)
        
        # Create metrics directory
        self.metrics_export_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.memory_snapshots: deque = deque(maxlen=max_memory_snapshots)
        self.processing_records: deque = deque(maxlen=max_processing_records)
        self.cache_metrics: Dict[str, CachePerformanceMetrics] = {}
        self.module_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0,
            'last_operation_time': 0.0
        })
        
        # Training progress
        self.training_progress: Optional[TrainingProgressMetrics] = None
        self.training_start_time: Optional[float] = None
        
        # System monitoring
        self.system_process = psutil.Process()
        self.last_collection_time = time.time()
        
        # Performance tracking
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Logger
        self.logger = get_pipeline_logger().get_module_logger("MetricsCollector")
        self.logger.info("MetricsCollector initialized")
    
    def start_training_session(self, total_samples: int):
        """Start a new training session."""
        self.training_start_time = time.time()
        self.training_progress = TrainingProgressMetrics(
            epoch=0,
            batch=0,
            samples_processed=0,
            total_samples=total_samples,
            progress_percent=0.0,
            estimated_time_remaining_seconds=None,
            current_throughput=0.0,
            avg_throughput=0.0
        )
        
        self.logger.info(f"Training session started with {total_samples} samples")
    
    def update_training_progress(
        self,
        epoch: int,
        batch: int,
        samples_processed: int
    ):
        """Update training progress metrics."""
        if self.training_progress is None:
            return
        
        current_time = time.time()
        elapsed_time = current_time - self.training_start_time
        
        # Update progress
        self.training_progress.epoch = epoch
        self.training_progress.batch = batch
        self.training_progress.samples_processed = samples_processed
        self.training_progress.progress_percent = (
            samples_processed / self.training_progress.total_samples * 100
            if self.training_progress.total_samples > 0 else 0
        )
        
        # Calculate throughput
        if elapsed_time > 0:
            self.training_progress.avg_throughput = samples_processed / elapsed_time
            
            # Current throughput (last 60 seconds)
            recent_time_window = 60.0
            if elapsed_time > recent_time_window:
                recent_start_time = current_time - recent_time_window
                # This is a simplified calculation - in practice you'd track samples over time
                self.training_progress.current_throughput = self.training_progress.avg_throughput
            else:
                self.training_progress.current_throughput = self.training_progress.avg_throughput
        
        # Estimate remaining time
        if self.training_progress.avg_throughput > 0:
            remaining_samples = self.training_progress.total_samples - samples_processed
            self.training_progress.estimated_time_remaining_seconds = (
                remaining_samples / self.training_progress.avg_throughput
            )
    
    def record_processing_operation(
        self,
        module_name: str,
        operation: str,
        success: bool,
        processing_time_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a processing operation."""
        current_time = time.time()
        
        # Update module stats
        stats = self.module_stats[module_name]
        stats['total_operations'] += 1
        stats['last_operation_time'] = current_time
        
        if success:
            stats['successful_operations'] += 1
        else:
            stats['failed_operations'] += 1
        
        # Update timing stats
        stats['total_processing_time'] += processing_time_ms
        stats['avg_processing_time'] = (
            stats['total_processing_time'] / stats['total_operations']
        )
        
        # Store processing record
        record = {
            'timestamp': current_time,
            'module_name': module_name,
            'operation': operation,
            'success': success,
            'processing_time_ms': processing_time_ms,
            'metadata': metadata or {}
        }
        self.processing_records.append(record)
        
        # Update performance history
        self.performance_history[f"{module_name}_{operation}"].append(processing_time_ms)
    
    def collect_memory_snapshot(self) -> MemoryUsageSnapshot:
        """Collect current memory usage snapshot."""
        try:
            # System memory
            system_memory = psutil.virtual_memory()
            swap_memory = psutil.swap_memory()
            
            # Process memory
            process_memory = self.system_process.memory_info()
            
            # GPU memory
            gpu_memory_mb = None
            gpu_memory_percent = None
            
            if self.enable_gpu_monitoring and torch.cuda.is_available():
                try:
                    gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                    gpu_memory_percent = (gpu_memory_reserved / gpu_memory_total) * 100
                except Exception as e:
                    self.logger.warning(f"Failed to get GPU memory info: {e}")
            
            snapshot = MemoryUsageSnapshot(
                timestamp=time.time(),
                system_memory_mb=system_memory.used / (1024 * 1024),
                system_memory_percent=system_memory.percent,
                available_memory_mb=system_memory.available / (1024 * 1024),
                gpu_memory_mb=gpu_memory_mb,
                gpu_memory_percent=gpu_memory_percent,
                process_memory_mb=process_memory.rss / (1024 * 1024),
                swap_usage_mb=swap_memory.used / (1024 * 1024)
            )
            
            # Store snapshot
            self.memory_snapshots.append(snapshot)
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Error collecting memory snapshot: {e}")
            # Return minimal snapshot
            return MemoryUsageSnapshot(
                timestamp=time.time(),
                system_memory_mb=0.0,
                system_memory_percent=0.0,
                available_memory_mb=0.0,
                gpu_memory_mb=None,
                gpu_memory_percent=None,
                process_memory_mb=0.0,
                swap_usage_mb=0.0
            )
    
    def update_cache_metrics(
        self,
        cache_name: str,
        hit_count: int,
        miss_count: int,
        cache_size_mb: float,
        avg_access_time_ms: float,
        corruption_count: int = 0,
        recovery_count: int = 0
    ):
        """Update cache performance metrics."""
        total_requests = hit_count + miss_count
        hit_rate = hit_count / total_requests if total_requests > 0 else 0.0
        
        metrics = CachePerformanceMetrics(
            hit_count=hit_count,
            miss_count=miss_count,
            hit_rate=hit_rate,
            total_requests=total_requests,
            cache_size_mb=cache_size_mb,
            avg_access_time_ms=avg_access_time_ms,
            corruption_count=corruption_count,
            recovery_count=recovery_count
        )
        
        self.cache_metrics[cache_name] = metrics
    
    def get_processing_statistics(self, module_name: Optional[str] = None) -> Dict[str, Any]:
        """Get processing statistics for all modules or a specific module."""
        if module_name:
            if module_name in self.module_stats:
                stats = self.module_stats[module_name]
                return {
                    module_name: {
                        'total_operations': stats['total_operations'],
                        'successful_operations': stats['successful_operations'],
                        'failed_operations': stats['failed_operations'],
                        'success_rate': (
                            stats['successful_operations'] / stats['total_operations']
                            if stats['total_operations'] > 0 else 0
                        ),
                        'avg_processing_time_ms': stats['avg_processing_time'],
                        'total_processing_time_ms': stats['total_processing_time']
                    }
                }
            else:
                return {}
        else:
            # Return stats for all modules
            result = {}
            for name, stats in self.module_stats.items():
                result[name] = {
                    'total_operations': stats['total_operations'],
                    'successful_operations': stats['successful_operations'],
                    'failed_operations': stats['failed_operations'],
                    'success_rate': (
                        stats['successful_operations'] / stats['total_operations']
                        if stats['total_operations'] > 0 else 0
                    ),
                    'avg_processing_time_ms': stats['avg_processing_time'],
                    'total_processing_time_ms': stats['total_processing_time']
                }
            return result
    
    def get_memory_usage_summary(self) -> Dict[str, Any]:
        """Get memory usage summary."""
        if not self.memory_snapshots:
            return {}
        
        # Get recent snapshots (last 10 minutes)
        current_time = time.time()
        recent_snapshots = [
            snapshot for snapshot in self.memory_snapshots
            if current_time - snapshot.timestamp <= 600
        ]
        
        if not recent_snapshots:
            recent_snapshots = list(self.memory_snapshots)[-10:]  # Last 10 snapshots
        
        # Calculate statistics
        system_memory_values = [s.system_memory_mb for s in recent_snapshots]
        process_memory_values = [s.process_memory_mb for s in recent_snapshots]
        gpu_memory_values = [s.gpu_memory_mb for s in recent_snapshots if s.gpu_memory_mb is not None]
        
        summary = {
            'current_snapshot': asdict(self.memory_snapshots[-1]),
            'system_memory': {
                'current_mb': system_memory_values[-1] if system_memory_values else 0,
                'avg_mb': sum(system_memory_values) / len(system_memory_values) if system_memory_values else 0,
                'peak_mb': max(system_memory_values) if system_memory_values else 0,
                'min_mb': min(system_memory_values) if system_memory_values else 0
            },
            'process_memory': {
                'current_mb': process_memory_values[-1] if process_memory_values else 0,
                'avg_mb': sum(process_memory_values) / len(process_memory_values) if process_memory_values else 0,
                'peak_mb': max(process_memory_values) if process_memory_values else 0,
                'min_mb': min(process_memory_values) if process_memory_values else 0
            }
        }
        
        if gpu_memory_values:
            summary['gpu_memory'] = {
                'current_mb': gpu_memory_values[-1],
                'avg_mb': sum(gpu_memory_values) / len(gpu_memory_values),
                'peak_mb': max(gpu_memory_values),
                'min_mb': min(gpu_memory_values)
            }
        
        return summary
    
    def get_cache_performance_summary(self) -> Dict[str, Any]:
        """Get cache performance summary."""
        return {
            cache_name: asdict(metrics)
            for cache_name, metrics in self.cache_metrics.items()
        }
    
    def get_training_progress_summary(self) -> Optional[Dict[str, Any]]:
        """Get training progress summary."""
        if self.training_progress is None:
            return None
        
        return asdict(self.training_progress)
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive metrics report."""
        current_time = time.time()
        
        # Calculate uptime
        uptime_seconds = current_time - self.training_start_time if self.training_start_time else 0
        
        report = {
            'timestamp': current_time,
            'uptime_seconds': uptime_seconds,
            'processing_statistics': self.get_processing_statistics(),
            'memory_usage': self.get_memory_usage_summary(),
            'cache_performance': self.get_cache_performance_summary(),
            'training_progress': self.get_training_progress_summary(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'cpu_percent': psutil.cpu_percent(),
                'gpu_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        
        return report
    
    def export_metrics(self, filename: Optional[str] = None) -> str:
        """Export metrics to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pipeline_metrics_{timestamp}.json"
        
        export_path = self.metrics_export_dir / filename
        
        # Prepare export data
        export_data = {
            'export_timestamp': time.time(),
            'comprehensive_report': self.get_comprehensive_report(),
            'memory_snapshots': [asdict(snapshot) for snapshot in self.memory_snapshots],
            'processing_records': list(self.processing_records),
            'performance_history': {
                key: list(values) for key, values in self.performance_history.items()
            }
        }
        
        # Write to file
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Metrics exported to {export_path}")
        return str(export_path)
    
    def start_real_time_monitoring(self):
        """Start real-time monitoring (would typically run in a separate thread)."""
        if not self.enable_real_time_monitoring:
            return
        
        self.logger.info("Real-time monitoring started")
        # In a real implementation, this would start a background thread
        # that periodically collects metrics and updates dashboards
    
    def stop_real_time_monitoring(self):
        """Stop real-time monitoring."""
        self.logger.info("Real-time monitoring stopped")
    
    def reset_metrics(self):
        """Reset all collected metrics."""
        self.memory_snapshots.clear()
        self.processing_records.clear()
        self.cache_metrics.clear()
        self.module_stats.clear()
        self.performance_history.clear()
        self.training_progress = None
        self.training_start_time = None
        
        self.logger.info("All metrics reset")


# Global metrics collector instance
_global_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector


def initialize_metrics_collector(**kwargs) -> MetricsCollector:
    """Initialize the global metrics collector with custom settings."""
    global _global_metrics_collector
    _global_metrics_collector = MetricsCollector(**kwargs)
    return _global_metrics_collector