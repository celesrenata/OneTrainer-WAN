"""
Performance monitoring utilities for WAN 2.2 training.
Provides memory usage tracking and performance optimization.
"""
import time
import psutil
import os
from typing import Dict, Any, Optional
from contextlib import contextmanager


class PerformanceMonitor:
    """Monitor performance metrics during training."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.metrics = {
            'start_time': self.start_time,
            'peak_memory_mb': 0,
            'gpu_memory_mb': 0,
            'step_times': []
        }
    
    def record_step_time(self, step_time: float):
        """Record time for a training step."""
        if 'step_times' not in self.metrics:
            self.metrics['step_times'] = []
        self.metrics['step_times'].append(step_time)
    
    def update_memory_usage(self):
        """Update memory usage metrics."""
        # CPU memory
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.metrics['peak_memory_mb'] = max(
            self.metrics.get('peak_memory_mb', 0), 
            memory_mb
        )
        
        # GPU memory (if available)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                self.metrics['gpu_memory_mb'] = max(
                    self.metrics.get('gpu_memory_mb', 0),
                    gpu_memory_mb
                )
        except ImportError:
            pass
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if self.start_time is None:
            return {}
        
        total_time = time.time() - self.start_time
        step_times = self.metrics.get('step_times', [])
        
        summary = {
            'total_time_seconds': total_time,
            'peak_memory_mb': self.metrics.get('peak_memory_mb', 0),
            'gpu_memory_mb': self.metrics.get('gpu_memory_mb', 0),
            'total_steps': len(step_times)
        }
        
        if step_times:
            summary.update({
                'avg_step_time': sum(step_times) / len(step_times),
                'min_step_time': min(step_times),
                'max_step_time': max(step_times)
            })
        
        return summary


@contextmanager
def monitor_memory():
    """Context manager to monitor memory usage."""
    import gc
    
    # Clear cache before monitoring
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            start_gpu_memory = torch.cuda.memory_allocated()
        else:
            start_gpu_memory = 0
    except ImportError:
        start_gpu_memory = 0
    
    process = psutil.Process(os.getpid())
    start_cpu_memory = process.memory_info().rss
    
    try:
        yield
    finally:
        # Measure final memory usage
        end_cpu_memory = process.memory_info().rss
        cpu_diff_mb = (end_cpu_memory - start_cpu_memory) / 1024 / 1024
        
        try:
            import torch
            if torch.cuda.is_available():
                end_gpu_memory = torch.cuda.memory_allocated()
                gpu_diff_mb = (end_gpu_memory - start_gpu_memory) / 1024 / 1024
            else:
                gpu_diff_mb = 0
        except ImportError:
            gpu_diff_mb = 0
        
        print(f"Memory usage - CPU: {cpu_diff_mb:.1f}MB, GPU: {gpu_diff_mb:.1f}MB")


def optimize_batch_size(model, device, max_memory_mb: int = 8000) -> int:
    """
    Automatically determine optimal batch size based on available memory.
    
    Args:
        model: Model to test
        device: Device to test on
        max_memory_mb: Maximum memory to use in MB
        
    Returns:
        Optimal batch size
    """
    try:
        import torch
    except ImportError:
        return 1
    
    batch_size = 1
    max_batch_size = 16
    
    model.eval()
    
    for test_batch_size in range(1, max_batch_size + 1):
        try:
            # Test with dummy data
            dummy_input = torch.randn(test_batch_size, 3, 16, 256, 256, device=device)
            
            with torch.no_grad():
                # Simulate forward pass
                if hasattr(model, 'vae') and model.vae is not None:
                    _ = model.vae.encode(dummy_input)
            
            # Check memory usage
            if torch.cuda.is_available() and device.type == 'cuda':
                memory_used_mb = torch.cuda.memory_allocated() / 1024 / 1024
                if memory_used_mb > max_memory_mb:
                    break
            
            batch_size = test_batch_size
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                break
            else:
                raise
        finally:
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return batch_size


def profile_training_step(func):
    """Decorator to profile training step performance."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            step_time = end_time - start_time
            print(f"Training step took {step_time:.3f} seconds")
    
    return wrapper
