"""
Cleanup utilities for WAN 2.2 training workflows.
Provides functions to clean up temporary files and optimize memory usage.
"""
import os
import shutil
import gc
import tempfile
from pathlib import Path
from typing import List, Optional


def cleanup_training_artifacts(output_dir: str, keep_checkpoints: bool = True) -> int:
    """
    Clean up training artifacts and temporary files.
    
    Args:
        output_dir: Training output directory
        keep_checkpoints: Whether to keep checkpoint files
        
    Returns:
        Number of files cleaned up
    """
    cleaned_count = 0
    
    if not os.path.exists(output_dir):
        return cleaned_count
    
    # Files to clean up
    cleanup_patterns = [
        "*.tmp",
        "*.temp", 
        "debug_*",
        "sample_temp_*"
    ]
    
    if not keep_checkpoints:
        cleanup_patterns.extend([
            "checkpoint_*.pt",
            "optimizer_*.pt"
        ])
    
    for pattern in cleanup_patterns:
        import glob
        files = glob.glob(os.path.join(output_dir, pattern))
        for file_path in files:
            try:
                os.remove(file_path)
                cleaned_count += 1
            except Exception:
                pass
    
    return cleaned_count


def cleanup_cache_directories(cache_dirs: Optional[List[str]] = None) -> int:
    """
    Clean up cache directories.
    
    Args:
        cache_dirs: List of cache directories to clean
        
    Returns:
        Number of directories cleaned
    """
    if cache_dirs is None:
        cache_dirs = [
            ".cache",
            "cache", 
            "__pycache__",
            ".pytest_cache"
        ]
    
    cleaned_count = 0
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                cleaned_count += 1
            except Exception:
                pass
    
    return cleaned_count


def optimize_memory_usage():
    """Optimize memory usage by clearing caches and forcing garbage collection."""
    # Clear Python garbage collection
    gc.collect()
    
    # Clear PyTorch CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def create_temp_directory(prefix: str = "wan_training_") -> str:
    """
    Create a temporary directory with proper cleanup.
    
    Args:
        prefix: Prefix for temporary directory name
        
    Returns:
        Path to temporary directory
    """
    return tempfile.mkdtemp(prefix=prefix)


def safe_remove_file(file_path: str) -> bool:
    """
    Safely remove a file with error handling.
    
    Args:
        file_path: Path to file to remove
        
    Returns:
        True if file was removed successfully
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
    except Exception:
        pass
    return False


def safe_remove_directory(dir_path: str) -> bool:
    """
    Safely remove a directory with error handling.
    
    Args:
        dir_path: Path to directory to remove
        
    Returns:
        True if directory was removed successfully
    """
    try:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            return True
    except Exception:
        pass
    return False


def get_directory_size(dir_path: str) -> int:
    """
    Get the total size of a directory in bytes.
    
    Args:
        dir_path: Path to directory
        
    Returns:
        Size in bytes
    """
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(dir_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
    except Exception:
        pass
    return total_size


def cleanup_large_files(directory: str, max_size_mb: int = 100) -> List[str]:
    """
    Find and optionally remove large files.
    
    Args:
        directory: Directory to scan
        max_size_mb: Maximum file size in MB
        
    Returns:
        List of large files found
    """
    large_files = []
    max_size_bytes = max_size_mb * 1024 * 1024
    
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    if file_size > max_size_bytes:
                        large_files.append(file_path)
    except Exception:
        pass
    
    return large_files
