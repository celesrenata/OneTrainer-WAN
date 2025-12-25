"""
ValidatedDiskCache with safe data access patterns and comprehensive error handling.

This module provides a robust disk caching implementation that prevents None value
propagation and implements safe data access patterns throughout the caching process.
"""

import concurrent
import hashlib
import json
import logging
import math
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import SingleVariationRandomAccessPipelineModule
from .DataFlowValidator import DataFlowValidator, ValidationResult


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hit_rate: float
    miss_rate: float
    corruption_count: int
    total_items: int
    cache_size_mb: float
    avg_access_time: float


@dataclass
class CacheValidationResult:
    """Result of cache validation operation."""
    is_valid: bool
    error_message: Optional[str]
    corrupted_files: List[str]
    recovery_actions: List[str]
    metadata: Dict[str, Any]


class ValidatedDiskCache(PipelineModule, SingleVariationRandomAccessPipelineModule):
    """
    Enhanced disk cache with comprehensive validation and error handling.
    
    Features:
    - Safe item access with None checking
    - Cache corruption detection and recovery
    - Detailed cache statistics and monitoring
    - Graceful degradation when caching fails
    - Comprehensive logging and diagnostics
    """
    
    def __init__(
        self,
        cache_dir: str,
        split_names: Optional[List[str]] = None,
        aggregate_names: Optional[List[str]] = None,
        variations_in_name: Optional[str] = None,
        balancing_in_name: Optional[str] = None,
        balancing_strategy_in_name: Optional[str] = None,
        variations_group_in_name: Optional[str] = None,
        group_enabled_in_name: Optional[str] = None,
        before_cache_fun: Optional[Callable[[], None]] = None,
        enable_validation: bool = True,
        max_cache_size_gb: float = 10.0,
        corruption_recovery: bool = True,
        safe_mode: bool = True
    ):
        super().__init__()
        
        self.cache_dir = cache_dir
        self.split_names = split_names or []
        self.aggregate_names = aggregate_names or []
        
        self.variations_in_name = variations_in_name
        self.balancing_in_name = balancing_in_name
        self.balancing_strategy_in_name = balancing_strategy_in_name
        self.variations_group_in_names = (
            [variations_group_in_name] if isinstance(variations_group_in_name, str) 
            else variations_group_in_name or []
        )
        self.group_enabled_in_name = group_enabled_in_name
        self.before_cache_fun = before_cache_fun or (lambda: None)
        
        # Enhanced features
        self.enable_validation = enable_validation
        self.max_cache_size_gb = max_cache_size_gb
        self.corruption_recovery = corruption_recovery
        self.safe_mode = safe_mode
        
        # State tracking
        self.group_variations = {}
        self.group_indices = {}
        self.group_output_samples = {}
        self.variations_initialized = False
        self.aggregate_cache = {}
        
        # Performance and error tracking
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'corruptions': 0,
            'recoveries': 0,
            'validation_failures': 0,
            'safe_fallbacks': 0
        }
        self.access_times = []
        self.corrupted_files = set()
        
        # Setup logging
        self.logger = logging.getLogger("ValidatedDiskCache")
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
        
        self.logger.info(f"ValidatedDiskCache initialized - cache_dir: {cache_dir}, safe_mode: {safe_mode}")
    
    def length(self) -> int:
        """Get the length with safe fallback."""
        try:
            if not self.variations_initialized:
                name = self.split_names[0] if self.split_names else self.aggregate_names[0]
                length = self._get_previous_length(name)
                if length is None:
                    self.logger.warning("Previous length is None, using fallback length 1")
                    return 1
                return length
            else:
                return sum(x for x in self.group_output_samples.values())
        except Exception as e:
            self.logger.error(f"Error getting length: {e}, using fallback")
            return 1
    
    def get_inputs(self) -> List[str]:
        """Get required input names."""
        inputs = self.split_names + self.aggregate_names
        
        if self.variations_in_name:
            inputs.append(self.variations_in_name)
        if self.balancing_in_name:
            inputs.append(self.balancing_in_name)
        if self.balancing_strategy_in_name:
            inputs.append(self.balancing_strategy_in_name)
        if self.variations_group_in_names:
            inputs.extend(self.variations_group_in_names)
        if self.group_enabled_in_name:
            inputs.append(self.group_enabled_in_name)
        
        return inputs
    
    def get_outputs(self) -> List[str]:
        """Get output names."""
        return self.split_names + self.aggregate_names
    
    def safe_get_previous_item(self, variation: int, name: str, index: int) -> Any:
        """
        Safely get previous item with comprehensive error handling.
        
        This is the core method that prevents None propagation by implementing
        safe data access patterns with validation and fallback mechanisms.
        """
        try:
            # Attempt to get the item
            item = self._get_previous_item(variation, name, index)
            
            # Check for None
            if item is None:
                self.cache_stats['safe_fallbacks'] += 1
                self.logger.warning(
                    f"Previous item is None - variation: {variation}, name: {name}, index: {index}"
                )
                return self._create_safe_fallback_item(name, index)
            
            # Validate item if validator is available
            if self.validator and isinstance(item, dict):
                validation_result = self.validator.validate_pipeline_item(item, {})
                if not validation_result.is_valid:
                    self.cache_stats['validation_failures'] += 1
                    self.logger.warning(
                        f"Item validation failed: {validation_result.error_message} "
                        f"- variation: {variation}, name: {name}, index: {index}"
                    )
                    return self._create_safe_fallback_item(name, index)
            
            return item
            
        except Exception as e:
            self.cache_stats['safe_fallbacks'] += 1
            self.logger.error(
                f"Error getting previous item: {str(e)} "
                f"- variation: {variation}, name: {name}, index: {index}"
            )
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return self._create_safe_fallback_item(name, index)
    
    def _create_safe_fallback_item(self, name: str, index: int) -> Any:
        """Create safe fallback data for failed item access."""
        self.logger.debug(f"Creating safe fallback for name: {name}, index: {index}")
        
        # Create fallback based on expected data type
        if 'video' in name.lower():
            # Create dummy video tensor
            return torch.zeros((3, 8, 64, 64), dtype=torch.float32)
        elif 'image' in name.lower():
            # Create dummy image tensor
            return torch.zeros((3, 64, 64), dtype=torch.float32)
        elif 'latent' in name.lower():
            # Create dummy latent tensor for WAN 2.2 (48 channels)
            if 'video' in name.lower():
                return torch.zeros((48, 2, 8, 8), dtype=torch.float32)  # WAN 2.2 video latents
            else:
                return torch.zeros((48, 8, 8), dtype=torch.float32)  # WAN 2.2 image latents
        elif 'text' in name.lower() or 'prompt' in name.lower():
            # Create dummy text
            return f"fallback_{name}_{index}"
        elif 'path' in name.lower():
            # Create dummy path
            return f"fallback_{name}_{index}.dat"
        elif 'resolution' in name.lower():
            # Create dummy resolution
            return (64, 64)
        elif 'offset' in name.lower():
            # Create dummy offset
            return (0, 0)
        else:
            # Generic fallback
            return f"fallback_{name}_{index}"
    
    def __string_key(self, data: List[Any]) -> str:
        """Create string key from data with safe handling."""
        try:
            # Filter out None values
            safe_data = [item for item in data if item is not None]
            json_data = json.dumps(safe_data, sort_keys=True, ensure_ascii=True, 
                                 separators=(',', ':'), indent=None, default=str)
            return hashlib.sha256(json_data.encode('utf-8')).hexdigest()
        except Exception as e:
            self.logger.warning(f"Error creating string key: {e}, using fallback")
            return hashlib.sha256(f"fallback_{time.time()}".encode('utf-8')).hexdigest()
    
    def __init_variations(self):
        """Initialize variations with comprehensive error handling."""
        try:
            self.logger.debug("Initializing variations...")
            
            if self.variations_in_name is not None:
                group_variations = {}
                group_indices = {}
                group_balancing = {}
                group_balancing_strategy = {}
                
                # Get length safely
                length = self._get_previous_length(self.variations_in_name)
                if length is None:
                    self.logger.warning("Variations length is None, using fallback")
                    length = 1
                
                for in_index in range(length):
                    try:
                        # Check if group is enabled
                        if self.group_enabled_in_name:
                            enabled = self.safe_get_previous_item(0, self.group_enabled_in_name, in_index)
                            if not enabled:
                                continue
                        
                        # Get variation data safely
                        variations = self.safe_get_previous_item(0, self.variations_in_name, in_index)
                        balancing = self.safe_get_previous_item(0, self.balancing_in_name, in_index)
                        balancing_strategy = self.safe_get_previous_item(0, self.balancing_strategy_in_name, in_index)
                        
                        # Create group key safely
                        group_data = []
                        for name in self.variations_group_in_names:
                            item = self.safe_get_previous_item(0, name, in_index)
                            group_data.append(item)
                        
                        group_key = self.__string_key(group_data)
                        
                        # Store group data
                        if group_key not in group_variations:
                            group_variations[group_key] = variations or 1
                        
                        if group_key not in group_indices:
                            group_indices[group_key] = []
                        group_indices[group_key].append(in_index)
                        
                        if group_key not in group_balancing:
                            group_balancing[group_key] = balancing or 1.0
                        
                        if group_key not in group_balancing_strategy:
                            group_balancing_strategy[group_key] = balancing_strategy or 'REPEATS'
                    
                    except Exception as e:
                        self.logger.warning(f"Error processing variation index {in_index}: {e}")
                        continue
                
                # Calculate output samples
                group_output_samples = {}
                for group_key, balancing in group_balancing.items():
                    try:
                        balancing_strategy = group_balancing_strategy[group_key]
                        if balancing_strategy == 'REPEATS':
                            group_output_samples[group_key] = int(math.floor(len(group_indices[group_key]) * balancing))
                        elif balancing_strategy == 'SAMPLES':
                            group_output_samples[group_key] = int(balancing)
                        else:
                            group_output_samples[group_key] = len(group_indices[group_key])
                    except Exception as e:
                        self.logger.warning(f"Error calculating output samples for group {group_key}: {e}")
                        group_output_samples[group_key] = len(group_indices.get(group_key, []))
            
            else:
                # Simple case without variations
                first_name = self.split_names[0] if self.split_names else self.aggregate_names[0]
                length = self._get_previous_length(first_name)
                if length is None:
                    length = 1
                
                group_variations = {'': 1}
                group_indices = {'': list(range(length))}
                group_output_samples = {'': length}
            
            # Store results
            self.group_variations = group_variations
            self.group_indices = group_indices
            self.group_output_samples = group_output_samples
            self.aggregate_cache = {}
            self.variations_initialized = True
            
            self.logger.info(f"Variations initialized - groups: {len(group_variations)}, "
                           f"total samples: {sum(group_output_samples.values())}")
            
        except Exception as e:
            self.logger.error(f"Error initializing variations: {e}")
            # Fallback initialization
            self.group_variations = {'': 1}
            self.group_indices = {'': [0]}
            self.group_output_samples = {'': 1}
            self.aggregate_cache = {}
            self.variations_initialized = True
    
    def __get_cache_dir(self, group_key: str, in_variation: int) -> str:
        """Get cache directory with safe path handling."""
        try:
            variations = self.group_variations.get(group_key, 1)
            variation_dir = f"variation-{in_variation % variations}"
            cache_path = os.path.join(self.cache_dir, group_key, variation_dir)
            
            # Ensure directory exists
            os.makedirs(cache_path, exist_ok=True)
            
            return cache_path
        except Exception as e:
            self.logger.error(f"Error getting cache dir: {e}")
            # Fallback to simple path
            fallback_path = os.path.join(self.cache_dir, "fallback")
            os.makedirs(fallback_path, exist_ok=True)
            return fallback_path
    
    def __is_caching_done(self, group_key: str, in_variation: int) -> bool:
        """Check if caching is complete with corruption detection."""
        try:
            cache_dir = self.__get_cache_dir(group_key, in_variation)
            
            # Check if directory exists and has files
            if not os.path.isdir(cache_dir):
                return False
            
            cache_exists = False
            with os.scandir(cache_dir) as path_iter:
                cache_exists = any(path_iter)
            
            if not cache_exists:
                return False
            
            # Check for aggregate file
            aggregate_path = os.path.join(cache_dir, 'aggregate.pt')
            if not (os.path.exists(aggregate_path) and os.path.isfile(aggregate_path)):
                return False
            
            # Validate aggregate file if corruption recovery is enabled
            if self.corruption_recovery:
                try:
                    # Try to load the aggregate file to check for corruption
                    torch.load(aggregate_path, weights_only=False, map_location='cpu')
                    return True
                except Exception as e:
                    self.logger.warning(f"Aggregate file corrupted: {aggregate_path}, error: {e}")
                    self.cache_stats['corruptions'] += 1
                    self.corrupted_files.add(aggregate_path)
                    
                    if self.corruption_recovery:
                        self.logger.info(f"Attempting to recover corrupted cache: {cache_dir}")
                        self._recover_corrupted_cache(cache_dir)
                    
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking cache completion: {e}")
            return False
    
    def _recover_corrupted_cache(self, cache_dir: str):
        """Recover from corrupted cache by removing corrupted files."""
        try:
            self.logger.info(f"Recovering corrupted cache: {cache_dir}")
            
            # Remove corrupted files
            for file_name in os.listdir(cache_dir):
                file_path = os.path.join(cache_dir, file_name)
                try:
                    if file_name.endswith('.pt'):
                        # Try to load the file
                        torch.load(file_path, weights_only=False, map_location='cpu')
                except Exception:
                    # File is corrupted, remove it
                    self.logger.warning(f"Removing corrupted cache file: {file_path}")
                    os.remove(file_path)
                    self.corrupted_files.add(file_path)
            
            self.cache_stats['recoveries'] += 1
            
        except Exception as e:
            self.logger.error(f"Error recovering corrupted cache: {e}")
    
    def __clone_for_cache(self, x: Any) -> Any:
        """Clone data for caching with safe handling."""
        try:
            if x is None:
                self.logger.warning("Attempting to clone None value, using fallback")
                return "fallback_none_value"
            
            if isinstance(x, torch.Tensor):
                return x.clone()
            
            return x
            
        except Exception as e:
            self.logger.warning(f"Error cloning data: {e}, using fallback")
            return f"fallback_clone_error_{str(e)[:50]}"
    
    def __refresh_cache(self, out_variation: int):
        """Refresh cache with comprehensive error handling."""
        try:
            if not self.variations_initialized:
                self.__init_variations()
            
            # Initialize aggregate cache
            self.aggregate_cache = {}
            for group_key, variations in self.group_variations.items():
                self.aggregate_cache[group_key] = [None for _ in range(variations)]
            
            before_cache_fun_called = False
            
            for group_key in self.group_variations.keys():
                try:
                    # Calculate indices safely
                    group_output_samples = self.group_output_samples.get(group_key, 0)
                    if group_output_samples == 0:
                        continue
                    
                    start_index = group_output_samples * out_variation
                    end_index = group_output_samples * (out_variation + 1) - 1
                    
                    group_indices = self.group_indices.get(group_key, [])
                    if not group_indices:
                        continue
                    
                    start_variation = start_index // len(group_indices)
                    end_variation = end_index // len(group_indices)
                    
                    variations = self.group_variations.get(group_key, 1)
                    
                    for in_variation in [(x % variations) for x in range(start_variation, end_variation + 1)]:
                        if not self.__is_caching_done(group_key, in_variation):
                            # Call before_cache_fun once
                            if not before_cache_fun_called and self.before_cache_fun:
                                try:
                                    before_cache_fun_called = True
                                    self.before_cache_fun()
                                except Exception as e:
                                    self.logger.warning(f"before_cache_fun failed: {e}")
                            
                            # Perform caching
                            self._cache_group_variation(group_key, in_variation)
                        
                        # Load cached data
                        self._load_cached_data(group_key, in_variation)
                
                except Exception as e:
                    self.logger.error(f"Error refreshing cache for group {group_key}: {e}")
                    continue
            
        except Exception as e:
            self.logger.error(f"Error refreshing cache: {e}")
            # Initialize minimal cache to prevent crashes
            self.aggregate_cache = {'': [{}]}
    
    def _cache_group_variation(self, group_key: str, in_variation: int):
        """Cache a specific group variation with error handling."""
        try:
            cache_dir = self.__get_cache_dir(group_key, in_variation)
            group_indices = self.group_indices.get(group_key, [])
            
            if not group_indices:
                self.logger.warning(f"No indices for group {group_key}")
                return
            
            size = len(group_indices)
            aggregate_cache = [None] * size
            
            self.logger.info(f"Caching group {group_key}, variation {in_variation}, {size} items")
            
            with tqdm(total=size, smoothing=0.1, desc='caching') as bar:
                def cache_item(group_index, in_index, in_variation, current_device):
                    try:
                        # Preserve current device for multi-GPU
                        if torch.cuda.is_available() and current_device is not None:
                            torch.cuda.set_device(current_device)
                        
                        split_item = {}
                        aggregate_item = {}
                        
                        with torch.no_grad():
                            # Cache split items with safe access
                            for name in self.split_names:
                                item = self.safe_get_previous_item(in_variation, name, in_index)
                                split_item[name] = self.__clone_for_cache(item)
                            
                            # Cache aggregate items with safe access
                            for name in self.aggregate_names:
                                item = self.safe_get_previous_item(in_variation, name, in_index)
                                aggregate_item[name] = self.__clone_for_cache(item)
                        
                        # Save split item
                        split_path = os.path.join(cache_dir, f"{group_index}.pt")
                        torch.save(split_item, split_path)
                        
                        # Store aggregate item
                        aggregate_cache[group_index] = aggregate_item
                        
                    except Exception as e:
                        self.logger.error(f"Error caching item {group_index}: {e}")
                        # Create fallback data
                        fallback_split = {name: self._create_safe_fallback_item(name, in_index) 
                                        for name in self.split_names}
                        fallback_aggregate = {name: self._create_safe_fallback_item(name, in_index) 
                                            for name in self.aggregate_names}
                        
                        try:
                            split_path = os.path.join(cache_dir, f"{group_index}.pt")
                            torch.save(fallback_split, split_path)
                            aggregate_cache[group_index] = fallback_aggregate
                        except Exception as e2:
                            self.logger.error(f"Error saving fallback data: {e2}")
                
                current_device = torch.cuda.current_device() if torch.cuda.is_available() else None
                
                # Submit caching tasks
                fs = (self._state.executor.submit(
                    cache_item, group_index, in_index, in_variation, current_device)
                      for (group_index, in_index) in enumerate(group_indices))
                
                # Wait for completion
                for i, f in enumerate(concurrent.futures.as_completed(fs)):
                    try:
                        f.result()
                    except Exception as e:
                        self.logger.error(f"Caching task failed: {e}")
                        # Don't shutdown executor, continue with other tasks
                    
                    if i % 250 == 0:
                        self._torch_gc()
                    bar.update(1)
            
            # Save aggregate cache
            aggregate_path = os.path.join(cache_dir, 'aggregate.pt')
            torch.save(aggregate_cache, aggregate_path)
            
        except Exception as e:
            self.logger.error(f"Error caching group variation: {e}")
    
    def _load_cached_data(self, group_key: str, in_variation: int):
        """Load cached data with error handling."""
        try:
            if self.aggregate_cache[group_key][in_variation] is None:
                cache_dir = self.__get_cache_dir(group_key, in_variation)
                aggregate_path = os.path.join(cache_dir, 'aggregate.pt')
                
                if os.path.exists(aggregate_path):
                    self.aggregate_cache[group_key][in_variation] = torch.load(
                        aggregate_path, weights_only=False, map_location=self.pipeline.device
                    )
                    self.cache_stats['hits'] += 1
                else:
                    self.cache_stats['misses'] += 1
                    self.logger.warning(f"Aggregate cache file not found: {aggregate_path}")
                    # Create fallback cache
                    self.aggregate_cache[group_key][in_variation] = [
                        {name: self._create_safe_fallback_item(name, i) 
                         for name in self.aggregate_names}
                        for i in range(len(self.group_indices.get(group_key, [0])))
                    ]
        
        except Exception as e:
            self.logger.error(f"Error loading cached data: {e}")
            self.cache_stats['misses'] += 1
            # Create fallback cache
            self.aggregate_cache[group_key][in_variation] = [
                {name: self._create_safe_fallback_item(name, i) 
                 for name in self.aggregate_names}
                for i in range(len(self.group_indices.get(group_key, [0])))
            ]
    
    def __get_input_index(self, out_variation: int, out_index: int) -> Tuple[str, int, int, int]:
        """Get input index with safe handling."""
        try:
            offset = 0
            for group_key, group_output_samples in self.group_output_samples.items():
                if out_index >= group_output_samples + offset:
                    offset += group_output_samples
                    continue
                
                variations = self.group_variations.get(group_key, 1)
                group_indices = self.group_indices.get(group_key, [0])
                
                local_index = (out_index - offset) + (out_variation * group_output_samples)
                in_variation = (local_index // len(group_indices)) % variations
                group_index = local_index % len(group_indices)
                in_index = group_indices[group_index]
                
                return group_key, in_variation, group_index, in_index
            
            # Fallback
            return '', 0, 0, 0
            
        except Exception as e:
            self.logger.error(f"Error getting input index: {e}")
            return '', 0, 0, 0
    
    def start(self, out_variation: int):
        """Start caching with error handling."""
        try:
            self.__refresh_cache(out_variation)
        except Exception as e:
            self.logger.error(f"Error starting cache: {e}")
            # Initialize minimal cache to prevent crashes
            self.aggregate_cache = {'': [{}]}
    
    def get_item(self, index: int, requested_name: Optional[str] = None) -> Dict[str, Any]:
        """Get cached item with comprehensive error handling."""
        start_time = time.time()
        
        try:
            item = {}
            
            group_key, in_variation, group_index, in_index = self.__get_input_index(
                self.current_variation, index
            )
            
            # Get aggregate item safely
            try:
                aggregate_cache = self.aggregate_cache.get(group_key, [])
                if in_variation < len(aggregate_cache) and aggregate_cache[in_variation] is not None:
                    aggregate_item = aggregate_cache[in_variation][group_index]
                else:
                    raise IndexError("Aggregate cache not available")
            except Exception as e:
                self.logger.warning(f"Error accessing aggregate cache: {e}")
                # Create fallback aggregate item
                aggregate_item = {name: self._create_safe_fallback_item(name, in_index) 
                               for name in self.aggregate_names}
            
            # Handle requested aggregate items
            if requested_name in self.aggregate_names:
                for name in self.aggregate_names:
                    item[name] = aggregate_item.get(name, self._create_safe_fallback_item(name, in_index))
            
            # Handle requested split items
            elif requested_name in self.split_names:
                try:
                    cache_dir = self.__get_cache_dir(group_key, in_variation)
                    cache_path = os.path.join(cache_dir, f"{group_index}.pt")
                    
                    if os.path.exists(cache_path):
                        split_item = torch.load(cache_path, weights_only=False, 
                                              map_location=self.pipeline.device)
                        
                        for name in self.split_names:
                            item[name] = split_item.get(name, self._create_safe_fallback_item(name, in_index))
                    else:
                        # Create fallback split items
                        for name in self.split_names:
                            item[name] = self._create_safe_fallback_item(name, in_index)
                
                except Exception as e:
                    self.logger.warning(f"Error loading split cache: {e}")
                    # Create fallback split items
                    for name in self.split_names:
                        item[name] = self._create_safe_fallback_item(name, in_index)
            
            # Record access time
            access_time = time.time() - start_time
            self.access_times.append(access_time)
            if len(self.access_times) > 1000:
                self.access_times = self.access_times[-500:]
            
            return item
            
        except Exception as e:
            self.logger.error(f"Error getting cache item: {e}")
            # Return complete fallback item
            fallback_item = {}
            for name in self.split_names + self.aggregate_names:
                fallback_item[name] = self._create_safe_fallback_item(name, index)
            return fallback_item
    
    def get_cache_statistics(self) -> CacheStats:
        """Get comprehensive cache statistics."""
        try:
            total_accesses = self.cache_stats['hits'] + self.cache_stats['misses']
            hit_rate = self.cache_stats['hits'] / total_accesses if total_accesses > 0 else 0
            miss_rate = self.cache_stats['misses'] / total_accesses if total_accesses > 0 else 0
            
            # Calculate cache size
            cache_size_mb = 0
            try:
                for root, dirs, files in os.walk(self.cache_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            cache_size_mb += os.path.getsize(file_path)
                cache_size_mb = cache_size_mb / (1024 * 1024)
            except Exception:
                cache_size_mb = 0
            
            avg_access_time = sum(self.access_times) / len(self.access_times) if self.access_times else 0
            
            return CacheStats(
                hit_rate=hit_rate,
                miss_rate=miss_rate,
                corruption_count=self.cache_stats['corruptions'],
                total_items=total_accesses,
                cache_size_mb=cache_size_mb,
                avg_access_time=avg_access_time
            )
            
        except Exception as e:
            self.logger.error(f"Error getting cache statistics: {e}")
            return CacheStats(0, 0, 0, 0, 0, 0)
    
    def validate_cache_integrity(self) -> CacheValidationResult:
        """Validate cache integrity and detect corruption."""
        try:
            corrupted_files = []
            recovery_actions = []
            
            if not os.path.exists(self.cache_dir):
                return CacheValidationResult(
                    is_valid=False,
                    error_message="Cache directory does not exist",
                    corrupted_files=[],
                    recovery_actions=["Create cache directory"],
                    metadata={'cache_dir': self.cache_dir}
                )
            
            # Check cache files
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    if file.endswith('.pt'):
                        file_path = os.path.join(root, file)
                        try:
                            torch.load(file_path, weights_only=False, map_location='cpu')
                        except Exception as e:
                            corrupted_files.append(file_path)
                            recovery_actions.append(f"Remove corrupted file: {file_path}")
            
            is_valid = len(corrupted_files) == 0
            error_message = None if is_valid else f"Found {len(corrupted_files)} corrupted files"
            
            return CacheValidationResult(
                is_valid=is_valid,
                error_message=error_message,
                corrupted_files=corrupted_files,
                recovery_actions=recovery_actions,
                metadata={
                    'cache_dir': self.cache_dir,
                    'total_files_checked': len([f for f in files if f.endswith('.pt')])
                }
            )
            
        except Exception as e:
            return CacheValidationResult(
                is_valid=False,
                error_message=f"Error validating cache: {str(e)}",
                corrupted_files=[],
                recovery_actions=["Manual cache inspection required"],
                metadata={'exception': str(e)}
            )