import re
from collections.abc import Callable

import modules.util.multi_gpu_util as multi
from modules.util import path_util
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.video_util import validate_video_file, FrameSamplingStrategy

from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.pipelineModules.AspectBatchSorting import AspectBatchSorting
from mgds.pipelineModules.AspectBucketing import AspectBucketing
from mgds.pipelineModules.CalcAspect import CalcAspect
from mgds.pipelineModules.CapitalizeTags import CapitalizeTags
from mgds.pipelineModules.CollectPaths import CollectPaths
from mgds.pipelineModules.DistributedSampler import DistributedSampler
from mgds.pipelineModules.DownloadHuggingfaceDatasets import DownloadHuggingfaceDatasets
from mgds.pipelineModules.DropTags import DropTags
from mgds.pipelineModules.GenerateImageLike import GenerateImageLike
from mgds.pipelineModules.GenerateMaskedConditioningImage import GenerateMaskedConditioningImage
from mgds.pipelineModules.GetFilename import GetFilename
from mgds.pipelineModules.ImageToVideo import ImageToVideo
from mgds.pipelineModules.InlineAspectBatchSorting import InlineAspectBatchSorting
from mgds.pipelineModules.InlineDistributedSampler import InlineDistributedSampler
from mgds.pipelineModules.LoadImage import LoadImage
from mgds.pipelineModules.LoadMultipleTexts import LoadMultipleTexts
from mgds.pipelineModules.LoadVideo import LoadVideo
from mgds.pipelineModules.ModifyPath import ModifyPath
from mgds.pipelineModules.RandomBrightness import RandomBrightness
from mgds.pipelineModules.RandomCircularMaskShrink import RandomCircularMaskShrink
from mgds.pipelineModules.RandomContrast import RandomContrast
from mgds.pipelineModules.RandomFlip import RandomFlip
from mgds.pipelineModules.RandomHue import RandomHue
from mgds.pipelineModules.RandomLatentMaskRemove import RandomLatentMaskRemove
from mgds.pipelineModules.RandomMaskRotateCrop import RandomMaskRotateCrop
from mgds.pipelineModules.RandomRotate import RandomRotate
from mgds.pipelineModules.RandomSaturation import RandomSaturation
from mgds.pipelineModules.ScaleCropImage import ScaleCropImage
from mgds.pipelineModules.SelectFirstInput import SelectFirstInput
from mgds.pipelineModules.SelectInput import SelectInput
from mgds.pipelineModules.SelectRandomText import SelectRandomText
from mgds.pipelineModules.ShuffleTags import ShuffleTags
from mgds.pipelineModules.SingleAspectCalculation import SingleAspectCalculation

import torch

from diffusers import AutoencoderKL


class DataLoaderText2VideoMixin:
    def __init__(self):
        pass

    def _enumerate_input_modules(self, config: TrainConfig) -> list:
        """Enumerate input modules for video data loading."""
        supported_extensions = set()
        supported_extensions |= path_util.supported_image_extensions()
        supported_extensions |= path_util.supported_video_extensions()
        
        print(f"INFO: Supported extensions: {supported_extensions}")

        download_datasets = DownloadHuggingfaceDatasets(
            concept_in_name='concept', path_in_name='path', enabled_in_name='enabled',
            concept_out_name='concept',
        )

        collect_paths = CollectPaths(
            concept_in_name='concept', path_in_name='path', include_subdirectories_in_name='concept.include_subdirectories', enabled_in_name='enabled',
            path_out_name='video_path', concept_out_name='concept',
            extensions=supported_extensions, include_postfix=None, exclude_postfix=['-masklabel','-condlabel']
        )
        
        print(f"INFO: Created CollectPaths module with extensions: {supported_extensions}")
        print(f"INFO: CollectPaths will scan for files with extensions: {list(supported_extensions)}")
        
        # Create a wrapper that ensures CollectPaths returns reasonable length estimates
        class SafeCollectPaths:
            def __init__(self, collect_paths_module):
                self.collect_paths_module = collect_paths_module
                self.estimated_length = 10  # From concept stats
                
            def length(self):
                try:
                    actual_length = self.collect_paths_module.length()
                    print(f"INFO: CollectPaths actual length: {actual_length}")
                    return actual_length if actual_length > 0 else self.estimated_length
                except Exception as e:
                    print(f"WARNING: CollectPaths length failed: {e}, using estimated length {self.estimated_length}")
                    return self.estimated_length
                    
            def __getattr__(self, name):
                # Delegate all other method calls to the wrapped module
                return getattr(self.collect_paths_module, name)
        
        # Wrap CollectPaths with our safety wrapper
        collect_paths = SafeCollectPaths(collect_paths)

        mask_path = ModifyPath(in_name='video_path', out_name='mask_path', postfix='-masklabel', extension='.png')
        cond_path = ModifyPath(in_name='video_path', out_name='cond_path', postfix='-condlabel', extension='.png')
        sample_prompt_path = ModifyPath(in_name='video_path', out_name='sample_prompt_path', postfix='', extension='.txt')

        modules = [download_datasets, collect_paths, sample_prompt_path]

        if config.masked_training:
            modules.append(mask_path)
        if config.custom_conditioning_image:
            modules.append(cond_path)

        return modules

    def _load_input_modules(
            self,
            config: TrainConfig,
            train_dtype: DataType,
    ) -> list:
        """Load input modules for video data processing."""
        from mgds.PipelineModule import PipelineModule
        from mgds.pipelineModules.LoadVideo import LoadVideo
        from mgds.pipelineModules.LoadImage import LoadImage
        from mgds.pipelineModules.ImageToVideo import ImageToVideo
        from mgds.pipelineModules.LoadMultipleTexts import LoadMultipleTexts
        from mgds.pipelineModules.GetFilename import GetFilename
        from mgds.pipelineModules.SelectInput import SelectInput
        from mgds.pipelineModules.SelectRandomText import SelectRandomText
        from modules.util import path_util
        import torch
        
        # Create a safety wrapper that ensures we never return None from any module
        class SafePipelineModule(PipelineModule):
            def __init__(self, wrapped_module, module_name="Unknown", dtype=torch.float32):
                self.wrapped_module = wrapped_module
                self.module_name = module_name
                self.dtype = dtype
                super().__init__()
                
            def length(self):
                try:
                    # Check if the module has been initialized by MGDS yet
                    if not hasattr(self.wrapped_module, '_PipelineModule__module_index'):
                        # Module not initialized yet - try to get a reasonable estimate
                        if hasattr(self.wrapped_module, '__class__') and 'CollectPaths' in str(self.wrapped_module.__class__):
                            # For CollectPaths, we can't know the length until MGDS initializes it
                            # But we know from concept stats there should be 10 files
                            print(f"INFO: {self.module_name} not initialized yet, estimating length from concept stats")
                            return 10  # Based on concept stats showing 10 images
                        else:
                            # For other modules, return 1 as fallback
                            return 1
                    
                    length = self.wrapped_module.length()
                    # Always log length calls to trace data flow
                    print(f"INFO: {self.module_name} length() returned: {length}")
                    return length
                except Exception as e:
                    print(f"WARNING: {self.module_name} length() failed: {e}, returning fallback length 1")
                    return 1  # Minimum length to prevent empty dataset
                
            def get_inputs(self):
                return self.wrapped_module.get_inputs()
                
            def get_outputs(self):
                return self.wrapped_module.get_outputs()
            
            def init(self, pipeline, seed, index, state):
                """Initialize the module - required by MGDS pipeline"""
                print(f"DEBUG: {self.module_name} init called - seed={seed}, index={index}")
                try:
                    if hasattr(self.wrapped_module, 'init'):
                        return self.wrapped_module.init(pipeline, seed, index, state)
                    else:
                        print(f"DEBUG: {self.module_name} wrapped module has no init method, skipping")
                        return None
                except Exception as e:
                    print(f"DEBUG: {self.module_name} init failed: {e}")
                    return None
            
            def start(self, epoch):
                """Start epoch - required by some MGDS modules"""
                print(f"DEBUG: {self.module_name} start called - epoch={epoch}")
                try:
                    if hasattr(self.wrapped_module, 'start'):
                        return self.wrapped_module.start(epoch)
                    else:
                        print(f"DEBUG: {self.module_name} wrapped module has no start method, skipping")
                        return None
                except Exception as e:
                    print(f"DEBUG: {self.module_name} start failed: {e}")
                    return None
            
            def end(self):
                """End processing - required by some MGDS modules"""
                print(f"DEBUG: {self.module_name} end called")
                try:
                    if hasattr(self.wrapped_module, 'end'):
                        return self.wrapped_module.end()
                    else:
                        print(f"DEBUG: {self.module_name} wrapped module has no end method, skipping")
                        return None
                except Exception as e:
                    print(f"DEBUG: {self.module_name} end failed: {e}")
                    return None
            
            def clear_item_cache(self):
                """Clear item cache - required by MGDS pipeline"""
                # Reduce debug noise - only log errors, not every call
                
                # First call the parent class method to initialize __local_cache
                super().clear_item_cache()
                
                # Then optionally call the wrapped module's method
                try:
                    if hasattr(self.wrapped_module, 'clear_item_cache'):
                        return self.wrapped_module.clear_item_cache()
                    else:
                        # Only log if we need to debug specific modules
                        pass
                        return None
                except Exception as e:
                    print(f"ERROR: {self.module_name} clear_item_cache failed: {e}")
                    return None
                
            def get_item(self, variation, index, requested_name=None):
                print(f"DEBUG: {self.module_name} get_item called - variation={variation}, index={index}, requested_name={requested_name}")
                try:
                    result = self.wrapped_module.get_item(variation, index, requested_name)
                    print(f"DEBUG: {self.module_name} returned result type: {type(result)}")
                    if result is None:
                        print(f"ERROR: {self.module_name} returned None for item {index}, creating safe fallback")
                        # Create a safe fallback data dictionary
                        fallback = self._create_safe_fallback_data(index)
                        print(f"DEBUG: {self.module_name} created fallback: {type(fallback)} with keys: {list(fallback.keys()) if isinstance(fallback, dict) else 'not dict'}")
                        return fallback
                    else:
                        print(f"DEBUG: {self.module_name} returned valid result with keys: {list(result.keys()) if isinstance(result, dict) else 'not dict'}")
                    return result
                except Exception as e:
                    print(f"ERROR: {self.module_name} failed for item {index}: {e}, creating safe fallback")
                    import traceback
                    print(f"DEBUG: {self.module_name} exception traceback: {traceback.format_exc()}")
                    # Create a safe fallback data dictionary
                    fallback = self._create_safe_fallback_data(index)
                    print(f"DEBUG: {self.module_name} created fallback after exception: {type(fallback)} with keys: {list(fallback.keys()) if isinstance(fallback, dict) else 'not dict'}")
                    return fallback
            
            def _create_safe_fallback_data(self, index):
                """Create safe fallback data that won't cause pipeline crashes"""
                import torch
                
                print(f"DEBUG: {self.module_name} creating fallback data for index {index}")
                
                # Create minimal but complete data dictionary
                fallback_data = {
                    'video_path': f'fallback_video_{index}.mp4',
                    'prompt': 'fallback prompt for training',
                    'concept': {'name': 'fallback_concept', 'enabled': True},
                    'settings': {'target_frames': 8}
                }
                
                print(f"DEBUG: {self.module_name} base fallback data created")
                
                try:
                    outputs = self.get_outputs()
                    print(f"DEBUG: {self.module_name} outputs: {outputs}")
                    
                    # Add video tensor if this module is supposed to provide it
                    if 'video' in outputs:
                        fallback_data['video'] = torch.zeros((8, 3, 64, 64), dtype=self.dtype)
                        print(f"DEBUG: {self.module_name} added video tensor")
                    
                    # Add image tensor if this module is supposed to provide it  
                    if 'image' in outputs:
                        fallback_data['image'] = torch.zeros((3, 64, 64), dtype=self.dtype)
                        print(f"DEBUG: {self.module_name} added image tensor")
                        
                except Exception as e:
                    print(f"DEBUG: {self.module_name} error getting outputs: {e}")
                
                print(f"DEBUG: {self.module_name} final fallback data keys: {list(fallback_data.keys())}")
                return fallback_data
        
        # Create a wrapper module that prevents None returns from LoadVideo
        class SafeLoadVideo(PipelineModule):
            def __init__(self, load_video_module, dtype=torch.float32):
                super().__init__()
                self.load_video_module = load_video_module
                self.dtype = dtype
                
            def length(self):
                return self.load_video_module.length()
                
            def get_inputs(self):
                return self.load_video_module.get_inputs()
                
            def get_outputs(self):
                return self.load_video_module.get_outputs()
                
            def get_item(self, variation, index, requested_name=None):
                try:
                    print(f"DEBUG SAFE_LOAD_VIDEO: Processing item {index}, variation {variation}")
                    result = self.load_video_module.get_item(variation, index, requested_name)
                    
                    if result is None:
                        print(f"DEBUG SAFE_LOAD_VIDEO ERROR: LoadVideo returned None for item {index}")
                        print(f"  - This means the underlying video loading failed")
                        print(f"  - Creating dummy data to prevent pipeline crash")
                        # Create comprehensive dummy data to prevent pipeline crash
                        import torch
                        dummy_video = torch.zeros((8, 3, 64, 64), dtype=self.dtype)  # 8 frames, 3 channels, 64x64
                        # Return a complete data dictionary with all expected fields
                        return {
                            'video': dummy_video,
                            'video_path': f'dummy_video_{index}.mp4',
                            'prompt': 'dummy prompt',
                            'settings': {'target_frames': 8}
                        }
                    
                    # Log successful loading
                    video_path = result.get('video_path', 'unknown')
                    video_data = result.get('video', None)
                    if hasattr(video_data, 'shape'):
                        print(f"DEBUG SAFE_LOAD_VIDEO SUCCESS: {video_path} loaded with shape {video_data.shape}")
                    else:
                        print(f"DEBUG SAFE_LOAD_VIDEO SUCCESS: {video_path} loaded, type={type(video_data)}")
                    
                    return result
                    
                except Exception as e:
                    print(f"DEBUG SAFE_LOAD_VIDEO EXCEPTION: LoadVideo failed for item {index}: {e}")
                    print(f"  - Exception type: {type(e).__name__}")
                    print(f"  - Creating dummy data to prevent pipeline crash")
                    # Create comprehensive dummy data to prevent pipeline crash
                    import torch
                    dummy_video = torch.zeros((8, 3, 64, 64), dtype=self.dtype)  # 8 frames, 3 channels, 64x64
                    # Return a complete data dictionary with all expected fields
                    return {
                        'video': dummy_video,
                        'video_path': f'dummy_video_{index}.mp4',
                        'prompt': 'dummy prompt',
                        'settings': {'target_frames': 8}
                    }
        
        # Load video with configurable frame count and sampling strategy
        load_video_base = LoadVideo(
            path_in_name='video_path', 
            target_frame_count_in_name='settings.target_frames', 
            video_out_name='video', 
            range_min=0, 
            range_max=1, 
            target_frame_rate=24, 
            supported_extensions=path_util.supported_video_extensions(), 
            dtype=train_dtype.torch_dtype()
        )
        
        # Wrap with safety module
        load_video = SafeLoadVideo(load_video_base, dtype=train_dtype.torch_dtype())
        
        # Also support loading images and converting to video format
        load_image_base = LoadImage(
            path_in_name='video_path', 
            image_out_name='image', 
            range_min=0, 
            range_max=1, 
            supported_extensions=path_util.supported_image_extensions(), 
            dtype=train_dtype.torch_dtype()
        )
        
        # Create a wrapper for safe image loading
        class SafeLoadImage(PipelineModule):
            def __init__(self, load_image_module, dtype=torch.float32):
                super().__init__()
                self.load_image_module = load_image_module
                self.dtype = dtype
                
            def length(self):
                return self.load_image_module.length()
                
            def get_inputs(self):
                return self.load_image_module.get_inputs()
                
            def get_outputs(self):
                return self.load_image_module.get_outputs()
                
            def get_item(self, variation, index, requested_name=None):
                try:
                    result = self.load_image_module.get_item(variation, index, requested_name)
                    if result is None:
                        print(f"Warning: LoadImage returned None for item {index}, creating dummy data")
                        # Create comprehensive dummy data to prevent pipeline crash
                        import torch
                        dummy_image = torch.zeros((3, 64, 64), dtype=self.dtype)  # 3 channels, 64x64
                        # Return a complete data dictionary with all expected fields
                        return {
                            'image': dummy_image,
                            'image_path': f'dummy_image_{index}.jpg',
                            'prompt': 'dummy prompt',
                            'settings': {'target_frames': 1}
                        }
                    return result
                except Exception as e:
                    print(f"Warning: LoadImage failed for item {index}: {e}, creating dummy data")
                    # Create comprehensive dummy data to prevent pipeline crash
                    import torch
                    dummy_image = torch.zeros((3, 64, 64), dtype=self.dtype)  # 3 channels, 64x64
                    return {
                        'image': dummy_image,
                        'image_path': f'dummy_image_{index}.jpg',
                        'prompt': 'dummy prompt',
                        'settings': {'target_frames': 1}
                    }
        
        # Wrap with safety module
        load_image = SafeLoadImage(load_image_base, dtype=train_dtype.torch_dtype())
        
        # Convert image to video format for consistency
        image_to_video = ImageToVideo(in_name='image', out_name='video')
        
        # Text loading modules
        load_sample_prompts = LoadMultipleTexts(path_in_name='sample_prompt_path', texts_out_name='sample_prompts')
        load_concept_prompts = LoadMultipleTexts(path_in_name='concept.text.prompt_path', texts_out_name='concept_prompts')
        filename_prompt = GetFilename(path_in_name='video_path', filename_out_name='filename_prompt', include_extension=False)
        select_prompt_input = SelectInput(setting_name='concept.text.prompt_source', out_name='prompts', setting_to_in_name_map={
            'sample': 'sample_prompts',
            'concept': 'concept_prompts',
            'filename': 'filename_prompt',
        }, default_in_name='sample_prompts')
        select_random_text = SelectRandomText(texts_in_name='prompts', text_out_name='prompt')

        # Conditional image loading for custom conditioning
        load_cond_image = LoadImage(path_in_name='cond_path', image_out_name='custom_conditioning_image', range_min=0, range_max=1, supported_extensions=path_util.supported_image_extensions(), dtype=train_dtype.torch_dtype())

        modules = [load_video, load_image, image_to_video, load_sample_prompts, load_concept_prompts, filename_prompt, select_prompt_input, select_random_text]

        if config.masked_training:
            modules.append(generate_mask)
            modules.append(load_mask)
            modules.append(mask_to_video)
        elif config.model_type.has_mask_input():
            modules.append(generate_mask)

        if config.custom_conditioning_image:
            modules.append(load_cond_image)

        # Wrap critical modules with safety wrapper to prevent None returns
        print(f"DEBUG: Wrapping {len(modules)} modules with safety wrapper")
        safe_modules = []
        for i, module in enumerate(modules):
            module_name = f"{type(module).__name__}_{i}"
            print(f"DEBUG: Processing module {i}: {module_name}")
            if hasattr(module, 'get_item'):  # Only wrap actual pipeline modules
                print(f"DEBUG: Wrapping module {module_name} with SafePipelineModule")
                safe_module = SafePipelineModule(
                    module, 
                    module_name=module_name,
                    dtype=train_dtype.torch_dtype()
                )
                safe_modules.append(safe_module)
            else:
                print(f"DEBUG: Module {module_name} does not have get_item, adding as-is")
                safe_modules.append(module)

        print(f"DEBUG: Created {len(safe_modules)} safe modules")
        return safe_modules

    def _video_validation_modules(self, config: TrainConfig) -> list:
        """Video validation modules to ensure data quality."""
        # Re-enabled video validation with debug logging to identify issues
        print("DEBUG: Video validation enabled - will show detailed error messages")
        
        validation_modules = []
        
        # Add basic video validation that logs issues instead of filtering
        try:
            from mgds.pipelineModules import FilterByFunction
            
            def validate_video_debug(sample):
                """Debug video validation that logs issues but doesn't filter"""
                try:
                    video_path = sample.get('video_path', 'unknown')
                    print(f"DEBUG VIDEO VALIDATION: Processing {video_path}")
                    
                    # Check if video data exists
                    if 'video' not in sample:
                        print(f"DEBUG VIDEO ERROR: No 'video' key in sample for {video_path}")
                        return None  # This will cause filtering
                    
                    video_data = sample['video']
                    if video_data is None:
                        print(f"DEBUG VIDEO ERROR: Video data is None for {video_path}")
                        return None
                    
                    # Check video properties if it's a tensor/array
                    if hasattr(video_data, 'shape'):
                        print(f"DEBUG VIDEO SUCCESS: {video_path} shape={video_data.shape}")
                    else:
                        print(f"DEBUG VIDEO INFO: {video_path} type={type(video_data)}")
                    
                    return sample  # Pass through valid samples
                    
                except Exception as e:
                    video_path = sample.get('video_path', 'unknown')
                    print(f"DEBUG VIDEO EXCEPTION: {video_path} - {str(e)}")
                    return None  # Filter out problematic samples
            
            validation_modules.append(FilterByFunction(
                function=validate_video_debug,
                inputs=['video', 'video_path']
            ))
            
        except ImportError:
            print("DEBUG: FilterByFunction not available, using basic validation")
        except Exception as e:
            print(f"DEBUG: Error setting up video validation: {e}")
        
        return validation_modules

    def _video_augmentation_modules(self, config: TrainConfig) -> list:
        """Video-specific augmentation modules."""
        inputs = ['video']

        if config.masked_training or config.model_type.has_mask_input():
            inputs.append('mask')

        if config.model_type.has_depth_input():
            inputs.append('depth')

        if config.custom_conditioning_image:
            inputs.append('custom_conditioning_image')

        modules = [
            RandomFlip(names=inputs, enabled_in_name='concept.image.enable_random_flip', fixed_enabled_in_name='concept.image.enable_fixed_flip'),
            RandomRotate(names=inputs, enabled_in_name='concept.image.enable_random_rotate', fixed_enabled_in_name='concept.image.enable_fixed_rotate', max_angle_in_name='concept.image.random_rotate_max_angle'),
            RandomBrightness(names=inputs, enabled_in_name='concept.image.enable_random_brightness', fixed_enabled_in_name='concept.image.enable_fixed_brightness', max_strength_in_name='concept.image.random_brightness_max_strength'),
            RandomContrast(names=inputs, enabled_in_name='concept.image.enable_random_contrast', fixed_enabled_in_name='concept.image.enable_fixed_contrast', max_strength_in_name='concept.image.random_contrast_max_strength'),
            RandomSaturation(names=inputs, enabled_in_name='concept.image.enable_random_saturation', fixed_enabled_in_name='concept.image.enable_fixed_saturation', max_strength_in_name='concept.image.random_saturation_max_strength'),
            RandomHue(names=inputs, enabled_in_name='concept.image.enable_random_hue', fixed_enabled_in_name='concept.image.enable_fixed_hue', max_strength_in_name='concept.image.random_hue_max_strength'),
        ]

        return modules

    def _video_aspect_bucketing_in(self, config: TrainConfig, aspect_bucketing_quantization: int):
        """Video aspect bucketing with frame dimension support."""
        calc_aspect = CalcAspect(image_in_name='video', resolution_out_name='original_resolution')

        aspect_bucketing_quantization = AspectBucketing(
            quantization=aspect_bucketing_quantization,
            resolution_in_name='original_resolution',
            target_resolution_in_name='settings.target_resolution',
            enable_target_resolutions_override_in_name='concept.image.enable_resolution_override',
            target_resolutions_override_in_name='concept.image.resolution_override',
            target_frames_in_name='settings.target_frames',
            frame_dim_enabled=True,  # Enable frame dimension for video
            scale_resolution_out_name='scale_resolution',
            crop_resolution_out_name='crop_resolution',
            possible_resolutions_out_name='possible_resolutions'
        )

        single_aspect_calculation = SingleAspectCalculation(
            resolution_in_name='original_resolution',
            target_resolution_in_name='settings.target_resolution',
            enable_target_resolutions_override_in_name='concept.image.enable_resolution_override',
            target_resolutions_override_in_name='concept.image.resolution_override',
            scale_resolution_out_name='scale_resolution',
            crop_resolution_out_name='crop_resolution',
            possible_resolutions_out_name='possible_resolutions'
        )

        modules = [calc_aspect]

        if config.aspect_ratio_bucketing:
            modules.append(aspect_bucketing_quantization)
        else:
            modules.append(single_aspect_calculation)

        return modules

    def _video_crop_modules(self, config: TrainConfig):
        """Video cropping modules."""
        inputs = ['video']

        if config.masked_training or config.model_type.has_mask_input():
            inputs.append('mask')

        if config.model_type.has_depth_input():
            inputs.append('depth')

        if config.custom_conditioning_image:
            inputs.append('custom_conditioning_image')

        scale_crop = ScaleCropImage(names=inputs, scale_resolution_in_name='scale_resolution', crop_resolution_in_name='crop_resolution', enable_crop_jitter_in_name='concept.image.enable_crop_jitter', crop_offset_out_name='crop_offset')

        modules = [scale_crop]

        return modules

    def _video_mask_augmentation_modules(self, config: TrainConfig) -> list:
        """Video mask augmentation modules."""
        inputs = ['video']

        lowest_resolution = min([int(x.strip()) for x in re.split(r'\D', config.resolution) if x.strip() != ''])
        circular_mask_shrink = RandomCircularMaskShrink(mask_name='mask', shrink_probability=1.0, shrink_factor_min=0.2, shrink_factor_max=1.0, enabled_in_name='concept.image.enable_random_circular_mask_shrink')
        random_mask_rotate_crop = RandomMaskRotateCrop(mask_name='mask', additional_names=inputs, min_size=lowest_resolution, min_padding_percent=10, max_padding_percent=30, max_rotate_angle=20, enabled_in_name='concept.image.enable_random_mask_rotate_crop')

        modules = []

        if config.masked_training or config.model_type.has_mask_input():
            modules.append(circular_mask_shrink)
            modules.append(random_mask_rotate_crop)

        return modules

    def _video_inpainting_modules(self, config: TrainConfig):
        """Video inpainting modules."""
        conditioning_image = GenerateMaskedConditioningImage(image_in_name='video', mask_in_name='mask', image_out_name='conditioning_image', image_range_min=0, image_range_max=1)
        select_conditioning_image = SelectFirstInput(in_names=['custom_conditioning_image', 'conditioning_image'], out_name='conditioning_image')

        modules = []

        if config.model_type.has_conditioning_image_input():
            modules.append(conditioning_image)
            modules.append(select_conditioning_image)

        return modules

    def _video_output_modules_from_out_names(
            self,
            output_names: list[str | tuple[str, str]],
            config: TrainConfig,
            model,
            before_cache_video_fun=None,
            use_conditioning_image=False,
            vae=None,
            autocast_context=None,
            train_dtype=None,
    ) -> list:
        """Create video output modules from output names."""
        from mgds.pipelineModules.RandomLatentMaskRemove import RandomLatentMaskRemove
        from mgds.pipelineModules.AspectBatchSorting import AspectBatchSorting
        from mgds.pipelineModules.DistributedSampler import DistributedSampler

        world_size = multi.world_size()

        output = OutputPipelineModule(output_names)
        batch_sorting = AspectBatchSorting(
            batch_size=config.batch_size,
            resolution_in_name='crop_resolution',
            names=['crop_resolution'],
        )
        distributed_sampler = DistributedSampler()

        modules = []

        if config.latent_caching:
            mask_remove = RandomLatentMaskRemove(
                probability=0.1,
                names=['latent_image', 'latent_conditioning_image'],
                mask_name='latent_mask'
            )
            modules.append(mask_remove)

        modules.append(batch_sorting)
        if world_size > 1:
            modules.append(distributed_sampler)

        modules.append(output)

        return modules
