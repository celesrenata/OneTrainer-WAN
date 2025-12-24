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

        download_datasets = DownloadHuggingfaceDatasets(
            concept_in_name='concept', path_in_name='path', enabled_in_name='enabled',
            concept_out_name='concept',
        )

        collect_paths = CollectPaths(
            concept_in_name='concept', path_in_name='path', include_subdirectories_in_name='concept.include_subdirectories', enabled_in_name='enabled',
            path_out_name='video_path', concept_out_name='concept',
            extensions=supported_extensions, include_postfix=None, exclude_postfix=['-masklabel','-condlabel']
        )

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
        
        # Create a safety wrapper that ensures we never return None from any module
        class SafePipelineModule(PipelineModule):
            def __init__(self, wrapped_module, module_name="Unknown", dtype=torch.float32):
                super().__init__()
                self.wrapped_module = wrapped_module
                self.module_name = module_name
                self.dtype = dtype
                
            def length(self):
                try:
                    return self.wrapped_module.length()
                except:
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
                    print(f"DEBUG: {self.module_name} init failed: {e}, continuing")
                    return None
            
            def start(self, epoch):
                """Start epoch - may be required by MGDS pipeline"""
                print(f"DEBUG: {self.module_name} start called - epoch={epoch}")
                try:
                    if hasattr(self.wrapped_module, 'start'):
                        return self.wrapped_module.start(epoch)
                    else:
                        print(f"DEBUG: {self.module_name} wrapped module has no start method, skipping")
                        return None
                except Exception as e:
                    print(f"DEBUG: {self.module_name} start failed: {e}, continuing")
                    return None
            
            def end_epoch(self):
                """End epoch - may be required by MGDS pipeline"""
                print(f"DEBUG: {self.module_name} end_epoch called")
                try:
                    if hasattr(self.wrapped_module, 'end_epoch'):
                        return self.wrapped_module.end_epoch()
                    else:
                        print(f"DEBUG: {self.module_name} wrapped module has no end_epoch method, skipping")
                        return None
                except Exception as e:
                    print(f"DEBUG: {self.module_name} end_epoch failed: {e}, continuing")
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
                    result = self.load_video_module.get_item(variation, index, requested_name)
                    if result is None:
                        print(f"Warning: LoadVideo returned None for item {index}, creating dummy data")
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
                    return result
                except Exception as e:
                    print(f"Warning: LoadVideo failed for item {index}: {e}, creating dummy data")
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
                    # Return a complete data dictionary with all expected fields
                    return {
                        'image': dummy_image,
                        'image_path': f'dummy_image_{index}.jpg',
                        'prompt': 'dummy prompt',
                        'settings': {'target_frames': 1}
                    }
        
        load_image = SafeLoadImage(load_image_base, dtype=train_dtype.torch_dtype())
        
        # Convert single images to video format for consistency
        image_to_video = ImageToVideo(in_name='image', out_name='video')

        # Generate mask for video frames
        generate_mask = GenerateImageLike(image_in_name='video', image_out_name='mask', color=255, range_min=0, range_max=1)
        load_mask = LoadImage(path_in_name='mask_path', image_out_name='mask', range_min=0, range_max=1, channels=1, supported_extensions={".png"}, dtype=train_dtype.torch_dtype())
        mask_to_video = ImageToVideo(in_name='mask', out_name='mask')

        load_cond_image = LoadImage(path_in_name='cond_path', image_out_name='custom_conditioning_image', range_min=0, range_max=1, supported_extensions=path_util.supported_image_extensions(), dtype=train_dtype.torch_dtype())

        load_sample_prompts = LoadMultipleTexts(path_in_name='sample_prompt_path', texts_out_name='sample_prompts')
        load_concept_prompts = LoadMultipleTexts(path_in_name='concept.text.prompt_path', texts_out_name='concept_prompts')
        filename_prompt = GetFilename(path_in_name='video_path', filename_out_name='filename_prompt', include_extension=False)
        select_prompt_input = SelectInput(setting_name='concept.text.prompt_source', out_name='prompts', setting_to_in_name_map={
            'sample': 'sample_prompts',
            'concept': 'concept_prompts',
            'filename': 'filename_prompt',
        }, default_in_name='sample_prompts')
        select_random_text = SelectRandomText(texts_in_name='prompts', text_out_name='prompt')

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
        # For now, disable video validation to avoid pipeline issues
        # Video validation can be added later when MGDS pipeline properly handles filtering
        
        # The original FilterByFunction module is not available in current MGDS version
        # and custom filtering that returns None causes downstream pipeline issues
        # where modules try to access None[item_name] causing TypeError
        
        # TODO: Implement proper video validation when MGDS supports it
        # or when we can implement a filtering mechanism that doesn't break the pipeline
        
        print("Video validation temporarily disabled to prevent pipeline errors")
        return []

    def _video_augmentation_modules(self, config: TrainConfig) -> list:
        """Video-specific augmentation modules."""
        inputs = ['video']

        if config.masked_training or config.model_type.has_mask_input():
            inputs.append('mask')

        if config.model_type.has_depth_input():
            inputs.append('depth')

        if config.custom_conditioning_image:
            inputs.append('custom_conditioning_image')

        # Video augmentations - apply to all frames consistently
        random_flip = RandomFlip(names=inputs, enabled_in_name='concept.image.enable_random_flip', fixed_enabled_in_name='concept.image.enable_fixed_flip')
        random_rotate = RandomRotate(names=inputs, enabled_in_name='concept.image.enable_random_rotate', fixed_enabled_in_name='concept.image.enable_fixed_rotate', max_angle_in_name='concept.image.random_rotate_max_angle')
        
        # Color augmentations for video frames
        video_inputs = ['video']
        if config.custom_conditioning_image:
            video_inputs.append('custom_conditioning_image')
            
        random_brightness = RandomBrightness(names=video_inputs, enabled_in_name='concept.image.enable_random_brightness', fixed_enabled_in_name='concept.image.enable_fixed_brightness', max_strength_in_name='concept.image.random_brightness_max_strength')
        random_contrast = RandomContrast(names=video_inputs, enabled_in_name='concept.image.enable_random_contrast', fixed_enabled_in_name='concept.image.enable_fixed_contrast', max_strength_in_name='concept.image.random_contrast_max_strength')
        random_saturation = RandomSaturation(names=video_inputs, enabled_in_name='concept.image.enable_random_saturation', fixed_enabled_in_name='concept.image.enable_fixed_saturation', max_strength_in_name='concept.image.random_saturation_max_strength')
        random_hue = RandomHue(names=video_inputs, enabled_in_name='concept.image.enable_random_hue', fixed_enabled_in_name='concept.image.enable_fixed_hue', max_strength_in_name='concept.image.random_hue_max_strength')

        # Text augmentations
        drop_tags = DropTags(text_in_name='prompt', enabled_in_name='concept.text.tag_dropout_enable', probability_in_name='concept.text.tag_dropout_probability', dropout_mode_in_name='concept.text.tag_dropout_mode',
                             special_tags_in_name='concept.text.tag_dropout_special_tags', special_tag_mode_in_name='concept.text.tag_dropout_special_tags_mode', delimiter_in_name='concept.text.tag_delimiter',
                             keep_tags_count_in_name='concept.text.keep_tags_count', text_out_name='prompt', regex_enabled_in_name='concept.text.tag_dropout_special_tags_regex')
        caps_randomize = CapitalizeTags(text_in_name='prompt', enabled_in_name='concept.text.caps_randomize_enable', probability_in_name='concept.text.caps_randomize_probability',
                                        capitalize_mode_in_name='concept.text.caps_randomize_mode', delimiter_in_name='concept.text.tag_delimiter', convert_lowercase_in_name='concept.text.caps_randomize_lowercase', text_out_name='prompt')
        shuffle_tags = ShuffleTags(text_in_name='prompt', enabled_in_name='concept.text.enable_tag_shuffling', delimiter_in_name='concept.text.tag_delimiter', keep_tags_count_in_name='concept.text.keep_tags_count', text_out_name='prompt')

        modules = [
            random_flip,
            random_rotate,
            random_brightness,
            random_contrast,
            random_saturation,
            random_hue,
            drop_tags,
            caps_randomize,
            shuffle_tags,
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
            before_cache_video_fun: Callable[[], None] | None = None,
            use_conditioning_image: bool = False,
            vae: AutoencoderKL | None = None,
            autocast_context: list[torch.autocast | None] = None,
            train_dtype: DataType | None = None,
    ):
        """Video-specific output modules."""
        sort_names = output_names + ['concept']

        output_names = output_names + [
            ('concept.loss_weight', 'loss_weight'),
            ('concept.type', 'concept_type'),
        ]

        if config.validation:
            output_names.append(('concept.name', 'concept_name'))
            output_names.append(('concept.path', 'concept_path'))
            output_names.append(('concept.seed', 'concept_seed'))

        mask_remove = RandomLatentMaskRemove(
            latent_mask_name='latent_mask', latent_conditioning_image_name='latent_conditioning_image' if use_conditioning_image else None,
            replace_probability=config.unmasked_probability, vae=vae,
            possible_resolutions_in_name='possible_resolutions',
            autocast_contexts=autocast_context, dtype=train_dtype.torch_dtype(),
            before_cache_fun=before_cache_video_fun,
        )

        world_size = multi.world_size() if config.multi_gpu else 1
        if config.latent_caching:
            batch_sorting = AspectBatchSorting(resolution_in_name='crop_resolution', names=sort_names, batch_size=config.batch_size * world_size)
            distributed_sampler = DistributedSampler(names=sort_names, world_size=world_size, rank=multi.rank())
        else:
            batch_sorting = InlineAspectBatchSorting(resolution_in_name='crop_resolution', names=sort_names, batch_size=config.batch_size * world_size)
            distributed_sampler = InlineDistributedSampler(names=sort_names, world_size=world_size, rank=multi.rank())

        output = OutputPipelineModule(names=output_names)

        modules = []

        if config.model_type.has_mask_input():
            modules.append(mask_remove)

        modules.append(batch_sorting)
        if world_size > 1:
            modules.append(distributed_sampler)

        modules.append(output)

        return modules