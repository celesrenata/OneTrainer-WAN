import copy
import os

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.mixin.DataLoaderText2VideoMixin import DataLoaderText2VideoMixin
from modules.dataLoader.wan.TemporalConsistencyVAE import TemporalConsistencyVAE
from modules.dataLoader.wan.VideoFrameSampler import VideoFrameSampler
from modules.dataLoader.wan.WanVideoTextEncoder import WanVideoTextEncoder
from modules.model.WanModel import WanModel
from modules.util.config.TrainConfig import TrainConfig
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress
from modules.util.video_util import (
    validate_video_dataset, 
    preprocess_video_for_training,
    FrameSamplingStrategy,
    VideoValidationError
)

from mgds.MGDS import MGDS, TrainDataLoader
from mgds.pipelineModules.DecodeTokens import DecodeTokens
from mgds.pipelineModules.DecodeVAE import DecodeVAE
from mgds.pipelineModules.DiskCache import DiskCache
from mgds.pipelineModules.EncodeClipText import EncodeClipText
from mgds.pipelineModules.EncodeVAE import EncodeVAE
from mgds.pipelineModules.MapData import MapData
from mgds.pipelineModules.RescaleImageChannels import RescaleImageChannels
from mgds.pipelineModules.SampleVAEDistribution import SampleVAEDistribution
from mgds.pipelineModules.SaveImage import SaveImage
from mgds.pipelineModules.SaveText import SaveText
from mgds.pipelineModules.ScaleImage import ScaleImage
from mgds.pipelineModules.Tokenize import Tokenize
from mgds.pipelineModules.VariationSorting import VariationSorting

import torch


class WanBaseDataLoader(
    BaseDataLoader,
    DataLoaderText2VideoMixin,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            config: TrainConfig,
            model: WanModel,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        super().__init__(
            train_device,
            temp_device,
        )

        if is_validation:
            config = copy.copy(config)
            config.batch_size = 1
            config.multi_gpu = False

        # Validate video data constraints for WAN 2.2
        self._validate_video_config(config)

        self.__ds = self.create_dataset(
            config=config,
            model=model,
            train_progress=train_progress,
            is_validation=is_validation,
        )
        self.__dl = TrainDataLoader(self.__ds, config.batch_size)

    def get_data_set(self) -> MGDS:
        return self.__ds

    def get_data_loader(self) -> TrainDataLoader:
        return self.__dl

    def _validate_video_config(self, config: TrainConfig):
        """Validate video-specific configuration for WAN 2.2."""
        # Set default video constraints for WAN 2.2
        min_resolution = getattr(config, 'min_video_resolution', (256, 256))
        max_resolution = getattr(config, 'max_video_resolution', (1024, 1024))
        max_duration = getattr(config, 'max_video_duration', 10.0)  # 10 seconds max
        target_frames = getattr(config, 'target_frames', 16)
        
        # Validate that target_frames is reasonable
        if target_frames < 1:
            raise ValueError(f"target_frames must be at least 1, got {target_frames}")
        if target_frames > 64:
            print(f"Warning: target_frames={target_frames} is quite high, this may cause memory issues")
        
        # Store validation constraints for later use
        self._video_constraints = {
            'min_resolution': min_resolution,
            'max_resolution': max_resolution,
            'max_duration': max_duration,
            'target_frames': target_frames,
        }

    def _get_frame_sampling_strategy(self, config: TrainConfig) -> FrameSamplingStrategy:
        """Get frame sampling strategy from config."""
        strategy_name = getattr(config, 'frame_sample_strategy', 'uniform')
        
        try:
            return FrameSamplingStrategy(strategy_name.lower())
        except ValueError:
            print(f"Warning: Unknown frame sampling strategy '{strategy_name}', using 'uniform'")
            return FrameSamplingStrategy.UNIFORM

    def _preparation_modules(self, config: TrainConfig, model: WanModel):
        """Prepare video data for WAN 2.2 training."""
        # Sample video frames using the configured strategy
        video_frame_sampler = VideoFrameSampler(
            video_in_name='video',
            video_path_in_name='video_path',
            video_out_name='sampled_video',
            target_frames_in_name='settings.target_frames',
            sampling_strategy=self._get_frame_sampling_strategy(config),
            seed=getattr(config, 'seed', None)
        )
        
        # Rescale video frames from [0, 1] to [-1, 1] range
        rescale_video = RescaleImageChannels(
            image_in_name='sampled_video', 
            image_out_name='scaled_video', 
            in_range_min=0, 
            in_range_max=1, 
            out_range_min=-1, 
            out_range_max=1
        )
        
        # Encode video frames using VAE with temporal consistency
        temporal_consistency_weight = getattr(config, 'temporal_consistency_weight', 1.0)
        encode_video = TemporalConsistencyVAE(
            video_in_name='scaled_video',
            latent_out_name='latent_video',
            vae=model.vae,
            temporal_consistency_weight=temporal_consistency_weight,
            autocast_contexts=[model.autocast_context],
            dtype=model.train_dtype.torch_dtype()
        )
        
        # Downscale mask for latent space
        downscale_mask = ScaleImage(
            in_name='mask', 
            out_name='latent_mask', 
            factor=0.125
        )
        
        # Add embeddings to prompt for text encoder
        add_embeddings_to_prompt = MapData(
            in_name='prompt', 
            out_name='prompt_with_embeddings', 
            map_fn=model.add_text_encoder_embeddings_to_prompt
        )
        
        # Encode text using WAN-specific video text encoder
        encode_video_text = WanVideoTextEncoder(
            prompt_in_name='prompt_with_embeddings',
            tokens_out_name='tokens',
            hidden_state_out_name='text_encoder_hidden_state',
            pooled_out_name='text_encoder_pooled_state',
            tokenizer=model.tokenizer,
            text_encoder=model.text_encoder,
            max_token_length=77,
            autocast_contexts=[model.autocast_context],
            dtype=model.train_dtype.torch_dtype()
        )

        modules = [video_frame_sampler, rescale_video, encode_video]

        if model.tokenizer and model.text_encoder:
            modules.append(add_embeddings_to_prompt)
            
            if not config.train_text_encoder_or_embedding():
                modules.append(encode_video_text)

        if config.masked_training:
            modules.append(downscale_mask)

        return modules

    def _cache_modules(self, config: TrainConfig, model: WanModel):
        """Configure caching for video data."""
        video_split_names = ['latent_video', 'original_resolution', 'crop_offset']

        if config.masked_training or config.model_type.has_mask_input():
            video_split_names.append('latent_mask')

        if config.model_type.has_conditioning_image_input():
            video_split_names.append('latent_conditioning_image')

        video_aggregate_names = ['crop_resolution', 'video_path']

        text_split_names = []

        sort_names = video_aggregate_names + video_split_names + [
            'prompt_with_embeddings', 'tokens', 'text_encoder_hidden_state', 'text_encoder_pooled_state',
            'concept'
        ]

        if not config.train_text_encoder_or_embedding():
            text_split_names.extend(['tokens', 'text_encoder_hidden_state', 'text_encoder_pooled_state'])

        video_cache_dir = os.path.join(config.cache_dir, "video")
        text_cache_dir = os.path.join(config.cache_dir, "text")

        def before_cache_video_fun():
            model.to(self.temp_device)
            model.vae_to(self.train_device)
            model.eval()
            torch_gc()

        def before_cache_text_fun():
            model.to(self.temp_device)

            if not config.train_text_encoder_or_embedding():
                model.text_encoder_to(self.train_device)

            model.eval()
            torch_gc()

        video_disk_cache = DiskCache(
            cache_dir=video_cache_dir, 
            split_names=video_split_names, 
            aggregate_names=video_aggregate_names, 
            variations_in_name='concept.image_variations', 
            balancing_in_name='concept.balancing', 
            balancing_strategy_in_name='concept.balancing_strategy', 
            variations_group_in_name=['concept.path', 'concept.seed', 'concept.include_subdirectories', 'concept.image'], 
            group_enabled_in_name='concept.enabled', 
            before_cache_fun=before_cache_video_fun
        )

        text_disk_cache = DiskCache(
            cache_dir=text_cache_dir, 
            split_names=text_split_names, 
            aggregate_names=[], 
            variations_in_name='concept.text_variations', 
            balancing_in_name='concept.balancing', 
            balancing_strategy_in_name='concept.balancing_strategy', 
            variations_group_in_name=['concept.path', 'concept.seed', 'concept.include_subdirectories', 'concept.text'], 
            group_enabled_in_name='concept.enabled', 
            before_cache_fun=before_cache_text_fun
        )

        modules = []

        if config.latent_caching:
            modules.append(video_disk_cache)

        if config.latent_caching:
            sort_names = [x for x in sort_names if x not in video_aggregate_names]
            sort_names = [x for x in sort_names if x not in video_split_names]

            if not config.train_text_encoder_or_embedding():
                modules.append(text_disk_cache)
                sort_names = [x for x in sort_names if x not in text_split_names]

        if len(sort_names) > 0:
            variation_sorting = VariationSorting(
                names=sort_names, 
                balancing_in_name='concept.balancing', 
                balancing_strategy_in_name='concept.balancing_strategy', 
                variations_group_in_name=['concept.path', 'concept.seed', 'concept.include_subdirectories', 'concept.text'], 
                group_enabled_in_name='concept.enabled'
            )
            modules.append(variation_sorting)

        return modules

    def _output_modules(self, config: TrainConfig, model: WanModel):
        """Configure output modules for WAN 2.2 training."""
        output_names = [
            'video_path', 'latent_video',
            'prompt_with_embeddings',
            'tokens',
            'original_resolution', 'crop_resolution', 'crop_offset',
        ]

        if config.masked_training or config.model_type.has_mask_input():
            output_names.append('latent_mask')

        if config.model_type.has_conditioning_image_input():
            output_names.append('latent_conditioning_image')

        if not config.train_text_encoder_or_embedding():
            output_names.extend(['text_encoder_hidden_state', 'text_encoder_pooled_state'])

        def before_cache_video_fun():
            model.to(self.temp_device)
            model.vae_to(self.train_device)
            model.eval()
            torch_gc()

        return self._video_output_modules_from_out_names(
            output_names=output_names,
            config=config,
            before_cache_video_fun=before_cache_video_fun,
            use_conditioning_image=True,
            vae=model.vae,
            autocast_context=[model.autocast_context],
            train_dtype=model.train_dtype,
        )

    def _debug_modules(self, config: TrainConfig, model: WanModel):
        """Debug modules for video data visualization."""
        debug_dir = os.path.join(config.debug_dir, "dataloader")

        def before_save_fun():
            model.vae_to(self.train_device)

        decode_video = DecodeVAE(
            in_name='latent_video', 
            out_name='decoded_video', 
            vae=model.vae, 
            autocast_contexts=[model.autocast_context], 
            dtype=model.train_dtype.torch_dtype()
        )
        
        upscale_mask = ScaleImage(
            in_name='latent_mask', 
            out_name='decoded_mask', 
            factor=8
        )
        
        decode_prompt = DecodeTokens(
            in_name='tokens', 
            out_name='decoded_prompt', 
            tokenizer=model.tokenizer
        )

        # Save first frame of video for debugging
        save_video_frame = SaveImage(
            image_in_name='decoded_video', 
            original_path_in_name='video_path', 
            path=debug_dir, 
            in_range_min=-1, 
            in_range_max=1, 
            before_save_fun=before_save_fun
        )
        
        save_mask = SaveImage(
            image_in_name='decoded_mask', 
            original_path_in_name='video_path', 
            path=debug_dir, 
            in_range_min=0, 
            in_range_max=1, 
            before_save_fun=before_save_fun
        )
        
        save_prompt = SaveText(
            text_in_name='decoded_prompt', 
            original_path_in_name='video_path', 
            path=debug_dir, 
            before_save_fun=before_save_fun
        )

        modules = []

        modules.append(decode_video)
        modules.append(save_video_frame)

        if config.masked_training or config.model_type.has_mask_input():
            modules.append(upscale_mask)
            modules.append(save_mask)

        modules.append(decode_prompt)
        modules.append(save_prompt)

        return modules

    def create_dataset(
            self,
            config: TrainConfig,
            model: WanModel,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        """Create the complete video dataset pipeline."""
        enumerate_input = self._enumerate_input_modules(config)
        video_validation = self._video_validation_modules(config)
        load_input = self._load_input_modules(config, model.train_dtype)
        video_mask_augmentation = self._video_mask_augmentation_modules(config)
        video_aspect_bucketing_in = self._video_aspect_bucketing_in(config, 64)
        video_crop_modules = self._video_crop_modules(config)
        video_augmentation_modules = self._video_augmentation_modules(config)
        video_inpainting_modules = self._video_inpainting_modules(config)
        preparation_modules = self._preparation_modules(config, model)
        cache_modules = self._cache_modules(config, model)
        output_modules = self._output_modules(config, model)

        debug_modules = self._debug_modules(config, model)

        return self._create_mgds(
            config,
            [
                enumerate_input,
                video_validation,  # Add video validation early in pipeline
                load_input,
                video_mask_augmentation,
                video_aspect_bucketing_in,
                video_crop_modules,
                video_augmentation_modules,
                video_inpainting_modules,
                preparation_modules,
                cache_modules,
                output_modules,

                debug_modules if config.debug_mode else None,
            ],
            train_progress,
            is_validation
        )