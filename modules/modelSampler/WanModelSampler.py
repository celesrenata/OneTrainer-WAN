import copy
import inspect
from collections.abc import Callable

from modules.model.WanModel import WanModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.torch_util import torch_gc
from modules.util.video_util import (
    calculate_video_quality_metrics,
    save_video_with_quality_options,
    validate_generated_video,
    create_video_sampling_config
)

import torch

from PIL import Image
from tqdm import tqdm


class WanModelSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: WanModel,
            model_type: ModelType,
    ):
        super().__init__(train_device, temp_device)

        self.model = model
        self.model_type = model_type
        self.pipeline = model.create_pipeline()

    @torch.no_grad()
    def __sample_base(
            self,
            prompt: str,
            negative_prompt: str,
            height: int,
            width: int,
            num_frames: int,
            seed: int,
            random_seed: bool,
            diffusion_steps: int,
            cfg_scale: float,
            noise_scheduler: NoiseScheduler,
            text_encoder_layer_skip: int = 0,
            transformer_attention_mask: bool = False,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> ModelSamplerOutput:
        with self.model.autocast_context:
            generator = torch.Generator(device=self.train_device)
            if random_seed:
                generator.seed()
            else:
                generator.manual_seed(seed)

            noise_scheduler = copy.deepcopy(self.model.noise_scheduler)
            transformer = self.pipeline.transformer if hasattr(self.pipeline, 'transformer') else self.model.transformer
            vae = self.pipeline.vae if hasattr(self.pipeline, 'vae') else self.model.vae
            
            # WAN 2.2 specific parameters - these may need adjustment based on actual model
            vae_temporal_scale_factor = 4
            vae_spatial_scale_factor = 8
            num_latent_channels = 16

            # prepare prompt
            self.model.text_encoder_to(self.train_device)

            prompt_embedding = self.model.encode_text(
                text=prompt,
                train_device=self.train_device,
                text_encoder_layer_skip=text_encoder_layer_skip,
            )

            self.model.text_encoder_to(self.temp_device)
            torch_gc()

            # prepare latent video
            num_latent_frames = (num_frames - 1) // vae_temporal_scale_factor + 1
            latent_video = torch.randn(
                size=(
                    1,  # batch size
                    num_latent_channels,
                    num_latent_frames,
                    height // vae_spatial_scale_factor,
                    width // vae_spatial_scale_factor
                ),
                generator=generator,
                device=self.train_device,
                dtype=torch.float32,
            )

            # prepare timesteps
            noise_scheduler.set_timesteps(
                num_inference_steps=diffusion_steps,
                device=self.train_device,
            )
            timesteps = noise_scheduler.timesteps

            # denoising loop
            extra_step_kwargs = {}
            if "generator" in set(inspect.signature(noise_scheduler.step).parameters.keys()):
                extra_step_kwargs["generator"] = generator

            self.model.transformer_to(self.train_device)
            for i, timestep in enumerate(tqdm(timesteps, desc="sampling video")):
                latent_model_input = torch.cat([latent_video])
                expanded_timestep = timestep.expand(latent_model_input.shape[0])

                # handle guidance for WAN 2.2
                if hasattr(transformer.config, 'guidance_embeds') and transformer.config.guidance_embeds:
                    guidance = torch.tensor([cfg_scale], device=self.train_device)
                    guidance = guidance.expand(latent_model_input.shape[0])
                else:
                    guidance = None

                # pack latents for transformer processing
                packed_latents = self.model.pack_latents(latent_model_input)

                with self.model.transformer_autocast_context:
                    # predict the noise residual
                    # Note: This will need to be adapted based on actual WAN 2.2 transformer interface
                    noise_pred = transformer(
                        hidden_states=packed_latents.to(dtype=self.model.transformer_train_dtype.torch_dtype()),
                        timestep=expanded_timestep,
                        encoder_hidden_states=prompt_embedding.to(dtype=self.model.transformer_train_dtype.torch_dtype()),
                        guidance=guidance.to(dtype=self.model.transformer_train_dtype.torch_dtype()) if guidance is not None else None,
                        return_dict=True
                    ).sample

                # unpack latents from transformer output
                noise_pred = self.model.unpack_latents(
                    noise_pred, 
                    num_latent_frames, 
                    height // vae_spatial_scale_factor, 
                    width // vae_spatial_scale_factor
                )

                # compute the previous noisy sample x_t -> x_t-1
                latent_video = noise_scheduler.step(
                    noise_pred, timestep, latent_video, return_dict=False, **extra_step_kwargs
                )[0]

                on_update_progress(i + 1, len(timesteps))

            self.model.transformer_to(self.temp_device)
            torch_gc()

            # decode video
            self.model.vae_to(self.train_device)

            # Decode video latents to frames
            if hasattr(vae.config, 'scaling_factor'):
                latents = latent_video / vae.config.scaling_factor
            else:
                latents = latent_video

            video = vae.decode(latents, return_dict=False)[0]

            # Process video output
            if hasattr(self.pipeline, 'video_processor'):
                video = self.pipeline.video_processor.postprocess(video, output_type='pt')
            else:
                # Fallback processing
                video = video.clamp(0, 1)

            self.model.vae_to(self.temp_device)
            torch_gc()

            # Determine if output is single image or video
            is_single_frame = video.shape[2] == 1 or num_frames == 1
            if is_single_frame:
                # Convert to image
                frame = video[:, :, 0] if video.shape[2] > 0 else video.squeeze(2)
                frame = frame.cpu().permute(0, 2, 3, 1).float().numpy()
                frame = (frame * 255).round().astype("uint8")
                image = Image.fromarray(frame[0])

                return ModelSamplerOutput(
                    file_type=FileType.IMAGE,
                    data=image,
                )
            else:
                # Convert to video tensor
                video = video.cpu().permute(0, 2, 3, 4, 1).float()
                video = (video.clamp(0, 1) * 255).round().to(dtype=torch.uint8)
                video = video[0]

                return ModelSamplerOutput(
                    file_type=FileType.VIDEO,
                    data=video,
                )

    def sample(
            self,
            sample_config: SampleConfig,
            destination: str,
            image_format: ImageFormat | None = None,
            video_format: VideoFormat | None = None,
            audio_format: AudioFormat | None = None,
            on_sample: Callable[[ModelSamplerOutput], None] = lambda _: None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        # Create video sampling configuration
        video_config = create_video_sampling_config(
            frames=max(1, sample_config.frames),
            fps=24.0,
            resolution=(
                self.quantize_resolution(sample_config.width, 64),
                self.quantize_resolution(sample_config.height, 64)
            ),
            quality_preset="medium",
            format_preference="mp4"
        )
        
        sampler_output = self.__sample_base(
            prompt=sample_config.prompt,
            negative_prompt=sample_config.negative_prompt,
            height=video_config['resolution'][1],
            width=video_config['resolution'][0],
            num_frames=video_config['frames'],
            seed=sample_config.seed,
            random_seed=sample_config.random_seed,
            diffusion_steps=sample_config.diffusion_steps,
            cfg_scale=sample_config.cfg_scale,
            noise_scheduler=sample_config.noise_scheduler,
            text_encoder_layer_skip=sample_config.text_encoder_1_layer_skip,
            transformer_attention_mask=sample_config.transformer_attention_mask,
            on_update_progress=on_update_progress,
        )

        # Calculate quality metrics for video outputs
        if sampler_output.file_type == FileType.VIDEO:
            try:
                quality_metrics = calculate_video_quality_metrics(sampler_output.data)
                print(f"Video quality metrics: {quality_metrics}")
            except Exception as e:
                print(f"Could not calculate quality metrics: {e}")

        # Save output with enhanced video handling
        if sampler_output.file_type == FileType.VIDEO and video_format and video_format.is_video_format():
            # Use enhanced video saving with quality options
            success = save_video_with_quality_options(
                sampler_output.data,
                destination + video_format.extension(),
                fps=video_config['fps'],
                quality_preset=video_config['quality_preset'],
                codec_options=video_format.codec_options()
            )
            
            if success:
                # Validate the saved video
                is_valid, issues = validate_generated_video(
                    destination + video_format.extension(),
                    expected_frames=video_config['frames'],
                    expected_resolution=video_config['resolution']
                )
                
                if not is_valid:
                    print(f"Video validation issues: {issues}")
            else:
                print("Failed to save video with enhanced options, falling back to default")
                self.save_sampler_output(
                    sampler_output, destination,
                    image_format, video_format, audio_format,
                )
        else:
            # Use default saving method
            self.save_sampler_output(
                sampler_output, destination,
                image_format, video_format, audio_format,
            )

        on_sample(sampler_output)