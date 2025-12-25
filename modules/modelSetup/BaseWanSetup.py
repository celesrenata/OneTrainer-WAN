from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.WanModel import WanModel, WanModelEmbedding
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupEmbeddingMixin import ModelSetupEmbeddingMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.util.checkpointing_util import (
    enable_checkpointing_for_clip_encoder_layers,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.conv_util import apply_circular_padding_to_conv2d
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BaseWanSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupEmbeddingMixin,
    metaclass=ABCMeta
):

    def setup_optimizations(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        if config.gradient_checkpointing.enabled():
            # Enable gradient checkpointing for WAN 2.2 transformer
            # TODO: Implement WAN-specific checkpointing when available
            pass
            
        if config.force_circular_padding:
            apply_circular_padding_to_conv2d(model.vae)
            apply_circular_padding_to_conv2d(model.transformer)
            if model.transformer_lora is not None:
                apply_circular_padding_to_conv2d(model.transformer_lora)

        model.autocast_context, model.train_dtype = create_autocast_context(self.train_device, config.train_dtype, [
            config.weight_dtypes().transformer,
            config.weight_dtypes().text_encoder,
            config.weight_dtypes().vae,
            config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
            config.weight_dtypes().embedding if config.train_any_embedding() else None,
        ], config.enable_autocast_cache)

        model.transformer_autocast_context, model.transformer_train_dtype = \
            disable_fp16_autocast_context(
                self.train_device,
                config.train_dtype,
                config.fallback_train_dtype,
                [
                    config.weight_dtypes().transformer,
                    config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
                    config.weight_dtypes().embedding if config.train_any_embedding() else None,
                ],
                config.enable_autocast_cache,
            )

        quantize_layers(model.text_encoder, self.train_device, model.train_dtype, config)
        quantize_layers(model.vae, self.train_device, model.train_dtype, config)
        quantize_layers(model.transformer, self.train_device, model.transformer_train_dtype, config)

        # Enable VAE tiling for memory efficiency with video data
        if hasattr(model.vae, 'enable_tiling'):
            model.vae.enable_tiling()

    def _setup_embeddings(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        additional_embeddings = []
        for embedding_config in config.all_embedding_configs():
            embedding_state = model.embedding_state_dicts.get(embedding_config.uuid, None)
            if embedding_state is None:
                with model.autocast_context:
                    embedding_state = self._create_new_embedding(
                        model,
                        embedding_config,
                        model.tokenizer,
                        model.text_encoder,
                        lambda text: model.encode_text(
                            text=text,
                            train_device=self.temp_device,
                        )[0],
                    )
            else:
                embedding_state = embedding_state.get("text_encoder_out", embedding_state.get("text_encoder", None))

            if embedding_state is not None:
                embedding_state = embedding_state.to(
                    dtype=model.text_encoder.get_input_embeddings().weight.dtype,
                    device=self.train_device,
                ).detach()

            embedding = WanModelEmbedding(
                embedding_config.uuid,
                embedding_state,
                embedding_config.placeholder,
                embedding_config.is_output_embedding,
            )
            if embedding_config.uuid == config.embedding.uuid:
                model.embedding = embedding
            else:
                additional_embeddings.append(embedding)

        model.additional_embeddings = additional_embeddings

        if model.tokenizer is not None:
            self._add_embeddings_to_tokenizer(model.tokenizer, model.all_text_encoder_embeddings())

    def _setup_embedding_wrapper(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        if model.tokenizer is not None and model.text_encoder is not None:
            model.embedding_wrapper = AdditionalEmbeddingWrapper(
                tokenizer=model.tokenizer,
                orig_module=model.text_encoder.text_model.embeddings.token_embedding,
                embeddings=model.all_text_encoder_embeddings(),
            )

        if model.embedding_wrapper is not None:
            model.embedding_wrapper.hook_to_module()

    def _setup_embeddings_requires_grad(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        if model.text_encoder is not None:
            for embedding, embedding_config in zip(model.all_text_encoder_embeddings(),
                                                   config.all_embedding_configs(), strict=True):
                train_embedding = \
                    embedding_config.train \
                    and config.text_encoder.train_embedding \
                    and not self.stop_embedding_training_elapsed(embedding_config, model.train_progress)
                embedding.requires_grad_(train_embedding)

    def predict(
            self,
            model: WanModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        with model.autocast_context:
            batch_seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)
            rand = Random(batch_seed)

            vae_scaling_factor = getattr(model.vae.config, 'scaling_factor', 0.18215)

            # Encode text prompt
            text_encoder_output = model.encode_text(
                train_device=self.train_device,
                batch_size=batch['latent_video'].shape[0],
                rand=rand,
                tokens=batch.get("tokens"),
                text_encoder_layer_skip=config.text_encoder_layer_skip,
                text_encoder_output=batch['text_encoder_hidden_state'] \
                    if 'text_encoder_hidden_state' in batch and not config.train_text_encoder_or_embedding() else None,
                text_encoder_dropout_probability=config.text_encoder.dropout_probability,
            )

            # Process video latents
            latent_video = batch['latent_video']
            scaled_latent_video = latent_video * vae_scaling_factor

            # Ensure video has correct dimensions (B, C, F, H, W)
            if scaled_latent_video.ndim == 4:
                scaled_latent_video = scaled_latent_video.unsqueeze(2)

            latent_noise = self._create_noise(scaled_latent_video, config, generator)

            timestep = self._get_timestep_discrete(
                model.noise_scheduler.config['num_train_timesteps'],
                deterministic,
                generator,
                scaled_latent_video.shape[0],
                config,
            )

            scaled_noisy_latent_video, sigma = self._add_noise_discrete(
                scaled_latent_video,
                latent_noise,
                timestep,
                model.noise_scheduler.timesteps,
            )

            latent_input = scaled_noisy_latent_video

            # Add guidance if supported
            if hasattr(model.transformer.config, 'guidance_embeds') and model.transformer.config.guidance_embeds:
                guidance = torch.tensor([config.transformer.guidance_scale * 1000.0], device=self.train_device)
                guidance = guidance.expand(latent_input.shape[0])
            else:
                guidance = None

            with model.transformer_autocast_context:
                predicted_flow = model.transformer(
                    hidden_states=latent_input.to(dtype=model.train_dtype.torch_dtype()),
                    timestep=timestep,
                    guidance=guidance.to(dtype=model.train_dtype.torch_dtype()) if guidance is not None else None,
                    encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                    return_dict=True
                ).sample

            flow = latent_noise - scaled_latent_video
            model_output_data = {
                'loss_type': 'target',
                'timestep': timestep,
                'predicted': predicted_flow,
                'target': flow,
            }

            if config.debug_mode:
                with torch.no_grad():
                    self._save_text(
                        self._decode_tokens(batch['tokens'], model.tokenizer),
                        config.debug_dir + "/training_batches",
                        "7-prompt",
                        train_progress.global_step,
                    )

                    # noise
                    self._save_image(
                        self._project_latent_to_image(latent_noise[:, :, 0]),  # Save first frame
                        config.debug_dir + "/training_batches",
                        "1-noise",
                        train_progress.global_step,
                    )

                    # noisy video
                    self._save_image(
                        self._project_latent_to_image(scaled_noisy_latent_video[:, :, 0]),
                        config.debug_dir + "/training_batches",
                        "2-noisy_video",
                        train_progress.global_step,
                    )

                    # predicted flow
                    self._save_image(
                        self._project_latent_to_image(predicted_flow[:, :, 0]),
                        config.debug_dir + "/training_batches",
                        "3-predicted_flow",
                        train_progress.global_step,
                    )

                    # flow
                    self._save_image(
                        self._project_latent_to_image(flow[:, :, 0]),
                        config.debug_dir + "/training_batches",
                        "4-flow",
                        train_progress.global_step,
                    )

                    predicted_scaled_latent_video = scaled_noisy_latent_video - predicted_flow * sigma

                    # predicted video
                    self._save_image(
                        self._project_latent_to_image(predicted_scaled_latent_video[:, :, 0]),
                        config.debug_dir + "/training_batches",
                        "5-predicted_video",
                        train_progress.global_step,
                    )

                    # video
                    self._save_image(
                        self._project_latent_to_image(scaled_latent_video[:, :, 0]),
                        config.debug_dir + "/training_batches",
                        "6-video",
                        model.train_progress.global_step,
                    )

        return model_output_data

    def calculate_loss(
            self,
            model: WanModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        return self._flow_matching_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            sigmas=model.noise_scheduler.sigmas,
        ).mean()

# WAN 2.2 Training Presets
PRESETS = {
    "WAN 2.2 LoRA 8GB": {
        "learning_rate": 1e-4,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_epochs": 10,
        "transformer": {
            "train": True,
            "learning_rate": 1e-4,
            "gradient_checkpointing": True,
        },
        "text_encoder": {
            "train": False,
        },
        "vae": {
            "train": False,
        },
        "lora": {
            "rank": 16,
            "alpha": 16,
        },
        "optimizer": {
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
        },
        "lr_scheduler": {
            "scheduler": "cosine",
            "warmup_steps": 100,
        },
    },
    "WAN 2.2 LoRA 16GB": {
        "learning_rate": 1e-4,
        "batch_size": 2,
        "gradient_accumulation_steps": 2,
        "max_epochs": 10,
        "transformer": {
            "train": True,
            "learning_rate": 1e-4,
            "gradient_checkpointing": True,
        },
        "text_encoder": {
            "train": False,
        },
        "vae": {
            "train": False,
        },
        "lora": {
            "rank": 32,
            "alpha": 32,
        },
        "optimizer": {
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
        },
        "lr_scheduler": {
            "scheduler": "cosine",
            "warmup_steps": 100,
        },
    },
    "WAN 2.2 Fine-tune 24GB": {
        "learning_rate": 5e-5,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_epochs": 5,
        "transformer": {
            "train": True,
            "learning_rate": 5e-5,
            "gradient_checkpointing": True,
        },
        "text_encoder": {
            "train": True,
            "learning_rate": 1e-5,
        },
        "vae": {
            "train": False,
        },
        "optimizer": {
            "optimizer": "AdamW",
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
        },
        "lr_scheduler": {
            "scheduler": "cosine",
            "warmup_steps": 200,
        },
    },
}