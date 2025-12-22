import copy
import os
import traceback

from modules.model.WanModel import WanModel
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    DiffusionPipeline,
)
from transformers import PreTrainedTokenizer, PreTrainedModel


class WanModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def __load_internal(
            self,
            model: WanModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            include_text_encoder: bool,
            quantization: QuantizationConfig,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, transformer_model_name, vae_model_name,
                include_text_encoder, quantization,
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: WanModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            include_text_encoder: bool,
            quantization: QuantizationConfig,
    ):
        diffusers_sub = []
        transformers_sub = []

        if not transformer_model_name:
            diffusers_sub.append("transformer")
        if include_text_encoder:
            transformers_sub.append("text_encoder")
        if not vae_model_name:
            diffusers_sub.append("vae")

        self._prepare_sub_modules(
            base_model_name,
            diffusers_modules=diffusers_sub,
            transformers_modules=transformers_sub,
        )

        if include_text_encoder:
            # Load tokenizer - will need to be adapted for actual WAN 2.2 tokenizer
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name,
                    subfolder="tokenizer",
                )
            except Exception:
                # Fallback to a default tokenizer if WAN 2.2 specific one not available
                tokenizer = None
        else:
            tokenizer = None

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        if include_text_encoder:
            # Load text encoder - will need to be adapted for actual WAN 2.2 text encoder
            try:
                text_encoder = self._load_transformers_sub_module(
                    PreTrainedModel,  # Will be replaced with actual WAN 2.2 text encoder class
                    weight_dtypes.text_encoder,
                    weight_dtypes.train_dtype,
                    base_model_name,
                    "text_encoder",
                )
            except Exception:
                text_encoder = None
        else:
            text_encoder = None

        if vae_model_name:
            vae = self._load_diffusers_sub_module(
                AutoencoderKL,  # May need to be WAN 2.2 specific VAE
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                vae_model_name,
            )
        else:
            vae = self._load_diffusers_sub_module(
                AutoencoderKL,  # May need to be WAN 2.2 specific VAE
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                base_model_name,
                "vae",
            )

        if transformer_model_name:
            # Load transformer from single file - will need actual WAN 2.2 transformer class
            try:
                transformer = PreTrainedModel.from_single_file(
                    transformer_model_name,
                    config=base_model_name,
                    subfolder="transformer",
                    torch_dtype = torch.bfloat16 if weight_dtypes.transformer.torch_dtype() is None else weight_dtypes.transformer.torch_dtype(),
                )
                transformer = self._convert_diffusers_sub_module_to_dtype(
                    transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization
                )
            except Exception:
                transformer = None
        else:
            # Load transformer from diffusers format - will need actual WAN 2.2 transformer class
            try:
                transformer = self._load_diffusers_sub_module(
                    PreTrainedModel,  # Will be replaced with actual WAN 2.2 transformer class
                    weight_dtypes.transformer,
                    weight_dtypes.train_dtype,
                    base_model_name,
                    "transformer",
                    quantization,
                )
            except Exception:
                transformer = None

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.transformer = transformer

    def __load_safetensors(
            self,
            model: WanModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            include_text_encoder: bool,
            quantization: QuantizationConfig,
    ):
        # Load from single safetensors file - will need actual WAN 2.2 pipeline
        try:
            pipeline = DiffusionPipeline.from_single_file(
                pretrained_model_link_or_path=base_model_name,
                safety_checker=None,
            )
        except Exception:
            raise Exception("Could not load WAN 2.2 pipeline from safetensors")

        if vae_model_name:
            vae = self._load_diffusers_sub_module(
                AutoencoderKL,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                vae_model_name,
            )
        else:
            vae = self._convert_diffusers_sub_module_to_dtype(
                pipeline.vae, weight_dtypes.vae, weight_dtypes.train_dtype
            )

        if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None and include_text_encoder:
            text_encoder = self._convert_transformers_sub_module_to_dtype(
                pipeline.text_encoder, weight_dtypes.text_encoder, weight_dtypes.train_dtype
            )
            tokenizer = pipeline.tokenizer
        else:
            text_encoder = None
            tokenizer = None
            print("text encoder not loaded, continuing without it")

        if transformer_model_name:
            # Load transformer from single file
            try:
                transformer = PreTrainedModel.from_single_file(
                    transformer_model_name,
                    config=pipeline.config.transformer if hasattr(pipeline.config, 'transformer') else None,
                    torch_dtype = torch.bfloat16 if weight_dtypes.transformer.torch_dtype() is None else weight_dtypes.transformer.torch_dtype(),
                )
                transformer = self._convert_diffusers_sub_module_to_dtype(
                    transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization
                )
            except Exception:
                transformer = getattr(pipeline, 'transformer', None)
                if transformer:
                    transformer = self._convert_diffusers_sub_module_to_dtype(
                        transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
                    )
        else:
            transformer = getattr(pipeline, 'transformer', None)
            if transformer:
                transformer = self._convert_diffusers_sub_module_to_dtype(
                    transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
                )

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.noise_scheduler = pipeline.scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.transformer = transformer

    def __after_load(self, model: WanModel):
        if model.tokenizer is not None:
            model.orig_tokenizer = copy.deepcopy(model.tokenizer)

    def load(
            self,
            model: WanModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model,
                model_names.include_text_encoder, quantization,
            )
            self.__after_load(model)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model,
                model_names.include_text_encoder, quantization,
            )
            self.__after_load(model)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_safetensors(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model,
                model_names.include_text_encoder, quantization,
            )
            self.__after_load(model)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)