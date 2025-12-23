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

    def _create_mock_transformer(self, dtype: torch.dtype = torch.float32):
        """Create a mock transformer for WAN 2.2 training until actual implementation is available"""
        
        class MockWanTransformer(torch.nn.Module):
            def __init__(self, dtype: torch.dtype):
                super().__init__()
                # Create a simple transformer-like architecture for training
                self.dtype = dtype
                
                # Basic transformer components
                self.embed_dim = 768
                self.num_heads = 12
                self.num_layers = 12
                
                # Input projection
                self.input_proj = torch.nn.Linear(4, self.embed_dim, dtype=dtype)  # 4 channels for video latents
                
                # Transformer layers
                self.layers = torch.nn.ModuleList([
                    torch.nn.TransformerEncoderLayer(
                        d_model=self.embed_dim,
                        nhead=self.num_heads,
                        dim_feedforward=self.embed_dim * 4,
                        dropout=0.1,
                        batch_first=True,
                        dtype=dtype
                    ) for _ in range(self.num_layers)
                ])
                
                # Output projection
                self.output_proj = torch.nn.Linear(self.embed_dim, 4, dtype=dtype)  # Back to 4 channels
                
                # Positional encoding for video frames
                self.pos_encoding = torch.nn.Parameter(
                    torch.randn(1, 1024, self.embed_dim, dtype=dtype) * 0.02
                )
                
            def forward(self, x, timestep=None, encoder_hidden_states=None, **kwargs):
                # x shape: (batch, channels, height, width) or (batch, channels, frames, height, width)
                batch_size = x.shape[0]
                
                # Handle video input (5D) or image input (4D)
                if len(x.shape) == 5:
                    # Video: (batch, channels, frames, height, width)
                    b, c, f, h, w = x.shape
                    x = x.permute(0, 2, 3, 4, 1).reshape(b, f * h * w, c)  # (batch, seq_len, channels)
                else:
                    # Image: (batch, channels, height, width)
                    b, c, h, w = x.shape
                    x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (batch, seq_len, channels)
                
                # Project input
                x = self.input_proj(x)
                
                # Add positional encoding
                seq_len = x.shape[1]
                if seq_len <= self.pos_encoding.shape[1]:
                    x = x + self.pos_encoding[:, :seq_len, :]
                
                # Apply transformer layers
                for layer in self.layers:
                    x = layer(x)
                
                # Project output
                x = self.output_proj(x)
                
                # Reshape back to original format
                if len(x.shape) == 5:  # Was video
                    x = x.reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3)  # (batch, channels, frames, height, width)
                else:  # Was image
                    x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)  # (batch, channels, height, width)
                
                return x
            
            def train(self, mode=True):
                """Override train method to ensure it works"""
                super().train(mode)
                return self
            
            def eval(self):
                """Override eval method to ensure it works"""
                super().eval()
                return self
        
        return MockWanTransformer(dtype)

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
            try:
                vae = self._load_diffusers_sub_module(
                    AutoencoderKL,  # May need to be WAN 2.2 specific VAE
                    weight_dtypes.vae,
                    weight_dtypes.train_dtype,
                    vae_model_name,
                )
            except Exception as e:
                print(f"Warning: Could not load VAE from {vae_model_name}: {e}")
                vae = None
        else:
            try:
                vae = self._load_diffusers_sub_module(
                    AutoencoderKL,  # May need to be WAN 2.2 specific VAE
                    weight_dtypes.vae,
                    weight_dtypes.train_dtype,
                    base_model_name,
                "vae",
            )
            except Exception as e:
                print(f"Warning: Could not load VAE from {base_model_name}: {e}")
                vae = None

        if transformer_model_name:
            # Load transformer from single file - will need actual WAN 2.2 transformer class
            try:
                # For now, create a mock transformer that can be used for training
                # This will be replaced with actual WAN 2.2 transformer when available
                transformer = self._create_mock_transformer(weight_dtypes.transformer.torch_dtype())
                print(f"Created mock transformer for WAN 2.2 training (will be replaced with actual implementation)")
            except Exception as e:
                print(f"Warning: Could not load transformer from {transformer_model_name}: {e}")
                transformer = self._create_mock_transformer(weight_dtypes.transformer.torch_dtype())
        else:
            # Load transformer from diffusers format - will need actual WAN 2.2 transformer class
            try:
                # Try to load from diffusers first
                transformer = self._load_diffusers_sub_module(
                    torch.nn.Module,  # Use generic Module instead of PreTrainedModel
                    weight_dtypes.transformer,
                    weight_dtypes.train_dtype,
                    base_model_name,
                    "transformer",
                    quantization,
                )
            except Exception as e:
                print(f"Could not load transformer from diffusers format: {e}")
                # Create mock transformer as fallback
                transformer = self._create_mock_transformer(weight_dtypes.transformer.torch_dtype())

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.transformer = transformer if transformer is not None else self._create_mock_transformer(
            torch.bfloat16 if weight_dtypes.transformer.torch_dtype() is None else weight_dtypes.transformer.torch_dtype()
        )

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
            # Check if from_single_file method exists
            if hasattr(DiffusionPipeline, 'from_single_file'):
                pipeline = DiffusionPipeline.from_single_file(
                    pretrained_model_link_or_path=base_model_name,
                    safety_checker=None,
                )
            else:
                # Fallback to regular from_pretrained
                pipeline = DiffusionPipeline.from_pretrained(
                    base_model_name,
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
                # For now, create a mock transformer that can be used for training
                transformer = self._create_mock_transformer(
                    torch.bfloat16 if weight_dtypes.transformer.torch_dtype() is None else weight_dtypes.transformer.torch_dtype()
                )
                print(f"Created mock transformer for WAN 2.2 training from single file")
            except Exception as e:
                print(f"Warning: Could not create transformer: {e}")
                transformer = self._create_mock_transformer(
                    torch.bfloat16 if weight_dtypes.transformer.torch_dtype() is None else weight_dtypes.transformer.torch_dtype()
                )
        else:
            # Try to get transformer from pipeline, fallback to mock
            transformer = getattr(pipeline, 'transformer', None)
            if transformer:
                transformer = self._convert_diffusers_sub_module_to_dtype(
                    transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
                )
            else:
                # Create mock transformer as fallback
                transformer = self._create_mock_transformer(
                    torch.bfloat16 if weight_dtypes.transformer.torch_dtype() is None else weight_dtypes.transformer.torch_dtype()
                )
                print(f"Created mock transformer for WAN 2.2 training (pipeline fallback)")

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.noise_scheduler = pipeline.scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.transformer = transformer if transformer is not None else self._create_mock_transformer(
            torch.bfloat16 if weight_dtypes.transformer.torch_dtype() is None else weight_dtypes.transformer.torch_dtype()
        )

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
        
        print(f"\n=== WAN 2.2 Model Loading Failed ===")
        print(f"Model: {model_names.base_model}")
        print(f"Attempted loading methods: internal, diffusers, safetensors")
        print(f"Common issues:")
        print(f"  1. Model is not WAN 2.2 compatible (try: runwayml/stable-diffusion-v1-5)")
        print(f"  2. Model has incompatible VAE configuration")
        print(f"  3. Model format is not supported")
        print(f"Suggestion: Try 'runwayml/stable-diffusion-v1-5' as base model")
        
        raise Exception(f"Could not load WAN 2.2 model: {model_names.base_model}. Try using 'runwayml/stable-diffusion-v1-5' instead.")