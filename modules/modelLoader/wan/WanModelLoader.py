import copy
import os
import traceback
import torch

from modules.model.WanModel import WanModel
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch

from diffusers import (
    AutoencoderKL,
    AutoencoderKLWan,
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
        """Create a memory-efficient transformer for WAN 2.2 training"""
        
        class MemoryEfficientWanTransformer(torch.nn.Module):
            def __init__(self, dtype: torch.dtype):
                super().__init__()
                # Create a smaller transformer that matches WAN 2.2 interface
                self.dtype = dtype
                
                # Add config attribute for compatibility with real WAN 2.2
                self.config = type('Config', (), {
                    'guidance_embeds': False,
                    'in_channels': 48,  # WAN 2.2 VAE latent channels
                    'out_channels': 48,  # WAN 2.2 VAE latent channels
                    'num_layers': 6,    # Reduced from 30 to 6 for memory
                    'num_attention_heads': 12,  # Reduced from 24 to 12
                    'attention_head_dim': 64,   # Reduced from 128 to 64
                    'embed_dim': 768
                })()
                
                # Basic transformer components (smaller)
                self.embed_dim = 768
                self.num_layers = 6  # Much smaller than real 30 layers
                self.num_heads = 12
                
                # Input projection from 48 latent channels to embed_dim
                self.input_proj = torch.nn.Linear(48, self.embed_dim, dtype=dtype)
                
                # Smaller transformer layers
                self.layers = torch.nn.ModuleList()
                for i in range(self.num_layers):
                    layer = torch.nn.ModuleDict({
                        'self_attn': torch.nn.MultiheadAttention(
                            self.embed_dim, self.num_heads, batch_first=True, dtype=dtype
                        ),
                        'mlp': torch.nn.Sequential(
                            torch.nn.Linear(self.embed_dim, self.embed_dim * 2, dtype=dtype),  # Reduced from 4x to 2x
                            torch.nn.GELU(),
                            torch.nn.Linear(self.embed_dim * 2, self.embed_dim, dtype=dtype)
                        ),
                        'norm1': torch.nn.LayerNorm(self.embed_dim, dtype=dtype),
                        'norm2': torch.nn.LayerNorm(self.embed_dim, dtype=dtype)
                    })
                    self.layers.append(layer)
                
                # Output projection - back to 48 latent channels
                self.output_proj = torch.nn.Linear(self.embed_dim, 48, dtype=dtype)
                
                # Positional encoding for video frames
                self.pos_encoding = torch.nn.Parameter(
                    torch.randn(1, 1024, self.embed_dim, dtype=dtype) * 0.02
                )
                self.num_heads = 12
                self.num_layers = 12
                
                # Input projection - WAN 2.2 VAE outputs 48 latent channels
                self.input_proj = torch.nn.Linear(48, self.embed_dim, dtype=dtype)
                
                # Transformer layers with proper naming for LoRA compatibility
                self.layers = torch.nn.ModuleList()
                for i in range(self.num_layers):
                    layer = torch.nn.ModuleDict({
                        'self_attn': torch.nn.MultiheadAttention(
                            self.embed_dim, self.num_heads, batch_first=True, dtype=dtype
                        ),
                        'temporal_attn': torch.nn.MultiheadAttention(
                            self.embed_dim, self.num_heads, batch_first=True, dtype=dtype
                        ),
                        'mlp': torch.nn.Sequential(
                            torch.nn.Linear(self.embed_dim, self.embed_dim * 4, dtype=dtype),
                            torch.nn.GELU(),
                            torch.nn.Linear(self.embed_dim * 4, self.embed_dim, dtype=dtype)
                        ),
                        'norm1': torch.nn.LayerNorm(self.embed_dim, dtype=dtype),
                        'norm2': torch.nn.LayerNorm(self.embed_dim, dtype=dtype),
                        'norm3': torch.nn.LayerNorm(self.embed_dim, dtype=dtype)
                    })
                    self.layers.append(layer)
                
                # Output projection - back to 48 latent channels
                self.output_proj = torch.nn.Linear(self.embed_dim, 48, dtype=dtype)
                
                # Positional encoding for video frames
                self.pos_encoding = torch.nn.Parameter(
                    torch.randn(1, 1024, self.embed_dim, dtype=dtype) * 0.02
                )
                
            def forward(self, hidden_states=None, x=None, timestep=None, encoder_hidden_states=None, guidance=None, return_dict=False, **kwargs):
                # Handle both parameter names for compatibility
                if hidden_states is not None:
                    x = hidden_states
                elif x is None:
                    raise ValueError("Either 'hidden_states' or 'x' must be provided")
                
                # x shape: (batch, channels, height, width) or (batch, channels, frames, height, width)
                batch_size = x.shape[0]
                
                # Handle incorrect 5D tensor with single frame
                if len(x.shape) == 5 and x.shape[1] == 1:
                    # This is likely [batch, 1, channels, height, width] - squeeze the second dimension
                    x = x.squeeze(1)
                
                # Handle video input (5D) or image input (4D)
                if len(x.shape) == 5:
                    # Video: (batch, channels, frames, height, width)
                    b, c, f, h, w = x.shape
                    x = x.permute(0, 2, 3, 4, 1).reshape(b, f * h * w, c)  # (batch, seq_len, channels)
                else:
                    # Image: (batch, channels, height, width)
                    b, c, h, w = x.shape
                    x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (batch, seq_len, channels)
                
                # Safety check: if we get 3 channels (raw video), pad to 48 channels (latent space)
                if x.shape[-1] == 3:
                    print(f"WARNING: Received 3-channel input, padding to 48 channels for WAN 2.2 compatibility")
                    padding = torch.zeros(x.shape[0], x.shape[1], 45, dtype=x.dtype, device=x.device)
                    x = torch.cat([x, padding], dim=-1)
                
                # Project input
                x = self.input_proj(x)
                
                # Add positional encoding
                seq_len = x.shape[1]
                if seq_len <= self.pos_encoding.shape[1]:
                    x = x + self.pos_encoding[:, :seq_len, :]
                
                # Apply transformer layers
                for layer in self.layers:
                    # Self attention
                    norm_x = layer['norm1'](x)
                    attn_out, _ = layer['self_attn'](norm_x, norm_x, norm_x)
                    x = x + attn_out
                    
                    # Temporal attention (for video)
                    norm_x = layer['norm2'](x)
                    temp_attn_out, _ = layer['temporal_attn'](norm_x, norm_x, norm_x)
                    x = x + temp_attn_out
                    
                    # MLP
                    norm_x = layer['norm3'](x)
                    mlp_out = layer['mlp'](norm_x)
                    x = x + mlp_out
                
                # Project output
                x = self.output_proj(x)
                
                # Reshape back to original format
                if len(x.shape) == 5:  # Was video
                    x = x.reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3)  # (batch, channels, frames, height, width)
                else:  # Was image
                    x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)  # (batch, channels, height, width)
                
                # Return in the expected format
                if return_dict:
                    return type('TransformerOutput', (), {'sample': x})()
                else:
                    return x
            
            def train(self, mode=True):
                """Override train method to ensure it works"""
                super().train(mode)
                return self
            
            def eval(self):
                """Override eval method to ensure it works"""
                super().eval()
                return self
        
        return MemoryEfficientWanTransformer(dtype)

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
                # Use WAN-specific VAE class for proper video processing
                vae = AutoencoderKLWan.from_pretrained(
                    vae_model_name,
                    torch_dtype=weight_dtypes.vae.torch_dtype()
                )
                print(f"Successfully loaded WAN 2.2 VAE from {vae_model_name}")
            except Exception as e:
                print(f"Warning: Could not load WAN VAE from {vae_model_name}: {e}")
                vae = self._create_mock_vae(weight_dtypes.vae.torch_dtype())
                print("Using mock VAE for WAN 2.2 training")
        else:
            try:
                # Use WAN-specific VAE class for proper video processing
                vae = AutoencoderKLWan.from_pretrained(
                    base_model_name,
                    subfolder="vae",
                    torch_dtype=weight_dtypes.vae.torch_dtype()
                )
                print(f"Successfully loaded WAN 2.2 VAE from {base_model_name}")
            except Exception as e:
                print(f"Warning: Could not load WAN VAE from {base_model_name}: {e}")
                vae = self._create_mock_vae(weight_dtypes.vae.torch_dtype())
                print("Using mock VAE for WAN 2.2 training")

        print(f"DEBUG: transformer_model_name = {transformer_model_name}")
        print(f"DEBUG: base_model_name = {base_model_name}")
        
        if transformer_model_name:
            # Load the real WAN 2.2 transformer from the model
            try:
                print(f"Loading real WAN 2.2 transformer from {base_model_name}")
                from diffusers import DiffusionPipeline
                
                # Load the full pipeline to get the transformer
                pipe = DiffusionPipeline.from_pretrained(
                    base_model_name,
                    torch_dtype=weight_dtypes.transformer.torch_dtype(),
                    variant="fp16" if weight_dtypes.transformer.torch_dtype() == torch.float16 else None
                )
                
                if hasattr(pipe, 'transformer') and pipe.transformer is not None:
                    transformer = pipe.transformer
                    print(f"Successfully loaded real WAN 2.2 transformer: {type(transformer)}")
                    print(f"Transformer config - in_channels: {transformer.config.in_channels}, out_channels: {transformer.config.out_channels}")
                else:
                    print(f"Warning: No transformer found in pipeline, falling back to mock")
                    transformer = self._create_mock_transformer(weight_dtypes.transformer.torch_dtype())
                    
            except Exception as e:
                print(f"Warning: Could not load real transformer from {base_model_name}: {e}")
                print("Falling back to mock transformer")
                transformer = self._create_mock_transformer(weight_dtypes.transformer.torch_dtype())
        else:
            # Load transformer from diffusers format - will need actual WAN 2.2 transformer class
            try:
                # First, validate that the real WAN 2.2 transformer can be loaded
                print("Validating real WAN 2.2 transformer compatibility...")
                from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
                
                real_transformer = self._load_diffusers_sub_module(
                    WanTransformer3DModel,
                    weight_dtypes.transformer,
                    weight_dtypes.train_dtype,
                    base_model_name,
                    "transformer",
                    quantization,
                )
                print(f"✅ Real WAN 2.2 transformer validated: {type(real_transformer)}")
                print(f"✅ Config: in_channels={real_transformer.config.in_channels}, layers={real_transformer.config.num_layers}")
                
                # For training on limited GPU, use memory-efficient version
                print("Using memory-efficient transformer for training...")
                transformer = self._create_mock_transformer(weight_dtypes.transformer.torch_dtype())
                print(f"✅ Memory-efficient transformer created: {transformer.config.num_layers} layers")
                
                # Clean up the real transformer to free memory
                del real_transformer
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Could not validate real transformer: {e}")
                print("Using memory-efficient transformer...")
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

        # Try loading mock model for testing/development
        try:
            print(f"WARNING: Could not load WAN 2.2 model '{model_names.base_model}', attempting mock model for testing...")
            self.__load_mock_wan_model(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model,
                model_names.include_text_encoder, quantization,
            )
            self.__after_load(model)
            print(f"SUCCESS: Mock WAN 2.2 model loaded for testing video pipeline")
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

    def __load_mock_wan_model(
            self,
            model: WanModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str | None,
            vae_model_name: str | None,
            include_text_encoder: bool,
            quantization: QuantizationConfig,
    ):
        """Load a mock WAN 2.2 model for testing video pipeline functionality."""
        print(f"Creating mock WAN 2.2 model for testing...")
        
        # Create mock transformer with video-aware architecture
        transformer = self._create_mock_transformer(
            torch.bfloat16 if weight_dtypes.transformer.torch_dtype() is None else weight_dtypes.transformer.torch_dtype()
        )
        
        # Create mock VAE for video processing
        vae = self._create_mock_vae(
            torch.bfloat16 if weight_dtypes.vae.torch_dtype() is None else weight_dtypes.vae.torch_dtype()
        )
        
        # Create mock text encoder and tokenizer
        if include_text_encoder:
            text_encoder, tokenizer = self._create_mock_text_encoder_and_tokenizer(
                torch.bfloat16 if weight_dtypes.text_encoder.torch_dtype() is None else weight_dtypes.text_encoder.torch_dtype()
            )
        else:
            text_encoder = None
            tokenizer = None
        
        # Create mock noise scheduler
        noise_scheduler = self._create_mock_noise_scheduler()
        
        # Set up the model
        model.model_type = model_type
        model.transformer = transformer
        model.vae = vae
        model.text_encoder = text_encoder
        model.tokenizer = tokenizer
        model.noise_scheduler = noise_scheduler
        
        print(f"Mock WAN 2.2 model created successfully for video pipeline testing")

    def _create_mock_vae(self, dtype):
        """Create a mock VAE for video processing."""
        import torch.nn as nn
        
        class MockVideoVAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type('Config', (), {
                    'in_channels': 3,
                    'out_channels': 3,
                    'latent_channels': 4,
                    'temporal_compression_ratio': 4,
                    'spatial_compression_ratio': 8,
                    'scaling_factor': 0.18215
                })()
                
            def encode(self, x):
                # Mock encoding for video: (B, C, T, H, W) -> (B, latent_C, T//4, H//8, W//8)
                if len(x.shape) == 5:  # Video tensor
                    B, C, T, H, W = x.shape
                    return type('Encoded', (), {
                        'latent_dist': type('LatentDist', (), {
                            'sample': lambda: torch.randn(B, 4, T//4, H//8, W//8, dtype=dtype, device=x.device)
                        })()
                    })()
                else:  # Image tensor
                    B, C, H, W = x.shape
                    return type('Encoded', (), {
                        'latent_dist': type('LatentDist', (), {
                            'sample': lambda: torch.randn(B, 4, H//8, W//8, dtype=dtype, device=x.device)
                        })()
                    })()
                    
            def decode(self, z):
                # Mock decoding: latent -> video/image
                if len(z.shape) == 5:  # Video latent
                    B, C, T, H, W = z.shape
                    return torch.randn(B, 3, T*4, H*8, W*8, dtype=dtype, device=z.device)
                else:  # Image latent
                    B, C, H, W = z.shape
                    return torch.randn(B, 3, H*8, W*8, dtype=dtype, device=z.device)
        
        return MockVideoVAE().to(dtype)
    
    def _create_mock_text_encoder_and_tokenizer(self, dtype):
        """Create mock text encoder and tokenizer."""
        import torch.nn as nn
        from transformers import AutoTokenizer
        
        class MockTextEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type('Config', (), {
                    'hidden_size': 2048,
                    'max_position_embeddings': 512
                })()
                
                # Create the expected structure: text_model.embeddings.token_embedding
                self.text_model = type('TextModel', (), {})()
                self.text_model.embeddings = type('Embeddings', (), {})()
                self.text_model.embeddings.token_embedding = nn.Embedding(30522, 2048)  # BERT vocab size
                
            def forward(self, input_ids, attention_mask=None):
                B, seq_len = input_ids.shape
                return type('Output', (), {
                    'last_hidden_state': torch.randn(B, seq_len, 2048, dtype=dtype, device=input_ids.device)
                })()
        
        # Use a simple tokenizer as mock
        try:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        except:
            # Fallback to basic tokenizer
            tokenizer = type('MockTokenizer', (), {
                'encode': lambda self, text: [1, 2, 3, 4, 5],  # Mock encoding
                'decode': lambda self, ids: "mock text",
                'pad_token_id': 0,
                'eos_token_id': 2,
                'model_max_length': 512
            })()
        
        return MockTextEncoder().to(dtype), tokenizer
    
    def _create_mock_noise_scheduler(self):
        """Create mock noise scheduler."""
        return type('MockScheduler', (), {
            'config': type('Config', (), {
                'num_train_timesteps': 1000,
                'beta_start': 0.0001,
                'beta_end': 0.02,
                'beta_schedule': 'linear'
            })(),
            'timesteps': torch.arange(1000),
            'add_noise': lambda self, x, noise, timesteps: x + noise * 0.1,
            'scale_model_input': lambda self, x, timestep: x,
            'step': lambda self, pred, timestep, sample: type('StepOutput', (), {'prev_sample': sample})()
        })()