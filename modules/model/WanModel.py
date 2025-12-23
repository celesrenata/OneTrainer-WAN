from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel, BaseModelEmbedding
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
from torch import Tensor

from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import PreTrainedTokenizer, PreTrainedModel


class WanModelEmbedding:
    def __init__(
            self,
            uuid: str,
            text_encoder_vector: Tensor | None,
            placeholder: str,
            is_output_embedding: bool,
    ):
        self.text_encoder_embedding = BaseModelEmbedding(
            uuid=uuid,
            placeholder=placeholder,
            vector=text_encoder_vector,
            is_output_embedding=is_output_embedding,
        )



    def clear_cache(self):
        """Clear model cache to free memory."""
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear any cached computations
        if hasattr(self, '_cached_text_embeddings'):
            delattr(self, '_cached_text_embeddings')
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def enable_memory_efficient_attention(self):
        """Enable memory efficient attention if available."""
        try:
            # Try to enable xformers if available
            if hasattr(self.transformer, 'enable_xformers_memory_efficient_attention'):
                self.transformer.enable_xformers_memory_efficient_attention()
                return True
        except Exception:
            pass
        
        try:
            # Try to enable flash attention if available
            if hasattr(self.transformer, 'enable_flash_attention'):
                self.transformer.enable_flash_attention()
                return True
        except Exception:
            pass
        
        return False
    
    def optimize_for_inference(self):
        """Optimize model for inference to reduce memory usage."""
        self.eval()
        
        # Disable gradient computation
        for param in self.parameters():
            param.requires_grad = False
        
        # Enable memory efficient attention
        self.enable_memory_efficient_attention()
        
        # Clear cache
        self.clear_cache()

class WanModel(BaseModel):
    # base model data
    tokenizer: PreTrainedTokenizer | None
    text_encoder: PreTrainedModel | None
    vae: AutoencoderKL | None
    transformer: PreTrainedModel | None  # WAN 2.2 transformer (will be specific type when available)
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None

    # autocast context
    text_encoder_autocast_context: torch.autocast | nullcontext
    transformer_autocast_context: torch.autocast | nullcontext

    text_encoder_train_dtype: DataType
    transformer_train_dtype: DataType

    text_encoder_offload_conductor: LayerOffloadConductor | None
    transformer_offload_conductor: LayerOffloadConductor | None

    # persistent embedding training data
    embedding: WanModelEmbedding | None
    additional_embeddings: list[WanModelEmbedding] | None
    embedding_wrapper: AdditionalEmbeddingWrapper | None

    # persistent lora training data
    text_encoder_lora: LoRAModuleWrapper | None
    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    def __init__(
            self,
            model_type: ModelType,
    ):
        super().__init__(
            model_type=model_type,
        )

        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.transformer = None
        self.noise_scheduler = None

        self.text_encoder_autocast_context = nullcontext()
        self.transformer_autocast_context = nullcontext()
        
        # General autocast context (will be set by model setup)
        self.autocast_context = nullcontext()

        self.text_encoder_train_dtype = DataType.FLOAT_32
        self.transformer_train_dtype = DataType.FLOAT_32
        
        # General train dtype (will be set by model setup)
        self.train_dtype = DataType.FLOAT_32

        self.text_encoder_offload_conductor = None
        self.transformer_offload_conductor = None

        self.embedding = None
        self.additional_embeddings = []
        self.embedding_wrapper = None

        self.text_encoder_lora = None
        self.transformer_lora = None
        self.lora_state_dict = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [a for a in [
            self.text_encoder_lora,
            self.transformer_lora,
        ] if a is not None]

    def all_embeddings(self) -> list[WanModelEmbedding]:
        return self.additional_embeddings \
               + ([self.embedding] if self.embedding is not None else [])

    def all_text_encoder_embeddings(self) -> list[BaseModelEmbedding]:
        return [embedding.text_encoder_embedding for embedding in self.additional_embeddings] \
               + ([self.embedding.text_encoder_embedding] if self.embedding is not None else [])

    def vae_to(self, device: torch.device):
        if self.vae is not None:
            self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        if self.text_encoder is not None:
            if self.text_encoder_offload_conductor is not None and \
                    self.text_encoder_offload_conductor.layer_offload_activated():
                self.text_encoder_offload_conductor.to(device)
            else:
                self.text_encoder.to(device=device)

        if self.text_encoder_lora is not None:
            self.text_encoder_lora.to(device)

    def transformer_to(self, device: torch.device):
        if self.transformer is not None:
            if self.transformer_offload_conductor is not None and \
                    self.transformer_offload_conductor.layer_offload_activated():
                self.transformer_offload_conductor.to(device)
            else:
                self.transformer.to(device=device)

        if self.transformer_lora is not None:
            self.transformer_lora.to(device)

    def to(self, device: torch.device):
        self.vae_to(device)
        self.text_encoder_to(device)
        self.transformer_to(device)

    def eval(self):
        if self.vae is not None:
            self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        if self.transformer is not None:
            self.transformer.eval()

    def add_text_encoder_embeddings_to_prompt(self, prompt: str) -> str:
        return self._add_embeddings_to_prompt(self.all_text_encoder_embeddings(), prompt)

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str = None,
            tokens: Tensor = None,
            text_encoder_layer_skip: int = 0,
            text_encoder_dropout_probability: float | None = None,
            text_encoder_output: Tensor = None,
    ) -> Tensor:
        # tokenize prompt
        if tokens is None and text is not None and self.tokenizer is not None:
            tokenizer_output = self.tokenizer(
                self.add_text_encoder_embeddings_to_prompt(text),
                padding='max_length',
                truncation=True,
                max_length=77,  # Standard sequence length for text encoders
                return_tensors="pt",
            )
            tokens = tokenizer_output.input_ids.to(self.text_encoder.device)

        # encode text
        with self.text_encoder_autocast_context:
            if text_encoder_output is None and self.text_encoder is not None:
                text_encoder_output = self.text_encoder(tokens)[0]
            
            if text_encoder_output is None:
                text_encoder_output = torch.zeros(
                    size=(batch_size, 77, 768),  # Default dimensions
                    device=train_device,
                    dtype=self.train_dtype.torch_dtype(),
                )

        text_encoder_output = self._apply_output_embeddings(
            self.all_text_encoder_embeddings(),
            self.tokenizer,
            tokens,
            text_encoder_output,
        )

        # apply dropout
        if text_encoder_dropout_probability is not None:
            dropout_text_encoder_mask = (torch.tensor(
                [rand.random() > text_encoder_dropout_probability for _ in range(batch_size)],
                device=train_device)).float()
            text_encoder_output = text_encoder_output * dropout_text_encoder_mask[:, None, None]

        return text_encoder_output

    def pack_latents(self, latents: Tensor) -> Tensor:
        """Pack video latents for transformer processing"""
        batch_size, channels, frames, height, width = latents.shape
        # Reshape for video processing - this will need to be adapted based on WAN 2.2 specifics
        latents = latents.view(batch_size, channels * frames, height, width)
        return latents

    def unpack_latents(self, latents: Tensor, frames: int, height: int, width: int) -> Tensor:
        """Unpack video latents from transformer output"""
        batch_size, channels_frames, h, w = latents.shape
        channels = channels_frames // frames
        latents = latents.view(batch_size, channels, frames, h, w)
        return latents

    def create_pipeline(self) -> DiffusionPipeline:
        """Create inference pipeline for WAN 2.2"""
        # Create a mock pipeline that works with our components
        from diffusers import DiffusionPipeline
        
        print("Creating WAN 2.2 mock pipeline...")
        
        # Create a custom pipeline class that works with our mock components
        class MockWanPipeline(DiffusionPipeline):
            def __init__(self, transformer, scheduler, vae, text_encoder, tokenizer):
                super().__init__()
                self.transformer = transformer
                self.scheduler = scheduler
                self.vae = vae
                self.text_encoder = text_encoder
                self.tokenizer = tokenizer
                
                # Add video processor for compatibility
                self.video_processor = None  # Will be set if needed
                
            def __call__(self, prompt, **kwargs):
                # Mock implementation for sampling
                # This will be replaced with actual WAN 2.2 pipeline logic
                import torch
                
                # Return a dummy tensor for now
                batch_size = kwargs.get('batch_size', 1)
                height = kwargs.get('height', 512)
                width = kwargs.get('width', 512)
                frames = kwargs.get('frames', 8)
                
                # Create dummy video output
                dummy_video = torch.randn(batch_size, frames, 3, height, width)
                
                return {"videos": dummy_video}
                
            def to(self, device):
                """Move pipeline to device"""
                if self.transformer is not None:
                    self.transformer.to(device)
                if self.vae is not None:
                    self.vae.to(device)
                if self.text_encoder is not None:
                    self.text_encoder.to(device)
                return self
        
        try:
            # Return the mock pipeline with our components
            pipeline = MockWanPipeline(
                transformer=self.transformer,
                scheduler=self.noise_scheduler,
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
            )
            print("✓ WAN 2.2 mock pipeline created successfully")
            return pipeline
        except Exception as e:
            print(f"Error creating WAN 2.2 pipeline: {e}")
            # Return a minimal pipeline as fallback
            class MinimalPipeline(DiffusionPipeline):
                def __init__(self):
                    super().__init__()
                    
                def __call__(self, *args, **kwargs):
                    import torch
                    return {"videos": torch.randn(1, 8, 3, 512, 512)}
                    
                def to(self, device):
                    return self
            
            print("Using minimal fallback pipeline")
            return MinimalPipeline()