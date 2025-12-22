from modules.model.WanModel import WanModel
from modules.modelSaver.mixin.LoRASaverMixin import LoRASaverMixin
from modules.util.enum.ModelFormat import ModelFormat

import torch
from torch import Tensor

from omi_model_standards.convert.lora.convert_lora_util import LoraConversionKeySet


class WanLoRASaver(
    LoRASaverMixin,
):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: WanModel) -> list[LoraConversionKeySet] | None:
        # TODO: Implement WAN 2.2 specific LoRA conversion key sets when available
        # This will need to be updated with actual WAN 2.2 LoRA conversion utilities
        # from omi_model_standards.convert.lora.convert_wan_lora import convert_wan_lora_key_sets
        # return convert_wan_lora_key_sets()
        
        # For now, return None to use default LoRA saving without conversion
        return None

    def _get_state_dict(
            self,
            model: WanModel,
    ) -> dict[str, Tensor]:
        state_dict = {}
        
        # Add text encoder LoRA weights if available
        if model.text_encoder_lora is not None:
            state_dict |= model.text_encoder_lora.state_dict()
        
        # Add transformer LoRA weights if available
        if model.transformer_lora is not None:
            state_dict |= model.transformer_lora.state_dict()
        
        # Add any additional LoRA state dict
        if model.lora_state_dict is not None:
            state_dict |= model.lora_state_dict

        # Bundle additional embeddings if configured
        if model.additional_embeddings and model.train_config.bundle_additional_embeddings:
            for embedding in model.additional_embeddings:
                placeholder = embedding.text_encoder_embedding.placeholder

                if embedding.text_encoder_embedding.vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.text_encoder"] = embedding.text_encoder_embedding.vector
                if embedding.text_encoder_embedding.output_vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.text_encoder_out"] = embedding.text_encoder_embedding.output_vector

        return state_dict

    def save(
            self,
            model: WanModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        self._save(model, output_model_format, output_model_destination, dtype)