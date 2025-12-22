from modules.model.BaseModel import BaseModel
from modules.model.WanModel import WanModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.ModelNames import ModelNames

from omi_model_standards.convert.lora.convert_lora_util import LoraConversionKeySet


class WanLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        # TODO: Implement WAN 2.2 specific LoRA conversion key sets when available
        # This will need to be updated with actual WAN 2.2 LoRA conversion utilities
        # from omi_model_standards.convert.lora.convert_wan_lora import convert_wan_lora_key_sets
        # return convert_wan_lora_key_sets()
        
        # For now, return None to use default LoRA loading without conversion
        # This allows basic LoRA functionality while waiting for WAN 2.2 specific conversion
        return None

    def load(
            self,
            model: WanModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)