from modules.model.WanModel import WanModel
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.wan.WanEmbeddingLoader import WanEmbeddingLoader
from modules.modelLoader.wan.WanModelLoader import WanModelLoader
from modules.util.enum.ModelType import ModelType

WanFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={ModelType.WAN_2_2: "resources/sd_model_spec/wan_2_2.json"},
    model_class=WanModel,
    model_loader_class=WanModelLoader,
    embedding_loader_class=WanEmbeddingLoader,
)