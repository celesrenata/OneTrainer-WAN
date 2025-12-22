from modules.model.WanModel import WanModel
from modules.modelLoader.GenericEmbeddingModelLoader import make_embedding_model_loader
from modules.modelLoader.wan.WanEmbeddingLoader import WanEmbeddingLoader
from modules.modelLoader.wan.WanModelLoader import WanModelLoader
from modules.util.enum.ModelType import ModelType

WanEmbeddingModelLoader = make_embedding_model_loader(
    model_spec_map={ModelType.WAN_2_2: "resources/sd_model_spec/wan_2_2-embedding.json"},
    model_class=WanModel,
    model_loader_class=WanModelLoader,
    embedding_loader_class=WanEmbeddingLoader,
)