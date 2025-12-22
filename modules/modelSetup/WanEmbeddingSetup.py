from modules.model.WanModel import WanModel
from modules.modelSetup.BaseWanSetup import BaseWanSetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


class WanEmbeddingSetup(
    BaseWanSetup,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super().__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: WanModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        # Only add embedding parameters for textual inversion training
        if config.train_any_embedding() or config.train_any_output_embedding():
            if config.text_encoder.train_embedding and model.text_encoder is not None:
                self._add_embedding_param_groups(
                    model.all_text_encoder_embeddings(), 
                    parameter_group_collection, 
                    config.embedding_learning_rate,
                    "embeddings"
                )

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        self._setup_embeddings_requires_grad(model, config)
        
        # Freeze all model parameters except embeddings
        if model.text_encoder is not None:
            model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)

    def setup_model(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        # Setup embedding dtype
        if config.train_any_embedding():
            if model.text_encoder is not None:
                model.text_encoder.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())

        # Setup embeddings and embedding wrapper
        self._setup_embeddings(model, config)
        self._setup_embedding_wrapper(model, config)
        
        # Initialize output embedding for WAN 2.2
        model.output_embedding = torch.zeros(
            size=(4, 4096), 
            dtype=config.train_dtype.torch_dtype(), 
            device=self.train_device
        )
        
        self.__setup_requires_grad(model, config)

        # Initialize model parameters
        init_model_parameters(model, self.create_parameters(model, config), self.train_device)

    def setup_train_device(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        # For embedding training, we need text encoder on train device
        # VAE and transformer can be on temp device since they're frozen
        model.text_encoder_to(self.train_device)
        model.vae_to(self.temp_device)
        model.transformer_to(self.temp_device)

        # Set all components to eval mode since we're only training embeddings
        if model.text_encoder:
            model.text_encoder.eval()
        model.vae.eval()
        model.transformer.eval()

    def after_optimizer_step(
            self,
            model: WanModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        # Normalize embeddings if required
        if config.preserve_embedding_norm:
            self._normalize_output_embeddings(model.all_text_encoder_embeddings())
            if model.embedding_wrapper is not None:
                model.embedding_wrapper.normalize_embeddings()
        
        # Update requires_grad settings
        self.__setup_requires_grad(model, config)