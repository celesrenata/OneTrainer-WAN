from modules.model.WanModel import WanModel
from modules.modelSetup.BaseWanSetup import BaseWanSetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.ModuleFilter import ModuleFilter
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


class WanFineTuneSetup(
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

        # Add text encoder parameters
        self._create_model_part_parameters(
            parameter_group_collection, 
            "text_encoder", 
            model.text_encoder, 
            config.text_encoder
        )

        # Add embedding parameters if training embeddings
        if config.train_any_embedding() or config.train_any_output_embedding():
            if config.text_encoder.train_embedding and model.text_encoder is not None:
                self._add_embedding_param_groups(
                    model.all_text_encoder_embeddings(), 
                    parameter_group_collection, 
                    config.embedding_learning_rate,
                    "embeddings"
                )

        # Add transformer parameters with optional layer filtering
        self._create_model_part_parameters(
            parameter_group_collection, 
            "transformer", 
            model.transformer, 
            config.transformer,
            freeze=ModuleFilter.create(config), 
            debug=config.debug_mode
        )

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        self._setup_embeddings_requires_grad(model, config)

        self._setup_model_part_requires_grad(
            "text_encoder", 
            model.text_encoder, 
            config.text_encoder, 
            model.train_progress
        )
        self._setup_model_part_requires_grad(
            "transformer", 
            model.transformer, 
            config.transformer, 
            model.train_progress
        )

        # VAE is always frozen during training
        model.vae.requires_grad_(False)

    def setup_model(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        # Setup embedding dtype if training embeddings
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
        # Determine device placement based on caching and training settings
        vae_on_train_device = not config.latent_caching
        text_encoder_on_train_device = \
            config.train_text_encoder_or_embedding() \
            or not config.latent_caching

        # Move components to appropriate devices
        model.text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)

        # Set training/eval modes
        if model.text_encoder:
            if config.text_encoder.train:
                model.text_encoder.train()
            else:
                model.text_encoder.eval()

        # VAE is always in eval mode
        model.vae.eval()

        # Set transformer mode
        if config.transformer.train:
            model.transformer.train()
        else:
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