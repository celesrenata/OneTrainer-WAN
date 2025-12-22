#!/usr/bin/env python3
"""
WAN 2.2 Training Examples for OneTrainer

This script provides example configurations and workflows for training WAN 2.2 models
with different hardware setups and training objectives.

Usage:
    python examples/wan22_training_examples.py --example <example_name>

Examples:
    - character_lora_8gb: Character LoRA training for 8GB VRAM
    - style_lora_16gb: Style transfer LoRA for 16GB VRAM  
    - domain_finetune_24gb: Domain adaptation fine-tuning for 24GB+ VRAM
    - concept_embedding: Quick concept learning with embeddings
    - rocm_optimized: ROCm-optimized configuration for AMD GPUs
"""

import argparse
import json
import os
from pathlib import Path


def create_character_lora_8gb():
    """Character LoRA training optimized for 8GB VRAM"""
    return {
        "model_type": "WAN_2_2",
        "base_model_name": "wan-ai/WAN_2_2",
        "training_method": "LORA",
        
        # Memory optimization for 8GB
        "batch_size": 1,
        "gradient_checkpointing": "CPU_OFFLOADED",
        "layer_offload_fraction": 0.8,
        "train_dtype": "BFLOAT_16",
        
        # Training parameters
        "learning_rate": 0.0003,
        "epochs": 15,
        "resolution": "384",
        "frames": 8,
        
        # Model configuration
        "transformer": {
            "train": True,
            "weight_dtype": "NFLOAT_4"
        },
        "text_encoder": {
            "train": False,
            "weight_dtype": "NFLOAT_4"
        },
        
        # Video-specific settings
        "video_config": {
            "max_frames": 8,
            "frame_sample_strategy": "uniform",
            "target_fps": 12.0,
            "max_duration": 8.0,
            "temporal_consistency_weight": 1.2,
            "use_temporal_attention": True,
            "spatial_compression_ratio": 16,
            "temporal_compression_ratio": 8,
            "video_batch_size_multiplier": 0.25,
            "frame_dropout_probability": 0.0,
            "temporal_augmentation": False
        },
        
        # Output settings
        "output_model_destination": "models/character_lora_8gb.safetensors",
        "output_model_format": "SAFETENSORS",
        
        # Backup and sampling
        "backup_after": 20,
        "sample_after": 100,
        "sample_prompts": [
            "the character walking in a park",
            "the character sitting at a table", 
            "the character waving hello"
        ]
    }


def create_style_lora_16gb():
    """Style transfer LoRA training for 16GB VRAM"""
    return {
        "model_type": "WAN_2_2",
        "base_model_name": "wan-ai/WAN_2_2", 
        "training_method": "LORA",
        
        # Balanced settings for 16GB
        "batch_size": 2,
        "gradient_checkpointing": "CPU_OFFLOADED",
        "layer_offload_fraction": 0.5,
        "train_dtype": "BFLOAT_16",
        
        # Training parameters
        "learning_rate": 0.0002,
        "epochs": 20,
        "resolution": "512",
        "frames": 16,
        
        # Model configuration
        "transformer": {
            "train": True,
            "weight_dtype": "FLOAT_8"
        },
        "text_encoder": {
            "train": False,
            "weight_dtype": "FLOAT_8"
        },
        
        # Video-specific settings
        "video_config": {
            "max_frames": 16,
            "frame_sample_strategy": "uniform",
            "target_fps": 24.0,
            "max_duration": 10.0,
            "temporal_consistency_weight": 1.0,
            "use_temporal_attention": True,
            "spatial_compression_ratio": 8,
            "temporal_compression_ratio": 4,
            "video_batch_size_multiplier": 0.5,
            "frame_dropout_probability": 0.05,
            "temporal_augmentation": True
        },
        
        # Output settings
        "output_model_destination": "models/style_lora_16gb.safetensors",
        "output_model_format": "SAFETENSORS",
        
        # Backup and sampling
        "backup_after": 15,
        "sample_after": 75,
        "sample_prompts": [
            "a person walking through a forest",
            "ocean waves crashing on rocks",
            "city street at golden hour"
        ]
    }


def create_domain_finetune_24gb():
    """Domain adaptation fine-tuning for 24GB+ VRAM"""
    return {
        "model_type": "WAN_2_2",
        "base_model_name": "wan-ai/WAN_2_2",
        "training_method": "FINE_TUNE",
        
        # High-end settings for 24GB+
        "batch_size": 2,
        "gradient_checkpointing": "DEFAULT",
        "layer_offload_fraction": 0.2,
        "train_dtype": "BFLOAT_16",
        
        # Training parameters
        "learning_rate": 0.0001,
        "epochs": 30,
        "resolution": "512",
        "frames": 16,
        
        # Model configuration
        "transformer": {
            "train": True,
            "learning_rate": 0.0001,
            "weight_dtype": "BFLOAT_16"
        },
        "text_encoder": {
            "train": True,
            "learning_rate": 0.00005,
            "weight_dtype": "BFLOAT_16"
        },
        
        # Video-specific settings
        "video_config": {
            "max_frames": 16,
            "frame_sample_strategy": "uniform",
            "target_fps": 24.0,
            "max_duration": 12.0,
            "temporal_consistency_weight": 1.0,
            "use_temporal_attention": True,
            "spatial_compression_ratio": 8,
            "temporal_compression_ratio": 4,
            "video_batch_size_multiplier": 0.75,
            "frame_dropout_probability": 0.1,
            "temporal_augmentation": True
        },
        
        # Output settings
        "output_model_destination": "models/domain_finetune_24gb",
        "output_model_format": "DIFFUSERS",
        
        # Backup and sampling
        "backup_after": 10,
        "sample_after": 50,
        "sample_prompts": [
            "domain-specific scene 1",
            "domain-specific scene 2", 
            "domain-specific scene 3"
        ]
    }


def create_concept_embedding():
    """Quick concept learning with embedding training"""
    return {
        "model_type": "WAN_2_2",
        "base_model_name": "wan-ai/WAN_2_2",
        "training_method": "EMBEDDING",
        
        # Embedding-optimized settings
        "batch_size": 4,
        "gradient_checkpointing": "DEFAULT",
        "layer_offload_fraction": 0.3,
        "train_dtype": "BFLOAT_16",
        
        # Training parameters
        "learning_rate": 0.001,
        "embedding_learning_rate": 0.005,
        "epochs": 50,
        "resolution": "512",
        "frames": 12,
        
        # Video-specific settings
        "video_config": {
            "max_frames": 12,
            "frame_sample_strategy": "uniform",
            "target_fps": 24.0,
            "max_duration": 8.0,
            "temporal_consistency_weight": 1.0,
            "use_temporal_attention": True,
            "spatial_compression_ratio": 8,
            "temporal_compression_ratio": 4,
            "video_batch_size_multiplier": 0.8,
            "frame_dropout_probability": 0.0,
            "temporal_augmentation": False
        },
        
        # Output settings
        "output_model_destination": "models/concept_embedding.safetensors",
        "output_model_format": "SAFETENSORS",
        
        # Backup and sampling
        "backup_after": 25,
        "sample_after": 100,
        "sample_prompts": [
            "a <concept> in a natural setting",
            "close-up of <concept>",
            "<concept> with dramatic lighting"
        ]
    }


def create_rocm_optimized():
    """ROCm-optimized configuration for AMD GPUs"""
    return {
        "model_type": "WAN_2_2",
        "base_model_name": "wan-ai/WAN_2_2",
        "training_method": "LORA",
        
        # ROCm-optimized settings
        "batch_size": 1,
        "gradient_checkpointing": "CPU_OFFLOADED",
        "layer_offload_fraction": 0.6,
        "train_dtype": "BFLOAT_16",
        "dataloader_threads": 2,
        
        # Conservative training parameters for ROCm
        "learning_rate": 0.0003,
        "epochs": 15,
        "resolution": "448",  # ROCm-friendly resolution
        "frames": 12,
        
        # Model configuration
        "transformer": {
            "train": True,
            "weight_dtype": "FLOAT_8"
        },
        "text_encoder": {
            "train": False,
            "weight_dtype": "FLOAT_8"
        },
        
        # Video-specific settings
        "video_config": {
            "max_frames": 12,
            "frame_sample_strategy": "uniform",
            "target_fps": 20.0,
            "max_duration": 8.0,
            "temporal_consistency_weight": 1.0,
            "use_temporal_attention": True,
            "spatial_compression_ratio": 8,
            "temporal_compression_ratio": 6,
            "video_batch_size_multiplier": 0.4,
            "frame_dropout_probability": 0.0,
            "temporal_augmentation": False
        },
        
        # Output settings
        "output_model_destination": "models/rocm_lora.safetensors",
        "output_model_format": "SAFETENSORS",
        
        # Backup and sampling
        "backup_after": 20,
        "sample_after": 100
    }


def create_cpu_fallback():
    """CPU-only configuration for development/testing"""
    return {
        "model_type": "WAN_2_2",
        "base_model_name": "wan-ai/WAN_2_2",
        "training_method": "LORA",
        
        # CPU-optimized settings
        "batch_size": 1,
        "gradient_checkpointing": "NONE",
        "layer_offload_fraction": 0.0,
        "train_dtype": "FLOAT_32",
        "dataloader_threads": 1,
        
        # Minimal settings for CPU
        "learning_rate": 0.001,
        "epochs": 5,
        "resolution": "256",
        "frames": 4,
        
        # Model configuration
        "transformer": {
            "train": True,
            "weight_dtype": "FLOAT_32"
        },
        "text_encoder": {
            "train": False,
            "weight_dtype": "FLOAT_32"
        },
        
        # Video-specific settings
        "video_config": {
            "max_frames": 4,
            "frame_sample_strategy": "uniform",
            "target_fps": 12.0,
            "max_duration": 4.0,
            "temporal_consistency_weight": 0.5,
            "use_temporal_attention": False,
            "spatial_compression_ratio": 16,
            "temporal_compression_ratio": 8,
            "video_batch_size_multiplier": 1.0,
            "frame_dropout_probability": 0.0,
            "temporal_augmentation": False
        },
        
        # Output settings
        "output_model_destination": "models/cpu_test_lora.safetensors",
        "output_model_format": "SAFETENSORS",
        
        # Minimal backup and sampling
        "backup_after": 50,
        "sample_after": 200
    }


EXAMPLES = {
    "character_lora_8gb": create_character_lora_8gb,
    "style_lora_16gb": create_style_lora_16gb,
    "domain_finetune_24gb": create_domain_finetune_24gb,
    "concept_embedding": create_concept_embedding,
    "rocm_optimized": create_rocm_optimized,
    "cpu_fallback": create_cpu_fallback
}


def main():
    parser = argparse.ArgumentParser(description="Generate WAN 2.2 training configuration examples")
    parser.add_argument("--example", choices=EXAMPLES.keys(), required=True,
                       help="Example configuration to generate")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path (default: <example_name>.json)")
    parser.add_argument("--print", action="store_true",
                       help="Print configuration to stdout instead of saving")
    
    args = parser.parse_args()
    
    # Generate configuration
    config_func = EXAMPLES[args.example]
    config = config_func()
    
    # Add common settings
    config.update({
        "timestep_distribution": "LOGIT_NORMAL",
        "dataloader_threads": config.get("dataloader_threads", 1),
        "cache_latents": True,
        "cache_text_encoder_outputs": True,
        "cache_vae_outputs": True
    })
    
    if args.print:
        print(json.dumps(config, indent=2))
    else:
        output_path = args.output or f"{args.example}.json"
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to: {output_path}")
        print(f"\nTo use this configuration:")
        print(f"1. Load the configuration in OneTrainer")
        print(f"2. Set your dataset path in the Concepts tab")
        print(f"3. Adjust paths and settings as needed")
        print(f"4. Start training")


if __name__ == "__main__":
    main()