# Video Training with OneTrainer

OneTrainer supports training video generation models including WAN 2.2, Hunyuan Video, and HiDream. This guide covers the specific considerations and configuration options for video model training.

## Supported Video Models

- **WAN 2.2**: World Animator Network 2.2 for high-quality video generation
- **Hunyuan Video**: Hunyuan's video generation model
- **HiDream**: High-resolution video generation model

## Video-Specific Configuration

When training video models, additional configuration options are available under the `video_config` section:

### Data Processing Parameters

- **max_frames**: Maximum number of frames to use for training (default: 16)
  - Higher values require more memory but can improve temporal consistency
  - Recommended: 8-16 for most setups, 32+ for high-end hardware

- **frame_sample_strategy**: Strategy for sampling frames from videos
  - `uniform`: Evenly spaced frames (recommended for most cases)
  - `random`: Randomly selected frames (good for data augmentation)
  - `keyframe`: Prefer keyframes (useful for scene-based training)

- **target_fps**: Target frames per second for video processing (default: 24.0)
  - Videos will be resampled to this FPS during preprocessing
  - Common values: 12, 24, 30

- **max_duration**: Maximum duration of video clips in seconds (default: 10.0)
  - Longer clips require more memory
  - Recommended: 5-10 seconds for most training scenarios

### Temporal Consistency Parameters

- **temporal_consistency_weight**: Weight for temporal consistency loss (default: 1.0)
  - Higher values enforce stronger consistency between frames
  - Range: 0.0-2.0, start with 1.0 and adjust based on results

- **use_temporal_attention**: Enable temporal attention mechanisms (default: true)
  - Improves frame-to-frame consistency
  - Disable only if experiencing memory issues

### Memory Management

- **spatial_compression_ratio**: Compression ratio for spatial dimensions (default: 8)
  - Higher values reduce memory usage but may affect quality
  - Common values: 4, 8, 16

- **temporal_compression_ratio**: Compression ratio for temporal dimension (default: 4)
  - Higher values reduce memory usage for longer sequences
  - Common values: 2, 4, 8

- **video_batch_size_multiplier**: Multiplier for batch size when processing video (default: 0.5)
  - Lower values reduce memory usage
  - Adjust based on available VRAM

### Training Parameters

- **frame_dropout_probability**: Probability of dropping frames during training (default: 0.0)
  - Used for regularization, range: 0.0-0.3
  - Higher values can improve generalization but may hurt temporal consistency

- **temporal_augmentation**: Enable temporal augmentations (default: false)
  - Includes frame shuffling and temporal cropping
  - Can improve model robustness but increases training time

## Memory Requirements

Video training is significantly more memory-intensive than image training. Here are rough guidelines:

### WAN 2.2 Memory Requirements

| Configuration | VRAM | Batch Size | Frames | Resolution |
|---------------|------|------------|--------|------------|
| 8GB Setup    | 8GB  | 1          | 8      | 384x384    |
| 16GB Setup   | 16GB | 2          | 16     | 512x512    |
| 24GB Setup   | 24GB | 4          | 16     | 512x512    |
| 48GB+ Setup  | 48GB | 8          | 32     | 768x768    |

### Memory Optimization Tips

1. **Use CPU Offloading**: Enable `gradient_checkpointing: "CPU_OFFLOADED"`
2. **Increase Layer Offload**: Set `layer_offload_fraction` to 0.5-0.8
3. **Reduce Batch Size**: Start with batch_size: 1 for video models
4. **Lower Precision**: Use `BFLOAT_16` or `FLOAT_16` for train_dtype
5. **Quantization**: Use `FLOAT_8` or `NFLOAT_4` for model weights
6. **Reduce Frames**: Lower `max_frames` and `frames` settings
7. **Lower Resolution**: Start with 384x384 or 256x256 resolution

## Training Methods

### LoRA Training (Recommended)

LoRA is the most memory-efficient method for video model training:

```json
{
    "training_method": "LORA",
    "batch_size": 2,
    "learning_rate": 0.0003,
    "transformer": {
        "train": true
    },
    "text_encoder": {
        "train": false
    }
}
```

### Fine-Tuning

Full model fine-tuning requires significantly more memory:

```json
{
    "training_method": "FINE_TUNE",
    "batch_size": 1,
    "learning_rate": 0.0001,
    "transformer": {
        "train": true
    },
    "text_encoder": {
        "train": true,
        "learning_rate": 0.00005
    }
}
```

### Embedding Training

Textual inversion for video models:

```json
{
    "training_method": "EMBEDDING",
    "batch_size": 4,
    "learning_rate": 0.001,
    "embedding_learning_rate": 0.005
}
```

## Data Preparation

### Video Format Support

Supported video formats:
- MP4 (recommended)
- AVI
- MOV
- WebM

### Data Organization

Organize your video data similar to image datasets:

```
dataset/
├── videos/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── captions/
    ├── video1.txt
    ├── video2.txt
    └── ...
```

### Caption Guidelines

- Keep captions descriptive but concise
- Focus on visual elements, actions, and scene descriptions
- Avoid temporal references like "first", "then", "finally"
- Example: "A person walking through a forest with sunlight filtering through trees"

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size to 1
   - Enable CPU offloading
   - Reduce number of frames
   - Lower resolution

2. **Poor Temporal Consistency**
   - Increase temporal_consistency_weight
   - Enable temporal attention
   - Use uniform frame sampling
   - Reduce frame dropout probability

3. **Slow Training**
   - Increase dataloader_threads (if CPU allows)
   - Use faster storage (SSD)
   - Enable gradient checkpointing
   - Consider mixed precision training

4. **Quality Issues**
   - Check video preprocessing settings
   - Ensure consistent frame rates in source videos
   - Verify caption quality
   - Adjust learning rates

### Performance Tips

1. **Use Presets**: Start with provided presets and adjust as needed
2. **Monitor Memory**: Use tools like `nvidia-smi` to monitor VRAM usage
3. **Gradual Scaling**: Start with small configurations and scale up
4. **Regular Sampling**: Enable sampling to monitor training progress
5. **Backup Frequently**: Video training can be unstable, backup often

## Example Configurations

See the provided training presets:
- `#wan 2.2 LoRA.json`: Standard LoRA training
- `#wan 2.2 LoRA 8GB.json`: Memory-optimized for 8GB VRAM
- `#wan 2.2 Finetune.json`: Full fine-tuning setup
- `#wan 2.2 Embedding.json`: Textual inversion training

These presets provide good starting points that can be customized for your specific needs.