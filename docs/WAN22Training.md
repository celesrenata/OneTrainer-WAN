# WAN 2.2 Training Guide

WAN 2.2 (World Animator Network 2.2) is a state-of-the-art video generation model that can be trained using OneTrainer. This guide covers everything you need to know about training WAN 2.2 models, from data preparation to advanced configuration options.

## Table of Contents

- [Quick Start](#quick-start)
- [System Requirements](#system-requirements)
- [GPU Platform Setup](#gpu-platform-setup)
- [Data Preparation](#data-preparation)
- [Training Methods](#training-methods)
- [Configuration Reference](#configuration-reference)
- [Memory Optimization](#memory-optimization)
- [Troubleshooting](#troubleshooting)
- [Example Workflows](#example-workflows)

## Quick Start

### 1. Prepare Your Data

Organize your video dataset:
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

### 2. Choose a Training Preset

OneTrainer provides several WAN 2.2 presets:
- **`#wan 2.2 LoRA.json`**: Standard LoRA training (16GB+ VRAM)
- **`#wan 2.2 LoRA 8GB.json`**: Memory-optimized LoRA (8GB VRAM)
- **`#wan 2.2 Finetune.json`**: Full fine-tuning (24GB+ VRAM)
- **`#wan 2.2 Embedding.json`**: Textual inversion training

### 3. Configure Your Training

1. Load a preset in OneTrainer
2. Set your dataset path in the Concepts tab
3. Configure output model destination
4. Adjust video-specific settings if needed
5. Start training

## System Requirements

### Minimum Requirements

- **VRAM**: 8GB (with optimizations)
- **RAM**: 16GB system memory
- **Storage**: 50GB+ free space (for models and cache)
- **Python**: 3.10-3.12

### Recommended Requirements

- **VRAM**: 16GB+ for comfortable training
- **RAM**: 32GB+ system memory
- **Storage**: SSD with 100GB+ free space
- **CPU**: Multi-core processor for video preprocessing

### Memory Usage by Configuration

| Setup Type | VRAM | Batch Size | Frames | Resolution | Training Method |
|------------|------|------------|--------|------------|-----------------|
| 8GB Budget | 8GB  | 1          | 8      | 384x384    | LoRA           |
| 16GB Standard | 16GB | 2        | 16     | 512x512    | LoRA           |
| 24GB High-End | 24GB | 4        | 16     | 512x512    | LoRA/Fine-tune |
| 48GB+ Workstation | 48GB+ | 8   | 32     | 768x768    | Fine-tune      |

## GPU Platform Setup

### NVIDIA GPUs (CUDA)

OneTrainer automatically detects CUDA-capable NVIDIA GPUs. Ensure you have:

1. **NVIDIA Driver**: Latest stable driver
2. **CUDA Toolkit**: Installed automatically with PyTorch
3. **cuDNN**: Included with PyTorch installation

**Installation verification:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

### AMD GPUs (ROCm)

For AMD GPU acceleration, OneTrainer supports ROCm through PyTorch:

1. **Install ROCm**: Follow AMD's ROCm installation guide for your OS
2. **PyTorch ROCm**: Install PyTorch with ROCm support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
   ```
3. **Verify ROCm**: Check ROCm detection:
   ```bash
   python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}')"
   rocm-smi  # Check GPU status
   ```

**Supported AMD GPUs:**
- RX 6000 series (RDNA2)
- RX 7000 series (RDNA3)
- Radeon Pro series
- Instinct MI series

**ROCm Platform Notes:**
- ROCm support varies by Linux distribution
- Windows ROCm support is experimental
- Some features may have different performance characteristics compared to CUDA

### CPU-Only Training

For development or testing without GPU:

1. **Reduced Performance**: Expect significantly slower training
2. **Memory Requirements**: Higher system RAM usage
3. **Configuration**: Use CPU-optimized settings:
   ```json
   {
       "train_dtype": "FLOAT_32",
       "batch_size": 1,
       "frames": 4,
       "resolution": "256"
   }
   ```

## Data Preparation

### Video Requirements

**Supported Formats:**
- MP4 (recommended)
- AVI
- MOV
- WebM

**Quality Guidelines:**
- **Resolution**: 256x256 minimum, 1024x1024 maximum
- **Frame Rate**: 12-30 FPS (will be resampled to target_fps)
- **Duration**: 2-30 seconds per clip
- **Bitrate**: Higher quality source videos produce better results

### Data Organization

#### Option 1: Separate Caption Files
```
dataset/
├── videos/
│   ├── dancing_person.mp4
│   ├── sunset_ocean.mp4
│   └── city_traffic.mp4
└── captions/
    ├── dancing_person.txt
    ├── sunset_ocean.txt
    └── city_traffic.txt
```

#### Option 2: Filename-Based Captions
```
dataset/
├── a_person_dancing_in_a_studio.mp4
├── beautiful_sunset_over_ocean_waves.mp4
└── busy_city_street_with_traffic.mp4
```

#### Option 3: Single Caption File
```
dataset/
├── videos/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── video3.mp4
└── captions.txt  # Multiple lines, one per video
```

### Caption Writing Guidelines

**Good Captions:**
- Describe visual elements, actions, and scenes
- Be specific but concise (10-50 words)
- Focus on what's visible in the video
- Avoid temporal references ("first", "then", "finally")

**Examples:**
```
✅ "A person in red jacket walking through snowy forest with tall pine trees"
✅ "Ocean waves crashing against rocky coastline under cloudy sky"
✅ "Cat sitting on windowsill watching birds outside during daytime"

❌ "First a person walks, then they stop, finally they turn around"
❌ "This video shows someone doing something interesting"
❌ "Amazing footage of nature"
```

### Video Preprocessing

OneTrainer automatically handles:
- **Frame Extraction**: Samples frames according to your configuration
- **Resizing**: Scales videos to training resolution
- **Frame Rate**: Resamples to target FPS
- **Format Conversion**: Converts to training-compatible format

**Manual Preprocessing (Optional):**
If you want to preprocess videos manually:
```bash
# Resize and convert to MP4
ffmpeg -i input.mov -vf scale=512:512 -r 24 -c:v libx264 output.mp4

# Extract specific duration
ffmpeg -i input.mp4 -ss 00:00:05 -t 00:00:10 -c copy output.mp4
```

## Training Methods

### LoRA Training (Recommended)

LoRA (Low-Rank Adaptation) is the most efficient method for WAN 2.2 training:

**Advantages:**
- Low memory usage
- Fast training
- Small output files (50-200MB)
- Easy to share and combine

**Configuration:**
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

**When to Use:**
- Limited VRAM (8-16GB)
- Style adaptation
- Character/object training
- Quick experimentation

### Fine-Tuning

Full model fine-tuning trains all model parameters:

**Advantages:**
- Maximum flexibility
- Can learn complex patterns
- Better for dramatic style changes

**Disadvantages:**
- High memory usage
- Slower training
- Large output files (5-15GB)

**Configuration:**
```json
{
    "training_method": "FINE_TUNE",
    "batch_size": 1,
    "learning_rate": 0.0001,
    "transformer": {
        "train": true,
        "learning_rate": 0.0001
    },
    "text_encoder": {
        "train": true,
        "learning_rate": 0.00005
    }
}
```

**When to Use:**
- High-end hardware (24GB+ VRAM)
- Completely new domains
- Maximum quality requirements

### Embedding Training

Textual inversion creates new tokens for specific concepts:

**Advantages:**
- Very low memory usage
- Fast training
- Tiny output files (<1MB)
- Easy to use with any model

**Configuration:**
```json
{
    "training_method": "EMBEDDING",
    "batch_size": 4,
    "learning_rate": 0.001,
    "embedding_learning_rate": 0.005
}
```

**When to Use:**
- Learning specific objects/characters
- Minimal hardware requirements
- Quick concept learning

## Configuration Reference

### Core Training Settings

```json
{
    "model_type": "WAN_2_2",
    "base_model_name": "wan-ai/WAN_2_2",
    "training_method": "LORA",
    "batch_size": 2,
    "learning_rate": 0.0003,
    "epochs": 10,
    "resolution": "512",
    "frames": 16
}
```

### Video-Specific Configuration

```json
{
    "video_config": {
        "max_frames": 16,
        "frame_sample_strategy": "uniform",
        "target_fps": 24.0,
        "max_duration": 10.0,
        "temporal_consistency_weight": 1.0,
        "use_temporal_attention": true,
        "spatial_compression_ratio": 8,
        "temporal_compression_ratio": 4,
        "video_batch_size_multiplier": 0.5,
        "frame_dropout_probability": 0.0,
        "temporal_augmentation": false
    }
}
```

#### Video Configuration Parameters

**Frame Processing:**
- `max_frames`: Maximum frames per video clip (8-32)
- `frame_sample_strategy`: How to sample frames
  - `"uniform"`: Evenly spaced frames (recommended)
  - `"random"`: Random frame selection
  - `"keyframe"`: Prefer keyframes
- `target_fps`: Target frame rate for processing (12-30)
- `max_duration`: Maximum clip duration in seconds (5-30)

**Temporal Consistency:**
- `temporal_consistency_weight`: Strength of temporal loss (0.0-2.0)
- `use_temporal_attention`: Enable temporal attention (true/false)
- `frame_dropout_probability`: Regularization dropout (0.0-0.3)
- `temporal_augmentation`: Enable temporal augmentations (true/false)

**Memory Management:**
- `spatial_compression_ratio`: Spatial compression (4, 8, 16)
- `temporal_compression_ratio`: Temporal compression (2, 4, 8)
- `video_batch_size_multiplier`: Batch size adjustment (0.25-1.0)

### Memory Optimization Settings

```json
{
    "gradient_checkpointing": "CPU_OFFLOADED",
    "layer_offload_fraction": 0.5,
    "train_dtype": "BFLOAT_16",
    "transformer": {
        "weight_dtype": "FLOAT_8"
    },
    "text_encoder": {
        "weight_dtype": "FLOAT_8"
    }
}
```

## Memory Optimization

### Gradient Checkpointing

**Options:**
- `"NONE"`: No checkpointing (fastest, most memory)
- `"DEFAULT"`: Standard checkpointing
- `"CPU_OFFLOADED"`: Offload to CPU (slowest, least memory)

### Layer Offloading

Controls how much of the model is kept in VRAM:
- `0.0`: Keep all in VRAM (fastest)
- `0.5`: Offload 50% to system RAM
- `0.8`: Offload 80% to system RAM (memory-efficient)

### Data Types

**Training Precision:**
- `FLOAT_32`: Highest precision, most memory
- `BFLOAT_16`: Good balance (recommended)
- `FLOAT_16`: Lower precision, less memory

**Model Weights:**
- `FLOAT_32`: Full precision
- `BFLOAT_16`: Half precision
- `FLOAT_8`: 8-bit quantization
- `NFLOAT_4`: 4-bit quantization (most memory-efficient)

### Memory Optimization Strategies

#### 8GB VRAM Setup
```json
{
    "batch_size": 1,
    "frames": 8,
    "resolution": "384",
    "gradient_checkpointing": "CPU_OFFLOADED",
    "layer_offload_fraction": 0.8,
    "train_dtype": "BFLOAT_16",
    "transformer": {"weight_dtype": "NFLOAT_4"},
    "video_config": {
        "max_frames": 8,
        "spatial_compression_ratio": 16,
        "temporal_compression_ratio": 8,
        "video_batch_size_multiplier": 0.25
    }
}
```

#### 16GB VRAM Setup
```json
{
    "batch_size": 2,
    "frames": 16,
    "resolution": "512",
    "gradient_checkpointing": "CPU_OFFLOADED",
    "layer_offload_fraction": 0.5,
    "train_dtype": "BFLOAT_16",
    "transformer": {"weight_dtype": "FLOAT_8"},
    "video_config": {
        "max_frames": 16,
        "spatial_compression_ratio": 8,
        "temporal_compression_ratio": 4,
        "video_batch_size_multiplier": 0.5
    }
}
```

## Troubleshooting

### Common Issues

#### Out of Memory Errors

**Symptoms:**
- "CUDA out of memory" or "ROCm out of memory"
- Training crashes during forward pass

**Solutions:**
1. Reduce batch size to 1
2. Enable CPU offloading: `"gradient_checkpointing": "CPU_OFFLOADED"`
3. Increase layer offloading: `"layer_offload_fraction": 0.8`
4. Reduce frames: `"frames": 8` or `"max_frames": 8`
5. Lower resolution: `"resolution": "384"`
6. Use quantization: `"weight_dtype": "NFLOAT_4"`

#### Poor Video Quality

**Symptoms:**
- Blurry or inconsistent frames
- Poor temporal consistency
- Artifacts in generated videos

**Solutions:**
1. Increase temporal consistency weight: `"temporal_consistency_weight": 1.5`
2. Enable temporal attention: `"use_temporal_attention": true`
3. Use uniform frame sampling: `"frame_sample_strategy": "uniform"`
4. Reduce frame dropout: `"frame_dropout_probability": 0.0`
5. Check source video quality
6. Verify caption quality

#### Slow Training

**Symptoms:**
- Very low steps per second
- Long epoch times

**Solutions:**
1. Increase dataloader threads: `"dataloader_threads": 2`
2. Use faster storage (SSD)
3. Enable mixed precision: `"train_dtype": "BFLOAT_16"`
4. Reduce video preprocessing overhead
5. Check CPU usage during training

#### ROCm-Specific Issues

**Common ROCm Problems:**
1. **ROCm not detected**: Verify ROCm installation and PyTorch ROCm build
2. **Performance issues**: Some operations may be slower than CUDA
3. **Memory management**: ROCm may have different memory patterns

**ROCm Troubleshooting:**
```bash
# Check ROCm installation
rocm-smi
rocminfo

# Verify PyTorch ROCm
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.hip)"

# Monitor GPU usage
watch -n 1 rocm-smi
```

### Performance Monitoring

#### NVIDIA GPUs
```bash
# Monitor VRAM usage
watch -n 1 nvidia-smi

# Detailed monitoring
nvidia-smi dmon -s pucvmet -d 1
```

#### AMD GPUs
```bash
# Monitor GPU usage
watch -n 1 rocm-smi

# Detailed monitoring
rocm-smi -a
```

#### System Resources
```bash
# Monitor system memory and CPU
htop

# Monitor disk I/O
iotop
```

## Example Workflows

### Workflow 1: Character Training (8GB VRAM)

**Goal:** Train a LoRA to generate videos of a specific character

**Setup:**
1. Collect 20-50 videos of the character (5-10 seconds each)
2. Write descriptive captions focusing on the character
3. Use the 8GB preset as starting point

**Configuration:**
```json
{
    "base_model_name": "wan-ai/WAN_2_2",
    "training_method": "LORA",
    "batch_size": 1,
    "learning_rate": 0.0003,
    "epochs": 15,
    "resolution": "384",
    "frames": 8,
    "video_config": {
        "max_frames": 8,
        "frame_sample_strategy": "uniform",
        "target_fps": 12.0,
        "max_duration": 8.0,
        "temporal_consistency_weight": 1.2
    }
}
```

### Workflow 2: Style Transfer (16GB VRAM)

**Goal:** Adapt WAN 2.2 to a specific visual style

**Setup:**
1. Collect 50-100 videos in the target style
2. Focus captions on style elements (lighting, colors, composition)
3. Use standard LoRA preset

**Configuration:**
```json
{
    "base_model_name": "wan-ai/WAN_2_2",
    "training_method": "LORA",
    "batch_size": 2,
    "learning_rate": 0.0002,
    "epochs": 20,
    "resolution": "512",
    "frames": 16,
    "video_config": {
        "max_frames": 16,
        "frame_sample_strategy": "uniform",
        "target_fps": 24.0,
        "temporal_consistency_weight": 1.0,
        "temporal_augmentation": true
    }
}
```

### Workflow 3: Domain Adaptation (24GB+ VRAM)

**Goal:** Fine-tune for a completely new domain (e.g., medical videos)

**Setup:**
1. Large dataset (200+ videos)
2. Comprehensive captions
3. Fine-tuning approach

**Configuration:**
```json
{
    "base_model_name": "wan-ai/WAN_2_2",
    "training_method": "FINE_TUNE",
    "batch_size": 2,
    "learning_rate": 0.0001,
    "epochs": 30,
    "resolution": "512",
    "frames": 16,
    "transformer": {
        "train": true,
        "learning_rate": 0.0001
    },
    "text_encoder": {
        "train": true,
        "learning_rate": 0.00005
    }
}
```

### Workflow 4: Quick Concept Learning

**Goal:** Learn a specific object or concept quickly

**Setup:**
1. 10-20 videos of the concept
2. Simple, focused captions
3. Embedding training

**Configuration:**
```json
{
    "base_model_name": "wan-ai/WAN_2_2",
    "training_method": "EMBEDDING",
    "batch_size": 4,
    "learning_rate": 0.001,
    "embedding_learning_rate": 0.005,
    "epochs": 50,
    "resolution": "512",
    "frames": 12
}
```

## Advanced Tips

### Data Augmentation

Enable temporal augmentation for better generalization:
```json
{
    "video_config": {
        "temporal_augmentation": true,
        "frame_dropout_probability": 0.1
    }
}
```

### Multi-Resolution Training

Train on multiple resolutions simultaneously:
```json
{
    "resolution": "384,512,640"
}
```

### Learning Rate Scheduling

Use different learning rates for different components:
```json
{
    "learning_rate": 0.0003,
    "transformer": {
        "learning_rate": 0.0003
    },
    "text_encoder": {
        "learning_rate": 0.0001
    }
}
```

### Sampling During Training

Monitor progress with regular sampling:
```json
{
    "sample_after": 100,
    "sample_prompts": [
        "a person walking in a park",
        "ocean waves at sunset",
        "city street at night"
    ]
}
```

This comprehensive guide should help you successfully train WAN 2.2 models with OneTrainer. For additional support, check the OneTrainer Discord community or GitHub discussions.