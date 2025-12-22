# WAN 2.2 Troubleshooting Guide

This guide covers common issues and solutions when training WAN 2.2 models with OneTrainer.

## Table of Contents

- [Memory Issues](#memory-issues)
- [Video Data Problems](#video-data-problems)
- [Training Quality Issues](#training-quality-issues)
- [Performance Problems](#performance-problems)
- [GPU Platform Issues](#gpu-platform-issues)
- [Configuration Errors](#configuration-errors)
- [Sampling Issues](#sampling-issues)

## Memory Issues

### Out of Memory (OOM) Errors

**Symptoms:**
- "CUDA out of memory" or "ROCm out of memory"
- Training crashes during forward pass
- System becomes unresponsive

**Solutions (in order of effectiveness):**

1. **Reduce batch size:**
   ```json
   {
       "batch_size": 1
   }
   ```

2. **Enable CPU offloading:**
   ```json
   {
       "gradient_checkpointing": "CPU_OFFLOADED",
       "layer_offload_fraction": 0.8
   }
   ```

3. **Reduce video parameters:**
   ```json
   {
       "frames": 8,
       "resolution": "384",
       "video_config": {
           "max_frames": 8,
           "spatial_compression_ratio": 16,
           "temporal_compression_ratio": 8,
           "video_batch_size_multiplier": 0.25
       }
   }
   ```

4. **Use quantization:**
   ```json
   {
       "transformer": {"weight_dtype": "NFLOAT_4"},
       "text_encoder": {"weight_dtype": "NFLOAT_4"}
   }
   ```

5. **Switch to LoRA training:**
   ```json
   {
       "training_method": "LORA"
   }
   ```

### Memory Leaks

**Symptoms:**
- Memory usage increases over time
- Training becomes slower after several epochs
- System runs out of memory after hours of training

**Solutions:**
1. **Restart training periodically** using backups
2. **Reduce dataloader threads:**
   ```json
   {
       "dataloader_threads": 1
   }
   ```
3. **Clear cache between epochs:**
   ```json
   {
       "cache_latents": false
   }
   ```

## Video Data Problems

### Video Loading Errors

**Symptoms:**
- "Could not load video" errors
- Training fails during data loading
- Corrupted or missing frames

**Solutions:**

1. **Check video format compatibility:**
   - Use MP4 with H.264 codec (most compatible)
   - Avoid exotic codecs or containers
   - Convert problematic videos:
     ```bash
     ffmpeg -i input.mov -c:v libx264 -c:a aac output.mp4
     ```

2. **Verify video integrity:**
   ```bash
   ffmpeg -v error -i video.mp4 -f null -
   ```

3. **Check file permissions:**
   - Ensure OneTrainer can read video files
   - Avoid network drives for video storage

### Frame Sampling Issues

**Symptoms:**
- Inconsistent frame counts
- Poor temporal consistency
- Training instability

**Solutions:**

1. **Use uniform sampling:**
   ```json
   {
       "video_config": {
           "frame_sample_strategy": "uniform"
       }
   }
   ```

2. **Standardize video properties:**
   - Convert all videos to same FPS:
     ```bash
     ffmpeg -i input.mp4 -r 24 -c:v libx264 output.mp4
     ```
   - Ensure consistent durations

3. **Adjust frame parameters:**
   ```json
   {
       "video_config": {
           "max_frames": 16,
           "target_fps": 24.0,
           "max_duration": 10.0
       }
   }
   ```

### Caption Synchronization

**Symptoms:**
- Mismatched video-caption pairs
- Training on wrong descriptions
- Poor text-video alignment

**Solutions:**

1. **Verify naming consistency:**
   ```
   video1.mp4 → video1.txt
   video2.mp4 → video2.txt
   ```

2. **Check caption file encoding:**
   - Use UTF-8 encoding
   - Avoid special characters that might cause parsing issues

3. **Validate caption content:**
   - Ensure captions describe visual content
   - Remove temporal references ("first", "then", "finally")

## Training Quality Issues

### Poor Temporal Consistency

**Symptoms:**
- Flickering between frames
- Inconsistent object appearance
- Temporal artifacts

**Solutions:**

1. **Increase temporal consistency weight:**
   ```json
   {
       "video_config": {
           "temporal_consistency_weight": 1.5
       }
   }
   ```

2. **Enable temporal attention:**
   ```json
   {
       "video_config": {
           "use_temporal_attention": true
       }
   }
   ```

3. **Reduce frame dropout:**
   ```json
   {
       "video_config": {
           "frame_dropout_probability": 0.0
       }
   }
   ```

4. **Use higher quality source videos:**
   - Ensure source videos have good temporal consistency
   - Avoid heavily compressed or low-quality videos

### Blurry or Low-Quality Output

**Symptoms:**
- Generated videos lack detail
- Blurry or soft appearance
- Loss of fine details

**Solutions:**

1. **Increase resolution:**
   ```json
   {
       "resolution": "512"
   }
   ```

2. **Reduce compression ratios:**
   ```json
   {
       "video_config": {
           "spatial_compression_ratio": 4,
           "temporal_compression_ratio": 2
       }
   }
   ```

3. **Improve source data quality:**
   - Use high-resolution source videos
   - Ensure good lighting and focus in training data

4. **Adjust learning rate:**
   ```json
   {
       "learning_rate": 0.0001
   }
   ```

### Training Instability

**Symptoms:**
- Loss spikes or NaN values
- Training diverges
- Inconsistent results between runs

**Solutions:**

1. **Lower learning rate:**
   ```json
   {
       "learning_rate": 0.0001,
       "transformer": {"learning_rate": 0.0001},
       "text_encoder": {"learning_rate": 0.00005}
   }
   ```

2. **Use gradient clipping:**
   ```json
   {
       "gradient_clipping": 1.0
   }
   ```

3. **Enable EMA:**
   ```json
   {
       "ema": {
           "enabled": true,
           "decay": 0.999
       }
   }
   ```

## Performance Problems

### Slow Training Speed

**Symptoms:**
- Very low steps per second
- Long epoch times
- High CPU usage during training

**Solutions:**

1. **Optimize data loading:**
   ```json
   {
       "dataloader_threads": 2,
       "cache_latents": true,
       "cache_text_encoder_outputs": true
   }
   ```

2. **Use faster storage:**
   - Move dataset to SSD
   - Use local storage instead of network drives

3. **Enable mixed precision:**
   ```json
   {
       "train_dtype": "BFLOAT_16"
   }
   ```

4. **Reduce video preprocessing:**
   - Pre-process videos to target resolution
   - Use consistent frame rates in source data

### High Memory Usage During Data Loading

**Symptoms:**
- System RAM usage spikes
- Slow data loading
- System becomes unresponsive

**Solutions:**

1. **Reduce dataloader threads:**
   ```json
   {
       "dataloader_threads": 1
   }
   ```

2. **Lower video batch multiplier:**
   ```json
   {
       "video_config": {
           "video_batch_size_multiplier": 0.25
       }
   }
   ```

3. **Reduce frame count:**
   ```json
   {
       "frames": 8,
       "video_config": {"max_frames": 8}
   }
   ```

## GPU Platform Issues

### NVIDIA CUDA Issues

**Common Problems:**
1. **CUDA out of memory**: See [Memory Issues](#memory-issues)
2. **CUDA version mismatch**: Reinstall PyTorch with correct CUDA version
3. **Driver issues**: Update NVIDIA drivers

**Diagnostics:**
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"
```

### AMD ROCm Issues

**Common Problems:**

1. **ROCm not detected:**
   ```bash
   # Check ROCm installation
   rocm-smi
   rocminfo
   
   # Verify PyTorch ROCm
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Performance issues:**
   - Some operations may be slower than CUDA
   - Use ROCm-optimized configurations:
     ```json
     {
         "batch_size": 1,
         "resolution": "448",
         "video_config": {
             "spatial_compression_ratio": 8,
             "temporal_compression_ratio": 6
         }
     }
     ```

3. **Memory management differences:**
   - ROCm may have different memory patterns
   - Use conservative memory settings:
     ```json
     {
         "layer_offload_fraction": 0.6,
         "video_config": {
             "video_batch_size_multiplier": 0.4
         }
     }
     ```

**ROCm Troubleshooting Steps:**

1. **Verify ROCm installation:**
   ```bash
   /opt/rocm/bin/rocminfo
   ```

2. **Check GPU compatibility:**
   ```bash
   rocm-smi --showproductname
   ```

3. **Install correct PyTorch version:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
   ```

4. **Set environment variables if needed:**
   ```bash
   export HSA_OVERRIDE_GFX_VERSION=10.3.0  # For unsupported GPUs
   ```

### CPU-Only Issues

**Common Problems:**
1. **Extremely slow training**: Expected behavior, use minimal settings
2. **High memory usage**: Use FLOAT_32 and minimal batch sizes
3. **Threading issues**: Set `dataloader_threads: 1`

**CPU-Optimized Configuration:**
```json
{
    "batch_size": 1,
    "frames": 4,
    "resolution": "256",
    "train_dtype": "FLOAT_32",
    "gradient_checkpointing": "NONE",
    "dataloader_threads": 1,
    "video_config": {
        "max_frames": 4,
        "use_temporal_attention": false,
        "spatial_compression_ratio": 16
    }
}
```

## Configuration Errors

### Invalid Model Type

**Error:** `Unknown model type: WAN_2_2`

**Solution:**
- Ensure you're using the latest OneTrainer version
- Verify WAN 2.2 support is properly installed
- Check model type spelling: `"model_type": "WAN_2_2"`

### Missing Video Configuration

**Error:** Video-specific parameters not recognized

**Solution:**
Add video_config section:
```json
{
    "video_config": {
        "max_frames": 16,
        "frame_sample_strategy": "uniform",
        "target_fps": 24.0
    }
}
```

### Incompatible Training Method

**Error:** Training method not supported for WAN 2.2

**Solution:**
Use supported methods:
```json
{
    "training_method": "LORA"  // or "FINE_TUNE" or "EMBEDDING"
}
```

## Sampling Issues

### Sampling Failures

**Symptoms:**
- Sampling crashes during training
- Generated videos are corrupted
- Sampling takes extremely long

**Solutions:**

1. **Reduce sampling complexity:**
   ```json
   {
       "sample_steps": 20,
       "sample_cfg_scale": 7.0
   }
   ```

2. **Use simpler prompts:**
   - Avoid very long or complex prompts
   - Test with basic prompts first

3. **Check sampling configuration:**
   ```json
   {
       "sample_after": 100,
       "sample_prompts": [
           "a person walking",
           "ocean waves"
       ]
   }
   ```

### Video Output Issues

**Symptoms:**
- Generated videos won't play
- Corrupted video files
- Wrong format or codec

**Solutions:**

1. **Check output format:**
   - Ensure MP4 output is selected
   - Verify codec compatibility

2. **Validate video files:**
   ```bash
   ffmpeg -v error -i generated_video.mp4 -f null -
   ```

3. **Convert if necessary:**
   ```bash
   ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4
   ```

## Getting Help

If you continue to experience issues:

1. **Check the logs:** Look for error messages in the OneTrainer console output
2. **Generate debug report:** Use `export_debug.bat` or `./run-cmd.sh generate_debug_report`
3. **Join the community:** Ask for help in the [OneTrainer Discord](https://discord.gg/KwgcQd5scF)
4. **Report bugs:** Create an issue on [GitHub](https://github.com/Nerogar/OneTrainer/issues) with debug information

## Common Error Messages

### "Could not load model"
- Check model path and format
- Verify internet connection for Hugging Face models
- Ensure sufficient disk space

### "Video preprocessing failed"
- Check video file integrity
- Verify supported format (MP4, AVI, MOV, WebM)
- Ensure sufficient disk space for temporary files

### "Temporal consistency loss is NaN"
- Reduce temporal_consistency_weight
- Check for corrupted video data
- Lower learning rate

### "CUDA/ROCm initialization failed"
- Update GPU drivers
- Reinstall PyTorch with correct GPU support
- Check GPU compatibility

This troubleshooting guide should help resolve most common WAN 2.2 training issues. For additional support, consult the OneTrainer community resources.