# WAN 2.2 Video Dataset Remote Testing Instructions

## 🎯 Problem Summary
The WAN 2.2 training shows:
- `Group0_SafeCollectPaths_1 wrapped length: 10` (files found)
- `Group10_AspectBatchSorting_0 length() returned: 0` (final dataset empty)
- `step: 0it [00:00, ?it/s]` (no training steps)

This means videos are being filtered out somewhere in the pipeline.

## ✅ Debug Logging Enabled
I've enabled comprehensive debug logging in the video pipeline:
- **SafeLoadVideo**: Will show detailed video loading attempts
- **Video Validation**: Will show why videos are rejected
- **Pipeline Processing**: Will trace video processing steps

## 📋 Remote Testing Commands

### 1. Copy Debug Files to Remote Host
```bash
# Copy these files to your remote host
scp -P 3701 debug_video_dataset_remote.py 10.1.1.12:/workspace/OneTrainer-WAN/
scp -P 3701 remote_debug_commands.sh 10.1.1.12:/workspace/OneTrainer-WAN/
```

### 2. Run Comprehensive Video Analysis
```bash
# SSH to remote host
ssh -p 3701 10.1.1.12

# Navigate to OneTrainer-WAN directory
cd /workspace/OneTrainer-WAN

# Run comprehensive video debugging
python debug_video_dataset_remote.py

# Or run the shell script for full analysis
chmod +x remote_debug_commands.sh
./remote_debug_commands.sh
```

### 3. Quick Video Check Commands
```bash
# Check if videos exist
ls -la /workspace/input/training/cube/

# Check video properties
for video in /workspace/input/training/cube/*.mp4; do
    echo "=== $video ==="
    ffprobe -v quiet -show_format -show_streams "$video" | grep -E "(duration|width|height|codec_name)"
done

# Test video loading with OpenCV
python3 -c "
import cv2
import os
for video in os.listdir('/workspace/input/training/cube'):
    if video.endswith('.mp4'):
        cap = cv2.VideoCapture(f'/workspace/input/training/cube/{video}')
        if cap.isOpened():
            print(f'{video}: OK - {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames')
        else:
            print(f'{video}: FAILED to open')
        cap.release()
"
```

### 4. Run Training with Debug Logging
```bash
# Run training - you'll now see detailed debug output
python train.py

# Look for these debug messages:
# - "DEBUG SAFE_LOAD_VIDEO: Processing item X"
# - "DEBUG VIDEO VALIDATION: Processing /path/to/video"
# - "DEBUG SAFE_LOAD_VIDEO SUCCESS:" or "DEBUG SAFE_LOAD_VIDEO ERROR:"
```

### 5. Create Test Video (if needed)
```bash
# Create a known-good test video
ffmpeg -y -f lavfi -i testsrc=duration=5:size=512x512:rate=30 \
    -c:v libx264 -pix_fmt yuv420p -preset fast \
    /workspace/input/training/cube/test_debug.mp4

# Verify it was created
ls -la /workspace/input/training/cube/test_debug.mp4
ffprobe /workspace/input/training/cube/test_debug.mp4
```

## 🔍 What to Look For

### Expected Debug Output During Training:
```
DEBUG: Video validation enabled - will show detailed error messages
DEBUG SAFE_LOAD_VIDEO: Processing item 0, variation 0
DEBUG VIDEO VALIDATION: Processing /workspace/input/training/cube/video1.mp4
```

### Success Indicators:
```
DEBUG SAFE_LOAD_VIDEO SUCCESS: video1.mp4 loaded with shape torch.Size([8, 3, 384, 384])
DEBUG VIDEO SUCCESS: /workspace/input/training/cube/video1.mp4 shape=(8, 3, 384, 384)
```

### Error Indicators:
```
DEBUG SAFE_LOAD_VIDEO ERROR: LoadVideo returned None for item 0
DEBUG VIDEO ERROR: No 'video' key in sample for /path/to/video
DEBUG SAFE_LOAD_VIDEO EXCEPTION: LoadVideo failed for item 0: [specific error]
```

## 🛠️ Common Issues and Fixes

### Issue 1: Videos Too Short
**Symptom**: `DEBUG VIDEO ERROR: Video too short`
**Fix**: Ensure videos are at least 2-3 seconds long

### Issue 2: Invalid Video Format
**Symptom**: `DEBUG SAFE_LOAD_VIDEO ERROR: LoadVideo returned None`
**Fix**: Convert videos to standard MP4 with H.264 codec

### Issue 3: Resolution Too Low
**Symptom**: `DEBUG VIDEO ERROR: Resolution too low`
**Fix**: Ensure videos are at least 256x256 pixels

### Issue 4: Frame Count Issues
**Symptom**: Videos load but get filtered later
**Fix**: Reduce `frames=8` to `frames=4` or `frames=2` in training config

## 📊 Expected Results

After running the debug commands, you should see:
1. **Video files exist** in `/workspace/input/training/cube/`
2. **Video properties** showing duration, resolution, codec
3. **Debug output** during training showing exactly where videos fail
4. **Specific error messages** explaining why videos are rejected

## 🎯 Next Steps Based on Results

### If videos load successfully:
- The issue is in later pipeline stages (VideoFrameSampler, AspectBucketing)
- Check frame sampling requirements
- Verify aspect ratio handling

### If videos fail to load:
- Check video format compatibility
- Verify file permissions and accessibility
- Test with the generated test video

### If no videos found:
- Copy videos to the correct directory
- Update concepts.json configuration
- Verify directory permissions

## 📞 Reporting Results

When reporting back, please include:
1. Output from `debug_video_dataset_remote.py`
2. Any debug messages from training run
3. Video file properties (duration, resolution, codec)
4. Specific error messages if any

This will help identify the exact cause of the empty dataset issue.