# WAN 2.2 UI Testing Guide

## 🚀 How to Test WAN 2.2 from the OneTrainer GUI

This guide walks you through testing the complete WAN 2.2 implementation using the OneTrainer graphical user interface.

## 📋 Prerequisites

### 1. Environment Setup
```bash
# Install OneTrainer dependencies
pip install torch torchvision torchaudio
pip install transformers diffusers accelerate
pip install opencv-python pillow numpy

# Clone and setup OneTrainer-WAN
git clone https://github.com/celesrenata/OneTrainer-WAN.git
cd OneTrainer-WAN
```

### 2. Launch OneTrainer GUI
```bash
# Start the GUI
python scripts/train_ui.py
```

## 🎯 Step-by-Step UI Testing

### Step 1: Model Selection
1. **Open OneTrainer GUI**
2. **Navigate to Model Tab**
3. **Select Model Type**:
   - Click on the "Model Type" dropdown
   - Look for **"WAN_2_2"** in the list
   - Select **WAN_2_2**

✅ **Expected Result**: WAN_2_2 should appear in the model type dropdown

### Step 2: Load Training Presets
1. **Go to Training Tab**
2. **Load Preset**:
   - Click "Load Preset" button
   - Navigate to `training_presets/` folder
   - Choose one of the WAN 2.2 presets:
     - `#wan 2.2 Finetune.json` - Full fine-tuning
     - `#wan 2.2 LoRA.json` - LoRA training
     - `#wan 2.2 LoRA 8GB.json` - LoRA for 8GB VRAM
     - `#wan 2.2 Embedding.json` - Textual inversion

✅ **Expected Result**: Preset loads successfully and populates training parameters

### Step 3: Configure Video Settings
1. **Navigate to Video Config Tab** (new tab for WAN 2.2)
2. **Set Video Parameters**:
   - **Target Frames**: 16 (default)
   - **Frame Sample Strategy**: "uniform", "random", or "keyframe"
   - **Min Resolution**: 256x256
   - **Max Resolution**: 1024x1024
   - **Max Duration**: 10.0 seconds
   - **Temporal Consistency Weight**: 1.0

✅ **Expected Result**: Video-specific parameters are available and configurable

### Step 4: Configure Data Source
1. **Go to Data Tab**
2. **Set Training Data**:
   - **Concept List**: Point to a folder containing:
     - Video files (`.mp4`, `.avi`, `.mov`, `.webm`)
     - Corresponding text files (`.txt`) with descriptions
   - **Example structure**:
     ```
     training_data/
     ├── video_001.mp4
     ├── video_001.txt
     ├── video_002.mp4
     ├── video_002.txt
     └── ...
     ```

✅ **Expected Result**: Video files are detected and validated

### Step 5: Model Configuration
1. **Return to Model Tab**
2. **Configure Model Settings**:
   - **Base Model**: Leave empty for new training or specify WAN 2.2 model path
   - **Output Model Format**: Choose "diffusers" (recommended)
   - **Output Directory**: Set where to save the trained model

✅ **Expected Result**: Model configuration accepts WAN 2.2 settings

### Step 6: Training Configuration
1. **Go to Training Tab**
2. **Configure Training Parameters**:
   - **Batch Size**: 1 (recommended for video)
   - **Learning Rate**: 1e-4 (fine-tune) or 1e-3 (embedding)
   - **Max Epochs**: 10
   - **Gradient Accumulation Steps**: 4
   - **Mixed Precision**: Enable for memory efficiency

✅ **Expected Result**: Training parameters are appropriate for video training

### Step 7: Advanced Settings (Optional)
1. **Check Advanced Options**:
   - **LoRA Settings** (if using LoRA preset):
     - LoRA Rank: 16
     - LoRA Alpha: 32
   - **Embedding Settings** (if using embedding preset):
     - Embedding Learning Rate: 1e-3
     - Train Text Encoder: Disabled
   - **Memory Optimization**:
     - Gradient Checkpointing: Enabled
     - Cache Latents: Enabled for speed

✅ **Expected Result**: Advanced settings are available and properly configured

### Step 8: Validation Check
1. **Review Configuration Summary**
2. **Check for Errors**:
   - Look for any red error messages
   - Ensure all required fields are filled
   - Verify video data is properly detected

✅ **Expected Result**: No validation errors, all settings are valid

### Step 9: Start Training (Test Run)
1. **Click "Start Training"**
2. **Monitor Progress**:
   - Training should start without errors
   - Check console output for WAN 2.2 specific messages
   - Monitor GPU/CPU usage
   - Watch for video processing logs

✅ **Expected Result**: Training starts successfully with WAN 2.2 model

### Step 10: Sampling Test (During Training)
1. **Enable Sampling** (if configured):
   - Set sample prompts
   - Configure sampling frequency
   - Set output format (MP4 recommended)

✅ **Expected Result**: Video samples are generated during training

## 🔍 What to Look For

### ✅ Success Indicators
- **Model Type**: WAN_2_2 appears in dropdown
- **Presets**: WAN 2.2 presets load correctly
- **Video Tab**: Video configuration tab is available
- **Video Formats**: MP4, AVI, MOV, WebM files are accepted
- **Training**: Starts without import errors
- **Memory**: Efficient memory usage with video data
- **Sampling**: Generates video outputs

### ❌ Potential Issues
- **Import Errors**: Missing dependencies (install PyTorch, etc.)
- **Model Type Missing**: WAN_2_2 not in dropdown (check ModelType enum)
- **Video Tab Missing**: VideoConfigTab not loaded (check UI integration)
- **Preset Errors**: JSON format issues (check preset files)
- **Memory Issues**: Insufficient VRAM (try LoRA or reduce batch size)

## 🛠️ Troubleshooting

### Issue: WAN_2_2 Not in Model Type Dropdown
**Solution**:
```bash
# Check ModelType enum
python -c "from modules.util.enum.ModelType import ModelType; print(ModelType.WAN_2_2)"
```

### Issue: Video Config Tab Missing
**Solution**:
- Check `modules/ui/VideoConfigTab.py` exists
- Verify UI integration in `modules/ui/TrainingTab.py`

### Issue: Training Presets Don't Load
**Solution**:
- Verify JSON format in `training_presets/` folder
- Check file permissions

### Issue: Video Files Not Recognized
**Solution**:
- Ensure video files have supported extensions (`.mp4`, `.avi`, `.mov`, `.webm`)
- Check corresponding `.txt` files exist
- Verify file permissions

### Issue: Out of Memory Errors
**Solutions**:
1. **Use LoRA preset**: `#wan 2.2 LoRA 8GB.json`
2. **Reduce batch size**: Set to 1
3. **Enable gradient checkpointing**: In advanced settings
4. **Reduce video resolution**: Lower max resolution
5. **Reduce target frames**: Use 8 or 12 instead of 16

## 📊 Performance Testing

### Memory Usage Test
1. **Monitor GPU Memory**:
   ```bash
   # In another terminal
   watch -n 1 nvidia-smi
   ```

2. **Test Different Configurations**:
   - Fine-tuning: Highest memory usage
   - LoRA: Medium memory usage
   - Embedding: Lowest memory usage

### Speed Test
1. **Time Training Steps**:
   - Note time per epoch
   - Compare with other model types
   - Monitor data loading speed

### Quality Test
1. **Generate Samples**:
   - Use various prompts
   - Check video quality
   - Verify temporal consistency

## 🎥 Sample Data for Testing

### Create Test Dataset
```bash
# Create test directory
mkdir test_videos

# Add sample videos (you'll need actual video files)
# Each video should have a corresponding .txt file with description
```

### Example Text Descriptions
```
video_001.txt: "A person walking through a forest path"
video_002.txt: "Ocean waves crashing on rocks"
video_003.txt: "A cat playing with a ball of yarn"
```

## 📈 Expected Performance

### Training Times (Approximate)
- **Fine-tuning**: 2-5 minutes per epoch (16GB VRAM)
- **LoRA**: 1-3 minutes per epoch (8GB VRAM)
- **Embedding**: 30 seconds - 1 minute per epoch (6GB VRAM)

### Memory Usage (Approximate)
- **Fine-tuning**: 12-16GB VRAM
- **LoRA**: 6-10GB VRAM
- **Embedding**: 4-8GB VRAM

## 🎯 Success Criteria

Your WAN 2.2 implementation is working correctly if:

1. ✅ **Model Selection**: WAN_2_2 appears in model type dropdown
2. ✅ **Preset Loading**: All 4 WAN 2.2 presets load without errors
3. ✅ **Video Configuration**: Video config tab is available and functional
4. ✅ **Data Loading**: Video files are detected and processed
5. ✅ **Training Start**: Training begins without import/initialization errors
6. ✅ **Memory Management**: Training runs within expected memory limits
7. ✅ **Sample Generation**: Video samples are generated (if enabled)
8. ✅ **Model Saving**: Trained model saves successfully

## 🚀 Next Steps After Successful Testing

1. **Train with Real Data**: Use your own video dataset
2. **Experiment with Settings**: Try different parameters
3. **Generate Videos**: Use the trained model for inference
4. **Share Results**: Document your training results
5. **Contribute**: Report any issues or improvements

## 📞 Getting Help

If you encounter issues:

1. **Check Documentation**: `docs/WAN22Training.md`
2. **Review Troubleshooting**: `docs/WAN22Troubleshooting.md`
3. **Run Validation**: `python test_wan_final_validation.py`
4. **Check Logs**: Look at console output for error messages
5. **Create Issue**: Report bugs on GitHub

---

**Happy Training with WAN 2.2!** 🎉