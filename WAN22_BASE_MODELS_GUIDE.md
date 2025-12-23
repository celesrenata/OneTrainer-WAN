# WAN 2.2 Base Models Guide

## 🎯 **Quick Answer: Recommended WAN 2.2 Base Models**

### **Primary Recommendation:**
```
genmo/mochi-1-preview
```

### **Alternative Options:**
```
stabilityai/stable-video-diffusion-img2vid-xt
microsoft/DiT-XL-2-256
facebook/AnimateDiff
```

---

## 📋 **Complete WAN 2.2 Base Model Options**

### **1. Mochi-1 (Recommended)**
- **Model ID**: `genmo/mochi-1-preview`
- **Type**: Video generation transformer
- **Resolution**: Up to 1024x1024
- **Frames**: 16-24 frames
- **Best for**: High-quality video generation
- **Memory**: 16GB+ VRAM recommended

### **2. Stable Video Diffusion**
- **Model ID**: `stabilityai/stable-video-diffusion-img2vid-xt`
- **Type**: Video diffusion model
- **Resolution**: 576x1024
- **Frames**: 25 frames
- **Best for**: Image-to-video generation
- **Memory**: 12GB+ VRAM

### **3. DiT (Diffusion Transformer)**
- **Model ID**: `microsoft/DiT-XL-2-256`
- **Type**: Transformer-based diffusion
- **Resolution**: 256x256 (can be upscaled)
- **Best for**: Learning transformer architecture
- **Memory**: 8GB+ VRAM

### **4. AnimateDiff**
- **Model ID**: `guoyww/animatediff-motion-adapter-v1-5-2`
- **Type**: Motion adapter for video
- **Best for**: Adding motion to static images
- **Memory**: 10GB+ VRAM

---

## 🚀 **How to Use in OneTrainer GUI**

### **Step 1: Set Base Model Path**
1. Open OneTrainer GUI
2. Go to **Model Tab**
3. Set **Model Type**: `WAN_2_2`
4. Set **Base Model**: `genmo/mochi-1-preview`

### **Step 2: Alternative - Local Model**
If you have a local WAN 2.2 model:
```
/path/to/your/wan22/model/directory
```

### **Step 3: Model Format**
- **Preferred**: Hugging Face diffusers format
- **Supported**: SafeTensors, PyTorch checkpoints
- **Structure**: Should contain `transformer/`, `vae/`, `text_encoder/`, `scheduler/`

---

## 📁 **Expected Model Structure**

A proper WAN 2.2 model should have this structure:
```
model_directory/
├── model_index.json
├── transformer/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── text_encoder/
│   ├── config.json
│   └── pytorch_model.safetensors
├── tokenizer/
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── scheduler/
    └── scheduler_config.json
```

---

## 🔧 **Troubleshooting Base Model Issues**

### **Error: "Model not found"**
**Solutions:**
1. **Check internet connection** (for Hugging Face models)
2. **Verify model ID spelling**: `genmo/mochi-1-preview`
3. **Try alternative model**: `stabilityai/stable-video-diffusion-img2vid-xt`
4. **Check Hugging Face access**: Some models require authentication

### **Error: "Invalid model format"**
**Solutions:**
1. **Use diffusers format**: Models with `model_index.json`
2. **Convert checkpoint**: Use `convert_original_stable_diffusion_to_diffusers.py`
3. **Check model structure**: Ensure all required components exist

### **Error: "Out of memory"**
**Solutions:**
1. **Use smaller model**: Try `microsoft/DiT-XL-2-256`
2. **Reduce batch size**: Set to 1
3. **Enable gradient checkpointing**: In advanced settings
4. **Use LoRA training**: Instead of full fine-tuning

---

## 🎯 **Recommended Setup by GPU Memory**

### **24GB+ VRAM (RTX 4090, A100)**
- **Model**: `genmo/mochi-1-preview`
- **Training**: Full fine-tuning
- **Batch Size**: 2-4
- **Resolution**: 1024x1024
- **Frames**: 24

### **16GB VRAM (RTX 4080, RTX 3090)**
- **Model**: `stabilityai/stable-video-diffusion-img2vid-xt`
- **Training**: Full fine-tuning or LoRA
- **Batch Size**: 1-2
- **Resolution**: 768x768
- **Frames**: 16

### **12GB VRAM (RTX 4070 Ti, RTX 3080)**
- **Model**: `microsoft/DiT-XL-2-256`
- **Training**: LoRA recommended
- **Batch Size**: 1
- **Resolution**: 512x512
- **Frames**: 16

### **8GB VRAM (RTX 4060 Ti, RTX 3070)**
- **Model**: `microsoft/DiT-XL-2-256`
- **Training**: LoRA only
- **Batch Size**: 1
- **Resolution**: 256x256
- **Frames**: 8-12

---

## 📝 **Quick Start Commands**

### **Download Model Manually (Optional)**
```bash
# Install huggingface-hub
pip install huggingface-hub

# Download model
huggingface-cli download genmo/mochi-1-preview --local-dir ./models/mochi-1-preview
```

### **Set in OneTrainer**
```
Base Model: genmo/mochi-1-preview
# OR
Base Model: ./models/mochi-1-preview
```

---

## 🔍 **How to Find More WAN 2.2 Models**

### **Hugging Face Search**
1. Go to [huggingface.co/models](https://huggingface.co/models)
2. Search for: `"video generation" OR "video diffusion" OR "transformer"`
3. Filter by: `diffusers` library
4. Look for models with video capabilities

### **Popular Collections**
- **Stability AI**: Video diffusion models
- **Genmo**: Mochi video models  
- **Microsoft**: DiT transformer models
- **Community**: Fine-tuned variants

### **Model Requirements**
- ✅ **Transformer-based** architecture
- ✅ **Video generation** capability
- ✅ **Diffusers format** preferred
- ✅ **Flow matching** compatible (for WAN 2.2)

---

## ⚠️ **Important Notes**

### **Model Licensing**
- Check model license before use
- Some models require attribution
- Commercial use may be restricted

### **Model Size**
- Base models are typically 5-20GB
- Download time depends on internet speed
- Ensure sufficient disk space

### **Compatibility**
- Not all video models work with WAN 2.2
- Test with small datasets first
- Check model architecture compatibility

---

## 🎉 **Success Checklist**

When you have the right base model, you should see:
- ✅ Model loads without errors
- ✅ All components (transformer, VAE, text encoder) detected
- ✅ Training starts successfully
- ✅ No "model not found" errors
- ✅ Memory usage within limits

---

## 🆘 **Still Having Issues?**

1. **Try the simplest option first**: `microsoft/DiT-XL-2-256`
2. **Check the troubleshooting guide**: `docs/WAN22Troubleshooting.md`
3. **Verify your setup**: Run `python test_wan_final_validation.py`
4. **Check model availability**: Visit the Hugging Face model page
5. **Ask for help**: Provide the exact error message

**Most Common Solution**: Use `genmo/mochi-1-preview` as your base model! 🎯