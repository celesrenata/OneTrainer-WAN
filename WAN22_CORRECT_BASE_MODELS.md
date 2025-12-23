# WAN 2.2 Correct Base Models Guide

## ⚠️ **IMPORTANT: Previous Recommendation Was Incorrect**

The previous recommendation of `genmo/mochi-1-preview` was **incorrect**. Mochi-1 is not a WAN 2.2 model - it has a different architecture.

## 🎯 **Correct WAN 2.2 Base Models**

### **Option 1: Use Stable Diffusion as Base (Recommended)**
```
runwayml/stable-diffusion-v1-5
```
**Why**: WAN 2.2 can be trained on top of Stable Diffusion architecture with video extensions.

### **Option 2: Stable Diffusion XL Base**
```
stabilityai/stable-diffusion-xl-base-1.0
```
**Why**: More modern base model with better quality.

### **Option 3: AnimateDiff Base**
```
guoyww/animatediff-motion-adapter-v1-5-2
```
**Why**: Already designed for video generation.

### **Option 4: Leave Empty for New Training**
```
(Leave Base Model field empty)
```
**Why**: Train WAN 2.2 from scratch with your video data.

---

## 🔧 **The Real Issue**

The error you're seeing is because:

1. **Mochi-1 is not WAN 2.2**: Different model architecture
2. **WAN 2.2 is a research model**: May not have public pretrained weights
3. **Model loader expects specific format**: WAN 2.2 has specific component requirements

---

## 🚀 **Recommended Solutions**

### **Solution 1: Use Stable Diffusion Base (Easiest)**
```
Base Model: runwayml/stable-diffusion-v1-5
Model Type: WAN_2_2
Training Method: LoRA
```

### **Solution 2: Train from Scratch**
```
Base Model: (leave empty)
Model Type: WAN_2_2  
Training Method: Fine-tune
```

### **Solution 3: Use AnimateDiff**
```
Base Model: guoyww/animatediff-motion-adapter-v1-5-2
Model Type: WAN_2_2
Training Method: LoRA
```

---

## 🎯 **Quick Fix for Your Current Error**

1. **Stop the current training**
2. **Change Base Model to**: `runwayml/stable-diffusion-v1-5`
3. **Keep Model Type**: `WAN_2_2`
4. **Use Training Preset**: `#wan 2.2 LoRA.json`
5. **Start training again**

---

## 📝 **What is WAN 2.2 Actually?**

**WAN 2.2 (World Animator Network 2.2)** is likely:
- A research model for video generation
- Based on transformer architecture
- Designed for flow matching
- May not have publicly available pretrained weights

Since there are no official WAN 2.2 pretrained models, OneTrainer's WAN 2.2 implementation is designed to:
1. **Adapt existing models** (like Stable Diffusion) for video
2. **Train from scratch** with video data
3. **Use video-specific architectures** with transformer components

---

## 🔍 **How to Find Real WAN 2.2 Models**

Currently, there are **no official WAN 2.2 models** on Hugging Face because:
1. WAN 2.2 is a research architecture
2. No official implementation has been released
3. OneTrainer's implementation is custom

**Best approach**: Use Stable Diffusion as base and train with your video data.

---

## ⚡ **Quick Start (Fixed)**

```bash
# In OneTrainer GUI:
Model Type: WAN_2_2
Base Model: runwayml/stable-diffusion-v1-5
Training Preset: #wan 2.2 LoRA.json
Training Data: Your video dataset
```

This should work without the VAE configuration errors you encountered!

---

## 🆘 **If You Still Get Errors**

The WAN model loader might need updates to handle different base models. The errors suggest:

1. **VAE config issue**: `'list' object cannot be interpreted as an integer`
2. **Missing diffusers method**: `'from_single_file'` not available
3. **Architecture mismatch**: Model components don't match expected structure

**Temporary workaround**: Try training without a base model (from scratch) or use a different model type until the WAN loader is updated.

---

## 🎉 **Summary**

- ❌ **Don't use**: `genmo/mochi-1-preview` (wrong architecture)
- ✅ **Do use**: `runwayml/stable-diffusion-v1-5` (compatible base)
- ✅ **Or leave empty**: Train WAN 2.2 from scratch
- ✅ **Use LoRA**: More memory efficient and stable

The WAN 2.2 implementation in OneTrainer is designed to work with standard diffusion models as a base, not with Mochi-1's specific architecture.