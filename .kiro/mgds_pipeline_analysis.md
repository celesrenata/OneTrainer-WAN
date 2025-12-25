# MGDS Pipeline Analysis - WAN 2.2 Training

## Pipeline Structure (32 modules total)

### Group 0: Data Input (3 modules)
- 0: ConceptPipelineModule → outputs: ['concept']
- 1: SettingsPipelineModule → outputs: ['settings'] 
- 2: DownloadHuggingfaceDatasets → outputs: ['concept']
- 3: CollectPaths → outputs: ['video_path', 'concept']
- 4: ModifyPath → outputs: ['sample_prompt_path']

### Group 2: Video Loading (9 SafePipelineModules)
- 5: SafePipelineModule (ProvideTargetFrames_0) → outputs: ['video_path', 'target_frames']
- 6: SafePipelineModule (SafeLoadVideo_1) → outputs: ['video']
- 7: SafePipelineModule (SafeLoadImage_2) → outputs: ['image'] 
- 8: SafePipelineModule (ImageToVideo_3) → outputs: ['video']
- 9: SafePipelineModule (LoadMultipleTexts_4) → outputs: ['sample_prompts']
- 10: SafePipelineModule (LoadMultipleTexts_5) → outputs: ['concept_prompts']
- 11: SafePipelineModule (GetFilename_6) → outputs: ['filename_prompt']
- 12: SafePipelineModule (SelectInput_7) → outputs: ['prompts']
- 13: SafePipelineModule (SelectRandomText_8) → outputs: ['prompt']

### Group 4: Aspect Processing (2 modules)
- 14: CalcAspect → outputs: ['original_resolution']
- 15: AspectBucketing → outputs: ['scale_resolution', 'crop_resolution', 'possible_resolutions']

### Group 5: Image Processing (1 module)
- 16: ScaleCropImage → outputs: ['video', 'crop_offset']

### Group 6: Augmentation (6 modules)
- 17: RandomFlip → outputs: ['video']
- 18: RandomRotate → outputs: ['video']
- 19: RandomBrightness → outputs: ['video']
- 20: RandomContrast → outputs: ['video']
- 21: RandomSaturation → outputs: ['video']
- 22: RandomHue → outputs: ['video']

### Group 8: Video Processing (3 modules) - **CRITICAL SECTION**
- 23: VideoFrameSampler → outputs: ['sampled_video']
- 24: RescaleImageChannels → outputs: ['scaled_video']
- **25: TemporalConsistencyVAE → outputs: ['latent_video']** ⚠️ **NOT BEING CALLED**

### Group 9: Caching (3 modules)
- 26: ValidatedDiskCache → outputs: ['original_resolution', 'crop_offset', 'crop_resolution', 'video_path']
- 27: ValidatedDiskCache → outputs: ['tokens', 'text_encoder_hidden_state', 'text_encoder_pooled_state']
- 28: VariationSorting → outputs: ['prompt_with_embeddings', 'concept']

### Group 10: Final Processing (3 modules)
- 29: RandomLatentMaskRemove → outputs: ['latent_mask', 'latent_conditioning_image']
- **30: AspectBatchSorting → outputs: ['video_path', 'latent_video', ...]** ⚠️ **FAILS HERE**
- 31: OutputPipelineModule

## Problem Analysis

### The Issue
- **AspectBatchSorting (module 30)** needs `latent_video`
- **TemporalConsistencyVAE (module 25)** should provide `latent_video`
- **But TemporalConsistencyVAE.get_item() is NEVER called**
- **AspectBatchSorting gets None instead of latent_video**
- **Error: `TypeError: 'NoneType' object is not subscriptable`**

### Root Cause Hypothesis
The MGDS pipeline system is not calling TemporalConsistencyVAE.get_item() because:
1. **Module dependency issue** - Pipeline doesn't recognize the dependency chain
2. **Module registration problem** - Module not properly registered in MGDS system
3. **Exception during pipeline setup** - Module fails during initialization phase
4. **Pipeline routing bug** - MGDS takes different path that bypasses the module

### Expected Data Flow
```
Raw Video (3 channels) 
  ↓ (modules 23-24: VideoFrameSampler, RescaleImageChannels)
Scaled Video (3 channels)
  ↓ (module 25: TemporalConsistencyVAE + WAN 2.2 VAE)
Latent Video (48 channels) 
  ↓ (module 30: AspectBatchSorting)
Batched Training Data
```

### Actual Data Flow
```
Raw Video (3 channels)
  ↓ (modules 23-24: VideoFrameSampler, RescaleImageChannels)  
Scaled Video (3 channels)
  ↓ (module 25: TemporalConsistencyVAE) ❌ **SKIPPED/FAILS**
None
  ↓ (module 30: AspectBatchSorting) ❌ **CRASHES**
TypeError: 'NoneType' object is not subscriptable
```

## Next Steps
1. **Verify module is callable** - Test TemporalConsistencyVAE in isolation
2. **Check MGDS integration** - Ensure module is properly registered
3. **Fix pipeline routing** - Make sure AspectBatchSorting gets data from TemporalConsistencyVAE
4. **Test VAE encoding** - Verify 3→48 channel conversion works
