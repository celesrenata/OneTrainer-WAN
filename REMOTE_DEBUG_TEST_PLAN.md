# Remote Debug Test Plan - Data Pipeline Issue

## Current Status
- ✅ WAN 2.2 pipeline initializes successfully 
- ✅ 10 supported files exist in `/workspace/input/training/cube`
- ❌ Data loader returns 0 items - no training steps occur

## Test Commands to Run (in order)

### 1. First, pull the latest changes:
```bash
git pull
```

### 2. Test basic file detection:
```bash
python debug_mgds_simple.py
```
**Expected**: Should show 10 supported files and confirm path matching

### 3. Run training with enhanced debug logging:
```bash
python scripts/train.py --config-path "training_presets/#wan 2.2 LoRA flexible.json" 2>&1 | head -100
```
**Look for these new debug lines:**
- `DEBUG: CollectPaths created and wrapped with safety wrapper`
- `DEBUG: AspectBatchSorting created with batch_size=X, names=[...]`

### 4. If training starts, let it run for a few epochs and check:
```bash
python scripts/train.py --config-path "training_presets/#wan 2.2 LoRA flexible.json" 2>&1 | grep -E "(length|step:|DEBUG:)"
```
**Look for**:
- `Group0_SafeCollectPaths_1 wrapped length: 10` (should be > 0)
- `Group10_AspectBatchSorting_1 length() returned: 0` (this is the problem)
- Any new DEBUG lines we added

### 5. Alternative test with local data:
```bash
python scripts/train.py --config-path "training_presets/#wan 2.2 LoRA local.json" 2>&1 | head -50
```

## What to Look For

### ✅ Good Signs:
- CollectPaths shows length > 0
- AspectBatchSorting shows length > 0  
- `step: X/Y` instead of `step: 0it`

### ❌ Problem Signs:
- `Group10_AspectBatchSorting_1 length() returned: 0`
- `step: 0it [00:00, ?it/s]` (no progress)
- Missing DEBUG lines (means our changes didn't apply)

## Key Question
**Where exactly does the data get lost between CollectPaths (10 items) and AspectBatchSorting (0 items)?**

The pipeline flow is:
1. CollectPaths finds files ✅
2. Video processing modules ✅  
3. AspectBatchSorting gets 0 items ❌

## Copy Results
Please copy the output from commands 2, 3, and 4 so I can see:
1. File detection results
2. New debug logging output
3. Where the data disappears in the pipeline

This will help identify the exact point where the 10 files become 0 items.