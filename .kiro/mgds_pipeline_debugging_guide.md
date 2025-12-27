# MGDS Pipeline Debugging Guide - WAN 2.2 Integration

## Current Status: MAJOR BREAKTHROUGH ✅

### What's Working
- ✅ **TemporalConsistencyVAE is being called** - Core video processing pipeline functional
- ✅ **Real WAN 2.2 VAE integration** - AutoencoderKLWan produces 48-channel latents
- ✅ **MGDS pipeline routing** - Requests reach correct modules (TemporalConsistencyVAE)
- ✅ **Dataset configuration** - 8 training samples detected, concepts.json working
- ✅ **AspectBatchSorting includes latent_video** - Pipeline expects correct data types

### Current Issue: MGDS Architectural Gap

**The Problem:**
MGDS pipeline has **dependency resolution gaps** where modules are too far apart to access each other's outputs.

**Pipeline Structure:**
```
Load Input (0-13)     Preparation (14-22)     Cache (23+)
├─ Module 5: video_path, target_frames
├─ Module 8: video           [GAP: 15 modules]    ├─ Module 23: VideoFrameSampler (needs video)
├─ Module 13: prompt         [GAP: 10 modules]    ├─ Module 25: TemporalConsistencyVAE ✅ WORKING
                                                  ├─ Module 26: MapData (needs prompt)
```

**The Gap:**
- VideoFrameSampler (Module 23) needs `'video'` from Module 8
- MapData (Module 26) needs `'prompt'` from Module 13  
- **15+ module gap** causes MGDS backward search to fail
- Intermediate modules (14-22) don't pass through the original data

**Error Pattern:**
```
VideoFrameSampler → can't get 'video' → returns None
RescaleImageChannels → can't get 'sampled_video' → returns None  
TemporalConsistencyVAE → can't get 'scaled_video' → uses fallback ✅
```

## External Testing Strategy (No Torch Required)

### 1. Mock MGDS Pipeline Tester

Create a standalone Python script that simulates the MGDS pipeline without PyTorch dependencies:

```python
# test_mgds_pipeline.py
class MockModule:
    def __init__(self, name, inputs, outputs, index):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.index = index
        self.cache = {}
    
    def get_inputs(self):
        return self.inputs
    
    def get_outputs(self):
        return self.outputs
    
    def get_item(self, variation, index, item_name):
        # Simulate the actual MGDS behavior
        return {item_name: f"mock_{item_name}_data"}

class MockPipeline:
    def __init__(self, modules):
        self.modules = modules
    
    def find_dependency(self, requesting_module_index, item_name):
        """Simulate MGDS backward search"""
        for i in range(requesting_module_index - 1, -1, -1):
            module = self.modules[i]
            if item_name in module.get_outputs():
                return module
        return None
    
    def test_dependency_chain(self):
        """Test if all modules can find their dependencies"""
        results = {}
        for module in self.modules:
            results[module.name] = {}
            for input_name in module.get_inputs():
                provider = self.find_dependency(module.index, input_name)
                results[module.name][input_name] = {
                    'provider': provider.name if provider else None,
                    'gap': module.index - provider.index if provider else 'MISSING'
                }
        return results
```

### 2. WAN Pipeline Configuration Tester

```python
# wan_pipeline_config.py
def create_wan_pipeline_mock():
    """Create mock WAN pipeline matching current structure"""
    modules = [
        # Load Input (0-13)
        MockModule("CollectPaths", [], ["video_path", "concept"], 3),
        MockModule("SafePipelineModule_5", [], ["video_path", "target_frames"], 5),
        MockModule("SafePipelineModule_8", [], ["video"], 8),
        MockModule("SafePipelineModule_13", [], ["prompt"], 13),
        
        # Preparation (14-22) - Gap modules
        MockModule("CalcAspect", [], ["original_resolution"], 14),
        MockModule("AspectBucketing", [], ["scale_resolution", "crop_resolution"], 15),
        MockModule("RandomFlip", ["video"], ["video"], 17),
        MockModule("RandomRotate", ["video"], ["video"], 18),
        
        # Cache (23+)
        MockModule("VideoFrameSampler", ["video", "video_path", "target_frames"], ["sampled_video"], 23),
        MockModule("RescaleImageChannels", ["sampled_video"], ["scaled_video"], 24),
        MockModule("TemporalConsistencyVAE", ["scaled_video"], ["latent_video"], 25),
        MockModule("MapData_prompt", ["prompt"], ["prompt_passthrough"], 26),
        MockModule("AspectBatchSorting", ["latent_video", "prompt_with_embeddings"], [], 32),
    ]
    
    return MockPipeline(modules)

def test_wan_dependencies():
    pipeline = create_wan_pipeline_mock()
    results = pipeline.test_dependency_chain()
    
    print("WAN Pipeline Dependency Analysis:")
    for module_name, deps in results.items():
        print(f"\n{module_name}:")
        for input_name, info in deps.items():
            status = "✅" if info['provider'] else "❌"
            gap = f"(gap: {info['gap']})" if info['gap'] != 'MISSING' else "(MISSING)"
            print(f"  {status} {input_name} <- {info['provider']} {gap}")
```

### 3. Pipeline Optimization Recommendations

```python
# pipeline_optimizer.py
def analyze_gaps(pipeline_results):
    """Identify problematic gaps in the pipeline"""
    issues = []
    
    for module_name, deps in pipeline_results.items():
        for input_name, info in deps.items():
            if info['provider'] is None:
                issues.append(f"MISSING: {module_name} needs {input_name}")
            elif isinstance(info['gap'], int) and info['gap'] > 10:
                issues.append(f"LARGE GAP: {module_name} needs {input_name} from {info['gap']} modules away")
    
    return issues

def suggest_fixes(issues):
    """Suggest architectural fixes"""
    fixes = []
    
    for issue in issues:
        if "VideoFrameSampler needs video" in issue:
            fixes.append("Move VideoFrameSampler to load_input phase (modules 0-13)")
        elif "MapData needs prompt" in issue:
            fixes.append("Add prompt pass-through in preparation phase")
        elif "LARGE GAP" in issue:
            fixes.append(f"Add pass-through module for: {issue}")
    
    return fixes
```

### 4. Usage Instructions

**To test the pipeline without OneTrainer:**

1. **Create the test files** above in `/workspace/OneTrainer-WAN/`
2. **Run the mock pipeline test:**
   ```bash
   cd /workspace/OneTrainer-WAN
   python test_mgds_pipeline.py
   ```
3. **Analyze the results** to identify dependency gaps
4. **Apply fixes** based on the recommendations
5. **Re-test** until all dependencies are satisfied

**Benefits:**
- ✅ **No PyTorch dependency** - Pure Python testing
- ✅ **Fast iteration** - No model loading or GPU requirements  
- ✅ **Clear visualization** - See exactly which modules can't find their inputs
- ✅ **Architectural validation** - Test pipeline changes before implementation

### 5. Current Recommended Fixes

Based on the analysis:

1. **Move VideoFrameSampler to load_input phase** - Closer to video data sources
2. **Add explicit pass-through modules** - Bridge gaps between phases
3. **Simplify text processing** - Reduce dependency chain complexity
4. **Use ValidatedDiskCache more effectively** - Better fallback handling

### 6. Next Steps

1. **Create and run the mock pipeline tester**
2. **Identify all dependency gaps** 
3. **Design minimal fixes** that don't break existing functionality
4. **Test fixes in mock environment first**
5. **Apply fixes to actual WAN data loader**
6. **Validate with real training**

This approach allows **rapid iteration** on pipeline architecture without the overhead of loading models, initializing GPUs, or dealing with PyTorch dependencies.
