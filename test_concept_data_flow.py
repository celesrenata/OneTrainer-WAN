#!/usr/bin/env python3
"""
Test the concept data flow to see where it breaks
"""

import sys
import os
import json
import torch
from pathlib import Path

# Add the modules to the path
sys.path.append('.')

def test_concept_data_flow():
    """Test the concept data flow step by step"""
    print("🔍 Testing Concept Data Flow...")
    
    try:
        # Import required modules
        from modules.util.config.ConceptConfig import ConceptConfig
        from modules.util.enum.ConceptType import ConceptType
        from mgds.ConceptPipelineModule import ConceptPipelineModule
        from mgds.SettingsPipelineModule import SettingsPipelineModule
        from mgds.pipelineModules.DownloadHuggingfaceDatasets import DownloadHuggingfaceDatasets
        from mgds.pipelineModules.CollectPaths import CollectPaths
        from mgds.PipelineModule import PipelineState
        from mgds.LoadingPipeline import LoadingPipeline
        
        print("   ✓ Imports successful")
        
        # Create concept data using proper ConceptConfig
        concept = ConceptConfig.default_values()
        concept.name = 'Cube'
        concept.path = '/workspace/input/training/cube'
        concept.enabled = True
        concept.type = ConceptType.STANDARD
        concept.include_subdirectories = False
        
        concepts = [concept]
        
        # Convert ConceptConfig objects to dictionaries for MGDS
        concept_dicts = [c.to_dict() for c in concepts]
        
        # Create settings
        settings = {
            "target_resolution": 384,
            "target_frames": 2,
        }
        
        print(f"   ✓ Created concept: {concept.name} -> {concept.path}")
        print(f"   ✓ Converted to dict format for MGDS")
        
        # Test ConceptPipelineModule
        print("   Testing ConceptPipelineModule...")
        concept_module = ConceptPipelineModule(concept_dicts)
        
        # Test SettingsPipelineModule
        print("   Testing SettingsPipelineModule...")
        settings_module = SettingsPipelineModule(settings)
        
        # Test DownloadHuggingfaceDatasets
        print("   Testing DownloadHuggingfaceDatasets...")
        download_datasets = DownloadHuggingfaceDatasets(
            concept_in_name='concept', 
            path_in_name='path', 
            enabled_in_name='enabled',
            concept_out_name='concept',
        )
        
        # Test CollectPaths
        print("   Testing CollectPaths...")
        collect_paths = CollectPaths(
            concept_in_name='concept', 
            path_in_name='path', 
            include_subdirectories_in_name='include_subdirectories', 
            enabled_in_name='enabled',
            path_out_name='video_path', 
            concept_out_name='concept',
            extensions=['.mp4'], 
            include_postfix=None, 
            exclude_postfix=[]
        )
        
        # Add OutputPipelineModule for full pipeline test
        from mgds.OutputPipelineModule import OutputPipelineModule
        output_module = OutputPipelineModule(['video_path', 'concept'])
        
        # Create a minimal pipeline to test the data flow
        print("   Creating minimal pipeline...")
        modules = [concept_module, settings_module, download_datasets, collect_paths, output_module]
        
        pipeline = LoadingPipeline(
            torch.device('cpu'),
            modules,
            batch_size=1,
            seed=42,
            state=PipelineState(1),
            initial_epoch=0,
            initial_index=0,
        )
        
        print("   ✓ Pipeline created")
        
        # Start the pipeline
        print("   Starting pipeline...")
        pipeline.start_next_epoch()
        
        # Test each module individually
        print("   Testing individual modules after start...")
        
        # Test ConceptPipelineModule
        print("   Testing ConceptPipelineModule...")
        try:
            concept_item = concept_module.get_item(0, 0)
            print(f"     Concept item: {concept_item}")
        except Exception as e:
            print(f"     ❌ ConceptPipelineModule failed: {e}")
            return False
        
        # Test SettingsPipelineModule
        print("   Testing SettingsPipelineModule...")
        try:
            settings_item = settings_module.get_item(0, 0)
            print(f"     Settings item: {settings_item}")
        except Exception as e:
            print(f"     ❌ SettingsPipelineModule failed: {e}")
            return False
        
        # Test DownloadHuggingfaceDatasets
        print("   Testing DownloadHuggingfaceDatasets...")
        try:
            print(f"     DownloadHuggingfaceDatasets concepts length: {len(download_datasets.concepts) if hasattr(download_datasets, 'concepts') else 'no concepts attr'}")
            download_item = download_datasets.get_item(0, 0)
            print(f"     Download item: {download_item}")
        except Exception as e:
            print(f"     ❌ DownloadHuggingfaceDatasets failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test CollectPaths
        print("   Testing CollectPaths...")
        try:
            print(f"     CollectPaths paths length: {len(collect_paths.paths) if hasattr(collect_paths, 'paths') else 'no paths attr'}")
            collect_item = collect_paths.get_item(0, 0)
            print(f"     Collect item: {collect_item}")
        except Exception as e:
            print(f"     ❌ CollectPaths failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("   ✓ All modules work individually")
        
        # Test full pipeline iteration
        print("   Testing full pipeline iteration...")
        try:
            first_item = next(iter(pipeline))
            print(f"   ✓ Got first item: {type(first_item)}")
            
            if isinstance(first_item, dict):
                print(f"   Item keys: {list(first_item.keys())}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error in full pipeline iteration: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"   ❌ Error in concept data flow test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Concept Data Flow Test")
    print("=" * 30)
    
    success = test_concept_data_flow()
    
    if success:
        print("\n✅ Concept data flow works correctly")
        print("   The issue is elsewhere in the complex pipeline")
    else:
        print("\n❌ Concept data flow has issues")
        print("   This is the root cause of the problem")

if __name__ == "__main__":
    main()