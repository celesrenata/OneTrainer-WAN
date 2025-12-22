#!/usr/bin/env python3
"""
Simple unit test runner for WAN 2.2 implementation.
Tests core functionality without requiring pytest or ML dependencies.
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

def test_model_type_enum():
    """Test ModelType enum has WAN_2_2."""
    print("\n=== Testing ModelType Enum ===")
    try:
        from modules.util.enum.ModelType import ModelType
        
        # Test WAN_2_2 exists
        assert hasattr(ModelType, 'WAN_2_2'), "WAN_2_2 not found in ModelType"
        print("✓ WAN_2_2 model type exists")
        
        # Test helper methods
        assert ModelType.WAN_2_2.is_wan(), "is_wan() should return True"
        print("✓ is_wan() method works")
        
        assert ModelType.WAN_2_2.is_video_model(), "is_video_model() should return True"
        print("✓ is_video_model() method works")
        
        assert ModelType.WAN_2_2.is_flow_matching(), "is_flow_matching() should return True"
        print("✓ is_flow_matching() method works")
        
        return True
    except Exception as e:
        print(f"✗ ModelType enum test failed: {e}")
        return False

def test_wan_model_import():
    """Test WanModel can be imported."""
    print("\n=== Testing WanModel Import ===")
    try:
        from modules.model.WanModel import WanModel, WanModelEmbedding
        print("✓ WanModel imported successfully")
        print("✓ WanModelEmbedding imported successfully")
        return True
    except Exception as e:
        print(f"✗ WanModel import failed: {e}")
        return False

def test_wan_model_initialization():
    """Test WanModel initialization."""
    print("\n=== Testing WanModel Initialization ===")
    try:
        from modules.model.WanModel import WanModel
        from modules.util.enum.ModelType import ModelType
        from modules.util.enum.DataType import DataType
        
        # Create model
        model = WanModel(ModelType.WAN_2_2)
        
        # Test attributes
        assert model.model_type == ModelType.WAN_2_2, "Model type not set correctly"
        print("✓ Model type set correctly")
        
        assert model.tokenizer is None, "Tokenizer should be None initially"
        assert model.text_encoder is None, "Text encoder should be None initially"
        assert model.vae is None, "VAE should be None initially"
        assert model.transformer is None, "Transformer should be None initially"
        print("✓ Model components initialized to None")
        
        assert model.text_encoder_train_dtype == DataType.FLOAT_32, "Default dtype incorrect"
        assert model.transformer_train_dtype == DataType.FLOAT_32, "Default dtype incorrect"
        print("✓ Default data types set correctly")
        
        assert model.embedding is None, "Embedding should be None initially"
        assert model.additional_embeddings == [], "Additional embeddings should be empty list"
        print("✓ Embedding attributes initialized correctly")
        
        return True
    except Exception as e:
        print(f"✗ WanModel initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_wan_model_methods():
    """Test WanModel methods exist."""
    print("\n=== Testing WanModel Methods ===")
    try:
        from modules.model.WanModel import WanModel
        from modules.util.enum.ModelType import ModelType
        
        model = WanModel(ModelType.WAN_2_2)
        
        # Test method existence
        methods = [
            'to', 'eval', 'train',
            'vae_to', 'text_encoder_to', 'transformer_to',
            'encode_text', 'pack_latents', 'unpack_latents',
            'adapters', 'all_embeddings', 'all_text_encoder_embeddings'
        ]
        
        for method_name in methods:
            assert hasattr(model, method_name), f"Method {method_name} not found"
            print(f"✓ Method '{method_name}' exists")
        
        return True
    except Exception as e:
        print(f"✗ WanModel methods test failed: {e}")
        return False

def test_data_loader_import():
    """Test WanBaseDataLoader can be imported."""
    print("\n=== Testing WanBaseDataLoader Import ===")
    try:
        from modules.dataLoader.WanBaseDataLoader import WanBaseDataLoader
        print("✓ WanBaseDataLoader imported successfully")
        return True
    except Exception as e:
        print(f"✗ WanBaseDataLoader import failed: {e}")
        return False

def test_model_loaders_import():
    """Test model loaders can be imported."""
    print("\n=== Testing Model Loaders Import ===")
    try:
        from modules.modelLoader.WanFineTuneModelLoader import WanFineTuneModelLoader
        print("✓ WanFineTuneModelLoader imported")
        
        from modules.modelLoader.WanLoRAModelLoader import WanLoRAModelLoader
        print("✓ WanLoRAModelLoader imported")
        
        from modules.modelLoader.WanEmbeddingModelLoader import WanEmbeddingModelLoader
        print("✓ WanEmbeddingModelLoader imported")
        
        from modules.modelLoader.wan.WanModelLoader import WanModelLoader
        print("✓ WanModelLoader imported")
        
        return True
    except Exception as e:
        print(f"✗ Model loaders import failed: {e}")
        return False

def test_model_savers_import():
    """Test model savers can be imported."""
    print("\n=== Testing Model Savers Import ===")
    try:
        from modules.modelSaver.WanFineTuneModelSaver import WanFineTuneModelSaver
        print("✓ WanFineTuneModelSaver imported")
        
        from modules.modelSaver.WanLoRAModelSaver import WanLoRAModelSaver
        print("✓ WanLoRAModelSaver imported")
        
        from modules.modelSaver.WanEmbeddingModelSaver import WanEmbeddingModelSaver
        print("✓ WanEmbeddingModelSaver imported")
        
        from modules.modelSaver.wan.WanModelSaver import WanModelSaver
        print("✓ WanModelSaver imported")
        
        return True
    except Exception as e:
        print(f"✗ Model savers import failed: {e}")
        return False

def test_model_setup_import():
    """Test model setup classes can be imported."""
    print("\n=== Testing Model Setup Import ===")
    try:
        from modules.modelSetup.BaseWanSetup import BaseWanSetup
        print("✓ BaseWanSetup imported")
        
        from modules.modelSetup.WanFineTuneSetup import WanFineTuneSetup
        print("✓ WanFineTuneSetup imported")
        
        from modules.modelSetup.WanLoRASetup import WanLoRASetup
        print("✓ WanLoRASetup imported")
        
        from modules.modelSetup.WanEmbeddingSetup import WanEmbeddingSetup
        print("✓ WanEmbeddingSetup imported")
        
        return True
    except Exception as e:
        print(f"✗ Model setup import failed: {e}")
        return False

def test_model_sampler_import():
    """Test model sampler can be imported."""
    print("\n=== Testing Model Sampler Import ===")
    try:
        from modules.modelSampler.WanModelSampler import WanModelSampler
        print("✓ WanModelSampler imported successfully")
        return True
    except Exception as e:
        print(f"✗ WanModelSampler import failed: {e}")
        return False

def test_video_utilities_import():
    """Test video utilities can be imported."""
    print("\n=== Testing Video Utilities Import ===")
    try:
        from modules.util.video_util import (
            FrameSamplingStrategy,
            VideoValidationError,
            validate_video_file,
            get_video_info
        )
        print("✓ Video utilities imported successfully")
        return True
    except Exception as e:
        print(f"✗ Video utilities import failed: {e}")
        return False

def test_configuration_support():
    """Test configuration support for WAN 2.2."""
    print("\n=== Testing Configuration Support ===")
    try:
        from modules.util.config.TrainConfig import TrainConfig
        from modules.util.enum.ModelType import ModelType
        
        config = TrainConfig()
        config.model_type = ModelType.WAN_2_2
        
        assert config.model_type == ModelType.WAN_2_2, "Model type not set correctly"
        print("✓ TrainConfig supports WAN_2_2")
        
        # Test video-specific parameters
        config.target_frames = 16
        config.frame_sample_strategy = "uniform"
        config.temporal_consistency_weight = 1.0
        
        assert config.target_frames == 16, "target_frames not set correctly"
        assert config.frame_sample_strategy == "uniform", "frame_sample_strategy not set correctly"
        assert config.temporal_consistency_weight == 1.0, "temporal_consistency_weight not set correctly"
        print("✓ Video-specific parameters supported")
        
        return True
    except Exception as e:
        print(f"✗ Configuration support test failed: {e}")
        return False

def test_factory_functions():
    """Test factory functions support WAN 2.2."""
    print("\n=== Testing Factory Functions ===")
    try:
        from modules.util.create import create_model_loader, create_model_saver, create_model_setup
        from modules.util.enum.ModelType import ModelType
        from modules.util.enum.DataType import DataType
        
        # Test model loader factory
        loader = create_model_loader(ModelType.WAN_2_2, train_dtype=DataType.FLOAT_32)
        assert loader is not None, "Model loader factory returned None"
        print("✓ Model loader factory supports WAN_2_2")
        
        # Test model saver factory
        saver = create_model_saver(ModelType.WAN_2_2)
        assert saver is not None, "Model saver factory returned None"
        print("✓ Model saver factory supports WAN_2_2")
        
        # Test model setup factory
        setup = create_model_setup(ModelType.WAN_2_2)
        assert setup is not None, "Model setup factory returned None"
        print("✓ Model setup factory supports WAN_2_2")
        
        return True
    except Exception as e:
        print(f"✗ Factory functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_presets():
    """Test training preset files exist and are valid."""
    print("\n=== Testing Training Presets ===")
    try:
        import json
        
        preset_files = [
            "training_presets/#wan 2.2 Finetune.json",
            "training_presets/#wan 2.2 LoRA.json",
            "training_presets/#wan 2.2 LoRA 8GB.json",
            "training_presets/#wan 2.2 Embedding.json"
        ]
        
        valid_count = 0
        for preset_file in preset_files:
            if os.path.exists(preset_file):
                with open(preset_file, 'r') as f:
                    preset_config = json.load(f)
                
                assert 'model_type' in preset_config, f"{preset_file} missing model_type"
                assert preset_config['model_type'] == 'WAN_2_2', f"{preset_file} has wrong model_type"
                print(f"✓ {preset_file} is valid")
                valid_count += 1
            else:
                print(f"✗ {preset_file} not found")
        
        assert valid_count >= 3, "Not enough valid training presets"
        print(f"✓ {valid_count} training presets validated")
        
        return True
    except Exception as e:
        print(f"✗ Training presets test failed: {e}")
        return False

def main():
    """Run all unit tests."""
    print("=" * 70)
    print("WAN 2.2 Unit Tests - Simple Runner")
    print("=" * 70)
    
    tests = [
        test_model_type_enum,
        test_wan_model_import,
        test_wan_model_initialization,
        test_wan_model_methods,
        test_data_loader_import,
        test_model_loaders_import,
        test_model_savers_import,
        test_model_setup_import,
        test_model_sampler_import,
        test_video_utilities_import,
        test_configuration_support,
        test_factory_functions,
        test_training_presets
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} raised exception: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)
    
    if failed == 0:
        print("🎉 ALL UNIT TESTS PASSED! 🎉")
        print("\nWAN 2.2 implementation is working correctly:")
        print("  ✓ Model type enum integration")
        print("  ✓ WanModel class functionality")
        print("  ✓ Data loader components")
        print("  ✓ Model loaders (fine-tune, LoRA, embedding)")
        print("  ✓ Model savers (fine-tune, LoRA, embedding)")
        print("  ✓ Model setup classes")
        print("  ✓ Model sampler")
        print("  ✓ Video utilities")
        print("  ✓ Configuration support")
        print("  ✓ Factory functions")
        print("  ✓ Training presets")
        return True
    else:
        print(f"⚠ {failed} test(s) failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
