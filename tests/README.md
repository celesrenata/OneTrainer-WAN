# OneTrainer WAN 2.2 Test Suite

This directory contains comprehensive tests for the WAN 2.2 (World Animator Network 2.2) implementation in OneTrainer.

## Test Structure

```
tests/
├── conftest.py                 # Pytest configuration and fixtures
├── unit/                       # Unit tests for individual components
│   ├── test_wan_model.py      # WanModel class tests
│   ├── test_wan_data_loader.py # WanBaseDataLoader tests
│   └── test_wan_lora.py       # LoRA functionality tests
├── integration/                # Integration tests for workflows
│   ├── test_wan_training_workflow.py # End-to-end training tests
│   └── test_wan_sampling.py   # Sampling integration tests
├── nix/                       # Nix environment and platform tests
│   ├── test_nix_environment.py # Nix environment compatibility
│   ├── test_gpu_compatibility.py # Multi-platform GPU tests
│   ├── test_virtual_environment.py # Virtual environment isolation
│   └── test_cpu_fallback.py  # CPU-only functionality tests
└── README.md                  # This file
```

## Test Categories

### Unit Tests (`tests/unit/`)

Test individual WAN 2.2 components in isolation:

- **WanModel Tests**: Model initialization, device movement, text encoding, video latent processing
- **WanBaseDataLoader Tests**: Video data processing, validation, frame sampling strategies
- **LoRA Tests**: LoRA adapter functionality, loading, saving, and training setup

### Integration Tests (`tests/integration/`)

Test complete workflows and component interactions:

- **Training Workflow Tests**: End-to-end training simulation with synthetic data
- **Sampling Tests**: Video generation and sampling integration during training

### Nix Environment Tests (`tests/nix/`)

Test compatibility with different environments and platforms:

- **Nix Environment**: Dependency resolution, package compatibility, reproducible builds
- **GPU Compatibility**: CUDA (NVIDIA) and ROCm (AMD) support testing
- **Virtual Environment**: Isolation and dependency management
- **CPU Fallback**: Functionality when GPU acceleration is unavailable

## Running Tests

### Prerequisites

1. Install pytest:
   ```bash
   pip install pytest
   ```

2. Ensure OneTrainer dependencies are installed:
   ```bash
   pip install torch torchvision numpy pillow transformers diffusers
   ```

### Quick Start

Run quick smoke tests to verify basic functionality:
```bash
python run_tests.py quick
```

### Test Categories

Run specific test categories:

```bash
# Unit tests only
python run_tests.py unit

# Integration tests only
python run_tests.py integration

# Nix environment tests
python run_tests.py nix

# GPU compatibility tests
python run_tests.py gpu

# CPU fallback tests
python run_tests.py cpu

# All tests
python run_tests.py all
```

### Direct pytest Usage

You can also run tests directly with pytest:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_wan_model.py

# Run specific test class
pytest tests/unit/test_wan_model.py::TestWanModel

# Run specific test method
pytest tests/unit/test_wan_model.py::TestWanModel::test_wan_model_initialization

# Run with verbose output
pytest tests/ -v

# Run with specific markers
pytest tests/ -m "not slow"
```

## Test Markers

Tests are marked with the following pytest markers:

- `unit`: Unit tests for individual components
- `integration`: Integration tests for workflows
- `nix`: Tests for Nix environment compatibility
- `gpu`: Tests requiring GPU acceleration
- `cuda`: Tests specifically for CUDA (NVIDIA GPU)
- `rocm`: Tests specifically for ROCm (AMD GPU)
- `cpu`: Tests for CPU-only functionality
- `slow`: Tests that take a long time to run

## Environment-Specific Testing

### Nix Environment

When running in a Nix environment (NixOS or nix-shell):

```bash
# Test Nix-specific functionality
python run_tests.py nix

# Check dependency resolution
pytest tests/nix/test_nix_environment.py::TestNixEnvironment::test_dependency_resolution_completeness
```

### GPU Testing

#### NVIDIA CUDA

```bash
# Test CUDA functionality
pytest tests/nix/test_gpu_compatibility.py -k "cuda"

# Check CUDA availability
pytest tests/nix/test_gpu_compatibility.py::TestGPUCompatibility::test_cuda_availability
```

#### AMD ROCm

```bash
# Test ROCm functionality
pytest tests/nix/test_gpu_compatibility.py -k "rocm"

# Check ROCm availability
pytest tests/nix/test_gpu_compatibility.py::TestGPUCompatibility::test_rocm_availability
```

### CPU-Only Development

For development environments without GPU:

```bash
# Test CPU fallback functionality
python run_tests.py cpu

# Test complete CPU workflow
pytest tests/nix/test_cpu_fallback.py::TestCPUFallback::test_cpu_development_workflow
```

## Test Data and Fixtures

### Fixtures Available

- `temp_dir`: Temporary directory for test files
- `mock_device`: Mock torch device for testing
- `mock_train_config`: Mock training configuration
- `mock_weight_dtypes`: Mock weight data types
- `mock_tokenizer`: Mock tokenizer for text processing
- `mock_text_encoder`: Mock text encoder
- `mock_vae`: Mock VAE for video processing
- `mock_transformer`: Mock transformer model
- `sample_video_tensor`: Sample video tensor for testing
- `skip_if_no_gpu`: Skip test if GPU not available
- `skip_if_no_rocm`: Skip test if ROCm not available

### Synthetic Test Data

Tests use synthetic data to avoid dependencies on large model files:

- Mock video tensors with realistic dimensions
- Synthetic text prompts for testing
- Mock model components with appropriate interfaces
- Temporary directories for file I/O testing

## Continuous Integration

### GitHub Actions

Example workflow for CI testing:

```yaml
name: WAN 2.2 Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.10, 3.11, 3.12]
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install pytest torch torchvision numpy pillow
    - name: Run tests
      run: python run_tests.py all
```

### Nix CI

For Nix-based CI:

```yaml
- name: Install Nix
  uses: cachix/install-nix-action@v20
- name: Run tests in Nix environment
  run: |
    nix-shell --run "python run_tests.py nix"
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure OneTrainer modules are in Python path
2. **GPU Tests Failing**: Check CUDA/ROCm installation and availability
3. **Nix Tests Failing**: Verify Nix environment setup and dependencies
4. **Slow Tests**: Use `-m "not slow"` to skip time-intensive tests

### Debug Mode

Run tests with debug information:

```bash
pytest tests/ -v --tb=long --capture=no
```

### Test Coverage

Generate test coverage report:

```bash
pip install pytest-cov
pytest tests/ --cov=modules --cov-report=html
```

## Contributing

When adding new tests:

1. Follow the existing test structure and naming conventions
2. Use appropriate fixtures from `conftest.py`
3. Add proper test markers for categorization
4. Include docstrings explaining test purpose
5. Mock external dependencies appropriately
6. Test both success and failure cases
7. Ensure tests are deterministic and isolated

### Test Guidelines

- **Unit tests**: Test single functions/methods in isolation
- **Integration tests**: Test component interactions and workflows
- **Environment tests**: Test platform-specific functionality
- **Use mocks**: Avoid dependencies on large model files or external services
- **Be specific**: Test specific functionality, not general behavior
- **Clean up**: Use fixtures and context managers for resource cleanup

## Performance Considerations

- Tests use small tensor sizes to run quickly
- GPU tests are skipped when GPU is unavailable
- Slow tests are marked and can be excluded
- Mock objects are used instead of real model loading
- Temporary directories are cleaned up automatically

## Security Considerations

- Tests do not require network access
- No sensitive data is used in tests
- Temporary files are created in isolated directories
- Mock objects prevent accidental model downloads