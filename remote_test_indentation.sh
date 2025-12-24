#!/bin/bash

echo "=== Remote Testing: Indentation Fix ==="
echo "Connecting to remote server to test the indentation fix..."

ssh -p 3701 10.1.1.12 << 'EOF'
cd ~/OneTrainer-WAN
source venv/bin/activate

echo "=== Testing Python syntax compilation ==="
python -m py_compile modules/dataLoader/mixin/DataLoaderText2VideoMixin.py
if [ $? -eq 0 ]; then
    echo "✓ DataLoaderText2VideoMixin.py compiles without syntax errors"
else
    echo "✗ DataLoaderText2VideoMixin.py has syntax errors"
    exit 1
fi

echo ""
echo "=== Testing basic imports ==="
python -c "
try:
    from modules.dataLoader.mixin.DataLoaderText2VideoMixin import DataLoaderText2VideoMixin
    print('✓ DataLoaderText2VideoMixin imported successfully')
except SyntaxError as e:
    print(f'✗ Syntax error: {e}')
    exit(1)
except ImportError as e:
    print(f'✓ Syntax OK, but missing dependencies: {e}')
except Exception as e:
    print(f'✗ Other error: {e}')
    exit(1)
"

echo ""
echo "=== Testing training script startup ==="
timeout 10s python scripts/train.py --config-path "training_presets/#wan 2.2 LoRA.json" 2>&1 | head -20

echo ""
echo "=== Test completed ==="
EOF