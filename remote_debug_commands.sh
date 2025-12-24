#!/bin/bash
# Remote debugging commands for WAN 2.2 video dataset issue
# Run these commands on the remote host where training is happening

echo "🎯 WAN 2.2 Video Dataset Remote Debugging"
echo "========================================"

echo ""
echo "📁 1. Checking training directory structure..."
ls -la /workspace/input/training/ 2>/dev/null || echo "❌ Training directory does not exist"

echo ""
echo "🎥 2. Checking cube video files..."
if [ -d "/workspace/input/training/cube" ]; then
    echo "✓ Cube directory exists"
    ls -la /workspace/input/training/cube/
    echo "Video file count: $(find /workspace/input/training/cube -name "*.mp4" -o -name "*.avi" -o -name "*.mov" | wc -l)"
else
    echo "❌ Cube directory does not exist"
    echo "Creating cube directory..."
    mkdir -p /workspace/input/training/cube
fi

echo ""
echo "🔍 3. Analyzing video properties..."
for video in /workspace/input/training/cube/*.{mp4,avi,mov}; do
    if [ -f "$video" ]; then
        echo "📹 Analyzing: $(basename "$video")"
        ffprobe -v quiet -print_format json -show_format -show_streams "$video" 2>/dev/null | jq -r '
            .format.duration as $duration |
            .streams[] | select(.codec_type=="video") |
            "  Duration: \($duration)s, Resolution: \(.width)x\(.height), FPS: \(.r_frame_rate), Codec: \(.codec_name)"
        ' 2>/dev/null || echo "  ❌ Could not analyze video"
    fi
done

echo ""
echo "📝 4. Checking concepts configuration..."
if [ -f "training_concepts/concepts.json" ]; then
    echo "✓ Concepts file exists"
    cat training_concepts/concepts.json | jq . 2>/dev/null || cat training_concepts/concepts.json
else
    echo "❌ Concepts file does not exist"
fi

echo ""
echo "🧪 5. Testing video loading with Python..."
python3 -c "
import sys
import os
sys.path.append('.')

try:
    import cv2
    print('✓ OpenCV available')
    
    cube_path = '/workspace/input/training/cube'
    if os.path.exists(cube_path):
        videos = [f for f in os.listdir(cube_path) if f.endswith(('.mp4', '.avi', '.mov'))]
        print(f'Found {len(videos)} video files')
        
        for video in videos[:2]:  # Test first 2
            video_path = os.path.join(cube_path, video)
            cap = cv2.VideoCapture(video_path)
            
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                print(f'  {video}: {frame_count} frames, {fps:.2f}fps, {width}x{height}')
                
                # Try reading first frame
                ret, frame = cap.read()
                if ret:
                    print(f'    ✓ First frame: {frame.shape}')
                else:
                    print(f'    ❌ Could not read first frame')
                    
                cap.release()
            else:
                print(f'  ❌ Could not open: {video}')
    else:
        print('❌ Cube directory not found')
        
except ImportError:
    print('❌ OpenCV not available')
except Exception as e:
    print(f'❌ Error: {e}')
"

echo ""
echo "🎬 6. Creating test video..."
ffmpeg -y -f lavfi -i testsrc=duration=5:size=512x512:rate=30 \
    -c:v libx264 -pix_fmt yuv420p -preset fast \
    /workspace/input/training/cube/debug_test.mp4 2>/dev/null

if [ -f "/workspace/input/training/cube/debug_test.mp4" ]; then
    echo "✓ Test video created"
    ls -la /workspace/input/training/cube/debug_test.mp4
else
    echo "❌ Failed to create test video"
fi

echo ""
echo "🔧 7. Checking WAN 2.2 pipeline modules..."
python3 -c "
import sys
import os
sys.path.append('.')

# Check if key modules exist
modules_to_check = [
    'modules/dataLoader/WanBaseDataLoader.py',
    'modules/dataLoader/mixin/DataLoaderMgdsMixin.py', 
    'modules/dataLoader/wan/VideoFrameSampler.py',
    'modules/util/video_util.py'
]

for module in modules_to_check:
    if os.path.exists(module):
        print(f'✓ {module}')
        
        # Check for video validation disable
        with open(module, 'r') as f:
            content = f.read()
            if 'validation temporarily disabled' in content.lower():
                print(f'  ⚠️  Video validation disabled in {module}')
            if 'video_validation' in content.lower() and '0' in content:
                print(f'  ⚠️  Video validation set to 0 in {module}')
    else:
        print(f'❌ {module}')
"

echo ""
echo "📊 8. Running comprehensive debug script..."
if [ -f "debug_video_dataset_remote.py" ]; then
    python3 debug_video_dataset_remote.py
else
    echo "❌ debug_video_dataset_remote.py not found"
    echo "Please copy this file to the remote host first"
fi

echo ""
echo "🎯 Summary and Next Steps:"
echo "========================="
echo "1. If videos exist but dataset is empty:"
echo "   - Videos are being filtered out by the pipeline"
echo "   - Enable video validation logging"
echo "   - Check SafeLoadVideo and VideoFrameSampler modules"
echo ""
echo "2. If no videos found:"
echo "   - Copy videos to /workspace/input/training/cube/"
echo "   - Ensure videos are valid (duration >1s, resolution >256x256)"
echo ""
echo "3. To enable debug logging:"
echo "   - Run: python3 enable_video_validation_debug.py"
echo "   - Then run training again"
echo ""
echo "4. To test with minimal setup:"
echo "   - Use only the debug_test.mp4 created above"
echo "   - Reduce frame count in training config (frames=2 instead of 8)"