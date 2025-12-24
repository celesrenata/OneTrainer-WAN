#!/usr/bin/env python3
"""
Script to enable video validation debugging in WAN 2.2 pipeline.

This will modify the code to enable video validation logging so we can see
why videos are being rejected by the pipeline.
"""

import os
import re
from pathlib import Path

def find_video_validation_disable():
    """Find where video validation is disabled"""
    print("🔍 Searching for video validation disable...")
    
    # Search for the disable message
    search_patterns = [
        "Video validation temporarily disabled",
        "video_validation.*disabled",
        "disable.*video.*validation"
    ]
    
    found_files = []
    
    # Search in common directories
    search_dirs = [
        "modules/dataLoader",
        "modules/modelSetup", 
        "modules/util",
        "."
    ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for py_file in Path(search_dir).rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern in search_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            found_files.append(py_file)
                            print(f"   📄 Found in: {py_file}")
                            
                            # Show context
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if re.search(pattern, line, re.IGNORECASE):
                                    start = max(0, i-2)
                                    end = min(len(lines), i+3)
                                    print(f"      Context (lines {start+1}-{end}):")
                                    for j in range(start, end):
                                        marker = ">>>" if j == i else "   "
                                        print(f"      {marker} {j+1}: {lines[j]}")
                                    print()
                            break
                            
                except Exception as e:
                    continue
    
    return found_files

def enable_video_validation_logging():
    """Enable video validation logging"""
    print("\n🛠️  Enabling video validation logging...")
    
    # Common patterns to look for and fix
    fixes = [
        {
            'pattern': r'#.*video_validation.*modules.*=.*0',
            'replacement': 'video_validation_modules = []  # Re-enabled for debugging',
            'description': 'Enable video validation modules'
        },
        {
            'pattern': r'video_validation.*=.*\[\].*#.*disabled',
            'replacement': 'video_validation = []  # Re-enabled for debugging', 
            'description': 'Enable video validation list'
        },
        {
            'pattern': r'Video validation temporarily disabled.*',
            'replacement': '# Video validation re-enabled for debugging',
            'description': 'Remove disable message'
        }
    ]
    
    modified_files = []
    
    # Search in likely files
    search_files = [
        "modules/dataLoader/WanBaseDataLoader.py",
        "modules/dataLoader/mixin/DataLoaderMgdsMixin.py",
        "modules/modelSetup/BaseWanSetup.py"
    ]
    
    for file_path in search_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                for fix in fixes:
                    if re.search(fix['pattern'], content, re.IGNORECASE):
                        content = re.sub(fix['pattern'], fix['replacement'], content, flags=re.IGNORECASE)
                        print(f"   ✓ Applied fix: {fix['description']} in {file_path}")
                
                if content != original_content:
                    # Backup original
                    backup_path = f"{file_path}.backup"
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(original_content)
                    
                    # Write modified
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    modified_files.append(file_path)
                    print(f"   ✓ Modified: {file_path} (backup: {backup_path})")
                    
            except Exception as e:
                print(f"   ❌ Error modifying {file_path}: {e}")
    
    return modified_files

def add_debug_logging():
    """Add debug logging to video loading modules"""
    print("\n📝 Adding debug logging...")
    
    # Add logging to SafeLoadVideo if we can find it
    debug_code = '''
# DEBUG: Added for video validation debugging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_video_file(file_path, error=None):
    """Debug video file loading"""
    logger.debug(f"DEBUG VIDEO: Processing {file_path}")
    if error:
        logger.error(f"DEBUG VIDEO ERROR: {file_path} - {error}")
    else:
        logger.debug(f"DEBUG VIDEO SUCCESS: {file_path}")
'''
    
    # Look for video loading modules
    video_modules = [
        "modules/dataLoader/wan/VideoFrameSampler.py",
        "modules/util/video_util.py"
    ]
    
    for module_path in video_modules:
        if os.path.exists(module_path):
            try:
                with open(module_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if debug code already added
                if "DEBUG: Added for video validation debugging" not in content:
                    # Add debug code at the top after imports
                    lines = content.split('\n')
                    insert_pos = 0
                    
                    # Find good insertion point (after imports)
                    for i, line in enumerate(lines):
                        if line.strip().startswith('import ') or line.strip().startswith('from '):
                            insert_pos = i + 1
                    
                    lines.insert(insert_pos, debug_code)
                    
                    # Backup and write
                    backup_path = f"{module_path}.backup"
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    with open(module_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(lines))
                    
                    print(f"   ✓ Added debug logging to: {module_path}")
                else:
                    print(f"   ⚠️  Debug logging already present in: {module_path}")
                    
            except Exception as e:
                print(f"   ❌ Error adding debug to {module_path}: {e}")

def create_debug_commands():
    """Create commands for remote testing"""
    print("\n📋 Creating remote testing commands...")
    
    commands = [
        "# Remote debugging commands for WAN 2.2 video dataset",
        "",
        "# 1. Run comprehensive video debugging",
        "python debug_video_dataset_remote.py",
        "",
        "# 2. Check video files exist",
        "ls -la /workspace/input/training/cube/",
        "",
        "# 3. Check video properties with ffprobe",
        "ffprobe -v quiet -print_format json -show_format -show_streams /workspace/input/training/cube/*.mp4",
        "",
        "# 4. Test video with OpenCV",
        "python -c \"import cv2; cap=cv2.VideoCapture('/workspace/input/training/cube/test_cube.mp4'); print('Opened:', cap.isOpened()); print('Frames:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT))); cap.release()\"",
        "",
        "# 5. Check concepts configuration", 
        "cat training_concepts/concepts.json | jq .",
        "",
        "# 6. Enable video validation debugging",
        "python enable_video_validation_debug.py",
        "",
        "# 7. Run training with debug logging",
        "python -c \"import logging; logging.basicConfig(level=logging.DEBUG)\" && python train.py",
        "",
        "# 8. Check for specific error patterns in logs",
        "grep -i 'video\\|error\\|fail' training.log | tail -20",
        "",
        "# 9. Create minimal test video",
        "ffmpeg -y -f lavfi -i testsrc=duration=3:size=512x512:rate=30 -c:v libx264 -pix_fmt yuv420p /workspace/input/training/cube/minimal_test.mp4",
        "",
        "# 10. Test with single video file",
        "# Temporarily move all but one video file and test"
    ]
    
    with open("remote_debug_commands.txt", 'w') as f:
        f.write('\n'.join(commands))
    
    print(f"   ✓ Created: remote_debug_commands.txt")

def main():
    """Main function"""
    print("WAN 2.2 Video Validation Debug Enabler")
    print("=" * 40)
    
    # Find where validation is disabled
    found_files = find_video_validation_disable()
    
    # Enable validation logging
    modified_files = enable_video_validation_logging()
    
    # Add debug logging
    add_debug_logging()
    
    # Create remote commands
    create_debug_commands()
    
    print("\n🎯 Summary:")
    print(f"   Found validation disable in: {len(found_files)} files")
    print(f"   Modified files: {len(modified_files)}")
    
    if modified_files:
        print("\n✅ Video validation debugging enabled!")
        print("   Backup files created with .backup extension")
        print("   Run training again to see detailed video validation logs")
    else:
        print("\n⚠️  Could not automatically enable validation")
        print("   Manual intervention may be required")
    
    print("\n📋 Next steps:")
    print("1. Copy debug_video_dataset_remote.py to remote host")
    print("2. Run: python debug_video_dataset_remote.py")
    print("3. Use commands from remote_debug_commands.txt")
    print("4. Run training with debug logging enabled")

if __name__ == "__main__":
    main()