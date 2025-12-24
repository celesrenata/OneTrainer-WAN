#!/usr/bin/env python3
"""
Pipeline trace script to monitor video processing during training.

This script can be run alongside training to see what's happening in the pipeline.
"""

import time
import os

def monitor_pipeline():
    """Monitor pipeline execution"""
    print("🔍 Monitoring pipeline execution...")
    print("This will show debug output from the training process.")
    print("Run this in a separate terminal while training is running.")
    print()
    
    # Instructions for the user
    print("📋 What to look for in the training output:")
    print()
    print("1. CollectPaths debugging:")
    print("   - 'DEBUG MGDS: CollectPaths length() = X'")
    print("   - 'DEBUG MGDS: CollectPaths get_item called'")
    print()
    print("2. LoadVideo debugging:")
    print("   - 'DEBUG MGDS: LoadVideo get_item called'")
    print("   - 'DEBUG MGDS: LoadVideo returned X for item Y'")
    print()
    print("3. SafeLoadVideo debugging:")
    print("   - 'DEBUG SAFE_LOAD_VIDEO: Processing item X'")
    print("   - 'DEBUG SAFE_LOAD_VIDEO SUCCESS/ERROR'")
    print()
    print("4. Pipeline module debugging:")
    print("   - 'DEBUG PIPELINE: SafePipelineModule processing item X'")
    print("   - 'DEBUG PIPELINE: SafePipelineModule got result: X'")
    print()
    print("🎯 Key Questions to Answer:")
    print("- Are video files reaching LoadVideo?")
    print("- Is LoadVideo returning None or valid data?")
    print("- Where in the pipeline are videos being lost?")
    print()
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    monitor_pipeline()
