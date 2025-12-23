#!/usr/bin/env python3
"""
Troubleshoot OneTrainer GUI launch issues.
Helps diagnose why the GUI window isn't appearing.
"""
import sys
import os
import subprocess
import importlib

def check_display_environment():
    """Check display environment variables."""
    print("=== Checking Display Environment ===")
    
    display_vars = ['DISPLAY', 'WAYLAND_DISPLAY', 'XDG_SESSION_TYPE']
    
    for var in display_vars:
        value = os.environ.get(var)
        if value:
            print(f"✓ {var}: {value}")
        else:
            print(f"⚠ {var}: not set")
    
    # Check if we're in SSH
    ssh_vars = ['SSH_CLIENT', 'SSH_TTY', 'SSH_CONNECTION']
    in_ssh = any(os.environ.get(var) for var in ssh_vars)
    
    if in_ssh:
        print("⚠ Running in SSH session - GUI may not display")
        print("  Try: ssh -X username@hostname (for X11 forwarding)")
    else:
        print("✓ Not in SSH session")

def check_gui_dependencies():
    """Check GUI framework dependencies."""
    print("\n=== Checking GUI Dependencies ===")
    
    gui_packages = [
        ('tkinter', 'Tkinter (built-in Python GUI)'),
        ('PyQt5', 'PyQt5'),
        ('PyQt6', 'PyQt6'),
        ('PySide2', 'PySide2'),
        ('PySide6', 'PySide6'),
        ('gradio', 'Gradio (web-based GUI)')
    ]
    
    available_guis = []
    
    for package, description in gui_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}: {description}")
            available_guis.append(package)
        except ImportError:
            print(f"✗ {package}: {description} - not available")
    
    if not available_guis:
        print("❌ No GUI frameworks available!")
        return False
    else:
        print(f"✓ Available GUI frameworks: {available_guis}")
        return True

def check_onetrainer_gui_script():
    """Check OneTrainer GUI script."""
    print("\n=== Checking OneTrainer GUI Script ===")
    
    possible_scripts = [
        'scripts/train_ui.py',
        'train_ui.py',
        'ui.py',
        'main.py',
        'OneTrainer.py'
    ]
    
    found_scripts = []
    
    for script in possible_scripts:
        if os.path.exists(script):
            print(f"✓ Found: {script}")
            found_scripts.append(script)
        else:
            print(f"✗ Not found: {script}")
    
    if not found_scripts:
        print("❌ No GUI script found!")
        return None
    
    return found_scripts[0]

def test_basic_gui():
    """Test basic GUI functionality."""
    print("\n=== Testing Basic GUI ===")
    
    try:
        import tkinter as tk
        print("✓ Tkinter available, testing basic window...")
        
        # Create a simple test window
        root = tk.Tk()
        root.title("OneTrainer GUI Test")
        root.geometry("300x200")
        
        label = tk.Label(root, text="If you see this window,\nGUI is working!")
        label.pack(expand=True)
        
        # Show window briefly
        root.update()
        print("✓ Test window created successfully")
        
        # Close after 2 seconds
        root.after(2000, root.destroy)
        root.mainloop()
        
        return True
        
    except Exception as e:
        print(f"✗ GUI test failed: {e}")
        return False

def check_onetrainer_imports():
    """Check OneTrainer imports."""
    print("\n=== Checking OneTrainer Imports ===")
    
    try:
        # Add current directory to path
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Test basic imports
        from modules.util.enum.ModelType import ModelType
        print("✓ ModelType import successful")
        
        # Check if WAN_2_2 is available
        if hasattr(ModelType, 'WAN_2_2'):
            print("✓ WAN_2_2 model type available")
        else:
            print("⚠ WAN_2_2 model type not found")
        
        return True
        
    except Exception as e:
        print(f"✗ OneTrainer imports failed: {e}")
        return False

def suggest_solutions():
    """Suggest solutions based on findings."""
    print("\n=== Suggested Solutions ===")
    
    print("1. **Try Alternative Launch Methods:**")
    print("   python scripts/train_ui.py")
    print("   python -m scripts.train_ui")
    print("   python OneTrainer.py")
    
    print("\n2. **Check for GUI Process:**")
    print("   ps aux | grep python")
    print("   ps aux | grep OneTrainer")
    
    print("\n3. **Try Web-based Interface (if available):**")
    print("   Look for Gradio or Streamlit interface")
    print("   Check for --port or --host options")
    
    print("\n4. **Environment Issues:**")
    print("   export DISPLAY=:0  # For X11")
    print("   xhost +local:  # Allow local connections")
    
    print("\n5. **Install Missing GUI Dependencies:**")
    print("   pip install PyQt5")
    print("   pip install gradio")
    print("   sudo apt-get install python3-tk  # Ubuntu/Debian")
    
    print("\n6. **Check OneTrainer Documentation:**")
    print("   Look for specific launch instructions")
    print("   Check for environment setup requirements")

def main():
    """Run GUI troubleshooting."""
    print("🔧 OneTrainer GUI Launch Troubleshooting")
    print("=" * 50)
    
    # Run checks
    check_display_environment()
    gui_available = check_gui_dependencies()
    gui_script = check_onetrainer_gui_script()
    onetrainer_imports = check_onetrainer_imports()
    
    if gui_available:
        gui_test = test_basic_gui()
    else:
        gui_test = False
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 TROUBLESHOOTING SUMMARY")
    print("=" * 50)
    
    if gui_available and gui_script and onetrainer_imports:
        print("✅ Basic requirements met")
        
        if gui_test:
            print("✅ GUI functionality working")
            print("\n💡 OneTrainer GUI should work. Try:")
            print(f"   python {gui_script}")
        else:
            print("⚠ GUI test failed - display issues")
    else:
        print("❌ Missing requirements:")
        if not gui_available:
            print("  - GUI framework not available")
        if not gui_script:
            print("  - GUI script not found")
        if not onetrainer_imports:
            print("  - OneTrainer imports failing")
    
    suggest_solutions()

if __name__ == "__main__":
    main()