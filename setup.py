#!/usr/bin/env python3
"""
Setup script for Image Search System
Handles PyTorch installation and environment setup
"""
import subprocess
import sys
import platform
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def install_pytorch():
    """Install PyTorch based on platform and preferences"""
    system = platform.system().lower()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    print(f"🖥️ Detected: {system} with Python {python_version}")
    
    # PyTorch installation commands for different platforms
    if system == "darwin":  # macOS
        pytorch_cmd = "pip install torch torchvision torchaudio"
    elif system == "linux":
        # Check if CUDA is available
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, check=True)
            pytorch_cmd = "pip install torch torchvision torchaudio"
            print("🚀 CUDA detected - installing GPU version")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
            print("💻 No CUDA detected - installing CPU version")
    elif system == "windows":
        pytorch_cmd = "pip install torch torchvision torchaudio"
    else:
        # Fallback to CPU version
        pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    return run_command(pytorch_cmd, "Installing PyTorch")

def install_requirements():
    """Install other requirements"""
    return run_command("pip install -r requirements.txt", "Installing other dependencies")

def test_installation():
    """Test if installation was successful"""
    print("🧪 Testing installation...")
    
    test_imports = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("faiss", "FAISS"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("fastapi", "FastAPI"),
        ("streamlit", "Streamlit"),
        ("requests", "Requests"),
        ("tqdm", "TQDM"),
        ("loguru", "Loguru"),
        ("psutil", "PSUtil"),
    ]
    
    all_good = True
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name}")
            all_good = False
    
    return all_good

def main():
    """Main setup function"""
    print("🚀 IMAGE SEARCH SYSTEM - SETUP")
    print("=" * 40)
    
    # Check if we're in a virtual environment
    if sys.prefix == sys.base_prefix:
        print("⚠️  Warning: Not in a virtual environment")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Please create a virtual environment first:")
            print("  python -m venv .venv")
            print("  source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate")
            return False
    else:
        print("✅ Virtual environment detected")
    
    # Step 1: Install PyTorch
    if not install_pytorch():
        return False
    
    # Step 2: Install other requirements
    if not install_requirements():
        return False
    
    # Step 3: Test installation
    if not test_installation():
        print("\n❌ Some packages failed to install correctly")
        return False
    
    print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("\n🚀 Next steps:")
    print("  1. Test setup: python test_setup.py")
    print("  2. Run pipeline: python main.py")
    print("  3. Start servers: python main.py --servers")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 