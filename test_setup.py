#!/usr/bin/env python3
"""Test script to validate Image Search System setup"""
import sys
import importlib

def test_dependencies():
    """Test if all required dependencies are installed"""
    packages = ['transformers', 'torch', 'faiss', 'pandas', 'numpy', 'PIL', 'fastapi', 'streamlit', 'requests', 'tqdm', 'loguru', 'psutil']
    
    print("🔍 Testing Dependencies...")
    missing = []
    
    for pkg in packages:
        try:
            if pkg == 'PIL':
                importlib.import_module('PIL')
            else:
                importlib.import_module(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg}")
            missing.append(pkg)
    
    if missing:
        print(f"\n❌ Missing: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed!")
    return True

def test_files():
    """Test if required files exist"""
    from pathlib import Path
    
    print("🔍 Testing Files...")
    files = ['config.py', 'main.py', 'check-shirts.csv', 'utils/logging_utils.py']
    
    for file in files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")
            return False
    
    print("✅ All files present!")
    return True

def main():
    print("🧪 IMAGE SEARCH SYSTEM - SETUP TEST")
    print("=" * 40)
    
    deps_ok = test_dependencies()
    files_ok = test_files()
    
    if deps_ok and files_ok:
        print("\n🎉 SETUP VALIDATION PASSED!")
        print("\n🚀 Ready to run:")
        print("   python main.py --help")
    else:
        print("\n❌ Setup validation failed!")
    
    return deps_ok and files_ok

if __name__ == "__main__":
    sys.exit(0 if main() else 1) 