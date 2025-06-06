#!/usr/bin/env python3
"""
Convenience script to run the enhanced Streamlit frontend
"""
import sys
import subprocess
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

if __name__ == "__main__":
    subprocess.run([
        "streamlit", "run", "ui/app_streamlit.py",
        "--server.port", "8501",
        "--server.address", "127.0.0.1"
    ]) 