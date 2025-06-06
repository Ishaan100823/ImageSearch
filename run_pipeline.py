#!/usr/bin/env python3
"""
Convenience script to run the main pipeline from the root directory
"""
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

if __name__ == "__main__":
    from pipeline.main import main
    main() 