#!/usr/bin/env python3
"""
Convenience script to run training from the root directory
"""
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

if __name__ == "__main__":
    from training.train_model import main
    main() 