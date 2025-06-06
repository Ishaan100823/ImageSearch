#!/usr/bin/env python3
"""
Convenience script to run the enhanced API from the root directory
"""
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.enhanced_api:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    ) 