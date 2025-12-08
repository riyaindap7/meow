#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import uvicorn
import uvicorn

if __name__ == "__main__":
    # Run the FastAPI app
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(current_dir)]
    )