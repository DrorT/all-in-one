#!/usr/bin/env python3
"""
Audio Processing HTTP Server Startup Script
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path to enable imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from server import app
import uvicorn

def main():
    """Run the audio processing server"""
    print("🎵 Audio Processing HTTP Server")
    print("=" * 40)
    print("Server starting on http://localhost:8000")
    print("API Documentation available at http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    print("=" * 40)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main()
