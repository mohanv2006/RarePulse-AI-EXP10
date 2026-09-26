"""
RarePulse AI - Entry Point Server Runner
========================================
Run this file to launch the RarePulse AI Clinical Full-Stack Platform.

Usage:
    python run_server.py
"""

import sys
import uvicorn

if __name__ == "__main__":
    print("=" * 80)
    print("                     STARTING RAREPULSE AI PLATFORM")
    print("=" * 80)
    print("  - Backend Server & REST API : http://127.0.0.1:8000")
    print("  - Clinical Web Dashboard    : http://127.0.0.1:8000/")
    print("  - Interactive API Docs      : http://127.0.0.1:8000/docs")
    print("=" * 80)
    print("Press CTRL+C to stop the server.\n")

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
