#!/usr/bin/env python3
"""
DataGenie AI Co-Pilot - Quick Start Script
"""
import os
import sys
import subprocess
import webbrowser
import time

def check_ollama():
    """Check if Ollama is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            return True
    except:
        print("❌ Ollama is not running or not accessible")
        print("Please start Ollama first:")
        print("  - Download from: https://ollama.ai/")
        print("  - Install and run: ollama serve")
        print("  - Pull SQLCoder model: ollama pull sqlcoder")
        return False

def main():
    print("🚀 DataGenie AI Co-Pilot - Quick Start")
    print("=" * 50)
    
    # Check virtual environment
    if not os.path.exists('venv'):
        print("❌ Virtual environment not found. Please run:")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # or venv\\Scripts\\activate on Windows")
        print("   pip install -r requirements.txt")
        return
    
    # Check dependencies
    try:
        import fastapi
        import uvicorn
        import pandas
        print("✅ All dependencies installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return
    
    # Check Ollama
    if not check_ollama():
        return
    
    print("\n🎯 Starting DataGenie AI Co-Pilot Server...")
    print("📊 Access Points:")
    print("   • Main UI:      http://localhost:8000/ui")
    print("   • Admin:        http://localhost:8000/admin")
    print("   • API Docs:     http://localhost:8000/docs")
    print("   • Health Check: http://localhost:8000/health")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the server
    try:
        import uvicorn
        uvicorn.run("main_ds:app", host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()