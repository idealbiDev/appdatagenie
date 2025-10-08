##main.py
import sys
import os
import traceback
from pathlib import Path

def handle_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    
    # Write to log file
    log_path = Path("datagenie_crash.log")
    with open(log_path, "w") as f:
        f.write(f"DataGenie Crash Report\n")
        f.write(f"Error: {error_msg}\n")
    
    print(f"💥 CRITICAL ERROR - Check {log_path} for details")
    print(error_msg)
    
    # Keep window open if not frozen
    if not getattr(sys, 'frozen', False):
        input("Press Enter to exit...")

# Set global exception handler
sys.excepthook = handle_exception

def main():
    try:
        # Add current directory to path for frozen apps
        if getattr(sys, 'frozen', False):
            app_path = Path(sys.executable).parent
            os.chdir(app_path)
            sys.path.insert(0, str(app_path))
        
        from PyQt6.QtWidgets import QApplication, QMainWindow
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        from PyQt6.QtCore import QUrl, QTimer
        import threading
        import uvicorn
        from main_ds import app as fastapi_app
        
        class MainWindow(QMainWindow):
            def __init__(self):
                super().__init__()
                self.setWindowTitle("DataGenie AI Co-Pilot 🤖")
                self.resize(1400, 900)
                
                # Create browser
                self.browser = QWebEngineView()
                self.setCentralWidget(self.browser)
                
                # Start server
                self.start_fastapi_server()
                
                # Load page after delay
                QTimer.singleShot(2000, self.load_app)
            
            def start_fastapi_server(self):
                def run_server():
                    try:
                        print("🌐 Starting FastAPI server...")
                        uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, log_level="warning")
                    except Exception as e:
                        print(f"❌ Server error: {e}")
                        traceback.print_exc()
                
                self.server_thread = threading.Thread(target=run_server, daemon=True)
                self.server_thread.start()
                print("✅ Server thread started")
            
            def load_app(self):
                try:
                    print("🌐 Loading web interface...")
                    self.browser.load(QUrl("http://127.0.0.1:8000/ui"))
                    print("✅ Web interface loaded")
                except Exception as e:
                    print(f"❌ Browser error: {e}")
                    traceback.print_exc()
        
        # Create QApplication instance
        app = QApplication(sys.argv)
        app.setApplicationName("DataGenie")
        app.setApplicationVersion("0.0.1")
        
        # Create and show main window
        window = MainWindow()
        window.show()
        
        print("🎯 Application started successfully!")
        
        # Run the application
        return app.exec()
        
    except Exception as e:
        print(f"💥 Failed to start: {e}")
        traceback.print_exc()
        if not getattr(sys, 'frozen', False):
            input("Press Enter to exit...")
        return 1

if __name__ == "__main__":
    sys.exit(main())