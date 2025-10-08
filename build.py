import PyInstaller.__main__
import os
import platform
import shutil
from pathlib import Path

def build_app():
    print("🤖 Building DataGenie Desktop v0.0.1...")
    
    # Clean previous builds
    for folder in ['dist', 'build']:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"🧹 Cleaned {folder}/")
    
    # Common arguments
    common_args = [
        'main.py',
        '--name=DataGenie',
        # '--onefile',  # Comment out for debugging
        '--windowed',
        '--noconsole',
        '--log-level=DEBUG',
        '--add-data=main_ds.py;.',
        '--add-data=authentication_prompts.json;.',
        '--add-data=config_personal.json;.',
        '--add-data=config_business.json;.',
        '--add-data=data;data',
        '--add-data=C:\\Users\\xds\\AppDataGenie\\venv\\Lib\\site-packages\\PyQt6\\Qt6\\translations;PyQt6\\Qt6\\translations',
        '--add-data=C:\\Users\\xds\\AppDataGenie\\venv\\Lib\\site-packages\\PyQt6\\Qt6\\resources;PyQt6\\Qt6\\resources',
        '--add-data=C:\\Users\\xds\\AppDataGenie\\venv\\Lib\\site-packages\\PyQt6\\Qt6\\bin;PyQt6\\Qt6\\bin',
        
        # FastAPI & Uvicorn
        '--hidden-import=uvicorn',
        '--hidden-import=uvicorn.loops.asyncio',
        '--hidden-import=uvicorn.loops.auto',
        '--hidden-import=uvicorn.protocols.http',
        '--hidden-import=uvicorn.protocols.websockets',
        '--hidden-import=uvicorn.lifespan.on',
        '--hidden-import=fastapi',
        '--hidden-import=starlette',
        '--hidden-import=starlette.applications',
        '--hidden-import=starlette.routing',
        '--hidden-import=starlette.responses',
        '--hidden-import=starlette.middleware',
        '--hidden-import=websockets',
        '--hidden-import=httpx',
        
        # Pydantic
        '--hidden-import=pydantic',
        '--hidden-import=pydantic.functional_validators',
        '--hidden-import=pydantic.json',
        '--hidden-import=pydantic.types',
        '--hidden-import=pydantic.networks',
        
        # PyQt6
        '--hidden-import=PyQt6',
        '--hidden-import=PyQt6.QtCore',
        '--hidden-import=PyQt6.QtWidgets',
        '--hidden-import=PyQt6.QtGui',
        '--hidden-import=PyQt6.QtWebEngineWidgets',
        '--hidden-import=PyQt6.QtWebEngineCore',
        '--hidden-import=PyQt6.QtNetwork',
        '--hidden-import=PyQt6.QtWebChannel',
        '--hidden-import=PyQt6.QtWebEngineQuick',
        
        # Standard Library and Others
        '--hidden-import=asyncio',
        '--hidden-import=requests',
        '--hidden-import=pandas',
        '--hidden-import=numpy',
        '--hidden-import=yaml',
        '--hidden-import=multiprocessing',
        '--hidden-import=ssl',
        '--hidden-import=hashlib',
        '--hidden-import=pathlib',
    ]
    
    # Platform-specific arguments
    if platform.system() == 'Windows':
        common_args.append('--icon=assets/icon.ico')
    
    print("🔨 Running PyInstaller...")
    try:
        PyInstaller.__main__.run(common_args)
        print("✅ PyInstaller build completed!")
    except Exception as e:
        print(f"❌ Build failed: {e}")
        return
    
    # Verify the build
    exe_path = Path('dist/DataGenie/DataGenie.exe')
    if exe_path.exists():
        file_size = exe_path.stat().st_size / (1024 * 1024)  # MB
        print(f"📦 Executable created: {exe_path} ({file_size:.1f} MB)")
        print("✅ Build successful!")
    else:
        print("❌ Executable not found!")
    
    print("\n🎉 Build completed! Check 'dist' folder for DataGenie")

if __name__ == "__main__":
    build_app()