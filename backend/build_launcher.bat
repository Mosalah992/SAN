@echo off
REM build_launcher.bat — Builds sancta_launcher.exe
REM Run from the sancta-main directory (project root) or backend/

echo.
echo  ⬡ Building Sancta Launcher...
echo.

REM Ensure we're in backend/ where sancta_launcher.spec and sancta_launcher.py live
cd /d "%~dp0"
if not exist "sancta_launcher.spec" (
    echo  ERROR: Run from project root or backend/ — sancta_launcher.spec not found
    pause
    exit /b 1
)

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found in PATH
    pause
    exit /b 1
)

REM Install PyInstaller if needed
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo  Installing PyInstaller...
    pip install pyinstaller
)

REM Install requests if needed  
pip show requests >nul 2>&1
if errorlevel 1 (
    echo  Installing requests...
    pip install requests
)

REM Clean old build
if exist "dist\sancta_launcher.exe" del "dist\sancta_launcher.exe"
if exist "build" rmdir /s /q build

REM Build
echo  Building exe...
pyinstaller sancta_launcher.spec --clean

if errorlevel 1 (
    echo.
    echo  BUILD FAILED — check output above
    pause
    exit /b 1
)

echo.
echo  ✓ Built: dist\sancta_launcher.exe
echo.
echo  IMPORTANT: Place the exe in the project (or keep in backend\dist\).
echo  It needs the backend/ folder with sancta.py, siem_server.py, etc.
echo.
echo  To run: dist\sancta_launcher.exe
echo  Or double-click it from Windows Explorer.
echo.
pause
