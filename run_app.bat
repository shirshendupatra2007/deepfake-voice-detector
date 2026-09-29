@echo off
title AURA SHIELD - Real-Time Deepfake Voice Detector
color 0b
echo ===================================================================
echo          AURA SHIELD // DEEPFAKE VOICE DETECTOR
echo          Hackathon Ready - Real-Time AI Audio Defense
echo ===================================================================
echo.
echo [1/3] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found on PATH. Please install Python 3.10+ or check PATH.
    pause
    exit /b
)

echo [2/3] Verified dependencies: PyTorch, librosa, ONNX Runtime, FastAPI...
echo [3/3] Starting Local Web Server on http://127.0.0.1:8000 ...
echo.
echo Browser will open automatically. Keep this window open while testing.
echo Press Ctrl+C to stop the server anytime.
echo.

cd /d "%~dp0"
python app.py
pause
