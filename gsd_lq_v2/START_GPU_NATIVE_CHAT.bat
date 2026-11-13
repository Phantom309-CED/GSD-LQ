@echo off
title GSD-LQ v2.0 Native GPU Chat - Phantom Lattice Labs
cd /d "%~dp0"

echo ===============================================================================
echo             GSD-LQ v2.0: Silicon-Native Consumer GPU LLM Engine
echo                        by Phantom Lattice Labs
echo ===============================================================================
echo.

:: 1. Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [!] ERROR: Python 3 was not found on your system PATH.
    echo.
    echo Please install Python 3.10 or higher from:
    echo   https://www.python.org/downloads/
    echo.
    echo IMPORTANT: Make sure to check "Add Python to PATH" during setup.
    echo.
    pause
    exit /b 1
)

:: 2. Auto-install required minimal dependencies
echo [*] Verifying dependencies (numpy, huggingface_hub, gguf)...
python -m pip install -q -r requirements.txt
if %errorlevel% neq 0 (
    echo [!] Warning: Automatic package installation encountered an issue. Attempting launch...
)

:: 3. Launch native GPU engine
echo [*] Initializing GSD-LQ native runtime and scanning GPU hardware...
echo.
python -u run_gpu_native_chat.py

if %errorlevel% neq 0 (
    echo.
    echo [!] Engine exited with status code %errorlevel%.
)
echo.
pause
