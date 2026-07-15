@echo off
setlocal

rem Capture configuration
set "ENABLE_CAPTURE=1"
set "OUTPUT_DIR=dataset\raw-20260715-001-2023E-yolopose"
set "GAIN=15"
set "EXPOSURE=10000"
set "INTERVAL=0.2"
set "MAX_IMAGES=10000"
set "IMAGE_FORMAT=png"

if not "%ENABLE_CAPTURE%"=="0" if not "%ENABLE_CAPTURE%"=="1" (
    echo ENABLE_CAPTURE must be 0 or 1.
    pause
    exit /b 1
)

set "CAPTURE_OPTION="
if "%ENABLE_CAPTURE%"=="0" set "CAPTURE_OPTION=--preview-only"

cd /d "%~dp0\..\.."
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

py -3 scripts\data_collection\hikvision_capture.py ^
    --output "%OUTPUT_DIR%" ^
    --gain "%GAIN%" ^
    --exposure "%EXPOSURE%" ^
    --interval "%INTERVAL%" ^
    --max-images "%MAX_IMAGES%" ^
    --format "%IMAGE_FORMAT%" %CAPTURE_OPTION%
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Capture script failed with exit code %EXIT_CODE%.
)

pause
exit /b %EXIT_CODE%
