@echo off
setlocal

rem Edit these values for another dataset. Leave FILTER_THRESHOLD empty for interactive input.
set "CONDA_ENV=YOLO"
set "INPUT_DIR=%~dp0..\..\dataset\raw-20260715-001-2023E-yolopose"
set "OUTPUT_DIR=%~dp0..\..\dataset\filtered-20260715-001"
set "FILTER_THRESHOLD=0.03"
set "CONSOLE_CODE_PAGE=65001"
set "PYTHONIOENCODING=utf-8"

chcp %CONSOLE_CODE_PAGE% > nul

for %%I in ("%INPUT_DIR%") do set "INPUT_DIR=%%~fI"
for %%I in ("%OUTPUT_DIR%") do set "OUTPUT_DIR=%%~fI"

call conda activate "%CONDA_ENV%"
if errorlevel 1 (
    echo [ERROR] Unable to activate the conda environment: %CONDA_ENV%
    pause
    exit /b 1
)

if defined FILTER_THRESHOLD (
    python "%~dp0filter_motion_blur.py" "%INPUT_DIR%" "%OUTPUT_DIR%" --threshold "%FILTER_THRESHOLD%"
) else (
    python "%~dp0filter_motion_blur.py" "%INPUT_DIR%" "%OUTPUT_DIR%"
)

set "EXIT_CODE=%errorlevel%"
pause
exit /b %EXIT_CODE%
