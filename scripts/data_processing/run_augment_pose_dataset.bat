@echo off
chcp 65001 >nul
setlocal

set "CONDA_EXE=D:\anaconda3\Scripts\conda.exe"
set "CONDA_ENV=YOLO"
set "INPUT_DIR=dataset\annotated-20260715-001"
set "OUTPUT_DIR=dataset\augmented-20260716-001"
set "PREVIEW_COUNT=5"
set "RANDOM_SEED=20260716"
set "WORKERS=8"
set "BATCH_SIZE=128"
set "PYTHONIOENCODING=utf-8"

cd /d "%~dp0\..\.."
set "PYTHONPATH=%CD%\src"
"%CONDA_EXE%" run --no-capture-output -n "%CONDA_ENV%" python scripts\data_processing\augment_pose_dataset.py --input "%INPUT_DIR%" --output "%OUTPUT_DIR%" --preview-count %PREVIEW_COUNT% --seed %RANDOM_SEED% --workers %WORKERS% --batch-size %BATCH_SIZE%
if errorlevel 1 goto :error
echo Dataset augmentation completed: %OUTPUT_DIR%
pause
exit /b 0

:error
echo Dataset augmentation failed.
pause
exit /b 1
