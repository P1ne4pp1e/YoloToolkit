@echo off
chcp 65001 >nul
setlocal

set "CONDA_EXE=D:\anaconda3\Scripts\conda.exe"
set "CONDA_ENV=YOLO"
set "INPUT_DIR=dataset\augmented-20260716-001"
set "OUTPUT_DIR=dataset\augmented-20260716-001-split"
set "TRAIN_RATIO=0.8"
set "VAL_RATIO=0.2"
set "TEST_RATIO=0.0"
set "RANDOM_SEED=20260716"
set "PYTHONIOENCODING=utf-8"

cd /d "%~dp0\..\.."
set "PYTHONPATH=%CD%\src"
"%CONDA_EXE%" run --no-capture-output -n "%CONDA_ENV%" python scripts\data_processing\split_pose_dataset.py --input "%INPUT_DIR%" --output "%OUTPUT_DIR%" --train-ratio %TRAIN_RATIO% --val-ratio %VAL_RATIO% --test-ratio %TEST_RATIO% --seed %RANDOM_SEED%
if errorlevel 1 goto :error
echo Dataset split completed: %OUTPUT_DIR%
pause
exit /b 0

:error
echo Dataset split failed.
pause
exit /b 1
