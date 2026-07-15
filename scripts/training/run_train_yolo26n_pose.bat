@echo off
chcp 65001 >nul
setlocal

rem Training configuration
set "CONDA_EXE=D:\anaconda3\Scripts\conda.exe"
set "CONDA_ENV=YOLO"
set "DATASET_DIR=dataset\augmented-20260716-001-split"
set "MODEL_WEIGHTS=yolo26n-pose.pt"
set "RUN_PROJECT=runs\pose"
set "RUN_NAME=yolo26n_pose_augmented_20260716_001"
set "EPOCHS=100"
set "IMAGE_SIZE=640"
set "BATCH_SIZE=16"
set "WORKERS=8"
set "DEVICE=0"
set "PATIENCE=30"
set "SEED=20260716"
set "PYTHONIOENCODING=utf-8"
if not defined DRY_RUN set "DRY_RUN=0"

cd /d "%~dp0\..\.."

if not exist "%DATASET_DIR%\images\train" (
    echo Training images not found: %DATASET_DIR%\images\train
    pause
    exit /b 1
)

if not exist "%MODEL_WEIGHTS%" (
    echo Pretrained weights not found: %MODEL_WEIGHTS%
    pause
    exit /b 1
)

if "%DRY_RUN%"=="1" (
    echo Dry run passed. Training command is ready.
    exit /b 0
)

"%CONDA_EXE%" run --no-capture-output -n "%CONDA_ENV%" python scripts\training\train_yolo26n_pose.py ^
    --dataset "%DATASET_DIR%" ^
    --model "%MODEL_WEIGHTS%" ^
    --epochs %EPOCHS% ^
    --imgsz %IMAGE_SIZE% ^
    --batch %BATCH_SIZE% ^
    --workers %WORKERS% ^
    --device %DEVICE% ^
    --patience %PATIENCE% ^
    --seed %SEED% ^
    --project "%RUN_PROJECT%" ^
    --name "%RUN_NAME%"

if errorlevel 1 goto :error
echo.
echo Training completed. Best model: %RUN_PROJECT%\%RUN_NAME%\weights\best.pt
pause
exit /b 0

:error
echo.
echo Training failed. Review the errors above.
pause
exit /b 1
