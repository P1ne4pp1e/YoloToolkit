# YoloToolkit

## Project layout

The repository is organized by responsibility:

- `src/yolo_toolkit/camera`: Hikvision camera integration and vendor SDK bindings.
- `src/yolo_toolkit/conversion`: ONNX, OpenVINO, PyTorch, and K210 conversion tools.
- `scripts/inference`: runnable demos and real-time inference processes.
- `scripts/system`: host performance tuning scripts.
- `dataset/`, `images/`, `test_models/`, `output/`, and caches: local-only artifacts ignored by Git.

Install the package in editable mode before running scripts:

```bash
conda activate YOLO
python -m pip install -e .
python scripts/inference/yolo_pose_realtime.py --help
```

The inference scripts intentionally keep their hardware-specific dependencies optional. Use the script matching the installed backend: Ultralytics, ONNX Runtime, OpenVINO, or OpenCV DNN.

YOLO 全流程工具箱，面向模型训练、导出与转换、数据采集、数据清洗和数据增强等工作。

## 当前内容

- YOLO 推理与测试脚本
- PyTorch、ONNX、OpenVINO 等模型转换脚本
- 相机采集与相关设备接口代码
- OpenVINO 和 ONNX 推理示例

## 目录说明

本地数据集、图片、模型文件、推理输出和缓存默认不会提交到 Git，请根据实际环境配置路径。

## 环境

具体依赖取决于使用的模型格式和推理后端，建议在对应脚本中确认所需 Python 包及运行时版本。

## 数据集运动模糊筛选

针对 LabelMe 格式数据集，可按每张有标注图片中 `rectangle` 目标 ROI 的拉普拉斯方差倒数筛选运动模糊图片。模糊分数越大代表越模糊；无对应 JSON 的图片始终保留。

```powershell
py scripts/data_processing/filter_motion_blur.py `
  dataset/raw-20260715-001-2023E-yolopose `
  dataset/filtered-20260715-001 `
  --no-show
```

也可使用会自动激活 `YOLO` conda 环境的批处理脚本：

```bat
scripts\data_processing\run_filter_motion_blur.bat
```

输入目录、输出目录、`YOLO` conda 环境名和可选阈值均在批处理文件开头配置。

脚本会保存分布图、输出多个候选阈值的过滤占比和张数，并在终端等待输入阈值。也可以直接通过 `--threshold 0.001` 非交互执行。
