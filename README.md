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
