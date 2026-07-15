# 首先安装库
# pip install onnx2pytorch

# 保存转换后的模型
import torch
import onnx
from onnx2pytorch import ConvertModel

if __name__ == '__main__':
    # 加载 ONNX 模型
    onnx_model = onnx.load("test_models/yolo11n.onnx")

    # 转换为 PyTorch 模型
    pytorch_model = ConvertModel(onnx_model)

    torch.save(pytorch_model.state_dict(), "yolo11n_320.pt")