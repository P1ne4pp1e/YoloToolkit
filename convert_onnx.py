import hikcam as hik_cam
from ultralytics import YOLO
import cv2

def convert_to_onnx():
    # 加载模型
    model = YOLO("test_models/yolo11n_trained_06.pt")

    # 导出模型为ONNX格式，并指定输出文件名
    model.export(
        format='onnx',  # 导出格式
        imgsz=320,  # 输入图像尺寸
        half=True,  # 启用FP16精度
        simplify=True,  # 简化模型
        opset=12,  # ONNX操作集版本
        batch=1  # 批处理大小
    )

if __name__ == '__main__':
    convert_to_onnx()