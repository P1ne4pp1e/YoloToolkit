import cv2
import numpy as np
import time
import hikcam as hik_cam
from numba import jit
from openvino.runtime import Core

# 定义类别字典
classes = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}


# JIT编译的预处理函数
@jit(nopython=True)
def preprocess_image(input_img, input_width, input_height):
    """JIT编译的预处理函数，用于加速图像处理"""
    # 创建输出数组
    processed = np.empty((3, input_height, input_width), dtype=np.float32)

    # 手动执行归一化和转置操作
    for y in range(input_height):
        for x in range(input_width):
            processed[0, y, x] = input_img[y, x, 0] / 255.0  # R
            processed[1, y, x] = input_img[y, x, 1] / 255.0  # G
            processed[2, y, x] = input_img[y, x, 2] / 255.0  # B

    return processed


# 另一个JIT编译的函数，用于坐标转换
@jit(nopython=True)
def convert_boxes(boxes, x_factor, y_factor):
    """JIT编译的坐标转换函数"""
    result = np.empty((boxes.shape[0], 4), dtype=np.int32)

    for i in range(boxes.shape[0]):
        cx, cy, w, h = boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3]

        # 计算左上角坐标
        left = int((cx - w / 2) * x_factor)
        top = int((cy - h / 2) * y_factor)
        width = int(w * x_factor)
        height = int(h * y_factor)

        # 存储结果
        result[i, 0] = left
        result[i, 1] = top
        result[i, 2] = width
        result[i, 3] = height

    return result


class YOLOv8OpenVINO:
    """使用OpenVINO优化的YOLOv8目标检测模型类"""

    def __init__(self, model_path, confidence_thres=0.6, iou_thres=0.5, device="CPU"):
        """
        初始化YOLOv8 OpenVINO优化版本

        参数:
            model_path: ONNX或IR模型的路径
            confidence_thres: 置信度阈值
            iou_thres: IoU阈值
            device: 推理设备，如"CPU"、"GPU"、"MYRIAD"等
        """
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.device = device

        # 设置输入尺寸
        self.input_width = 320
        self.input_height = 320

        # 为每个类别生成颜色调色板
        np.random.seed(42)  # 设置随机种子以保持颜色一致
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # 加载并优化OpenVINO模型
        self._load_model(model_path)

        # 用于存储计时数据
        self.timing = {}

        # 预热JIT编译函数
        dummy_img = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        _ = preprocess_image(dummy_img, self.input_width, self.input_height)
        _ = convert_boxes(np.array([[0.5, 0.5, 0.1, 0.1]]), 1.0, 1.0)

    def _load_model(self, model_path):
        """加载并优化OpenVINO模型"""
        # 初始化OpenVINO Core
        self.core = Core()

        # 读取模型
        # 如果是ONNX格式，OpenVINO可以直接读取，或者先将其转换为IR格式
        self.model = self.core.read_model(model_path)

        # 配置推理请求
        # 启用性能优化
        self.compiled_model = self.core.compile_model(
            model=self.model,
            device_name=self.device,
            config={
                "PERFORMANCE_HINT": "THROUGHPUT",  # 更改为吞吐量优化
                "NUM_STREAMS": "AUTO",  # 自动选择最佳流数
                "INFERENCE_PRECISION_HINT": "f16"  # 使用FP16精度以提高速度
            }
        )

        # 获取模型输入和输出信息
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        # 获取输入形状
        self.input_shape = self.input_layer.shape

    def preprocess(self, img):
        """使用Numba JIT优化的预处理"""
        start_time = time.time()

        # 获取原始图像尺寸
        self.img_height, self.img_width = img.shape[:2]

        # 调整图像大小
        resized_img = cv2.resize(img, (self.input_width, self.input_height))

        # 转换颜色空间
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        # 使用JIT编译的函数进行预处理
        processed = preprocess_image(rgb_img, self.input_width, self.input_height)

        # 扩展维度
        input_tensor = np.expand_dims(processed, axis=0)

        self.timing['Preprocessing'] = time.time() - start_time
        return input_tensor

    def postprocess(self, image, outputs):
        """后处理优化版本，使用向量化操作代替循环"""
        start_time = time.time()

        # 将输出转换为更易处理的形式 [8400, 84]
        predictions = np.squeeze(outputs).T

        # 获取预测数量
        num_predictions = predictions.shape[0]

        # 使用向量化操作获取最大置信度和相应的类ID
        scores = predictions[:, 4:]  # 所有候选框的所有类别分数 [8400, 80]
        class_ids = np.argmax(scores, axis=1)  # 每个候选框的最佳类别ID [8400]
        confidences = np.max(scores, axis=1)  # 每个候选框的最佳置信度 [8400]

        # 应用置信度阈值筛选
        mask = confidences > self.confidence_thres

        # 提取满足阈值的边界框和类ID
        filtered_boxes = predictions[mask, :4]  # 满足置信度阈值的框 [n, 4]
        filtered_class_ids = class_ids[mask]  # 满足置信度阈值的类ID [n]
        filtered_confidences = confidences[mask]  # 满足阈值的置信度值 [n]

        # 计算缩放因子
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # 计算实际像素坐标 - 向量化操作
        if len(filtered_boxes) > 0:
            # 使用JIT编译的函数转换坐标
            boxes = convert_boxes(filtered_boxes, x_factor, y_factor)
        else:
            boxes = np.array([])

        self.timing['Filtering Candidates'] = time.time() - start_time
        nms_start = time.time()

        # 应用非最大抑制
        indices = []
        if len(boxes) > 0:
            # 将边界框转换为xyxy格式，用于NMS
            boxes_xyxy = np.zeros_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0]  # x1
            boxes_xyxy[:, 1] = boxes[:, 1]  # y1
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2

            indices = cv2.dnn.NMSBoxes(
                boxes_xyxy.tolist(),
                filtered_confidences.tolist(),
                self.confidence_thres,
                self.iou_thres
            )

        self.timing['Non-maximum Suppression'] = time.time() - nms_start
        draw_start = time.time()

        # 绘制检测结果
        result_image = image.copy()
        if len(indices) > 0:
            for i in indices:
                # OpenCV版本兼容性
                if isinstance(i, (list, np.ndarray)):
                    i = i[0]

                # 获取边界框和类别
                x, y, w, h = boxes[i]
                class_id = int(filtered_class_ids[i])
                confidence = filtered_confidences[i]

                # 绘制边界框
                color = tuple(map(int, self.color_palette[class_id % len(self.color_palette)]))
                cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)

                # 绘制标签
                label = f"{self.classes[class_id]}: {confidence:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(result_image, (x, y - text_height - 5), (x + text_width, y), color, -1)
                cv2.putText(result_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        self.timing['Drawing Boxes'] = time.time() - draw_start
        return result_image

    def predict(self, img):
        """执行预测，返回结果图像"""
        # 开始总计时
        total_start = time.time()

        # 重置计时字典
        self.timing = {}

        # 预处理图像
        input_tensor = self.preprocess(img)

        # 执行推理 - 使用异步推理
        infer_start = time.time()
        infer_req = self.compiled_model.create_infer_request()
        infer_req.start_async(inputs={self.input_layer.any_name: input_tensor})
        infer_req.wait()
        outputs = infer_req.get_output_tensor(0).data
        self.timing['Model Inference'] = time.time() - infer_start

        # 后处理并绘制结果
        result_image = self.postprocess(img, outputs)

        # 计算总时间和其他组件所占比例
        self.timing['Total Time'] = time.time() - total_start

        # 计算帧内开销 (非渲染时间)
        non_rendering_time = sum([
            t for step, t in self.timing.items()
            if step != 'Total Time'
        ])
        self.timing['Other Overhead'] = self.timing['Total Time'] - non_rendering_time

        # 显示计时信息
        y_pos = 30
        for step, t in self.timing.items():
            text = f"{step}: {t * 1000:.2f} ms"
            cv2.putText(result_image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 30

        # 打印控制台信息
        for step, t in self.timing.items():
            print(f"{step}: {t * 1000:.2f} ms")
        print("-" * 30)

        return result_image


if __name__ == "__main__":
    # 配置参数
    model_path = "test_models/yolo11n.onnx"  # 或者使用ONNX模型: "yolo11n.onnx"
    confidence_threshold = 0.4
    iou_threshold = 0.7
    device = "CPU"  # 可以根据需要更改为GPU、MYRIAD（NCS2）等

    # 创建检测器实例
    detector = YOLOv8OpenVINO(model_path, confidence_threshold, iou_threshold, device)

    # 用于计算FPS
    fps = 30
    frames = 0
    T1 = time.perf_counter()
    T2 = T1

    # 创建窗口
    # cv2.namedWindow("YOLOv8 OpenVINO", cv2.WINDOW_NORMAL)

    cam = hik_cam.HikCam()
    cam.start_camera()
    cam.set_camera(15.0, 3000)

    while True:
        st = time.time()
        # 读取相机帧
        frame = cam.get_image(False)

        # 执行预测
        result = detector.predict(frame)

        # 计算FPS
        frames += 1
        if frames >= int(fps / 2) + 1:
            T2 = time.perf_counter()
            fps = round(1 / (T2 - T1) * frames, 2)
            T1 = time.perf_counter()
            frames = 0

        # 显示FPS
        cv2.putText(result, f"FPS: {fps}", (10, result.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


        # 显示结果
        cv2.imshow("YOLOv8 OpenVINO", result)
        #
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cv2.destroyAllWindows()