import cv2
import numpy as np
import onnxruntime as ort
import time
import hikcam as hik_cam
from numba import jit

# Define class dictionary
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


class YOLOv8Optimized:
    """Optimized version of YOLOv8 object detection model class"""

    def __init__(self, onnx_model, confidence_thres=0.6, iou_thres=0.5):
        """
        Initialize YOLOv8 optimized version

        Parameters:
            onnx_model: Path to ONNX model
            confidence_thres: Confidence threshold
            iou_thres: IoU threshold
        """
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.classes = classes

        # Set input dimensions
        self.input_width = 320
        self.input_height = 320

        # Generate a color palette for each class
        np.random.seed(42)  # Set random seed for consistent colors
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # Optimize ONNX model
        self._optimize_model(onnx_model)

        # For storing timing data
        self.timing = {}

        # 预热JIT编译函数
        dummy_img = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        _ = preprocess_image(dummy_img, self.input_width, self.input_height)
        _ = convert_boxes(np.array([[0.5, 0.5, 0.1, 0.1]]), 1.0, 1.0)

    def _optimize_model(self, model_path):
        """Optimize ONNX model using ONNX Runtime session options"""
        # Define ONNX Runtime session options
        sess_options = ort.SessionOptions()

        # Enable graph optimization
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Enable parallel execution
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

        # Enable memory optimization
        sess_options.enable_mem_pattern = True
        sess_options.enable_mem_reuse = True

        # Enable CPU thread settings (adjust based on your hardware)
        sess_options.intra_op_num_threads = 8  # Threads for internal operations
        sess_options.inter_op_num_threads = 4  # Threads for node parallelism

        # Create optimized inference session
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

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
        """Postprocessing optimized version, using vectorized operations instead of loops"""
        start_time = time.time()

        # Transpose outputs to a more processable form [8400, 84]
        predictions = np.squeeze(outputs[0]).T

        # Get number of predictions
        num_predictions = predictions.shape[0]

        # Use vectorized operations to get maximum confidence and corresponding class ID
        scores = predictions[:, 4:]  # All class scores for all candidate boxes [8400, 80]
        class_ids = np.argmax(scores, axis=1)  # Best class ID for each candidate box [8400]
        confidences = np.max(scores, axis=1)  # Best confidence for each candidate box [8400]

        # Apply confidence threshold filter
        mask = confidences > self.confidence_thres

        # Extract bounding boxes and class IDs that meet the threshold
        filtered_boxes = predictions[mask, :4]  # Boxes that meet confidence threshold [n, 4]
        filtered_class_ids = class_ids[mask]  # Class IDs that meet confidence threshold [n]
        filtered_confidences = confidences[mask]  # Confidence values that meet threshold [n]

        # Calculate scaling factors
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Calculate actual pixel coordinates - vectorized operations
        if len(filtered_boxes) > 0:
            # 使用JIT编译的函数转换坐标
            boxes = convert_boxes(filtered_boxes, x_factor, y_factor)
        else:
            boxes = np.array([])

        self.timing['Filtering Candidates'] = time.time() - start_time
        nms_start = time.time()

        # Apply non-maximum suppression
        indices = []
        if len(boxes) > 0:
            # Convert bounding boxes to xyxy format for NMS
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

        # Draw detection results
        result_image = image.copy()
        if len(indices) > 0:
            for i in indices:
                # OpenCV version compatibility
                if isinstance(i, (list, np.ndarray)):
                    i = i[0]

                # Get bounding box and class
                x, y, w, h = boxes[i]
                class_id = int(filtered_class_ids[i])
                confidence = filtered_confidences[i]

                # Draw bounding box
                color = tuple(map(int, self.color_palette[class_id % len(self.color_palette)]))
                cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)

                # Draw label
                label = f"{self.classes[class_id]}: {confidence:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(result_image, (x, y - text_height - 5), (x + text_width, y), color, -1)
                cv2.putText(result_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        self.timing['Drawing Boxes'] = time.time() - draw_start
        return result_image

    def predict(self, img):
        """Perform prediction, return result image"""
        # Start total timing
        total_start = time.time()

        # Reset timing dictionary
        self.timing = {}

        # Preprocess image
        input_tensor = self.preprocess(img)

        # Execute inference
        infer_start = time.time()
        outputs = self.session.run(None, {self.input_name: input_tensor})
        self.timing['Model Inference'] = time.time() - infer_start

        # Postprocess and draw results
        result_image = self.postprocess(img, outputs)

        # Calculate total time
        self.timing['Total Time'] = time.time() - total_start

        # Display timing information
        y_pos = 30
        for step, t in self.timing.items():
            text = f"{step}: {t * 1000:.2f} ms"
            cv2.putText(result_image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 30

        # Print console information
        # for step, t in self.timing.items():
        #     print(f"{step}: {t * 1000:.2f} ms")
        # print("-" * 30)

        return result_image


if __name__ == "__main__":
    # Configure parameters
    model_path = "test_models/yolo11n_trained_05.onnx"
    confidence_threshold = 0.7
    iou_threshold = 0.7

    # Create detector instance
    detector = YOLOv8Optimized(model_path, confidence_threshold, iou_threshold)

    # For calculating FPS
    fps = 30
    frames = 0
    T1 = time.perf_counter()
    T2 = T1

    # Create window
    cv2.namedWindow("YOLOv8 Optimized", cv2.WINDOW_NORMAL)

    cam = hik_cam.HikCam()
    cam.start_camera()
    cam.set_camera(15.0, 2000)

    while True:
        # Read camera frame
        frame = cam.get_image(False)

        # Perform prediction
        result = detector.predict(frame)

        # Calculate FPS
        frames += 1
        if frames >= int(fps / 2) + 1:
            T2 = time.perf_counter()
            fps = round(1 / (T2 - T1) * frames, 2)
            T1 = time.perf_counter()
            frames = 0

        # Display FPS
        cv2.putText(result, f"FPS: {fps}", (10, result.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Display result
        cv2.imshow("YOLOv8 Optimized", result)

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cv2.destroyAllWindows()