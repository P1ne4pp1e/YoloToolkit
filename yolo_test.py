# 导入必要的库
import cv2
import hikcam as hik_cam
from ultralytics import YOLO
import torch
import torch.nn as nn
import time
import collections


# 使用torch.jit.script装饰器创建预处理函数
@torch.jit.script
def preprocess_image(img: torch.Tensor, img_size: int = 320, stride: int = 32):
    # 将图像转换为(1, C, H, W)格式
    img = img.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    # 标准化
    img = img / 255.0

    # 确保尺寸是stride的倍数
    h, w = img.shape[2:]
    h_f = float(h) / float(stride)
    w_f = float(w) / float(stride)
    h = int(torch.ceil(torch.tensor(h_f)) * stride)
    w = int(torch.ceil(torch.tensor(w_f)) * stride)

    # 调整大小，保持纵横比
    scale = min(float(img_size) / float(h), float(img_size) / float(w))
    nh, nw = int(h * scale), int(w * scale)

    # 调整大小和填充
    img = torch.nn.functional.interpolate(img, size=(nh, nw), mode='bilinear', align_corners=False)

    # 创建填充后的图像
    pad_img = torch.zeros(1, 3, img_size, img_size, dtype=img.dtype, device=img.device)
    pad_img[:, :, :nh, :nw] = img

    return pad_img


# 创建时间记录的类
class TimingTracker:
    def __init__(self, max_points=100):
        self.preprocess_times = collections.deque(maxlen=max_points)
        self.inference_times = collections.deque(maxlen=max_points)
        self.total_times = collections.deque(maxlen=max_points)
        self.max_points = max_points

    def add_times(self, preprocess_time, inference_time):
        self.preprocess_times.append(preprocess_time * 1000)  # 转换为毫秒
        self.inference_times.append(inference_time * 1000)  # 转换为毫秒
        self.total_times.append((preprocess_time + inference_time) * 1000)  # 转换为毫秒

    def get_avg_times(self):
        if not self.preprocess_times:
            return 0, 0, 0

        avg_preprocess = sum(self.preprocess_times) / len(self.preprocess_times)
        avg_inference = sum(self.inference_times) / len(self.inference_times)
        avg_total = sum(self.total_times) / len(self.total_times)

        return avg_preprocess, avg_inference, avg_total

    def plot_timing_graph(self, frame, width=400, height=200, x_offset=20, y_offset=50):
        # 创建绘图区域
        graph_img = frame.copy()

        # 如果没有数据，返回原始帧
        if not self.preprocess_times:
            return frame

        # 绘制背景和边框
        cv2.rectangle(graph_img, (x_offset, y_offset),
                      (x_offset + width, y_offset + height), (0, 0, 0), -1)
        cv2.rectangle(graph_img, (x_offset, y_offset),
                      (x_offset + width, y_offset + height), (255, 255, 255), 1)

        # 获取最大值用于缩放
        max_value = max(max(self.preprocess_times), max(self.inference_times))
        max_value = max(max_value, 1.0)  # 避免除以零

        # 绘制网格线
        for i in range(1, 5):
            y = y_offset + height - int((i / 5) * height)
            cv2.line(graph_img, (x_offset, y), (x_offset + width, y),
                     (100, 100, 100), 1)
            label = f"{int(max_value * i / 5)}ms"
            cv2.putText(graph_img, label, (x_offset - 40, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 绘制图例
        cv2.line(graph_img, (x_offset + 10, y_offset + 15),
                 (x_offset + 30, y_offset + 15), (0, 255, 0), 2)
        cv2.putText(graph_img, "Preprocess", (x_offset + 35, y_offset + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.line(graph_img, (x_offset + 120, y_offset + 15),
                 (x_offset + 140, y_offset + 15), (0, 0, 255), 2)
        cv2.putText(graph_img, "Inference", (x_offset + 145, y_offset + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 绘制时间曲线
        points_count = len(self.preprocess_times)
        point_spacing = width / min(points_count, self.max_points)

        # 绘制预处理时间曲线
        for i in range(1, points_count):
            if i >= self.max_points:
                break

            x1 = x_offset + int((i - 1) * point_spacing)
            y1 = y_offset + height - int((self.preprocess_times[i - 1] / max_value) * height)
            x2 = x_offset + int(i * point_spacing)
            y2 = y_offset + height - int((self.preprocess_times[i] / max_value) * height)

            cv2.line(graph_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制推理时间曲线
        for i in range(1, points_count):
            if i >= self.max_points:
                break

            x1 = x_offset + int((i - 1) * point_spacing)
            y1 = y_offset + height - int((self.inference_times[i - 1] / max_value) * height)
            x2 = x_offset + int(i * point_spacing)
            y2 = y_offset + height - int((self.inference_times[i] / max_value) * height)

            cv2.line(graph_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 绘制平均时间
        avg_preprocess, avg_inference, avg_total = self.get_avg_times()
        cv2.putText(graph_img, f"Avg Preprocess: {avg_preprocess:.2f}ms",
                    (x_offset, y_offset + height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(graph_img, f"Avg Inference: {avg_inference:.2f}ms",
                    (x_offset, y_offset + height + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(graph_img, f"Avg Total: {avg_total:.2f}ms",
                    (x_offset, y_offset + height + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return graph_img


# 主程序
if __name__ == "__main__":
    # 加载YOLOv8n模型
    model = YOLO('test_models/yolo11n_trained_05.pt')  # 从本地加载预训练模型

    cam = hik_cam.HikCam()
    cam.start_camera()
    cam.set_camera(15.0, 2000)

    device = torch.device('cuda')

    # 创建时间跟踪器
    timer = TimingTracker(max_points=100)

    # 预热JIT函数
    dummy_input = torch.zeros(480, 640, 3, dtype=torch.float32, device=device)
    _ = preprocess_image(dummy_input)

    while True:
        # 读取摄像头帧
        frame = cam.get_image(False)

        # 转换为RGB

        # 转换为tensor
        frame_tensor = torch.from_numpy(frame).float().to(device)

        # 测量预处理时间
        t0 = time.time()
        with torch.no_grad():
            processed_frame = preprocess_image(frame_tensor)
        t1 = time.time()
        preprocess_time = t1 - t0

        # 测量推理时间
        t0 = time.time()
        results = model(processed_frame, verbose=True)
        t1 = time.time()
        inference_time = t1 - t0

        # 添加时间到跟踪器
        timer.add_times(preprocess_time, inference_time)

        # 绘制结果
        annotated_frame = results[0].plot()

        # 绘制时间图表
        # final_frame = timer.plot_timing_graph(annotated_frame)
        final_frame = annotated_frame

        # 在当前帧上显示时间信息
        cv2.putText(final_frame, f"Preprocess: {preprocess_time * 1000:.2f}ms",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(final_frame, f"Inference: {inference_time * 1000:.2f}ms",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(final_frame, f"Total: {(preprocess_time + inference_time) * 1000:.2f}ms",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 显示帧
        cv2.imshow("YOLOv8 实时检测与时间跟踪", final_frame)

        # 按'q'键退出
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()