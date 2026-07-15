# 导入必要的库
import cv2
import hikcam as hik_cam
import numpy as np
import time
import collections
from ultralytics import YOLO
import torch

# 定义常量
IMG_SIZE = 320  # 模型输入尺寸
NUM_KEYPOINTS = 4  # 定义关键点数量为4
MAX_DETECTIONS = 20  # 最大检测数量


# 绘制结果
def draw_results(frame, boxes, keypoints):
    """在帧上绘制检测结果和关键点"""
    result_frame = frame.copy()

    # 关键点颜色
    colors = [
        (255, 0, 0),  # 蓝色
        (0, 255, 0),  # 绿色
        (0, 0, 255),  # 红色
        (255, 255, 0)  # 青色
    ]

    # 关键点连接关系 (根据需要调整)
    connections = [(0, 1), (1, 3), (2, 3), (0, 2)]

    # 绘制每个检测结果
    for i, (box, kpts) in enumerate(zip(boxes, keypoints)):
        x1, y1, x2, y2, conf = box

        # 绘制边界框
        cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # 绘制置信度
        cv2.putText(
            result_frame,
            f"{conf:.2f}",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        # 绘制关键点
        for k, (kx, ky, kp_conf) in enumerate(kpts):
            # 如果关键点置信度足够高
            if kp_conf > 0.5:
                # 绘制关键点
                cv2.circle(result_frame, (int(kx), int(ky)), 5, colors[k], -1)

                # 绘制关键点标签
                cv2.putText(
                    result_frame,
                    f"{k}",
                    (int(kx + 5), int(ky + 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    colors[k],
                    1
                )

        # 绘制关键点之间的连接
        for conn in connections:
            if conn[0] < len(kpts) and conn[1] < len(kpts):
                pt1 = kpts[conn[0]]
                pt2 = kpts[conn[1]]

                # 如果两个关键点都有足够高的置信度
                if pt1[2] > 0.5 and pt2[2] > 0.5:
                    cv2.line(
                        result_frame,
                        (int(pt1[0]), int(pt1[1])),
                        (int(pt2[0]), int(pt2[1])),
                        (255, 255, 255),
                        2
                    )

    return result_frame


# 时间跟踪器类
class TimingTracker:
    def __init__(self, max_points=100):
        self.preprocess_times = collections.deque(maxlen=max_points)
        self.inference_times = collections.deque(maxlen=max_points)
        self.postprocess_times = collections.deque(maxlen=max_points)
        self.total_times = collections.deque(maxlen=max_points)
        self.max_points = max_points

    def add_times(self, preprocess_time, inference_time, postprocess_time):
        self.preprocess_times.append(preprocess_time * 1000)  # 转换为毫秒
        self.inference_times.append(inference_time * 1000)  # 转换为毫秒
        self.postprocess_times.append(postprocess_time * 1000)  # 转换为毫秒
        self.total_times.append((preprocess_time + inference_time + postprocess_time) * 1000)  # 转换为毫秒

    def get_avg_times(self):
        if not self.preprocess_times:
            return 0, 0, 0, 0

        avg_preprocess = sum(self.preprocess_times) / len(self.preprocess_times)
        avg_inference = sum(self.inference_times) / len(self.inference_times)
        avg_postprocess = sum(self.postprocess_times) / len(self.postprocess_times)
        avg_total = sum(self.total_times) / len(self.total_times)

        return avg_preprocess, avg_inference, avg_postprocess, avg_total

    def draw_timing_info(self, frame):
        """在帧上绘制时间信息"""
        result_frame = frame.copy()

        # 获取平均时间
        avg_preprocess, avg_inference, avg_postprocess, avg_total = self.get_avg_times()

        # 绘制时间信息
        cv2.putText(result_frame, f"Preprocess: {avg_preprocess:.2f}ms",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result_frame, f"Inference: {avg_inference:.2f}ms",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(result_frame, f"Postprocess: {avg_postprocess:.2f}ms",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        cv2.putText(result_frame, f"Total: {avg_total:.2f}ms",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result_frame, f"FPS: {1000 / avg_total:.1f}",
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return result_frame


def main():
    # 模型路径 - 使用OpenVINO格式的模型
    model_path = "test_models/yolo11s-pose_trained_05_openvino_model/"  # 替换为实际的OpenVINO模型路径

    # 初始化海康摄像头
    cam = hik_cam.HikCam()
    cam.start_camera()
    cam.set_camera(15.0, 800)  # 设置曝光时间和增益

    # 加载模型
    try:
        model = YOLO(model_path)
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        return

    # 创建时间跟踪器
    timer = TimingTracker(max_points=50)

    print("按'q'键退出程序")

    try:
        while True:
            # 获取图像
            frame = cam.get_image(False)

            # 预处理阶段
            t_preprocess_start = time.time()
            # 调整图像大小以匹配模型输入要求
            resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            t_preprocess = time.time() - t_preprocess_start

            # 推理阶段
            t_inference_start = time.time()
            # 明确指定输入尺寸
            results = model(resized_frame, imgsz=IMG_SIZE, verbose=False)
            t_inference = time.time() - t_inference_start

            # 后处理阶段 - 从YOLO结果中提取边界框和关键点
            t_postprocess_start = time.time()

            boxes = []
            keypoints_list = []

            if len(results) > 0:
                result = results[0]

                # 提取边界框
                if result.boxes is not None and len(result.boxes) > 0:
                    for i in range(len(result.boxes)):
                        # 获取边界框坐标
                        box = result.boxes.xyxy[i].cpu().numpy()
                        conf = result.boxes.conf[i].cpu().numpy()

                        # 调整边界框坐标到原始图像尺寸
                        orig_h, orig_w = frame.shape[:2]
                        scale_x = orig_w / IMG_SIZE
                        scale_y = orig_h / IMG_SIZE

                        x1 = box[0] * scale_x
                        y1 = box[1] * scale_y
                        x2 = box[2] * scale_x
                        y2 = box[3] * scale_y

                        # 添加到边界框列表
                        boxes.append([x1, y1, x2, y2, conf])

                # 提取关键点
                if result.keypoints is not None:
                    keypoints = result.keypoints.data.cpu().numpy()  # [num_det, n_kpts, 3] - (x, y, confidence)

                    for i in range(len(keypoints)):
                        kpts = []
                        for k in range(NUM_KEYPOINTS):
                            if k < keypoints.shape[1]:  # 确保索引不越界
                                x, y, conf = keypoints[i, k]

                                # 调整关键点坐标到原始图像尺寸
                                x = x * scale_x
                                y = y * scale_y

                                kpts.append((x, y, conf))
                            else:
                                # 如果关键点不存在，添加无效的关键点
                                kpts.append((0, 0, 0))

                        keypoints_list.append(kpts)

            t_postprocess = time.time() - t_postprocess_start

            # 添加时间
            timer.add_times(t_preprocess, t_inference, t_postprocess)

            # 绘制结果
            result_frame = draw_results(frame, boxes, keypoints_list)
            result_frame = timer.draw_timing_info(result_frame)

            # 显示帧
            cv2.imshow("YOLO-Pose 实时检测", result_frame)

            # 按'q'键退出
            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        # 关闭窗口
        cv2.destroyAllWindows()
        print("程序已安全退出")


if __name__ == "__main__":
    main()