import os.path
import time
import numpy as np
import cv2
import platform
import multiprocessing
from multiprocessing import Process, shared_memory
from multiprocessing.managers import SharedMemoryManager
import threading
import argparse
import psutil
import torch
from ultralytics import YOLO  # 导入Ultralytics YOLO库

import os

os.environ['OPENCV_VIDEOIO_PRIORITY_LIST'] = 'GSTREAMER,V4L2'
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# 尝试导入海康摄像头模块
try:
    from yolo_toolkit.camera import hikvision as hik_cam

    HIKCAM_AVAILABLE = True
except ImportError:
    HIKCAM_AVAILABLE = False
    print("hikcam模块不可用，将使用OpenCV摄像头")


# 硬件加速探测
def detect_accelerators():
    """检测并配置可用加速硬件"""
    devices = {}

    # 检查CUDA是否可用
    if torch.cuda.is_available():
        devices["GPU"] = torch.cuda.get_device_name(0)
        print(f"CUDA GPU加速可用: {devices['GPU']}")

        # 获取GPU详细信息
        try:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # 转换为GB
            print(f"GPU内存: {gpu_mem:.2f} GB")
        except:
            pass
    else:
        print("CUDA GPU不可用，将使用CPU")
        devices["CPU"] = "PyTorch CPU"

        # 检查CPU架构和特性
        try:
            import subprocess
            if platform.system() == 'Linux':
                cpu_info = subprocess.check_output("lscpu | grep -E 'Model name|MHz|CPU(s)'", shell=True)
                print(f"CPU信息:\n{cpu_info.decode()}")
        except:
            pass

        # 检查MPS（MacOS上的GPU加速）
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices["MPS"] = "Apple Silicon GPU"
            print("Apple Silicon GPU加速可用")

    return devices


# 设置进程亲和性
def set_process_affinity(pid, cpu_list):
    system = platform.system()

    try:
        if system == 'Linux':
            # 使用taskset命令
            cpu_str = ','.join(map(str, cpu_list))
            os.system(f"taskset -pc {cpu_str} {pid} > /dev/null 2>&1")
            print(f"已将进程{pid}绑定到CPU核心{cpu_list}")
        elif system == 'Windows':
            import psutil
            p = psutil.Process(pid)
            p.cpu_affinity(cpu_list)
            print(f"已将进程{pid}绑定到CPU核心{cpu_list}")
        else:
            print(f"不支持的操作系统: {system}")
    except Exception as e:
        print(f"设置进程亲和性失败: {e}")


# 设置系统优先级和优化
def set_system_priorities():
    import os, psutil

    try:
        # 获取当前进程
        p = psutil.Process(os.getpid())

        # 在Linux上使用nice值（-20到19，越低优先级越高）
        if platform.system() == 'Linux':
            p.nice(-10)
            print("已设置进程优先级")
    except Exception as e:
        print(f"无法设置进程优先级: {e}")

    # 仅在Linux上执行的优化
    if platform.system() == 'Linux':
        try:
            # 设置CPU性能模式
            os.system(
                "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1")

            # 清除缓存
            os.system("echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1")

            print("已应用系统优化")
        except:
            print("无法应用系统优化，可能需要root权限")


def optimize_opencv():
    """优化OpenCV性能，避免使用Qt"""
    # 显式禁用Qt后端
    os.environ['QT_QPA_PLATFORM'] = 'xcb'

    # 设置OpenCV后端为GTK或其他非Qt后端
    try:
        cv2.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    except:
        pass

    # 启用OpenCL加速
    cv2.ocl.setUseOpenCL(True)

    # 优化线程数量，根据CPU核心数设置
    cpu_count = os.cpu_count()
    if cpu_count:
        cv2.setNumThreads(cpu_count)

    print(f"OpenCV优化: OpenCL状态={cv2.ocl.useOpenCL()}, 线程数={cv2.getNumThreads()}")


# 性能监控UI
def add_performance_overlay(frame, metrics):
    """添加性能监控覆盖层 (纯OpenCV实现)"""
    h, w = frame.shape[:2]

    # 创建半透明背景 (纯OpenCV方式)
    overlay_region = frame[10:150, 10:260].copy()
    cv2.rectangle(frame, (10, 10), (260, 150), (0, 0, 0), -1)

    # 混合 - 手动实现alpha混合
    alpha = 0.7
    cv2.addWeighted(
        frame[10:150, 10:260], alpha,
        overlay_region, 1 - alpha, 0,
        frame[10:150, 10:260]
    )

    # 绘制指标
    cv2.putText(frame, f"CAM FPS: {metrics['cam_fps']}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"INFER FPS: {metrics['infer_fps']:.1f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"INFER TIME: {metrics['infer_time']:.1f}ms", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"KEYPOINTS: {metrics.get('keypoints_count', 0)}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # CPU使用率
    if 'cpu_usage' in metrics:
        color = (0, 255, 0) if metrics['cpu_usage'] < 80 else (0, 0, 255)
        cv2.putText(frame, f"CPU: {metrics['cpu_usage']:.1f}%", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


def check_display_available():
    """检查是否可以使用显示功能"""
    print(f"DISPLAY环境变量: {os.environ.get('DISPLAY', '未设置')}")
    print(f"QT_QPA_PLATFORM环境变量: {os.environ.get('QT_QPA_PLATFORM', '未设置')}")
    try:
        # 尝试创建小窗口测试
        test_img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imshow('Test', test_img)
        cv2.waitKey(1)
        cv2.destroyWindow('Test')
        print("显示功能测试成功")
        return True
    except Exception as e:
        print(f"无法初始化显示: {e}")
        return False


class YOLOPoseInference:
    def __init__(self, model_path, confidence_thres=0.4, iou_thres=0.5, device="cpu"):
        """初始化YOLO-Pose推理类"""
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.device = device

        # 检查模型文件
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")

        # 初始化YOLO模型
        print(f"正在加载模型: {model_path}")
        t_start = time.time()
        self.model = YOLO(model_path)
        print(f"模型加载完成，耗时: {time.time() - t_start:.2f}秒")

        # 设置设备
        if device.lower() == "gpu" and torch.cuda.is_available():
            self.device = "cuda"
        elif device.lower() == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"使用设备: {self.device}")

        # 获取输入尺寸 (尝试获取模型的默认输入尺寸)
        try:
            self.input_width = self.model.model.args['imgsz']
            self.input_height = self.model.model.args['imgsz']
            if isinstance(self.input_width, list):
                self.input_width = self.input_width[0]
                self.input_height = self.input_height[0]
        except:
            # 使用默认尺寸
            self.input_height, self.input_width = 320, 320

        print(f"模型输入尺寸: {self.input_width}x{self.input_height}")

        # 设置关键点配置 - 自定义4点模型
        self.keypoint_names = [
            "lu", "ld", "ru", "rd"
        ]

        # 设置骨架连接
        self.skeleton = [
            [0, 1], [0, 2], [1, 3], [2, 3]
        ]

        # 骨架颜色
        self.skeleton_colors = [
            (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0)
        ]

        # 关键点颜色
        self.keypoint_colors = [
            (0, 255, 255), (0, 255, 0), (0, 0, 255), (255, 0, 0)
        ]

        # FPS计算
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()

    def infer(self, img):
        """执行推理并绘制姿态估计结果"""
        # 记录原始图像和尺寸
        original_img = img.copy()
        self.orig_height, self.orig_width = img.shape[:2]

        # 推理
        start_time = time.time()
        results = self.model(img, conf=self.confidence_thres, iou=self.iou_thres, device=self.device)
        infer_time = (time.time() - start_time) * 1000  # 转换为毫秒

        # 处理结果
        processed_img, keypoints_count = self.postprocess(original_img, results)

        # 更新FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time

        return processed_img, self.fps, 0, infer_time, 0, keypoints_count

    def postprocess(self, img, results):
        """后处理并绘制姿态关键点"""
        # 创建结果图像
        result_img = img.copy()
        keypoints_count = 0

        # 处理每个结果
        for result in results:
            # 检查是否有检测结果
            if len(result.boxes) > 0:
                # 获取边界框坐标
                boxes = result.boxes.xyxy.cpu().numpy()
                # 获取置信度
                confs = result.boxes.conf.cpu().numpy()

                print(boxes, confs)

                # 绘制每个检测框
                for i, (box, conf) in enumerate(zip(boxes, confs)):
                    # 提取边界框坐标
                    x1, y1, x2, y2 = map(int, box)

                    # 绘制矩形边界框
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 添加置信度标签
                    label = f"{conf:.2f}"
                    cv2.putText(result_img, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 获取关键点信息
            if result.keypoints is not None:
                keypoints = result.keypoints.data.cpu().numpy()  # [num_det, n_kpts, 3] - (x, y, confidence)

                # 为每个检测结果绘制关键点
                for kpts in keypoints:
                    valid_kps = []

                    # 遍历每个关键点
                    for k, (kp_x, kp_y, kp_conf) in enumerate(kpts):
                        # 只处理有效的关键点
                        if kp_conf > 0.2 and 0 <= kp_x < self.orig_width and 0 <= kp_y < self.orig_height:
                            # 绘制关键点
                            color = self.keypoint_colors[k % len(self.keypoint_colors)]
                            cv2.circle(result_img, (int(kp_x), int(kp_y)), 5, color, -1)

                            # 添加关键点标签
                            kp_text = f"{self.keypoint_names[k]}" if k < len(self.keypoint_names) else f"{k}"
                            cv2.putText(result_img, kp_text, (int(kp_x + 5), int(kp_y + 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                            valid_kps.append((int(kp_x), int(kp_y)))
                            keypoints_count += 1
                        else:
                            valid_kps.append(None)

                    # 绘制骨架连接
                    for s, (start_idx, end_idx) in enumerate(self.skeleton):
                        if start_idx < len(valid_kps) and end_idx < len(valid_kps) and \
                                valid_kps[start_idx] is not None and valid_kps[end_idx] is not None:
                            # 绘制骨架线
                            color = self.skeleton_colors[s % len(self.skeleton_colors)]
                            cv2.line(result_img, valid_kps[start_idx], valid_kps[end_idx],
                                     color, 2)

        return result_img, keypoints_count

    def save_results_as_json(self, results, output_path="result.json"):
        """将推理结果保存为JSON格式"""
        import json

        # 提取结果
        results_dict = {
            "objects": [],
            "meta": {
                "model": "YOLO-pose",
                "resolution": f"{self.orig_width}x{self.orig_height}"
            }
        }

        # 处理每个结果
        for result in results:
            # 检查是否有检测结果
            if len(result.boxes) > 0:
                # 获取边界框坐标
                boxes = result.boxes.xyxy.cpu().numpy()
                # 获取置信度
                confs = result.boxes.conf.cpu().numpy()

                # 获取关键点信息
                if result.keypoints is not None:
                    keypoints_data = result.keypoints.data.cpu().numpy()

                    # 处理每个检测结果
                    for i, (box, conf, kpts) in enumerate(zip(boxes, confs, keypoints_data)):
                        x1, y1, x2, y2 = map(int, box)

                        # 构建关键点信息
                        keypoints = []
                        for k, (kp_x, kp_y, kp_conf) in enumerate(kpts):
                            keypoint_name = self.keypoint_names[k] if k < len(self.keypoint_names) else f"kpt{k}"
                            keypoints.append({
                                "x": int(kp_x),
                                "y": int(kp_y),
                                "confidence": float(kp_conf),
                                "name": keypoint_name
                            })

                        # 添加到结果对象
                        obj = {
                            "bbox": {
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                                "width": x2 - x1,
                                "height": y2 - y1,
                                "confidence": float(conf)
                            },
                            "keypoints": keypoints
                        }

                        results_dict["objects"].append(obj)

        # 保存为JSON文件
        try:
            with open(output_path, 'w') as f:
                json.dump(results_dict, f, indent=4)
            print(f"结果已保存到: {output_path}")
            return True
        except Exception as e:
            print(f"保存JSON文件失败: {e}")
            return False


def predict_single_image(image_path, model_path, device="cpu", conf_thres=0.4, save_output=True, output_path=None):
    """
    对单张图片进行姿态估计预测

    参数:
        image_path (str): 输入图片路径
        model_path (str): 模型文件路径
        device (str): 推理设备 ('cpu', 'gpu', 'mps')
        conf_thres (float): 置信度阈值
        save_output (bool): 是否保存结果图片
        output_path (str): 输出图片路径，如果为None则自动生成

    返回:
        tuple: (处理后的图像, 检测到的关键点数量, 推理时间(ms))
    """
    # 检查输入文件
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"输入图片未找到: {image_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    print(f"图片尺寸: {img.shape}")

    # 初始化检测器
    try:
        print(f"使用设备: {device}")
        detector = YOLOPoseInference(model_path, confidence_thres=conf_thres, device=device)
    except Exception as e:
        print(f"初始化检测器失败: {e}")
        raise

    # 执行推理
    start_time = time.time()
    processed_img, _, _, infer_time, _, keypoints_count = detector.infer(img)
    total_time = time.time() - start_time

    print(f"推理完成: 耗时 {total_time * 1000:.2f}ms, 检测到 {keypoints_count} 个关键点")

    # 保存输出图片
    if save_output:
        if output_path is None:
            # 自动生成输出路径
            base_name, ext = os.path.splitext(image_path)
            output_path = f"{base_name}_result{ext}"

        cv2.imwrite(output_path, processed_img)
        print(f"结果已保存到: {output_path}")

    return processed_img, keypoints_count, infer_time


def Process_Cam(m_cam2main_img: SharedMemoryManager.SharedMemory,
                m_cam2main_fps: SharedMemoryManager.SharedMemory,
                m_close: SharedMemoryManager.SharedMemory,
                resolution=(1280, 720)):
    """摄像头进程"""
    width, height = resolution

    # 尝试设置进程亲和性
    try:
        if platform.system() == 'Linux':
            set_process_affinity(os.getpid(), [0, 1, 2, 3])
    except:
        print("无法设置摄像头进程亲和性")

    # 初始化摄像头
    use_cv_cam = not HIKCAM_AVAILABLE

    if not use_cv_cam:
        try:
            cam = hik_cam.HikCam()
            cam.start_camera()
            cam.set_camera(15.0, 2000)
            print("已初始化海康摄像头")
        except Exception as e:
            print(f"海康摄像头初始化失败: {e}")
            use_cv_cam = True

    if use_cv_cam:
        try:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print("已初始化OpenCV摄像头")
        except Exception as e:
            print(f"OpenCV摄像头初始化失败: {e}")
            # 设置结束标志
            np.ndarray((1,), dtype=np.uint8, buffer=m_close.buf)[:] = 1
            return

    # 创建直接视图
    fps_view = np.ndarray((1,), dtype=np.int64, buffer=m_cam2main_fps.buf)
    img_view = np.ndarray((height, width, 3), dtype=np.uint8, buffer=m_cam2main_img.buf)
    close_view = np.ndarray((1,), dtype=np.uint8, buffer=m_close.buf)

    fps = 30
    frames = 0
    T1 = time.perf_counter()

    print("摄像头进程已启动")

    while True:
        # 获取图像
        if use_cv_cam:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # 调整帧大小以匹配共享内存
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
        else:
            try:
                frame = cam.get_image(False)

                # 调整帧大小以匹配共享内存
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
            except:
                time.sleep(0.01)
                continue

        # 直接复制到共享内存
        np.copyto(img_view, frame)
        fps_view[0] = np.int64(fps)

        # 更新FPS
        frames += 1
        if frames >= int(fps / 2) + 1:
            T2 = time.perf_counter()
            fps = round(frames / (T2 - T1), 2)
            T1 = T2
            frames = 0

        # 检查退出信号
        if close_view[0] == 1:
            break

    # 清理资源
    if use_cv_cam:
        cap.release()

    print("摄像头进程已结束")


def Process_Main(m_cam2main_img: SharedMemoryManager.SharedMemory,
                 m_cam2main_fps: SharedMemoryManager.SharedMemory,
                 m_close: SharedMemoryManager.SharedMemory,
                 resolution=(960, 540),
                 model_path="yolo11s-pose_trained_02.pt",
                 device="cpu",
                 conf_thres=0.4,
                 show_display=True):
    """主处理进程"""
    width, height = resolution

    # 尝试设置进程亲和性
    try:
        if platform.system() == 'Linux':
            set_process_affinity(os.getpid(), [4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    except:
        print("无法设置主进程亲和性")

    # 优化OpenCV
    optimize_opencv()

    # 初始化检测器
    try:
        print(f"使用设备: {device}")
        detector = YOLOPoseInference(model_path, confidence_thres=conf_thres, device=device)
    except Exception as e:
        print(f"初始化检测器失败: {e}")
        np.ndarray((1,), dtype=np.uint8, buffer=m_close.buf)[:] = 1
        return

    # 创建直接视图
    fps_view = np.ndarray((1,), dtype=np.int64, buffer=m_cam2main_fps.buf)
    img_view = np.ndarray((height, width, 3), dtype=np.uint8, buffer=m_cam2main_img.buf)
    close_view = np.ndarray((1,), dtype=np.uint8, buffer=m_close.buf)

    # 性能测量
    fps_main = 30
    frames = 0
    T1 = time.perf_counter()
    frame_render_time = 0

    # CPU使用率监控
    p = psutil.Process(os.getpid())

    print("主进程已启动")

    try:
        while True:
            frame_start = time.time()

            # 检查摄像头数据
            if fps_view[0] == -1:
                time.sleep(0.001)
                continue

            cam_fps = int(fps_view[0])
            fps_view[0] = -1

            # 获取当前帧
            current_frame = img_view.copy()

            # 使用推理
            processed_img, detector_fps, preprocess_time, infer_time, postprocess_time, keypoints_count = detector.infer(
                current_frame)

            # 测量CPU使用率
            cpu_usage = p.cpu_percent()

            # 显示性能信息
            frame_end = time.time()
            frame_render_time = frame_end - frame_start

            # 准备性能指标
            metrics = {
                'cam_fps': cam_fps,
                'infer_fps': fps_main,
                'preprocess_time': preprocess_time,
                'infer_time': infer_time,
                'postprocess_time': postprocess_time,
                'cpu_usage': cpu_usage,
                'keypoints_count': keypoints_count
            }

            # 添加性能叠加层和显示图像
            if show_display:
                # 避免使用Qt相关代码
                processed_img = add_performance_overlay(processed_img, metrics)

                # 纯OpenCV显示
                cv2.imshow('自定义YOLO-Pose 检测', processed_img)

                # 确保使用纯OpenCV的键盘处理
                key = cv2.waitKey(1)
                if key == 27:  # ESC键
                    close_view[0] = 1
                    break
            else:
                # 无显示模式下仍然计算FPS
                frames += 1
                if frames >= int(fps_main / 2) + 1:
                    T2 = time.perf_counter()
                    fps_main = round(1 / (T2 - T1) * frames, 2)
                    T1 = time.perf_counter()
                    frames = 0

                # 每30帧打印性能信息
                if frames % 30 == 0:
                    print(
                        f"CAM FPS: {cam_fps}, INFER FPS: {fps_main:.1f}, Time: {preprocess_time:.1f}ms {infer_time:.1f}ms {postprocess_time:.1f}ms, CPU: {cpu_usage:.1f}%, 关键点: {keypoints_count}")

            # 检查退出信号
            if close_view[0] == 1:
                break

    except Exception as e:
        print(f"主进程错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 设置关闭信号
        close_view[0] = 1

        # 关闭显示窗口
        if show_display:
            cv2.destroyAllWindows()

        print("主进程已结束")

# 修改主程序，添加命令行参数支持
if __name__ == '__main__':
    import argparse

    # 命令行参数解析
    parser = argparse.ArgumentParser(description='YOLO-Pose 四点检测')
    parser.add_argument('--model', type=str, default='test_models/yolo11n-pose_trained_01.pt',
                        help='模型路径 (默认: yolo11s-pose_trained_04.pt)')
    parser.add_argument('--device', type=str, default='gpu',
                        choices=['cpu', 'gpu', 'mps', 'auto'],
                        help='推理设备 (默认: cpu)')
    parser.add_argument('--width', type=int, default=960,
                        help='图像宽度 (默认: 960)')
    parser.add_argument('--height', type=int, default=540,
                        help='图像高度 (默认: 540)')
    parser.add_argument('--conf', type=float, default=0.1,
                        help='置信度阈值 (默认: 0.1)')
    parser.add_argument('--no-display', action='store_true',
                        help='不显示画面 (适用于无GUI环境)')
    parser.add_argument('--optimize', action='store_true',
                        help='应用系统优化 (仅限Linux系统和root权限)')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式，显示更多信息')

    # 新增参数 - 单张图片模式
    parser.add_argument('--image', type=str, default=None,
                        help='单张图片路径 (启用单张图片模式)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出图片路径 (仅用于单张图片模式)')
    parser.add_argument('--view', action='store_true',
                        help='显示处理结果 (仅用于单张图片模式)')

    args = parser.parse_args()

    # 检查是否为单张图片模式
    if args.image:
        print("=== 单张图片模式 ===")
        print(f"输入图片: {args.image}")
        print(f"模型: {args.model}")
        print(f"设备: {args.device}")
        print(f"置信度阈值: {args.conf}")

        # 设备自动检测
        if args.device == 'auto':
            devices = detect_accelerators()
            if 'GPU' in devices:
                args.device = 'gpu'
                print(f"自动选择设备: GPU ({devices['GPU']})")
            elif 'MPS' in devices:
                args.device = 'mps'
                print(f"自动选择设备: MPS (Apple Silicon)")
            else:
                args.device = 'cpu'
                print(f"自动选择设备: CPU ({devices.get('CPU', 'Unknown')})")

        try:
            # 处理单张图片
            result_img, keypoints_count, infer_time = predict_single_image(
                args.image,
                args.model,
                device=args.device,
                conf_thres=args.conf,
                save_output=True,
                output_path=args.output
            )

            # 显示结果
            if args.view:
                # 添加性能信息
                metrics = {
                    'cam_fps': 0,
                    'infer_fps': 0,
                    'infer_time': infer_time,
                    'keypoints_count': keypoints_count,
                    'cpu_usage': 0
                }
                result_img = add_performance_overlay(result_img, metrics)

                cv2.imshow('YOLO-Pose 检测结果', result_img)
                print("按任意键关闭窗口...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        except Exception as e:
            print(f"处理图片时出错: {e}")
            import traceback
            traceback.print_exc()

        exit(0)

    # 在Linux上优化系统设置
    if args.optimize and platform.system() == 'Linux':
        try:
            set_system_priorities()
        except Exception as e:
            print(f"系统优化失败: {e}")

    # 设置多处理方法
    if platform.system() == 'Windows':
        multiprocessing.set_start_method('spawn', force=True)

    # 设备自动检测
    if args.device == 'auto':
        devices = detect_accelerators()
        if 'GPU' in devices:
            args.device = 'gpu'
            print(f"自动选择设备: GPU ({devices['GPU']})")
        elif 'MPS' in devices:
            args.device = 'mps'
            print(f"自动选择设备: MPS (Apple Silicon)")
        else:
            args.device = 'cpu'
            print(f"自动选择设备: CPU ({devices.get('CPU', 'Unknown')})")

    try:
        # 共享内存管理
        smm = SharedMemoryManager()
        smm.start()

        # 创建共享内存
        data_img = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        data_fps = np.zeros((1,), dtype=np.int64)
        data_close = np.zeros((1,), dtype=np.uint8)

        m_cam2main_img = smm.SharedMemory(size=data_img.nbytes)
        m_cam2main_fps = smm.SharedMemory(size=data_fps.nbytes)
        m_close = smm.SharedMemory(size=data_close.nbytes)

        # 初始化共享内存
        np.ndarray((1,), dtype=np.int64, buffer=m_cam2main_fps.buf)[:] = -1

        # 创建进程
        resolution = (args.width, args.height)
        p_cam = Process(target=Process_Cam, args=(m_cam2main_img, m_cam2main_fps, m_close, resolution))
        p_main = Process(target=Process_Main,
                         args=(m_cam2main_img, m_cam2main_fps, m_close, resolution,
                               args.model, args.device, args.conf, not args.no_display))

        # 启动进程
        p_cam.start()
        p_main.start()

        # 等待主进程结束
        p_main.join()

        # 设置关闭标志
        np.ndarray((1,), dtype=np.uint8, buffer=m_close.buf)[:] = 1

        # 等待摄像头进程结束
        p_cam.join()

        # 关闭共享内存
        smm.shutdown()

        print("程序正常退出")

    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序执行错误: {e}")
        import traceback
        traceback.print_exc()
