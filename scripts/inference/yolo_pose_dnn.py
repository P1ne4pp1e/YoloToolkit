import os.path
import time
import numpy as np
import cv2
import platform
import multiprocessing
from multiprocessing import Process, shared_memory
from multiprocessing.managers import SharedMemoryManager
import argparse
import psutil

# 设置环境变量
os.environ['OPENCV_VIDEOIO_PRIORITY_LIST'] = 'GSTREAMER,V4L2'
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# 尝试导入海康摄像头模块
try:
    from yolo_toolkit.camera import hikvision as hik_cam

    HIKCAM_AVAILABLE = True
except ImportError:
    HIKCAM_AVAILABLE = False
    print("hikcam模块不可用，将使用OpenCV摄像头")


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
    cv2.putText(frame, f"OBJECTS: {metrics.get('object_count', 0)}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # CPU使用率
    if 'cpu_usage' in metrics:
        color = (0, 255, 0) if metrics['cpu_usage'] < 80 else (0, 0, 255)
        cv2.putText(frame, f"CPU: {metrics['cpu_usage']:.1f}%", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


class YOLODNNDetector:
    def __init__(self, config_path, weights_path, class_names_path, confidence_thres=0.4, nms_thres=0.5,
                 backend=cv2.dnn.DNN_BACKEND_OPENCV, target=cv2.dnn.DNN_TARGET_CPU):
        """初始化YOLO DNN检测器"""
        self.confidence_thres = confidence_thres
        self.nms_thres = nms_thres

        # 检查文件存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"权重文件未找到: {weights_path}")
        if not os.path.exists(class_names_path):
            raise FileNotFoundError(f"类别名称文件未找到: {class_names_path}")

        # 加载类别名称
        with open(class_names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        print(f"加载了 {len(self.classes)} 个类别")

        # 加载神经网络
        print(f"正在加载模型: {weights_path}")
        t_start = time.time()

        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.net.setPreferableBackend(backend)
        self.net.setPreferableTarget(target)

        # 获取网络层名称
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        print(f"模型加载完成，耗时: {time.time() - t_start:.2f}秒")

        # 设置随机颜色
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype=np.uint8)

        # FPS计算
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()

        # 获取模型输入尺寸 (尝试从配置中获取)
        try:
            # 这里尝试获取一个空的blob，然后检查它的尺寸
            self.input_width = 416
            self.input_height = 416
            blob = cv2.dnn.blobFromImage(
                np.zeros((416, 416, 3), dtype=np.uint8),
                1 / 255.0, (self.input_width, self.input_height),
                swapRB=True, crop=False
            )
            self.net.setInput(blob)
            test_out = self.net.forward(self.output_layers)
            print(f"成功测试网络前向传播")
        except Exception as e:
            print(f"网络测试失败，设置默认尺寸: {e}")
            self.input_width = 416
            self.input_height = 416

        print(f"模型输入尺寸: {self.input_width}x{self.input_height}")

    def infer(self, img):
        """执行推理并绘制目标检测结果"""
        # 记录原始图像和尺寸
        original_img = img.copy()
        height, width = img.shape[:2]

        # 推理准备
        start_time = time.time()

        # 预处理 - 转换为blob
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (self.input_width, self.input_height),
                                     swapRB=True, crop=False)

        # 设置输入并执行前向传播
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # 计算推理时间
        infer_time = (time.time() - start_time) * 1000  # 转换为毫秒

        # 处理结果
        class_ids = []
        confidences = []
        boxes = []

        # 解析输出
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence_thres:
                    # YOLO输出是相对于网络输入尺寸的中心坐标和宽高
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # 左上角坐标
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # 应用非极大值抑制
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_thres, self.nms_thres)

        # 绘制检测结果
        result_img = original_img.copy()
        object_count = 0

        if len(indices) > 0:
            for i in indices.flatten():
                object_count += 1
                x, y, w, h = boxes[i]

                # 修正超出图像边界的框
                x = max(0, x)
                y = max(0, y)

                # 获取类别和颜色
                class_id = class_ids[i]
                color = self.colors[class_id].tolist()

                # 绘制边界框
                cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)

                # 添加标签
                label = f"{self.classes[class_id]}: {confidences[i]:.2f}"
                cv2.rectangle(result_img, (x, y - 30), (x + len(label) * 10, y), color, -1)
                cv2.putText(result_img, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 更新FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time

        return result_img, self.fps, infer_time, object_count


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
            cam.set_camera(15.0, 4000)
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
                 config_path="yolov3.cfg",
                 weights_path="yolov3.weights",
                 class_names_path="coco.names",
                 conf_thres=0.4,
                 nms_thres=0.5,
                 backend=cv2.dnn.DNN_BACKEND_OPENCV,
                 target=cv2.dnn.DNN_TARGET_CPU,
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
        print(f"使用后端: {backend}, 目标: {target}")
        detector = YOLODNNDetector(
            config_path,
            weights_path,
            class_names_path,
            confidence_thres=conf_thres,
            nms_thres=nms_thres,
            backend=backend,
            target=target
        )
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
            processed_img, detector_fps, infer_time, object_count = detector.infer(current_frame)

            # 测量CPU使用率
            cpu_usage = p.cpu_percent()

            # 显示性能信息
            frame_end = time.time()
            frame_render_time = frame_end - frame_start

            # 准备性能指标
            metrics = {
                'cam_fps': cam_fps,
                'infer_fps': fps_main,
                'infer_time': infer_time,
                'cpu_usage': cpu_usage,
                'object_count': object_count
            }

            # 添加性能叠加层和显示图像
            if show_display:
                # 避免使用Qt相关代码
                processed_img = add_performance_overlay(processed_img, metrics)

                # 纯OpenCV显示
                cv2.imshow('YOLO DNN 目标检测', processed_img)

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
                        f"CAM FPS: {cam_fps}, INFER FPS: {fps_main:.1f}, Time: {infer_time:.1f}ms, CPU: {cpu_usage:.1f}%, 检测到: {object_count}个对象")

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


def predict_single_image(image_path, config_path, weights_path, class_names_path, conf_thres=0.4, nms_thres=0.5,
                         backend=cv2.dnn.DNN_BACKEND_OPENCV, target=cv2.dnn.DNN_TARGET_CPU,
                         save_output=True, output_path=None):
    """
    对单张图片进行目标检测

    参数:
        image_path (str): 输入图片路径
        config_path (str): 配置文件路径
        weights_path (str): 权重文件路径
        class_names_path (str): 类别名称文件路径
        conf_thres (float): 置信度阈值
        nms_thres (float): 非极大值抑制阈值
        backend: DNN后端
        target: DNN目标设备
        save_output (bool): 是否保存结果图片
        output_path (str): 输出图片路径，如果为None则自动生成

    返回:
        tuple: (处理后的图像, 检测到的对象数量, 推理时间(ms))
    """
    # 检查输入文件
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"输入图片未找到: {image_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"权重文件未找到: {weights_path}")

    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"类别名称文件未找到: {class_names_path}")

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    print(f"图片尺寸: {img.shape}")

    # 初始化检测器
    try:
        detector = YOLODNNDetector(
            config_path,
            weights_path,
            class_names_path,
            confidence_thres=conf_thres,
            nms_thres=nms_thres,
            backend=backend,
            target=target
        )
    except Exception as e:
        print(f"初始化检测器失败: {e}")
        raise

    # 执行推理
    start_time = time.time()
    processed_img, _, infer_time, object_count = detector.infer(img)
    total_time = time.time() - start_time

    print(f"推理完成: 耗时 {total_time * 1000:.2f}ms, 检测到 {object_count} 个对象")

    # 保存输出图片
    if save_output:
        if output_path is None:
            # 自动生成输出路径
            base_name, ext = os.path.splitext(image_path)
            output_path = f"{base_name}_result{ext}"

        cv2.imwrite(output_path, processed_img)
        print(f"结果已保存到: {output_path}")

    return processed_img, object_count, infer_time


if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='YOLO DNN 目标检测系统')

    # 模型相关参数
    parser.add_argument('--config', type=str, default='yolov3.cfg',
                        help='YOLO配置文件路径 (默认: yolov3.cfg)')
    parser.add_argument('--weights', type=str, default='yolov3.weights',
                        help='YOLO权重文件路径 (默认: yolov3.weights)')
    parser.add_argument('--classes', type=str, default='coco.names',
                        help='类别名称文件路径 (默认: coco.names)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='置信度阈值 (默认: 0.5)')
    parser.add_argument('--nms', type=float, default=0.4,
                        help='非极大值抑制阈值 (默认: 0.4)')

    # 设备相关参数
    parser.add_argument('--backend', type=str, default='opencv',
                        choices=['opencv', 'cuda', 'halide', 'inference_engine', 'cpu_fx'],
                        help='DNN后端 (默认: opencv)')
    parser.add_argument('--target', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'opencl', 'vulkan', 'fpga'],
                        help='计算目标 (默认: cpu)')

    # 图像相关参数
    parser.add_argument('--width', type=int, default=960,
                        help='图像宽度 (默认: 960)')
    parser.add_argument('--height', type=int, default=540,
                        help='图像高度 (默认: 540)')

    # 其他参数
    parser.add_argument('--no-display', action='store_true',
                        help='不显示画面 (适用于无GUI环境)')
    parser.add_argument('--optimize', action='store_true',
                        help='应用系统优化 (仅限Linux系统和root权限)')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式，显示更多信息')

    # 单张图片模式参数
    parser.add_argument('--image', type=str, default=None,
                        help='单张图片路径 (启用单张图片模式)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出图片路径 (仅用于单张图片模式)')
    parser.add_argument('--view', action='store_true',
                        help='显示处理结果 (仅用于单张图片模式)')

    args = parser.parse_args()

    # 映射backend字符串到OpenCV常量
    backend_map = {
        'opencv': cv2.dnn.DNN_BACKEND_OPENCV,
        'cuda': cv2.dnn.DNN_BACKEND_CUDA,
        'halide': cv2.dnn.DNN_BACKEND_HALIDE,
        'inference_engine': cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE,
        'cpu_fx': cv2.dnn.DNN_BACKEND_TIMVX
    }

    # 映射target字符串到OpenCV常量
    target_map = {
        'cpu': cv2.dnn.DNN_TARGET_CPU,
        'cuda': cv2.dnn.DNN_TARGET_CUDA,
        'opencl': cv2.dnn.DNN_TARGET_OPENCL,
        'vulkan': cv2.dnn.DNN_TARGET_VULKAN,
        'fpga': cv2.dnn.DNN_TARGET_FPGA
    }

    backend = backend_map.get(args.backend, cv2.dnn.DNN_BACKEND_OPENCV)
    target = target_map.get(args.target, cv2.dnn.DNN_TARGET_CPU)

    # 检查是否为单张图片模式
    if args.image:
        print("=== 单张图片模式 ===")
        print(f"输入图片: {args.image}")
        print(f"配置文件: {args.config}")
        print(f"权重文件: {args.weights}")
        print(f"类别文件: {args.classes}")
        print(f"后端: {args.backend}, 目标: {args.target}")
        print(f"置信度阈值: {args.conf}, NMS阈值: {args.nms}")

        try:
            # 处理单张图片
            result_img, object_count, infer_time = predict_single_image(
                args.image,
                args.config,
                args.weights,
                args.classes,
                conf_thres=args.conf,
                nms_thres=args.nms,
                backend=backend,
                target=target,
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
                    'object_count': object_count,
                    'cpu_usage': 0
                }
                result_img = add_performance_overlay(result_img, metrics)

                cv2.imshow('YOLO DNN 目标检测结果', result_img)
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

        # 检查和打印配置
        print(f"=== YOLO DNN 目标检测系统 ===")
        print(f"配置文件: {args.config}")
        print(f"权重文件: {args.weights}")
        print(f"类别文件: {args.classes}")
        print(f"分辨率: {args.width}x{args.height}")
        print(f"置信度阈值: {args.conf}, NMS阈值: {args.nms}")
        print(f"使用后端: {args.backend}, 目标: {args.target}")
        print(f"显示界面: {not args.no_display}")

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
                                   args.config, args.weights, args.classes, args.conf, args.nms,
                                   backend, target, not args.no_display))

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
