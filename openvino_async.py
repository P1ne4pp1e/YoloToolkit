import os.path
import time
import numpy as np
import cv2
import platform
import multiprocessing
from multiprocessing import Process, shared_memory
from multiprocessing.managers import SharedMemoryManager
from openvino import Core
import threading
import argparse
import psutil

import os
os.environ['OPENCV_VIDEOIO_PRIORITY_LIST'] = 'GSTREAMER,V4L2'
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# 尝试导入海康摄像头模块
try:
    import hikcam as hik_cam
    HIKCAM_AVAILABLE = True
except ImportError:
    HIKCAM_AVAILABLE = False
    print("hikcam模块不可用，将使用OpenCV摄像头")

# 硬件加速探测
def detect_accelerators():
    """检测并配置可用加速硬件"""
    devices = {}
    
    try:
        # 初始化OpenVINO
        core = Core()
        available = core.available_devices
        
        # 获取设备信息
        for device in available:
            try:
                full_name = core.get_property(device, "FULL_DEVICE_NAME")
                devices[device] = full_name
            except:
                devices[device] = "Unknown"
                
        # 检查特定NUC加速硬件
        if "GPU" in devices:
            print(f"Intel GPU加速可用: {devices['GPU']}")
            # 检查GPU频率
            try:
                import subprocess
                if platform.system() == 'Linux':
                    gpu_freq = subprocess.check_output("cat /sys/class/drm/card0/gt_cur_freq_mhz", shell=True)
                    print(f"GPU频率: {gpu_freq.decode().strip()} MHz")
            except:
                pass
                
        if "MYRIAD" in devices:
            print(f"Neural Compute Stick 2可用: {devices['MYRIAD']}")
            
        if "HDDL" in devices:
            print(f"HDDL加速可用: {devices['HDDL']}")
            
        if len(devices) == 1 and "CPU" in devices:
            # 检查CPU架构和特性
            try:
                import subprocess
                if platform.system() == 'Linux':
                    cpu_info = subprocess.check_output("lscpu | grep -E 'Model name|MHz|CPU(s)'", shell=True)
                    print(f"CPU信息:\n{cpu_info.decode()}")
            except:
                pass
                
            # 检查AVX-512支持
            try:
                if platform.system() == 'Linux':
                    with open("/proc/cpuinfo", "r") as f:
                        if "avx512" in f.read():
                            print("CPU支持AVX-512指令集，可获得最佳性能")
            except:
                pass
                
        return devices
    except Exception as e:
        print(f"检测加速硬件失败: {e}")
        return {"CPU": "Fallback"}

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
            os.system("echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1")
            
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
    overlay_region = frame[10:130, 10:260].copy()
    cv2.rectangle(frame, (10, 10), (260, 130), (0, 0, 0), -1)

    # 混合 - 手动实现alpha混合
    alpha = 0.7
    cv2.addWeighted(
        frame[10:130, 10:260], alpha,
        overlay_region, 1 - alpha, 0,
        frame[10:130, 10:260]
    )

    # 绘制指标
    cv2.putText(frame, f"CAM FPS: {metrics['cam_fps']}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"INFER FPS: {metrics['infer_fps']:.1f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"INFER TIME: {metrics['infer_time']:.1f}ms", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # CPU使用率
    if 'cpu_usage' in metrics:
        cv2.putText(frame, f"CPU: {metrics['cpu_usage']:.1f}%", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if metrics['cpu_usage'] < 80 else (0, 0, 255), 2)

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

# 模型缓存管理
class ModelCache:
    """模型缓存管理器"""
    def __init__(self, cache_dir="/tmp/model_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def get_cached_model(self, model_path, device):
        """获取缓存的模型或重新编译"""
        model_name = os.path.basename(model_path)
        cache_path = os.path.join(self.cache_dir, f"{model_name}_{device}.xml")
        
        if os.path.exists(cache_path):
            print(f"使用缓存模型: {cache_path}")
            return cache_path
            
        print(f"模型未缓存，需要编译...")
        return self._compile_and_cache(model_path, device)
        
    def _compile_and_cache(self, model_path, device):
        """编译并缓存模型"""
        try:
            from openvino import serialize
            
            # 读取模型
            core = Core()
            model = core.read_model(model_path)
            
            # 优化配置
            config = {"CACHE_DIR": self.cache_dir}
            
            # 编译模型
            compiled_model = core.compile_model(model, device, config)
            
            # 缓存路径
            model_name = os.path.basename(model_path)
            cache_path = os.path.join(self.cache_dir, f"{model_name}_{device}.xml")
            
            # 序列化模型
            serialize(model, cache_path)
            print(f"模型已缓存到: {cache_path}")
            
            return cache_path
        except Exception as e:
            print(f"缓存模型失败: {e}")
            return model_path

class YOLOv11Inference:
    def __init__(self, model_path, confidence_thres=0.4, iou_thres=0.5, device="CPU"):
        """初始化YOLOv11推理类"""
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.device = device
        
        # 检查模型文件
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
        
        # 尝试使用缓存模型
        try:
            cache = ModelCache()
            model_path = cache.get_cached_model(model_path, device)
        except Exception as e:
            print(f"模型缓存失败，使用原始模型: {e}")
        
        # 初始化OpenVINO
        self.core = Core()
        
        # 配置优化参数 - 特别适合NUC
        if device == "CPU":
            config = {
                "PERFORMANCE_HINT": "LATENCY",  # 对于实时应用
                "NUM_STREAMS": "2",             # 适合四核处理器
                "INFERENCE_PRECISION_HINT": "f16",  # 使用FP16精度
                "ENABLE_CPU_PINNING": "YES"   # 启用CPU固定
            }
        else:  # GPU
            config = {
                "PERFORMANCE_HINT": "LATENCY",
                "CACHE_DIR": "/tmp/openvino_cache",
                "INFERENCE_PRECISION_HINT": "f16"
            }
        
        print(f"正在加载模型: {model_path}")
        self.model = self.core.read_model(model_path)
        
        print(f"编译模型到 {device}...")
        t_start = time.time()
        self.compiled_model = self.core.compile_model(
            model=self.model,
            device_name=self.device,
            config=config
        )
        print(f"模型编译完成，耗时: {time.time() - t_start:.2f}秒")
        
        # 获取输入输出信息
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
        # 设置输入尺寸
        self.input_shape = self.input_layer.shape
        if len(self.input_shape) == 4:
            _, _, self.input_height, self.input_width = self.input_shape
        else:
            # 较小的尺寸以获得更好的性能
            self.input_height, self.input_width = 416, 416
            
        print(f"模型输入尺寸: {self.input_width}x{self.input_height}")
        
        # 创建推理请求
        self.infer_request = self.compiled_model.create_infer_request()
        
        # 生成COCO类别
        self.classes = self._generate_classes()
        
        # 生成颜色调色板
        np.random.seed(42)
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        # FPS计算
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        
    def _generate_classes(self):
        """生成COCO数据集类别"""
        return {
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
        
    def preprocess(self, img):
        """图像预处理"""
        # 调整图像大小
        resized_img = cv2.resize(img, (self.input_width, self.input_height))
        
        # 转换颜色空间
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        
        # 归一化并转换为NCHW格式
        processed = rgb_img.astype(np.float32) / 255.0
        processed = processed.transpose(2, 0, 1)  # HWC到CHW
        processed = np.expand_dims(processed, 0)  # 添加批次维度
        
        return processed
    
    def preprocess_batch(self, img_list, batch_size=2):
        """批量图像预处理"""
        batch_input = np.zeros((batch_size, 3, self.input_height, self.input_width), dtype=np.float32)
        
        for i, img in enumerate(img_list[:batch_size]):
            # 调整图像大小
            resized_img = cv2.resize(img, (self.input_width, self.input_height))
            
            # 转换颜色空间
            rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            
            # 归一化并转换格式
            processed = rgb_img.astype(np.float32) / 255.0
            processed = processed.transpose(2, 0, 1)  # HWC到CHW
            
            batch_input[i] = processed
        
        return batch_input
        
    def postprocess(self, image, outputs):
        """后处理并绘制检测框"""
        # 获取原始图像尺寸
        img_height, img_width = image.shape[:2]
        
        # 转换输出
        outputs = outputs.squeeze().T  # 形状变为 [8400, 85] 或类似的形状
        
        # 分离类别分数
        boxes = outputs[:, :4]  # 前4个值为中心点坐标和宽高
        scores = outputs[:, 4:]  # 剩余的为类别分数
        
        # 获取最高置信度和对应的类别
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # 应用置信度阈值
        mask = confidences > self.confidence_thres
        
        # 提取满足阈值的候选框
        filtered_boxes = boxes[mask]
        filtered_class_ids = class_ids[mask]
        filtered_confidences = confidences[mask]
        
        # 如果没有检测到任何物体，返回原始图像
        if len(filtered_boxes) == 0:
            return image
            
        # 计算缩放因子
        x_factor = img_width / self.input_width
        y_factor = img_height / self.input_height
        
        # 将中心点坐标、宽度和高度转换为左上右下坐标
        boxes_xyxy = []
        original_boxes = []
        for i, (cx, cy, w, h) in enumerate(filtered_boxes):
            left = int((cx - w / 2) * x_factor)
            top = int((cy - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            
            # 存储原始框的格式
            original_boxes.append([left, top, width, height])
            
            # 转换为xyxy格式
            boxes_xyxy.append([left, top, left + width, top + height])
            
        # 应用非最大抑制
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy,
            filtered_confidences.tolist(),
            self.confidence_thres,
            self.iou_thres
        )
        
        # 绘制检测结果
        result_image = image.copy()
        
        # 处理不同版本的OpenCV返回的索引格式
        if len(indices) > 0:
            if isinstance(indices[0], list) or isinstance(indices[0], np.ndarray):
                indices = [i[0] for i in indices]
                
        for i in indices:
            # 获取检测框信息
            left, top, width, height = original_boxes[i]
            class_id = int(filtered_class_ids[i])
            confidence = filtered_confidences[i]
            
            # 获取类别颜色
            color = tuple(map(int, self.color_palette[class_id % len(self.color_palette)]))
            
            # 绘制检测框
            cv2.rectangle(result_image, (left, top), (left + width, top + height), color, 2)
            
            # 绘制标签
            label = f"{self.classes[class_id]}: {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(result_image, (left, top - text_height - 5), (left + text_width, top), color, -1)
            cv2.putText(result_image, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        return result_image
        
    def infer(self, img):
        """执行推理并绘制检测框"""
        # 记录原始图像尺寸
        original_img = img.copy()
        
        # 预处理
        start_time = time.time()
        preprocessed = self.preprocess(img)
        preprocess_time = time.time() - start_time

        # 推理
        start_time = time.time()
        self.infer_request.infer({self.input_layer.any_name: preprocessed})
        infer_time = time.time() - start_time
        
        # 获取结果
        output = self.infer_request.get_output_tensor(0).data
        
        # 后处理和绘制
        start_time = time.time()
        processed_img = self.postprocess(original_img, output)
        processed_time = time.time() - start_time

        # 更新FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
            
        return processed_img, self.fps, preprocess_time * 1000, infer_time * 1000, processed_time * 1000  # 返回毫秒
    
    def infer_async(self, img, infer_request=None):
        """异步执行推理"""
        # 如果没有提供推理请求，使用默认的
        if infer_request is None:
            infer_request = self.infer_request
            
        # 预处理
        preprocessed = self.preprocess(img)
        
        # 开始异步推理
        infer_request.start_async({self.input_layer.any_name: preprocessed})
        
        return infer_request


def Process_Cam(m_cam2main_img: SharedMemoryManager.SharedMemory,
                m_cam2main_fps: SharedMemoryManager.SharedMemory,
                m_close: SharedMemoryManager.SharedMemory,
                resolution=(1440, 1080)):
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
            cam.set_camera(15.0, 3000)
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
                model_path="yolo11n.onnx",
                device="CPU",
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
        detector = YOLOv11Inference(model_path, confidence_thres=conf_thres, device=device)
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
    
    # 简化：使用同步推理替代异步
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
            
            # 使用同步推理
            processed_img, detector_fps, preprocess_time, infer_time, postprocess_time = detector.infer(current_frame)
            
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
                'cpu_usage': cpu_usage
            }
            
            # 添加性能叠加层
            # 添加性能叠加层和显示图像
            if show_display:
                # 避免使用Qt相关代码
                processed_img = add_performance_overlay(processed_img, metrics)

                # try:
                # print("showing...")
                # print(processed_img.shape)
                # 纯OpenCV显示
                cv2.imshow('YOLOv11 Detection', processed_img)
                # print("end showing")

                # 确保使用纯OpenCV的键盘处理
                key = cv2.waitKey(1)
                # print(key)
                if key == 27:  # ESC键
                    close_view[0] = 1
                    break
                # except Exception as e:
                #     print(f"显示错误: {e}")
                #     # 如果显示失败，切换到无显示模式
                #     show_display = False
                #     print("显示模式出错，切换到无显示模式")
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
                    print(f"CAM FPS: {cam_fps}, INFER FPS: {fps_main:.1f}, Time: {preprocess_time:.1f}ms {infer_time:.1f}ms {postprocess_time:.1f}ms, CPU: {cpu_usage:.1f}%")
            
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

if __name__ == '__main__':
    # 命令行参数解析
    # 然后在主程序中使用

    # os.environ['QT_QPA_PLATFORM'] = ''
    # os.environ['OPENCV_VIDEOIO_PRIORITY_LIST'] = 'GSTREAMER,V4L2'  # 优先使用非Qt后端

    parser = argparse.ArgumentParser(description='YOLOv11目标检测')
    parser.add_argument('--model', type=str, default='yolo11s-pose_trained_04.onnx',
                        help='模型路径 (默认: yolo11s_trained_02.onnx)')
    parser.add_argument('--device', type=str, default='CPU',
                        choices=['CPU', 'GPU', 'MYRIAD', 'AUTO'],
                        help='推理设备 (默认: CPU)')
    parser.add_argument('--width', type=int, default=960,
                        help='图像宽度 (默认: 960)')
    parser.add_argument('--height', type=int, default=540,
                        help='图像高度 (默认: 540)')
    parser.add_argument('--conf', type=float, default=0.8,
                        help='置信度阈值 (默认: 0.4)')
    parser.add_argument('--no-display', action='store_true',
                        help='不显示画面 (适用于无GUI环境)')
    parser.add_argument('--optimize', action='store_true',
                        help='应用系统优化 (仅限Linux系统和root权限)')
    args = parser.parse_args()

    # 在 __main__ 部分添加

    if not args.no_display:
        if not check_display_available():
            print("警告: 无法使用显示功能，切换到无显示模式")
            args.no_display = True

    # 显示系统信息
    print(f"=== YOLOv11 目标检测 ===")
    print(f"系统: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"模型: {args.model}")
    print(f"设备: {args.device}")
    print(f"分辨率: {args.width}x{args.height}")
    print(f"置信度阈值: {args.conf}")
    print(f"显示模式: {'关闭' if args.no_display else '开启'}")
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"错误: 模型文件未找到 {args.model}")
        exit(1)
    
    # 设备自动检测
    if args.device == 'AUTO':
        devices = detect_accelerators()
        if 'GPU' in devices:
            args.device = 'GPU'
            print(f"自动选择设备: GPU ({devices['GPU']})")
        elif 'MYRIAD' in devices:
            args.device = 'MYRIAD'
            print(f"自动选择设备: MYRIAD ({devices['MYRIAD']})")
        else:
            args.device = 'CPU'
            print(f"自动选择设备: CPU ({devices.get('CPU', 'Unknown')})")
    
    # 在Linux上优化系统设置
    if args.optimize and platform.system() == 'Linux':
        try:
            set_system_priorities()
        except Exception as e:
            print(f"系统优化失败: {e}")
    
    # 设置多处理方法
    if platform.system() == 'Windows':
        multiprocessing.set_start_method('spawn', force=True)
    
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
