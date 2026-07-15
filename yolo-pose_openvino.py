# Import necessary libraries
import os
import time
import numpy as np
import cv2
import platform
import multiprocessing
from multiprocessing import Process, shared_memory
from multiprocessing.managers import SharedMemoryManager
import psutil
import torch
import numba
from numba import jit, prange

# Try to import HikVision camera module
try:
    import hikcam as hik_cam

    HIKCAM_AVAILABLE = True
except ImportError:
    HIKCAM_AVAILABLE = False
    print("hikcam module not available, will use OpenCV camera")

# Set environment variables
os.environ['OPENCV_VIDEOIO_PRIORITY_LIST'] = 'GSTREAMER,V4L2'
os.environ['QT_QPA_PLATFORM'] = 'xcb'


# Numba JIT compiled optimization functions
@jit(nopython=True, parallel=True, fastmath=True)
def resize_image_jit(img, width, height):
    """JIT compiled image resizing function"""
    resized = np.zeros((height, width, 3), dtype=np.uint8)
    h_ratio = img.shape[0] / height
    w_ratio = img.shape[1] / width

    for y in prange(height):
        for x in prange(width):
            src_y = min(int(y * h_ratio), img.shape[0] - 1)
            src_x = min(int(x * w_ratio), img.shape[1] - 1)
            resized[y, x] = img[src_y, src_x]

    return resized


@jit(nopython=True, fastmath=True)
def normalize_image_jit(img):
    """JIT compiled image normalization function"""
    normalized = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            normalized[y, x, 0] = img[y, x, 0] / 255.0
            normalized[y, x, 1] = img[y, x, 1] / 255.0
            normalized[y, x, 2] = img[y, x, 2] / 255.0
    return normalized


@jit(nopython=True, fastmath=True)
def calculate_fps_jit(prev_time, curr_time, prev_frames):
    """JIT compiled FPS calculation function"""
    fps = prev_frames / max(curr_time - prev_time, 1e-6)
    return fps


# Set process affinity
def set_process_affinity(pid, cpu_list):
    """Set process to specific CPU cores"""
    system = platform.system()

    try:
        if system == 'Linux':
            # Use taskset command
            cpu_str = ','.join(map(str, cpu_list))
            os.system(f"taskset -pc {cpu_str} {pid} > /dev/null 2>&1")
            print(f"Process {pid} bound to CPU cores {cpu_list}")
        elif system == 'Windows':
            import psutil
            p = psutil.Process(pid)
            p.cpu_affinity(cpu_list)
            print(f"Process {pid} bound to CPU cores {cpu_list}")
        else:
            print(f"Unsupported operating system: {system}")
    except Exception as e:
        print(f"Failed to set process affinity: {e}")


# Set system priorities and optimize
def set_system_priorities():
    """Optimize system settings, increase program priority"""
    import os, psutil

    try:
        # Get current process
        p = psutil.Process(os.getpid())

        # On Linux use nice value (-20 to 19, lower is higher priority)
        if platform.system() == 'Linux':
            p.nice(-10)
            print("Process priority set")
    except Exception as e:
        print(f"Unable to set process priority: {e}")

    # Optimization only for Linux
    if platform.system() == 'Linux':
        try:
            # Set CPU performance mode
            os.system(
                "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1")

            # Clear cache
            os.system("echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1")

            print("System optimizations applied")
        except:
            print("Unable to apply system optimizations, may need root privileges")


# Optimize OpenCV
def optimize_opencv():
    """Optimize OpenCV performance, avoid using Qt"""
    # Explicitly disable Qt backend
    os.environ['QT_QPA_PLATFORM'] = 'xcb'

    # Set OpenCV backend to GTK or other non-Qt backend
    try:
        cv2.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    except:
        pass

    # Enable OpenCL acceleration
    cv2.ocl.setUseOpenCL(True)

    # Optimize thread count based on CPU cores
    cpu_count = os.cpu_count()
    if cpu_count:
        cv2.setNumThreads(cpu_count)

    print(f"OpenCV optimized: OpenCL status={cv2.ocl.useOpenCL()}, threads={cv2.getNumThreads()}")


def add_performance_overlay(frame, metrics):
    """Add performance monitoring overlay (pure OpenCV implementation)"""
    h, w = frame.shape[:2]

    # Create semi-transparent background (pure OpenCV method)
    overlay_region = frame[10:200, 10:260]  # Expanded area to show more info
    cv2.rectangle(frame, (10, 10), (260, 200), (0, 0, 0), -1)

    # Blend - manual alpha blending
    alpha = 0.7
    cv2.addWeighted(
        frame[10:200, 10:260], alpha,
        overlay_region, 1 - alpha, 0,
        frame[10:200, 10:260]
    )

    # Draw metrics
    cv2.putText(frame, f"CAM FPS: {metrics['cam_fps']}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"INFER FPS: {metrics['infer_fps']:.1f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"INFER TIME: {metrics['infer_time']:.1f}ms", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"KEYPOINTS: {metrics.get('keypoints_count', 0)}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # CPU usage
    if 'cpu_usage' in metrics:
        color = (0, 255, 0) if metrics['cpu_usage'] < 80 else (0, 0, 255)
        cv2.putText(frame, f"CPU: {metrics['cpu_usage']:.1f}%", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Add corner confidence information (if provided)
    if 'corner_confidences' in metrics and metrics['corner_confidences']:
        cv2.putText(frame, f"Corner Confidences:", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame


# Detect hardware accelerators
def detect_accelerators():
    """Detect and configure available acceleration hardware"""
    devices = {}

    # Check if CUDA is available
    if torch.cuda.is_available():
        devices["GPU"] = torch.cuda.get_device_name(0)
        print(f"CUDA GPU acceleration available: {devices['GPU']}")

        # Get GPU details
        try:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB
            print(f"GPU memory: {gpu_mem:.2f} GB")
        except:
            pass
    else:
        print("CUDA GPU not available, will use CPU")
        devices["CPU"] = "PyTorch CPU"

        # Check CPU architecture and features
        try:
            import subprocess
            if platform.system() == 'Linux':
                cpu_info = subprocess.check_output("lscpu | grep -E 'Model name|MHz|CPU(s)'", shell=True)
                print(f"CPU info:\n{cpu_info.decode()}")
        except:
            pass

        # Check MPS (GPU acceleration on MacOS)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices["MPS"] = "Apple Silicon GPU"
            print("Apple Silicon GPU acceleration available")

    return devices


class YOLOPoseProcessor:
    """YOLO pose detection model JIT optimized processor class"""

    def __init__(self, model=None):
        """Initialize processor"""
        self.model = model

        # Set keypoint configuration - custom 4-point model
        self.keypoint_names = [
            "lu", "ld", "ru", "rd"
        ]

        # Set skeleton connections
        self.skeleton = [
            [0, 1], [0, 2], [1, 3], [2, 3]
        ]

        # Skeleton colors
        self.skeleton_colors = [
            (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0)
        ]

        # Keypoint colors
        self.keypoint_colors = [
            (0, 255, 255), (0, 255, 0), (0, 0, 255), (255, 0, 0)
        ]

    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def preprocess_image_jit(img, input_height, input_width):
        """JIT compiled image preprocessing function"""
        # Resize image
        resized = np.zeros((input_height, input_width, 3), dtype=np.uint8)
        h_ratio = img.shape[0] / input_height
        w_ratio = img.shape[1] / input_width

        for y in prange(input_height):
            for x in prange(input_width):
                src_y = min(int(y * h_ratio), img.shape[0] - 1)
                src_x = min(int(x * w_ratio), img.shape[1] - 1)
                resized[y, x] = img[src_y, src_x]

        # Normalize
        normalized = np.zeros((input_height, input_width, 3), dtype=np.float32)
        for y in prange(input_height):
            for x in prange(input_width):
                normalized[y, x, 0] = resized[y, x, 0] / 255.0
                normalized[y, x, 1] = resized[y, x, 1] / 255.0
                normalized[y, x, 2] = resized[y, x, 2] / 255.0

        return normalized

    def process_results(self, results, orig_width, orig_height, input_width=320, input_height=320):
        """Post-process YOLO results, but do not draw"""
        boxes = []
        keypoints_list = []
        keypoints_count = 0

        # Calculate scaling ratios
        scale_x = orig_width / input_width
        scale_y = orig_height / input_height

        # Process each result
        for result in results:
            # Extract bounding boxes
            if result.boxes is not None and len(result.boxes) > 0:
                for i in range(len(result.boxes)):
                    # Get bounding box coordinates and confidence
                    box = result.boxes.xyxy[i].cpu().numpy()
                    conf = result.boxes.conf[i].cpu().numpy()

                    # Adjust bounding box coordinates to original image size
                    x1 = box[0] * scale_x
                    y1 = box[1] * scale_y
                    x2 = box[2] * scale_x
                    y2 = box[3] * scale_y

                    # Add to bounding box list
                    boxes.append([x1, y1, x2, y2, conf])

            # Extract keypoints
            if result.keypoints is not None:
                keypoints = result.keypoints.data.cpu().numpy()  # [num_det, n_kpts, 3] - (x, y, confidence)

                for i in range(len(keypoints)):
                    kpts = []
                    for k in range(len(keypoints[i])):
                        x, y, conf = keypoints[i, k]

                        # Adjust keypoint coordinates to original image size
                        x = x * scale_x
                        y = y * scale_y

                        kpts.append((x, y, conf))
                        if conf > 0.5:  # Only count keypoints with high confidence
                            keypoints_count += 1

                    keypoints_list.append(kpts)

        return boxes, keypoints_list, keypoints_count

    def draw_results(self, frame, boxes, keypoints):
        """Draw detection results and keypoints on frame"""
        result_frame = frame.copy()

        # Keypoint names
        keypoint_names = ["Upper Left", "Lower Left", "Upper Right", "Lower Right"]

        # Keypoint colors
        colors = [
            (255, 0, 0),  # Blue
            (0, 255, 0),  # Green
            (0, 0, 255),  # Red
            (255, 255, 0)  # Cyan
        ]

        # Keypoint connections
        connections = [(0, 1), (1, 3), (2, 3), (0, 2)]

        # Add an output area to display confidences
        cv2.rectangle(result_frame, (10, frame.shape[0] - 150), (300, frame.shape[0] - 10), (0, 0, 0), -1)

        # Draw each detection result
        for i, (box, kpts) in enumerate(zip(boxes, keypoints)):
            x1, y1, x2, y2, conf = box

            # Draw bounding box
            cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Draw confidence
            cv2.putText(
                result_frame,
                "{:.2f}".format(conf),
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            # Output corner confidence information to console
            print("Detection #{} corner confidences:".format(i))

            # Display each corner confidence on frame
            for k in range(min(4, len(kpts))):
                # Get keypoint info
                if k < len(kpts):
                    kx, ky, kp_conf = kpts[k]

                    # Output to console
                    print("  {} corner: {:.3f}".format(keypoint_names[k], kp_conf))

                    # Display confidence on frame
                    cv2.putText(
                        result_frame,
                        "{}: {:.3f}".format(keypoint_names[k], kp_conf),
                        (20, frame.shape[0] - 130 + k * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        colors[k],
                        2
                    )

                    # If keypoint confidence is high enough
                    if kp_conf > 0.5:
                        # Draw keypoint
                        cv2.circle(result_frame, (int(kx), int(ky)), 5, colors[k], -1)

                        # Draw keypoint label
                        cv2.putText(
                            result_frame,
                            "{}".format(keypoint_names[k]),
                            (int(kx + 5), int(ky + 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            colors[k],
                            1
                        )

            # Draw connections between keypoints
            for conn in connections:
                if conn[0] < len(kpts) and conn[1] < len(kpts):
                    pt1 = kpts[conn[0]]
                    pt2 = kpts[conn[1]]

                    # If both keypoints have high enough confidence
                    if pt1[2] > 0.5 and pt2[2] > 0.5:
                        cv2.line(
                            result_frame,
                            (int(pt1[0]), int(pt1[1])),
                            (int(pt2[0]), int(pt2[1])),
                            (255, 255, 255),
                            2
                        )

        return result_frame


class YOLOPoseInference:
    """YOLO pose estimation inference class"""

    def __init__(self, model_path, confidence_thres=0.4, iou_thres=0.5, device="cpu"):
        """Initialize YOLO-Pose inference class"""
        from ultralytics import YOLO

        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.device = device

        # Check model file
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Initialize YOLO model
        print(f"Loading model: {model_path}")
        t_start = time.time()
        self.model = YOLO(model_path)
        print(f"Model loaded, time taken: {time.time() - t_start:.2f} seconds")

        # Set device
        if device.lower() == "gpu" and torch.cuda.is_available():
            self.device = "cuda"
        elif device.lower() == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"Using device: {self.device}")

        # Get input dimensions (try to get model's default input size)
        try:
            self.input_width = self.model.model.args['imgsz']
            self.input_height = self.model.model.args['imgsz']
            if isinstance(self.input_width, list):
                self.input_width = self.input_width[0]
                self.input_height = self.input_height[0]
        except:
            # Use default size
            self.input_height, self.input_width = 320, 320

        print(f"Model input size: {self.input_width}x{self.input_height}")

        # Create processor
        self.processor = YOLOPoseProcessor(self.model)

        # FPS calculation
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()

    def infer(self, img):
        """Perform inference and draw pose estimation results"""
        # Record original image dimensions
        original_img = img.copy()
        orig_height, orig_width = img.shape[:2]

        # Preprocess - resize image to match model input requirements
        t_preprocess_start = time.time()
        resized_img = cv2.resize(img, (self.input_width, self.input_height))
        t_preprocess = time.time() - t_preprocess_start

        # Inference
        t_inference_start = time.time()
        results = self.model(resized_img, imgsz=(self.input_width, self.input_height),
                             conf=self.confidence_thres, iou=self.iou_thres, device=self.device, verbose=False)
        t_inference = time.time() - t_inference_start

        # Post-process - extract bounding boxes and keypoints from YOLO results
        t_postprocess_start = time.time()

        # Use separated post-processing function
        boxes, keypoints, keypoints_count = self.processor.process_results(
            results, orig_width, orig_height, self.input_width, self.input_height
        )

        # Store keypoints list for later use
        self.processor.keypoints_list = keypoints

        # Get corner confidences (for debug)
        corner_confidences = []
        if keypoints and len(keypoints) > 0:
            for i in range(min(4, len(keypoints[0]))):
                if i < len(keypoints[0]):
                    # Get first detection result's corner confidences
                    corner_confidences.append(keypoints[0][i][2])

            # Print corner confidence information (console debug output)
            names = ["Upper Left", "Lower Left", "Upper Right", "Lower Right"]
            print("Corner confidences:")
            for i, conf in enumerate(corner_confidences):
                if i < len(names):
                    print(f"  {names[i]}: {conf:.3f}")

        # Use separated drawing function
        processed_img = self.processor.draw_results(original_img, boxes, keypoints)

        t_postprocess = time.time() - t_postprocess_start

        # Update FPS
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.last_time

        if elapsed_time >= 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.last_time = current_time

        # Return processed image and time metrics
        return processed_img, self.fps, t_preprocess * 1000, t_inference * 1000, t_postprocess * 1000, keypoints_count, corner_confidences


def Process_Cam(m_cam2main_img: SharedMemoryManager.SharedMemory,
                m_cam2main_fps: SharedMemoryManager.SharedMemory,
                m_close: SharedMemoryManager.SharedMemory,
                resolution=(1280, 720)):
    """Camera process"""
    width, height = resolution

    # Try to set process affinity
    try:
        if platform.system() == 'Linux':
            set_process_affinity(os.getpid(), [0, 1])
    except:
        print("Unable to set camera process affinity")

    # Initialize camera
    use_cv_cam = not HIKCAM_AVAILABLE

    if not use_cv_cam:
        try:
            cam = hik_cam.HikCam()
            cam.start_camera()
            cam.set_camera(15.0, 4000)
            print("HikVision camera initialized")
        except Exception as e:
            print(f"HikVision camera initialization failed: {e}")
            use_cv_cam = True

    if use_cv_cam:
        try:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print("OpenCV camera initialized")
        except Exception as e:
            print(f"OpenCV camera initialization failed: {e}")
            # Set end flag
            np.ndarray((1,), dtype=np.uint8, buffer=m_close.buf)[:] = 1
            return

    # Create direct views
    fps_view = np.ndarray((1,), dtype=np.int64, buffer=m_cam2main_fps.buf)
    img_view = np.ndarray((height, width, 3), dtype=np.uint8, buffer=m_cam2main_img.buf)
    close_view = np.ndarray((1,), dtype=np.uint8, buffer=m_close.buf)

    fps = 30
    frames = 0
    T1 = time.perf_counter()

    print("Camera process started")

    while True:
        # Get image
        if use_cv_cam:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Adjust frame size to match shared memory
            if frame.shape[:2] != (height, width):
                # Use JIT compiled resize function
                try:
                    frame = resize_image_jit(frame, width, height)
                except:
                    # If JIT fails, fall back to OpenCV resize
                    frame = cv2.resize(frame, (width, height))
        else:
            try:
                frame = cam.get_image(False)

                # Adjust frame size to match shared memory
                if frame.shape[:2] != (height, width):
                    # Use JIT compiled resize function
                    try:
                        frame = resize_image_jit(frame, width, height)
                    except:
                        # If JIT fails, fall back to OpenCV resize
                        frame = cv2.resize(frame, (width, height))
            except:
                time.sleep(0.01)
                continue

        # Copy directly to shared memory
        np.copyto(img_view, frame)
        fps_view[0] = np.int64(fps)

        # Update FPS
        frames += 1
        if frames >= (fps/2):  # Update FPS every 10 frames
            T2 = time.perf_counter()
            fps = round(frames / (T2 - T1), 2)
            T1 = T2
            frames = 0

        # Check exit signal
        if close_view[0] == 1:
            break

    # Clean up resources
    if use_cv_cam:
        cap.release()

    print("Camera process ended")


def Process_Main(m_cam2main_img: SharedMemoryManager.SharedMemory,
                 m_cam2main_fps: SharedMemoryManager.SharedMemory,
                 m_close: SharedMemoryManager.SharedMemory,
                 resolution=(960, 540),
                 model_path="yolo11s-pose_trained_05_openvino_model/",
                 device="cpu",
                 conf_thres=0.4,
                 show_display=True):
    """Main processing process"""
    width, height = resolution

    # Try to set process affinity
    try:
        if platform.system() == 'Linux':
            set_process_affinity(os.getpid(), [2, 3, 4, 5, 6, 7])
    except:
        print("Unable to set main process affinity")

    # Optimize OpenCV
    optimize_opencv()

    # Initialize detector
    try:
        print(f"Using device: {device}")
        detector = YOLOPoseInference(model_path, confidence_thres=conf_thres, device=device)
    except Exception as e:
        print(f"Detector initialization failed: {e}")
        np.ndarray((1,), dtype=np.uint8, buffer=m_close.buf)[:] = 1
        return

    # Create direct views
    fps_view = np.ndarray((1,), dtype=np.int64, buffer=m_cam2main_fps.buf)
    img_view = np.ndarray((height, width, 3), dtype=np.uint8, buffer=m_cam2main_img.buf)
    close_view = np.ndarray((1,), dtype=np.uint8, buffer=m_close.buf)

    # Performance measurement
    fps_main = 30
    frames = 0
    T1 = time.perf_counter()

    # CPU usage monitoring
    p = psutil.Process(os.getpid())

    print("Main process started")

    try:
        while True:
            frame_start = time.time()

            # Check camera data
            if fps_view[0] == -1:
                time.sleep(0.001)
                continue

            cam_fps = int(fps_view[0])
            fps_view[0] = -1

            # Get current frame
            current_frame = img_view

            # Use inference
            # Modified
            processed_img, detector_fps, preprocess_time, infer_time, postprocess_time, keypoints_count, corner_confidences = detector.infer(
                current_frame)

            # Measure CPU usage
            cpu_usage = p.cpu_percent()

            # Prepare performance metrics
            metrics = {
                'cam_fps': cam_fps,
                'infer_fps': fps_main,
                'preprocess_time': preprocess_time,
                'infer_time': infer_time,
                'postprocess_time': postprocess_time,
                'cpu_usage': cpu_usage,
                'keypoints_count': keypoints_count,
                'corner_confidences': corner_confidences
            }

            # Update FPS
            frames += 1
            if frames >= 10:  # Update FPS every 10 frames
                T2 = time.perf_counter()
                fps_main = round(frames / (T2 - T1), 2)
                T1 = T2
                frames = 0

            # Add performance overlay and display image
            if show_display:
                # Avoid using Qt related code
                processed_img = add_performance_overlay(processed_img, metrics)

                # Pure OpenCV display
                cv2.imshow('YOLO-Pose JIT Optimized Detection', processed_img)

                # Ensure using pure OpenCV keyboard handling
                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    close_view[0] = 1
                    break
            else:
                # No display mode still calculates FPS

                # Print performance info every 30 frames
                if frames % 30 == 0:
                    print(
                        f"CAM FPS: {cam_fps}, INFER FPS: {fps_main:.1f}, Time: {preprocess_time:.1f}ms {infer_time:.1f}ms {postprocess_time:.1f}ms, CPU: {cpu_usage:.1f}%, Keypoints: {keypoints_count}")

            # Check exit signal
            if close_view[0] == 1:
                break

    except Exception as e:
        print(f"Main process error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Set close signal
        close_view[0] = 1

        # Close display windows
        if show_display:
            cv2.destroyAllWindows()

        print("Main process ended")


# Main function
def main():
    import argparse

    # Command line argument parsing
    parser = argparse.ArgumentParser(description='YOLO-Pose JIT Optimized Detection')
    parser.add_argument('--model', type=str, default='test_models/yolo11s-pose_trained_06_openvino_model/',
                        help='Model path (default: yolo11s-pose_trained_05_openvino_model/)')
    parser.add_argument('--device', type=str, default='gpu',
                        choices=['cpu', 'gpu', 'mps', 'auto'],
                        help='Inference device (default: cpu)')
    parser.add_argument('--width', type=int, default=960,
                        help='Image width (default: 960)')
    parser.add_argument('--height', type=int, default=540,
                        help='Image height (default: 540)')
    parser.add_argument('--conf', type=float, default=0.6,
                        help='Confidence threshold (default: 0.6)')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not show display (for environments without GUI)')
    parser.add_argument('--optimize', action='store_true',
                        help='Apply system optimizations (Linux only with root privileges)')

    args = parser.parse_args()

    # Optimize system settings on Linux
    if args.optimize and platform.system() == 'Linux':
        try:
            set_system_priorities()
        except Exception as e:
            print(f"System optimization failed: {e}")

    # Set multiprocessing method
    if platform.system() == 'Windows':
        multiprocessing.set_start_method('spawn', force=True)

    # Auto-detect device
    if args.device == 'auto':
        devices = detect_accelerators()
        if 'GPU' in devices:
            args.device = 'gpu'
            print(f"Automatically selected device: GPU ({devices['GPU']})")
        elif 'MPS' in devices:
            args.device = 'mps'
            print(f"Automatically selected device: MPS (Apple Silicon)")
        else:
            args.device = 'cpu'
            print(f"Automatically selected device: CPU ({devices.get('CPU', 'Unknown')})")

    try:
        # Shared memory management
        smm = SharedMemoryManager()
        smm.start()

        # Create shared memory
        data_img = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        data_fps = np.zeros((1,), dtype=np.int64)
        data_close = np.zeros((1,), dtype=np.uint8)

        m_cam2main_img = smm.SharedMemory(size=data_img.nbytes)
        m_cam2main_fps = smm.SharedMemory(size=data_fps.nbytes)
        m_close = smm.SharedMemory(size=data_close.nbytes)

        # Initialize shared memory
        np.ndarray((1,), dtype=np.int64, buffer=m_cam2main_fps.buf)[:] = -1

        # Create processes
        resolution = (args.width, args.height)
        p_cam = Process(target=Process_Cam, args=(m_cam2main_img, m_cam2main_fps, m_close, resolution))
        p_main = Process(target=Process_Main,
                         args=(m_cam2main_img, m_cam2main_fps, m_close, resolution,
                               args.model, args.device, args.conf, not args.no_display))

        # Start processes
        p_cam.start()
        p_main.start()

        # Wait for main process to end
        p_main.join()

        # Set close flag
        np.ndarray((1,), dtype=np.uint8, buffer=m_close.buf)[:] = 1

        # Wait for camera process to end
        p_cam.join()

        # Close shared memory
        smm.shutdown()

        print("Program exited normally")

    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Program execution error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()