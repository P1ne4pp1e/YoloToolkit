import cv2
import numpy as np
import onnxruntime as ort
import time
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import os


class YOLOPoseInference:
    """YOLO-Pose模型推理类"""

    def __init__(self, model_path, img_size=320, conf_thres=0.25, iou_thres=0.45):
        """
        初始化YOLO-Pose推理类

        Args:
            model_path: ONNX模型路径
            img_size: 输入图像大小
            conf_thres: 置信度阈值
            iou_thres: IOU阈值
        """
        self.model_path = model_path
        self.kpt_shape = [4, 3]  # 4个关键点，每个关键点有3个值(x,y,conf)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = (img_size, img_size)

        # 加载ONNX模型
        print(f"加载模型: {model_path}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            print(f"CUDA加速不可用，使用CPU: {e}")
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        # 获取模型信息
        self._get_model_info()

        # 设置关键点颜色 (按照原始PT代码设置)
        self.colors = {
            0: (255, 0, 0),  # 蓝色
            1: (0, 255, 0),  # 绿色
            2: (0, 0, 255),  # 红色
            3: (255, 255, 0)  # 青色
        }

    def _get_model_info(self):
        """获取模型的输入输出信息"""
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape

        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

        print(f"ONNX模型输入名称: {self.input_names}")
        print(f"ONNX模型输入形状: {self.input_shape}")
        print(f"ONNX模型输出名称: {self.output_names}")

    def _letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True,
                   scaleup=True, stride=32):
        """
        调整图像大小并添加填充

        Args:
            img: 输入图像
            new_shape: 目标尺寸
            color: 填充颜色
            auto: 是否自动裁剪填充
            scaleup: 是否允许放大
            stride: 步长

        Returns:
            调整后的图像、缩放比例和填充值
        """
        # 获取原始图像形状
        shape = img.shape[:2]  # 当前形状 [height, width]

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # 计算缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # 只缩小，不放大
            r = min(r, 1.0)

        # 计算填充
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # 最小矩形
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # 将填充分为两侧
        dh /= 2

        # 调整图像大小
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # 添加填充
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return img, r, (dw, dh)

    def preprocess(self, img):
        """
        预处理图像

        Args:
            img: 输入图像

        Returns:
            预处理后的图像数据
        """
        # 保存原始图像尺寸
        self.orig_shape = img.shape[:2]

        # BGR转RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 调整图像大小并添加填充
        img_resized, ratio, pad = self._letterbox(img_rgb, self.img_size, auto=False)

        # 保存比例和填充信息用于后处理
        self.ratio = ratio
        self.pad = pad

        # 打印调试信息
        print(f"原始图像形状: {img.shape}")
        print(f"调整大小后的图像形状: {img_resized.shape}")
        print(f"缩放比例: {ratio}, 填充: {pad}")

        # 转换为模型输入格式: [batch, channels, height, width]
        img_input = img_resized.transpose(2, 0, 1).astype(np.float32)  # HWC -> CHW
        img_input = np.expand_dims(img_input, axis=0)  # 添加批次维度
        img_input /= 255.0  # 归一化

        print(f"模型输入形状: {img_input.shape}")
        return img_input

    def inference(self, img_input):
        """
        执行推理

        Args:
            img_input: 预处理后的图像数据

        Returns:
            模型输出结果
        """
        start_time = time.time()
        inputs = {self.input_names[0]: img_input}
        outputs = self.session.run(self.output_names, inputs)

        np.save("onnx.npy", outputs[0])
        inference_time = time.time() - start_time

        print(f"推理时间: {inference_time * 1000:.2f}ms")
        print(f"输出形状: {[output.shape for output in outputs]}")

        return outputs

    def postprocess(self, output):
        """
        后处理模型输出

        Args:
            output: 模型输出结果

        Returns:
            处理后的目标框和关键点数据
        """
        # 获取输出张量
        tensor = output[0]
        if tensor.ndim == 3:
            tensor = tensor[0]  # 只处理第一个批次

        print(f"处理tensor形状: {tensor.shape}")

        # 找出最佳目标框（置信度最高的那个）
        conf = tensor[:, 4]
        best_idx = np.argmax(conf)

        # 检查置信度是否满足阈值
        if conf[best_idx] < self.conf_thres:
            print(f"未找到置信度大于{self.conf_thres}的检测")
            return []

        print(f"最佳检测索引: {best_idx}, 置信度: {conf[best_idx]:.4f}")

        # 提取目标框数据
        box = tensor[best_idx]

        # 解析边界框坐标 (x,y,w,h格式)
        x, y, w, h = box[0:4]

        # 转换为像素坐标
        box_scaled = np.zeros(4)
        box_scaled[0] = (x - w / 2) * self.img_size[0]  # x1 左上角x
        box_scaled[1] = (y - h / 2) * self.img_size[1]  # y1 左上角y
        box_scaled[2] = (x + w / 2) * self.img_size[0]  # x2 右下角x
        box_scaled[3] = (y + h / 2) * self.img_size[1]  # y2 右下角y

        # 调整为原始图像坐标
        box_orig = np.zeros(4)
        box_orig[0] = (box_scaled[0] - self.pad[0]) / self.ratio  # x1
        box_orig[1] = (box_scaled[1] - self.pad[1]) / self.ratio  # y1
        box_orig[2] = (box_scaled[2] - self.pad[0]) / self.ratio  # x2
        box_orig[3] = (box_scaled[3] - self.pad[1]) / self.ratio  # y2

        # 提取关键点数据
        keypoints = []
        start_idx = 5  # 关键点数据从索引5开始

        print(f"检测数据: {box[:15]}")  # 只打印前15个值以免输出过长

        # 处理4个关键点
        for i in range(self.kpt_shape[0]):
            idx = start_idx + i * 3  # 每个关键点有3个值(x,y,conf)

            if idx + 2 < len(box):
                kp_x, kp_y, kp_conf = box[idx:idx + 3]

                # YOLO-Pose模型输出的关键点坐标为百分比值(0-100)
                # 需要转换为归一化坐标(0-1)
                kp_x_norm = kp_x / 100.0
                kp_y_norm = kp_y / 100.0

                # 转换为像素坐标
                kp_x_img = kp_x_norm * self.img_size[0]
                kp_y_img = kp_y_norm * self.img_size[1]

                # 调整为原始图像坐标
                kp_x_orig = (kp_x_img - self.pad[0]) / self.ratio
                kp_y_orig = (kp_y_img - self.pad[1]) / self.ratio

                print(f"关键点 {i}: 原始值=({kp_x:.4f}, {kp_y:.4f}, {kp_conf:.4f}), "
                      f"归一化=({kp_x_norm:.4f}, {kp_y_norm:.4f}), "
                      f"调整后=({kp_x_orig:.1f}, {kp_y_orig:.1f})")

                keypoints.append([i, kp_x_orig, kp_y_orig, kp_conf])

        return [box_orig.tolist(), keypoints]

    def visualize(self, img, detection_result):
        """
        可视化检测结果

        Args:
            img: 原始图像
            detection_result: 检测结果

        Returns:
            绘制了检测框和关键点的图像
        """
        if not detection_result:
            print("没有有效的检测结果")
            return img.copy()

        # 获取检测框和关键点
        box, keypoints = detection_result
        result_img = img.copy()

        # 绘制边界框
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制关键点
        for kp in keypoints:
            if len(kp) >= 4:  # 确保有足够的数据
                idx, x, y, conf = kp

                if conf > self.conf_thres:
                    # 确保坐标在图像范围内
                    x = max(0, min(int(x), result_img.shape[1] - 1))
                    y = max(0, min(int(y), result_img.shape[0] - 1))

                    # 获取当前关键点的颜色
                    color = self.colors[int(idx)]

                    # 绘制关键点
                    cv2.circle(result_img, (x, y), 5, color, -1)

                    # 标注关键点索引
                    cv2.putText(result_img, f"{int(idx)}", (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return result_img

    def save_debug_info(self, outputs, detection_result):
        """
        保存调试信息

        Args:
            outputs: 模型原始输出
            detection_result: 处理后的检测结果
        """
        # 创建调试输出目录
        output_dir = "debug_output"
        os.makedirs(output_dir, exist_ok=True)

        # 保存原始输出数据
        np.save(os.path.join(output_dir, "raw_output.npy"), outputs[0])

        # 保存处理后的检测结果
        if detection_result:
            with open(os.path.join(output_dir, "processed_output.txt"), "w") as f:
                f.write(str(detection_result))

    def run(self, img_path, save_debug=True):
        """
        运行完整的推理流程

        Args:
            img_path: 图像路径或图像数据
            save_debug: 是否保存调试信息

        Returns:
            可视化后的结果图像
        """
        # 读取图像
        if isinstance(img_path, str):
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"无法读取图像: {img_path}")
        else:
            img = img_path

        # 图像预处理
        img_input = self.preprocess(img)

        # 执行推理
        outputs = self.inference(img_input)

        # 后处理
        detection_result = self.postprocess(outputs[0])

        # 保存调试信息
        if save_debug:
            self.save_debug_info(outputs, detection_result)

        # 可视化结果
        result_img = self.visualize(img, detection_result)

        return result_img


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="YOLO-Pose ONNX推理")
    parser.add_argument('--model', type=str, default='yolo11s-pose_trained_04.onnx', help='ONNX模型路径')
    parser.add_argument('--img', type=str, default='00000285_aug0.jpg', help='图像路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU阈值')
    parser.add_argument('--size', type=int, default=320, help='输入图像大小')
    parser.add_argument('--no-debug', action='store_true', help='不保存调试信息')
    args = parser.parse_args()

    try:
        # 初始化推理器
        yolo_pose = YOLOPoseInference(
            args.model,
            img_size=args.size,
            conf_thres=args.conf,
            iou_thres=args.iou
        )

        # 运行推理
        result_img = yolo_pose.run(args.img, save_debug=not args.no_debug)

        # 保存结果
        save_path = f"result_{Path(args.img).name}"
        cv2.imwrite(save_path, result_img)
        print(f"结果已保存至: {save_path}")

        # 显示结果
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    except Exception as e:
        import traceback
        print(f"运行时发生错误: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()