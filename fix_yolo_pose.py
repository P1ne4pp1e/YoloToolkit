from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch


def predict_with_four_keypoints_and_bbox(model_path, image_path, save_path=None, conf_thres=0.5):
    """
    使用YOLOv8进行姿态预测，显示四个关键点和检测框，不进行连接

    参数:
        model_path (str): 模型的路径，可以是预训练模型或自定义模型
        image_path (str): 需要预测的图像路径，可以是本地路径或URL
        save_path (str, optional): 保存结果图像的路径，默认为None
        conf_thres (float, optional): 置信度阈值，默认0.5

    返回:
        PIL.Image: 可视化结果的图像
    """
    # 加载模型
    try:
        model = YOLO(model_path)
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        return None

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 进行预测
    try:
        results = model(image_path, conf=conf_thres, device=device)
        print(f"预测完成，检测到 {len(results[0].boxes)} 个目标")
    except Exception as e:
        print(f"预测失败: {str(e)}")
        return None

    # np.save("result.npy", results[0].cpu().numpy())
    # 获取原始图像
    result = results[0]
    orig_img = result.orig_img.copy()

    # 检查是否有检测结果
    if len(result.boxes) > 0:
        # 获取边界框坐标
        boxes = result.boxes.xyxy.cpu().numpy()
        # 获取置信度
        confs = result.boxes.conf.cpu().numpy()

        # 绘制每个检测框
        for i, (box, conf) in enumerate(zip(boxes, confs)):
            # 提取边界框坐标
            x1, y1, x2, y2 = map(int, box)

            # 绘制矩形边界框
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 添加置信度标签
            label = f"{conf:.2f}"
            cv2.putText(orig_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 获取关键点信息
    if result.keypoints is not None:
        keypoints = result.keypoints.data  # [num_det, n_kpts, 3] - (x, y, confidence)
        print(f"keypoints.shape: {keypoints.shape}")
        print(f"检测到 {keypoints.shape[0]} 个对象的关键点")

        # 定义四个关键点的索引
        keypoint_indices = [0, 1, 2, 3]  # 这里需要根据您的实际模型调整

        # 定义四个不同的颜色
        colors = [(255, 0, 0),  # 蓝色
                  (0, 255, 0),  # 绿色
                  (0, 0, 255),  # 红色
                  (255, 255, 0)]  # 青色

        # 为每个检测结果绘制四个关键点
        for kpts in keypoints:
            # 只处理指定的四个关键点
            for k, idx in enumerate(keypoint_indices):
                if idx < len(kpts):  # 确保索引不越界
                    x, y, conf = kpts[idx]
                    if conf > 0:  # 只绘制有效的关键点
                        color = colors[k % len(colors)]
                        cv2.circle(orig_img, (int(x), int(y)), 5, color, -1)
                        # 添加关键点标签
                        cv2.putText(orig_img, f"{idx}", (int(x + 5), int(y + 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        print("未检测到关键点")

    # 转换回PIL图像以便显示
    result_img = Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))

    # 保存结果图像
    if save_path:
        result_img.save(save_path)
        print(f"结果已保存至: {save_path}")

    return result_img

# 使用示例
if __name__ == "__main__":
    # 可以使用预训练模型或自定义模型
    model_path = "test_models/yolo11s-pose_trained_04.pt"  # 您的四关键点自定义模型

    # 图像路径，可以是本地路径或URL
    image_path = "images/00000048.png"

    # 保存路径
    save_path = "images/result_four_keypoints.jpg"

    # 预测并显示结果
    result_img = predict_with_four_keypoints_and_bbox(model_path, image_path, save_path)

    if result_img:
        plt.figure(figsize=(12, 8))
        plt.imshow(result_img)
        plt.axis('off')
        plt.title('YOLOv8 四关键点预测结果')
        plt.show()

        # 打印关键点信息
        print("\n关键点详细信息:")
        model = YOLO(model_path)
        results = model(image_path)
        for i, result in enumerate(results):
            if result.keypoints is not None:
                print(f"目标 #{i + 1}:")
                # 只输出四个关键点的信息
                kpts = result.keypoints.data[0]  # 假设只有一个检测结果
                keypoint_indices = [0, 1, 2, 3]  # 四个关键点的索引

                for idx in keypoint_indices:
                    if idx < len(kpts):
                        x, y, conf = kpts[idx]
                        print(f"  关键点 {idx}: x={x:.2f}, y={y:.2f}, conf={conf:.2f}")