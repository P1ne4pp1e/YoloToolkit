"""Filter a LabelMe dataset by target-ROI motion blur."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yolo_toolkit.dataset.motion_blur_filter import annotated_image_scores, filter_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按目标矩形 ROI 的运动模糊程度筛选 LabelMe 数据集")
    parser.add_argument("input_dir", type=Path, help="原始数据集目录")
    parser.add_argument("output_dir", type=Path, help="筛选结果输出目录")
    parser.add_argument("--threshold", type=float, help="过滤模糊分数大于此值的有标注图片")
    parser.add_argument("--plot", type=Path, help="分布图保存路径，默认保存至输出目录")
    parser.add_argument("--no-show", action="store_true", help="不弹出 matplotlib 图窗")
    return parser


def print_threshold_table(scores: list[float]) -> None:
    print("\n阈值影响（过滤条件：模糊分数 > 阈值）")
    print(f"{'阈值':>14} {'过滤占比':>12} {'过滤张数':>12} {'保留张数':>12}")
    for percentile in (0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100):
        threshold = float(np.percentile(scores, percentile))
        removed = sum(score > threshold for score in scores)
        print(f"{threshold:14.8f} {removed / len(scores):11.2%} {removed:12d} {len(scores) - removed:12d}")


def save_distribution_plot(scores: list[float], output_path: Path, show: bool) -> None:
    values = np.asarray(scores, dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    bins = min(50, max(10, int(np.sqrt(len(values)))))
    axes[0].hist(values, bins=bins, density=True, color="#2b6cb0", edgecolor="white", alpha=0.9)
    axes[0].set_title("目标 ROI 模糊分数分布")
    axes[0].set_xlabel("模糊分数（越大越模糊）")
    axes[0].set_ylabel("概率密度")
    sorted_values = np.sort(values)
    axes[1].plot(sorted_values, np.arange(1, len(values) + 1) / len(values), color="#c05621", linewidth=2)
    axes[1].set_title("目标 ROI 模糊分数累计分布")
    axes[1].set_xlabel("模糊分数（越大越模糊）")
    axes[1].set_ylabel("累计占比")
    axes[1].set_ylim(0, 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    print(f"分布图已保存：{output_path}")
    if show:
        plt.show()
    plt.close(figure)


def main() -> int:
    args = build_parser().parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f"输入目录不存在：{input_dir}")
    records = annotated_image_scores(input_dir)
    if not records:
        raise ValueError("未找到包含有效矩形标注框的图片，无法计算模糊分布")
    scores = [record.score for record in records]
    print(f"已统计 {len(scores)} 张有有效矩形标注框的图片。")
    print_threshold_table(scores)
    plot_path = args.plot or output_dir / "motion_blur_distribution.png"
    save_distribution_plot(scores, plot_path, show=not args.no_show)
    threshold = args.threshold
    if threshold is None:
        threshold = float(input("请输入过滤阈值（模糊分数大于该值的有标注图片将被移除）：").strip())
    removed = sum(score > threshold for score in scores)
    result_scores = filter_dataset(input_dir, output_dir, threshold, records)
    print(f"阈值 {threshold:.8f}：过滤 {removed}/{len(scores)} 张有标注图片，已输出至：{output_dir}")
    print(f"已记录 {len(result_scores)} 张有有效 ROI 的模糊分数。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
