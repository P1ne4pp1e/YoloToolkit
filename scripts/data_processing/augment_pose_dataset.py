"""Create a streaming, LabelMe-compatible 20k A4-target pose dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from yolo_toolkit.dataset.pose_augmentation import (
    composite_target,
    cutout_negative,
    fill_black,
    letterbox_to_square,
    load_pose_annotation,
    normal_augmentation,
    relocate_target,
    yolo_pose_line,
)


QUOTAS = {"normal": 7800, "relocate": 2200, "black_negative": 800, "cutout": 2300, "composite": 6090}
JPEG_QUALITY = 95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/annotated-20260715-001"))
    parser.add_argument("--output", type=Path, default=Path("dataset/augmented-20260716-001"))
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--preview-count", type=int, default=5)
    parser.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--preview-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        existing_entries = [entry for entry in args.output.iterdir() if entry.name != "previews"]
        if existing_entries:
            raise FileExistsError(f"Output already contains generated data: {args.output}")
    positives = sorted(args.input.joinpath("with_target").glob("*.json"))
    backgrounds = sorted(args.input.joinpath("without_target").glob("*.png"))
    if not positives or not backgrounds:
        raise ValueError("Input must contain LabelMe files in with_target and PNG files in without_target")
    rng = random.Random(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    write_previews(args.output / "previews", positives, backgrounds, rng, args.preview_count)
    if args.preview_only:
        return
    write_originals(positives, backgrounds, args.output, args.workers, args.batch_size)
    generated = Counter()
    tasks = build_augmentation_jobs(positives, backgrounds, args.output, rng)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for strategy in tqdm(
            executor.map(generate_augmentation_job, tasks, chunksize=args.batch_size),
            total=len(tasks),
            desc="多进程生成增强样本",
            unit="张",
            dynamic_ncols=True,
        ):
            generated[strategy] += 1
    write_data_yaml(args.output)
    manifest = {"image_size": 640, "original_positive": len(positives), "original_negative": len(backgrounds), "generated": dict(generated), "total": len(positives) + len(backgrounds) + sum(generated.values())}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_previews(directory: Path, positives: list[Path], backgrounds: list[Path], rng: random.Random, count: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for strategy in QUOTAS:
        for index in range(count):
            annotation_path = rng.choice(positives)
            annotation = load_pose_annotation(annotation_path)
            image = read_image(annotation_path.with_suffix(".png"))
            points = annotation.points
            labelled = strategy not in {"black_negative", "cutout"}
            if strategy == "normal": output, points = normal_augmentation(image, points, rng)
            elif strategy == "relocate": output, points = relocate_target(image, points, rng)
            elif strategy == "black_negative": output = fill_black(image, points)
            elif strategy == "cutout": output = cutout_negative(image, points, rng)
            else: output, points = composite_target(read_image(rng.choice(backgrounds)), image, points, rng)
            output, points = letterbox_to_square(output, points if labelled else None)
            if labelled:
                draw_points(output, points)
            cv2.imwrite(str(directory / f"{strategy}_{index:02d}.jpg"), output, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def write_originals(positives: list[Path], backgrounds: list[Path], output_dir: Path, workers: int, batch_size: int) -> None:
    jobs = [("positive", path, output_dir) for path in positives]
    jobs.extend(("negative", path, output_dir) for path in backgrounds)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for _ in tqdm(
            executor.map(convert_original_job, jobs, chunksize=batch_size),
            total=len(jobs),
            desc="多进程转换原始样本",
            unit="张",
            dynamic_ncols=True,
        ):
            pass


def build_augmentation_jobs(positives: list[Path], backgrounds: list[Path], output_dir: Path, rng: random.Random) -> list[tuple]:
    jobs: list[tuple] = []
    for strategy, quota in QUOTAS.items():
        for index in range(quota):
            annotation_path = rng.choice(positives)
            background_path = rng.choice(backgrounds) if strategy == "composite" else None
            jobs.append((strategy, index, annotation_path, background_path, output_dir, rng.randrange(2**63)))
    return jobs


def convert_original_job(job: tuple[str, Path, Path]) -> None:
    kind, path, output_dir = job
    if kind == "positive":
        annotation = load_pose_annotation(path)
        write_yolo_sample(output_dir, f"original_positive_{path.stem}", read_image(path.with_suffix(".png")), annotation.points)
    else:
        write_yolo_sample(output_dir, f"original_negative_{path.stem}", read_image(path), None)


def generate_augmentation_job(job: tuple) -> str:
    strategy, index, annotation_path, background_path, output_dir, seed = job
    rng = random.Random(seed)
    annotation = load_pose_annotation(annotation_path)
    image = read_image(annotation_path.with_suffix(".png"))
    if strategy == "normal":
        output_image, points = normal_augmentation(image, annotation.points, rng)
        write_positive(output_dir, strategy, index, output_image, points)
    elif strategy == "relocate":
        output_image, points = relocate_target(image, annotation.points, rng)
        write_positive(output_dir, strategy, index, output_image, points)
    elif strategy == "black_negative":
        write_negative(output_dir, strategy, index, fill_black(image, annotation.points))
    elif strategy == "cutout":
        write_negative(output_dir, strategy, index, cutout_negative(image, annotation.points, rng))
    else:
        background = read_image(background_path)
        output_image, points = composite_target(background, image, annotation.points, rng)
        write_positive(output_dir, strategy, index, output_image, points)
    return strategy


def write_positive(output_dir: Path, strategy: str, index: int, image, points) -> None:
    write_yolo_sample(output_dir, f"aug_{strategy}_{index:05d}", image, points)


def write_negative(output_dir: Path, strategy: str, index: int, image) -> None:
    write_yolo_sample(output_dir, f"aug_{strategy}_{index:05d}", image, None)


def write_yolo_sample(output_dir: Path, stem: str, image, points) -> None:
    images_dir, labels_dir = output_dir / "images", output_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    resized, transformed = letterbox_to_square(image, points, size=640)
    if not cv2.imwrite(str(images_dir / f"{stem}.jpg"), resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
        raise ValueError(f"Unable to write image: {stem}")
    label = "" if transformed is None else yolo_pose_line(transformed, size=640)
    (labels_dir / f"{stem}.txt").write_text(label + ("\n" if label else ""), encoding="ascii")


def write_data_yaml(output_dir: Path) -> None:
    (output_dir / "data.yaml").write_text(
        "path: .\ntrain: images\nval: images\nnames:\n  0: target\nkpt_shape: [4, 3]\n",
        encoding="utf-8",
    )


def read_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def draw_points(image, points) -> None:
    for label, point in zip(("LU", "RU", "RL", "LL"), points, strict=True):
        x, y = map(int, np.rint(point))
        cv2.circle(image, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(image, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


if __name__ == "__main__":
    main()
