"""Split a flat YOLO pose dataset into reproducible train, validation, and optional test sets."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = ("train", "val", "test")
KNOWN_PREFIXES = (
    "original_positive",
    "original_negative",
    "aug_black_negative",
    "aug_normal",
    "aug_relocate",
    "aug_cutout",
    "aug_composite",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/augmented-20260716-001"))
    parser.add_argument("--output", type=Path, default=Path("dataset/augmented-20260716-001-split"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    counts = split_dataset(args.input, args.output, ratios, args.seed)
    print(f"Dataset split completed: {args.output}")
    print(", ".join(f"{name}={counts[name]}" for name in SPLIT_NAMES if ratios[name] > 0))


def split_dataset(input_dir: Path, output_dir: Path, ratios: dict[str, float], seed: int) -> dict[str, int]:
    validate_inputs(input_dir, output_dir, ratios)
    samples = collect_samples(input_dir)
    if not samples:
        raise ValueError(f"No images found in: {input_dir / 'images'}")

    assignments = assign_splits(samples, ratios, seed)
    create_output_directories(output_dir, ratios)
    copy_assignments(assignments, output_dir)
    counts = {name: len(assignments[name]) for name in SPLIT_NAMES}
    write_data_yaml(output_dir, ratios)
    write_manifest(output_dir, input_dir, ratios, seed, counts)
    return counts


def validate_inputs(input_dir: Path, output_dir: Path, ratios: dict[str, float]) -> None:
    if not input_dir.joinpath("images").is_dir() or not input_dir.joinpath("labels").is_dir():
        raise ValueError("Input must contain images and labels directories")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    if any(value < 0 for value in ratios.values()) or not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Train, validation, and test ratios must be non-negative and sum to 1")
    if ratios["train"] == 0 or ratios["val"] == 0:
        raise ValueError("Train and validation ratios must both be greater than 0")


def collect_samples(input_dir: Path) -> list[tuple[Path, Path, str]]:
    samples: list[tuple[Path, Path, str]] = []
    for image_path in sorted(input_dir.joinpath("images").iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = input_dir / "labels" / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for image: {image_path.name}")
        label_kind = "positive" if label_path.read_text(encoding="utf-8").strip() else "negative"
        samples.append((image_path, label_path, f"{source_family(image_path.stem)}:{label_kind}"))
    return samples


def source_family(stem: str) -> str:
    for prefix in KNOWN_PREFIXES:
        if stem.startswith(f"{prefix}_"):
            return prefix
    return "other"


def assign_splits(
    samples: list[tuple[Path, Path, str]], ratios: dict[str, float], seed: int
) -> dict[str, list[tuple[Path, Path]]]:
    grouped: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for image_path, label_path, stratum in samples:
        grouped[stratum].append((image_path, label_path))

    rng = random.Random(seed)
    assignments: dict[str, list[tuple[Path, Path]]] = {name: [] for name in SPLIT_NAMES}
    for stratum in sorted(grouped):
        group = grouped[stratum]
        rng.shuffle(group)
        allocation = allocate_counts(len(group), ratios)
        start = 0
        for name in SPLIT_NAMES:
            end = start + allocation[name]
            assignments[name].extend(group[start:end])
            start = end
    return assignments


def allocate_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {name: total * ratios[name] for name in SPLIT_NAMES}
    counts = {name: math.floor(raw[name]) for name in SPLIT_NAMES}
    remaining = total - sum(counts.values())
    priority = {name: index for index, name in enumerate(SPLIT_NAMES)}
    for name in sorted(SPLIT_NAMES, key=lambda item: (-(raw[item] - counts[item]), priority[item]))[:remaining]:
        counts[name] += 1
    return counts


def create_output_directories(output_dir: Path, ratios: dict[str, float]) -> None:
    for name in SPLIT_NAMES:
        if ratios[name] > 0:
            output_dir.joinpath("images", name).mkdir(parents=True, exist_ok=True)
            output_dir.joinpath("labels", name).mkdir(parents=True, exist_ok=True)


def copy_assignments(assignments: dict[str, list[tuple[Path, Path]]], output_dir: Path) -> None:
    total = sum(len(items) for items in assignments.values())
    with tqdm(total=total, desc="Copying dataset", unit="image", dynamic_ncols=True) as progress:
        for name in SPLIT_NAMES:
            for image_path, label_path in assignments[name]:
                shutil.copy2(image_path, output_dir / "images" / name / image_path.name)
                shutil.copy2(label_path, output_dir / "labels" / name / label_path.name)
                progress.update()


def write_data_yaml(output_dir: Path, ratios: dict[str, float]) -> None:
    lines = ["path: .", "train: images/train", "val: images/val"]
    if ratios["test"] > 0:
        lines.append("test: images/test")
    lines.extend(("names:", "  0: target", "kpt_shape: [4, 3]", ""))
    output_dir.joinpath("data.yaml").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(
    output_dir: Path, input_dir: Path, ratios: dict[str, float], seed: int, counts: dict[str, int]
) -> None:
    manifest = {"input": str(input_dir), "seed": seed, "ratios": ratios, "counts": counts}
    output_dir.joinpath("split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
