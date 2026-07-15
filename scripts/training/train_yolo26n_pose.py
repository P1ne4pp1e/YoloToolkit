"""Train yolo26n-pose with the project's four-keypoint pose dataset."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from ultralytics import YOLO
from ultralytics.data import augment as ultralytics_augment


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class NoOpAlbumentations:
    """Disable Ultralytics' built-in low-probability Albumentations transforms."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, labels: dict) -> dict:
        return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def resolve_from_project(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_dataset(dataset_dir: Path) -> None:
    required_directories = (
        dataset_dir / "images" / "train",
        dataset_dir / "images" / "val",
        dataset_dir / "labels" / "train",
        dataset_dir / "labels" / "val",
    )
    missing = [str(path) for path in required_directories if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing dataset directories: " + ", ".join(missing))


def write_resolved_data_yaml(dataset_dir: Path, destination: Path) -> None:
    train_dir = dataset_dir / "images" / "train"
    val_dir = dataset_dir / "images" / "val"
    content = "\n".join(
        (
            f"train: {json.dumps(train_dir.as_posix())}",
            f"val: {json.dumps(val_dir.as_posix())}",
            "names:",
            "  0: target",
            "kpt_shape: [4, 3]",
            "",
        )
    )
    destination.write_text(content, encoding="utf-8")


def disable_augmentations() -> None:
    ultralytics_augment.Albumentations = NoOpAlbumentations


def main() -> None:
    args = parse_args()
    dataset_dir = resolve_from_project(args.dataset).resolve()
    model_path = resolve_from_project(args.model).resolve()
    project_dir = resolve_from_project(args.project).resolve()
    validate_dataset(dataset_dir)
    if not model_path.is_file():
        raise FileNotFoundError(f"Pretrained weights not found: {model_path}")

    disable_augmentations()
    with tempfile.TemporaryDirectory(prefix="yolo26n_pose_") as directory:
        data_yaml = Path(directory) / "data.yaml"
        write_resolved_data_yaml(dataset_dir, data_yaml)
        model = YOLO(str(model_path))
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            patience=args.patience,
            seed=args.seed,
            project=str(project_dir),
            name=args.name,
            pretrained=True,
            amp=True,
            cache="disk",
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.0,
            bgr=0.0,
            mosaic=0.0,
            mixup=0.0,
            cutmix=0.0,
            copy_paste=0.0,
            erasing=0.0,
            multi_scale=0.0,
        )


if __name__ == "__main__":
    main()
