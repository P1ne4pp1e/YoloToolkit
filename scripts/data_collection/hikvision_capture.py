"""Collect an image-only dataset from a Hikvision industrial camera."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import time

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a raw image dataset with the first Hikvision camera."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/raw"),
        help="Output directory, default: dataset/raw",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=15.0,
        help="Camera gain, default: 15.0",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=800.0,
        help="Exposure time in the unit expected by MVS, usually microseconds",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Automatic capture interval in seconds; 0 disables automatic capture",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Stop after this many images; 0 means no limit",
    )
    parser.add_argument(
        "--format",
        choices=("png", "jpg"),
        default="png",
        help="Image format, default: png",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Preview the camera without saving images",
    )
    return parser.parse_args()


def next_index(output_dir: Path, extension: str) -> int:
    pattern = re.compile(rf"^frame_(\d+)\.{re.escape(extension)}$")
    indices = []
    for path in output_dir.glob(f"frame_*.{extension}"):
        match = pattern.match(path.name)
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=0) + 1


def save_frame(frame_bgr, output_dir: Path, index: int, extension: str) -> Path:
    path = output_dir / f"frame_{index:06d}.{extension}"
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if extension == "jpg" else []
    if not cv2.imwrite(str(path), frame_bgr, params):
        raise OSError(f"Failed to write image: {path}")
    return path


def main() -> int:
    args = parse_args()
    if args.interval < 0:
        raise ValueError("--interval must be greater than or equal to 0")
    if args.max_images < 0:
        raise ValueError("--max-images must be greater than or equal to 0")

    args.output.mkdir(parents=True, exist_ok=True)
    image_index = next_index(args.output, args.format)
    saved_count = 0
    next_capture_at = time.monotonic()
    from yolo_toolkit.camera import HikCam

    camera = HikCam()
    camera_started = False

    try:
        camera.start_camera()
        camera_started = True
        camera.set_camera(args.gain, args.exposure)
        print(f"Saving images to: {args.output.resolve()}")
        if args.preview_only:
            print("Preview-only mode. Press 'q' to quit.")
        else:
            print("Press 's' to save, 'q' to quit.")

        while True:
            frame_rgb = camera.get_image(False)
            if frame_rgb is None:
                continue

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("Hikvision Capture", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            now = time.monotonic()
            should_capture = not args.preview_only and (key == ord("s") or (
                args.interval > 0 and now >= next_capture_at
            ))

            if should_capture:
                path = save_frame(frame_bgr, args.output, image_index, args.format)
                print(f"Saved [{saved_count + 1}]: {path.name}")
                image_index += 1
                saved_count += 1
                next_capture_at = now + args.interval
                if args.max_images and saved_count >= args.max_images:
                    break

            if key == ord("q"):
                break
    finally:
        if camera_started:
            camera.close_device()
        cv2.destroyAllWindows()

    print(f"Capture finished. Saved {saved_count} image(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
