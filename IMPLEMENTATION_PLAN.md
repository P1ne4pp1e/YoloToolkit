## Stage 1: Dataset Contract
**Goal**: Read LabelMe target rectangles and named corner keypoints safely.
**Success Criteria**: The four semantic keypoints are validated and ordered consistently.
**Tests**: Parse a minimal LabelMe annotation and reject incomplete targets.
**Status**: Complete

## Stage 2: Augmentation Primitives
**Goal**: Implement the approved normal, black-domain, cutout, and compositing augmentations.
**Success Criteria**: Positive outputs retain all four points; incomplete targets become unlabelled negatives.
**Tests**: Validate translation, polygon intersection, and compositing labels.
**Status**: Complete

## Stage 3: Streaming Dataset Writer
**Goal**: Generate the configured approximately 20k dataset without retaining images in memory.
**Success Criteria**: Output count and per-strategy quotas match configuration, with a progress bar.
**Tests**: Generate a small temporary dataset and check its manifest.
**Status**: In Progress

## Stage 4: Visual Verification
**Goal**: Export labelled preview images for every augmentation strategy.
**Success Criteria**: Preview images visibly show correct corner locations and expected negative samples.
**Tests**: Confirm preview files are generated.
**Status**: Not Started

## Stage 5: Dataset Split
**Goal**: Create a reproducible train/validation split for the generated pose dataset.
**Success Criteria**: Images and matching labels are copied into YOLO-compatible split directories without changing the source dataset.
**Tests**: Verify stratification, deterministic assignment, and generated data.yaml paths.
**Status**: In Progress

## Stage 6: Pose Training Launcher
**Goal**: Provide a double-clickable GPU training launcher for the split YOLO pose dataset.
**Success Criteria**: The launcher activates the `YOLO` Conda environment, uses `yolo26n-pose.pt`, and stores results under `runs/pose`.
**Tests**: Validate the batch syntax and configured dataset, model, and CUDA device paths.
**Status**: Complete

## Stage 7: Training Configuration Verification
**Goal**: Confirm the launcher settings match the dataset and available RTX 4070 GPU.
**Success Criteria**: The launcher uses the four-keypoint dataset, 640 image size, GPU 0, and a conservative 8 GB VRAM batch size.
**Tests**: Inspect all configured variables and the dataset metadata.
**Status**: Complete
