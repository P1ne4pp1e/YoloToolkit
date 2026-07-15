from pathlib import Path

import yolo_toolkit


def test_package_has_stable_version():
    assert yolo_toolkit.__version__


def test_capability_directories_exist():
    package_root = Path(yolo_toolkit.__file__).parent
    assert (package_root / "camera").is_dir()
    assert (package_root / "conversion").is_dir()
    assert (package_root / "inference").is_dir()
