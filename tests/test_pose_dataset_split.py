from pathlib import Path

from scripts.data_processing.split_pose_dataset import split_dataset


def write_sample(root: Path, stem: str, label: str) -> None:
    root.joinpath("images").mkdir(parents=True, exist_ok=True)
    root.joinpath("labels").mkdir(parents=True, exist_ok=True)
    root.joinpath("images", f"{stem}.jpg").write_bytes(b"image")
    root.joinpath("labels", f"{stem}.txt").write_text(label, encoding="utf-8")


def test_split_dataset_copies_matching_files_and_writes_pose_yaml(tmp_path):
    source = tmp_path / "source"
    for index in range(5):
        write_sample(source, f"aug_normal_{index:05d}", "0 0.5 0.5")
        write_sample(source, f"aug_black_negative_{index:05d}", "")

    output = tmp_path / "split"
    counts = split_dataset(source, output, {"train": 0.8, "val": 0.2, "test": 0.0}, seed=7)

    assert counts == {"train": 8, "val": 2, "test": 0}
    assert len(list(output.joinpath("images", "train").glob("*.jpg"))) == 8
    assert len(list(output.joinpath("labels", "val").glob("*.txt"))) == 2
    assert "kpt_shape: [4, 3]" in output.joinpath("data.yaml").read_text(encoding="utf-8")


def test_split_dataset_is_deterministic_and_stratifies_positive_negative_samples(tmp_path):
    source = tmp_path / "source"
    for index in range(10):
        write_sample(source, f"aug_composite_{index:05d}", "0 0.5 0.5")
        write_sample(source, f"aug_cutout_{index:05d}", "")

    ratios = {"train": 0.8, "val": 0.2, "test": 0.0}
    split_dataset(source, tmp_path / "split_a", ratios, seed=11)
    split_dataset(source, tmp_path / "split_b", ratios, seed=11)

    train_a = sorted(path.name for path in (tmp_path / "split_a" / "images" / "train").glob("*.jpg"))
    train_b = sorted(path.name for path in (tmp_path / "split_b" / "images" / "train").glob("*.jpg"))
    val_labels = [path.read_text(encoding="utf-8") for path in (tmp_path / "split_a" / "labels" / "val").glob("*.txt")]

    assert train_a == train_b
    assert any(label for label in val_labels)
    assert any(not label for label in val_labels)
