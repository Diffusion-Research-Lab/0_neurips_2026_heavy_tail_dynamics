"""Tests for datakit raw-data staging helpers."""

from datakit._prepare import prepare_cifar100_lt, prepare_imagenet_lt, prepare_wildfires


def test_prepare_cifar100_lt_symlinks_source(tmp_path):
    source = tmp_path / "source" / "cifar-100-python"
    source.mkdir(parents=True)
    for name in ("train", "test", "meta"):
        (source / name).write_bytes(b"fixture")

    root = tmp_path / "root"
    prepare_cifar100_lt(root=root, source=source)

    assert (root / "cifar-100-python" / "train").is_file()


def test_prepare_imagenet_lt_copies_annotations_and_symlinks_source(tmp_path):
    source = tmp_path / "imagenet"
    source.mkdir()
    root = tmp_path / "imagenet_lt"

    prepare_imagenet_lt(root=root, source=source)

    assert (root / "annotations" / "ImageNet_LT_train.txt").is_file()
    assert (root / "imagenet").exists()


def test_prepare_wildfires_keeps_existing_file(tmp_path):
    root = tmp_path / "powerlaws"
    root.mkdir()
    target = root / "fires.txt"
    target.write_text("1\n2\n", encoding="utf-8")

    prepare_wildfires(root=root)

    assert target.read_text(encoding="utf-8") == "1\n2\n"
