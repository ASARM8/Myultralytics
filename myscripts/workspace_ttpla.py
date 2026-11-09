"""Locate the TTPLA server split and prepare its detector config without editing the dataset."""
import argparse
import os
import tempfile
from pathlib import Path

SERVER_DATA_YAML = Path("/workspace/datasets/TTPLA-640-811/dataset.yaml")


def refine_training_dir(work=Path("/workspace/workdir")):
    override = os.environ.get("REFINE_DIR")
    if override:
        return Path(override)
    default = work / "refine_v311_ca_seed0"
    candidates = [default, *sorted(work.glob("refine_v311_ca_seed0_*"))]
    runs = [path for path in candidates if (path / "checkpoints/best.pt").is_file()]
    if len(runs) > 1:
        raise ValueError("Multiple Refine runs found. Set REFINE_DIR to the desired run: " + ", ".join(map(str, runs)))
    if runs:
        return runs[0]
    legacy = work / "paper_exports/refine_v311_seed0/train_geometry_only"
    return legacy if (legacy / "checkpoints/best.pt").is_file() else default


def dataset_root():
    base = Path("/workspace/datasets")
    override = os.environ.get("TTPLA_ROOT")
    candidates = [Path(override)] if override else [base / "TTPLA-640-811", base]
    for candidate in candidates:
        if (candidate / "images/train").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing images/train in {candidates}. Set TTPLA_ROOT to the dataset directory.")


def prepare_detector_yaml():
    import yaml

    root = dataset_root()
    original = root / "dataset.yaml"
    if original.is_file():
        return original
    config = {}
    for split in ("train", "val", "test"):
        config[split] = f"images/{split}"
    config["path"] = str(root)
    config.setdefault("names", {0: "cable"})
    dest = Path("/workspace/workdir/configs/TTPLA-640-811.yaml")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=dest.parent, encoding="utf-8", delete=False) as f:
        temp = Path(f.name)
    try:
        temp.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        os.replace(temp, dest)
    finally:
        temp.unlink(missing_ok=True)
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("root", "yaml", "refine-dir"))
    args = parser.parse_args()
    actions = {"root": dataset_root, "yaml": prepare_detector_yaml, "refine-dir": refine_training_dir}
    print(actions[args.action]())
