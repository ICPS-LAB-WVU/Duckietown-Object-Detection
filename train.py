#!/usr/bin/env python3
"""Minimal, portable Duckietown YOLO detector trainer (YOLOv8 by default).

No hard-coded machine paths and no mandatory Weights & Biases setup.
Use `python train.py --check-data` before training.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    import yaml
except ImportError as exc:
    raise SystemExit("Missing PyYAML. Run: python -m pip install -r requirements.txt") from exc

ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_NAMES = ["Bot", "Intersection_sign", "Stop_sign", "Traffic_light"]

# Standard Ultralytics folder styles, including Roboflow's `valid` spelling.
CANDIDATES = {
    "train": ("images/train", "train/images"),
    "val": ("images/val", "val/images", "images/valid", "valid/images"),
    "test": ("images/test", "test/images"),
}


def locate_data_yaml(requested: str) -> Path:
    path = Path(requested).expanduser()
    path = path if path.is_absolute() else ROOT / path
    if path.is_file():
        return path.resolve()
    # Previous ZIP releases may contain one enclosing dataset-named folder.
    if path.name == "data.yaml" and path.parent.is_dir():
        choices = sorted(p for p in path.parent.glob("*/data.yaml") if p.is_file())
        if len(choices) == 1:
            print(f"[info] Found nested dataset YAML: {choices[0]}")
            return choices[0].resolve()
    raise FileNotFoundError(
        f"Dataset YAML not found: {path}\n"
        "Extract the detection ZIP under this repository's dataset/ directory, "
        "or pass --data /path/to/data.yaml."
    )


def images_in(path: Path) -> list[Path]:
    return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def resolve_images_folder(root: Path, key: str, requested: object) -> Path | None:
    choices = []
    if isinstance(requested, str):
        specified = Path(requested)
        # Ignore stale /home/... image folders from someone else's machine.
        if not specified.is_absolute():
            choices.append(specified.as_posix())
    choices.extend(CANDIDATES[key])
    seen = set()
    existing = []
    for relative in choices:
        if relative in seen:
            continue
        seen.add(relative)
        folder = (root / relative).resolve()
        if folder.is_dir() and images_in(folder):
            existing.append(folder)
    if not existing:
        return None
    # Prefer the explicitly requested folder, otherwise the first known layout.
    return existing[0]


def labels_for(image_dir: Path, root: Path) -> Path:
    relative = image_dir.relative_to(root)
    segments = list(relative.parts)
    if "images" not in segments:
        raise ValueError(f"Expected an 'images' path component in {image_dir}")
    segments[segments.index("images")] = "labels"
    return root.joinpath(*segments)


def inspect_dataset(data_yaml: Path) -> dict:
    root = data_yaml.parent.resolve()
    with data_yaml.open("r", encoding="utf-8") as stream:
        supplied = yaml.safe_load(stream) or {}
    names = supplied.get("names")
    if isinstance(names, dict):
        normalized = [names.get(i, names.get(str(i))) for i in range(len(names))]
    elif isinstance(names, list):
        normalized = names
    else:
        raise ValueError("data.yaml must define names as a list or class-ID mapping")
    if normalized != CLASS_NAMES:
        raise ValueError(f"Class order mismatch. Expected {CLASS_NAMES}, found {normalized}")

    resolved = {"path": str(root), "names": {i: name for i, name in enumerate(CLASS_NAMES)}, "nc": 4}
    for split in ("train", "val", "test"):
        folder = resolve_images_folder(root, split, supplied.get(split))
        if folder is None:
            if split == "test":
                print("[info] No test split found; training/validation will still work.")
                continue
            raise FileNotFoundError(
                f"Could not find nonempty {split} images under {root}. "
                f"Supported relative paths: {', '.join(CANDIDATES[split])}"
            )
        labels_dir = labels_for(folder, root)
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"Missing corresponding labels directory: {labels_dir}")
        image_files = images_in(folder)
        missing = []
        for image in image_files:
            label = labels_dir / image.relative_to(folder).with_suffix(".txt")
            if not label.is_file():
                missing.append(str(image.relative_to(folder)))
                if len(missing) >= 5:
                    break
        if missing:
            raise ValueError(f"{split}: missing YOLO label files in {labels_dir}: {missing}")
        relative = folder.relative_to(root).as_posix()
        resolved[split] = relative
        print(f"[{split:5}] {len(image_files):5} images  |  {relative}  |  labels: OK")

    return resolved


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="dataset/data.yaml", help="Dataset YAML relative to repo root, or absolute")
    p.add_argument("--check-data", action="store_true", help="Validate file paths/labels; do not train")
    p.add_argument("--model", default="yolov8n.pt", help="Ultralytics-compatible checkpoint or model YAML")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="Optional, e.g. cpu, 0 or 0,1. Default: auto")
    p.add_argument("--name", default="yolov8n_course")
    p.add_argument("--test", action="store_true", help="After training, evaluate best.pt on held-out test split")
    args = p.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32:
        p.error("epochs and batch must be positive; imgsz must be >= 32")

    data_yaml = locate_data_yaml(args.data)
    print("Dataset YAML:", data_yaml)
    resolved = inspect_dataset(data_yaml)
    if args.check_data:
        print("PASS: Dataset paths and image/label pairs verified. No training started.")
        return

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics not installed. Run: python -m pip install -r requirements.txt") from exc

    runs_dir = ROOT / "runs" / "detect"
    runs_dir.mkdir(parents=True, exist_ok=True)
    # Ultralytics receives an absolute root regardless of ZIP extraction path,
    # current working directory, or local Ultralytics datasets_dir setting.
    generated_yaml = runs_dir / "resolved_data.yaml"
    with generated_yaml.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(resolved, stream, sort_keys=False)
    print("Runtime training YAML:", generated_yaml)
    print("Model:", args.model)

    # Accept any checkpoint supported by the installed Ultralytics package.
    # YOLOv8n is tested by design; YOLOv13 custom checkpoints may need a
    # matching research implementation/version and are not course defaults.
    model = YOLO(args.model)
    training_options = dict(
        data=str(generated_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        seed=args.seed,
        project=str(runs_dir),
        name=args.name,
        plots=True,
    )
    if args.device:
        training_options["device"] = args.device
    model.train(**training_options)
    # Unlike inference results, the training result is not guaranteed to expose
    # `save_dir` across Ultralytics versions. The trainer does.
    run_dir = Path(model.trainer.save_dir)
    print("Training results:", run_dir)

    if args.test:
        if "test" not in resolved:
            raise SystemExit("--test requested, but no test/images directory was found")
        best = run_dir / "weights" / "best.pt"
        if not best.is_file():
            raise FileNotFoundError(f"Expected trained checkpoint missing: {best}")
        evaluator = YOLO(str(best))
        test_options = dict(
            data=str(generated_yaml), split="test", imgsz=args.imgsz,
            batch=args.batch, project=str(runs_dir), name=args.name + "_test",
        )
        if args.device:
            test_options["device"] = args.device
        scores = evaluator.val(**test_options)
        print(f"Test mAP50={scores.box.map50:.4f}, mAP50-95={scores.box.map:.4f}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as err:
        sys.exit(f"ERROR: {err}")
