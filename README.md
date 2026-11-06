# Duckietown Object Detection — YOLOv8 Training

A small student-facing training repository for four-class object detection in Duckietown camera imagery. **YOLOv8n** is the default model. The code is separate from the dataset, which is distributed via Google Drive.

## Dataset

**Google Drive ZIP:** `[ADD DIRECT DATASET FILE LINK]` (instructor: insert the shared ZIP file link).

The recommended course collection is **iCPS Duckietown Detection Dataset**, containing 4,886 annotated images in the audited existing split: 4,143 train / 371 validation / 372 test. This is a released split, not a verified reconstruction of the accepted paper's original training split.

The dataset ZIP should contain either of these standard YOLO layouts. The repository also includes a portable `data.yaml` template: if the dataset ZIP lacks one, copy the template to `dataset/data.yaml`. The script resolves the root at runtime even if the ZIP contains an older absolute `path:` entry. `train.py` accepts both forms as well as the `valid` spelling.

```text
# Recommended original dataset layout
<repo>/dataset/
    data.yaml
    images/train/    images/val/    images/test/
    labels/train/    labels/val/    labels/test/

# Also supported: Roboflow-style layout
<repo>/dataset/
    data.yaml
    train/images/    train/labels/
    valid/images/    valid/labels/
    test/images/     test/labels/
```

An extra top-level dataset-named directory inside `dataset/` is also accepted if it contains exactly one `data.yaml`. Do not combine the original clean validation split with adversarial sets such as `valid_fgsmv8`.

Class IDs are fixed: `0 Bot`, `1 Intersection_sign`, `2 Stop_sign`, `3 Traffic_light`.

## Setup

```bash
git clone https://github.com/ICPS-LAB-WVU/Duckietown-Object-Detection.git
cd Duckietown-Object-Detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

mkdir -p dataset
unzip -q ~/Downloads/iCPS_Duckietown_Detection_Dataset.zip -d dataset/
```

Replace the Git clone URL if your repository is named differently; replace the ZIP filename with the actual download name. For GPU installations, choose the matching PyTorch build for your platform before installing requirements.

If the extracted ZIP does not include a `data.yaml`, run `cp data.yaml dataset/data.yaml` (if it has an enclosing dataset-named directory, place it inside that directory instead).

## Check before training

```bash
python train.py --check-data
```

This prints the actual image paths and verifies a corresponding label file for every image. It does **not** start training.

## Train YOLOv8

Quick one-epoch smoke test:

```bash
python train.py --model yolov8n.pt --epochs 1 --batch 8
```

Course assignment example:

```bash
python train.py --model yolov8n.pt --epochs 50 --batch 8 --imgsz 640
```

Pretrained `yolov8n.pt` may be downloaded automatically by Ultralytics. Training outputs, including `best.pt`, are saved under `runs/detect/`. The script uses a runtime dataset YAML with an absolute `path` automatically; students do not edit local paths.

To evaluate the held-out test split after training:

```bash
python train.py --model yolov8n.pt --epochs 50 --batch 8 --test
```

This trains a new model and evaluates its best checkpoint. Do not repeatedly tune hyperparameters on the test split.

## Optional different models

The script accepts any model file that the installed Ultralytics version can train; for example, `--model yolov8m.pt`. **YOLOv13 is not a required course dependency**. If you already have a compatible custom YOLOv13 implementation/checkpoint, you may try it through `--model /path/to/yolov13n.pt`, but compatibility with stock Ultralytics is not guaranteed and has not been tested here.

## Notes

- Only detection labels and normal validation data are required; adversarial attack datasets are outside this assignment.
- Weights & Biases, flash-attention, COCO calibration code, and plotting comparisons are not required.
- The packaged dataset should not be committed to GitHub: it belongs in the Google Drive ZIP.
- The detector deployed in a related 2026 research implementation was verified as **YOLOv8m**, whereas one paper description says YOLOv8n. This course repository trains a new YOLOv8n model by default, **not** a reproduction of the deployed checkpoint.
