# Duckietown Object Detection — Training

A training repository for four-class object detection in Duckietown RGB camera imagery. **YOLOv8n** is the default model. The code is separate from the dataset, which is distributed via Google Drive.

## Dataset

**Google Drive ZIP:** https://drive.google.com/file/d/1JclpoeUNNpNY3tl0JWeg1ozM_2D8Hp4s/view?usp=sharing.

The **iCPS Duckietown Detection Dataset**, containing 4,886 annotated images in the audited existing split: 4,143 train / 371 validation / 372 test. 

The dataset ZIP should contain either of these standard YOLO layouts. The repository also includes a portable `data.yaml` template: if the dataset ZIP lacks one, copy the template to `dataset/data.yaml`.

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

The script accepts any model file that the installed Ultralytics version can train; for example, `--model yolov8m.pt`. 


