# 졸업운전감지시스템 — Drowsy driving detection system

Graduation project, Hanyang University ERICA.

A YOLO detector reads the driver's eye and mouth state frame by frame; a fatigue
rule turns those detections into a warning.

Four classes: `closed_eye`, `closed_mouth`, `open_eye`, `open_mouth`.

```
frame ──► YOLO (4 states) ──► fatigue counter ──► alert
```

## Repository contents

```
data/data.yaml                 class definition and dataset paths
data/images/                   a few sample frames
train.py                       training entry point (Ultralytics)
val.py                         validation entry point
requirements.txt               dependencies
runs/detect/train/             100-epoch run: results.csv, args.yaml, curves
runs/detect/train2/            a shorter 18-epoch run, kept for comparison
runs/detect/val/               PR / F1 curves, confusion matrices, prediction batches
```

## Training

```bash
pip install -r requirements.txt
pip install ultralytics

# point data/data.yaml at your local copy of the dataset, then:
python train.py
python val.py
```

The dataset itself is not distributed here; `data/data.yaml` uses relative paths
and must be edited to match where the images live.

Fatigue rule used in the evaluation:

* single image — a closed eye **or** an open mouth counts as fatigue;
* video or camera — count closed-eye and open-mouth frames in a sliding window
  of 10 frames; above a threshold the driver is flagged as fatigued.

## Results

Run `runs/detect/train` (100 epochs, YOLOv8n):

| precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.969 | 0.988 | 0.993 | 0.642 |

The per-epoch numbers are in `runs/detect/train/results.csv`; the validation
curves and confusion matrices are in `runs/detect/val/`.

## Notes

* Model weights (`*.pt`) are not committed — regenerate them with `train.py`.
* The desktop application used for the live demo is not part of this repository.
* Presentation slides are not part of this repository.
