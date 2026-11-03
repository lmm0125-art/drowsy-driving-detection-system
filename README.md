# 졸업 프로젝트 — Drowsy driving detection

Detection and warning system for driver fatigue. A YOLO detector reads eye and
mouth states frame by frame; a fatigue rule turns the detections into a warning
that is spoken aloud and, on Windows, accompanied by a system sound.

Four classes: `closed_eye`, `closed_mouth`, `open_eye`, `open_mouth`.

```
camera / video ──► YOLO (4 states) ──► fatigue counter ──► TTS + sound alert
                                          │
                                          └──► annotated frame in the GUI
```

## Running it

```bash
pip install -r requirements.txt
pip install ultralytics          # not vendored here, see below
python gui_ko_alert2.py          # latest GUI: Korean UI + spoken alerts
```

`train.py` and `val.py` call the Ultralytics CLI with `data/data.yaml`. The
dataset itself is not in this repository; the paths in `data/data.yaml` are
relative and must point at your local copy.

| file | what it is |
|---|---|
| `gui_ko_alert2.py` | latest version — Korean UI, spoken warnings (`pyttsx3`), Windows sound |
| `gui_ko_alert.py` | same idea, previous iteration |
| `gui.py` / `gui_en.py` | earlier interfaces, Chinese and English labels |
| `train.py` / `val.py` | training and validation entry points |
| `data/` | class definition (`data.yaml`) and a handful of sample frames |
| `runs/detect/` | training logs and validation artefacts — PR curves, confusion matrices, prediction batches |
| `项目使用说明.txt` | usage notes |

## Results

Run `runs/detect/train` (100 epochs):

| precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.969 | 0.988 | 0.993 | 0.642 |

`runs/detect/val` holds the validation curves for that run; `results.csv` and
`args.yaml` are kept for every run so the numbers can be reproduced.

## Not in this repository

| excluded | why |
|---|---|
| `*.pptx` (4 files) | presentation material — excluded on request |
| `runs/**/weights/*.pt`, `yolo11n.pt`, `yolov8n.pt` | ~60 MB of weights; regenerate with `train.py`, or download the public checkpoints |
| `ultralytics/` | a pristine vendored copy of the library (no local modifications); `pip install ultralytics` |
| `Arial.ttf` | font for the GUI, licence does not allow redistribution |
| `.idea/`, `.vscode/`, `desktop.ini` | editor and OS metadata |

## Provenance

The GUI layout started from a purchased template. `使用前必读.txt` carries the
original notice: personal use only, no resale. The detection model, the training
configuration and the four-class dataset are this project's own.
