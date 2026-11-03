# 졸업운전감지시스템 — Drowsy driving detection system

Graduation project, Hanyang University ERICA.

A YOLO detector reads the driver's eye and mouth state frame by frame; a fatigue
rule turns those detections into a warning.

Four classes: `closed_eye`, `closed_mouth`, `open_eye`, `open_mouth`.

```
frame ──► YOLO (4 states) ──► fatigue rule ──► alert (speech / beep)
                                   │
                              sliding window
```

## Repository contents

```
app.py                         desktop application: camera, video, still image
fatigue.py                     the fatigue rule, independent of UI and model
tests/test_fatigue_rule.py     nine unit tests for the rule (no camera needed)
data/data.yaml                 class definition and dataset paths
data/images/                   a few sample frames
train.py                       training entry point (Ultralytics)
val.py                         validation entry point
requirements.txt               dependencies
runs/detect/train/             100-epoch run: results.csv, args.yaml, curves
runs/detect/train2/            a shorter 18-epoch run, kept for comparison
runs/detect/val/               PR / F1 curves, confusion matrices, prediction batches
```

## Running the application

```bash
pip install -r requirements.txt
pip install ultralytics

python app.py --model runs/detect/train/weights/best.pt            # pick a source in the UI
python app.py --model best.pt --source 0                            # default camera
python app.py --model best.pt --source drive.mp4                    # a recorded clip
python app.py --model best.pt --source drive.mp4 --threshold 5       # more sensitive
```

The window shows the annotated frame, the current verdict and how many of the
last ten frames were drowsy. The warning is spoken through `pyttsx3`; pass
`--no-voice` to fall back to a system beep.

| option | meaning |
|---|---|
| `--model` | path to the trained weights (default `runs/detect/train/weights/best.pt`) |
| `--source` | `0` for the default camera, or a path to a video |
| `--window` | frames in the sliding window (default 10) |
| `--threshold` | drowsy frames inside the window that trigger the warning (default 6) |
| `--confidence` | detection confidence threshold (default 0.35) |
| `--no-voice` | beep instead of speaking |

## Training

```bash
# point data/data.yaml at your local copy of the dataset, then:
python train.py
python val.py
```

The dataset itself is not distributed here; `data/data.yaml` uses relative paths
and must be edited to match where the images live.

## The fatigue rule

`fatigue.py` holds the rule on its own, so it can be tested without a camera, a
model or a display:

```bash
python -m unittest discover -s tests -t .
```

* a still image is flagged as soon as a closed eye **or** an open mouth is
  detected — there is no temporal context to lean on;
* a stream counts drowsy frames inside a sliding window and fires when the count
  reaches the threshold, which tolerates ordinary blinks and swallows.

With the defaults (6 of the last 10 frames) a single blink or yawn cannot raise
the alarm; sustained closure does.

## Results

Run `runs/detect/train` (100 epochs, YOLOv8n):

| precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.969 | 0.988 | 0.993 | 0.642 |

The per-epoch numbers are in `runs/detect/train/results.csv`; the validation
curves and confusion matrices are in `runs/detect/val/`.

## Notes

* Model weights (`*.pt`) are not committed — regenerate them with `train.py`.
* Presentation slides are not part of this repository.
