# 졸업운전감지시스템

한양대학교 ERICA 졸업 프로젝트.

운전자의 눈과 입 상태를 프레임 단위로 검출하고, 그 결과를 피로 판정 규칙에 넣어
경고를 띄우는 시스템입니다.

검출 클래스는 네 가지입니다: `closed_eye`, `closed_mouth`, `open_eye`, `open_mouth`.

```
프레임 ──► YOLO (4개 상태) ──► 피로 판정 규칙 ──► 경고 (음성 / 비프음)
                                      │
                                 슬라이딩 윈도우
```

## 구성

```
app.py                         데스크톱 애플리케이션 (카메라 · 동영상 · 이미지)
fatigue.py                     피로 판정 규칙 (UI·모델과 분리된 순수 모듈)
tests/test_fatigue_rule.py     규칙 단위 테스트 9개 (카메라 없이 실행)
data/data.yaml                 클래스 정의와 데이터 경로
data/images/                   샘플 프레임
train.py                       학습 진입점 (Ultralytics)
val.py                         검증 진입점
requirements.txt               의존성
runs/detect/train/             100 에폭 학습 결과 (results.csv, args.yaml, 곡선)
runs/detect/train2/            비교용 18 에폭 학습 결과
runs/detect/val/               PR · F1 곡선, 혼동 행렬, 예측 배치
```

## 실행 방법

```bash
pip install -r requirements.txt
pip install ultralytics

python app.py --model runs/detect/train/weights/best.pt             # 실행 후 화면에서 입력 선택
python app.py --model best.pt --source 0                            # 기본 카메라
python app.py --model best.pt --source drive.mp4                    # 저장된 영상
python app.py --model best.pt --source drive.mp4 --threshold 5       # 더 민감하게
```

화면에는 검출 결과가 그려진 프레임, 현재 판정, 최근 프레임 중 몇 개가 피로로
판정됐는지가 함께 표시됩니다. 경고는 `pyttsx3`로 음성 출력되며, `--no-voice`를
주면 시스템 비프음으로 대체됩니다.

| 옵션 | 설명 |
|---|---|
| `--model` | 학습된 가중치 경로 (기본값 `runs/detect/train/weights/best.pt`) |
| `--source` | `0`이면 기본 카메라, 그 외에는 동영상 파일 경로 |
| `--window` | 슬라이딩 윈도우 크기, 프레임 단위 (기본 10) |
| `--threshold` | 윈도우 안에서 경고를 띄우는 피로 프레임 수 (기본 6) |
| `--confidence` | 검출 신뢰도 임계값 (기본 0.35) |
| `--no-voice` | 음성 대신 비프음 |

## 학습

```bash
# data/data.yaml의 train/val 경로를 로컬 데이터셋 위치로 수정한 뒤
python train.py
python val.py
```

데이터셋은 저장소에 포함되어 있지 않습니다. `data/data.yaml`은 상대 경로를 쓰고
있으므로 이미지가 있는 위치에 맞게 수정해야 합니다.

## 피로 판정 규칙

판정 로직은 `fatigue.py`에 따로 두어 카메라·모델·화면 없이도 검증할 수 있습니다.

```bash
python -m unittest discover -s tests -t .
```

* **이미지** — 시간 정보가 없으므로 눈 감김 또는 입 벌림이 하나라도 검출되면 피로로
  판정합니다.
* **스트림** — 최근 `window` 프레임 중 피로 프레임이 `threshold`개 이상이면 경고합니다.

기본값(10프레임 중 6프레임)에서는 한 번의 눈 깜빡임이나 하품으로는 경고가 뜨지
않고, 눈을 계속 감고 있는 경우에만 경고가 발생합니다.

## 결과

`runs/detect/train` (100 에폭, YOLOv8n) 기준입니다.

| 정밀도 | 재현율 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| 0.969 | 0.988 | 0.993 | 0.642 |

에폭별 수치는 `runs/detect/train/results.csv`, 검증 곡선과 혼동 행렬은
`runs/detect/val/`에 있습니다.

## 참고

* 모델 가중치(`*.pt`)는 저장소에 포함하지 않았습니다. `train.py`로 다시 만들 수 있습니다.
* 발표 자료는 저장소에 포함하지 않았습니다.
