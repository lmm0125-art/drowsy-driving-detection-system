"""Real-time driver fatigue monitor — desktop front end.

Reads frames from a camera, a video file or a still image, runs the detector
trained by ``train.py``, applies the rule in :mod:`fatigue` and warns the driver
when the rule fires.

    python app.py --model runs/detect/train/weights/best.pt
    python app.py --model best.pt --source 0          # default camera
    python app.py --model best.pt --source drive.mp4  # a recorded clip

The window and threshold are command line options, so a demo can be made more
or less sensitive without editing code.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from fatigue import FatigueRule

try:                                    # speech is a nicety, not a dependency
    import pyttsx3
except Exception:                       # pragma: no cover - optional
    pyttsx3 = None


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
class Detector:
    """Thin wrapper around Ultralytics YOLO that also returns the labels."""

    def __init__(self, weights: str, confidence: float = 0.35):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.confidence = confidence
        self.names = self.model.names

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, list[str]]:
        result = self.model.predict(frame, conf=self.confidence, verbose=False)[0]
        plotted = result.plot()

        labels: list[str] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and boxes.cls is not None:
            for cls_id in boxes.cls.tolist():
                name = self.names.get(int(cls_id), str(int(cls_id)))
                labels.append(name)
        return plotted, labels


# --------------------------------------------------------------------------- #
# Capture thread
# --------------------------------------------------------------------------- #
class Stream:
    """Read frames off the UI thread so the window stays responsive."""

    def __init__(self, source: int | str):
        self.source = source
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        # a file would otherwise be read as fast as the disk allows; pace it to
        # roughly the source frame rate so the window plays at a watchable speed
        is_file = not isinstance(self.source, int)
        fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        interval = 1.0 / fps if is_file and 1 <= fps <= 240 else 0.0

        while self._running:
            ok, frame = self.capture.read()
            if not ok:
                self._running = False
                break
            with self._lock:
                self._frame = frame
            if interval:
                time.sleep(interval)

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    @property
    def running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.capture.release()


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
class MonitorApp:
    """Video canvas, a status line and four buttons."""

    POLL_MS = 30
    IDLE_COLOUR = "#8fa3b8"
    OK_COLOUR = "#4fd08a"
    ALERT_COLOUR = "#ff5d5d"

    def __init__(self, detector: Detector, rule: FatigueRule,
                 source: int | str | None = None, speak: bool = True):
        self.detector = detector
        self.rule = rule
        self.source = source
        self.speak = speak
        self.stream: Optional[Stream] = None
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.warned = False

        self.root = tk.Tk()
        self.root.title("Drowsy driving monitor")
        self.root.configure(bg="#101418")
        self.root.geometry("960x720")

        self.canvas = tk.Canvas(self.root, bg="#0b0f13", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=12, pady=(12, 8))

        status_bar = tk.Frame(self.root, bg="#101418")
        status_bar.pack(fill="x", padx=12)
        self.status = tk.Label(status_bar, text="idle", fg=self.IDLE_COLOUR,
                               bg="#101418", font=("Helvetica", 15, "bold"),
                               anchor="w")
        self.status.pack(side="left")
        self.detail = tk.Label(status_bar, text="", fg=self.IDLE_COLOUR,
                               bg="#101418", font=("Helvetica", 11), anchor="e")
        self.detail.pack(side="right")

        buttons = tk.Frame(self.root, bg="#101418")
        buttons.pack(fill="x", padx=12, pady=12)
        for text, command in (("Camera", self.open_camera),
                              ("Open video", self.open_video),
                              ("Open image", self.open_image),
                              ("Stop", self.stop)):
            tk.Button(buttons, text=text, command=command, width=12,
                      relief="flat", bg="#1e2a36", fg="#dbe6f0",
                      activebackground="#2b3d4f", activeforeground="#ffffff",
                      font=("Helvetica", 12)).pack(side="left", padx=(0, 8))

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # -- sources ---------------------------------------------------------- #
    def open_camera(self) -> None:
        self.start_stream(0, "camera")

    def open_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Open video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if path:
            self.start_stream(path, Path(path).name)

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")])
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Drowsy driving monitor", f"Cannot read {path}")
            return
        self.stop()
        plotted, labels = self.detector(frame)
        self.show(plotted)
        fatigued = FatigueRule.judge_image(labels)
        self.set_status(fatigued, None, labels)
        if fatigued:
            self.alert()

    # -- stream ----------------------------------------------------------- #
    def start_stream(self, source: int | str, title: str) -> None:
        self.stop()
        try:
            self.stream = Stream(source)
        except RuntimeError as exc:
            messagebox.showerror("Drowsy driving monitor", str(exc))
            return
        self.rule.reset()
        self.warned = False
        self.stream.start()
        self.status.config(text=f"{title} — monitoring", fg=self.IDLE_COLOUR)
        self.root.after(self.POLL_MS, self.tick)

    def tick(self) -> None:
        if self.stream is None:
            return
        frame = self.stream.read()
        if frame is None:
            if not self.stream.running:
                self.status.config(text="stream finished", fg=self.IDLE_COLOUR)
                return
            self.root.after(self.POLL_MS, self.tick)
            return

        plotted, labels = self.detector(frame)
        fatigued, count = self.rule.update(labels)
        self.show(plotted)
        self.set_status(fatigued, count, labels)

        if fatigued and not self.warned:
            self.alert()
            self.warned = True
        elif not fatigued:
            self.warned = False

        self.root.after(self.POLL_MS, self.tick)

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream = None
        self.rule.reset()
        self.warned = False
        self.status.config(text="idle", fg=self.IDLE_COLOUR)
        self.detail.config(text="")

    # -- rendering -------------------------------------------------------- #
    def show(self, frame_bgr: np.ndarray) -> None:
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        if width > 1 and height > 1:
            scale = min(width / image.width, height / image.height)
            if scale < 1:
                image = image.resize((int(image.width * scale),
                                      int(image.height * scale)))
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(width // 2, height // 2, image=self.photo)

    def set_status(self, fatigued: bool, count: Optional[int],
                   labels: Iterable[str]) -> None:
        self.status.config(
            text="FATIGUE DETECTED" if fatigued else "normal",
            fg=self.ALERT_COLOUR if fatigued else self.OK_COLOUR)

        seen = ", ".join(sorted(set(labels))) or "no detection"
        if count is None:
            self.detail.config(text=seen)
        else:
            self.detail.config(
                text=f"{count}/{self.rule.window} drowsy · {seen}")

    def alert(self) -> None:
        """Speak a warning, or beep when speech is unavailable.

        Speech runs on its own thread: ``runAndWait`` blocks, and blocking the
        UI thread would freeze the video for the duration of the sentence.
        """
        if self.speak and pyttsx3 is not None:
            threading.Thread(target=self._speak, daemon=True).start()
            return
        self._beep()

    @staticmethod
    def _speak() -> None:
        try:
            engine = pyttsx3.init()
            engine.say("Warning. Driver fatigue detected.")
            engine.runAndWait()
            engine.stop()
        except Exception:
            pass

    def _beep(self) -> None:
        try:
            self.root.bell()
        except Exception:
            pass

    # -- lifecycle -------------------------------------------------------- #
    def run(self) -> None:
        if self.source is not None:
            self.start_stream(self.source, "source")
        self.root.mainloop()

    def close(self) -> None:
        self.stop()
        self.root.destroy()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Driver fatigue monitor")
    parser.add_argument("--model", default="runs/detect/train/weights/best.pt",
                        help="path to the trained YOLO weights")
    parser.add_argument("--source", default=None,
                        help="0 for the default camera, or a path to a video")
    parser.add_argument("--window", type=int, default=10,
                        help="sliding window size in frames (default 10)")
    parser.add_argument("--threshold", type=int, default=6,
                        help="drowsy frames in the window that trigger a warning")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--no-voice", action="store_true",
                        help="beep instead of speaking")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not Path(args.model).exists():
        raise SystemExit(
            f"Model not found: {args.model}\n"
            "Train one with `python train.py`, or pass --model explicitly.")
    detector = Detector(args.model, confidence=args.confidence)
    rule = FatigueRule(window=args.window, threshold=args.threshold)
    MonitorApp(detector, rule, args.source, speak=not args.no_voice).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
