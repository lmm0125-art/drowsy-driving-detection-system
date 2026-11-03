"""The fatigue rule, kept free of any UI or model dependency.

Two ways of deciding that a driver is fatigued:

* a still image has no temporal context, so any drowsy sign is enough;
* a stream is judged over a sliding window of frames, which suppresses the odd
  blink or yawn that a per-frame rule would flag.

Keeping this in its own module means the rule can be unit-tested without a
camera, a model or a display.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

#: Detections that count as evidence of fatigue.
DROWSY_CLASSES = ("closed_eye", "open_mouth")

#: Every class the detector can produce.
CLASSES = ("closed_eye", "closed_mouth", "open_eye", "open_mouth")


@dataclass
class FatigueRule:
    """Sliding-window counter over per-frame drowsiness flags.

    Parameters
    ----------
    window:
        How many recent frames the counter looks at.
    threshold:
        How many drowsy frames inside the window trigger the warning. A value
        equal to ``window`` means "every frame in the window must be drowsy",
        which is very conservative; the default 6/10 tolerates blinks.
    """

    window: int = 10
    threshold: int = 6
    history: deque = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if not 1 <= self.threshold <= self.window:
            raise ValueError("threshold must be between 1 and window")
        self.history = deque(maxlen=self.window)

    def update(self, labels: Iterable[str]) -> tuple[bool, int]:
        """Feed one frame's labels; return ``(is_fatigued, drowsy_count)``."""
        self.history.append(1 if self._drowsy(labels) else 0)
        count = int(sum(self.history))
        return count >= self.threshold, count

    def reset(self) -> None:
        self.history.clear()

    @property
    def frames_seen(self) -> int:
        return len(self.history)

    @staticmethod
    def judge_image(labels: Iterable[str]) -> bool:
        """A still image: any drowsy sign makes it a fatigued snapshot."""
        return FatigueRule._drowsy(labels)

    @staticmethod
    def _drowsy(labels: Iterable[str]) -> bool:
        return any(name in DROWSY_CLASSES for name in labels)
