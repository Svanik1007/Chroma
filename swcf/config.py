from dataclasses import dataclass

PATCH_NAMES = ("Forehead", "Left cheek", "Right cheek", "Nose", "Chin")


@dataclass(frozen=True)
class Config:
    fps: float = 30.0
    window: float = 6.0
    history: float = 65.0
    hrv_seconds: float = 60.0
    update: float = 0.5
    low: float = 0.75
    high: float = 2.5
    min_quality: float = 0.20
    min_concentration: float = 0.40
    max_gap: float = 0.25

    def __post_init__(self):
        if not (0 < self.low < self.high < self.fps / 2):
            raise ValueError("Require 0 < low < high < fps/2")
        if self.window < 6 or self.history < self.hrv_seconds + 2:
            raise ValueError("Window must be >=6 s; history must exceed HRV interval by 2 s")
