from dataclasses import dataclass

PATCH_NAMES = ("Forehead", "Left cheek", "Right cheek", "Nose", "Chin")


@dataclass(frozen=True)
class Config:
    fps: float = 30.0
    window: float = 6.0          # first estimate after this many seconds
    window_max: float = 12.0     # analysis window grows up to this length
    history: float = 65.0
    hrv_seconds: float = 60.0
    update: float = 0.5
    low: float = 0.75
    high: float = 2.5
    min_quality: float = 0.20
    min_concentration: float = 0.40
    max_gap: float = 1.0         # short gaps (blink of tracking, slow frame) are bridged
    bad_reset_s: float = 2.0     # reset only after this long of continuously bad quality
    face_loss_reset_s: float = 1.5
    hold_s: float = 5.0          # keep showing the last good BPM this long after rejection
    quality_smooth_s: float = 0.3
    track_sigma_bpm: float = 15.0
    smooth_count: int = 5        # median of this many accepted estimates
    smooth_alpha: float = 0.3    # then exponential smoothing

    def __post_init__(self):
        if not (0 < self.low < self.high < self.fps / 2):
            raise ValueError("Require 0 < low < high < fps/2")
        if self.window < 6 or self.history < self.hrv_seconds + 2:
            raise ValueError("Window must be >=6 s; history must exceed HRV interval by 2 s")
        if self.window_max < self.window:
            raise ValueError("window_max must be >= window")
