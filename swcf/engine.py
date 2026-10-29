from collections import deque
from dataclasses import dataclass, field
import numpy as np
from .config import Config
from .dsp import Analysis, analyze, pulse_intervals


@dataclass
class State:
    status: str = "Waiting for face"
    elapsed: float = 0.0
    analysis: Analysis | None = None
    hrv: dict | None = None
    timestamp: float = 0.0
    hrv_progress: float = 0.0


class Engine:
    def __init__(self, config=None, mode="weighted"):
        self.config = config or Config()
        self.mode = mode
        self.samples = deque(maxlen=round(self.config.history * self.config.fps * 3))
        self.state = State()
        self.last_update = -float("inf")
        self.last_hrv = -float("inf")

    def reset(self, status="Waiting for face"):
        self.samples.clear()
        self.state = State(status=status)
        self.last_update = self.last_hrv = -float("inf")

    def push(self, timestamp, rgb, quality):
        cfg = self.config
        rgb, quality = np.asarray(rgb, float), np.asarray(quality, float)
        if (rgb.shape != (5, 3) or quality.shape != (5,) or not np.isfinite(timestamp)
                or not np.all(np.isfinite(rgb)) or not np.all(np.isfinite(quality))):
            self.reset("Invalid sample")
            return False
        if np.max(quality) < cfg.min_quality:
            self.reset("Poor signal - improve lighting / hold still")
            return False
        if self.samples:
            gap = timestamp - self.samples[-1][0]
            if gap <= 0:
                raise ValueError("Timestamps must increase")
            if gap > cfg.max_gap:
                self.reset("Capture gap - reacquiring")
        self.samples.append((timestamp, rgb.copy(), np.clip(quality, 0, 1)))
        while self.samples and timestamp - self.samples[0][0] > cfg.history:
            self.samples.popleft()
        elapsed = timestamp - self.samples[0][0]
        self.state.elapsed = elapsed
        self.state.timestamp = timestamp
        self.state.hrv_progress = min(1, elapsed / (cfg.hrv_seconds + 1))
        if elapsed < cfg.window:
            self.state.status = f"Acquiring pulse: {elapsed:.1f} / {cfg.window:.0f} s"
            return False
        if timestamp - self.last_update < cfg.update - 1e-6:
            return False
        self.last_update = timestamp
        short = [s for s in self.samples if s[0] >= timestamp - cfg.window - 1e-6]
        try:
            self.state.analysis = analyze(*self._arrays(short), cfg, self.mode)
        except ValueError:
            self.reset("Timing unstable - reacquiring")
            return False
        good = self.state.analysis.bpm is not None
        self.state.status = "Tracking pulse" if good else "Poor signal - improve lighting / hold still"
        # A rejected short window breaks HRV continuity, even when a single
        # periodic artifact looks plausible in a full-history spectrum.
        if not good:
            self.state.hrv = None
            self.samples = deque(short, maxlen=self.samples.maxlen)
            self.state.hrv_progress = 0.0
        elif elapsed >= cfg.hrv_seconds + 1 and timestamp - self.last_hrv >= 5:
            self.last_hrv = timestamp
            history = [s for s in self.samples if s[0] >= timestamp - cfg.hrv_seconds - 2]
            long = analyze(*self._arrays(history), cfg, self.mode)
            # Require sustained physical quality in every half-second block.
            q = np.asarray([s[2].max() for s in history])
            self.state.hrv = pulse_intervals(long, cfg.hrv_seconds) if np.min(q) >= .35 else None
        return True

    @staticmethod
    def _arrays(samples):
        return tuple(np.asarray(x) for x in zip(*samples))
