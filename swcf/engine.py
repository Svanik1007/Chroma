from collections import deque
from dataclasses import dataclass
import numpy as np
from .config import Config
from .dsp import Analysis, analyze, pulse_intervals

NO_BG = np.full(3, np.nan)


@dataclass
class State:
    status: str = "Waiting for face"
    elapsed: float = 0.0
    analysis: Analysis | None = None
    hrv: dict | None = None
    timestamp: float = 0.0
    hrv_progress: float = 0.0
    bpm_display: float | None = None   # smoothed value shown to the user
    held: bool = False                 # True while showing the last good value
    window_s: float = 0.0


class Engine:
    """Time buffer, update schedule, tracking and output smoothing.

    Brief problems (a nod, talking, a dropped face for a few frames) no longer
    wipe the buffer: their samples are kept with low quality, so the DSP
    down-weights them, and the last good BPM is held. The buffer is reset only
    after a sustained problem (config.bad_reset_s / face_loss_reset_s) or a
    real capture gap (config.max_gap).
    """

    def __init__(self, config=None, mode="weighted"):
        self.config = config or Config()
        self.mode = mode
        self.samples = deque(maxlen=round(self.config.history * self.config.fps * 3))
        self.reset()

    def reset(self, status="Waiting for face"):
        self.samples.clear()
        self.state = State(status=status)
        self.last_update = self.last_hrv = -float("inf")
        self.bad_since = None
        self.lost_since = None
        self.last_bad_t = -float("inf")
        self.last_good_t = -float("inf")
        self.track_bpm = None
        self.recent = deque(maxlen=self.config.smooth_count)
        self.challenge = 0

    # ------------------------------------------------------------------
    def face_lost(self, timestamp):
        """Call when the tracker finds no face in a frame."""
        if not self.samples:
            self.state.status = "No face - center your face and improve lighting"
            return False
        if self.lost_since is None:
            self.lost_since = timestamp
        if timestamp - self.lost_since > self.config.face_loss_reset_s:
            self.reset("No face - center your face and improve lighting")
            return False
        # Keep the timeline continuous with a zero-quality placeholder so the
        # missing frames are gated out rather than becoming a capture gap.
        last_t, rgb, _, bg = self.samples[-1]
        if timestamp > last_t:
            self.samples.append((timestamp, rgb, np.zeros(5), bg))
        self.last_bad_t = timestamp
        self.state.status = "Face lost - hold on"
        self._hold(timestamp)
        return False

    def push(self, timestamp, rgb, quality, background=None):
        cfg = self.config
        rgb, quality = np.asarray(rgb, float), np.asarray(quality, float)
        if (rgb.shape != (5, 3) or quality.shape != (5,) or not np.isfinite(timestamp)
                or not np.all(np.isfinite(rgb)) or not np.all(np.isfinite(quality))):
            return False  # skip a corrupt sample; the timeline tolerates it
        bg = NO_BG if background is None else np.asarray(background, float).reshape(3)
        self.lost_since = None
        if self.samples:
            gap = timestamp - self.samples[-1][0]
            if gap <= 0:
                raise ValueError("Timestamps must increase")
            if gap > cfg.max_gap:
                self.reset("Capture gap - reacquiring")
        if np.max(quality) < cfg.min_quality:
            if self.bad_since is None:
                self.bad_since = timestamp
            self.last_bad_t = timestamp
            if timestamp - self.bad_since > cfg.bad_reset_s:
                self.reset("Poor signal - improve lighting / hold still")
                return False
        else:
            self.bad_since = None
        self.samples.append((timestamp, rgb.copy(), np.clip(quality, 0, 1), bg))
        while self.samples and timestamp - self.samples[0][0] > cfg.history:
            self.samples.popleft()
        elapsed = timestamp - self.samples[0][0]
        self.state.elapsed = elapsed
        self.state.timestamp = timestamp
        clean = timestamp - max(self.samples[0][0], self.last_bad_t)
        self.state.hrv_progress = float(np.clip(clean / (cfg.hrv_seconds + 1), 0, 1))
        if elapsed < cfg.window:
            self.state.status = f"Acquiring pulse: {elapsed:.1f} / {cfg.window:.0f} s"
            return False
        if timestamp - self.last_update < cfg.update - 1e-6:
            return False
        self.last_update = timestamp
        length = min(elapsed, cfg.window_max)
        short = [s for s in self.samples if s[0] >= timestamp - length - 1e-6]
        try:
            t, rgb, q, bg = self._arrays(short)
            a = analyze(t, rgb, q, cfg, self.mode, background=bg, prior_bpm=self.track_bpm)
        except ValueError:
            self.state.status = "Timing unstable - waiting"
            return False
        self.state.analysis = a
        self.state.window_s = a.window_s
        if a.bpm is None:
            self.last_bad_t = timestamp
            self.state.hrv = None
            self.state.status = ("Motion / weak signal - holding" if self.bad_since is not None
                                 else "Weak signal - hold still")
            self._hold(timestamp)
        else:
            self._accept(a, timestamp)
            self._update_hrv(timestamp)
        return True

    # ------------------------------------------------------------------
    def _accept(self, a, timestamp):
        cfg = self.config
        # A clearly stronger peak elsewhere must persist for 3 updates
        # before tracking switches to it (e.g. the real rate changed).
        if (self.track_bpm is not None and a.bpm_raw is not None
                and abs(a.bpm_raw - a.bpm) > 10 and a.peak_ratio > 2):
            self.challenge += 1
        else:
            self.challenge = 0
        if self.challenge >= 3:
            self.challenge = 0
            self.recent.clear()
            self.track_bpm = a.bpm_raw
            self.state.bpm_display = None
            value = a.bpm_raw
        else:
            self.track_bpm = a.bpm
            value = a.bpm
        self.recent.append(value)
        median = float(np.median(self.recent))
        if self.state.bpm_display is None:
            self.state.bpm_display = median
        else:
            self.state.bpm_display += cfg.smooth_alpha * (median - self.state.bpm_display)
        self.state.held = False
        self.last_good_t = timestamp
        self.state.status = "Tracking pulse"

    def _hold(self, timestamp):
        if self.state.bpm_display is not None and timestamp - self.last_good_t <= self.config.hold_s:
            self.state.held = True
        else:
            self.state.bpm_display = None
            self.state.held = False
            self.track_bpm = None
            self.recent.clear()

    def _update_hrv(self, timestamp):
        cfg = self.config
        start = max(self.samples[0][0], self.last_bad_t)
        if timestamp - start < cfg.hrv_seconds + 1 or timestamp - self.last_hrv < 5:
            return
        self.last_hrv = timestamp
        history = [s for s in self.samples if s[0] >= max(timestamp - cfg.hrv_seconds - 2, start)]
        try:
            t, rgb, q, bg = self._arrays(history)
            long = analyze(t, rgb, q, cfg, self.mode, background=bg, prior_bpm=self.track_bpm)
        except ValueError:
            self.state.hrv = None
            return
        q = np.asarray([s[2].max() for s in history])
        self.state.hrv = pulse_intervals(long, cfg.hrv_seconds) if np.min(q) >= .35 else None

    @staticmethod
    def _arrays(samples):
        t, rgb, q, bg = (np.asarray(x) for x in zip(*samples))
        background = bg if np.all(np.isfinite(bg)) else None
        return t, rgb, q, background
