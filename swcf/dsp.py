"""Timestamp-aware POS, quality-weighted fusion, and spectral/interval estimates.

All color arrays are RGB (not OpenCV BGR). No claimed clinical accuracy.
"""
from dataclasses import dataclass
import numpy as np
from scipy import signal
from scipy.fft import next_fast_len

EPS = 1e-12


@dataclass
class Analysis:
    times: np.ndarray
    pulse: np.ndarray
    frequencies: np.ndarray
    power: np.ndarray
    weights: np.ndarray
    patch_sqi: np.ndarray
    bpm: float | None
    concentration: float
    quality: float
    fs: float


def resample(times, rgb, quality):
    times = np.asarray(times, float)
    rgb, quality = np.asarray(rgb, float), np.asarray(quality, float)
    if len(times) < 3 or np.any(np.diff(times) <= 0):
        raise ValueError("At least 3 strictly increasing timestamps required")
    if rgb.shape != (len(times), 5, 3) or quality.shape != (len(times), 5):
        raise ValueError("Expected RGB (N,5,3) and quality (N,5)")
    if not all(np.all(np.isfinite(a)) for a in (times, rgb, quality)):
        raise ValueError("Inputs must be finite")
    fs = (len(times) - 1) / (times[-1] - times[0])
    grid = np.linspace(times[0], times[-1], len(times))
    colors = np.stack([np.interp(grid, times, rgb[:, p, c])
                       for p in range(5) for c in range(3)], axis=1).reshape(-1, 5, 3)
    q = np.stack([np.interp(grid, times, quality[:, p]) for p in range(5)], axis=1)
    return grid, colors, np.clip(q, 0, 1), fs


def pos(rgb, fs):
    """Wang et al. sliding 1.6 s POS; overlap-add and count normalization."""
    n = len(rgb)
    width = min(n, max(8, round(1.6 * fs)))
    output, counts = np.zeros(n), np.zeros(n)
    windows = np.lib.stride_tricks.sliding_window_view(np.asarray(rgb, float), width, axis=0)
    c = windows / np.maximum(windows.mean(axis=2, keepdims=True), EPS) - 1
    s1 = c[:, 1] - c[:, 2]
    s2 = c[:, 1] + c[:, 2] - 2*c[:, 0]
    alpha = np.std(s1, axis=1) / np.maximum(np.std(s2, axis=1), EPS)
    h = s1 + alpha[:, None]*s2
    h -= h.mean(axis=1, keepdims=True)
    for offset in range(width):
        output[offset:offset+len(h)] += h[:, offset]
        counts[offset:offset+len(h)] += 1
    return output / np.maximum(counts, 1)


def bandpass(x, fs, low=0.75, high=2.5):
    if fs <= 2 * high:
        raise ValueError("Sampling rate too low for requested passband")
    # N=4 is the low-pass prototype order; bandpass has 8 poles and
    # forward/backward filtering squares its magnitude response.
    sos = signal.butter(4, [low, high], fs=fs, btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, signal.detrend(x, axis=0), axis=0)


def spectrum(x, fs, low=.75, high=2.5):
    f, p = signal.welch(x, fs=fs, window="hann", nperseg=min(len(x), round(6 * fs)),
                        nfft=next_fast_len(max(2048, len(x))), detrend="constant")
    keep = (f >= low) & (f <= high)
    candidates = np.flatnonzero(keep)
    if not len(candidates) or np.std(x) < 1e-10:
        return f, p, None, 0.0
    k = candidates[np.argmax(p[keep])]
    offset = 0.0
    if 0 < k < len(p) - 1:
        y = np.log(p[k-1:k+2] + EPS)
        denom = y[0] - 2 * y[1] + y[2]
        if abs(denom) > EPS:
            offset = float(np.clip(.5 * (y[0] - y[2]) / denom, -.5, .5))
    peak_hz = np.clip(f[k] + offset * (f[1] - f[0]), low, high)
    near = keep & (np.abs(f - peak_hz) <= .18)
    concentration = float(p[near].sum() / max(p[keep].sum(), EPS))
    return f, p, float(60 * peak_hz), concentration


def analyze(times, rgb, quality, config, mode="weighted"):
    t, c, q, fs = resample(times, rgb, quality)
    if fs < 8 or np.max(np.diff(times)) > config.max_gap:
        raise ValueError("Insufficient frame rate or a capture gap")
    waves = np.column_stack([bandpass(pos(c[:, p], fs), fs, config.low, config.high)
                             for p in range(5)])
    # Trim edges for SQI/PSD; keep displayed waveform for transparency.
    trim = max(1, round(.5 * fs))
    core = waves[trim:-trim]
    concentrations = np.array([spectrum(core[:, p], fs, config.low, config.high)[3]
                               for p in range(5)])
    amplitude = np.std(core, axis=0)
    valid = amplitude > 1e-8
    patch_sqi = np.mean(q, axis=0) * concentrations * valid
    if mode == "equal":
        scores = valid.astype(float)
    elif mode == "green":
        waves = bandpass(c[:, :, 1] / np.maximum(c[:, :, 1].mean(axis=0), EPS) - 1,
                         fs, config.low, config.high)
        amplitude = np.std(waves[trim:-trim], axis=0)
        scores = (amplitude > 1e-8).astype(float)
    elif mode == "weighted":
        scores = patch_sqi ** 2
    else:
        raise ValueError("Unknown fusion mode")
    normalized = waves / np.maximum(amplitude, EPS)
    # Local per-frame gates suppress newly moving/occluded patches immediately.
    gates = q if mode == "weighted" else np.ones_like(q)
    weights_over_time = gates * scores
    sums = weights_over_time.sum(axis=1, keepdims=True)
    weights_over_time = np.divide(weights_over_time, sums, out=np.zeros_like(gates), where=sums > EPS)
    fused = (normalized * weights_over_time).sum(axis=1)
    fused = bandpass(fused, fs, config.low, config.high)
    weights = weights_over_time.mean(axis=0)
    overall_q = float(np.sum(weights * np.mean(q, axis=0)))
    f, power, bpm, concentration = spectrum(fused[trim:-trim], fs, config.low, config.high)
    if overall_q < config.min_quality or concentration < config.min_concentration:
        bpm = None
    return Analysis(t, fused, f, power, weights, patch_sqi, bpm, concentration, overall_q, fs)


def pulse_intervals(analysis, min_seconds=60.0):
    """Experimental PRV; never diagnoses stress or identifies ECG normal beats."""
    t, x, fs = analysis.times, analysis.pulse, analysis.fs
    if t[-1] - t[0] < min_seconds + 1 or analysis.bpm is None:
        return None
    edge = round(.5 * fs)
    peaks, _ = signal.find_peaks(x, distance=max(1, int(fs * .40)),
                                 prominence=max(.35 * np.std(x), EPS))
    peaks = peaks[(peaks >= edge) & (peaks < len(x) - edge)]
    # Parabolic sub-frame refinement reduces quantization but adds no new data.
    refined = []
    for k in peaks:
        denom = x[k-1] - 2*x[k] + x[k+1]
        shift = .5*(x[k-1]-x[k+1])/denom if abs(denom) > EPS else 0
        refined.append(t[k] + np.clip(shift, -.5, .5)/fs)
    beat_t = np.asarray(refined)
    if len(beat_t) < 20 or beat_t[-1] - beat_t[0] < min_seconds - 3:
        return None
    intervals = np.diff(beat_t) * 1000
    median = np.median(intervals)
    valid = ((intervals >= 400) & (intervals <= 1334) &
             (np.abs(intervals - median) <= .20 * median))
    if valid.mean() < .90 or abs(60000 / median - analysis.bpm) > 12:
        return None
    kept = intervals[valid]
    return {"sdnn_ms": float(np.std(kept, ddof=1)),
            "intervals_ms": kept.tolist(), "accepted_fraction": float(valid.mean()),
            "duration_s": float(beat_t[-1] - beat_t[0]), "n_intervals": len(kept)}
