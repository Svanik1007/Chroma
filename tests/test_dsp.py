import numpy as np
import pytest
from swcf.config import Config
from swcf.dsp import analyze, bandpass, pos, pulse_intervals, resample
from swcf.engine import Engine
from swcf.synthetic import generate


@pytest.mark.parametrize("bpm", [48, 60, 72, 90, 120, 144])
def test_known_pulse_recovery(bpm):
    t, rgb, q = generate(10, bpm=bpm, corrupted=False, jitter=True)
    a = analyze(t, rgb, q, Config())
    assert a.bpm is not None
    assert abs(a.bpm-bpm) < 3
    assert abs(a.weights.sum()-1) < 1e-6


def test_corrupted_patches_are_downweighted():
    t, rgb, q = generate(10, corrupted=True)
    a = analyze(t, rgb, q, Config())
    assert abs(a.bpm-72) < 3
    assert a.weights[3:].sum() < .03


def test_no_pulse_or_bad_lighting_does_not_make_bpm():
    t, rgb, q = generate(8, corrupted=False)
    a = analyze(t, np.full_like(rgb, 100), q, Config())
    assert a.bpm is None
    assert analyze(t, rgb, q*0, Config()).bpm is None


def test_gap_and_order_validation():
    t, rgb, q = generate(8, corrupted=False)
    t[40:] += 1
    with pytest.raises(ValueError, match="gap"):
        analyze(t, rgb, q, Config())
    t[1] = t[0]
    with pytest.raises(ValueError, match="increasing"):
        resample(t, rgb, q)


def test_filter_rejects_out_of_band():
    fs = 30
    t = np.arange(1800)/fs
    wanted = np.sin(2*np.pi*1.2*t)
    x = wanted + np.sin(2*np.pi*.15*t) + np.sin(2*np.pi*5*t)
    y = bandpass(x, fs)
    assert np.corrcoef(y[90:-90], wanted[90:-90])[0,1] > .995


def run(engine, t, rgb, q, bg=None, lost=None):
    """Feed samples; `lost` = (start, end) seconds during which the face is missing."""
    shown = []
    for i, stamp in enumerate(t):
        if lost and lost[0] <= stamp < lost[1]:
            engine.face_lost(stamp)
        else:
            engine.push(stamp, rgb[i], q[i], None if bg is None else bg[i])
        shown.append((stamp, engine.state.bpm_display, engine.state.held))
    return shown


def test_engine_resets_after_long_face_loss_and_real_capture_gap():
    e = Engine()
    t, rgb, q = generate(7, corrupted=False)
    run(e, t, rgb, q)
    assert e.state.analysis.bpm is not None
    for stamp in np.arange(7.1, 9.0, 1/30):      # 1.9 s without a face
        e.face_lost(stamp)
    assert e.state.analysis is None and not e.samples
    e.push(9, rgb[0], q[0])
    e.push(10.5, rgb[1], q[1])                   # 1.5 s capture gap
    assert len(e.samples) == 1
    assert e.state.analysis is None


def test_short_face_loss_keeps_tracking():
    t, rgb, q = generate(16, corrupted=False)
    shown = run(Engine(), t, rgb, q, lost=(9.0, 9.6))
    during = [v for stamp, v, _ in shown if 9.0 <= stamp < 9.6]
    assert all(v is not None for v in during)
    assert abs(shown[-1][1] - 72) < 3


def test_motion_burst_does_not_reset_or_blank():
    t, rgb, q = generate(20, corrupted=False, jitter=True, motion_bursts=[(8, .5), (14, .7)])
    e = Engine()
    shown = run(e, t, rgb, q)
    after_start = [v for stamp, v, _ in shown if stamp >= 6.6]
    assert all(v is not None for v in after_start), "display went blank during motion"
    assert e.state.elapsed > 19                  # buffer was never wiped
    assert max(abs(v - 72) for v in after_start) < 3


def test_sustained_bad_quality_resets_after_timeout():
    t, rgb, q = generate(12, corrupted=False)
    q[t >= 8] = .05
    e = Engine()
    run(e, t, rgb, q)
    assert e.state.bpm_display is None
    assert e.state.elapsed < 4                   # reset after ~2 s of bad data


def test_display_is_steady_with_noisy_input():
    t, rgb, q = generate(30, corrupted=True, jitter=True, noise=.25, seed=11)
    shown = run(Engine(), t, rgb, q)
    values = np.array([v for stamp, v, _ in shown if stamp >= 10 and v is not None])
    assert len(values) > 500
    assert np.std(values) < 2 and abs(np.mean(values) - 72) < 3


def test_background_reference_removes_common_flicker():
    t, rgb, q, bg = generate(14, corrupted=False, jitter=True, flicker_hz=1.5,
                             flicker_amp=.01, return_background=True)
    without = analyze(t[-360:], rgb[-360:], q[-360:], Config())
    with_bg = analyze(t[-360:], rgb[-360:], q[-360:], Config(), background=bg[-360:])
    assert abs(without.bpm_raw - 90) < 3          # flicker (1.5 Hz) wins without a reference
    assert with_bg.background_used
    assert abs(with_bg.bpm - 72) < 3


def test_rejected_window_holds_then_blanks():
    t, rgb, q = generate(36, corrupted=False)
    flat = rgb.copy()
    flat[t >= 12] = rgb[t >= 12].mean(axis=0)    # pulse disappears, quality stays fine
    shown = run(Engine(), t, flat, q)
    held = [h for stamp, v, h in shown if stamp >= 12]
    assert any(held)
    assert shown[-1][1] is None


def test_sdnn_waits_and_constant_rate_is_near_zero():
    t, rgb, q = generate(63, corrupted=False)
    a = analyze(t, rgb, q, Config())
    result = pulse_intervals(a)
    assert result is not None
    assert result["sdnn_ms"] < 8
    short = analyze(t[:180], rgb[:180], q[:180], Config())
    assert pulse_intervals(short) is None


def test_bad_sample_is_skipped_not_reset():
    e = Engine()
    e.push(0, np.ones((5,3))*100, np.ones(5))
    e.push(1/30, np.full((5,3), np.nan), np.ones(5))
    assert len(e.samples) == 1
    e.push(2/30, np.ones((5,3))*100, np.zeros(5))
    assert e.state.analysis is None and len(e.samples) == 2
