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


def test_engine_resets_after_face_loss_and_capture_gap():
    e = Engine()
    t, rgb, q = generate(7, corrupted=False)
    for stamp, c, quality in zip(t, rgb, q):
        e.push(stamp, c, quality)
    assert e.state.analysis.bpm is not None
    e.reset("No face")
    assert e.state.analysis is None and not e.samples
    e.push(9, rgb[0], q[0])
    e.push(10, rgb[1], q[1])
    assert len(e.samples) == 1
    assert e.state.analysis is None


def test_sdnn_waits_and_constant_rate_is_near_zero():
    t, rgb, q = generate(63, corrupted=False)
    a = analyze(t, rgb, q, Config())
    result = pulse_intervals(a)
    assert result is not None
    assert result["sdnn_ms"] < 8
    short = analyze(t[:180], rgb[:180], q[:180], Config())
    assert pulse_intervals(short) is None


def test_bad_sample_does_not_retain_stale_result():
    e = Engine()
    e.push(0, np.ones((5,3))*100, np.ones(5))
    e.push(1/30, np.ones((5,3))*100, np.zeros(5))
    assert e.state.analysis is None
    assert not e.samples
