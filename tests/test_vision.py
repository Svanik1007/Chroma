import numpy as np
from swcf.vision import PATCHES, extract_patches


def test_quality_suppresses_dark_saturated_and_moving_patches():
    rng = np.random.default_rng(2)
    lm = rng.uniform(.3,.7, (478,2))
    lm[234], lm[454] = [.2,.5], [.8,.5]
    frame = np.full((480,640,3), [100,130,170], np.uint8)
    rgb, q, polygons = extract_patches(frame, lm)
    assert rgb.shape == (5,3) and q.min() > .9
    np.testing.assert_allclose(rgb, np.tile([170,130,100], (5,1)))
    assert len(polygons) == 5
    assert extract_patches(np.zeros_like(frame), lm)[1].max() == 0
    assert extract_patches(np.full_like(frame,255), lm)[1].max() == 0
    moved = extract_patches(frame, lm, lm-.1, 1/30)[1]
    assert moved.max() < .01


def test_model_can_initialize_and_process_blank_image():
    from swcf.vision import DEFAULT_MODEL, FaceTracker
    import pytest
    if not DEFAULT_MODEL.exists():
        pytest.skip("Download model to run integration check")
    tracker = FaceTracker()
    try:
        assert tracker.extract(np.zeros((480,640,3), np.uint8), 0) is None
        assert tracker.extract(np.zeros((480,640,3), np.uint8), 1/30) is None
    finally:
        tracker.close()
