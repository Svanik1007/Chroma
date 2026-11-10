import numpy as np
from swcf.vision import PATCHES, extract_patches


def test_quality_suppresses_dark_saturated_and_moving_patches():
    rng = np.random.default_rng(2)
    lm = rng.uniform(.3,.7, (478,2))
    lm[234], lm[454] = [.2,.5], [.8,.5]
    frame = np.full((480,640,3), [100,130,170], np.uint8)
    rgb, q, polygons, bg = extract_patches(frame, lm)
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


def test_landmark_filter_reduces_jitter_but_follows_movement():
    from swcf.vision import LandmarkFilter
    rng = np.random.default_rng(0)
    f = LandmarkFilter()
    still = np.full((478, 2), .5)
    out = [f(still + rng.normal(0, .002, still.shape), i/30) for i in range(60)]
    assert np.std(np.asarray(out)[30:, :, 0]) < .001       # jitter of .002 strongly reduced
    for i in range(60, 90):                                 # face moves 0.1 in 1 s
        moved = f(still + (i-59)/300, i/30)
    assert abs(moved[0, 0] - (.5 + 30/300)) < .02


def test_small_jitter_is_not_treated_as_motion():
    rng = np.random.default_rng(2)
    lm = rng.uniform(.3,.7, (478,2))
    lm[234], lm[454] = [.2,.5], [.8,.5]
    frame = np.full((480,640,3), [100,130,170], np.uint8)
    jitter = lm + rng.normal(0, .0003, lm.shape)            # ~0.2 px per frame
    q = extract_patches(frame, jitter, lm, 1/30)[1]
    assert q.min() > .9


def test_background_patch_is_placed_beside_a_small_face():
    from swcf.vision import background_patch
    lm = np.random.default_rng(3).uniform(.4, .6, (478, 2))
    frame = np.full((480, 640, 3), 90, np.uint8)
    rgb, box = background_patch(frame, lm)
    x0, y0, x1, y1 = box
    assert x1 <= .4*640 or x0 >= .6*640                     # outside the face
    np.testing.assert_allclose(rgb, 90)
    assert background_patch(frame, np.random.default_rng(3).uniform(.05, .95, (478, 2))) is None
