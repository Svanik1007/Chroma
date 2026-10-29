import cv2
import numpy as np
from swcf.app import main


def test_video_eof_exports_last_frame_without_crashing(tmp_path):
    path = tmp_path/"blank.avi"
    writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),30,(320,240))
    assert writer.isOpened()
    for _ in range(10):
        writer.write(np.zeros((240,320,3), np.uint8))
    writer.release()
    from swcf.vision import DEFAULT_MODEL
    if not DEFAULT_MODEL.exists():
        import pytest
        pytest.skip("Download model for EOF integration check")
    out = tmp_path/"out"
    assert main(["video",str(path),"--headless","--output",str(out)]) == 0
    assert (out/"dashboard.png").stat().st_size > 1000
    assert (out/"session.json").exists()
