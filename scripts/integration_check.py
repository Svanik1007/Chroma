"""Optional test using Google's public portrait, never a physiological recording.

Downloads a test image, overlays polygons, and creates a color-modulated still
video. Run main.py video on it to exercise the complete video path.
The fixture is excluded from the submission ZIP.
"""
from pathlib import Path
import sys
import urllib.request
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swcf.vision import FaceTracker


def main():
    folder = Path("data/integration")
    folder.mkdir(parents=True, exist_ok=True)
    portrait = folder/"portrait.jpg"
    if not portrait.exists():
        urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-assets/portrait.jpg",portrait)
    frame = cv2.imread(str(portrait))
    frame = cv2.resize(frame, (int(frame.shape[1]*480/frame.shape[0]),480))
    tracker = FaceTracker()
    try:
        data = tracker.extract(frame, 0)
        assert data is not None, "Public portrait face was not detected"
        rgb,q,polygons = data
        assert np.min(q) > .5, f"Bad polygon geometry/quality: {q}"
        overlay = frame.copy()
        for i,poly in enumerate(polygons):
            cv2.polylines(overlay,[poly],True,(60,230,220),1)
            cv2.putText(overlay,str(i+1),tuple(poly.mean(axis=0).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1)
        cv2.imwrite(str(folder/"regions.png"),overlay)
        writer = cv2.VideoWriter(str(folder/"modulated_still.avi"),cv2.VideoWriter_fourcc(*"FFV1"),30,(frame.shape[1],480))
        if not writer.isOpened():
            raise RuntimeError("FFV1 video writer unavailable")
        for i in range(300):
            # Artificial pulse added to EVERY pixel, including background.
            # This validates wiring/timing, not actual remote physiology.
            modulation = np.sin(2*np.pi*1.2*i/30)*np.array([.18,.65,.10])
            writer.write(np.clip(np.rint(frame.astype(float)+modulation),0,255).astype(np.uint8))
        writer.release()
        print("Face polygons and artificial 72 BPM still-image video created in",folder)
        print("This is synthetic integration evidence, not a webcam accuracy test.")
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
