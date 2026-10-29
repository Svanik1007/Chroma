"""MediaPipe Tasks face tracker and compact, interior facial polygons."""
from pathlib import Path
import hashlib
import os
import urllib.request
import numpy as np
import cv2

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
# Only stable indices shared with the original 468-point mesh are used.
PATCHES = (
    (109, 10, 338, 337, 151, 108),
    (50, 101, 205, 187, 123),
    (280, 330, 425, 411, 352),
    (196, 3, 51, 45, 4, 275, 281, 248, 419),
    (201, 200, 421, 379, 152, 149),
)


def download_model(path=DEFAULT_MODEL):
    path = Path(path)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".download")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as source, temp.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        if temp.stat().st_size < 1_000_000:
            raise RuntimeError("Model download incomplete")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path


class FaceTracker:
    def __init__(self, model=DEFAULT_MODEL):
        import mediapipe as mp
        self.mp = mp
        if not Path(model).exists():
            raise FileNotFoundError("Face model missing. Run: python main.py download-model")
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(Path(model).resolve())),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1, min_face_detection_confidence=.5,
            min_face_presence_confidence=.5, min_tracking_confidence=.5)
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.previous = None
        self.previous_t = None
        self.last_ms = -1

    def extract(self, frame, timestamp):
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        millis = max(self.last_ms + 1, int(timestamp * 1000))
        self.last_ms = millis
        result = self.detector.detect_for_video(image, millis)
        if not result.face_landmarks:
            self.previous = self.previous_t = None
            return None
        landmarks = np.array([(v.x, v.y) for v in result.face_landmarks[0]])
        extracted = extract_patches(frame, landmarks, self.previous,
                                     timestamp - self.previous_t if self.previous_t is not None else 1/30)
        self.previous, self.previous_t = landmarks, timestamp
        return extracted

    def close(self):
        self.detector.close()


def extract_patches(frame, landmarks, previous=None, dt=1/30):
    h, w = frame.shape[:2]
    scale = np.array([w, h])
    face_scale = max(np.linalg.norm((landmarks[234] - landmarks[454]) * scale), 1)
    colors, qualities, polygons = [], [], []
    for indices in PATCHES:
        indices = list(indices)
        points = landmarks[indices] * scale
        # Convex hull and modest shrink keep patches away from eyes/lips/contours.
        center = points.mean(axis=0)
        points = center + .80 * (points - center)
        visible = bool(np.all((points[:, 0] >= 0) & (points[:, 0] < w) &
                             (points[:, 1] >= 0) & (points[:, 1] < h)))
        polygon = cv2.convexHull(np.rint(points).astype(np.int32)).reshape(-1, 2)
        polygons.append(polygon)
        x, y, pw, ph = cv2.boundingRect(polygon)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x+pw), min(h, y+ph)
        if x1 <= x0 or y1 <= y0:
            colors.append([0, 0, 0]); qualities.append(0.0); continue
        mask = np.zeros((y1-y0, x1-x0), np.uint8)
        cv2.fillConvexPoly(mask, polygon - [x0, y0], 255)
        pixels = frame[y0:y1, x0:x1][mask > 0][:, ::-1]  # BGR -> RGB
        if len(pixels) < 25:
            colors.append([0, 0, 0]); qualities.append(0.0); continue
        mean = pixels.mean(axis=0)
        brightness = float(np.mean(pixels))
        clipping = float(np.mean(np.any((pixels <= 8) | (pixels >= 247), axis=1)))
        light = np.clip((brightness - 20)/40, 0, 1) * np.clip((240-brightness)/30, 0, 1)
        motion = 0.0
        if previous is not None:
            motion = np.mean(np.linalg.norm((landmarks[indices] - previous[indices])*scale, axis=1))
            motion /= face_scale * max(dt, 1e-3)
        # Motion score uses face-widths per second, not pixels per frame.
        quality = light * (1 - clipping)**2 * np.exp(-3 * motion) * visible
        colors.append(mean)
        qualities.append(float(quality))
    return np.asarray(colors), np.asarray(qualities), polygons
