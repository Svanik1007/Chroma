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


class LandmarkFilter:
    """One-Euro filter (Casiez et al., 2012) applied to every landmark coordinate.

    Slow movement is smoothed strongly (removes tracker jitter); fast movement
    raises the cutoff so the regions still follow the head without lag.
    """

    def __init__(self, min_cutoff=1.5, beta=8.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x = self.dx = self.t = None

    def __call__(self, x, t):
        x = np.asarray(x, float)
        if self.x is None or self.x.shape != x.shape or t <= self.t:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        speed = np.linalg.norm(self.dx, axis=-1, keepdims=True)
        a = self._alpha(self.min_cutoff + self.beta * speed, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()


class FaceTracker:
    def __init__(self, model=DEFAULT_MODEL, smooth=True):
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
        self.filter = LandmarkFilter() if smooth else None
        self.previous = None
        self.previous_t = None
        self.last_ms = -1
        self.bg_rect = None

    def extract(self, frame, timestamp):
        """Returns (rgb (5,3), quality (5,), polygons, background_rgb or None), or None."""
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        millis = max(self.last_ms + 1, int(timestamp * 1000))
        self.last_ms = millis
        result = self.detector.detect_for_video(image, millis)
        if not result.face_landmarks:
            self.previous = self.previous_t = None
            self.bg_rect = None
            # Keep the filter state: a brief dropout should not make regions jump.
            return None
        landmarks = np.array([(v.x, v.y) for v in result.face_landmarks[0]])
        if self.filter is not None:
            landmarks = self.filter(landmarks, timestamp)
        rgb, q, polygons, bg = extract_patches(
            frame, landmarks, self.previous,
            timestamp - self.previous_t if self.previous_t is not None else 1/30)
        self.previous, self.previous_t = landmarks, timestamp
        self.bg_rect = bg[1] if bg is not None else None
        return rgb, q, polygons, (bg[0] if bg is not None else None)

    def close(self):
        self.detector.close()


MOTION_FLOOR = 0.08   # face-widths/s treated as tracker jitter, not movement
MOTION_GAIN = 1.5
LUMA = np.array([0.299, 0.587, 0.114])


def trimmed_mean(pixels, lo=5, hi=95):
    """Mean RGB of pixels whose luminance lies between the lo and hi percentiles.

    Drops specular highlights, shadows, eyebrow/hair pixels and edge pixels that
    slip into a patch when it shifts, which otherwise cause step changes.
    """
    luma = pixels @ LUMA
    a, b = np.percentile(luma, [lo, hi])
    keep = (luma >= a) & (luma <= b)
    if keep.sum() < 10:
        return pixels.mean(axis=0)
    return pixels[keep].mean(axis=0)


def background_patch(frame, landmarks):
    """A non-skin reference area beside the head at forehead height.

    Returns (mean RGB, (x0, y0, x1, y1)) or None when neither side is in frame.
    """
    h, w = frame.shape[:2]
    pts = landmarks * [w, h]
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    face_w = max(x_max - x_min, 1)
    cy = pts[151, 1] if len(pts) > 151 else pts[:, 1].mean()
    half = 0.12 * face_w
    candidates = []
    for side in (-1, 1):
        if side < 0:
            x1, x0 = x_min - 0.15 * face_w, x_min - 0.45 * face_w
        else:
            x0, x1 = x_max + 0.15 * face_w, x_max + 0.45 * face_w
        box = tuple(int(round(v)) for v in (x0, cy - half, x1, cy + half))
        if box[0] >= 0 and box[1] >= 0 and box[2] <= w and box[3] <= h and box[2] - box[0] >= 6:
            margin = min(box[0], w - box[2])
            candidates.append((margin, box))
    if not candidates:
        return None
    x0, y0, x1, y1 = max(candidates)[1]
    pixels = frame[y0:y1, x0:x1].reshape(-1, 3)[:, ::-1].astype(float)
    if len(pixels) < 25:
        return None
    return pixels.mean(axis=0), (x0, y0, x1, y1)


def extract_patches(frame, landmarks, previous=None, dt=1/30, background=True):
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
        pixels = frame[y0:y1, x0:x1][mask > 0][:, ::-1].astype(float)  # BGR -> RGB
        if len(pixels) < 25:
            colors.append([0, 0, 0]); qualities.append(0.0); continue
        mean = trimmed_mean(pixels)
        brightness = float(np.mean(pixels))
        clipping = float(np.mean(np.any((pixels <= 8) | (pixels >= 247), axis=1)))
        light = np.clip((brightness - 20)/40, 0, 1) * np.clip((240-brightness)/30, 0, 1)
        motion = 0.0
        if previous is not None:
            motion = np.mean(np.linalg.norm((landmarks[indices] - previous[indices])*scale, axis=1))
            motion /= face_scale * max(dt, 1e-3)
        # Motion in face-widths per second; small residual jitter is ignored.
        motion_score = np.exp(-MOTION_GAIN * max(0.0, motion - MOTION_FLOOR))
        quality = light * (1 - clipping)**2 * motion_score * visible
        colors.append(mean)
        qualities.append(float(quality))
    bg = background_patch(frame, landmarks) if background else None
    return np.asarray(colors), np.asarray(qualities), polygons, bg
