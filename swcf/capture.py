"""Single-slot live capture: discard old frames if processing falls behind."""
import platform
import threading
import time
import cv2


class Camera:
    def __init__(self, index=0, width=640, height=480, fps=30, backend="auto"):
        apis = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF,
                "v4l2": cv2.CAP_V4L2, "any": cv2.CAP_ANY}
        if backend == "auto":
            preferred = cv2.CAP_DSHOW if platform.system() == "Windows" else (
                cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY)
            candidates = list(dict.fromkeys([preferred, cv2.CAP_ANY]))
        else:
            candidates = [apis[backend]]
        self.cap = None
        for api in candidates:
            cap = cv2.VideoCapture(index, api)
            if cap.isOpened():
                self.cap = cap; break
            cap.release()
        if self.cap is None:
            raise RuntimeError("Cannot open webcam. Close other camera apps; try --camera 1 or --backend msmf.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.buffer_accepted = bool(self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1))
        self.backend = self.cap.getBackendName()
        self.condition = threading.Condition()
        self.latest = None
        self.sequence = 0
        self.stopped = False
        self.error = None
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _capture(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            stamp = time.perf_counter()
            with self.condition:
                if not ok:
                    self.error = "Webcam stopped delivering frames"
                    self.condition.notify_all(); break
                self.sequence += 1
                self.latest = (self.sequence, stamp, frame)
                self.condition.notify_all()

    def read(self, after=0, timeout=3):
        with self.condition:
            available = self.condition.wait_for(
                lambda: self.sequence > after or self.error or self.stopped, timeout)
            if self.error:
                raise RuntimeError(self.error)
            if not available or self.stopped:
                raise RuntimeError("Timed out waiting for webcam")
            return self.latest

    def close(self):
        self.stopped = True
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=2)
        self.cap.release()
