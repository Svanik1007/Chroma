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
        self.lock_request = False
        self.lock_result = None
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def request_lock(self):
        """Ask the capture thread to freeze auto exposure / white balance.

        Auto exposure and white balance make the whole image brighten and
        change colour while the webcam "hunts", which looks like a slow pulse.
        Done in the capture thread so cap.set never races cap.read.
        """
        self.lock_request = True

    def _apply_lock(self):
        cap, name = self.cap, self.cap.getBackendName().upper()
        exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
        # Manual-mode value differs per backend (DirectShow 0.25, V4L2 1).
        manual = 1 if name == "V4L2" else 0.25
        exp_ok = bool(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual))
        if exp_ok and exposure:
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        wb_ok = bool(cap.set(cv2.CAP_PROP_AUTO_WB, 0))
        self.lock_result = {"auto_exposure_off": exp_ok, "auto_white_balance_off": wb_ok}

    def _capture(self):
        while not self.stopped:
            if self.lock_request:
                self.lock_request = False
                try:
                    self._apply_lock()
                except cv2.error:
                    self.lock_result = {"auto_exposure_off": False, "auto_white_balance_off": False}
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
