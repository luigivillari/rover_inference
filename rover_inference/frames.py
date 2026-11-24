"""
Camera frames: buffer of the last JPEG frames and the image preprocessing of the training.

preprocess_image is a copy of rover_dataset.preprocess_image (rover_finetune/rover_data): crop 28 px top and
bottom (scaled to the image height, NAVSIM cam_f0.image[28:-28] on 1080 px), bilinear resize to 512 x 256,
RGB in [0, 1], channels first. The JPEG is decoded with OpenCV as in training (cv2.imread = cv2.imdecode),
so the model sees the same pixels on the rover.
"""
from collections import OrderedDict, deque

import cv2
import numpy as np

OUT_W, OUT_H = 512, 256
NAVSIM_CROP, NAVSIM_H = 28, 1080


def decode_jpeg_rgb(data):
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("JPEG could not be decoded")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess_image(rgb):
    """RGB uint8 (H, W, 3) -> float32 (3, 256, 512) in [0, 1] (copy of the training preprocessing)."""
    h = rgb.shape[0]
    c = int(round(NAVSIM_CROP * h / NAVSIM_H))
    img = cv2.resize(rgb[c:h - c], (OUT_W, OUT_H))
    return np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32) / np.float32(255.0)


class FrameBuffer:
    """Last keep_s seconds of frames: (time on the camera time base, stamp in ns, JPEG bytes).
    The time a frame was taken is its time minus the camera latency (17 ms in training)."""

    def __init__(self, keep_s=2.0, latency_s=0.017, cache_size=8):
        self.keep_s, self.latency_s = keep_s, latency_s
        self.frames = deque()
        self._cache = OrderedDict()
        self._cache_size = cache_size

    def add(self, t_base, stamp_ns, jpeg):
        t = t_base - self.latency_s
        if self.frames and t <= self.frames[-1][0]:
            return
        self.frames.append((t, int(stamp_ns), bytes(jpeg)))
        while self.frames and self.frames[0][0] < t - self.keep_s:
            self.frames.popleft()

    def newest(self, not_after=None):
        """Newest frame taken at or before not_after (all frames if None)."""
        for f in reversed(self.frames):
            if not_after is None or f[0] <= not_after:
                return f
        return None

    def nearest(self, t, max_offset=0.05):
        """Frame taken closest to t, if within max_offset seconds (training: 50 ms)."""
        if not self.frames:
            return None
        best = min(self.frames, key=lambda f: abs(f[0] - t))
        return best if abs(best[0] - t) <= max_offset else None

    def by_stamp(self, stamp_ns):
        for f in self.frames:
            if f[1] == stamp_ns:
                return f
        return None

    def image(self, frame):
        """Preprocessed image of a frame (cached: the frame at t is the frame at t - 0.5 s later on)."""
        key = frame[1]
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        img = preprocess_image(decode_jpeg_rgb(frame[2]))
        self._cache[key] = img
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return img
