"""
Online clock alignment, the same rule as extract_samples.py --time-source aligned (training data).

Camera stamps are on the Jetson clock; Vicon header stamps are on the clock of the lab PC (15-20 s
apart, no NTP). The extraction maps a Vicon stamp onto the camera time base with

    aligned = vicon_stamp + env(vicon_receive - vicon_stamp) - env(image_receive - image_stamp)

where env() is the lower envelope (1st percentile) of the delays: clock offset + minimum network delay,
minus the minimum delay of an image from its stamp to the reader on the same computer. Offline it is fitted
over the whole bag (offset + drift); online it is the 1st percentile of the last `window_s` seconds,
which follows the same offset and its slow drift.
"""
from collections import deque

import numpy as np


class _Envelope:
    """Low percentile of (receive - stamp) over the last window_s seconds of receive time."""

    def __init__(self, window_s=30.0, pct=1.0, min_samples=20, refresh_s=1.0):
        self.window_s, self.pct, self.min_samples, self.refresh_s = window_s, pct, min_samples, refresh_s
        self.buf = deque()
        self.value, self._computed_at = None, -1e18

    def add(self, stamp, recv):
        self.buf.append((recv, recv - stamp))
        while self.buf and self.buf[0][0] < recv - self.window_s:
            self.buf.popleft()
        if len(self.buf) >= self.min_samples and (self.value is None or recv - self._computed_at >= self.refresh_s):
            self.value = float(np.percentile([d for _, d in self.buf], self.pct))
            self._computed_at = recv


class ClockAligner:
    """mode 'aligned': Vicon stamps mapped onto the camera time base (as in training).
    mode 'receive': Vicon receive times and image receive times (fallback, as --time-source log)."""

    def __init__(self, mode="aligned", window_s=30.0, pct=1.0):
        if mode not in ("aligned", "receive"):
            raise ValueError(f"unknown time mode {mode}")
        self.mode = mode
        self.vicon = _Envelope(window_s, pct)
        self.image = _Envelope(window_s, pct, min_samples=5)

    def add_vicon(self, stamp, recv):
        self.vicon.add(stamp, recv)

    def add_image(self, stamp, recv):
        self.image.add(stamp, recv)

    @property
    def ready(self):
        return self.mode == "receive" or (self.vicon.value is not None and self.image.value is not None)

    def vicon_time(self, stamp, recv):
        """Time of a Vicon pose on the camera time base, or None while the offset is not known yet."""
        if self.mode == "receive":
            return recv
        if not self.ready:
            return None
        return stamp + self.vicon.value - self.image.value

    def vicon_key(self, stamp, recv):
        """Time under which a pose is STORED: its own stamp (aligned) or receive time. The offset is applied
        at query time to all stored poses at once (offset()), so an update of the offset estimate never
        puts a step inside the buffer (a step would show up as noise in velocity / acceleration)."""
        return recv if self.mode == "receive" else stamp

    def offset(self):
        """camera time = stored pose time + offset(); None while not known."""
        if self.mode == "receive":
            return 0.0
        return None if not self.ready else self.vicon.value - self.image.value

    def image_time(self, stamp, recv):
        """Time base of the images: their stamp (aligned) or receive time."""
        return recv if self.mode == "receive" else stamp

    def report(self):
        return {"mode": self.mode,
                "vicon_offset_ms": None if self.vicon.value is None else round(self.vicon.value * 1000, 2),
                "image_delay_ms": None if self.image.value is None else round(self.image.value * 1000, 2)}
