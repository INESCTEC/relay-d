"""Drops TF transforms that tf2 would reject as TF_OLD_DATA (no ROS imports, so it can be tested without ROS).

tf2 ignores a transform that is older than ``cache_time`` behind the newest one of the same frame, and prints a warning for every
one (a bag replayed on the live /tf, another machine with a wrong clock, ...). Dropping them before ``set_transform`` gives the
same buffer content without the spam; one summary line names the frames instead.
"""

import time
from typing import Callable, Dict

from relay_d.utils.coloring_logger import logger


class StaleTfFilter:
    def __init__(self, cache_time: float = 10.0, warn_every: float = 10.0, clock: Callable[[], float] = time.monotonic):
        self._cache_ns = int(cache_time * 1e9)
        self._warn_every, self._clock = warn_every, clock
        self._newest: Dict[str, int] = {}
        self._stale: Dict[str, float] = {}  # frame -> largest gap in seconds since the last summary
        self._last_warn = clock()

    def accept(self, child_frame: str, stamp_ns: int) -> bool:
        """False if tf2 would ignore this transform (older than ``cache_time`` behind the newest of its frame)."""
        newest = self._newest.get(child_frame, 0)
        if stamp_ns < newest - self._cache_ns:
            self._stale[child_frame] = max(self._stale.get(child_frame, 0.0), (newest - stamp_ns) / 1e9)
            self._summarize()
            return False
        if stamp_ns > newest:
            self._newest[child_frame] = stamp_ns
        return True

    def _summarize(self):
        if self._clock() - self._last_warn < self._warn_every:
            return
        self._last_warn = self._clock()
        frames = ", ".join(f"{f} ({gap:.0f} s old)" for f, gap in sorted(self._stale.items())[:8])
        logger.warning(
            f"TF: ignored transforms with old stamps for {len(self._stale)} frame(s): {frames}. Another publisher on this TF topic "
            "(a bag played without a remap, a machine with a different clock) is sending stamps from the past."
        )
        self._stale.clear()
