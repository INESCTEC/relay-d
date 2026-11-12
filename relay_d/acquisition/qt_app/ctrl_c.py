"""Ctrl+C in the terminal closes the window, so its cleanup runs (stop a recording, stop the timers)."""

import signal

from PyQt5.QtCore import QTimer


def close_on_ctrl_c(window):
    """SIGINT / SIGTERM close ``window``. Python runs the handler from the next timer callback (the ROS spin timer)."""

    def handler(_signum, _frame):
        QTimer.singleShot(0, window.close)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
