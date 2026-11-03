"""Ctrl+C in the terminal closes the main window (so its cleanup runs)."""

import os
import signal
import time

import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication, QMainWindow  # noqa: E402

from relay_d.acquisition.qt_app.ctrl_c import close_on_ctrl_c  # noqa: E402


class _Window(QMainWindow):
    closed = False

    def closeEvent(self, event):  # the app's closeEvent is where recording and timers are stopped
        self.closed = True
        event.accept()


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_signal_closes_the_window(qapp, sig):
    old = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    window = _Window()
    window.show()
    try:
        close_on_ctrl_c(window)
        os.kill(os.getpid(), sig)
        deadline = time.time() + 5
        while not window.closed and time.time() < deadline:  # Python runs the handler between Qt events
            QApplication.processEvents()
            time.sleep(0.01)
        assert window.closed
    finally:
        for s, handler in old.items():
            signal.signal(s, handler)
