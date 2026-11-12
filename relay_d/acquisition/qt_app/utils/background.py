import threading

from PyQt5.QtCore import QTimer


def run_in_background(parent, work, on_done, busy=lambda delta: None):
    """Run ``work()`` in a thread and call ``on_done(value, error)`` from the GUI thread when it ends.

    Importing torch, reading a robot description or waiting for a process takes seconds; on the GUI thread that would
    also stop the ROS spin timer (a running recording would hold and repeat its last samples).
    ``busy(+1)`` / ``busy(-1)`` is called when it starts / ends, so a page can disable its controls meanwhile.
    """
    outcome = {}

    def target():
        try:
            outcome["value"] = work()
        except Exception as e:  # reported to on_done, not lost with the thread
            outcome["error"] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    busy(+1)
    timer = QTimer(parent)
    timer.setInterval(100)

    def poll():
        if thread.is_alive():
            return
        timer.stop()
        timer.deleteLater()
        busy(-1)
        on_done(outcome.get("value"), outcome.get("error"))

    timer.timeout.connect(poll)
    timer.start()
