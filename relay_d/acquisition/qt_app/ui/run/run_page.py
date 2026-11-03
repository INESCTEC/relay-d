"""Run tab: start a trained policy on the robot (``lfd-inference-node``) and stop it safely.

The node runs as a child process; its output is shown in the log view. Stop sends SIGINT, the node's
Ctrl+C path, which sends the safe stop and closes.
"""

import os
import shutil
import signal
import sys

from PyQt5.QtCore import QProcess, QProcessEnvironment, QTimer
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QWidget,
)

from relay_d.utils import user_settings
from relay_d.utils.coloring_logger import logger
from ..page_style import add_page_header, page_layout
from ..settings.config_manager import config

DISPATCH_FILES = ("obs_config.yaml", "action_config.yaml")
KILL_AFTER_MS = 5000
LOG_MAX_LINES = 5000


def build_args(checkpoint, dispatch_dir, device, hz, dry_run):
    """Arguments of ``python -m relay_d.dispatch.inference_node``. Blank ``hz``: the policy's own rate."""
    args = ["-m", "relay_d.dispatch.inference_node", "--checkpoint", checkpoint,
            "--obs-config", os.path.join(dispatch_dir, DISPATCH_FILES[0]),
            "--action-config", os.path.join(dispatch_dir, DISPATCH_FILES[1]),
            "--device", device]
    if hz.strip():
        args += ["--hz", hz.strip()]
    if dry_run:
        args.append("--dry-run")
    return args


def dispatch_dir_for(folder):
    """A Post-process output folder means its ``action_dispatcher/`` subfolder."""
    sub = os.path.join(folder, "action_dispatcher")
    return sub if os.path.isdir(sub) else folder


def missing_reason(checkpoint, dispatch_dir):
    """Why Start cannot run yet ("" when it can)."""
    if not checkpoint or not os.path.exists(checkpoint):
        return "Choose a checkpoint (a robomimic .pth file or a LeRobot pretrained_model folder)."
    missing = [f for f in DISPATCH_FILES if not os.path.isfile(os.path.join(dispatch_dir or "", f))]
    if not dispatch_dir or missing:
        return (f"The dispatcher folder needs {' and '.join(missing or DISPATCH_FILES)} "
                "(Post-process writes them to <output>/action_dispatcher).")
    return ""


class RunPage:
    def __init__(self, ui_instance):
        self.ui = ui_instance
        self.proc = QProcess(self.ui.page_run)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")  # stream the log as it is written
        self.proc.setProcessEnvironment(env)
        self.proc.readyReadStandardOutput.connect(self._read_output)
        self.proc.finished.connect(self._on_finished)
        self.proc.errorOccurred.connect(self._on_error)
        self._kill_timer = QTimer(self.ui.page_run)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._kill_if_alive)
        self._build_ui()
        self._update_buttons()

    def _build_ui(self):
        root = page_layout(self.ui.page_run)
        add_page_header(root, "Run a Policy")

        box = QGroupBox("Policy")
        form = QFormLayout(box)
        self.edit_checkpoint = QLineEdit(user_settings.get("run_last_checkpoint"))
        self.edit_checkpoint.setPlaceholderText("robomimic .pth file or LeRobot pretrained_model folder")
        btn_file, btn_folder = QPushButton("File..."), QPushButton("Folder...")
        btn_file.clicked.connect(lambda: self._choose_checkpoint(folder=False))
        btn_folder.clicked.connect(lambda: self._choose_checkpoint(folder=True))
        form.addRow("Checkpoint", self._row(self.edit_checkpoint, btn_file, btn_folder))

        self.edit_dispatch = QLineEdit(user_settings.get("run_last_dispatch_dir"))
        self.edit_dispatch.setPlaceholderText("Folder with obs_config.yaml and action_config.yaml")
        btn_dispatch = QPushButton("Browse...")
        btn_dispatch.clicked.connect(self._choose_dispatch_dir)
        form.addRow("Dispatcher folder", self._row(self.edit_dispatch, btn_dispatch))

        self.combo_device = QComboBox()
        self.combo_device.addItems(["cuda", "cpu"])
        self.combo_device.setCurrentText(user_settings.get("infer_device"))
        form.addRow("Device", self.combo_device)
        self.edit_hz = QLineEdit()
        self.edit_hz.setValidator(QDoubleValidator(0.0, 1000.0, 3))
        self.edit_hz.setPlaceholderText("blank: the rate the policy was trained at")
        form.addRow("Rate (Hz)", self.edit_hz)
        self.check_dry_run = QCheckBox("Dry run (log commands, move nothing)")
        self.check_dry_run.setChecked(True)
        form.addRow("", self.check_dry_run)
        root.addWidget(box)
        for edit in (self.edit_checkpoint, self.edit_dispatch):
            edit.textChanged.connect(self._update_buttons)

        self.label_reason = QLabel("")
        self.label_reason.setWordWrap(True)
        root.addWidget(self.label_reason)

        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self.start)
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setObjectName("btn_run_stop")  # red in both themes (styles/*.qss)
        self.btn_stop.setMinimumHeight(60)
        self.btn_stop.clicked.connect(self.stop)
        self.label_state = QLabel("Stopped")
        root.addWidget(self._row(self.btn_start, self.label_state))
        root.addWidget(self.btn_stop)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(LOG_MAX_LINES)  # a long run does not grow memory forever
        root.addWidget(self.log, 1)

    @staticmethod
    def _row(*widgets):
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        for i, x in enumerate(widgets):
            lay.addWidget(x, 1 if i == 0 else 0)
        return w

    def _choose_checkpoint(self, folder):
        start = os.path.dirname(self.edit_checkpoint.text())
        path = (QFileDialog.getExistingDirectory(self.ui.page_run, "Choose a pretrained_model folder", start) if folder
                else QFileDialog.getOpenFileName(self.ui.page_run, "Choose a checkpoint", start, "Checkpoints (*.pth *.pt);;All files (*)")[0])
        if path:
            self.edit_checkpoint.setText(path)

    def _choose_dispatch_dir(self):
        path = QFileDialog.getExistingDirectory(self.ui.page_run, "Choose the dispatcher (or Post-process output) folder",
                                                self.edit_dispatch.text())
        if path:
            self.edit_dispatch.setText(dispatch_dir_for(path))

    # ------------------------------------------------------------------ process

    def is_running(self):
        return self.proc.state() != QProcess.NotRunning

    def command(self):
        """(program, arguments) of the node; the node re-executes itself in the backend's environment."""
        return sys.executable, build_args(self.edit_checkpoint.text(), self.edit_dispatch.text(),
                                          self.combo_device.currentText(), self.edit_hz.text(),
                                          self.check_dry_run.isChecked())

    def _update_buttons(self):
        running = self.is_running()
        reason = "" if running else missing_reason(self.edit_checkpoint.text(), self.edit_dispatch.text())
        self.label_reason.setText(reason)
        self.btn_start.setEnabled(not running and not reason)
        self.btn_stop.setEnabled(running)

    def start(self):
        if self.is_running():
            return
        config.save_settings({"run_last_checkpoint": self.edit_checkpoint.text(),
                              "run_last_dispatch_dir": self.edit_dispatch.text()})
        program, args = self.command()
        self.log.appendPlainText(f"$ {program} {' '.join(args)}")
        if shutil.which("setsid"):  # own process group: a terminal Ctrl+C reaches only the GUI, which stops the node once
            program, args = "setsid", [program, *args]  # the child is no group leader, so setsid execs: same PID
        self.proc.start(program, args)
        self.label_state.setText("Running")
        self._update_buttons()

    def stop(self):
        """SIGINT (the node's Ctrl+C: safe stop, then exit); killed if still alive after 5 s."""
        if not self.is_running():
            return
        self.log.appendPlainText("Stopping (SIGINT)...")
        os.kill(int(self.proc.processId()), signal.SIGINT)
        self._kill_timer.start(KILL_AFTER_MS)

    def _kill_if_alive(self):
        if self.is_running():
            msg = (f"The policy did not stop {KILL_AFTER_MS // 1000} s after SIGINT: KILLED. "
                   "The safe stop may NOT have been sent: check the robot.")
            logger.error(msg)
            self.log.appendPlainText(f"!!! {msg}")
            self.proc.kill()

    def shutdown(self):
        """Stop a running policy before the window closes (blocks at most ~5 s)."""
        if not self.is_running():
            return
        self.stop()
        if not self.proc.waitForFinished(KILL_AFTER_MS):
            self._kill_if_alive()
            self.proc.waitForFinished(1000)

    def _read_output(self):
        self.log.appendPlainText(bytes(self.proc.readAllStandardOutput()).decode(errors="replace").rstrip("\n"))

    def _on_finished(self, code, status):
        self._kill_timer.stop()
        ok = status == QProcess.NormalExit and code == 0
        self.label_state.setText("Stopped" if ok else "Failed (killed by a signal)" if status == QProcess.CrashExit
                                 else f"Failed (exit {code})")
        self.log.appendPlainText(f"Process ended, exit code {code}.")
        self._update_buttons()

    def _on_error(self, error):
        if error == QProcess.FailedToStart:
            self.label_state.setText("Failed (did not start)")
            self.log.appendPlainText(f"Could not start: {self.proc.errorString()}")
            self._update_buttons()
