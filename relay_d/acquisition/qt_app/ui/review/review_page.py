import os
import glob
import json
from threading import Thread

import cv2
import h5py
import numpy as np
import pyqtgraph as pg

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QButtonGroup,
    QStackedWidget,
    QListWidget,
    QListWidgetItem,
    QTextEdit,
    QSlider,
    QFileDialog,
    QTabWidget,
    QGroupBox,
    QRadioButton,
    QLineEdit,
    QProgressBar,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QColor, QBrush

from relay_d.utils.coloring_logger import logger
from ...utils.custom_message_box import CustomMessageBox
from ...utils.demo_file_ops import delete_demo_file
from .demo_validator import DemoValidator, CheckStatus

_STATUS_COLORS = {
    CheckStatus.PASS: "#16A34A",
    CheckStatus.WARN: "#D97706",
    CheckStatus.FAIL: "#DC2626",
}

_STATUS_LABELS = {
    CheckStatus.PASS: "PASS",
    CheckStatus.WARN: "WARN",
    CheckStatus.FAIL: "FAIL",
}

# Groups under data/<demo_name>/ that are not "raw input" containers.
_NON_INPUT_GROUPS = {"streams", "metadata", "gripper_states"}


class ReviewPage:
    """Lists recorded demos in a session, plays back their data, and lets you
    delete/reject a bad one. Only ever reads static HDF5 files - no ROS/live
    topic dependency.
    """

    def __init__(self, ui_instance, record_page=None, settings_page=None):
        self.ui = ui_instance
        self.record_page = record_page
        self.settings_page = settings_page
        self.validator = DemoValidator()

        self.session_dir = None
        self.demo_entries = {}  # file_path -> DemoValidationResult

        self._current_h5 = None
        self._current_demo_group = None
        self._current_image_dataset = None
        self._current_image_kind = None  # "images" or "compressed_data"

        # Vertical cursor line overlaid on whichever stream/gripper plot is
        # currently shown, moved live to track ongoing playback position.
        self._playback_cursor_line = None

        # Playback (rviz2 visualization) state - the DataPlayer/PlaybackProcessManager
        # instances are created lazily on first Play click, keeping ReviewPage's
        # baseline "no ROS/live topic dependency" behavior intact until then.
        self._player = None
        self._process_manager = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        page = self.ui.page_review
        root_layout = QHBoxLayout(page)

        # --- Sidebar: session picker + demo list ---
        sidebar = QWidget()
        sidebar.setMaximumWidth(320)
        sidebar_layout = QVBoxLayout(sidebar)

        self.label_session = QLabel("No session selected")
        self.label_session.setWordWrap(True)
        sidebar_layout.addWidget(self.label_session)

        self.btn_choose_session = QPushButton("Choose Session Folder")
        self.btn_choose_session.clicked.connect(self.choose_session_folder)
        sidebar_layout.addWidget(self.btn_choose_session)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh_demo_list)
        sidebar_layout.addWidget(self.btn_refresh)

        self.list_demos = QListWidget()
        self.list_demos.currentItemChanged.connect(self._on_demo_selected)
        sidebar_layout.addWidget(self.list_demos, 1)

        self.btn_delete = QPushButton("Delete / Reject Demo")
        self.btn_delete.clicked.connect(self.delete_selected_demo)
        sidebar_layout.addWidget(self.btn_delete)

        root_layout.addWidget(sidebar)

        # --- Detail panel, split into a "Data" tab (existing plot/scrub view)
        # and a "Playback" tab (rviz2-driven visualization controls) ---
        self.tab_widget = QTabWidget()
        root_layout.addWidget(self.tab_widget, 1)

        data_tab = QWidget()
        detail_layout = QVBoxLayout(data_tab)

        self.text_validation = QTextEdit()
        self.text_validation.setReadOnly(True)
        self.text_validation.setMaximumHeight(160)
        detail_layout.addWidget(self.text_validation)

        # --- Channel matrix: one checkable button per selectable channel ---
        self.channel_matrix_widget = QWidget()
        self.channel_matrix_layout = QGridLayout(self.channel_matrix_widget)
        detail_layout.addWidget(self.channel_matrix_widget)

        self.channel_button_group = QButtonGroup(self.channel_matrix_widget)
        self.channel_button_group.setExclusive(True)
        self.channel_buttons = []

        # --- Single display area: only one of these pages is ever visible ---
        self.display_stack = QStackedWidget()

        self.graph_page = QWidget()
        self.layout_display_graph = QVBoxLayout(self.graph_page)
        self.display_stack.addWidget(self.graph_page)

        self.image_page = QWidget()
        layout_image = QVBoxLayout(self.image_page)
        self.label_image = QLabel("No image data")
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_image.setMinimumHeight(300)
        layout_image.addWidget(self.label_image, 1)
        self.slider_scrub = QSlider(Qt.Horizontal)
        self.slider_scrub.valueChanged.connect(self._on_scrub)
        layout_image.addWidget(self.slider_scrub)
        self.label_frame_index = QLabel("Frame 0 / 0")
        layout_image.addWidget(self.label_frame_index)
        self.display_stack.addWidget(self.image_page)

        detail_layout.addWidget(self.display_stack, 1)

        self.tab_widget.addTab(data_tab, "Data")
        self.tab_widget.addTab(self._build_playback_tab(), "Playback")

    def _build_playback_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        desc_group = QGroupBox("Robot Description")
        desc_layout = QVBoxLayout(desc_group)

        self.radio_urdf_file = QRadioButton("URDF/xacro file")
        self.radio_urdf_file.setChecked(True)
        self.radio_robot_desc_topic = QRadioButton("Robot description topic")
        self._robot_desc_mode_group = QButtonGroup(desc_group)
        self._robot_desc_mode_group.setExclusive(True)
        self._robot_desc_mode_group.addButton(self.radio_urdf_file)
        self._robot_desc_mode_group.addButton(self.radio_robot_desc_topic)
        self.radio_urdf_file.toggled.connect(self._on_robot_desc_mode_changed)

        desc_layout.addWidget(self.radio_urdf_file)
        self.widget_urdf_file_row = QWidget()
        file_row_layout = QHBoxLayout(self.widget_urdf_file_row)
        file_row_layout.setContentsMargins(20, 0, 0, 0)
        self.edit_urdf_path = QLineEdit()
        file_row_layout.addWidget(self.edit_urdf_path, 1)
        btn_browse_urdf = QPushButton("Browse...")
        btn_browse_urdf.clicked.connect(self._browse_urdf_file)
        file_row_layout.addWidget(btn_browse_urdf)
        desc_layout.addWidget(self.widget_urdf_file_row)

        desc_layout.addWidget(self.radio_robot_desc_topic)
        self.widget_robot_desc_topic_row = QWidget()
        topic_row_layout = QHBoxLayout(self.widget_robot_desc_topic_row)
        topic_row_layout.setContentsMargins(20, 0, 0, 0)
        topic_row_layout.addWidget(QLabel("Topic:"))
        self.edit_robot_desc_topic = QLineEdit("/robot_description")
        topic_row_layout.addWidget(self.edit_robot_desc_topic, 1)
        desc_layout.addWidget(self.widget_robot_desc_topic_row)
        self.widget_robot_desc_topic_row.setVisible(False)

        layout.addWidget(desc_group)

        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Fixed frame:"))
        self.edit_fixed_frame = QLineEdit("base_link")
        frame_row.addWidget(self.edit_fixed_frame, 1)
        layout.addLayout(frame_row)

        self.label_channel_summary = QLabel("")
        self.label_channel_summary.setWordWrap(True)
        layout.addWidget(self.label_channel_summary)

        controls_row = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self._on_play_clicked)
        controls_row.addWidget(self.btn_play)
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._on_pause_clicked)
        controls_row.addWidget(self.btn_pause)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        controls_row.addWidget(self.btn_stop)
        layout.addLayout(controls_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        self.label_playback_progress = QLabel("Frame 0 / 0")
        layout.addWidget(self.label_playback_progress)

        self.label_playback_status = QLabel("")
        self.label_playback_status.setWordWrap(True)
        layout.addWidget(self.label_playback_status)

        layout.addStretch(1)

        self._playback_progress_timer = QTimer()
        self._playback_progress_timer.timeout.connect(self._on_progress_tick)

        return tab

    # ------------------------------------------------------------------
    # Session handling
    # ------------------------------------------------------------------

    def sync_with_current_session(self):
        """Pick up the live recording session's directory if one exists."""
        session_dir = None
        if self.record_page is not None:
            data_recorder = getattr(self.record_page, "data_recorder", None)
            if data_recorder is not None:
                session_dir = getattr(data_recorder, "recorded_data_dir", None)

        if not session_dir:
            session_dir = self.session_dir

        if not session_dir and self.settings_page is not None and hasattr(
            self.settings_page, "get_hdf5_save_folder"
        ):
            session_dir = self.settings_page.get_hdf5_save_folder() or None

        if session_dir:
            self.session_dir = session_dir
            self.label_session.setText(f"Session: {session_dir}")

    def choose_session_folder(self):
        folder = QFileDialog.getExistingDirectory(
            None, "Choose Session Folder", self.session_dir or ""
        )
        if folder:
            self.session_dir = folder
            self.label_session.setText(f"Session: {folder}")
            self.refresh_demo_list()

    def refresh_demo_list(self):
        self.list_demos.clear()
        self.demo_entries = {}
        self._close_current_file()

        if not self.session_dir or not os.path.isdir(self.session_dir):
            self.label_session.setText("No session selected")
            return

        paths = sorted(
            p
            for p in glob.glob(os.path.join(self.session_dir, "*.h5"))
            if os.path.basename(p) != "combined_dataset.h5"
        )

        for path in paths:
            try:
                result = self.validator.validate(path)
            except Exception as e:
                logger.error(f"Error validating {path}: {e}")
                continue

            self.demo_entries[path] = result
            label = result.demo_name or os.path.basename(path)
            status_label = _STATUS_LABELS[result.overall_status]
            item = QListWidgetItem(
                f"[{status_label}] {label}  "
                f"({result.samples} samples, {result.duration:.2f}s)"
            )
            item.setData(Qt.UserRole, path)
            item.setForeground(QBrush(QColor(_STATUS_COLORS[result.overall_status])))
            self.list_demos.addItem(item)

        if not paths:
            self.label_session.setText(f"Session: {self.session_dir} (no demos found)")

    # ------------------------------------------------------------------
    # Demo loading / playback
    # ------------------------------------------------------------------

    def _on_demo_selected(self, current, previous):
        if current is None:
            return
        path = current.data(Qt.UserRole)
        self.load_demo(path)

    def load_demo(self, path):
        self._close_current_file()

        result = self.demo_entries.get(path)
        if result is None:
            result = self.validator.validate(path)
            self.demo_entries[path] = result
        self.text_validation.setPlainText(result.summary_text())

        try:
            self._current_h5 = h5py.File(path, "r")
        except Exception as e:
            logger.error(f"Error opening demo file {path}: {e}")
            return

        if "data" not in self._current_h5:
            return
        demo_names = list(self._current_h5["data"].keys())
        if not demo_names:
            return

        demo_group = self._current_h5["data"][demo_names[0]]
        self._current_demo_group = demo_group

        self._load_image_frames(demo_group)
        self._build_channel_matrix(demo_group)

    def _close_current_file(self):
        if self._current_h5 is not None:
            try:
                self._current_h5.close()
            except Exception:
                pass
        self._current_h5 = None
        self._current_demo_group = None
        self._current_image_dataset = None
        self._current_image_kind = None
        self.slider_scrub.setMaximum(0)
        self.label_image.setText("No image data")
        self.label_frame_index.setText("Frame 0 / 0")
        self._clear_channel_matrix()
        self.display_stack.setCurrentWidget(self.graph_page)
        self._clear_layout(self.layout_display_graph)
        self.layout_display_graph.addWidget(QLabel("No demo loaded"))

    # ------------------------------------------------------------------
    # Plot rendering helpers
    # ------------------------------------------------------------------

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _find_input_group(self, demo_group, dataset_key, name_hints=None):
        candidates = []
        for name, item in demo_group.items():
            if name in _NON_INPUT_GROUPS or not isinstance(item, h5py.Group):
                continue
            if dataset_key in item:
                candidates.append((name, item))

        if not candidates:
            return None, None

        if name_hints:
            for name, item in candidates:
                lowered = name.lower()
                if any(h in lowered for h in name_hints):
                    return name, item

        return candidates[0]

    def _gripper_state_keys(self, demo_group):
        if "gripper_states" not in demo_group:
            return []
        gripper_group = demo_group["gripper_states"]
        state_keys = [k for k in gripper_group.keys() if k.endswith("_states")]
        if not state_keys:
            state_keys = [
                k for k in gripper_group.keys() if not k.endswith("_timestamps")
            ]
        return state_keys

    def _has_image_data(self, demo_group):
        _, grp = self._find_input_group(demo_group, "images")
        if grp is None:
            _, grp = self._find_input_group(demo_group, "compressed_data")
        return grp is not None

    def _clear_channel_matrix(self):
        self._clear_layout(self.channel_matrix_layout)
        for btn in self.channel_buttons:
            self.channel_button_group.removeButton(btn)
        self.channel_buttons = []

    def _build_channel_matrix(self, demo_group):
        self._clear_channel_matrix()

        channels = []
        if self._gripper_state_keys(demo_group):
            channels.append(("gripper", None, "Gripper"))
        if self._has_image_data(demo_group):
            channels.append(("image", None, "Image"))
        if "streams" in demo_group:
            for name in sorted(demo_group["streams"].keys()):
                shape_text = "x".join(str(d) for d in demo_group["streams"][name].shape)
                channels.append(("stream", name, f"{name}\n({shape_text})"))

        columns = 3
        for i, (kind, name, label) in enumerate(channels):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, k=kind, n=name: self._select_channel(k, n))
            self.channel_button_group.addButton(btn)
            self.channel_matrix_layout.addWidget(btn, i // columns, i % columns)
            self.channel_buttons.append(btn)

        if channels:
            self.channel_buttons[0].setChecked(True)
            self._select_channel(channels[0][0], channels[0][1])
        else:
            self.display_stack.setCurrentWidget(self.graph_page)
            self._clear_layout(self.layout_display_graph)
            self.layout_display_graph.addWidget(QLabel("No data available for this demo"))

    def _select_channel(self, kind, name):
        # The plot widget backing any previous cursor line is about to be
        # discarded (by _clear_layout below, or simply not touched for the
        # image page) - drop the reference so _on_progress_tick doesn't try
        # to move a line that no longer belongs to a visible plot.
        self._playback_cursor_line = None

        if kind == "image":
            self.display_stack.setCurrentWidget(self.image_page)
            return

        self.display_stack.setCurrentWidget(self.graph_page)
        self._clear_layout(self.layout_display_graph)

        demo_group = self._current_demo_group
        if demo_group is None:
            return

        if kind == "gripper":
            self._render_gripper_plot(demo_group)
        elif kind == "stream":
            self._render_stream_plot(demo_group, name)

    def _add_playback_cursor(self, plot):
        line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("r", width=2))
        plot.addItem(line)
        self._playback_cursor_line = line

    def _render_stream_plot(self, demo_group, name):
        data = demo_group["streams"][name][:]
        plot = pg.PlotWidget(title=name)
        if data.ndim == 1:
            plot.plot(data, pen=pg.intColor(0, hues=1))
        else:
            plot.addLegend()
            n_dims = data.shape[1]
            for i in range(n_dims):
                plot.plot(
                    data[:, i],
                    pen=pg.intColor(i, hues=max(n_dims, 1)),
                    name=f"{name}[{i}]",
                )
        self._add_playback_cursor(plot)
        self.layout_display_graph.addWidget(plot)

    def _render_gripper_plot(self, demo_group):
        state_keys = self._gripper_state_keys(demo_group)
        if not state_keys:
            self.layout_display_graph.addWidget(QLabel("No gripper state data found"))
            return

        gripper_group = demo_group["gripper_states"]
        plot = pg.PlotWidget(title="Gripper State")
        plot.addLegend()
        for i, key in enumerate(state_keys):
            values = gripper_group[key][:]
            plot.plot(values, pen=pg.intColor(i, hues=max(len(state_keys), 1)), name=key)
        self._add_playback_cursor(plot)
        self.layout_display_graph.addWidget(plot)

    # ------------------------------------------------------------------
    # Image scrubbing
    # ------------------------------------------------------------------

    def _load_image_frames(self, demo_group):
        self._current_image_dataset = None
        self._current_image_kind = None

        name, grp = self._find_input_group(demo_group, "images")
        kind = "images"
        if grp is None:
            name, grp = self._find_input_group(demo_group, "compressed_data")
            kind = "compressed_data"

        if grp is None:
            self.slider_scrub.setMaximum(0)
            self.label_image.setText("No image data")
            self.label_frame_index.setText("Frame 0 / 0")
            return

        dataset = grp[kind]
        self._current_image_dataset = dataset
        self._current_image_kind = kind
        count = dataset.shape[0]
        self.slider_scrub.setMaximum(max(count - 1, 0))
        self.slider_scrub.setValue(0)
        self._show_frame(0)

    def _on_scrub(self, value):
        self._show_frame(value)

    def _show_frame(self, index):
        if self._current_image_dataset is None:
            return
        count = self._current_image_dataset.shape[0]
        if count == 0:
            return
        index = max(0, min(index, count - 1))

        try:
            if self._current_image_kind == "compressed_data":
                raw = self._current_image_dataset[index]
                np_arr = np.frombuffer(bytes(raw), np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                cv_image = self._current_image_dataset[index]
                if cv_image.ndim == 2:
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)

            if cv_image is None:
                return

            self._display_cv_frame(cv_image)
            self.label_frame_index.setText(f"Frame {index} / {count - 1}")
        except Exception as e:
            logger.error(f"Error showing frame {index}: {e}")

    def _display_cv_frame(self, cv_image):
        height, width = cv_image.shape[:2]
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * width
        qt_image = QImage(
            rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qt_image).scaled(
            self.label_image.width() or width,
            self.label_image.height() or height,
            Qt.KeepAspectRatio,
        )
        self.label_image.setPixmap(pixmap)

    # ------------------------------------------------------------------
    # Delete / reject
    # ------------------------------------------------------------------

    def delete_selected_demo(self):
        item = self.list_demos.currentItem()
        if item is None:
            CustomMessageBox.info(
                "Select a demo to delete first.", title="No Demo Selected"
            )
            return

        path = item.data(Qt.UserRole)
        confirmed = CustomMessageBox.question(
            f"Permanently delete this demo?\n\n{os.path.basename(path)}",
            title="Delete Demo",
        )
        if not confirmed:
            return

        self._close_current_file()

        if delete_demo_file(path):
            if self.record_page is not None and hasattr(
                self.record_page, "remove_session_file"
            ):
                self.record_page.remove_session_file(path)
            self.refresh_demo_list()
        else:
            CustomMessageBox.error(f"Failed to delete: {path}", title="Delete Failed")

    # ------------------------------------------------------------------
    # RViz2 playback (Playback tab)
    #
    # All ROS-touching modules (data_player, playback_process_manager,
    # robot_description_resolver) are imported lazily, only from inside these
    # handlers, so ReviewPage keeps working standalone on machines without
    # xacro/rviz2/robot_state_publisher installed.
    # ------------------------------------------------------------------

    def _on_robot_desc_mode_changed(self, _checked):
        is_file_mode = self.radio_urdf_file.isChecked()
        self.widget_urdf_file_row.setVisible(is_file_mode)
        self.widget_robot_desc_topic_row.setVisible(not is_file_mode)

    def _browse_urdf_file(self):
        path, _ = QFileDialog.getOpenFileName(
            None,
            "Choose Robot Description",
            self.edit_urdf_path.text() or "",
            "Robot description (*.urdf *.xacro);;All files (*)",
        )
        if path:
            self.edit_urdf_path.setText(path)

    def _set_playback_controls_locked(self, locked: bool):
        for widget in (
            self.list_demos,
            self.btn_delete,
            self.btn_choose_session,
            self.btn_refresh,
            # Locked during playback so the user can't fight the live-synced
            # position by dragging it manually - see _on_progress_tick.
            self.slider_scrub,
        ):
            widget.setEnabled(not locked)

    def _on_play_clicked(self):
        from .data_player import PlaybackState

        if self._player is not None and self._player.state == PlaybackState.PAUSED:
            self._player.play()
            self.btn_play.setEnabled(False)
            self.btn_pause.setEnabled(True)
            self.btn_stop.setEnabled(True)
            return

        item = self.list_demos.currentItem()
        if item is None:
            CustomMessageBox.info(
                "Select a demo to play first.", title="No Demo Selected"
            )
            return
        demo_path = item.data(Qt.UserRole)

        self.label_playback_status.setText("")

        if self.radio_urdf_file.isChecked():
            urdf_path = self.edit_urdf_path.text().strip()
            if not urdf_path:
                CustomMessageBox.error(
                    "Choose a URDF/xacro file first.", title="No File Selected"
                )
                return
            from .robot_description_resolver import (
                resolve_from_file,
                RobotDescriptionError,
            )

            try:
                urdf_xml = resolve_from_file(urdf_path)
            except RobotDescriptionError as e:
                CustomMessageBox.error(str(e), title="Robot Description Error")
                return
            self._launch_playback(demo_path, urdf_xml)
        else:
            self._resolve_topic_then_launch(demo_path)

    def _resolve_topic_then_launch(self, demo_path):
        topic = self.edit_robot_desc_topic.text().strip() or "/robot_description"
        self.label_playback_status.setText(f"Waiting for '{topic}'...")
        self.btn_play.setEnabled(False)

        result_holder = {}

        def _worker():
            from .robot_description_resolver import (
                resolve_from_topic,
                RobotDescriptionError,
            )

            try:
                result_holder["urdf"] = resolve_from_topic(topic, timeout_sec=5.0)
            except RobotDescriptionError as e:
                result_holder["error"] = str(e)

        thread = Thread(target=_worker, daemon=True)
        thread.start()

        poll_timer = QTimer(self.tab_widget)

        def _poll():
            if thread.is_alive():
                return
            poll_timer.stop()
            self.label_playback_status.setText("")
            if "error" in result_holder:
                self.btn_play.setEnabled(True)
                CustomMessageBox.error(
                    result_holder["error"], title="Robot Description Error"
                )
                return
            self._launch_playback(demo_path, result_holder["urdf"])

        poll_timer.timeout.connect(_poll)
        poll_timer.start(150)
        # Keep a reference so the timer isn't garbage-collected mid-poll.
        self._pending_desc_timer = poll_timer

    def _launch_playback(self, demo_path, urdf_xml):
        from .playback_process_manager import (
            PlaybackProcessManager,
            PlaybackEnvironmentError,
        )
        from .data_player import DataPlayer

        fixed_frame = self.edit_fixed_frame.text().strip() or "base_link"

        if self._player is None:
            self._player = DataPlayer(
                self.ui.register_ros_node, self.ui.unregister_ros_node
            )
        if self._process_manager is None:
            self._process_manager = PlaybackProcessManager()

        try:
            self._player.load(demo_path)
        except Exception as e:
            self.btn_play.setEnabled(True)
            CustomMessageBox.error(
                f"Failed to load demo for playback: {e}", title="Playback Error"
            )
            return

        if not self._player.has_playable_channels():
            self.btn_play.setEnabled(True)
            CustomMessageBox.error(
                "This demo has no channels that can be visualized "
                "(no joint states or TF).",
                title="Nothing to Visualize",
            )
            return

        self.label_channel_summary.setText(self._player.channel_summary())

        try:
            self._process_manager.start(urdf_xml, fixed_frame)
        except PlaybackEnvironmentError as e:
            self.btn_play.setEnabled(True)
            CustomMessageBox.error(str(e), title="Missing Dependencies")
            return
        except Exception as e:
            self._process_manager.stop()
            self.btn_play.setEnabled(True)
            CustomMessageBox.error(
                f"Failed to launch rviz2/robot_state_publisher: {e}",
                title="Playback Error",
            )
            return

        if not self._player.play():
            self._process_manager.stop()
            self.btn_play.setEnabled(True)
            CustomMessageBox.error("Failed to start playback.", title="Playback Error")
            return

        self._set_playback_controls_locked(True)
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self._playback_progress_timer.start(150)

    def _on_pause_clicked(self):
        if self._player is None:
            return
        self._player.pause()
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)

    def _on_stop_clicked(self):
        self.shutdown_playback()

    def _on_progress_tick(self):
        if self._player is None:
            return
        from .data_player import PlaybackState

        frame, total = self._player.progress
        elapsed, duration = self._player.time_progress
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(frame)
        self.label_playback_progress.setText(
            f"Frame {frame} / {total}   ({elapsed:.1f}s / {duration:.1f}s)"
        )

        # Live-sync the Data tab so switching to it during playback shows the
        # recorded image/signal that corresponds to what's currently playing.
        if self._current_image_dataset is not None:
            self.slider_scrub.setValue(min(frame, self.slider_scrub.maximum()))
        if self._playback_cursor_line is not None:
            self._playback_cursor_line.setValue(frame)

        if self._player.state == PlaybackState.FINISHED:
            self._playback_progress_timer.stop()
            self.btn_play.setEnabled(True)
            self.btn_pause.setEnabled(False)
            self._set_playback_controls_locked(False)

    def shutdown_playback(self):
        """Full teardown: stop publishing, reset to frame 0, close rviz2 and
        robot_state_publisher. Safe to call even if playback was never started.
        """
        if getattr(self, "_playback_progress_timer", None) is not None:
            self._playback_progress_timer.stop()
        if self._player is not None:
            self._player.close()
        if self._process_manager is not None:
            self._process_manager.stop()

        if hasattr(self, "progress_bar"):
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
        if hasattr(self, "label_playback_progress"):
            self.label_playback_progress.setText("Frame 0 / 0")
        if hasattr(self, "label_channel_summary"):
            self.label_channel_summary.setText("")
        if hasattr(self, "label_playback_status"):
            self.label_playback_status.setText("")

        self._set_playback_controls_locked(False)
        if hasattr(self, "btn_play"):
            self.btn_play.setEnabled(True)
            self.btn_pause.setEnabled(False)
            self.btn_stop.setEnabled(False)
