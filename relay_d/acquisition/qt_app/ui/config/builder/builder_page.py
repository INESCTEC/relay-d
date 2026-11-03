"""Config page > "Build from ROS": build an acquisition config from the live ROS 2 graph (topics, TF frames, message fields).

The edited state is a ``ConfigDraft``; the pure modules next to this file do the work (``ros_catalog`` lists what the graph
offers, ``config_check`` validates, ``urdf_limits`` fills normalization limits). This file only turns them into widgets.
Steps are sub-tabs: Sources, Inputs, Observations, Actions, Recording, Review.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from relay_d.utils.coloring_logger import logger
from ....utils.custom_message_box import CustomMessageBox
from . import ros_catalog, urdf_limits
from .config_check import ERROR, Problem, check_config, has_errors, stream_build_errors, stream_width
from .draft import ConfigDraft, InputDraft, StreamDraft, suggest_input_name

_PEEK_SECONDS = 3.0
_MANUAL_TYPES = sorted(ros_catalog.KINDS) + ["std_msgs/msg/Float64"]


def _table(headers, select_rows=True, stretch=0):
    """``stretch``: the column that takes the free width; the others fit their text."""
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    for c in range(len(headers)):
        table.horizontalHeader().setSectionResizeMode(c, QHeaderView.Stretch if c == stretch else QHeaderView.ResizeToContents)
    table.verticalHeader().setVisible(False)
    if select_rows:
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
    return table


def _item(text, editable=False):
    item = QTableWidgetItem(str(text))
    if not editable:
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item


def _button(text, slot):
    b = QPushButton(text)
    b.clicked.connect(lambda _checked=False: slot())
    return b


class BuilderPage(QWidget):
    """``get_node()``: the ROS node of the GUI. ``get_tf_buffer(tf_topic)``: a TF buffer for that topic (or None).
    ``use_config(path)``: load a saved file into the app. ``current_config()``: the loaded config as a dict, or None."""

    def __init__(self, get_node, get_tf_buffer, use_config, current_config, loader=ros_catalog.default_loader, dialog_dir=lambda: "", parent=None):
        super().__init__(parent)
        self._get_node, self._get_tf_buffer, self._use_config = get_node, get_tf_buffer, use_config
        self._current_config, self._loader, self._dialog_dir = current_config, loader, dialog_dir
        self.draft = ConfigDraft()
        self._topics = []
        self._peeks = []  # live subscriptions waiting for one message

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 8, 0, 0)
        row = QHBoxLayout()
        self.label_hint = QLabel("Pick what to record from the running ROS 2 graph, then build the observations. Save at the Review step.")
        self.label_hint.setWordWrap(True)
        row.addWidget(self.label_hint, 1)
        row.addWidget(_button("Edit loaded config", self.load_current))
        row.addWidget(_button("Start empty", self.reset))
        root.addLayout(row)
        self.tabs = QTabWidget()
        for title, build in (("1 Sources", self._build_sources), ("2 Inputs", self._build_inputs), ("3 Observations", self._build_observations),
                             ("4 Actions", self._build_actions), ("5 Recording", self._build_recording), ("6 Review", self._build_review)):
            self.tabs.addTab(build(), title)
        self.tabs.currentChanged.connect(lambda _i: self.refresh())
        root.addWidget(self.tabs, 1)
        self.refresh_topics()

    # ------------------------------------------------------------------ 1 Sources

    def _build_sources(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        self.edit_filter = QLineEdit()
        self.edit_filter.setPlaceholderText("Filter topics by name or type")
        self.edit_filter.textChanged.connect(lambda _t: self._fill_topics())
        row.addWidget(self.edit_filter, 1)
        row.addWidget(_button("Refresh", self.refresh_topics))
        row.addWidget(_button("Add as input", self._add_selected_topic))
        lay.addLayout(row)
        self.table_topics = _table(["Topic", "Type", "Kind"])
        self.table_topics.doubleClicked.connect(lambda _i: self._add_selected_topic())
        lay.addWidget(self.table_topics, 1)

        row = QHBoxLayout()  # a topic the graph does not show (yet)
        self.edit_manual_topic = QLineEdit()
        self.edit_manual_topic.setPlaceholderText("/topic not listed above")
        self.combo_manual_type = QComboBox()
        self.combo_manual_type.addItems(_MANUAL_TYPES)
        row.addWidget(self.edit_manual_topic, 1)
        row.addWidget(self.combo_manual_type)
        row.addWidget(_button("Add manually", lambda: self.add_topic_input(self.edit_manual_topic.text().strip(), self.combo_manual_type.currentText())))
        lay.addLayout(row)

        row = QHBoxLayout()  # a TF transform between two frames
        self.edit_tf_topic = QLineEdit("/tf")
        self.combo_parent, self.combo_child = QComboBox(), QComboBox()
        for combo in (self.combo_parent, self.combo_child):
            combo.setEditable(True)
            combo.setMinimumWidth(160)
        row.addWidget(QLabel("TF topic"))
        row.addWidget(self.edit_tf_topic)
        row.addWidget(QLabel("parent"))
        row.addWidget(self.combo_parent)
        row.addWidget(QLabel("child"))
        row.addWidget(self.combo_child)
        row.addWidget(_button("Refresh frames", self.refresh_frames))
        row.addWidget(_button("Add TF input", lambda: self.add_tf_input(self.combo_parent.currentText().strip(), self.combo_child.currentText().strip(), self.edit_tf_topic.text().strip())))
        lay.addLayout(row)
        return w

    def refresh_topics(self):
        node = self._get_node()
        try:
            self._topics = ros_catalog.list_topics(node) if node is not None else []
        except Exception as e:
            logger.warning(f"Could not list ROS topics: {e}")
            self._topics = []
        self._fill_topics()
        self.refresh_frames()

    def _fill_topics(self):
        text = self.edit_filter.text().lower()
        rows = [t for t in self._topics if text in t.name.lower() or text in t.type.lower() or text in t.kind]
        self.table_topics.setRowCount(len(rows))
        for r, t in enumerate(rows):
            for c, value in enumerate((t.name, t.type, t.kind)):
                self.table_topics.setItem(r, c, _item(value))

    def refresh_frames(self):
        buffer = self._get_tf_buffer(self.edit_tf_topic.text().strip() or "/tf")
        try:
            frames = ros_catalog.tf_frames(buffer) if buffer is not None else []
        except Exception as e:
            logger.warning(f"Could not read TF frames: {e}")
            frames = []
        for combo in (self.combo_parent, self.combo_child):
            current = combo.currentText()
            combo.clear()
            combo.addItems(frames)
            combo.setEditText(current)

    def _add_selected_topic(self):
        row = self.table_topics.currentRow()
        if row >= 0:
            self.add_topic_input(self.table_topics.item(row, 0).text(), self.table_topics.item(row, 1).text())

    def _ask_name(self, suggestion):
        name, ok = QInputDialog.getText(self, "Input name", "Name of this input (its first word picks the live view on the Record page):", text=suggestion)
        return name if ok and name.strip() else None

    def add_topic_input(self, topic, msg_type, name=None):
        if not topic.startswith("/"):
            CustomMessageBox.warning("A topic name starts with '/'.")
            return None
        kind = ros_catalog.classify(msg_type)
        name = name or self._ask_name(self.draft.unique_name(suggest_input_name(kind, topic)))
        if not name:
            return None
        modality = ros_catalog.guess_modality(topic) if kind == "image" else ""
        inp = self.draft.add_input(InputDraft(name=self.draft.unique_name(name), topic=topic, msg_type=msg_type, modality="depth" if modality == "depth" else ""))
        self.refresh_inputs()
        return inp

    def add_tf_input(self, parent, child, tf_topic="/tf", name=None):
        if not parent or not child or parent == child:
            CustomMessageBox.warning("Choose two different frames: a parent and a child.")
            return None
        name = name or self._ask_name(self.draft.unique_name(f"tf_{child}"))
        if not name:
            return None
        inp = InputDraft(name=self.draft.unique_name(name), kind="tf", parent_frame=parent, child_frame=child, tf_topic="" if tf_topic in ("", "/tf") else tf_topic)
        self.draft.add_input(inp)
        self.refresh_inputs()
        return inp

    # ------------------------------------------------------------------ 2 Inputs

    def _build_inputs(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.table_inputs = _table(["Name", "Source", "Output map", "Modality", "Joint names", "Fill gaps"], stretch=1)
        self.table_inputs.itemChanged.connect(self._input_edited)
        lay.addWidget(self.table_inputs, 1)
        row = QHBoxLayout()
        row.addWidget(_button("Read joint names from the topic", self.read_joint_names))
        row.addWidget(_button("Remove input", self.remove_selected_input))
        row.addStretch(1)
        lay.addLayout(row)
        note = QLabel("Output map: HDF5 group name, if not the input name. Modality: depth for depth images, scan for laser scans (empty = rgb).")
        note.setWordWrap(True)
        lay.addWidget(note)
        return w

    def refresh_inputs(self):
        table = self.table_inputs
        table.blockSignals(True)
        table.setRowCount(len(self.draft.inputs))
        for r, inp in enumerate(self.draft.inputs.values()):
            source = f"TF {inp.parent_frame} -> {inp.child_frame}" if inp.kind == "tf" else inp.topic
            cells = [_item(inp.name), _item(source), _item(inp.output_map, True), _item(inp.modality, True), _item(", ".join(inp.joint_names), True)]
            gaps = QTableWidgetItem()
            gaps.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            gaps.setCheckState(Qt.Checked if inp.fill_gaps else Qt.Unchecked)
            for c, item in enumerate(cells + [gaps]):
                table.setItem(r, c, item)
        table.blockSignals(False)

    def _input_edited(self, item):
        inp = list(self.draft.inputs.values())[item.row()]
        text = item.text().strip()
        col = item.column()
        if col == 2:
            inp.output_map = text
        elif col == 3:
            inp.modality = text
        elif col == 4:
            inp.joint_names = [n.strip() for n in text.split(",") if n.strip()]
        elif col == 5:
            inp.fill_gaps = item.checkState() == Qt.Checked

    def _selected_input(self):
        row = self.table_inputs.currentRow()
        return list(self.draft.inputs.values())[row] if 0 <= row < len(self.draft.inputs) else None

    def remove_selected_input(self):
        inp = self._selected_input()
        if inp:
            self.draft.remove_input(inp.name)
            self.refresh_inputs()

    def read_joint_names(self):
        """Take ``joint_names`` from one message of the selected JointState topic (a few seconds, the GUI keeps running)."""
        inp = self._selected_input()
        if inp is None or inp.kind != "topic":
            CustomMessageBox.warning("Select a JointState input first.")
            return
        self.peek(inp.topic, inp.msg_type or "sensor_msgs/msg/JointState", lambda msg: self._joint_names_read(inp, msg))

    def _joint_names_read(self, inp, msg):
        names = ros_catalog.joint_names(msg) if msg is not None else []
        if not names:
            CustomMessageBox.warning(f"No joint names received on {inp.topic} within {_PEEK_SECONDS:.0f} s.")
            return
        inp.joint_names = names
        self.refresh_inputs()

    def peek(self, topic, msg_type, on_msg, latched=False):
        """Call ``on_msg(message)`` with the next message on ``topic``, or ``on_msg(None)`` after a few seconds.

        Uses the GUI node (already spun by the app's timer), so nothing blocks and no second spin runs.
        ``latched``: also get the last message published before (``/robot_description``).
        """
        node = self._get_node()
        if node is None:
            on_msg(None)
            return
        box = {}
        try:
            from rclpy.qos import QoSDurabilityPolicy, QoSProfile, ReliabilityPolicy

            qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL) if latched else QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            sub = node.create_subscription(self._loader(msg_type), topic, lambda m: box.setdefault("msg", m), qos)
        except Exception as e:
            logger.warning(f"Could not subscribe to {topic}: {e}")
            on_msg(None)
            return
        timer = QTimer(self)
        waited = [0.0]

        def poll():
            waited[0] += 0.1
            if "msg" in box or waited[0] >= _PEEK_SECONDS:
                timer.stop()
                timer.deleteLater()
                node.destroy_subscription(sub)
                on_msg(box.get("msg"))

        timer.timeout.connect(poll)
        timer.start(100)

    # ------------------------------------------------------------------ 3 Observations

    def _build_observations(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        self.combo_obs_input = QComboBox()
        self.combo_obs_input.currentIndexChanged.connect(lambda _i: self._fill_fields())
        row.addWidget(QLabel("Input"))
        row.addWidget(self.combo_obs_input, 1)
        lay.addLayout(row)
        self.tree_fields = QTreeWidget()
        self.tree_fields.setHeaderLabels(["Field (tick to add to the specs)", "Kind"])
        self.tree_fields.setMinimumHeight(150)
        self.tree_fields.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree_fields.itemChanged.connect(self._field_toggled)
        lay.addWidget(self.tree_fields, 1)
        self.label_obs_note = QLabel("")
        self.label_obs_note.setWordWrap(True)
        lay.addWidget(self.label_obs_note)

        form = QFormLayout()
        self.edit_stream_name = QLineEdit()
        self.edit_stream_name.setPlaceholderText("robot0_joint_pos")
        self.text_specs = QPlainTextEdit()
        self.text_specs.setPlaceholderText("One spec per line, e.g. joint_1.position[:]  or  tf_eef.translation.x  or  a.x - b.x")
        self.text_specs.setMaximumHeight(80)
        self.text_specs.textChanged.connect(self._resize_limits)
        form.addRow("Name", self.edit_stream_name)
        form.addRow("Specs", self.text_specs)
        lay.addLayout(form)
        row = QHBoxLayout()
        self.check_normalize = QCheckBox("Normalize to [-1, 1] (needs limits)")
        self.check_normalize.toggled.connect(lambda _on: self._resize_limits())
        self.check_rpy = QCheckBox("Quaternion to roll / pitch / yaw (4 specs: x y z w)")
        row.addWidget(self.check_normalize)
        row.addWidget(self.check_rpy)
        row.addStretch(1)
        lay.addLayout(row)
        self.table_limits = _table(["Lower", "Upper"], select_rows=False)
        self.table_limits.setMaximumHeight(130)
        self.table_limits.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_limits.setVisible(False)  # shown once "Normalize" is ticked
        lay.addWidget(self.table_limits)
        row = QHBoxLayout()
        row.addWidget(_button("Fill limits from URDF file...", self.fill_limits_from_file))
        row.addWidget(_button("Fill limits from /robot_description", self.fill_limits_from_topic))
        row.addStretch(1)
        row.addWidget(_button("Add stream", self.add_stream))
        lay.addLayout(row)

        row = QHBoxLayout()  # cameras and point clouds are recorded as images / points, not as stream values
        self.edit_camera_key = QLineEdit()
        self.edit_camera_key.setPlaceholderText("observation name, e.g. agentview_image")
        row.addWidget(self.edit_camera_key, 1)
        row.addWidget(_button("Add selected input as camera / point cloud", self.add_camera))
        lay.addLayout(row)
        self.list_obs = QListWidget()
        self.list_obs.setMaximumHeight(110)
        lay.addWidget(self.list_obs)
        lay.addWidget(_button("Remove selected observation", self.remove_observation))
        return w

    def _obs_input(self):
        return self.draft.inputs.get(self.combo_obs_input.currentText())

    def _input_kind(self, inp):
        if inp.kind == "tf":
            return "tf"
        if not inp.msg_type:  # a loaded file: ask the live graph, else guess from the name
            inp.msg_type = next((t.type for t in self._topics if t.name == inp.topic), "")
        kind = ros_catalog.classify(inp.msg_type) if inp.msg_type else ""
        return kind if kind not in ("", "other") else ("image" if inp.name.startswith("image") else "pointcloud" if inp.name.startswith("pointcloud") else "other")

    def _fill_fields(self):
        self.tree_fields.blockSignals(True)
        self.tree_fields.clear()
        self.label_obs_note.setText("")
        inp = self._obs_input()
        if inp:
            kind = self._input_kind(inp)
            if kind in ("image", "pointcloud"):
                self.label_obs_note.setText("Images and point clouds are not stream values: add this input as a camera / point cloud observation below.")
            else:
                try:
                    fields = [(p, False) for p in ros_catalog.TF_FIELDS] if kind == "tf" else ros_catalog.message_fields(inp.msg_type, self._loader)
                except Exception as e:
                    fields = []
                    self.label_obs_note.setText(f"Fields of '{inp.msg_type or inp.topic}' are not known ({e}). Type the specs by hand.")
                for path, is_array in fields:
                    item = QTreeWidgetItem([path, "array [:]" if is_array else "value"])
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(0, Qt.Unchecked)
                    item.setData(0, Qt.UserRole, ros_catalog.spec_for(inp.name, path, is_array))
                    self.tree_fields.addTopLevelItem(item)
        self.tree_fields.blockSignals(False)

    def _field_toggled(self, item, _col):
        spec = item.data(0, Qt.UserRole)
        specs = [s for s in self.text_specs.toPlainText().splitlines() if s.strip()]
        if item.checkState(0) == Qt.Checked and spec not in specs:
            specs.append(spec)
        elif item.checkState(0) != Qt.Checked and spec in specs:
            specs.remove(spec)
        self.text_specs.setPlainText("\n".join(specs))

    def _specs(self):
        return [s.strip() for s in self.text_specs.toPlainText().splitlines() if s.strip()]

    def _inputs_dict(self):
        return self.draft.to_config()["config"]["input"]

    def _resize_limits(self):
        width = stream_width(self._specs(), self._inputs_dict()) if self.check_normalize.isChecked() else 0
        rows = width if width else (1 if self.check_normalize.isChecked() else 0)
        old = [(self.table_limits.item(r, 0), self.table_limits.item(r, 1)) for r in range(self.table_limits.rowCount())]
        self.table_limits.setRowCount(rows)
        for r in range(rows):
            for c in (0, 1):
                keep = old[r][c].text() if r < len(old) and old[r][c] else ""
                self.table_limits.setItem(r, c, _item(keep, True))
        self.table_limits.setVisible(rows > 0)

    def _limits(self):
        out = []
        for r in range(self.table_limits.rowCount()):
            lo, hi = (self.table_limits.item(r, c).text().strip() for c in (0, 1))
            try:
                out.append({"lower_lim": float(lo), "upper_lim": float(hi)})
            except ValueError:
                raise ValueError(f"limit row {r + 1} needs two numbers (lower, upper)") from None
        return out

    def add_stream(self):
        name, specs = self.edit_stream_name.text().strip(), self._specs()
        if not name or not specs:
            CustomMessageBox.warning("A stream needs a name and at least one spec.")
            return None
        try:
            limits = self._limits() if self.check_normalize.isChecked() else []
        except ValueError as e:
            CustomMessageBox.warning(str(e))
            return None
        stream = StreamDraft(name=name, specs=specs, normalize=self.check_normalize.isChecked(), limits=limits,
                             rotation_format="rpy" if self.check_rpy.isChecked() else "quat", explicit_dict=self.check_normalize.isChecked())
        self.draft.streams[name] = stream
        self.edit_stream_name.clear()
        self.text_specs.clear()
        self.check_normalize.setChecked(False)
        self.check_rpy.setChecked(False)
        self._fill_fields()
        self.refresh_observations()
        return stream

    def add_camera(self):
        inp, key = self._obs_input(), self.edit_camera_key.text().strip()
        kind = self._input_kind(inp) if inp else ""
        if not inp or kind not in ("image", "pointcloud") or not key:
            CustomMessageBox.warning("Select an image or point cloud input and give the observation a name.")
            return
        self.draft.cameras[key] = (inp.name, "images" if kind == "image" else "points")
        self.edit_camera_key.clear()
        self.refresh_observations()

    def remove_observation(self):
        row = self.list_obs.currentRow()
        keys = [("stream", n) for n in self.draft.streams] + [("camera", k) for k in self.draft.cameras]
        if 0 <= row < len(keys):
            kind, key = keys[row]
            self.draft.remove_stream(key) if kind == "stream" else self.draft.cameras.pop(key, None)
            self.refresh_observations()

    def refresh_observations(self):
        names = list(self.draft.inputs)
        if names != [self.combo_obs_input.itemText(i) for i in range(self.combo_obs_input.count())]:
            current = self.combo_obs_input.currentText()
            self.combo_obs_input.blockSignals(True)
            self.combo_obs_input.clear()
            self.combo_obs_input.addItems(names)
            self.combo_obs_input.setCurrentText(current)
            self.combo_obs_input.blockSignals(False)
            self._fill_fields()
        inputs = self._inputs_dict()
        self.list_obs.clear()
        for s in self.draft.streams.values():
            width = stream_width(s.specs, inputs)
            self.list_obs.addItem(f"{s.name}: {width if width is not None else '?'} values  <-  {', '.join(s.specs)}")
        for key, (inp, kind) in self.draft.cameras.items():
            self.list_obs.addItem(f"{key}: {kind[:-1]} of {inp}")

    # URDF limits for the stream being built

    def _fill_limits(self, xml):
        specs = self._specs()
        try:
            if len(specs) != 1 or not specs[0].endswith("[:]"):
                raise ValueError("limits from a URDF need exactly one spec like joint_1.position[:]")
            input_name, field = specs[0][:-3].split(".", 1)
            names = self.draft.inputs[input_name].joint_names
            if not names:
                raise ValueError(f"'{input_name}' has no joint_names: read them on the Inputs step first")
            limits = urdf_limits.limits_for(field, urdf_limits.parse_joint_limits(xml, names), names)
        except (ValueError, KeyError) as e:
            CustomMessageBox.warning(f"Could not fill the limits: {e}")
            return
        self.check_normalize.setChecked(True)
        self._resize_limits()
        for r, lim in enumerate(limits):
            self.table_limits.item(r, 0).setText(f"{lim['lower_lim']:.6g}")
            self.table_limits.item(r, 1).setText(f"{lim['upper_lim']:.6g}")

    def fill_limits_from_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Robot description", self._dialog_dir(), "URDF / xacro (*.urdf *.xacro *.xml)")
        if path:
            from ...review.robot_description_resolver import RobotDescriptionError, resolve_from_file

            try:
                self._fill_limits(resolve_from_file(path))
            except RobotDescriptionError as e:
                CustomMessageBox.warning(str(e))

    def fill_limits_from_topic(self):
        def done(msg):
            if msg is None:
                CustomMessageBox.warning(f"No /robot_description received within {_PEEK_SECONDS:.0f} s. Is robot_state_publisher running?")
            else:
                self._fill_limits(msg.data)

        self.peek("/robot_description", "std_msgs/msg/String", done, latched=True)

    # ------------------------------------------------------------------ 4 Actions

    def _build_actions(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Tick the streams that make up the robot action, in the order they are concatenated (the number is the order). "
                             "Delta: the action is the change between two steps instead of the value."))
        self.table_actions = _table(["Stream", "Action order", "Delta", "Values"], select_rows=False)
        self.table_actions.itemChanged.connect(self._action_toggled)
        lay.addWidget(self.table_actions, 1)
        self.label_action_width = QLabel("")
        lay.addWidget(self.label_action_width)
        return w

    def refresh_actions(self):
        table, inputs = self.table_actions, self._inputs_dict()
        table.blockSignals(True)
        table.setRowCount(len(self.draft.streams))
        total = 0
        for r, s in enumerate(self.draft.streams.values()):
            width = stream_width(s.specs, inputs)
            total += width or 0 if s.name in self.draft.actions else 0
            on = QTableWidgetItem(str(self.draft.actions.index(s.name) + 1) if s.name in self.draft.actions else "")
            on.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            on.setCheckState(Qt.Checked if s.name in self.draft.actions else Qt.Unchecked)
            delta = QTableWidgetItem()
            delta.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            delta.setCheckState(Qt.Checked if s.name in self.draft.delta_actions else Qt.Unchecked)
            for c, item in enumerate((_item(s.name), on, delta, _item(width if width is not None else "?"))):
                table.setItem(r, c, item)
        table.blockSignals(False)
        self.label_action_width.setText(f"Action vector: {total} values" if self.draft.actions else "No action selected: the dataset has observations only.")

    def _action_toggled(self, item):
        name = self.table_actions.item(item.row(), 0).text()
        checked = item.checkState() == Qt.Checked
        if item.column() == 1:
            if checked and name not in self.draft.actions:
                self.draft.actions.append(name)
            elif not checked:
                self.draft.actions = [a for a in self.draft.actions if a != name]
                self.draft.delta_actions = [a for a in self.draft.delta_actions if a != name]
        elif item.column() == 2:
            if checked and name not in self.draft.actions:
                self.draft.actions.append(name)
            self.draft.delta_actions = [a for a in self.draft.delta_actions if a != name] + ([name] if checked else [])
        self.refresh_actions()

    # ------------------------------------------------------------------ 5 Recording

    def _build_recording(self):
        w = QWidget()
        form = QFormLayout(w)
        self.edit_dataset = QLineEdit()
        self.spin_freq = QDoubleSpinBox()
        self.spin_freq.setRange(0.1, 1000)
        self.spin_buffer = QDoubleSpinBox()
        self.spin_buffer.setRange(0, 1e6)
        self.spin_buffer.setSpecialValueText("default")
        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setRange(0, 10)
        self.spin_tol.setDecimals(3)
        self.spin_tol.setSpecialValueText("default")
        self.combo_sync = QComboBox()
        self.combo_sync.addItems(["", "soft", "strict"])
        form.addRow("Dataset name", self.edit_dataset)
        form.addRow("Recording frequency (Hz)", self.spin_freq)
        form.addRow("Buffer limit (MB)", self.spin_buffer)
        form.addRow("Sync tolerance (s)", self.spin_tol)
        form.addRow("Sync mode", self.combo_sync)
        self.edit_dataset.textChanged.connect(lambda t: setattr(self.draft, "dataset_name", t.strip()))
        self.spin_freq.valueChanged.connect(lambda v: setattr(self.draft, "recording_frequency", v))
        self.spin_buffer.valueChanged.connect(lambda v: setattr(self.draft, "max_buffer_mb", v or None))
        self.spin_tol.valueChanged.connect(lambda v: setattr(self.draft, "sync_tolerance_sec", v or None))
        self.combo_sync.currentTextChanged.connect(lambda t: setattr(self.draft, "sync_mode", t))
        return w

    def refresh_recording(self):
        for widget in (self.edit_dataset, self.spin_freq, self.spin_buffer, self.spin_tol, self.combo_sync):
            widget.blockSignals(True)
        self.edit_dataset.setText(self.draft.dataset_name)
        self.spin_freq.setValue(self.draft.recording_frequency or 30)
        self.spin_buffer.setValue(self.draft.max_buffer_mb or 0)
        self.spin_tol.setValue(self.draft.sync_tolerance_sec or 0)
        self.combo_sync.setCurrentText(self.draft.sync_mode)
        for widget in (self.edit_dataset, self.spin_freq, self.spin_buffer, self.spin_tol, self.combo_sync):
            widget.blockSignals(False)

    # ------------------------------------------------------------------ 6 Review

    def _build_review(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.text_yaml = QPlainTextEdit()
        self.text_yaml.setReadOnly(True)
        lay.addWidget(self.text_yaml, 2)
        self.list_problems = QListWidget()
        self.list_problems.setMaximumHeight(140)
        lay.addWidget(self.list_problems)
        row = QHBoxLayout()
        self.btn_save = _button("Save as...", self.save_as)
        self.btn_use = _button("Save and use now", self.save_and_use)
        row.addStretch(1)
        row.addWidget(self.btn_save)
        row.addWidget(self.btn_use)
        lay.addLayout(row)
        return w

    def problems(self):
        """Everything wrong with the draft: validator rules plus the stream-spec syntax errors of the recorder's own parser."""
        data = self.draft.to_config()
        found = check_config(data)
        try:
            found += [Problem(ERROR, e) for e in stream_build_errors(data)]
        except Exception as e:  # a malformed draft must show up as a problem, not crash the page
            found.append(Problem(ERROR, f"could not check the stream specs: {e}"))
        if not self.draft.inputs:
            found.append(Problem(ERROR, "no input yet: add one on the Sources step"))
        return found

    def refresh_review(self):
        self.text_yaml.setPlainText(self.draft.to_yaml())
        found = self.problems()
        self.list_problems.clear()
        self.list_problems.addItems([str(p) for p in found] or ["No problems found."])
        ok = not has_errors(found)
        self.btn_save.setEnabled(ok)
        self.btn_use.setEnabled(ok)

    def _write(self, path):
        try:
            with open(path, "w") as f:
                f.write(self.draft.to_yaml())
        except OSError as e:
            CustomMessageBox.error(f"Could not save {path}:\n{e}")
            return False
        return True

    def save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save config", self._dialog_dir(), "YAML files (*.yaml *.yml)")
        if path and self._write(path):
            CustomMessageBox.info(f"Saved {path}")
            return path
        return None

    def save_and_use(self):
        path = self.save_as()
        if path:
            self._use_config(path)

    # ------------------------------------------------------------------ whole page

    def refresh(self):
        self.refresh_inputs()
        self.refresh_observations()
        self.refresh_actions()
        self.refresh_recording()
        self.refresh_review()

    def reset(self):
        self.draft = ConfigDraft()
        self.refresh()

    def load_current(self):
        data = self._current_config()
        if not data:
            CustomMessageBox.warning("No config is loaded. Load one on the 'Load / edit file' tab first.")
            return
        self.draft = ConfigDraft.from_config(data)
        self.refresh()
