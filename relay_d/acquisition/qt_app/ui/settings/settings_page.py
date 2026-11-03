"""Settings tab, generated from the table in ``relay_d.utils.user_settings``.

Every setting is a row of that table (type, default, tooltip, live / restart). The same table feeds the
command-line tools, so a value saved here is what ``relay-d-train``, ``lfd-inference-node`` and the other
tools use as their default.
"""

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QWidget,
)

from relay_d.utils import user_settings
from relay_d.utils.coloring_logger import logger
from ...utils.custom_message_box import CustomMessageBox
from ..page_style import add_page_header, page_layout
from ..settings.config_manager import config


class SettingsPage:
    def __init__(self, ui_instance):
        self.ui = ui_instance
        self.widgets = {}  # key -> (widget, getter, setter)
        self._tab_groups = []
        self._build_ui()
        self.apply_settings_to_ui(config.settings)
        self._shown = dict(config.settings)  # what the widgets last showed as saved
        config.settings_changed.connect(self._on_settings_changed)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        root = page_layout(self.ui.page_settings)
        add_page_header(root, "Settings")

        self.tabs = QTabWidget()
        for group in user_settings.GROUPS:
            settings = [s for s in user_settings.SETTINGS if s.group == group]
            if not settings:
                continue
            form_widget = QWidget()
            form = QFormLayout(form_widget)
            for s in settings:
                label = QLabel(s.label + (" (restart)" if s.applies == user_settings.RESTART else ""))
                label.setToolTip(s.tooltip)
                form.addRow(label, self._make_editor(s))
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(form_widget)
            self.tabs.addTab(scroll, group)
            self._tab_groups.append(group)
        root.addWidget(self.tabs, 1)

        row = QHBoxLayout()
        self.btn_save_settings = QPushButton("Save")
        self.btn_save_settings.clicked.connect(lambda _checked=False: self.save_settings_from_ui())  # clicked passes `checked`, not keys
        self.btn_reset_tab = QPushButton("Reset this tab")
        self.btn_reset_tab.clicked.connect(lambda: self.reset_to_default(self._tab_groups[self.tabs.currentIndex()]))
        self.btn_reset_settings = QPushButton("Reset all")
        self.btn_reset_settings.clicked.connect(lambda: self.reset_to_default(None))
        for b in (self.btn_save_settings, self.btn_reset_tab, self.btn_reset_settings):
            row.addWidget(b)
        row.addStretch(1)
        root.addLayout(row)

        self.label_status = QLabel(f"Settings file: {config.settings_file}")
        self.label_status.setWordWrap(True)
        root.addWidget(self.label_status)

    def _make_editor(self, s):
        """Editor widget for one setting; registers it in ``self.widgets``."""
        if s.kind == "bool":
            w = QCheckBox()
            get, put = w.isChecked, w.setChecked
        elif s.kind == "int":
            w = QSpinBox()
            w.setRange(int(s.minimum if s.minimum is not None else -2_000_000_000),
                       int(s.maximum if s.maximum is not None else 2_000_000_000))
            get, put = w.value, lambda v, w=w: w.setValue(int(v))
        elif s.kind == "float":
            w = QDoubleSpinBox()
            w.setDecimals(3)
            w.setSingleStep(0.1)
            w.setRange(s.minimum if s.minimum is not None else -1e9, s.maximum if s.maximum is not None else 1e9)
            get, put = w.value, lambda v, w=w: w.setValue(float(v))
        elif s.kind == "choice":
            w = QComboBox()
            w.addItems(list(s.choices))
            get, put = w.currentText, w.setCurrentText
        else:  # text, folder, file
            w = QLineEdit()
            get, put = (lambda w=w: w.text().strip()), w.setText
        w.setToolTip(s.tooltip)
        self.widgets[s.key] = (w, get, put)
        if s.kind not in ("folder", "file"):
            return w
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda _=False, s=s, w=w: self._browse(s, w))
        lay.addWidget(w, 1)
        lay.addWidget(btn)
        return box

    def _browse(self, s, line_edit):
        if s.kind == "folder":
            path = QFileDialog.getExistingDirectory(self.ui, s.label, line_edit.text())
        else:
            path, _ = QFileDialog.getOpenFileName(self.ui, s.label, line_edit.text())
        if path:
            line_edit.setText(path)

    # ------------------------------------------------------------------ values

    def apply_settings_to_ui(self, values):
        for key, (_w, _get, put) in self.widgets.items():
            put(values.get(key, user_settings.BY_KEY[key].default))

    def _on_settings_changed(self, values):
        """Another page saved a setting: show it, so the next Save here does not write the old value back.
        Unsaved edits to other settings stay as they are."""
        try:
            for key, (_w, _get, put) in self.widgets.items():
                if key in values and values[key] != self._shown.get(key):
                    put(values[key])
        except RuntimeError:  # the page's widgets were deleted (a window closed)
            return
        self._shown = dict(values)

    def values_from_ui(self):
        return {key: get() for key, (_w, get, _put) in self.widgets.items()}

    def load_settings(self):
        """Show the saved values and tell listeners (called once at start so they pick them up)."""
        config.settings = user_settings.reload()
        self.apply_settings_to_ui(config.settings)
        config.settings_changed.emit(config.settings)

    def save_settings_from_ui(self, keys=None):
        """Save the values on screen (only ``keys`` if given) and report what needs a restart."""
        old = dict(config.settings)
        values = self.values_from_ui()
        try:
            config.save_settings({k: values[k] for k in keys} if keys is not None else values)
        except (OSError, ValueError) as e:
            logger.error(f"Error saving settings: {e}")
            CustomMessageBox.error(f"Failed to save settings: {e}")
            return False
        restart = user_settings.changed_restart_keys(old, config.settings)
        note = ""
        if restart:
            names = ", ".join(user_settings.BY_KEY[k].label for k in restart)
            note = f" Restart Relay-D for: {names}."
        self.label_status.setText(f"Saved to {config.settings_file}.{note}")
        logger.info("Settings saved")
        return True

    def reset_to_default(self, group=None):
        what = f"the '{group}' settings" if group else "all settings"
        if not CustomMessageBox.question(f"Reset {what} to their defaults?", title="Confirm Reset"):
            return False
        defaults = {s.key: s.default for s in user_settings.SETTINGS if group is None or s.group == group}
        for key, value in defaults.items():  # only this group's widgets: edits on other tabs stay unsaved, untouched
            self.widgets[key][2](value)
        return self.save_settings_from_ui(keys=list(defaults))

    # Used by other pages
    def get_hdf5_save_folder(self):
        return config.get("hdf5_save_folder", "")
