import copy
import os

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QKeySequence
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from datetime import datetime

# Import utils
from ..config.yaml_parser import YamlParser
from ..postprocess.yaml_driven_converter import YAMLDrivenConverter
from ..settings.config_manager import config
from relay_d.utils import user_settings
from relay_d.utils.coloring_logger import logger
from ...utils.background import run_in_background
from ...utils.custom_message_box import CustomMessageBox
from relay_d.acquisition.convert_cli import collect_inputs
from ...utils.default_configs import generate_default_configs
from ..postprocess.postprocess_config_loader import PostprocessConfigLoader
from ..postprocess.postprocess_options import dataset_output_paths
from ..postprocess.offline_config import load_offline_config, write_dataset_files
from ..postprocess.postprocess_options import PostprocessOptions

PRECHECK_COLUMNS = ("Demo", "Samples", "Rate (Hz)", "Recorded with", "Result")
# Per-item colours cannot come from the QSS; the mid-tones the Review page uses read on both themes.
_OK_COLOR, _FAIL_COLOR = "#16A34A", "#DC2626"

class _PrecheckTable(QTableWidget):
    """The pre-check table; the Delete key removes the selected recordings from the conversion."""

    delete_pressed = pyqtSignal()

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Delete):
            self.delete_pressed.emit()
        else:
            super().keyPressEvent(event)


class PostProcessPage(QWidget):
    def __init__(self, ui_instance, yaml_parser: YamlParser = None, settings_page=None):
        super().__init__()
        self.ui = ui_instance
        self.yaml_parser = yaml_parser  # Shared YamlParser from ConfigPage
        # Config loaded on this page only (no live-ROS check, never shared with the
        # Record page); takes precedence over the shared parser while set.
        self._local_parser = None
        # Format options (postprocess_config.yaml): each config load resets them to the one next
        # to that config (auto-detected), else the defaults; a file the user picks applies until the next load.
        self._options_explicit = False
        self._options_error = None  # a broken postprocess_config.yaml next to the config: refuse to convert, as the CLI and AppAPI do
        self.settings_page = settings_page

        self.h5_files = []
        self.output_format = ""
        self._precheck_gen = 0  # bumped by every refresh: only the newest pre-check result is shown
        self._report = None  # last pre-check report (None: not run / running)

        self.yaml_converter = YAMLDrivenConverter(PostprocessConfigLoader())
        self.format_buttons = {}
        self.format_names = {
            "btn_format_robomimic": "robomimic",
            "btn_format_lerobot": "lerobot",
        }

        # If a yaml_parser was passed at construction and already has data loaded,
        # push its data_config into the converter immediately.
        if self.yaml_parser is not None:
            self._sync_data_config_from_parser()

        self._build_precheck_widgets()
        self.setup_connections()
        self.setup_format_buttons()
        self._update_config_label()
        self._update_options_label()

    # ------------------------------------------------------------------
    # Config signal wiring  (mirrors RecordPage.connect_to_config_signals)
    # ------------------------------------------------------------------

    def connect_to_config_signals(self, config_signal_manager):
        """Connect to ConfigPage signals - call this from app.py after construction."""
        config_signal_manager.config_loaded.connect(self._on_config_loaded)
        config_signal_manager.config_cleared.connect(self._on_config_cleared)
        config_signal_manager.config_saved.connect(self._on_config_saved)

    def _on_config_loaded(self, file_path: str):
        """Handle config loaded signal - reload the bridge map from the new file."""
        logger.info(f"PostProcessPage: config loaded from {file_path}")
        if self.yaml_parser is not None:
            self.yaml_parser.load_yaml_file(file_path)
        if self._local_parser is not None:
            logger.info(
                "PostProcessPage: keeping the config loaded on this page "
                "(use 'Use Config page' to switch back)."
            )
            return
        self._sync_data_config_from_parser()
        self._update_config_label()
        self._auto_load_sibling_options(file_path)

    def _on_config_cleared(self):
        """Handle config cleared signal - wipe the bridge map."""
        logger.info("PostProcessPage: config cleared")
        if self._local_parser is None:
            self.yaml_converter.data_config = {}
        self._update_config_label()

    def _on_config_saved(self, file_path: str):
        """Handle config saved signal - re-sync in case output paths changed."""
        logger.info(f"PostProcessPage: config saved, re-syncing from {file_path}")
        if self.yaml_parser is not None:
            self.yaml_parser.load_yaml_file(file_path)
        if self._local_parser is None:
            self._sync_data_config_from_parser()
            self.refresh_precheck()

    # ------------------------------------------------------------------
    # Config loaded on this page (works without any live ROS topics)
    # ------------------------------------------------------------------

    def _active_parser(self):
        """The parser conversions use: this page's own config if loaded, else the shared one."""
        return self._local_parser if self._local_parser is not None else self.yaml_parser

    def _update_config_label(self):
        label = self.ui.label_postprocess_config
        if self._local_parser is not None:
            name = os.path.basename(self._local_parser.get_current_config_path() or "")
            label.setText(f"{name} (post-process only)")
        elif self.yaml_parser is not None and self.yaml_parser.yaml_data:
            label.setText("from Config page")
        else:
            label.setText("No config loaded")
        self.ui.btn_clear_postprocess_config.setEnabled(self._local_parser is not None)
        self.refresh_precheck()  # every config change passes here

    def _load_postprocess_config(self):
        """Load a config for post-processing only: no topic/TF availability check,
        no containers, and the Record page is never touched."""
        try:
            path, _ = QFileDialog.getOpenFileName(
                None, "Open Config File", config.config_dialog_dir(), "YAML files (*.yaml *.yml)"
            )
        except Exception as e:
            logger.error(f"An error occured while opening file: {e}")
            return
        if not path:
            return
        self.load_postprocess_config_from_path(path)

    def load_postprocess_config_from_path(self, path: str) -> bool:
        parser, problems = load_offline_config(path)
        if parser is None:
            CustomMessageBox.error(
                "Config rejected:\n" + "\n".join(f"  {p}" for p in problems)
            )
            return False
        warnings = [p for p in problems if p.startswith("warning:")]
        if warnings:
            CustomMessageBox.warning("\n".join(w[len("warning: "):] for w in warnings))
        self._local_parser = parser
        self._sync_data_config_from_parser()
        self._update_config_label()
        self._auto_load_sibling_options(path)
        logger.info(f"PostProcessPage: using post-process-only config {path}")
        return True

    def _clear_postprocess_config(self):
        """Drop the page's own config and go back to the Config page's one (if any)."""
        self._local_parser = None
        if self.yaml_parser is not None and self.yaml_parser.yaml_data:
            self._sync_data_config_from_parser()
            self._auto_load_sibling_options(self.yaml_parser.get_current_config_path())
        else:
            self.yaml_converter.data_config = {}
        self._update_config_label()

    # ------------------------------------------------------------------
    # Format options (postprocess_config.yaml)
    # ------------------------------------------------------------------

    def _update_options_label(self):
        source = self.yaml_converter.options.source
        if source:
            kind = "" if self._options_explicit else " (auto-detected)"
            self.ui.label_postprocess_options.setText(f"{os.path.basename(source)}{kind}")
        else:
            self.ui.label_postprocess_options.setText("defaults")
        self.ui.btn_clear_postprocess_options.setEnabled(bool(source))
        self.refresh_precheck()  # every options change passes here

    def _load_postprocess_options(self):
        try:
            path, _ = QFileDialog.getOpenFileName(
                None, "Open Postprocess Options", config.config_dialog_dir(), "YAML files (*.yaml *.yml)"
            )
        except Exception as e:
            logger.error(f"An error occured while opening file: {e}")
            return
        if path:
            self.load_postprocess_options_from_path(path)

    def load_postprocess_options_from_path(self, path: str, explicit: bool = True) -> bool:
        options, problems = PostprocessOptions.load(path)
        if options is None:
            if explicit:
                CustomMessageBox.error(
                    "Options rejected:\n" + "\n".join(f"  {p}" for p in problems)
                )
            else:
                self._options_error = f"{path} is not usable:\n" + "\n".join(f"  {p}" for p in problems)
                logger.error(self._options_error)
            return False
        warnings = [p for p in problems if p.startswith("warning:")]
        if warnings and explicit:
            CustomMessageBox.warning("\n".join(w[len("warning: "):] for w in warnings))
        self.yaml_converter.options = options
        self._options_explicit = explicit
        self._options_error = None
        self._update_options_label()
        logger.info(f"PostProcessPage: using format options from {path}")
        return True

    def _auto_load_sibling_options(self, config_path: str):
        """A loaded config brings its options: the postprocess_config.yaml next to it, else the defaults
        (as relay-d-convert and AppAPI)."""
        self._clear_postprocess_options()
        sibling = PostprocessOptions.find_sibling(config_path)
        if sibling:
            self.load_postprocess_options_from_path(sibling, explicit=False)

    def _clear_postprocess_options(self):
        self.yaml_converter.options = PostprocessOptions()
        self._options_explicit = False
        self._options_error = None
        self._update_options_label()

    # ------------------------------------------------------------------
    # Bridge map sync
    # ------------------------------------------------------------------

    def _sync_data_config_from_parser(self):
        """
        Pull the `config` sub-dict out of the shared YamlParser and push it
        into both YAMLDrivenConverter and DatasetPostprocessor.

        data_config.yaml structure:
            config:
              input:  { ... }
              output: { robot0_eef_pos: robot0_eef/translations, ... }
        """
        parser = self._active_parser()
        if parser is None:
            logger.warning(
                "PostProcessPage: no yaml_parser available, cannot sync bridge map"
            )
            return

        raw = parser.yaml_data  # Full parsed YAML dict

        if not raw:
            logger.warning("PostProcessPage: yaml_parser has no data yet")
            return

        # The converter expects the `config` sub-dict (same level as output/metadata)
        data_config = raw.get("config", raw)

        bridge_entries = len(data_config.get("output", {}))
        logger.info(
            f"PostProcessPage: syncing bridge map - {bridge_entries} output entries"
        )

        # Push into YAMLDrivenConverter
        self.yaml_converter.data_config = data_config

    # ------------------------------------------------------------------
    # UI wiring
    # ------------------------------------------------------------------

    def setup_connections(self):
        """Setup all button connections"""
        ui = self.ui
        self.btn_convert.clicked.connect(self.convert_dataset)
        self.btn_add_files.clicked.connect(self._select_files_to_convert)
        self.btn_add_folder.clicked.connect(self._select_folder_to_convert)
        self.btn_clear_files.clicked.connect(lambda: self.set_files([]))
        self.btn_remove_files.clicked.connect(self._remove_selected_files)
        self.table_precheck.delete_pressed.connect(self._remove_selected_files)
        self.table_precheck.itemSelectionChanged.connect(
            lambda: self.btn_remove_files.setEnabled(bool(self._selected_files()))
        )
        ui.btn_generate_postprocess_configs.clicked.connect(lambda: generate_default_configs(ui))
        ui.btn_load_postprocess_config.clicked.connect(self._load_postprocess_config)
        ui.btn_clear_postprocess_config.clicked.connect(self._clear_postprocess_config)
        ui.btn_load_postprocess_options.clicked.connect(self._load_postprocess_options)
        ui.btn_clear_postprocess_options.clicked.connect(self._clear_postprocess_options)

    def setup_format_buttons(self):
        """Setup format selection buttons defined in .ui file"""
        # Create button group for exclusive selection
        self.format_button_group = QButtonGroup()
        self.format_button_group.setExclusive(False)

        for button_name, format_name in self.format_names.items():
            button = getattr(self.ui, button_name, None)
            if button is None:
                logger.warning(f"Button {button_name} not found in UI")
                continue
            button.setCheckable(True)
            button.clicked.connect(lambda checked, name=format_name: self._select_output_format(name))
            self.format_button_group.addButton(button)
            self.format_buttons[format_name] = button
            if format_name == user_settings.get("postprocess_default_format"):  # setting
                button.setChecked(True)
                self.output_format = format_name

    def _select_output_format(self, format_name):
        """Single-choice toggle: select `format_name`, or deselect it if it already is.

        The buttons live in a non-exclusive QButtonGroup (so the selected one can be
        clicked again to deselect), which makes Qt ignore their `autoExclusive` flag;
        the checked state of every button is therefore set explicitly here.
        """
        if format_name not in self.format_buttons:
            return

        if self.output_format == format_name:
            self.output_format = ""
            self._sync_format_buttons()
            logger.info(f"Deselected format: {format_name}")
        else:
            self.output_format = format_name
            self._sync_format_buttons()
            logger.info(f"Selected output format: {format_name}")
        self.refresh_precheck()

    def _sync_format_buttons(self):
        """Check exactly the button of the selected format (none if nothing is selected)."""
        for name, button in self.format_buttons.items():
            blocked = button.blockSignals(True)
            button.setChecked(name == self.output_format)
            button.blockSignals(blocked)

    # ------------------------------------------------------------------
    # File selection
    # ------------------------------------------------------------------

    def _recordings_dir(self):
        """Where recordings are saved (Settings page), for the file dialogs; else home."""
        save_folder = self.settings_page.get_hdf5_save_folder() if self.settings_page else ""
        return os.path.join(save_folder, "recorded_data") if save_folder else os.path.expanduser("~")

    def _select_files_to_convert(self):
        fn, _ = QFileDialog.getOpenFileNames(
            parent=None, caption="Add recordings", filter="H5 files (*.h5)", directory=self._recordings_dir()
        )
        if fn:
            self.add_inputs(fn)

    def _select_folder_to_convert(self):
        folder = QFileDialog.getExistingDirectory(None, "Add every recording under a folder", self._recordings_dir())
        if folder:
            self.add_inputs([folder])

    def add_inputs(self, paths):
        """Add files and folders (searched recursively for *.h5, as relay-d-convert --input) to the recordings."""
        try:
            self.set_files(collect_inputs(self.h5_files + list(paths)))
        except FileNotFoundError as e:
            CustomMessageBox.error(f"Not found: {e}")

    def _selected_files(self):
        rows = {i.row() for i in self.table_precheck.selectedIndexes()}
        return {self.table_precheck.item(r, 0).data(Qt.UserRole) for r in rows if self.table_precheck.item(r, 0)}

    def _remove_selected_files(self):
        """Leave the selected rows' files out of the conversion (a file holding several demos goes as a whole)."""
        selected = self._selected_files()
        if selected:
            self.set_files([f for f in self.h5_files if f not in selected])

    def set_files(self, files):
        self.h5_files = list(files)
        logger.info(f"Recordings to convert: {len(self.h5_files)} files")
        self.refresh_precheck()

    # ------------------------------------------------------------------
    # Pre-check (what conversion would do, without writing anything)
    # ------------------------------------------------------------------

    def _build_precheck_widgets(self):
        """Recordings row, pre-check table, output preview and Convert, under the .ui's config/options/format rows."""
        layout = self.ui.verticalLayout_9

        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(QLabel("Recordings:"))
        self.label_recordings = QLabel()
        row.addWidget(self.label_recordings, 1)
        self.btn_add_files = QPushButton("Add files...")
        self.btn_add_folder = QPushButton("Add folder...")
        self.btn_remove_files = QPushButton("Remove selected")
        self.btn_remove_files.setToolTip("Leave the selected recordings out of this conversion (Delete key). Nothing is deleted from disk.")
        self.btn_remove_files.setEnabled(False)
        self.btn_clear_files = QPushButton("Clear")
        for b in (self.btn_add_files, self.btn_add_folder, self.btn_remove_files, self.btn_clear_files):
            b.setMinimumHeight(36)
            row.addWidget(b)
        layout.addLayout(row)

        self.label_precheck_error = QLabel()
        self.label_precheck_error.setObjectName("precheckError")
        self.label_precheck_error.setWordWrap(True)
        self.label_precheck_error.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.label_precheck_error.hide()
        layout.addWidget(self.label_precheck_error)

        self.table_precheck = _PrecheckTable(0, len(PRECHECK_COLUMNS))
        self.table_precheck.setHorizontalHeaderLabels(PRECHECK_COLUMNS)
        self.table_precheck.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_precheck.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_precheck.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table_precheck.verticalHeader().hide()
        header = self.table_precheck.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(len(PRECHECK_COLUMNS) - 1, QHeaderView.Stretch)  # long reasons wrap instead of being cut
        self.table_precheck.setWordWrap(True)
        precheck_box = QGroupBox("Pre-check")
        QVBoxLayout(precheck_box).addWidget(self.table_precheck)

        self.output_box = QGroupBox("Output")
        self.txt_output = QPlainTextEdit()
        self.txt_output.setReadOnly(True)
        self.txt_output.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.txt_output.setFont(QFont("monospace"))
        QVBoxLayout(self.output_box).addWidget(self.txt_output)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 14, 0, 0)
        self.label_summary = QLabel()
        self.label_summary.setWordWrap(True)
        self.label_convert_reason = QLabel()
        self.label_convert_reason.setObjectName("pageHint")
        self.label_convert_reason.setWordWrap(True)
        self.btn_convert = QPushButton("Convert")
        self.btn_convert.setMinimumHeight(38)
        bold = self.btn_convert.font()
        bold.setBold(True)
        self.btn_convert.setFont(bold)
        side_layout.addWidget(self.label_summary)
        side_layout.addWidget(self.label_convert_reason)
        side_layout.addStretch(1)
        side_layout.addWidget(self.btn_convert)
        side.setMinimumWidth(180)

        bottom = QSplitter(Qt.Horizontal)
        bottom.addWidget(self.output_box)
        bottom.addWidget(side)
        bottom.setStretchFactor(0, 4)
        bottom.setStretchFactor(1, 1)
        split = QSplitter(Qt.Vertical)
        split.addWidget(precheck_box)
        split.addWidget(bottom)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setChildrenCollapsible(False)
        layout.addWidget(split, 1)

    def _blocked_before_check(self):
        """Why there is nothing to pre-check yet ("" when there is)."""
        if not self.output_format:
            return "Select an output format."
        if not self.h5_files:
            return "Add recordings to convert."
        if self._options_error:
            return self._options_error
        if not self.yaml_converter.data_config.get("output"):
            return "No configuration loaded: load one on the Config page, or use 'Load Config...' here."
        return ""

    def refresh_precheck(self):
        """Re-run the pre-check in the background; a refresh started later replaces the result of an earlier one."""
        self._precheck_gen += 1
        gen = self._precheck_gen
        self._report = None
        if self._blocked_before_check():
            self._show_report()
            return
        # The pre-check sets per-batch state on the converter: give it its own copy, so a running
        # check never mixes with another one or with a conversion.
        conv = copy.copy(self.yaml_converter)
        conv.data_config = copy.deepcopy(self.yaml_converter.data_config)
        fmt, files = self.output_format, list(self.h5_files)
        self._show_report(checking=True)
        run_in_background(
            self.ui.page_data, lambda: conv.precheck(fmt, files), lambda r, e: self._on_precheck_done(gen, r, e)
        )

    def _on_precheck_done(self, gen, report, error):
        if gen != self._precheck_gen:
            return  # stale: inputs changed while it ran
        if error is not None:
            logger.error(f"Pre-check failed: {error}")
            report = {"error": f"Pre-check failed: {error}", "demos": [], "output": [], "converted": 0, "total": 0}
        self._report = report
        self._show_report()

    def _convert_block_reason(self, checking=False):
        """Why Convert is disabled ("" when it is enabled)."""
        if reason := self._blocked_before_check():
            return reason
        if checking or self._report is None:
            return "Checking the recordings..."
        if self._report.get("error"):
            return self._report["error"]
        if not self._report.get("converted"):
            return "No demo would be converted: see the Result column."
        return ""

    def _show_report(self, checking=False):
        report = self._report or {}
        demos = report.get("demos") or []
        n_files = len(self.h5_files)
        files = f"{n_files} file{'s' * (n_files != 1)}"
        demos_text = f"{report.get('total')} demo{'s' * (report.get('total') != 1)}"
        self.label_recordings.setText(f"{demos_text} in {files}" if "total" in report else files)

        error = report.get("error")
        self.label_precheck_error.setText(error or "")
        self.label_precheck_error.setVisible(bool(error))

        self.table_precheck.setRowCount(len(demos))
        for row, d in enumerate(demos):
            name = d.get("demo", "")
            rate, rec_hz = d.get("rate_hz"), d.get("recorded_hz")
            cells = (
                name,
                str(d.get("samples", "")),
                f"{rate:.1f}" if rate else "-",
                (d.get("recorded_with") or "-") + (f" ({rec_hz:g} Hz)" if rec_hz else ""),
                "\u2714 converts" if d.get("ok") else f"\u2716 {d.get('reason', '')}",
            )
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(d.get("file", "") if col == 0 else text)
                if col == 0:
                    item.setData(Qt.UserRole, d.get("file", ""))  # a row stands for its recording file
                if col == len(cells) - 1:
                    item.setForeground(QBrush(QColor(_OK_COLOR if d.get("ok") else _FAIL_COLOR)))
                self.table_precheck.setItem(row, col, item)
        self.table_precheck.resizeRowsToContents()

        fmt = {"robomimic": "Robomimic", "lerobot": "LeRobot"}.get(self.output_format, self.output_format)
        self.output_box.setTitle(f"Output ({fmt})" if fmt else "Output")
        output = report.get("output") or []
        width = max((len(str(name)) for name, _ in output), default=0)
        self.txt_output.setPlainText("\n".join(f"{str(name):<{width}}  {detail}" for name, detail in output))
        self.label_summary.setText(
            f"{report['converted']} of {demos_text} will be converted" if "total" in report else ""
        )

        reason = self._convert_block_reason(checking)
        converted = report.get("converted") or 0
        self.btn_convert.setEnabled(not reason)
        self.btn_convert.setText(f"Convert {converted} demo{'s' * (converted != 1)}" if not reason else "Convert")
        self.btn_convert.setToolTip(reason)
        self.label_convert_reason.setText("See the error above." if error and reason == error else reason)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def convert_dataset(self):
        self._sync_data_config_from_parser()  # always use freshest config before converting

        logger.info(
            f"Convert called. output_format: {self.output_format}, h5_files: {self.h5_files}"
        )

        if not self.output_format:
            CustomMessageBox.warning("Please select an output format first.")
            return

        if not self.h5_files or len(self.h5_files) == 0:
            CustomMessageBox.warning("Please select at least one H5 file to convert.")
            return

        if self._options_error:
            CustomMessageBox.error(self._options_error + "\n\nFix it, or load another options file with 'Load Options...'.")
            return

        # Guard: ensure bridge map is populated before converting
        bridge_entries = len(self.yaml_converter.data_config.get("output", {}))
        if bridge_entries == 0:
            logger.warning("Bridge map is empty - no config loaded yet.")
            CustomMessageBox.warning(
                "No configuration loaded.\n\n"
                "Load a config on the Config page, or use 'Load Config...' on this page.")
            return

        logger.info(f"Converting dataset to {self.output_format} format.")

        # The Recordings folder setting, else ~/lfd_recordings (the recorder's and AppAPI's default too)
        save_folder = self.settings_page.get_hdf5_save_folder() if self.settings_page else ""
        if not save_folder:
            save_folder = os.path.expanduser("~/lfd_recordings")
            logger.warning(f"The recordings folder is not set (Settings page). Defaulting to {save_folder}")
        output_dir, combined_output_path = dataset_output_paths(save_folder, self.output_format, datetime.now())
        os.makedirs(output_dir, exist_ok=True)

        success = self.yaml_converter.convert_multiple_files(
            self.output_format,
            self.h5_files,
            combined_output_path,
        )

        # Keep the configs used next to the dataset, plus the dispatcher files.
        post_error = None
        if success:
            try:
                write_dataset_files(output_dir, self._active_parser(), self.yaml_converter.options.source)
            except Exception as e:
                post_error = str(e) or type(e).__name__
                logger.error(
                    f"Failed to write the config copies / action_dispatcher files: {e}"
                )

        if success:
            logger.info("Dataset successfully converted and validated!")
            if post_error:
                CustomMessageBox.warning(
                    "The dataset was converted, but some extra files (config copies / "
                    f"action_dispatcher files) could not be written:\n\n{post_error}"
                )
            CustomMessageBox.info(
                f"Dataset converted to {self.output_format} format.\n\n{self.yaml_converter.conversion_summary()}"
                f"\n\nOutput file: {combined_output_path}",
            )
        else:
            logger.error("Dataset conversion failed")
            reason = self.yaml_converter.last_error
            CustomMessageBox.error(
                "Failed to convert dataset." + (f"\n\n{reason}" if reason else "")
            )

        return
