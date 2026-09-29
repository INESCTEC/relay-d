import os
import sys
import tkinter as tk

from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QLabel,
    QTextEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QButtonGroup,
    QScrollArea,
)
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QFileDialog
from tkinter.filedialog import askopenfilename
from datetime import datetime
from relay_d.dispatch import DispatcherHelpers

# Import utils
from ..config.yaml_parser import YamlParser
from ..postprocess.yaml_driven_converter import YAMLDrivenConverter
from ..postprocess.h5_file_parser import h5FileParser
from ..postprocess.robomimic_config_handler import (
    RobomimicConfigHandler,
    classify_observations,
)
from ..settings.config_manager import config
from relay_d.utils.coloring_logger import logger
from ...utils.custom_message_box import CustomMessageBox
from ..postprocess.postprocess_config_loader import PostprocessConfigLoader

class PostProcessPage(QWidget):
    def __init__(self, ui_instance, yaml_parser: YamlParser = None, settings_page=None):
        super().__init__()
        self.ui = ui_instance
        self.yaml_parser = yaml_parser  # Shared YamlParser from ConfigPage
        self.settings_page = settings_page

        self.h5_files = []
        self.h5_html_structure = []
        self.h5_parser = h5FileParser()
        self.output_format = ""

        self.config_loader = PostprocessConfigLoader()
        self.yaml_converter = YAMLDrivenConverter(self.config_loader)
        self.format_buttons = {}
        self.format_names = {
            "btn_format_robomimic": "robomimic",
        }

        # Use dispatcher api for helpers
        self.dispatcher_helper = DispatcherHelpers()

        # If a yaml_parser was passed at construction and already has data loaded,
        # push its data_config into the converter immediately.
        if self.yaml_parser is not None:
            self._sync_data_config_from_parser()

        self.setup_connections()
        self.setup_format_buttons()

        # Connect to config changes
        config.settings_changed.connect(self._on_settings_changed)

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
        self._sync_data_config_from_parser()

    def _on_config_cleared(self):
        """Handle config cleared signal - wipe the bridge map."""
        logger.info("PostProcessPage: config cleared")
        self.yaml_converter.data_config = {}

    def _on_config_saved(self, file_path: str):
        """Handle config saved signal - re-sync in case output paths changed."""
        logger.info(f"PostProcessPage: config saved, re-syncing from {file_path}")
        if self.yaml_parser is not None:
            self.yaml_parser.load_yaml_file(file_path)
        self._sync_data_config_from_parser()

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
              metadata: { env_name: ..., n_actions: 7, ... }
        """
        if self.yaml_parser is None:
            logger.warning(
                "PostProcessPage: no yaml_parser available, cannot sync bridge map"
            )
            return

        raw = self.yaml_parser.yaml_data  # Full parsed YAML dict

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
    # Settings
    # ------------------------------------------------------------------

    def _on_settings_changed(self, settings):
        """Handle settings changes via signal"""
        if "hdf5_load_folder" in settings:
            folder = settings["hdf5_load_folder"]
            if folder and hasattr(self.ui, "lineEdit_hdf5_load_folder"):
                self.ui.lineEdit_hdf5_load_folder.setText(folder)

        if "enable_compression" in settings:
            logger.info(
                f"Post-process compression set to {settings['enable_compression']}"
            )

    # ------------------------------------------------------------------
    # UI wiring
    # ------------------------------------------------------------------

    def setup_connections(self):
        """Setup all button connections"""
        try:
            if hasattr(self.ui, "btn_convert_dataset"):
                self.ui.btn_convert_dataset.clicked.connect(self.convert_dataset)

            if hasattr(self.ui, "btn_load_h5"):
                self.ui.btn_load_h5.clicked.connect(self._select_files_to_convert)

            if hasattr(self.ui, "btn_clear_preview"):
                self.ui.btn_clear_preview.clicked.connect(self.clear_h5_preview)

        except Exception as e:
            logger.error("Error setting up connections.")

        return

    def setup_format_buttons(self):
        """Setup format selection buttons defined in .ui file"""
        formats = self.config_loader.get_all_format_names()
        logger.info(f"Available formats: {formats}")

        # Create button group for exclusive selection
        self.format_button_group = QButtonGroup()
        self.format_button_group.setExclusive(False)

        # Setup each format button
        for button_name, format_name in self.format_names.items():
            button = getattr(self.ui, button_name, None)

            if button is not None:
                format_info = self.config_loader.get_format_info(format_name)
                button.setCheckable(True)

                if format_info is not None:
                    button.setToolTip(format_info.get("description", ""))
                    button.clicked.connect(
                        lambda checked, name=format_name: self._select_output_format(
                            name
                        )
                    )
                    self.format_button_group.addButton(button)
                    self.format_buttons[format_name] = button

                    if format_name == "robomimic":
                        button.setChecked(True)
                        self.output_format = format_name
                        self.show_output_structure_preview(format_name)
                else:
                    logger.warning(f"Could not get format info for: {format_name}")
                    button.setEnabled(False)
            else:
                logger.warning(f"Button {button_name} not found in UI")

        logger.info(f"Setup {len(self.format_buttons)} format buttons")

    def _select_output_format(self, format_name):
        button = self.format_buttons.get(format_name)
        if button is None:
            return

        if self.output_format == format_name:
            self.output_format = ""
            button.setChecked(False)
            self.clear_output_preview()
            logger.info(f"Deselected format: {format_name}")
        else:
            self.output_format = format_name
            button.setChecked(True)
            self.show_output_structure_preview(format_name)
            logger.info(f"Selected output format: {format_name}")

    def clear_output_preview(self):
        if hasattr(self.ui, "txt_output_preview"):
            self.ui.txt_output_preview.clear()

    def show_output_structure_preview(self, format_name):
        html_preview = self.config_loader.generate_html_structure_preview(format_name)
        if hasattr(self.ui, "txt_output_preview"):
            self.ui.txt_output_preview.setHtml(html_preview)
            self.ui.txt_output_preview.show()
        else:
            logger.error("txt_output_preview widget not found")
            CustomMessageBox.warning("No output preview widget found.")

    # ------------------------------------------------------------------
    # File selection
    # ------------------------------------------------------------------

    def _select_files_to_convert(self):

        logger.info("Opening file browser")

        # Get the same folder where datasets are saved by default, for better UX. Fall back to home if not set.
        if self.settings_page:
            save_folder = self.settings_page.get_hdf5_save_folder()
            processed_dir = os.path.join(save_folder, "recorded_data") if save_folder else os.path.expanduser("~")

        fn, _ = QFileDialog.getOpenFileNames(
            parent=None,
            caption="Select H5 Files",
            filter="H5 files (*.h5)",
            directory=processed_dir,
        )

        logger.info(f"Selected files result: {fn}, type: {type(fn)}")

        if not fn or (isinstance(fn, (list, tuple)) and len(fn) == 0):
            logger.info("No file selected by user.")
            return

        file_list = list(fn) if fn else []

        if not file_list:
            logger.info("No file selected by user (empty list).")
            return

        self.h5_files = file_list
        logger.info(f"Files loaded: {self.h5_files}. Total files: {len(self.h5_files)}")

        self.h5_html_structure = self.h5_parser.convert_h5_to_html(self.h5_files[0])
        self.show_h5_preview(self.h5_html_structure)
        return

    def show_h5_preview(self, html_string):
        if hasattr(self.ui, "txt_input_structure"):
            self.h5_html_structure = []
            self.ui.txt_input_structure.clear()
            self.ui.txt_input_structure.setHtml(html_string)
        else:
            logger.error("No txt_input_structure found.")
            CustomMessageBox.warning("No preview widget found to display the H5 file structure.")
        return

    def clear_h5_preview(self):
        if hasattr(self.ui, "txt_input_structure"):
            self.ui.txt_input_structure.clear()
            self.h5_html_structure = []
            self.h5_files = []
            CustomMessageBox.info("Cleared H5 file preview and reset loaded files.")
        else:
            logger.error("No txt_input_structure found.")
            CustomMessageBox.warning("No preview widget found to clear.")
        return

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

        # Guard: ensure bridge map is populated before converting
        bridge_entries = len(self.yaml_converter.data_config.get("output", {}))
        if bridge_entries == 0:
            logger.warning("Bridge map is empty - no config loaded yet.")
            CustomMessageBox.warning(
                "No configuration loaded.\n\n"
                "Please go to the Config page and load your data_config.yaml first.")
            return

        logger.info(f"Converting dataset to {self.output_format} format.")

        fmt_config = self.config_loader.get_config(self.output_format)
        if not fmt_config:
            logger.error(f"Config not found for format: {self.output_format}")
            return

        timestamp_folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Determine output path - use settings save folder if available, otherwise default to home directory
        if self.settings_page:
            save_folder = self.settings_page.get_hdf5_save_folder()
            output_dir = os.path.join(save_folder, timestamp_folder)
        else:
            output_dir = os.path.expanduser("~/.RelayDApp/", timestamp_folder)
            logger.warning(f"The save_folder is not set, please set it in the Settings page. Defaulting to {timestamp_folder}")

        os.makedirs(output_dir, exist_ok=True)

        timestamp = int(__import__("time").time())
        output_filename = f"{self.output_format}_dataset_{timestamp}.h5"
        combined_output_path = os.path.join(output_dir, output_filename)

        if self.output_format == "raw":
            success = self.yaml_converter.convert_raw_format(
                self.h5_files, combined_output_path
            )
        else:
            success = self.yaml_converter.convert_multiple_files(
                self.output_format,
                self.h5_files,
                combined_output_path,
            )

        # Populate the training config's observation modalities from the
        # actual converted data and save it alongside the dataset, for
        # future reference. "raw" is a pass-through copy with no training-
        # config/observation-modality concept, so it's skipped.
        if success and self.output_format != "raw":
            try:
                # Copy the acquisition_config used to build the dataset into the output folder for reference
                if(not self.yaml_parser.save_yaml_file(file_path=f"{output_dir}/acquisition_config.yaml")):
                    logger.error("An error occured copying the acquisition_config to the file.")
                    return

                # Get the train config path from the format config and populate it with the actual train dir and observation modalities
                train_config_path = self.config_loader.get_output_train_config(
                    self.output_format, self.yaml_parser.get_train_model_name()
                )
                handler = RobomimicConfigHandler(train_config_path)
                handler.set_train_dir(f"~/Documents/robomimic_trains/{self.output_format}_dataset_{timestamp_folder}")

                # Classigy observations based on the actual converted data and populate the config
                classification = classify_observations(
                    self.yaml_converter.data_config,
                    self.yaml_converter.get_last_shape_cache(),
                )
                handler.set_observation_modalities(**classification)
                handler.save(
                    os.path.join(output_dir, f"{self.output_format}_config.json")
                )

                # For the dispatcher add the obs_config.yaml based on the acquired config
                action_dispatcher_folder = os.path.join(output_dir, "action_dispatcher")
                obs_config_path = os.path.join(action_dispatcher_folder, "obs_config.yaml")

                os.makedirs(action_dispatcher_folder, exist_ok=True)

                # Add the obs_config.yaml converted
                self.yaml_parser.save_yaml_file(file_path=obs_config_path, data=self.yaml_parser.get_obs_config())

                # Add the action_config_template (the user changes as it may need)
                self.dispatcher_helper.save_action_template(output_dir=action_dispatcher_folder)

                # Add the inference script for deploy
                self.dispatcher_helper.save_inference_script(output_dir=action_dispatcher_folder)

            except Exception as e:
                logger.error(
                    f"Failed to write populated {self.output_format} training config: {e}"
                )

        if success:
            logger.info("Dataset successfully converted and validated!")
            CustomMessageBox.info(
                f"Dataset successfully converted to {self.output_format} format!\n\nOutput file: {combined_output_path}",
            )
        else:
            logger.error("Dataset conversion failed")
            CustomMessageBox.error("Failed to convert dataset.")

        return
