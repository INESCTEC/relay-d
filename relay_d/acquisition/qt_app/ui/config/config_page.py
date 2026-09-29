import os
import sys
import tkinter as tk

# Import utils
from .yaml_parser import YamlParser
from ..settings.config_manager import config
from relay_d.utils.coloring_logger import logger
from ...utils.custom_message_box import CustomMessageBox
from ...utils.stream_builder import StreamBuilder

from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QTextEdit, QPushButton
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QFileDialog


# Create a signal manager class
class ConfigSignalManager(QObject):
    # Signal emitted when config is loaded
    config_loaded = pyqtSignal(str)  # Passes file path
    config_cleared = pyqtSignal()
    config_saved = pyqtSignal(str)  # Passes file path

# Class for the config page
class ConfigPage:
    def __init__(self, ui_instance):
        self.ui = ui_instance
        self.yaml_parser = YamlParser()

        # Create the signal manager
        self.signal_manager = ConfigSignalManager()

        # Setup all button connections
        self._setup_connections()

        # Connect to config changes
        config.settings_changed.connect(self._on_settings_loaded)

    def _on_settings_loaded(self, settings):
        """Handle settings changes via signal"""
        # Config page doesn't need to react to many settings
        # but we could refresh config preview or other things here

        if "hdf5_load_folder" in settings:
            self.load_folder = settings["hdf5_load_folder"]
            if self.load_folder and hasattr(self.ui, "lineEdit_hdf5_load_folder"):
                self.ui.lineEdit_hdf5_load_folder.setText(self.load_folder)

    ## Connect all the buttons related with config page
    def _setup_connections(self):

        try:
            if hasattr(self.ui, "btn_insert_config"):
                self.ui.btn_insert_config.clicked.connect(self.insert_config_file)
            else:
                logger.error(f"No button btn_insert_config found.")

            if hasattr(self.ui, "btn_remove_config"):
                self.ui.btn_remove_config.clicked.connect(self.remove_config_file)
            else:
                logger.error(f"No button btn_remove_config found.")

            if hasattr(self.ui, "btn_save_config"):
                self.ui.btn_save_config.clicked.connect(self.save_config_file)
            else:
                logger.error(f"No button btn_save_config found.")

            return True

        except Exception as e:
            logger.error(f"Exception occured: {e}")
            return False

    ## Function to insert a new config into the app
    def insert_config_file(self):

        # Open the file browser
        logger.info(f"Opening file browser to select a config file.")

        try:
            fn, _ = QFileDialog.getOpenFileName(
                parent=self.load_folder if self.load_folder else None,
                caption="Open Config File",
                filter="YAML files (*.yaml *.yml)",
            )
        except Exception as e:
            logger.error(f"An error occured while opening file: {e}")
            return

        if not fn:
            logger.error("The user did not select any file.")
            return
        else:
            logger.info(f"Loaded file {fn}.")

        ## Parse the YAML file
        if not self.yaml_parser.load_yaml_file(fn):
            logger.error(f"Failed to parse config file {fn}.")
            CustomMessageBox.error(f"Failed to parse config file:\n{fn}")
            return

        ## Validate that every configured input's topic/TF is actually
        ## available before accepting the config -- a config referencing a
        ## dead topic or an unreachable TF transform must not be allowed to
        ## load (it would otherwise only fail later, mid-recording).
        logger.info("Validating configured topics/TF against the live ROS graph...")
        failures = self._validate_input_availability()
        if failures:
            self.yaml_parser.clear()
            self.yaml_parser.current_config = None
            self.yaml_parser.yaml_data = None
            message = (
                "Config rejected -- the following inputs are not available:\n"
                + "\n".join(failures)
            )
            logger.error(message)
            CustomMessageBox.error(message)
            return

        ## Validate that every configured output.streams spec (including
        ## arithmetic '+'/'-' specs) actually parses -- a malformed spec must
        ## not be allowed to load silently, only to end up missing from the
        ## recorded data with nothing but a console log to explain why.
        stream_errors = self._validate_stream_specs()
        if stream_errors:
            self.yaml_parser.clear()
            self.yaml_parser.current_config = None
            self.yaml_parser.yaml_data = None
            message = (
                "Config rejected -- the following output.streams specs "
                "failed to parse:\n" + "\n".join(f"  {e}" for e in stream_errors)
            )
            logger.error(message)
            CustomMessageBox.error(message)
            return

        ## Send the signal to the signal_manager
        self.signal_manager.config_loaded.emit(fn)
        logger.debug(f"Emitted config_loaded signal with file path: {fn}")

        ## Display config file
        self.show_config_preview()
        logger.debug(f"Config file preview updated for file: {fn}")

        return

    def _validate_input_availability(self):
        """Check every configured input's topic/TF against the live ROS
        graph. Returns a list of human-readable failure descriptions (empty
        if every input is currently available)."""
        topic_subscribers = getattr(
            getattr(self.ui, "page_record", None), "topic_subscribers", None
        )
        if topic_subscribers is None:
            logger.warning(
                "No topic_subscribers node available -- skipping input "
                "availability check."
            )
            return []

        failures = []
        for item in self.yaml_parser.get_input_data():
            name = item["name"]
            if item.get("is_tf"):
                tf_topic = item.get("tf_topic", "/tf")
                tf_static_topic = item.get("tf_static_topic", "/tf_static")
                logger.info(
                    f"Validating TF input '{name}': {item['parent_frame']} -> "
                    f"{item['child_frame']} on {tf_topic}..."
                )
                if not topic_subscribers.is_tf_available(
                    item["parent_frame"],
                    item["child_frame"],
                    tf_topic=tf_topic,
                    tf_static_topic=tf_static_topic,
                    timeout=item.get("tf_wait_timeout"),
                ):
                    failures.append(
                        f"  {name}: TF {item['parent_frame']} -> "
                        f"{item['child_frame']} not available on "
                        f"tf_topic={tf_topic}"
                    )
            elif item.get("topic"):
                logger.info(f"Validating input '{name}': topic {item['topic']}...")
                if not topic_subscribers.is_topic_available(item["topic"]):
                    failures.append(f"  {name}: topic {item['topic']} not found")
        return failures

    def _validate_stream_specs(self):
        """Build a throwaway StreamBuilder against the currently-loaded
        yaml_parser and return every spec-parsing error it collected (empty
        list if every output.streams spec parsed cleanly). StreamBuilder is
        pure yaml-config + message-dict logic (no ROS node / Qt dependency),
        so it's safe to construct here just to validate."""
        return StreamBuilder(self.yaml_parser).get_build_errors()

    def remove_config_file(self):

        if not self.yaml_parser.current_config:
            logger.warning(f"There is no lodaded config to remove.")
            return

        # Removing config file
        logger.info(f"Removing config file")

        self.yaml_parser.clear()
        self.clear_config_preview()
        ## Emit a signal to say the config is not loaded anymore
        self.signal_manager.config_cleared.emit()

        self.yaml_parser.current_config = {}

        return

    def save_config_file(self):
        """Save the current YAML configuration back to the file"""
        if not self.yaml_parser.is_loaded():
            logger.warning("No config file loaded to save.")
            CustomMessageBox.warning("No config file loaded to save.")
            return

        if hasattr(self.ui, "txt_config_display"):
            yaml_text = self.ui.txt_config_display.toPlainText()
            if self.yaml_parser.load_from_yaml_string(yaml_text):
                stream_errors = self._validate_stream_specs()
                if stream_errors:
                    message = (
                        "Not saved -- the following output.streams specs "
                        "failed to parse:\n"
                        + "\n".join(f"  {e}" for e in stream_errors)
                    )
                    logger.error(message)
                    CustomMessageBox.error(message)
                    return
                if self.yaml_parser.save_yaml_file():
                    logger.info(
                        f"Config file saved successfully to {self.yaml_parser.current_config}"
                    )
                    CustomMessageBox.info("Config file saved successfully.")
                    self.show_config_preview()
                    self.signal_manager.config_saved.emit(
                        self.yaml_parser.current_config
                    )
                else:
                    logger.error("Failed to save config file.")
                    CustomMessageBox.warning("Failed to save config file.")
            else:
                logger.error("Failed to parse YAML from text display.")
                CustomMessageBox.warning("Failed to parse YAML from text display.")

        return

    def show_config_preview(self):

        # First retrieve answer from yaml
        yaml_str = self.yaml_parser.get_raw_yaml()

        # Append the message
        if hasattr(self.ui, "txt_config_display"):
            self.clear_config_preview()

            # Use the yaml_parser to convert to html
            html_str = self.yaml_parser.convert_yaml_to_html()

            self.ui.txt_config_display.setHtml(html_str)
        else:
            logger.error(f"No txt_config_display found.")

        # Update metrics
        self.update_metrics()

        return

    def update_metrics(self):
        """Update the configuration metrics display"""
        try:
            # Get input data
            input_data = self.yaml_parser.get_input_data()
            output_data = self.yaml_parser.get_output_data()

            # Count inputs and outputs
            num_inputs = len(input_data) if input_data else 0
            num_outputs = len(output_data) if output_data else 0

            # Get input types
            input_types = set()
            if input_data:
                for item in input_data:
                    name = item.get("name", "")
                    if name:
                        base_type = name.split("_")[0]
                        input_types.add(base_type)

            # Get output types
            output_types = set()
            if output_data:
                for item in output_data:
                    name = item.get("name", "")
                    if name:
                        base_type = name.split("_")[0]
                        output_types.add(base_type)

            # Update labels
            if hasattr(self.ui, "inputsCountLabel"):
                self.ui.inputsCountLabel.setText(f"Inputs: {num_inputs}")

            if hasattr(self.ui, "outputsCountLabel"):
                self.ui.outputsCountLabel.setText(f"Outputs: {num_outputs}")

            if hasattr(self.ui, "inputTypesLabel"):
                types_str = ", ".join(sorted(input_types)) if input_types else "None"
                self.ui.inputTypesLabel.setText(f"Input Types: {types_str}")

            if hasattr(self.ui, "outputTypesLabel"):
                types_str = ", ".join(sorted(output_types)) if output_types else "None"
                self.ui.outputTypesLabel.setText(f"Output Types: {types_str}")

            logger.info(f"Metrics updated: {num_inputs} inputs, {num_outputs} outputs")

        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

        return

    def clear_config_preview(self):

        # Append the message
        if hasattr(self.ui, "txt_config_display"):
            self.ui.txt_config_display.clear()
        else:
            logger.error(f"No txt_config_display found.")

        # Reset metrics
        self.reset_metrics()

        return

    def reset_metrics(self):
        """Reset metrics to default values"""
        if hasattr(self.ui, "inputsCountLabel"):
            self.ui.inputsCountLabel.setText("Inputs: 0")
        if hasattr(self.ui, "outputsCountLabel"):
            self.ui.outputsCountLabel.setText("Outputs: 0")
        if hasattr(self.ui, "inputTypesLabel"):
            self.ui.inputTypesLabel.setText("Input Types: None")
        if hasattr(self.ui, "outputTypesLabel"):
            self.ui.outputTypesLabel.setText("Output Types: None")

        return
