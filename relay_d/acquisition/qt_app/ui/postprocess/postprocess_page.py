import os
import shutil

from PyQt5.QtWidgets import QWidget, QButtonGroup
from PyQt5.QtWidgets import QFileDialog
from datetime import datetime
from relay_d.dispatch import save_dispatch_templates

# Import utils
from ..config.yaml_parser import YamlParser
from ..postprocess.yaml_driven_converter import YAMLDrivenConverter
from ..postprocess.h5_file_parser import h5FileParser
from ..postprocess.robomimic_config_handler import (
    RobomimicConfigHandler,
    classify_observations,
)
from ..settings.config_manager import config
from relay_d.utils import user_settings
from relay_d.utils.coloring_logger import logger
from ...utils.custom_message_box import CustomMessageBox
from ...utils.default_configs import generate_default_configs
from ..postprocess.postprocess_config_loader import PostprocessConfigLoader
from ..postprocess.postprocess_options import dataset_output_paths
from ..postprocess.offline_config import load_offline_config
from ..postprocess.postprocess_options import PostprocessOptions

class PostProcessPage(QWidget):
    def __init__(self, ui_instance, yaml_parser: YamlParser = None, settings_page=None):
        super().__init__()
        self.ui = ui_instance
        self.yaml_parser = yaml_parser  # Shared YamlParser from ConfigPage
        # Config loaded on this page only (no live-ROS check, never shared with the
        # Record page); takes precedence over the shared parser while set.
        self._local_parser = None
        # Format options (postprocess_config.yaml): loaded explicitly by the user, or
        # auto-detected next to a loaded config. Neither -> built-in defaults.
        self._options_explicit = False
        self.settings_page = settings_page

        self.h5_files = []
        self.h5_parser = h5FileParser()
        self.output_format = ""

        self.config_loader = PostprocessConfigLoader()
        self.yaml_converter = YAMLDrivenConverter(self.config_loader)
        self.format_buttons = {}
        self.format_names = {
            "btn_format_robomimic": "robomimic",
            "btn_format_lerobot": "lerobot",
        }

        # If a yaml_parser was passed at construction and already has data loaded,
        # push its data_config into the converter immediately.
        if self.yaml_parser is not None:
            self._sync_data_config_from_parser()

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
                logger.warning(f"Ignoring {path}: {'; '.join(problems)}")
            return False
        warnings = [p for p in problems if p.startswith("warning:")]
        if warnings and explicit:
            CustomMessageBox.warning("\n".join(w[len("warning: "):] for w in warnings))
        self.yaml_converter.options = options
        self._options_explicit = explicit
        self._update_options_label()
        logger.info(f"PostProcessPage: using format options from {path}")
        return True

    def _auto_load_sibling_options(self, config_path: str):
        """Use a postprocess_config.yaml next to the loaded config, unless the user chose one."""
        if self._options_explicit:
            return
        sibling = PostprocessOptions.find_sibling(config_path)
        if sibling:
            self.load_postprocess_options_from_path(sibling, explicit=False)

    def _clear_postprocess_options(self):
        self.yaml_converter.options = PostprocessOptions()
        self._options_explicit = False
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
              metadata: { env_name: ..., n_actions: 7, ... }
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
        ui.btn_convert_dataset.clicked.connect(self.convert_dataset)
        ui.btn_load_h5.clicked.connect(self._select_files_to_convert)
        ui.btn_clear_preview.clicked.connect(self.clear_h5_preview)
        ui.btn_generate_postprocess_configs.clicked.connect(lambda: generate_default_configs(ui))
        ui.btn_load_postprocess_config.clicked.connect(self._load_postprocess_config)
        ui.btn_clear_postprocess_config.clicked.connect(self._clear_postprocess_config)
        ui.btn_load_postprocess_options.clicked.connect(self._load_postprocess_options)
        ui.btn_clear_postprocess_options.clicked.connect(self._clear_postprocess_options)

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

                    if format_name == user_settings.get("postprocess_default_format"):  # setting
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
            self.clear_output_preview()
            logger.info(f"Deselected format: {format_name}")
        else:
            self.output_format = format_name
            self._sync_format_buttons()
            self.show_output_structure_preview(format_name)
            logger.info(f"Selected output format: {format_name}")

    def _sync_format_buttons(self):
        """Check exactly the button of the selected format (none if nothing is selected)."""
        for name, button in self.format_buttons.items():
            blocked = button.blockSignals(True)
            button.setChecked(name == self.output_format)
            button.blockSignals(blocked)

    def clear_output_preview(self):
        self.ui.txt_output_preview.clear()

    def show_output_structure_preview(self, format_name):
        self.ui.txt_output_preview.setHtml(self.config_loader.generate_html_structure_preview(format_name))
        self.ui.txt_output_preview.show()

    # ------------------------------------------------------------------
    # File selection
    # ------------------------------------------------------------------

    def _select_files_to_convert(self):

        logger.info("Opening file browser")

        # Get the same folder where datasets are saved by default, for better UX. Fall back to home if not set.
        save_folder = self.settings_page.get_hdf5_save_folder() if self.settings_page else ""
        processed_dir = os.path.join(save_folder, "recorded_data") if save_folder else os.path.expanduser("~")

        fn, _ = QFileDialog.getOpenFileNames(
            parent=None,
            caption="Select H5 Files",
            filter="H5 files (*.h5)",
            directory=processed_dir,
        )

        if not fn:
            logger.info("No file selected by user.")
            return

        self.h5_files = list(fn)
        logger.info(f"Files loaded: {self.h5_files}. Total files: {len(self.h5_files)}")
        self.show_h5_preview(self.h5_parser.convert_h5_to_html(self.h5_files[0]))

    def show_h5_preview(self, html_string):
        self.ui.txt_input_structure.clear()
        self.ui.txt_input_structure.setHtml(html_string)

    def clear_h5_preview(self):
        self.ui.txt_input_structure.clear()
        self.h5_files = []
        CustomMessageBox.info("Cleared H5 file preview and reset loaded files.")
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
                "Load a config on the Config page, or use 'Load Config...' on this page.")
            return

        logger.info(f"Converting dataset to {self.output_format} format.")

        # The Recordings folder setting, else ~/.RelayDApp
        save_folder = self.settings_page.get_hdf5_save_folder() if self.settings_page else ""
        if not save_folder:
            save_folder = os.path.expanduser("~/.RelayDApp")
            logger.warning(f"The recordings folder is not set (Settings page). Defaulting to {save_folder}")
        output_dir, combined_output_path = dataset_output_paths(save_folder, self.output_format, datetime.now())
        os.makedirs(output_dir, exist_ok=True)

        success = self.yaml_converter.convert_multiple_files(
            self.output_format,
            self.h5_files,
            combined_output_path,
        )

        # Populate the training config's observation modalities from the
        # actual converted data and save it alongside the dataset, for
        # future reference.
        post_error = None
        if success:
            try:
                # Copy the acquisition_config used to build the dataset into the output folder for reference
                if(not self._active_parser().save_yaml_file(file_path=f"{output_dir}/acquisition_config.yaml")):
                    raise RuntimeError("could not copy acquisition_config.yaml to the output folder")

                # Keep the format options used for this dataset next to the config
                if self.yaml_converter.options.source:
                    shutil.copy2(
                        self.yaml_converter.options.source,
                        os.path.join(output_dir, "postprocess_config.yaml"),
                    )

                # The robomimic training config only applies to the robomimic format
                if self.output_format != "lerobot":
                    # Get the train config path from the format config and populate it with the actual train dir and observation modalities
                    train_config_path = self.config_loader.get_output_train_config(
                        self.output_format,
                        self._active_parser().get_train_model_name(self.yaml_converter.options),
                    )
                    handler = RobomimicConfigHandler(train_config_path)
                    handler.set_train_dir(f"~/Documents/robomimic_trains/{os.path.basename(output_dir)}")

                    # Classigy observations based on the actual converted data and populate the config
                    classification = classify_observations(
                        self.yaml_converter.data_config,
                        self.yaml_converter.get_last_shape_cache(),
                    )
                    handler.set_observation_modalities(**classification)
                    handler.save(
                        os.path.join(output_dir, f"{self.output_format}_config.json")
                    )

                # The dispatcher files (robomimic and LeRobot policies both run through lfd-inference-node):
                # obs_config.yaml from the acquired config, an action_config to adapt, and the inference script
                action_dispatcher_folder = os.path.join(output_dir, "action_dispatcher")
                os.makedirs(action_dispatcher_folder, exist_ok=True)
                parser = self._active_parser()
                parser.save_yaml_file(
                    file_path=os.path.join(action_dispatcher_folder, "obs_config.yaml"), data=parser.get_obs_config()
                )
                save_dispatch_templates(action_dispatcher_folder)

            except Exception as e:
                post_error = str(e) or type(e).__name__
                logger.error(
                    f"Failed to write populated {self.output_format} training config: {e}"
                )

        if success:
            logger.info("Dataset successfully converted and validated!")
            if post_error:
                CustomMessageBox.warning(
                    "The dataset was converted, but some extra files (training config / "
                    f"action_dispatcher files) could not be written:\n\n{post_error}"
                )
            CustomMessageBox.info(
                f"Dataset successfully converted to {self.output_format} format!\n\nOutput file: {combined_output_path}",
            )
        else:
            logger.error("Dataset conversion failed")
            reason = self.yaml_converter.last_error
            CustomMessageBox.error(
                "Failed to convert dataset." + (f"\n\n{reason}" if reason else "")
            )

        return
