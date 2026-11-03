import os
from datetime import datetime

import rclpy
from typing import Optional

from ..qt_app.ui.config.yaml_parser import YamlParser
from ..qt_app.ui.record.topic_subscribers import TopicSubscribers
from ..qt_app.ui.record.data_recorder import DataRecorder
from ..qt_app.ui.record.demo_start import start_demo
from ..qt_app.ui.config.builder.config_check import check_config
from ..qt_app.utils.stream_builder import StreamBuilder
from relay_d.utils.tf_topic import TfTopicSubscriptionError
from ..qt_app.ui.postprocess.yaml_driven_converter import YAMLDrivenConverter
from ..qt_app.ui.postprocess.postprocess_config_loader import PostprocessConfigLoader
from ..qt_app.ui.postprocess.offline_config import load_conversion_config, write_dataset_files
from ..qt_app.ui.postprocess.postprocess_options import dataset_output_paths
from ..qt_app.ui.review.demo_validator import CheckStatus, DemoValidator
from relay_d.utils import user_settings
from relay_d.utils.coloring_logger import logger

class ContainerAdapter:
    def __init__(self, data: dict):
        self.container_id = data.get("name")
        self.topic_path = data.get("topic")
        self.input_name = data.get("name")
        self.parent_frame = data.get("parent_frame")
        self.child_frame = data.get("child_frame")
        self.tf_topic = data.get("tf_topic")  # None -> /tf, /tf_static (as in the GUI's ContainerManager)
        self.tf_static_topic = data.get("tf_static_topic")
        self.tf_wait_timeout = data.get("tf_wait_timeout")


class AppAPI:
    def __init__(
        self,
        config_path: str,
        output_dir: Optional[str] = None,
        postprocess_config_path: Optional[str] = None,
    ):
        # Conversion config + format options, checked like relay-d-convert: raises before any ROS setup
        self._load_conversion_config(config_path, postprocess_config_path)
        self._rclpy_initialized = False
        try:
            rclpy.init()
            self._rclpy_initialized = True
        except Exception:
            pass

        self._yaml_parser = YamlParser()
        self._topic_subscribers = TopicSubscribers()
        self._data_recorder = DataRecorder(self._yaml_parser)
        # An explicit output_dir wins; otherwise the recorder reads the hdf5_save_folder setting when a session starts,
        # so a changed setting applies to the next recording.
        self._data_recorder.output_directory = output_dir
        self._session_h5_files: list = []
        self._validator = DemoValidator()

        self._containers: list = []
        self._config_path = config_path

        if config_path is not None:
            self.initialize(config_path)

    def initialize(self, config_path: str) -> bool:
        if config_path is None:
            return False
        if not self._yaml_parser.load_yaml_file(config_path):
            return False

        # The Config page's checks: refuse a dead topic / unreachable TF and a stream spec that does not parse
        # (either would leave that input silently missing); the rest of check_config only warns, as there.
        failures = self._topic_subscribers.unavailable_inputs(self._yaml_parser.get_input_data())
        failures += [f"  {e}" for e in StreamBuilder(self._yaml_parser).get_build_errors()]
        if failures:
            self._yaml_parser.clear()
            self._yaml_parser.current_config = self._yaml_parser.yaml_data = None  # unloaded, as the Config page does
            logger.error("Config rejected:\n" + "\n".join(failures))
            return False
        problems = check_config(self._yaml_parser.yaml_data)
        if problems:
            logger.warning("Config check:\n" + "\n".join(f"  {p.message}" for p in problems))

        # Subscribed at every start_recording, as in the GUI
        self._containers = [ContainerAdapter(d) for d in self._yaml_parser.get_input_data()]
        self.start_spinning()
        return True

    def start_spinning(self):
        self._topic_subscribers.start_spinning()

    def start_recording(self, demo_name: Optional[str] = None) -> bool:
        """Start a demo, named like the GUI's (``{dataset_name}_{ms}``) unless ``demo_name`` is given."""
        try:
            started = start_demo(  # same steps as the GUI's Start
                self._data_recorder, self._topic_subscribers, self._containers,
                self._yaml_parser.get_dataset_name(), demo_name,
            ) is not None
        except TfTopicSubscriptionError as e:
            logger.error(f"Recording NOT started: a configured TF topic could not be read: {e}")
            return False
        if self._data_recorder.retried_file:  # an earlier demo saved only now: keep it in the session
            self._session_h5_files.append(self._data_recorder.retried_file)
        return started

    def stop_recording(self) -> bool:
        result = self._data_recorder.stop_recording()
        if not self._data_recorder.is_recording:  # saved, or kept in memory for a retry: subscribe again at the next start
            self._topic_subscribers.clear_all_subscribers()
        if result and self._data_recorder.h5_file_path:
            file_path = self._data_recorder.h5_file_path
            self._session_h5_files.append(file_path)
            try:
                check = self._validator.validate(file_path)
                for c in check.checks:  # the GUI shows these in a dialog
                    if c.status != CheckStatus.PASS:
                        (logger.error if c.status == CheckStatus.FAIL else logger.warning)(
                            f"Demo check [{c.status.value.upper()}] {c.name}: {c.message}"
                        )
            except Exception as e:
                logger.error(f"Error validating recording: {e}")
        return result

    def _load_conversion_config(self, config_path: Optional[str], options_path: Optional[str]) -> None:
        """The converter, with the config and format options of relay-d-convert (options: ``options_path``, else a
        postprocess_config.yaml next to the config, else the defaults). Raises ValueError if a file is unusable."""
        self._yaml_converter = YAMLDrivenConverter(PostprocessConfigLoader())
        self._conversion_parser = None
        if config_path:
            self._conversion_parser, self._yaml_converter.options = load_conversion_config(config_path, options_path)
            self._yaml_converter.data_config = self._conversion_parser.yaml_data["config"]

    def convert_recording_to_dataset(self, output_format: str = "robomimic") -> bool:
        """Convert this session's recordings like the Post-process page: a new ``<date>_<Format>`` folder in the
        recordings folder with the dataset (robomimic .h5 or LeRobot v3.0 folder), the configs used and action_dispatcher/."""
        try:
            if not self._session_h5_files:
                logger.warning("No recordings in this session to convert.")
                return False
            if self._conversion_parser is None:
                logger.error("No config loaded: nothing to convert with.")
                return False
            recordings = os.path.dirname(self._data_recorder.recorded_data_dir)  # the session folder's parent
            output_dir, output_path = dataset_output_paths(recordings, output_format, datetime.now())
            success = self._yaml_converter.convert_multiple_files(
                output_format, self._session_h5_files, output_path
            )
            if not success:
                logger.error(f"Dataset conversion failed: {self._yaml_converter.last_error or 'see the log above'}")
                return False
            logger.info(f"{self._yaml_converter.conversion_summary()} -> {output_path}")
            try:
                write_dataset_files(output_dir, self._conversion_parser, self._yaml_converter.options.source)
            except Exception as e:
                logger.warning(
                    "The dataset was converted, but some extra files (config copies / action_dispatcher files) "
                    f"could not be written: {str(e) or type(e).__name__}"
                )
            return True
        except Exception as e:
            logger.error(f"Error converting recording to dataset: {e}")
            return False

    def shutdown(self) -> bool:
        """Save a running recording, convert the session and stop ROS. Returns False, and leaves everything
        running, if a recording could not be saved: its samples are only in memory (fix the disk and call again)."""
        if self._data_recorder.is_recording:  # its samples are only in memory: save them before converting
            self.stop_recording()
        if getattr(self._data_recorder, "_unsaved", False):
            if not self._data_recorder.retry_save():
                logger.error(
                    f"Not shutting down: the last recording could not be saved to "
                    f"{self._data_recorder.h5_file_path} and is still only in memory."
                )
                return False
            self._session_h5_files.append(self._data_recorder.retried_file)
        self.convert_recording_to_dataset(user_settings.get("postprocess_default_format") or "robomimic")
        self._topic_subscribers.close()  # stops its spin thread (no self-join when called from a callback)
        if self._rclpy_initialized:
            rclpy.shutdown()
        return True
