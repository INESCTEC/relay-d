import os
from datetime import datetime

import rclpy
from typing import Optional

from .qt_app.ui.config.yaml_parser import YamlParser
from .qt_app.ui.record.topic_subscribers import TopicSubscribers
from .qt_app.ui.record.data_recorder import DataRecorder
from .qt_app.ui.record.demo_start import start_demo
from .qt_app.ui.config.builder.config_check import check_config
from .qt_app.utils.stream_builder import StreamBuilder
from relay_d.utils.tf_topic import TfTopicSubscriptionError
from .qt_app.ui.postprocess.yaml_driven_converter import YAMLDrivenConverter
from .qt_app.ui.postprocess.offline_config import load_conversion_config, write_dataset_files
from .qt_app.ui.postprocess.postprocess_options import dataset_output_paths
from .qt_app.ui.review.demo_validator import CheckStatus, DemoValidator
from .qt_app.utils.demo_file_ops import delete_demo_file
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
        self._options_path = postprocess_config_path
        self._load_conversion_config(config_path, postprocess_config_path)
        self._rclpy_initialized = False
        try:
            user_settings.apply_ros_domain()  # same domain as the GUI
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

        if config_path is not None and not self.initialize(config_path):
            self._topic_subscribers.close()
            if self._rclpy_initialized:
                rclpy.shutdown()
            raise ValueError(f"Config rejected: {config_path} (see the log)")  # the GUI shows an error dialog

    def initialize(self, config_path: str) -> bool:
        if config_path is None:
            return False
        if self._data_recorder.is_recording:  # the GUI ignores a config load during a recording too
            logger.warning("Config not loaded: a recording is in progress (stop it first).")
            return False
        try:  # conversion follows the config that records, as the Post-process page does; checked first, swapped last
            conversion = self._conversion_for(config_path, self._options_path)
        except ValueError as e:
            logger.error(f"Config rejected for conversion: {e}")
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

        self._yaml_converter, self._conversion_parser = conversion
        self._config_path = config_path
        # Subscribed at every start_recording, as in the GUI
        self._containers = [ContainerAdapter(d) for d in self._yaml_parser.get_input_data()]
        self.start_spinning()
        return True

    def start_spinning(self):
        self._topic_subscribers.start_spinning()

    @property
    def is_recording(self) -> bool:
        return bool(self._data_recorder.is_recording)

    @property
    def session_files(self) -> list:
        """Recorded demo HDF5 paths kept in this session (copy)."""
        return list(self._session_h5_files)

    @property
    def last_saved_file(self) -> Optional[str]:
        """Most recently saved demo path in this session, or None."""
        return self._session_h5_files[-1] if self._session_h5_files else None

    def recording_health(self) -> dict:
        """Snapshot of live recording progress and background write health."""
        rec = self._data_recorder
        return {
            "is_recording": bool(rec.is_recording),
            "sample_count": int(getattr(rec, "recording_sample_count", 0)),
            "buffer_full": bool(getattr(rec, "buffer_full", False)),
            "loop_failed": bool(getattr(rec, "loop_failed", False)),
            "write_error": getattr(rec, "write_error", None),
            "stop_problem": getattr(rec, "stop_problem", ""),
        }

    def start_recording(self, demo_name: Optional[str] = None) -> bool:
        """Start a demo, named like the GUI's (``{dataset_name}_{ms}``) unless ``demo_name`` is given."""
        if self._data_recorder.is_recording:  # refused before start_demo, which would subscribe every input again
            logger.warning("Recording already in progress")
            return False
        try:
            started = start_demo(  # same steps as the GUI's Start
                self._data_recorder, self._topic_subscribers, self._containers,
                self._yaml_parser.get_dataset_name(), demo_name,
            ) is not None
        except TfTopicSubscriptionError as e:
            logger.error(f"Recording NOT started: a configured TF topic could not be read: {e}")
            return False
        retried, self._data_recorder.retried_file = self._data_recorder.retried_file, None  # added once, not at every Start
        if retried:  # an earlier demo saved only now: keep it in the session
            self._session_h5_files.append(retried)
        return started

    def stop_recording(self) -> bool:
        """Stop and save the demo. False when it was not saved, or when it was saved but sampling had stopped early
        (buffer full, or the recording loop kept failing: the GUI shows a dialog for these); see the log."""
        write_err = getattr(self._data_recorder, "write_error", None)
        cut_short = ("the recording buffer was full" if getattr(self._data_recorder, "buffer_full", False) else
                     "the recording loop kept failing" if getattr(self._data_recorder, "loop_failed", False) else
                     f"disk write failed ({write_err})" if isinstance(write_err, str) and write_err else "")
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
        if result and cut_short:
            logger.error(f"Demo saved but INCOMPLETE: sampling stopped before Stop because {cut_short} (see the log).")
            return False
        return result

    def delete_demo(self, file_path: str) -> bool:
        """Delete a recorded demo and drop it from this session, like the GUI's Review > Delete / Reject Demo."""
        if not delete_demo_file(file_path, self._data_recorder):
            return False
        path = os.path.abspath(file_path)  # a relative path names the same demo
        self._session_h5_files = [p for p in self._session_h5_files if os.path.abspath(p) != path]
        return True

    @staticmethod
    def _conversion_for(config_path: Optional[str], options_path: Optional[str]):
        """(converter, parser) with the config and format options of relay-d-convert (options: ``options_path``, else
        a postprocess_config.yaml next to the config, else the defaults). Raises ValueError if a file is unusable."""
        converter, parser = YAMLDrivenConverter(), None
        if config_path:
            parser, converter.options = load_conversion_config(config_path, options_path)
            converter.data_config = parser.yaml_data["config"]
        return converter, parser

    def _load_conversion_config(self, config_path: Optional[str], options_path: Optional[str]) -> None:
        self._yaml_converter, self._conversion_parser = self._conversion_for(config_path, options_path)

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

    def shutdown(self, convert: bool = True, output_format: Optional[str] = None) -> bool:
        """Save a running recording, optionally convert the session, and stop ROS.

        Returns False, and leaves everything running, if a recording could not be saved: its samples are only
        in memory (fix the disk and call again). Set ``convert=False`` to skip dataset conversion on close, or
        pass ``output_format`` (``"robomimic"`` / ``"lerobot"``) to override ``postprocess_default_format``.
        """
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
        if convert:
            fmt = output_format or user_settings.get("postprocess_default_format") or "robomimic"
            self.convert_recording_to_dataset(fmt)
        self._topic_subscribers.close()  # stops its spin thread (no self-join when called from a callback)
        if self._rclpy_initialized:
            rclpy.shutdown()
        return True
