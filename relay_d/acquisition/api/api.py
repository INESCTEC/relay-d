import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rclpy
from threading import Thread, current_thread
from typing import Optional

from ..qt_app.ui.config.yaml_parser import YamlParser
from ..qt_app.ui.record.topic_subscribers import TopicSubscribers
from ..qt_app.ui.record.data_recorder import DataRecorder
from ..qt_app.ui.postprocess.yaml_driven_converter import YAMLDrivenConverter
from ..qt_app.ui.postprocess.postprocess_config_loader import PostprocessConfigLoader
from ..qt_app.ui.review.demo_validator import DemoValidator
from relay_d.utils.coloring_logger import logger
from ..qt_app.utils.demo_file_ops import delete_demo_file

class ContainerAdapter:
    def __init__(self, data: dict):
        self.container_id = data.get("name")
        self.topic_path = data.get("topic")
        self.input_name = data.get("name")
        self.parent_frame = data.get("parent_frame")
        self.child_frame = data.get("child_frame")
        self.transfer_rate = data.get("transfer_rate", 10)


class AppAPI:
    def __init__(self, config_path: str, output_dir: Optional[str] = None):
        self._rclpy_initialized = False
        try:
            rclpy.init()
            self._rclpy_initialized = True
        except Exception:
            pass

        self._yaml_parser = YamlParser()
        self._topic_subscribers = TopicSubscribers()
        self._data_recorder = DataRecorder(self._yaml_parser)
        self._config_loader = PostprocessConfigLoader()
        self._yaml_converter = YAMLDrivenConverter(self._config_loader)
        if config_path:
            self._yaml_converter.load_data_config(config_path)

        if output_dir is None:
            output_dir = os.path.join(os.path.expanduser("~"), "lfd_recordings")

        # Pass the output directory to the DataRecorder instance
        self._base_output_dir = output_dir
        self._data_recorder.output_directory = output_dir
        self._session_h5_files: list = []
        self._validator = DemoValidator()
        self._last_validation_results: dict = {}

        self._spin_thread = None
        self._should_spin = False
        self._config_path = config_path

        if config_path is not None:
            self.initialize(config_path)

    def initialize(self, config_path: str) -> bool:
        if config_path is None:
            return False
        if not self._yaml_parser.load_yaml_file(config_path):
            return False

        for input_data in self._yaml_parser.get_input_data():
            container = ContainerAdapter(input_data)
            self._topic_subscribers.create_subscriber(container)

        self.start_spinning()
        return True

    def start_spinning(self):
        self._should_spin = True
        self._spin_thread = Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

    def _spin_loop(self):
        while self._should_spin:
            with self._topic_subscribers._spin_lock:
                rclpy.spin_once(self._topic_subscribers, timeout_sec=0.01)
            rclpy.spin_once(self._data_recorder, timeout_sec=0.0)

    def start_recording(self, demo_name: Optional[str] = None) -> bool:
        self.demo_name_ = demo_name
        return self._data_recorder.start_recording(
            self._topic_subscribers, demo_name, flush_stale_data=True
        )

    def pause_recording(self) -> bool:
        return self._data_recorder.pause_recording()

    def resume_recording(self) -> bool:
        return self._data_recorder.resume_recording()

    def stop_recording(self) -> bool:
        result = self._data_recorder.stop_recording()
        if result and self._data_recorder.h5_file_path:
            file_path = self._data_recorder.h5_file_path
            self._session_h5_files.append(file_path)
            try:
                self._last_validation_results[file_path] = self._validator.validate(file_path)
            except Exception as e:
                logger.error(f"Error validating recording: {e}")
        return result

    def get_validation_result(self, file_path: str):
        return self._last_validation_results.get(file_path)

    def delete_recording(self, file_path: str) -> bool:
        ok = delete_demo_file(file_path)
        if ok:
            if file_path in self._session_h5_files:
                self._session_h5_files.remove(file_path)
            self._last_validation_results.pop(file_path, None)
        return ok

    def get_recording_status(self) -> dict:
        return {
            "is_recording": self._data_recorder.is_recording,
            "is_paused": self._data_recorder.is_paused,
            "file_path": self._data_recorder.h5_file_path,
            "samples": len(self._data_recorder.timestamps),
        }

    def convert_recording_to_dataset(self) -> bool:
        try:
            if not self._session_h5_files:
                logger.warning("No recordings in this session to convert.")
                return False
            output_path = os.path.join(self._data_recorder.recorded_data_dir, "combined_dataset.h5")
            success = self._yaml_converter.convert_multiple_files(
                "robomimic", self._session_h5_files, output_path
            )
            if success:
                logger.info(
                    f"Successfully converted {len(self._session_h5_files)} recording(s) "
                    f"to: {output_path}"
                )
            return success
        except Exception as e:
            logger.error(f"Error converting recording to dataset: {e}")
            return False

    def shutdown(self):
        self.convert_recording_to_dataset()
        self._should_spin = False

        called_from_spin_thread = self._spin_thread is current_thread()

        if called_from_spin_thread:
            spin_thread = self._spin_thread
            rclpy_initialized = self._rclpy_initialized
            topic_subscribers = self._topic_subscribers

            def _deferred_shutdown():
                if spin_thread and spin_thread.is_alive():
                    spin_thread.join(timeout=2.0)
                topic_subscribers.close()
                if rclpy_initialized:
                    rclpy.shutdown()

            Thread(target=_deferred_shutdown, daemon=True).start()
        else:
            if self._spin_thread and self._spin_thread.is_alive():
                self._spin_thread.join(timeout=2.0)
            self._topic_subscribers.close()
            if self._rclpy_initialized:
                rclpy.shutdown()
