import time

from PyQt5.QtCore import QTimer

from relay_d.utils.coloring_logger import logger
from relay_d.utils.tf_topic import TfTopicSubscriptionError

from ..review.demo_validator import CheckStatus
from .demo_start import start_demo
from ...utils.background import run_in_background
from ...utils.custom_message_box import CustomMessageBox


class RecordingController:
    """Recording lifecycle for the record page: start / stop.

    Owns the recording state (``is_recording``), the demos recorded in this session and the
    Start/Stop buttons, whose enabled state follows the state via ``update_buttons``.
    ``RecordPage`` delegates to this class and keeps its public method names.
    """

    def __init__(
        self,
        ui,
        yaml_parser,
        data_recorder,
        topic_subscribers,
        topic_containers,
        container_manager,
        demo_validator,
    ):
        self.ui = ui
        self.yaml_parser = yaml_parser
        self.data_recorder = data_recorder
        self.topic_subscribers = topic_subscribers
        self.topic_containers = topic_containers
        self.container_manager = container_manager
        self._demo_validator = demo_validator

        self.is_recording = False  # stays True while the demo is being saved
        self.is_saving = False  # Stop was pressed; the save runs in a background thread
        self.recording_start_time = None
        self.config_loaded = False

        # Demos recorded in this session (mirrors AppAPI._session_h5_files)
        self.session_h5_files: list = []
        self.last_validation_result = None

        # The recorder's sampling thread cannot open dialogs, so poll it for "buffer full".
        self._buffer_timer = QTimer()
        self._buffer_timer.setInterval(1000)
        self._buffer_timer.timeout.connect(self._check_buffer_full)

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    def connect_buttons(self):
        """Connect Start/Stop once and set their initial state."""
        self.ui.btn_start_record.clicked.connect(self.start_recording)
        self.ui.btn_stop_record.clicked.connect(self.stop_recording)
        self.update_buttons()

    def update_buttons(self):
        """Enable the buttons from the current state."""
        self.ui.btn_start_record.setEnabled(self.config_loaded and not self.is_recording)
        self.ui.btn_stop_record.setEnabled(self.is_recording and not self.is_saving)

    def set_config_loaded(self, loaded: bool):
        """A config was loaded (Start becomes available) or cleared."""
        self.config_loaded = loaded
        self.update_buttons()

    def _check_buffer_full(self):
        """Tell the operator once when the recorder stopped sampling (buffer full, or its loop failed)."""
        if self.data_recorder.buffer_full is True:
            self._buffer_timer.stop()
            CustomMessageBox.warning(
                "The recording buffer is full: sampling has stopped. "
                "Press Stop to save what was recorded.",
                title="Recording Buffer Full",
            )
        elif self.data_recorder.loop_failed is True:
            self._buffer_timer.stop()
            CustomMessageBox.error(
                "The recording loop kept failing and has stopped sampling (see the log). "
                "Press Stop to save what was recorded.",
                title="Recording Stopped",
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_recording(self):
        """Start recording process"""
        try:
            if self.is_recording:
                logger.warning("Recording already in progress")
                return False

            if not self.yaml_parser.is_loaded():
                logger.error("No configuration loaded - cannot start recording")
                CustomMessageBox.warning("Load a config first: recording needs one (Configuration page).", title="No config loaded")
                return False

            demo_name = start_demo(  # same steps as AppAPI.start_recording
                self.data_recorder, self.topic_subscribers, self.topic_containers.containers,
                self.yaml_parser.get_dataset_name(),
            )
            success = demo_name is not None
            if self.data_recorder.retried_file:  # an earlier demo saved only now: keep it in the session
                self.session_h5_files.append(self.data_recorder.retried_file)

            if success:
                CustomMessageBox.info(
                    "Please check Record Page to visualize the topics.",
                    title="Started Recording",
                )

                self.is_recording = True
                self.recording_start_time = time.time()
                self._buffer_timer.start()

                self.container_manager.set_statuses_by_data(
                    "Recording (Data Available)", "Recording (Waiting for Data)"
                )

                logger.info(f"Started recording: {demo_name}")
                self.update_buttons()
                return True
            else:
                logger.error("Failed to start recording")
                CustomMessageBox.error(
                    "Could not start recording. Please check the config and topics.",
                    title="Recording Error",
                )
                return False

        except TfTopicSubscriptionError as e:
            # A configured custom tf_topic/tf_static_topic could not be
            # subscribed to, or never produced the requested transform.
            # Recording must NOT start on a silent fallback to the default
            # /tf, /tf_static tree — fail loudly and visibly instead.
            logger.error(f"Custom TF topic subscription failed: {e}")
            CustomMessageBox.error(
                f"A configured TF topic could not be read:\n\n{e}\n\n"
                "Recording was NOT started — it will not silently fall back "
                "to the default /tf, /tf_static topics. Check that the "
                "topic is being published and that tf_topic/tf_static_topic "
                "in your config are correct.",
                title="TF Topic Error",
            )
            return False
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            return False

    def stop_recording(self):
        """Stop button: start saving the demo. True if the save was started.

        Writing a few GB of frames takes seconds to minutes: on the GUI thread the window would freeze (and the ROS
        spin timer stop), so the save runs in the background and ``_on_stopped`` finishes the job.
        """
        if not self.is_recording or self.is_saving:
            logger.warning("No recording in progress" if not self.is_recording else "The recording is already being saved")
            return False
        self._buffer_timer.stop()
        self.is_saving = True
        self.update_buttons()
        logger.info("Saving the recording...")
        run_in_background(self.ui.btn_stop_record, self.data_recorder.stop_recording, self._on_stopped)
        return True

    def _on_stopped(self, success, error):
        """GUI thread, after DataRecorder.stop_recording ended: update the session and the UI."""
        self.is_saving = False
        self.update_buttons()
        if error is not None:
            logger.error(f"Error stopping recording: {error}")
            success = False
        try:
            if success:
                self.is_recording = False
                recording_duration = (
                    time.time() - self.recording_start_time
                    if self.recording_start_time
                    else 0
                )

                # Track this demo for the session and run an automatic viability check
                file_path = self.data_recorder.h5_file_path
                if file_path:
                    self.session_h5_files.append(file_path)
                    try:
                        self.last_validation_result = self._demo_validator.validate(file_path)
                        if self.last_validation_result.overall_status == CheckStatus.FAIL:
                            CustomMessageBox.warning(
                                f"Demo saved but failed validation:\n\n"
                                f"{self.last_validation_result.summary_text()}",
                                title="Demo Validation Failed",
                            )
                    except Exception as e:
                        logger.error(f"Error validating demo: {e}")

                # Destroy all subscribers — they will be re-created on next recording start
                self.topic_subscribers.clear_all_subscribers()
                logger.info("Subscribers destroyed after recording stopped")

                self.container_manager.set_statuses_by_data(
                    "Stopped (Data Available)", "Stopped (No Data)"
                )

                logger.info(
                    f"Stopped recording. Duration: {recording_duration:.2f} seconds"
                )
                self.update_buttons()
                return True
            else:
                logger.error("Failed to stop recording")
                if not self.data_recorder.is_recording:
                    # The recorder already stopped but could not save: it keeps the samples in memory
                    # and retries the save on the next Start. Leave the UI in the "stopped" state so
                    # Start is available again instead of stuck on "Recording".
                    self.is_recording = False
                    self.topic_subscribers.clear_all_subscribers()
                    self.container_manager.set_statuses_by_data(
                        "Stopped (Data Available)", "Stopped (No Data)"
                    )
                    self.update_buttons()
                    CustomMessageBox.error(
                        "The recording could not be saved (check the disk). The data is still in "
                        "memory: fix the problem and press Start to retry the save.",
                        title="Recording Not Saved",
                    )
                return False

        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            return False

    def remove_session_file(self, path: str):
        """Prune a demo file from this session's tracked recordings (e.g. after deletion)."""
        if path in self.session_h5_files:
            self.session_h5_files.remove(path)
