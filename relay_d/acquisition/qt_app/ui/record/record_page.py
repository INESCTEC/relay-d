import os
import sys
import time
import traceback


from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QLabel,
    QTextEdit,
    QPushButton,
    QSizePolicy,
)
from PyQt5 import uic
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtCore import QObject, pyqtSignal
from relay_d.utils.coloring_logger import logger
from relay_d.utils.tf_topic import TfTopicSubscriptionError

# Import utils
from ..config.yaml_parser import YamlParser
from ..containers.topic_containers import TopicContainers
from ..record.topic_subscribers import TopicSubscribers
from ..record.container_visualizer import ContainerVisualizer
from ..record.data_recorder import DataRecorder
from ..review.demo_validator import DemoValidator, CheckStatus
from ..settings.config_manager import config
from ...utils.custom_message_box import CustomMessageBox

class RecordPage:
    def __init__(self, ui_instance, yaml_parser: YamlParser, settings_page=None):
        self.ui = ui_instance
        self.yaml_parser = yaml_parser
        self.settings_page = settings_page
        self.topic_containers = TopicContainers(ui_instance)

        # Initialize the TopicSubscribers
        # with the main ROS node
        self.topic_subscribers = TopicSubscribers()

        # Initialize visualization and recording
        self.container_visualizer = ContainerVisualizer()
        self.data_recorder = DataRecorder(self.yaml_parser)

        # Recording state
        self.is_recording = False
        self.recording_start_time = None

        # Session-level demo tracking (mirrors AppAPI._session_h5_files)
        self.session_h5_files: list = []
        self.last_validation_result = None
        self._demo_validator = DemoValidator()

        # Setup all button connections
        self.setup_connections()
        self.config_signal_manager = None

        # Setup recording directory
        # self.data_recorder.setup_recording(settings_page=self.settings_page)

        # Timer for TF updates
        self.tf_update_timer = QTimer()
        self.tf_update_timer.timeout.connect(self._update_tf_lookups)

        # Connect to config changes
        config.settings_changed.connect(self._on_settings_changed)

    def _on_settings_changed(self, settings):
        """Handle settings changes via signal"""
        if "recording_frequency" in settings:
            freq = settings["recording_frequency"]
            if freq and freq > 0:
                self.data_recorder.recording_frequency = freq
                logger.info(f"Recording frequency updated to {freq} Hz")

        if "hdf5_save_folder" in settings:
            self.data_recorder.output_directory = settings["hdf5_save_folder"]
            logger.info(f"HDF5 save folder updated to: {settings['hdf5_save_folder']}")

        if "enable_compression" in settings:
            logger.info(f"Compression set to {settings['enable_compression']}")

        if "freeze_containers" in settings:
            logger.info(f"Freeze containers set to {settings['freeze_containers']}")

    def _update_tf_lookups(self):
        """Update TF lookups periodically"""
        self.topic_subscribers.update_tf_lookups()

    def connect_to_config_signals(self, config_signal_manager):
        """Connect to ConfigPage signals"""
        # Connect the signals to appropriate slots
        config_signal_manager.config_loaded.connect(self._on_config_loaded)
        config_signal_manager.config_cleared.connect(self._on_config_cleared)
        config_signal_manager.config_saved.connect(self._on_config_saved)

    def _on_config_saved(self, file_path):
        """Handle config saved signal - reload containers"""
        logger.info(f"Config saved, reloading containers from: {file_path}")

        # Reload the config file
        self.yaml_parser.load_yaml_file(file_path)

        # Recreate containers with new configuration
        self._create_containers()

    def _on_config_loaded(self, file_path):
        """Handle config loaded signal"""
        logger.info(f"Config loaded from: {file_path}")

        # Load config first
        self.yaml_parser.load_yaml_file(file_path)

        # Register the topic_subscribers node with the UI for spinning
        if hasattr(self.ui, "register_ros_node"):
            self.ui.register_ros_node(self.topic_subscribers)
            self.ui.register_ros_node(self.data_recorder)
            logger.info("Registered topic_subscribers and data_recorder nodes with UI")

        # Create the containers with visualization
        self._create_containers()

        # Start TF update timer at recording_frequency (not hardcoded 10 Hz)
        freq = self.data_recorder.recording_frequency if self.data_recorder else 20
        interval_ms = max(50, int(1000 / freq))   # floor at 50 ms (20 Hz max for UI)
        self.tf_update_timer.start(interval_ms)
        logger.info(f"TF update timer started at {freq} Hz ({interval_ms} ms)")

        ## Enable the buttons
        ## Buttons to start, record and pause
        if hasattr(self.ui, "btn_start_record"):
            self.ui.btn_start_record.setEnabled(True)

        if hasattr(self.ui, "btn_pause_record"):
            self.ui.btn_pause_record.setEnabled(False)

        if hasattr(self.ui, "btn_stop_record"):
            self.ui.btn_stop_record.setEnabled(False)

        ## Set grippers keys for state modification
        self.gripper_key_map = {}
        input_data = self.yaml_parser.get_input_data()

        for entry in input_data:
            if entry["name"].startswith("gripper") and "topic" in entry:
                logger.info(f"Found gripper entry: {entry['name']}")
                # For gripper, we now use key events - open/close keys should be defined separately
                # or handled through the GUI

        logger.info(f"Total gripper mappings: {len(self.gripper_key_map)}")

        # Update status
        self.update_config_status(f"Config loaded: {file_path}")

    def _on_config_cleared(self):
        """Handle config cleared signal"""
        logger.info("Config cleared")

        # Stop TF update timer
        self.tf_update_timer.stop()

        # Stop all visualizations
        self.container_visualizer.stop_all_visualizations()

        # Clear containers and subscribers
        self.topic_subscribers.clear_all_subscribers()
        self.topic_containers.clear_all_containers()

        # RESET layout references
        if hasattr(self.ui, "page_start"):
            self.topic_containers.reset_layout_references(self.ui.page_start)

        ## Enable the buttons
        ## Buttons to start, record and pause
        if hasattr(self.ui, "btn_start_record"):
            self.ui.btn_start_record.setEnabled(False)

        if hasattr(self.ui, "btn_pause_record"):
            self.ui.btn_pause_record.setEnabled(False)

        if hasattr(self.ui, "btn_stop_record"):
            self.ui.btn_stop_record.setEnabled(False)

        # Update status
        self.update_config_status("No config loaded")

    def _create_containers(self):
        """Create containers based on YAML configuration data - WITH SUBSCRIBERS AND VISUALIZATION"""
        try:
            # Clear existing containers, subscribers, and visualizations first
            self.container_visualizer.stop_all_visualizations()
            self.topic_containers.clear_all_containers()
            self.topic_subscribers.clear_all_subscribers()

            # Small delay to ensure cleanup is complete
            time.sleep(0.1)

            # Get configuration data from YAML parser
            input_data = self.yaml_parser.get_input_data()

            logger.info(f"Creating containers for {len(input_data)} inputs")

            # Prepare container specifications
            container_specs = []

            for input_item in input_data:
                input_name = input_item["name"]

                # Get the base type (e.g., 'image' from 'image_1')
                base_type = input_name.split("_")[0]

                # Prepare container data - msg_type will be auto-detected in topic_subscribers
                container_data = {
                    "input_name": input_name,
                    "topic_type_msg": None,  # Will be auto-detected from topic
                    "base_type": base_type,
                    "output_map": input_item.get("output_map", input_name),
                }

                # Check if this is a TF lookup configuration
                if input_item.get("is_tf", False):
                    container_data["parent_frame"] = input_item.get("parent_frame")
                    container_data["child_frame"] = input_item.get("child_frame")
                    container_data["tf_topic"] = input_item.get("tf_topic", "/tf")
                    container_data["tf_static_topic"] = input_item.get(
                        "tf_static_topic", "/tf_static"
                    )
                    container_data["tf_wait_timeout"] = input_item.get(
                        "tf_wait_timeout"
                    )
                    container_data["transfer_rate"] = input_item.get(
                        "transfer_rate", 10
                    )
                    container_data["topic_path"] = None  # No topic for TF lookup
                    # For TF, use 'pose' or 'tf' as the container type
                    container_topic_type = "pose"
                    logger.info(
                        f"TF lookup config: {input_name} - {input_item.get('parent_frame')} -> {input_item.get('child_frame')} -> {input_item.get('output_map')}"
                    )
                # Add topic path for regular topic subscriptions
                elif "topic" in input_item:
                    container_data["topic_path"] = input_item["topic"]
                    container_topic_type = base_type
                else:
                    logger.warning(
                        f"Input {input_name} has no topic or TF frames, skipping"
                    )
                    continue

                # Create container specification
                container_spec = {
                    "topic_type": container_topic_type,
                    "data": container_data,
                }
                container_specs.append(container_spec)

                logger.info(
                    f"Added container spec: {input_name} -> {base_type} -> output_map: {container_data['output_map']}"
                )

            logger.info(f"Total container specs prepared: {len(container_specs)}")

            if not container_specs:
                logger.info("No container specs to create")
                return True

            # Create all containers at once
            created_containers = self.topic_containers.create_multiple_containers(
                container_specs, target_widget=self.ui.page_start
            )

            logger.info(f"Containers created: {len(created_containers)}")

            # Update container UI elements with the data
            for i, container in enumerate(created_containers):
                if i < len(container_specs):
                    container_data = container_specs[i]["data"]
                    container_type = container_specs[i]["topic_type"]

                    # Update all containers with common info first
                    info_dict = {
                        "input_name": container_data.get("input_name"),
                        "topic_path": container_data.get("topic_path"),
                        "status": "Ready",
                    }
                    if "topic_type_msg" in container_data:
                        info_dict["topic_type"] = container_data["topic_type_msg"]

                    self.topic_containers.update_container_info(container, info_dict)

                    logger.info(f"Updated display for container {i + 1}")

            # SET UP VISUALIZATION FOR ALL CONTAINERS
            logger.info("Setting up visualization for containers...")
            for container in created_containers:
                visualization_success = (
                    self.container_visualizer.setup_container_visualization(
                        container, self.topic_subscribers
                    )
                )
                container_id = getattr(container, "container_id", None)
                if visualization_success:
                    logger.info(f"Visualization setup successful for {container_id}")
                else:
                    logger.warning(f"Visualization setup failed for {container_id}")

            # Mark containers as ready (subscribers will be created at recording start)
            for container in created_containers:
                self.topic_containers.update_container_info(
                    container, {"status": "Ready (press Start Recording to subscribe)"}
                )

            logger.info(
                f"Created {len(created_containers)} containers — subscribers will start on record"
            )

            # Make sure all containers are visible and properly sized
            for container in created_containers:
                container.show()
                container.raise_()
                container.update()
                container.repaint()

            # Auto-arrange containers in optimal grid layout
            self.topic_containers.auto_arrange_containers()

            # Force UI refresh
            self.force_ui_refresh()

            # Debug container and subscriber status
            self.topic_containers.debug_containers()
            self.topic_subscribers.debug_subscribers()

            logger.info(
                f"Successfully created {len(created_containers)} containers (subscribers will start on record)"
            )
            return True

        except Exception as e:
            logger.error(f"Exception occurred in create_containers: {e}")
            traceback.print_exc()
            return False

    def update_config_status(self, status_text):
        """Update UI to reflect config status"""
        # Assuming you have a status label
        if hasattr(self, "config_status_label"):
            self.config_status_label.setText(status_text)

    def start_recording(self):
        """Start recording process"""
        try:
            if self.is_recording:
                logger.warning("Recording already in progress")
                return False

            if not self.yaml_parser.is_loaded():
                logger.error("No configuration loaded - cannot start recording")
                return False

            # Create subscribers now (at recording start, not at config load)
            logger.info("Creating ROS subscribers for recording...")
            subscriber_results = self.topic_subscribers.create_subscribers_for_containers(
                self.topic_containers.containers
            )
            successful = sum(subscriber_results.values())
            logger.info(
                f"Subscribers created: {successful}/{len(self.topic_containers.containers)}"
            )

            # Start recording with DataRecorder
            dataset_name = self.yaml_parser.get_dataset_name()
            demo_name = f"{dataset_name}_{int(time.time())}"
            success = self.data_recorder.start_recording(
                self.topic_subscribers, demo_name
            )

            if success:
                CustomMessageBox.info(
                    "Started Recording",
                    "Please check Record Page to visualize the topics.",
                )

                # IMPORTANT: Set RecordPage recording state
                self.is_recording = True
                self.recording_start_time = time.time()

                # Update container statuses
                for container in self.topic_containers.containers:
                    container_id = getattr(container, "container_id", None)
                    if container_id and self.topic_subscribers.has_data(container_id):
                        self.topic_containers.update_container_info(
                            container, {"status": "Recording (Data Available)"}
                        )
                    else:
                        self.topic_containers.update_container_info(
                            container, {"status": "Recording (Waiting for Data)"}
                        )

                logger.info(f"Started recording: {demo_name}")

                # Update UI buttons if they exist
                if hasattr(self.ui, "btn_start_record"):
                    self.ui.btn_start_record.setEnabled(False)
                if hasattr(self.ui, "btn_pause_record"):
                    self.ui.btn_pause_record.setEnabled(True)
                if hasattr(self.ui, "btn_stop_record"):
                    self.ui.btn_stop_record.setEnabled(True)

                return True
            else:
                logger.error("Failed to start recording")
                CustomMessageBox.error(
                    "Recording Error",
                    "Could not start recording. Please check the config and topics.",
                )
                return False

        except TfTopicSubscriptionError as e:
            # A configured custom tf_topic/tf_static_topic could not be
            # subscribed to, or never produced the requested transform.
            # Recording must NOT start on a silent fallback to the default
            # /tf, /tf_static tree — fail loudly and visibly instead.
            logger.error(f"Custom TF topic subscription failed: {e}")
            CustomMessageBox.error(
                "TF Topic Error",
                f"A configured TF topic could not be read:\n\n{e}\n\n"
                "Recording was NOT started — it will not silently fall back "
                "to the default /tf, /tf_static topics. Check that the "
                "topic is being published and that tf_topic/tf_static_topic "
                "in your config are correct.",
            )
            return False
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            return False

    def pause_recording(self):
        """Pause recording process"""
        try:
            if not self.is_recording:
                logger.warning("No recording in progress to pause")
                return False

            success = self.data_recorder.pause_recording()

            if success:
                # Update container statuses
                for container in self.topic_containers.containers:
                    container_id = getattr(container, "container_id", None)
                    if container_id:
                        self.topic_containers.update_container_info(
                            container, {"status": "Paused"}
                        )

                logger.info("Recording paused")

                # Update UI buttons
                if hasattr(self.ui, "btn_pause_record"):
                    self.ui.btn_pause_record.setText("Resume")
                    # Disconnect old connection and connect resume
                    self.ui.btn_pause_record.clicked.disconnect()
                    self.ui.btn_pause_record.clicked.connect(self.resume_recording)

                return True
            else:
                logger.error("Failed to pause recording")
                return False

        except Exception as e:
            logger.error(f"Error pausing recording: {e}")
            return False

    def resume_recording(self):
        """Resume recording process"""
        try:
            success = self.data_recorder.resume_recording()

            if success:
                # Update container statuses
                for container in self.topic_containers.containers:
                    container_id = getattr(container, "container_id", None)
                    if container_id and self.topic_subscribers.has_data(container_id):
                        self.topic_containers.update_container_info(
                            container, {"status": "Recording (Data Available)"}
                        )
                    else:
                        self.topic_containers.update_container_info(
                            container, {"status": "Recording (Waiting for Data)"}
                        )

                logger.info("Recording resumed")

                # Update UI buttons
                if hasattr(self.ui, "btn_pause_record"):
                    self.ui.btn_pause_record.setText("Pause")
                    # Disconnect old connection and connect pause
                    self.ui.btn_pause_record.clicked.disconnect()
                    self.ui.btn_pause_record.clicked.connect(self.pause_recording)

                return True
            else:
                logger.error("Failed to resume recording")
                return False

        except Exception as e:
            logger.error(f"Error resuming recording: {e}")
            return False

    def stop_recording(self):
        """Stop recording process"""
        try:
            if not self.is_recording:
                logger.warning("No recording in progress")
                return False

            # Stop recording
            success = self.data_recorder.stop_recording()

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

                # Update container statuses
                for container in self.topic_containers.containers:
                    container_id = getattr(container, "container_id", None)
                    if container_id and self.topic_subscribers.has_data(container_id):
                        self.topic_containers.update_container_info(
                            container, {"status": "Stopped (Data Available)"}
                        )
                    else:
                        self.topic_containers.update_container_info(
                            container, {"status": "Stopped (No Data)"}
                        )

                logger.info(
                    f"Stopped recording. Duration: {recording_duration:.2f} seconds"
                )

                # Update UI buttons
                if hasattr(self.ui, "btn_start_record"):
                    self.ui.btn_start_record.setEnabled(True)
                if hasattr(self.ui, "btn_pause_record"):
                    self.ui.btn_pause_record.setEnabled(False)
                    self.ui.btn_pause_record.setText("Pause")  # Reset text
                    # Reconnect to pause function
                    self.ui.btn_pause_record.clicked.disconnect()
                    self.ui.btn_pause_record.clicked.connect(self.pause_recording)
                if hasattr(self.ui, "btn_stop_record"):
                    self.ui.btn_stop_record.setEnabled(False)

                return True
            else:
                logger.error("Failed to stop recording")
                return False

        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            return False

    def remove_session_file(self, path: str):
        """Prune a demo file from this session's tracked recordings (e.g. after deletion)."""
        if path in self.session_h5_files:
            self.session_h5_files.remove(path)

    def get_container_data(self, input_name):
        """Get the latest ROS data for a container by input name"""
        container = self.get_container_by_input_name(input_name)
        if container:
            return self.topic_subscribers.get_container_data(container)
        return None

    def get_container_by_input_name(self, input_name):
        """Get container by input name"""
        for container in self.topic_containers.containers:
            if hasattr(container, "input_name") and container.input_name == input_name:
                return container
        return None

    def get_all_topic_data(self):
        """Get all current topic data"""
        return self.topic_subscribers.get_content()

    def get_recording_summary(self):
        """Get a comprehensive summary of the recording setup"""
        if not self.yaml_parser.is_loaded():
            return None

        # Get subscriber summary
        subscriber_summary = self.topic_subscribers.get_all_data_summary()

        # Get recording status
        recording_status = self.data_recorder.get_recording_status()

        config = {
            "input_data": self.yaml_parser.get_input_data(),
            "output_data": self.yaml_parser.get_output_data(),
            "config_data": self.yaml_parser.get_config_data(),
            "topic_data": self.yaml_parser.get_topic_data(),
            "prefix_path": self.yaml_parser.get_prefix_path(),
            "containers": self.get_all_container_info(),
            "subscribers": subscriber_summary,
            "recording_status": recording_status,
            "topics_with_data": [
                cid
                for cid in subscriber_summary["subscriber_details"]
                if cid.get("has_data", False)
            ],
        }

        return config

    def get_all_container_info(self):
        """Get information about all containers"""
        return self.topic_containers.get_all_container_info()

    def wait_for_all_data(self, timeout=10.0):
        """Wait for all containers to receive data"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            all_have_data = True
            for container in self.topic_containers.containers:
                container_id = getattr(container, "container_id", None)
                if container_id and not self.topic_subscribers.has_data(container_id):
                    all_have_data = False
                    break

            if all_have_data:
                logger.info("All containers have received data")
                return True

            time.sleep(0.1)

        logger.warning(f"Timeout waiting for all containers to receive data")
        return False

    def restart_failed_subscribers(self):
        """Restart subscribers that failed to create or stopped working"""
        restarted_count = 0

        for container in self.topic_containers.containers:
            container_id = getattr(container, "container_id", None)
            if container_id:
                status = self.topic_subscribers.get_subscriber_status(container_id)

                # Restart if subscriber failed or has no recent data
                if (
                    not status
                    or status.get("status") != "active"
                    or not self.topic_subscribers.has_data(container_id)
                ):
                    if self.topic_subscribers.restart_subscriber(container):
                        # Also restart visualization
                        self.container_visualizer.setup_container_visualization(
                            container, self.topic_subscribers
                        )
                        restarted_count += 1
                        self.topic_containers.update_container_info(
                            container, {"status": "Restarted"}
                        )
                        logger.info(
                            f"Restarted subscriber and visualization for {container_id}"
                        )
                    else:
                        self.topic_containers.update_container_info(
                            container, {"status": "Restart Failed"}
                        )

        logger.info(f"Restarted {restarted_count} subscribers")
        return restarted_count

    def update_container_statuses(self):
        """Update all container statuses based on current subscriber state"""
        for container in self.topic_containers.containers:
            container_id = getattr(container, "container_id", None)
            if container_id:
                if self.topic_subscribers.has_data(container_id):
                    message_count = self.topic_subscribers.get_message_count(
                        container_id
                    )
                    status = f"Active ({message_count} msgs)"
                else:
                    subscriber_status = self.topic_subscribers.get_subscriber_status(
                        container_id
                    )
                    if subscriber_status.get("status") == "active":
                        status = "Waiting for Data"
                    else:
                        status = "No Subscriber"

                self.topic_containers.update_container_info(
                    container, {"status": status}
                )

    def force_ui_refresh(self):
        """Force UI refresh"""
        try:
            self.ui.update()
            self.ui.repaint()
            if hasattr(self.ui, "page_start"):
                self.ui.page_start.update()
                self.ui.page_start.repaint()
        except Exception as e:
            logger.error(f"Error in force UI refresh: {e}")

    def on_window_resize(self):
        """Handle window resize events"""
        try:
            # Update container layout for new window size
            if hasattr(self.ui, "size"):
                window_size = self.ui.size()
                self.topic_containers.resize_containers_to_window(window_size)
        except Exception as e:
            logger.error(f"Error handling window resize: {e}")

    def set_visualization_rate(self, rate_hz):
        """Set the visualization update rate"""
        self.container_visualizer.set_update_rate(rate_hz)

    def setup_connections(self):
        """Setup all button connections"""
        try:
            # Container management
            if hasattr(self.ui, "btn_refresh_containers"):
                self.ui.btn_refresh_containers.clicked.connect(self._create_containers)

            if hasattr(self.ui, "btn_restart_subscribers"):
                self.ui.btn_restart_subscribers.clicked.connect(
                    self.restart_failed_subscribers
                )

            if hasattr(self.ui, "btn_update_status"):
                self.ui.btn_update_status.clicked.connect(
                    self.update_container_statuses
                )

            ## Buttons to start, record and pause
            if hasattr(self.ui, "btn_start_record"):
                self.ui.btn_start_record.clicked.connect(self.start_recording)
                self.ui.btn_start_record.setEnabled(False)

            # Connect pause button to pause_recording method
            if hasattr(self.ui, "btn_pause_record"):
                self.ui.btn_pause_record.clicked.connect(self.pause_recording)
                self.ui.btn_pause_record.setEnabled(False)
                self.ui.btn_pause_record.setText("Pause")  # Set initial text

            if hasattr(self.ui, "btn_stop_record"):
                self.ui.btn_stop_record.clicked.connect(self.stop_recording)
                self.ui.btn_stop_record.setEnabled(False)

        except Exception as e:
            logger.error("Error setting up connections.")

    # Add this debug function at the end of your record_page.py file
    def debug_containers(self):
        """Debug method to inspect container widgets"""
        try:
            logger.info("=== STARTING CONTAINER DEBUG ===")

            if hasattr(self, "topic_containers") and self.topic_containers.containers:
                containers = self.topic_containers.containers
                logger.info(f"Found {len(containers)} containers to debug")

                for i, container in enumerate(containers):
                    container_id = getattr(container, "container_id", f"container_{i}")
                    logger.info(f"\n=== DEBUG CONTAINER {i + 1}: {container_id} ===")

                    # Get all attributes
                    all_attrs = [
                        attr for attr in dir(container) if not attr.startswith("_")
                    ]

                    # Find widgets with setText method
                    setText_widgets = []
                    setPixmap_widgets = []

                    for attr in all_attrs:
                        try:
                            obj = getattr(container, attr)
                            if hasattr(obj, "setText"):
                                setText_widgets.append(attr)
                            if hasattr(obj, "setPixmap"):
                                setPixmap_widgets.append(attr)
                        except:
                            continue

                    logger.info(
                        f"Container size: {container.size().width()}x{container.size().height()}"
                    )
                    logger.info(f"Container visible: {container.isVisible()}")
                    logger.info(f"Widgets with setText: {setText_widgets}")
                    logger.info(f"Widgets with setPixmap: {setPixmap_widgets}")

                    # Test updating first few widgets
                    for widget_name in setText_widgets[:3]:
                        try:
                            widget = getattr(container, widget_name)
                            widget.setText(f"TEST: {widget_name}")
                            logger.info(f"SUCCESS: Updated {widget_name}")
                        except Exception as e:
                            logger.info(f"FAILED: {widget_name} - {e}")
            else:
                logger.info("No containers found to debug")

        except Exception as e:
            logger.error(f"Error in debug_containers: {e}")
