from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QLabel, QSizePolicy
from relay_d.utils.coloring_logger import logger

# Import utils
from ..config.yaml_parser import YamlParser
from ..containers.topic_containers import TopicContainers
from ..record.topic_subscribers import TopicSubscribers
from ..record.container_visualizer import ContainerVisualizer
from ..page_style import add_page_header, page_layout
from ..record.data_recorder import DataRecorder
from ..record.container_manager import ContainerManager
from ..record.recording_controller import RecordingController
from ..review.demo_validator import DemoValidator
from ..settings.config_manager import config

class RecordPage:
    def __init__(self, ui_instance, yaml_parser: YamlParser, settings_page=None):
        self.ui = ui_instance
        self.yaml_parser = yaml_parser
        self.settings_page = settings_page
        page = getattr(ui_instance, "page_start", None)
        self.label_no_config = None
        if page is not None and page.layout() is None:  # same title and margins as every page; the containers are added below it
            root = page_layout(page)
            add_page_header(root, "Record")
            self.label_no_config = QLabel(
                "No config loaded.\n\nLoad one on the Configuration page (insert a file, or build one with \"Build from ROS\"): "
                "the live view of its topics appears here, and recording starts from here."
            )
            self.label_no_config.setObjectName("pageHint")
            self.label_no_config.setWordWrap(True)
            self.label_no_config.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            self.label_no_config.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            root.addWidget(self.label_no_config, 1)
        self.topic_containers = TopicContainers(ui_instance)

        # Initialize the TopicSubscribers
        # with the main ROS node
        self.topic_subscribers = TopicSubscribers()

        # Initialize visualization and recording
        self.container_visualizer = ContainerVisualizer()
        self.data_recorder = DataRecorder(self.yaml_parser)
        self.container_manager = ContainerManager(
            ui_instance,
            yaml_parser,
            self.topic_containers,
            self.topic_subscribers,
            self.container_visualizer,
        )

        # Recording lifecycle: start/pause/resume/stop, session demos, record buttons
        self.recording = RecordingController(
            ui_instance,
            yaml_parser,
            self.data_recorder,
            self.topic_subscribers,
            self.topic_containers,
            self.container_manager,
            DemoValidator(),
        )

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
        if self.label_no_config is not None:
            self.label_no_config.hide()

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

        # Start becomes available once a config is loaded
        self.recording.set_config_loaded(True)

        ## Set grippers keys for state modification
        self.gripper_key_map = {}
        input_data = self.yaml_parser.get_input_data()

        for entry in input_data:
            if entry["name"].startswith("gripper") and "topic" in entry:
                logger.info(f"Found gripper entry: {entry['name']}")
                # For gripper, we now use key events - open/close keys should be defined separately
                # or handled through the GUI

        logger.info(f"Total gripper mappings: {len(self.gripper_key_map)}")


    def _on_config_cleared(self):
        """Handle config cleared signal"""
        logger.info("Config cleared")
        if self.label_no_config is not None:
            self.label_no_config.show()

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

        self.recording.set_config_loaded(False)


    def _create_containers(self):
        """Create containers based on YAML configuration data"""
        return self.container_manager.create_containers()

    # ------------------------------------------------------------------
    # Recording lifecycle (implemented by RecordingController)
    # ------------------------------------------------------------------

    def start_recording(self):
        """Start recording process"""
        return self.recording.start_recording()

    def pause_recording(self):
        """Pause recording process"""
        return self.recording.pause_recording()

    def resume_recording(self):
        """Resume recording process"""
        return self.recording.resume_recording()

    def stop_recording(self):
        """Stop recording process"""
        return self.recording.stop_recording()

    def remove_session_file(self, path: str):
        """Prune a demo file from this session's tracked recordings (e.g. after deletion)."""
        self.recording.remove_session_file(path)

    @property
    def is_recording(self):
        return self.recording.is_recording

    @property
    def recording_start_time(self):
        return self.recording.recording_start_time

    @property
    def session_h5_files(self):
        return self.recording.session_h5_files

    @property
    def last_validation_result(self):
        return self.recording.last_validation_result

    def restart_failed_subscribers(self):
        """Restart subscribers that failed to create or stopped working"""
        return self.container_manager.restart_failed_subscribers()

    def force_ui_refresh(self):
        """Force UI refresh"""
        self.container_manager.force_ui_refresh()

    def on_window_resize(self):
        """Handle window resize events"""
        self.container_manager.on_window_resize()

    def setup_connections(self):
        """Setup all button connections"""
        try:
            # Container management
            if hasattr(self.ui, "btn_restart_subscribers"):
                self.ui.btn_restart_subscribers.clicked.connect(
                    self.restart_failed_subscribers
                )

            # Start / Pause(Resume) / Stop: each connected once by the controller
            self.recording.connect_buttons()

        except Exception as e:
            logger.error(f"Error setting up connections: {e}")
